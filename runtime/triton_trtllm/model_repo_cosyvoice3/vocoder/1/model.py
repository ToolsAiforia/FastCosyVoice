import json
import os
import logging

import numpy as np
import torch
from torch.utils.dlpack import to_dlpack
import triton_python_backend_utils as pb_utils
from hyperpyyaml import load_hyperpyyaml

import sys
sys.path.insert(0, "/workspace/CosyVoice")
from cosyvoice.utils.common import TrtContextWrapper
from cosyvoice.utils.file_utils import convert_onnx_to_trt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

torch.set_num_threads(1)


class TritonPythonModel:
    """CosyVoice3 vocoder with hybrid PyTorch + TRT pipeline.

    - f0_predictor (CPU fp64), m_source, STFT, conv_pre, iSTFT — PyTorch
    - decode_core (ups + resblocks + conv_post + exp/sin) — TRT engine fp16
    """

    def initialize(self, args):
        parameters = json.loads(args['model_config'])['parameters']
        model_params = {k: v["string_value"] for k, v in parameters.items()}
        model_dir = model_params["model_dir"]

        self.device = torch.device("cuda")

        with open(os.path.join(model_dir, 'cosyvoice3.yaml'), 'r') as f:
            configs = load_hyperpyyaml(f, overrides={
                'qwen_pretrain_path': os.path.join(model_dir, 'CosyVoice-BlankEN')
            })
        self.hift = configs['hift']
        hift_state_dict = {
            k.replace('generator.', ''): v
            for k, v in torch.load(
                os.path.join(model_dir, 'hift.pt'),
                map_location='cpu', weights_only=True
            ).items()
        }
        self.hift.load_state_dict(hift_state_dict, strict=True)
        self.hift.to(self.device).eval()

        # Pre-compute streaming constants
        self.istft_n_fft = self.hift.istft_params["n_fft"]
        self.istft_hop_len = self.hift.istft_params["hop_len"]
        self.upsample_total = int(np.prod(self.hift.upsample_rates))
        self.conv_pre_look_right = self.hift.conv_pre_look_right
        self.n_fft_half_plus_one = self.istft_n_fft // 2 + 1
        # finalize=False trim length on the source stft side
        self.s_stft_trim_len = self.upsample_total * self.conv_pre_look_right
        # finalize=False trim length on the final audio side
        self.audio_trim_len = self.upsample_total * self.istft_hop_len

        # Load TRT engine(s) for decode_core. fp32 — fp16 overflows in exp/sigmoid.
        # We keep TWO plans to avoid B=1 fastpath regression (opt=B=4 on a
        # dynamic plan makes B=1 ~70% slower than B=1-optimized plan):
        #   - hift_decode_core.fp32.plan       (B=1 only) — used when batch=1
        #   - hift_decode_core.fp32_B8.plan    (B=1..8)   — used when batch>=2
        plan_path_b8 = os.path.join(model_dir, 'hift_decode_core.fp32_B8.plan')
        plan_path_b1 = os.path.join(model_dir, 'hift_decode_core.fp32.plan')
        onnx_path = os.path.join(model_dir, 'hift_decode_core.onnx')
        if not os.path.exists(plan_path_b1) or os.path.getsize(plan_path_b1) == 0:
            assert os.path.exists(onnx_path), f"Missing {onnx_path}"
            trt_kwargs_b1 = {
                'min_shape': [(1, 512, 1),    (1, 18, 121)],
                'opt_shape': [(1, 512, 16),   (1, 18, 1921)],
                'max_shape': [(1, 512, 2500), (1, 18, 300001)],
                'input_names': ['x_pre', 's_stft'],
            }
            logger.info(f"Building B=1 TRT engine from {onnx_path}")
            convert_onnx_to_trt(plan_path_b1, trt_kwargs_b1, onnx_path, fp16=False)

        import tensorrt as trt
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        with open(plan_path_b1, 'rb') as f:
            engine_b1 = runtime.deserialize_cuda_engine(f.read())
        assert engine_b1 is not None, f"failed to deserialize {plan_path_b1}"
        self.decode_core_b1 = TrtContextWrapper(engine_b1, trt_concurrent=1, device=self.device)
        logger.info(f"Loaded HiFT B=1 plan: {os.path.basename(plan_path_b1)}")

        self.decode_core_b8 = None
        if os.path.exists(plan_path_b8) and os.path.getsize(plan_path_b8) > 0:
            with open(plan_path_b8, 'rb') as f:
                engine_b8 = runtime.deserialize_cuda_engine(f.read())
            if engine_b8 is not None:
                self.decode_core_b8 = TrtContextWrapper(engine_b8, trt_concurrent=1, device=self.device)
                logger.info(f"Loaded HiFT B-dynamic plan: {os.path.basename(plan_path_b8)}")

        # Default decode_core points to B=1 fastpath (used by the monkey-patched
        # hybrid_decode when called from execute with B=1). For B>1 we manually
        # rebind decode_core to b8 just for the call.
        self.decode_core = self.decode_core_b1

        # Replace hift.decode with our hybrid implementation
        self._monkey_patch_decode()

        logger.info(f"CausalHiFTGenerator initialized (decode_core: TRT fp16, "
                    f"upsample_total={self.upsample_total}, n_fft={self.istft_n_fft})")

    def _run_decode_core_trt(self, x_pre: torch.Tensor, s_stft: torch.Tensor):
        """Run TRT engine fp32 on (x_pre, s_stft) → (magnitude, phase)."""
        # Defensive cap: if input exceeds TRT profile (e.g. hallucinating fp8 LLM),
        # trim tail so execute_async_v3 cannot hang on out-of-profile shape.
        # Profile max: x_pre [..,..,2500], s_stft [..,..,300001]
        if x_pre.shape[-1] > 2500:
            logger.warning(f"x_pre frames={x_pre.shape[-1]} > 2500 — trimming tail (LLM hallucination?)")
            x_pre = x_pre[..., :2500]
            s_stft_max = 2500 * (self.upsample_total // self.istft_hop_len)
            s_stft = s_stft[..., :min(s_stft.shape[-1], s_stft_max + 1)]
        if s_stft.shape[-1] > 300001:
            logger.warning(f"s_stft frames={s_stft.shape[-1]} > 300001 — trimming tail")
            s_stft = s_stft[..., :300001]
        x_pre_f = x_pre.float().contiguous()
        s_stft_f = s_stft.float().contiguous()

        [context, _ignored_stream], engine = self.decode_core.acquire_estimator()
        try:
            with torch.cuda.device(self.device):
                context.set_input_shape("x_pre", tuple(x_pre_f.shape))
                context.set_input_shape("s_stft", tuple(s_stft_f.shape))
                B, _, T_stft = s_stft_f.shape
                out_mag = torch.empty(B, self.n_fft_half_plus_one, T_stft,
                                      dtype=torch.float32, device=self.device).contiguous()
                out_pha = torch.empty(B, self.n_fft_half_plus_one, T_stft,
                                      dtype=torch.float32, device=self.device).contiguous()
                context.set_tensor_address("x_pre", x_pre_f.data_ptr())
                context.set_tensor_address("s_stft", s_stft_f.data_ptr())
                context.set_tensor_address("magnitude", out_mag.data_ptr())
                context.set_tensor_address("phase", out_pha.data_ptr())
                ok = context.execute_async_v3(stream_handle=torch.cuda.current_stream().cuda_stream)
                torch.cuda.current_stream().synchronize()
                assert ok, "TRT execute_async_v3 returned False"
            return out_mag, out_pha
        finally:
            self.decode_core.release_estimator(context, _ignored_stream)

    def _monkey_patch_decode(self):
        """Replace hift.decode with hybrid PyTorch + TRT implementation."""
        hift = self.hift
        run_trt = self._run_decode_core_trt
        conv_pre_look_right = self.conv_pre_look_right
        s_stft_trim_len = self.s_stft_trim_len
        audio_trim_len = self.audio_trim_len

        def hybrid_decode(x: torch.Tensor, s: torch.Tensor = torch.zeros(1, 1, 0),
                          finalize: bool = True) -> torch.Tensor:
            s_stft_real, s_stft_imag = hift._stft(s.squeeze(1))
            if finalize is True:
                x = hift.conv_pre(x)
            else:
                x = hift.conv_pre(x[:, :, :-conv_pre_look_right], x[:, :, -conv_pre_look_right:])
                s_stft_real = s_stft_real[:, :, :-s_stft_trim_len]
                s_stft_imag = s_stft_imag[:, :, :-s_stft_trim_len]
            s_stft = torch.cat([s_stft_real, s_stft_imag], dim=1)

            # TRT main GPU compute
            magnitude, phase = run_trt(x, s_stft)

            x = hift._istft(magnitude, phase)
            if finalize is False:
                x = x[:, :-audio_trim_len]
            x = torch.clamp(x, -hift.audio_limit, hift.audio_limit)
            return x

        hift.decode = hybrid_decode

    def execute(self, requests):
        """Coordinator-friendly execute: BLS may send ONE request with batched
        mel tensor [B, 80, T] (uniform shapes guaranteed by BLS shape_key).
        - B=1: B=1-optimized TRT plan (no regression vs single-instance path).
        - B>1: B-dynamic TRT plan (loaded only if hift_decode_core.fp32_B8.plan
          present); otherwise loop B=1 inference per row (slow fallback).
        Output [B, T_audio] returned as-is, BLS slices per-request future.
        """
        responses = []
        for request in requests:
            mel = pb_utils.get_input_tensor_by_name(request, "mel")
            mel = torch.utils.dlpack.from_dlpack(mel.to_dlpack()).to(self.device)
            if mel.dim() == 2:
                mel = mel.unsqueeze(0)
            fin_pb = pb_utils.get_input_tensor_by_name(request, "finalize").as_numpy()
            finalize = bool(fin_pb.flatten()[0])
            B = mel.shape[0]

            with torch.no_grad():
                if B == 1:
                    # Fastpath — B=1 plan
                    self.decode_core = self.decode_core_b1
                    speech, _ = self.hift.inference(speech_feat=mel, finalize=finalize)
                elif self.decode_core_b8 is not None:
                    # Batched path — B-dynamic plan
                    prev_core = self.decode_core
                    self.decode_core = self.decode_core_b8
                    try:
                        speech, _ = self.hift.inference(speech_feat=mel, finalize=finalize)
                    finally:
                        self.decode_core = prev_core
                else:
                    # No B-dynamic plan available — fallback per-row B=1
                    rows = []
                    for i in range(B):
                        s_i, _ = self.hift.inference(
                            speech_feat=mel[i:i+1], finalize=finalize)
                        rows.append(s_i.squeeze())
                    # Audio length per row should be uniform since mel time is uniform
                    max_T = max(r.shape[0] for r in rows)
                    speech = torch.stack([
                        torch.nn.functional.pad(r, (0, max_T - r.shape[0]))
                        for r in rows
                    ], dim=0)

            if speech.dim() == 1:
                speech = speech.unsqueeze(0)
            speech_tensor = pb_utils.Tensor.from_dlpack(
                "tts_speech", to_dlpack(speech.contiguous()))
            responses.append(pb_utils.InferenceResponse(output_tensors=[speech_tensor]))

        return responses
