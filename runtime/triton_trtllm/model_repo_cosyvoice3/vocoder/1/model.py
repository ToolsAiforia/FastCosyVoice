import json
import os
import queue
import logging

import numpy as np
import torch
from torch.utils.dlpack import to_dlpack
import triton_python_backend_utils as pb_utils
from hyperpyyaml import load_hyperpyyaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

torch.set_num_threads(1)


class TrtContextWrapper:
    """Simple TRT execution context pool wrapper."""

    def __init__(self, trt_engine, trt_concurrent=1, device='cuda:0'):
        self.trt_context_pool = queue.Queue(maxsize=trt_concurrent)
        self.trt_engine = trt_engine
        self.device = device
        for _ in range(trt_concurrent):
            ctx = trt_engine.create_execution_context()
            stream = torch.cuda.stream(torch.cuda.Stream(torch.device(device)))
            assert ctx is not None
            self.trt_context_pool.put([ctx, stream])

    def acquire_estimator(self):
        return self.trt_context_pool.get(), self.trt_engine

    def release_estimator(self, ctx, stream):
        self.trt_context_pool.put([ctx, stream])


def _build_hift_fp32_trt(plan_path, onnx_path):
    """Build pure-FP32 HiFT decode_core TRT plan.

    Used as the default safe baseline when layer-mixed plan is unavailable.
    Time-dim profile chosen for typical streaming chunk sizes (16-1921 mel frames).
    """
    import tensorrt as trt
    logger.info(f"Building HiFT FP32 TRT plan from {os.path.basename(onnx_path)}...")
    trt_logger = trt.Logger(trt.Logger.WARNING)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, trt_logger)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 33)  # 8 GB
    if hasattr(trt.BuilderFlag, "TF32"):
        config.set_flag(trt.BuilderFlag.TF32)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                logger.error(parser.get_error(i))
            raise ValueError(f"failed to parse {onnx_path}")

    profile = builder.create_optimization_profile()
    profile.set_shape("x_pre",  (1, 512, 1),    (1, 512, 16),  (1, 512, 2500))
    profile.set_shape("s_stft", (1, 18, 121),   (1, 18, 1921), (1, 18, 300001))
    config.add_optimization_profile(profile)

    engine_bytes = builder.build_serialized_network(network, config)
    assert engine_bytes is not None, "TRT FP32 build returned None"
    with open(plan_path, "wb") as f:
        f.write(bytes(engine_bytes))
    logger.info(f"Built {plan_path} ({os.path.getsize(plan_path)//(1024*1024)} MB)")


def _build_hift_layer_mixed_trt(plan_path, onnx_path):
    """Build HiFT decode_core TRT plan with per-layer mixed precision.

    Strategy mirrors DiT round-9 winner:
      - FP16 default for Conv/Mul/Add (most of the network)
      - FP32 override for precision-sensitive ops:
        * Sin (73 nodes) — phase output; FP16 underflows near zero
        * Pow (72 nodes) — large dynamic range
        * Reciprocal (72) — 1/x amplifies FP16 quantization error
        * Exp/Log/Sqrt — sensitive unary
        * Normalization — stats need FP32 precision
        * Sensitive activations (Sigmoid, Tanh, GELU, ELU)

    Pure-FP16 was tried earlier and produced 100% clipping (TRT auto-picked
    BF16 for exp/sigmoid). Per-layer manual override prevents that.

    IO kept FP32 (compatible with PyTorch wrapper feeding fp32 x_pre/s_stft).
    """
    import tensorrt as trt
    logger.info(f"Building HiFT layer-mixed TRT plan from {os.path.basename(onnx_path)} "
                f"(takes ~1-2 min, can OOM if VRAM <8 GB free)")

    trt_logger = trt.Logger(trt.Logger.WARNING)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, trt_logger)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 33)  # 8 GB
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                logger.error(parser.get_error(i))
            raise ValueError(f"failed to parse {onnx_path}")

    # IO stays FP32 to avoid changing PyTorch wrapper
    for i in range(network.num_inputs):
        network.get_input(i).dtype = trt.DataType.FLOAT
    for i in range(network.num_outputs):
        network.get_output(i).dtype = trt.DataType.FLOAT

    # NAME-based matching (TRT enum .op access for parsed network can be unreliable).
    # HiFT uses Snake-like activations: x + (1/α)·sin²(α·x) → /activations*/{Sin,Pow,Reciprocal}.
    # Layer names preserve ONNX op naming under /activations{N}.{M}/Op patterns.
    SKIP_TYPES = {trt.LayerType.CONSTANT, trt.LayerType.CAST, trt.LayerType.SHAPE,
                  trt.LayerType.GATHER, trt.LayerType.SLICE, trt.LayerType.SHUFFLE,
                  trt.LayerType.CONCATENATION, trt.LayerType.IDENTITY}
    # Substrings in layer.name that mark precision-sensitive math
    SENSITIVE_KEYWORDS = (
        'Sin', 'Cos', 'Pow', 'Reciprocal', 'Exp', 'Log', 'Sqrt',
        'Softmax', 'LayerNorm', 'Tanh', 'Sigmoid', 'GELU', 'Erf',
        'gelu', 'sigmoid', 'tanh', 'norm',
        # Whole Snake activation blocks (Sin/Pow/Reciprocal + surrounding
        # Add/Mul/Cast) — without these, TRT PWN fusion crosses precision
        # boundary and produces NaN. 218 → 507 layers FP32 with these.
        '/activations1', '/activations2', '/m_source',
    )
    FLOAT_LIKE = (trt.DataType.FLOAT, trt.DataType.HALF, trt.DataType.BF16)

    fp32_count = 0
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        if layer.type in SKIP_TYPES:
            continue
        name = layer.name
        should_fp32 = any(kw in name for kw in SENSITIVE_KEYWORDS)
        # Always FP32 for these structural types regardless of name
        if not should_fp32:
            if layer.type in (trt.LayerType.SOFTMAX, trt.LayerType.NORMALIZATION):
                should_fp32 = True
        if not should_fp32:
            continue
        try:
            for j in range(layer.num_outputs):
                if layer.get_output(j).dtype not in FLOAT_LIKE:
                    should_fp32 = False
                    break
        except Exception:
            pass
        if should_fp32:
            try:
                layer.precision = trt.DataType.FLOAT
                for j in range(layer.num_outputs):
                    layer.set_output_type(j, trt.DataType.FLOAT)
                fp32_count += 1
            except Exception:
                pass

    logger.info(f"HiFT layer-mixed: marked {fp32_count}/{network.num_layers} layers as FP32 "
                f"(Sin/Pow/Reciprocal/Norm/sensitive activations)")

    profile = builder.create_optimization_profile()
    profile.set_shape("x_pre",  (1, 512, 1),    (1, 512, 16),  (1, 512, 2500))
    profile.set_shape("s_stft", (1, 18, 121),   (1, 18, 1921), (1, 18, 300001))
    config.add_optimization_profile(profile)

    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError("HiFT layer-mixed build returned None — check VRAM")
    with open(plan_path, "wb") as f:
        f.write(bytes(engine_bytes))
    logger.info(f"Built {plan_path} ({os.path.getsize(plan_path)//(1024*1024)} MB)")


class TritonPythonModel:
    """CosyVoice3 vocoder with hybrid PyTorch + TRT pipeline.

    - f0_predictor (CPU fp64), m_source, STFT, conv_pre, iSTFT — PyTorch
    - decode_core (ups + resblocks + conv_post + exp/sin) — TRT engine
    - Per-layer mixed precision (default) or FP32 fallback
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

        # TRT decode_core: prefer layer-mixed plan, fall back to FP32 plan.
        layer_mixed_path = os.path.join(
            model_dir, 'hift_decode_core.layer_mixed_fp32io.plan')
        fp32_path = os.path.join(model_dir, 'hift_decode_core.fp32.plan')
        onnx_path = os.path.join(model_dir, 'hift_decode_core.onnx')
        assert os.path.exists(onnx_path), f"Missing {onnx_path}"

        # Build layer-mixed if missing
        if not os.path.exists(layer_mixed_path) or os.path.getsize(layer_mixed_path) == 0:
            _build_hift_layer_mixed_trt(layer_mixed_path, onnx_path)
        if not os.path.exists(fp32_path) or os.path.getsize(fp32_path) == 0:
            _build_hift_fp32_trt(fp32_path, onnx_path)

        plan_path = layer_mixed_path  # prefer layer-mixed
        import tensorrt as trt
        with open(plan_path, "rb") as f:
            engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(f.read())
        assert engine is not None, f"failed to deserialize {plan_path}"
        self.decode_core = TrtContextWrapper(engine, trt_concurrent=1, device=self.device)
        logger.info(f"Loaded HiFT TRT plan: {os.path.basename(plan_path)}")

        # Replace hift.decode with our hybrid implementation
        self._monkey_patch_decode()

        logger.info(f"CausalHiFTGenerator hybrid initialized "
                    f"(upsample_total={self.upsample_total}, n_fft={self.istft_n_fft})")

    def _run_decode_core_trt(self, x_pre: torch.Tensor, s_stft: torch.Tensor):
        """Run TRT engine fp32 IO on (x_pre, s_stft) → (magnitude, phase)."""
        # Defensive cap against out-of-profile shapes (e.g. hallucinating LLM)
        if x_pre.shape[-1] > 2500:
            logger.warning(f"x_pre frames={x_pre.shape[-1]} > 2500 — trimming")
            x_pre = x_pre[..., :2500]
            s_stft_max = 2500 * (self.upsample_total // self.istft_hop_len)
            s_stft = s_stft[..., :min(s_stft.shape[-1], s_stft_max + 1)]
        if s_stft.shape[-1] > 300001:
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
                # No sync needed — out_mag/out_pha are queued on current_stream,
                # subsequent torch ops on same stream are correctly ordered
                # (Path D micro-optimization from DiT, applied here too).
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
        responses = []
        for request in requests:
            mel = pb_utils.get_input_tensor_by_name(request, "mel")
            mel = torch.utils.dlpack.from_dlpack(mel.to_dlpack()).to(self.device)
            if mel.dim() == 2:
                mel = mel.unsqueeze(0)
            finalize = pb_utils.get_input_tensor_by_name(request, "finalize").as_numpy().item()

            with torch.no_grad():
                speech, _ = self.hift.inference(speech_feat=mel, finalize=finalize)

            speech = speech.squeeze()
            speech_tensor = pb_utils.Tensor.from_dlpack(
                "tts_speech", to_dlpack(speech.unsqueeze(0)))
            responses.append(pb_utils.InferenceResponse(output_tensors=[speech_tensor]))

        return responses
