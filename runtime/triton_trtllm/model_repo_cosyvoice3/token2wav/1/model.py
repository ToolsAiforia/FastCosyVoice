import json
import os
import logging
import queue

import torch
import numpy as np
from torch.utils.dlpack import to_dlpack
import triton_python_backend_utils as pb_utils
from hyperpyyaml import load_hyperpyyaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TrtContextWrapper:
    def __init__(self, trt_engine, trt_concurrent=1, device='cuda:0'):
        self.trt_context_pool = queue.Queue(maxsize=trt_concurrent)
        self.trt_engine = trt_engine
        self.device = device
        for _ in range(trt_concurrent):
            trt_context = trt_engine.create_execution_context()
            trt_stream = torch.cuda.stream(torch.cuda.Stream(torch.device(device)))
            assert trt_context is not None
            self.trt_context_pool.put([trt_context, trt_stream])

    def acquire_estimator(self):
        return self.trt_context_pool.get(), self.trt_engine

    def release_estimator(self, context, stream):
        self.trt_context_pool.put([context, stream])


def convert_onnx_to_trt(trt_model, trt_kwargs, onnx_model, fp16, autocast_mode=False):
    import tensorrt as trt
    logging.info("Converting onnx to trt...")
    if autocast_mode:
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    else:
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    trt_logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, trt_logger)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 32)
    if not autocast_mode and fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    profile = builder.create_optimization_profile()
    with open(onnx_model, "rb") as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise ValueError(f'failed to parse {onnx_model}')
    for i in range(len(trt_kwargs['input_names'])):
        profile.set_shape(trt_kwargs['input_names'][i],
                          trt_kwargs['min_shape'][i],
                          trt_kwargs['opt_shape'][i],
                          trt_kwargs['max_shape'][i])
    if not autocast_mode:
        tensor_dtype = trt.DataType.HALF if fp16 else trt.DataType.FLOAT
        for i in range(network.num_inputs):
            network.get_input(i).dtype = tensor_dtype
        for i in range(network.num_outputs):
            network.get_output(i).dtype = tensor_dtype
    config.add_optimization_profile(profile)
    engine_bytes = builder.build_serialized_network(network, config)
    with open(trt_model, "wb") as f:
        f.write(engine_bytes)
    logging.info("Successfully converted onnx to trt")

torch.set_num_threads(1)


class TritonPythonModel:
    """Triton Python model for CosyVoice3 token2wav (flow-only, stateless).

    Converts speech tokens to mel spectrogram using the CausalMaskedDiffWithDiT flow model.
    """

    def initialize(self, args):
        parameters = json.loads(args['model_config'])['parameters']
        model_params = {k: v["string_value"] for k, v in parameters.items()}
        model_dir = model_params["model_dir"]

        self.device = torch.device("cuda")

        # Load flow model from cosyvoice3.yaml
        with open(os.path.join(model_dir, 'cosyvoice3.yaml'), 'r') as f:
            configs = load_hyperpyyaml(f, overrides={
                'qwen_pretrain_path': os.path.join(model_dir, 'CosyVoice-BlankEN')
            })
        self.flow = configs['flow']
        self.fp16 = True
        self.flow.half()
        self.flow.load_state_dict(
            torch.load(os.path.join(model_dir, 'flow.pt'),
                        map_location='cpu', weights_only=True),
            strict=True
        )
        self.flow.to(self.device).eval()

        # n_timesteps experiment 2026-05-19: nt=6 показал peak/clip audit clean,
        # но subjective listening выявил артефакты — REVERTED to default nt=10.
        # См. bench_n_timesteps_sweep/SUMMARY.md
        # self.flow.n_timesteps stays at 10 (default)

        # TRT acceleration for flow decoder estimator
        self.load_trt(model_dir)

        self.token_mel_ratio = self.flow.token_mel_ratio
        logger.info(f"Token2wav (flow-only) initialized, token_mel_ratio={self.token_mel_ratio}")

    def load_trt(self, model_dir, trt_concurrent=1):
        """Load layer-mixed precision TRT engine (production winner 2026-05-19).

        Strategy: pure FP32 ONNX → FP16 default + FP32 override for sensitive layers
        (Normalization, Softmax, time_embed, proj_out). Per-call compute -25%, TTFA
        p95 @ N=8 -14%, all audio metrics equal or better than autocast baseline
        (WER 5.21% vs 5.96%, UTMOS 3.360 identical, peak/clip clean).
        See bench_n_timesteps_sweep/layer_mixed/EVAL_SUITE_RESULTS.md
        """
        device_id = torch.cuda.current_device()
        # Tier-4: B-dynamic plan (B=2..16) for real GPU batching from BLS coordinator
        trt_path = os.path.join(
            model_dir, f'flow.decoder.estimator.layer_mixed_B8_fp16.{device_id}.plan')

        if not os.path.exists(trt_path) or os.path.getsize(trt_path) == 0:
            onnx_path = os.path.join(model_dir, 'flow.decoder.estimator.fp32.onnx')
            self._build_layer_mixed_trt(onnx_path, trt_path)

        del self.flow.decoder.estimator
        import tensorrt as trt
        with open(trt_path, 'rb') as f:
            estimator_engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(f.read())
        assert estimator_engine is not None, f'failed to load trt {trt_path}'
        self.flow.decoder.estimator = TrtContextWrapper(
            estimator_engine, trt_concurrent=trt_concurrent, device=str(self.device))
        logger.info(f"Loaded layer-mixed TRT engine: {os.path.basename(trt_path)}")

    def _build_layer_mixed_trt(self, onnx_path, trt_path):
        """Build TRT engine with per-layer mixed precision from pure FP32 ONNX."""
        import tensorrt as trt
        assert os.path.exists(onnx_path), f"Missing FP32 ONNX: {onnx_path}"
        logger.info(f"Building layer-mixed TRT plan from {os.path.basename(onnx_path)} "
                    f"(takes ~1-2 min, can OOM if VRAM <12 GB free)")

        trt_logger = trt.Logger(trt.Logger.WARNING)
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        builder = trt.Builder(trt_logger)
        network = builder.create_network(network_flags)
        parser = trt.OnnxParser(network, trt_logger)
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 * (1 << 30))
        # Mixed precision: default FP16, TRT respects per-layer FP32 hints
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)

        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    logger.error(parser.get_error(i))
                raise ValueError(f'failed to parse {onnx_path}')

        # Force FP16 IO (compatible with PyTorch runtime)
        for i in range(network.num_inputs):
            network.get_input(i).dtype = trt.DataType.HALF
        for i in range(network.num_outputs):
            network.get_output(i).dtype = trt.DataType.HALF

        # Mark precision-sensitive layers as FP32
        SKIP_TYPES = {trt.LayerType.CONSTANT, trt.LayerType.CAST, trt.LayerType.SHAPE,
                      trt.LayerType.GATHER, trt.LayerType.SLICE, trt.LayerType.SHUFFLE,
                      trt.LayerType.CONCATENATION, trt.LayerType.IDENTITY}
        SENSITIVE_ACT = {trt.ActivationType.SIGMOID, trt.ActivationType.TANH,
                         trt.ActivationType.HARD_SIGMOID, trt.ActivationType.ELU,
                         trt.ActivationType.GELU_ERF, trt.ActivationType.GELU_TANH}
        SENSITIVE_UNARY = {trt.UnaryOperation.EXP, trt.UnaryOperation.LOG,
                           trt.UnaryOperation.SQRT}
        FLOAT_LIKE = (trt.DataType.FLOAT, trt.DataType.HALF, trt.DataType.BF16)
        fp32_count = 0
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            if layer.type in SKIP_TYPES:
                continue
            name_low = layer.name.lower()
            should_fp32 = False
            if layer.type == trt.LayerType.SOFTMAX:
                should_fp32 = True
            elif layer.type == trt.LayerType.NORMALIZATION:
                should_fp32 = True
            elif layer.type == trt.LayerType.ACTIVATION:
                try:
                    should_fp32 = layer.algo_type in SENSITIVE_ACT
                except Exception:
                    pass
            elif layer.type == trt.LayerType.UNARY:
                try:
                    should_fp32 = layer.op in SENSITIVE_UNARY
                except Exception:
                    pass
            elif 'time_embed' in name_low or 'time_mlp' in name_low:
                should_fp32 = True
            elif 'proj_out' in name_low or 'out_proj' in name_low:
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

        logger.info(f"Marked {fp32_count}/{network.num_layers} layers as FP32 "
                    f"(Normalization/Softmax/time_embed/proj_out/sensitive activations)")

        # Optimization profile (production shapes)
        profile = builder.create_optimization_profile()
        shapes = {
            "x":    [(2, 80, 4), (2, 80, 500), (2, 80, 3000)],
            "mask": [(2, 1, 4),  (2, 1, 500),  (2, 1, 3000)],
            "mu":   [(2, 80, 4), (2, 80, 500), (2, 80, 3000)],
            "cond": [(2, 80, 4), (2, 80, 500), (2, 80, 3000)],
        }
        for name, (mn, op, mx) in shapes.items():
            profile.set_shape(name, mn, op, mx)
        config.add_optimization_profile(profile)

        engine_bytes = builder.build_serialized_network(network, config)
        if engine_bytes is None:
            raise RuntimeError("TRT build_serialized_network returned None — check VRAM")
        with open(trt_path, 'wb') as f:
            f.write(bytes(engine_bytes))
        logger.info(f"Built {trt_path} ({os.path.getsize(trt_path) // (1024*1024)} MB)")

    def get_trt_kwargs(self):
        min_shape = [(2, 80, 4), (2, 1, 4), (2, 80, 4), (2, 80, 4)]
        opt_shape = [(2, 80, 500), (2, 1, 500), (2, 80, 500), (2, 80, 500)]
        max_shape = [(2, 80, 3000), (2, 1, 3000), (2, 80, 3000), (2, 80, 3000)]
        input_names = ["x", "mask", "mu", "cond"]
        return {'min_shape': min_shape, 'opt_shape': opt_shape,
                'max_shape': max_shape, 'input_names': input_names}

    def _extract_request(self, request):
        """Pull all input tensors from one request → dict (CUDA tensors)."""
        target = pb_utils.get_input_tensor_by_name(request, "target_speech_tokens")
        target = torch.utils.dlpack.from_dlpack(target.to_dlpack()).to(self.device)
        if target.dim() == 1:
            target = target.unsqueeze(0)

        prompt_pb = pb_utils.get_input_tensor_by_name(request, "prompt_speech_tokens")
        if prompt_pb is None:
            raise ValueError("prompt_speech_tokens is required")
        prompt = torch.utils.dlpack.from_dlpack(prompt_pb.to_dlpack()).to(self.device)
        if prompt.dim() == 1:
            prompt = prompt.unsqueeze(0)

        pfeat = pb_utils.get_input_tensor_by_name(request, "prompt_speech_feat")
        pfeat = torch.utils.dlpack.from_dlpack(pfeat.to_dlpack()).to(self.device)
        if pfeat.dim() == 2:
            pfeat = pfeat.unsqueeze(0)

        spk = pb_utils.get_input_tensor_by_name(request, "prompt_spk_embedding")
        spk = torch.utils.dlpack.from_dlpack(spk.to_dlpack()).to(self.device)
        if spk.dim() == 1:
            spk = spk.unsqueeze(0)

        tok_off_pb = pb_utils.get_input_tensor_by_name(request, "token_offset")
        fin_pb = pb_utils.get_input_tensor_by_name(request, "finalize")
        # For B>1 batches BLS already slices mel per-request, so token_offset
        # is uniform-zero placeholder — take first scalar without .item() crash.
        tok_off = int(tok_off_pb.as_numpy().flatten()[0]) if tok_off_pb is not None else None
        finalize = bool(fin_pb.as_numpy().flatten()[0]) if fin_pb is not None else True

        return {
            'target': target,           # [1, T_target]
            'prompt': prompt,           # [1, T_prompt]
            'pfeat':  pfeat,            # [1, T_pfeat, 80]
            'spk':    spk,              # [1, 192]
            'tok_off': tok_off,
            'finalize': finalize,
        }

    def _make_response(self, mel, tok_off):
        if tok_off is not None:
            mel = mel[:, tok_off * self.token_mel_ratio:]
        mel_out = mel.float().unsqueeze(0).cpu()  # [1, 80, T] for max_batch_size>1 compat
        return pb_utils.InferenceResponse(
            output_tensors=[pb_utils.Tensor.from_dlpack("mel", to_dlpack(mel_out))])

    def execute(self, requests):
        """Tier-4 coordinator-friendly execute:
        - BLS sends ONE pb_utils.InferenceRequest with batched tensors [B, ...].
        - We dispatch via flow.inference_batched() (B>=1, uniform shape).
        - Triton dynamic_batching with multiple requests fallback: group-by-shape.

        token_offset slicing is done in BLS (since per-request offset varies inside
        a batch). We always return full mel; BLS slices per-request.

        Tier-B B2: first chunk (token_offset==0, non-finalize) uses reduced
        n_timesteps=5 instead of default 10. Halves DiT compute on first chunk
        only — direct TTFA win. Subsequent chunks keep full quality at n=10.
        """
        responses = []
        for request in requests:
            d = self._extract_request(request)
            B = d['target'].shape[0]
            T_prompt = d['prompt'].shape[1]
            T_target = d['target'].shape[1]
            T_pfeat  = d['pfeat'].shape[1]
            finalize = d['finalize']
            streaming = not finalize
            # B2: first-chunk fast path — only for streaming (non-finalize),
            # first chunk has tok_off==0
            n_timesteps_override = 5 if (not finalize and d.get('tok_off') == 0) else None

            with torch.no_grad(), torch.cuda.amp.autocast(self.fp16):
                if B == 1:
                    mel, _ = self.flow.inference(
                        token=d['target'],
                        token_len=torch.tensor([T_target], dtype=torch.int32, device=self.device),
                        prompt_token=d['prompt'],
                        prompt_token_len=torch.tensor([T_prompt], dtype=torch.int32, device=self.device),
                        prompt_feat=d['pfeat'],
                        prompt_feat_len=torch.tensor([T_pfeat], dtype=torch.int32, device=self.device),
                        embedding=d['spk'],
                        streaming=streaming, finalize=finalize,
                        n_timesteps_override=n_timesteps_override)
                else:
                    # Real GPU batching — uniform shape (BLS coordinator guarantees)
                    lens_t = torch.tensor([T_target] * B, dtype=torch.int32, device=self.device)
                    lens_p = torch.tensor([T_prompt] * B, dtype=torch.int32, device=self.device)
                    lens_f = torch.tensor([T_pfeat]  * B, dtype=torch.int32, device=self.device)
                    mel, _ = self.flow.inference_batched(
                        token=d['target'], token_len=lens_t,
                        prompt_token=d['prompt'], prompt_token_len=lens_p,
                        prompt_feat=d['pfeat'], prompt_feat_len=lens_f,
                        embedding=d['spk'],
                        streaming=streaming, finalize=finalize,
                        n_timesteps_override=n_timesteps_override)
            # mel: [B, 80, T_mel] (B may be 1). Output to Triton — keep batch dim.
            mel_out = mel.float().cpu()  # already has [B, 80, T] shape from flow
            responses.append(pb_utils.InferenceResponse(
                output_tensors=[pb_utils.Tensor.from_dlpack("mel", to_dlpack(mel_out))]))
        return responses
