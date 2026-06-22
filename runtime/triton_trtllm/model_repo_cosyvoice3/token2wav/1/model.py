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
        # --- Re-optimization ladder knobs (config-driven; round-9 defaults). ---
        #   flow_precision: "fp16" (round-9) | "fp32" (vanilla baseline)
        #   flow_trt:       "1" (round-9, layer_mixed_fp16 plan) | "0" (pure PyTorch)
        self.fp16 = model_params.get("flow_precision", "fp16") == "fp16"
        self.flow_trt = model_params.get("flow_trt", "1") == "1"
        if self.fp16:
            self.flow.half()
        self.flow.load_state_dict(
            torch.load(os.path.join(model_dir, 'flow.pt'),
                        map_location='cpu', weights_only=True),
            strict=True
        )
        self.flow.to(self.device).eval()

        # TRT acceleration for flow decoder estimator (round-9 layer-mixed plan).
        # Disabled for the vanilla-equivalent baseline (pure PyTorch estimator).
        if self.flow_trt:
            self.load_trt(model_dir)
        logger.info(f"Token2wav flow_precision={'fp16' if self.fp16 else 'fp32'}, "
                    f"flow_trt={self.flow_trt}")

        self.token_mel_ratio = self.flow.token_mel_ratio
        logger.info(f"Token2wav (flow-only) initialized, token_mel_ratio={self.token_mel_ratio}")

    def load_trt(self, model_dir, trt_concurrent=1):
        """Load layer-mixed precision TRT engine (round-9 production winner).

        Strategy: pure FP32 ONNX → FP16 default + FP32 override for sensitive
        layers (Normalization, Softmax, time_embed, proj_out). Per-call compute
        −25%, TTFA p95 @ N=8 −14%, all audio metrics equal or better than
        autocast baseline (WER 5.21% vs 5.96%, UTMOS 3.360 identical).
        Opt profile B=2 (CFG conditioned/unconditioned packed).
        """
        device_id = torch.cuda.current_device()
        trt_path = os.path.join(
            model_dir, f'flow.decoder.estimator.layer_mixed_fp16.{device_id}.plan')

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

        # Mark precision-sensitive layers as FP32 (75 layers in round-9)
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

        # Optimization profile: B=2 (CFG conditioned/unconditioned packed)
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

    def execute(self, requests):
        responses = []
        for req_idx, request in enumerate(requests):
            target_speech_tokens = pb_utils.get_input_tensor_by_name(
                request, "target_speech_tokens")
            target_speech_tokens = torch.utils.dlpack.from_dlpack(
                target_speech_tokens.to_dlpack()).to(self.device)
            if target_speech_tokens.dim() == 1:
                target_speech_tokens = target_speech_tokens.unsqueeze(0)

            # Optional inputs
            prompt_speech_tokens_pb = pb_utils.get_input_tensor_by_name(
                request, "prompt_speech_tokens")
            if prompt_speech_tokens_pb is not None:
                prompt_speech_tokens = torch.utils.dlpack.from_dlpack(
                    prompt_speech_tokens_pb.to_dlpack()).to(self.device)
                if prompt_speech_tokens.dim() == 1:
                    prompt_speech_tokens = prompt_speech_tokens.unsqueeze(0)

                prompt_speech_feat = pb_utils.get_input_tensor_by_name(
                    request, "prompt_speech_feat")
                prompt_speech_feat = torch.utils.dlpack.from_dlpack(
                    prompt_speech_feat.to_dlpack()).to(self.device)
                if prompt_speech_feat.dim() == 2:
                    prompt_speech_feat = prompt_speech_feat.unsqueeze(0)  # [T, 80] -> [1, T, 80]

                prompt_spk_embedding = pb_utils.get_input_tensor_by_name(
                    request, "prompt_spk_embedding")
                prompt_spk_embedding = torch.utils.dlpack.from_dlpack(
                    prompt_spk_embedding.to_dlpack()).to(self.device)
                if prompt_spk_embedding.dim() == 1:
                    prompt_spk_embedding = prompt_spk_embedding.unsqueeze(0)
                # Inputs are transported as fp32; cast to the flow's dtype (fp16 or
                # fp32 per the flow_precision knob) so the matmuls match. Robust to
                # warmup samples and to either precision baseline.
                _dt = torch.float16 if self.fp16 else torch.float32
                prompt_speech_feat = prompt_speech_feat.to(_dt)
                prompt_spk_embedding = prompt_spk_embedding.to(_dt)
            else:
                raise ValueError("prompt_speech_tokens is required for CosyVoice3 token2wav")

            token_offset_pb = pb_utils.get_input_tensor_by_name(request, "token_offset")
            finalize_pb = pb_utils.get_input_tensor_by_name(request, "finalize")

            token_offset = token_offset_pb.as_numpy().item() if token_offset_pb is not None else None
            finalize = finalize_pb.as_numpy().item() if finalize_pb is not None else True
            streaming = not finalize

            with torch.no_grad(), torch.cuda.amp.autocast(self.fp16):
                mel, _ = self.flow.inference(
                    token=target_speech_tokens,
                    token_len=torch.tensor([target_speech_tokens.shape[1]], dtype=torch.int32).to(self.device),
                    prompt_token=prompt_speech_tokens,
                    prompt_token_len=torch.tensor([prompt_speech_tokens.shape[1]], dtype=torch.int32).to(self.device),
                    prompt_feat=prompt_speech_feat,
                    prompt_feat_len=torch.tensor([prompt_speech_feat.shape[1]], dtype=torch.int32).to(self.device),
                    embedding=prompt_spk_embedding,
                    streaming=streaming,
                    finalize=finalize,
                )

            # Slice mel from token_offset if provided
            if token_offset is not None:
                mel = mel[:, :, token_offset * self.token_mel_ratio:]

            # Output mel as [80, T] (squeeze batch dim for Triton)
            mel_out = mel.squeeze(0).float()  # [80, T]
            mel_out = mel_out.cpu() # otherwise, dlpack bug
            mel_tensor = pb_utils.Tensor.from_dlpack("mel", to_dlpack(mel_out))
            inference_response = pb_utils.InferenceResponse(output_tensors=[mel_tensor])
            responses.append(inference_response)

        return responses
