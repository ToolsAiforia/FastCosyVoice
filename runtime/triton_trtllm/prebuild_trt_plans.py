#!/usr/bin/env python3
"""Pre-build TRT plans for token2wav (DiT) and vocoder (HiFT) BEFORE Triton start.

Problem: with instance_group.count >= 2, multiple Python stubs initialize
in parallel, all detect missing plan, all call _build_*_trt() at the same
time, all write to the same file → partial writes corrupt the .plan →
deserialization assertion: `plan.header.size == blobSize failed`.

Solution: build plans once (single-process) in entrypoint Step 0.6, before
launching Triton. Then model.py initialize() sees existing files and just
loads them — no race.

This script is idempotent — skips any plan that already exists.

Used by entrypoint_cosyvoice3.sh:
    python3 /workdir/scripts/prebuild_trt_plans.py --model-dir "${MODEL_DIR}"
"""
import argparse
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s [prebuild_trt_plans] %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def build_dit_plan(model_dir: str) -> None:
    """Build DiT (flow.decoder.estimator) layer-mixed FP16 plan.

    Replicates token2wav/1/model.py:_build_layer_mixed_trt (round-9-stable).
    """
    import tensorrt as trt

    device_id = 0  # entrypoint pins CUDA_VISIBLE_DEVICES=0; model.py uses current_device()
    onnx_path = os.path.join(model_dir, 'flow.decoder.estimator.fp32.onnx')
    trt_path = os.path.join(model_dir, f'flow.decoder.estimator.layer_mixed_fp16.{device_id}.plan')

    if os.path.exists(trt_path) and os.path.getsize(trt_path) > 0:
        logger.info(f"DiT plan exists, skipping: {os.path.basename(trt_path)}")
        return

    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"DiT FP32 ONNX missing: {onnx_path}")

    logger.info(f"Building DiT layer-mixed FP16 plan from {os.path.basename(onnx_path)} ...")

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

    # FP16 IO
    for i in range(network.num_inputs):
        network.get_input(i).dtype = trt.DataType.HALF
    for i in range(network.num_outputs):
        network.get_output(i).dtype = trt.DataType.HALF

    # Mark precision-sensitive layers as FP32 (matches token2wav/1/model.py)
    SKIP_TYPES = {trt.LayerType.CONSTANT, trt.LayerType.CAST, trt.LayerType.SHAPE,
                  trt.LayerType.GATHER, trt.LayerType.SLICE, trt.LayerType.SHUFFLE,
                  trt.LayerType.CONCATENATION, trt.LayerType.IDENTITY}
    SENSITIVE_ACT = {trt.ActivationType.SIGMOID, trt.ActivationType.TANH,
                     trt.ActivationType.HARD_SIGMOID, trt.ActivationType.ELU,
                     trt.ActivationType.GELU_ERF, trt.ActivationType.GELU_TANH}
    SENSITIVE_UNARY = {trt.UnaryOperation.EXP, trt.UnaryOperation.LOG, trt.UnaryOperation.SQRT}
    FLOAT_LIKE = (trt.DataType.FLOAT, trt.DataType.HALF, trt.DataType.BF16)
    fp32_count = 0
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        if layer.type in SKIP_TYPES:
            continue
        name_low = layer.name.lower()
        should_fp32 = False
        if layer.type == trt.LayerType.SOFTMAX or layer.type == trt.LayerType.NORMALIZATION:
            should_fp32 = True
        elif layer.type == trt.LayerType.ACTIVATION:
            try: should_fp32 = layer.algo_type in SENSITIVE_ACT
            except Exception: pass
        elif layer.type == trt.LayerType.UNARY:
            try: should_fp32 = layer.op in SENSITIVE_UNARY
            except Exception: pass
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
        except Exception: pass
        if should_fp32:
            try:
                layer.precision = trt.DataType.FLOAT
                for j in range(layer.num_outputs):
                    layer.set_output_type(j, trt.DataType.FLOAT)
                fp32_count += 1
            except Exception: pass

    logger.info(f"DiT: marked {fp32_count}/{network.num_layers} layers as FP32")

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
    # Atomic write — write to temp, then rename, чтобы partial writes не были видны
    tmp_path = trt_path + ".tmp"
    with open(tmp_path, 'wb') as f:
        f.write(bytes(engine_bytes))
    os.replace(tmp_path, trt_path)
    logger.info(f"Built DiT plan: {trt_path} ({os.path.getsize(trt_path) // (1024*1024)} MB)")


def build_hift_plans(model_dir: str) -> None:
    """Build HiFT layer-mixed + FP32 fallback plans.

    Imports _build_hift_layer_mixed_trt and _build_hift_fp32_trt from
    vocoder/1/model.py (module-level functions in round-9-stable).
    """
    layer_mixed_path = os.path.join(model_dir, 'hift_decode_core.layer_mixed_fp32io.plan')
    fp32_path = os.path.join(model_dir, 'hift_decode_core.fp32.plan')
    onnx_path = os.path.join(model_dir, 'hift_decode_core.onnx')

    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"HiFT ONNX missing: {onnx_path} "
                                "(run export_hift_trt.py first)")

    # Import builders from vocoder model.py
    sys.path.insert(0, '/model_repo/vocoder/1')
    try:
        from model import _build_hift_layer_mixed_trt, _build_hift_fp32_trt
    finally:
        sys.path.pop(0)

    if os.path.exists(layer_mixed_path) and os.path.getsize(layer_mixed_path) > 0:
        logger.info(f"HiFT layer-mixed plan exists, skipping: {os.path.basename(layer_mixed_path)}")
    else:
        logger.info(f"Building HiFT layer-mixed plan ...")
        tmp = layer_mixed_path + ".tmp"
        _build_hift_layer_mixed_trt(tmp, onnx_path)
        os.replace(tmp, layer_mixed_path)
        logger.info(f"Built HiFT layer-mixed plan: {layer_mixed_path}")

    if os.path.exists(fp32_path) and os.path.getsize(fp32_path) > 0:
        logger.info(f"HiFT FP32 plan exists, skipping: {os.path.basename(fp32_path)}")
    else:
        logger.info(f"Building HiFT FP32 fallback plan ...")
        tmp = fp32_path + ".tmp"
        _build_hift_fp32_trt(tmp, onnx_path)
        os.replace(tmp, fp32_path)
        logger.info(f"Built HiFT FP32 plan: {fp32_path}")


def main():
    ap = argparse.ArgumentParser(description="Pre-build TRT plans (single-process, idempotent)")
    ap.add_argument("--model-dir", required=True,
                    help="Directory with flow.decoder.estimator.fp32.onnx + hift_decode_core.onnx")
    args = ap.parse_args()

    if not os.path.isdir(args.model_dir):
        sys.exit(f"ERROR: model_dir not found: {args.model_dir}")

    build_dit_plan(args.model_dir)
    build_hift_plans(args.model_dir)
    logger.info("All TRT plans ready.")


if __name__ == "__main__":
    main()
