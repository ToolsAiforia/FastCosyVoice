#!/usr/bin/env python3
"""Build HiFT decode_core TRT engine with dynamic batch dim (min=1, opt=4, max=8).

Output plan file: <model-dir>/hift_decode_core.fp32_B8.plan

The B=1 plan (hift_decode_core.fp32.plan) is built by vocoder/1/model.py on
first execute() call. This B-dynamic plan needs to be pre-built so the BLS
coordinator can dispatch batched B>=2 mel tensors without falling back to
per-row B=1 inference.

Profile time-dim matches the existing B=1 plan:
  - x_pre: (1..8, 512, 1..2500), opt=(4, 512, 16)
  - s_stft: (1..8, 18, 121..300001), opt=(4, 18, 1921)
"""
import argparse
import os
import sys


def main():
    p = argparse.ArgumentParser(description="Build HiFT B-dynamic TRT plan")
    p.add_argument("--model-dir", required=True,
                   help="Path to model dir containing hift_decode_core.onnx")
    p.add_argument("--output-name", default="hift_decode_core.fp32_B8.plan",
                   help="Plan filename (default: hift_decode_core.fp32_B8.plan)")
    p.add_argument("--max-batch", type=int, default=8, help="Max batch (default: 8)")
    p.add_argument("--opt-batch", type=int, default=4, help="Opt batch (default: 4)")
    args = p.parse_args()

    sys.path.insert(0, "/workspace/CosyVoice")
    from cosyvoice.utils.file_utils import convert_onnx_to_trt

    onnx_path = os.path.join(args.model_dir, "hift_decode_core.onnx")
    plan_path = os.path.join(args.model_dir, args.output_name)

    if not os.path.exists(onnx_path):
        sys.exit(f"ERROR: ONNX not found at {onnx_path}")

    if os.path.exists(plan_path) and os.path.getsize(plan_path) > 0:
        print(f"Plan already exists at {plan_path} ({os.path.getsize(plan_path)//1024//1024} MB), skipping.")
        return

    B_max = args.max_batch
    B_opt = args.opt_batch
    trt_kwargs = {
        "min_shape": [(1,    512, 1),    (1,    18, 121)],
        "opt_shape": [(B_opt, 512, 16),  (B_opt, 18, 1921)],
        "max_shape": [(B_max, 512, 2500), (B_max, 18, 300001)],
        "input_names": ["x_pre", "s_stft"],
    }
    print(f"Building B-dynamic HiFT plan: {plan_path}")
    print(f"  min={trt_kwargs['min_shape']}")
    print(f"  opt={trt_kwargs['opt_shape']}")
    print(f"  max={trt_kwargs['max_shape']}")

    convert_onnx_to_trt(plan_path, trt_kwargs, onnx_path, fp16=False)
    print(f"Built {plan_path}, size={os.path.getsize(plan_path)/(1024*1024):.1f} MB")


if __name__ == "__main__":
    main()
