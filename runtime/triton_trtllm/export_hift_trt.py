#!/usr/bin/env python3
"""Export CausalHiFTGenerator middle compute (after conv_pre, before iSTFT) to ONNX → TRT.

Boundary:
- Input:  x_pre [B, base_channels, T_mel]   (already conv_pre output)
          s_stft [B, n_fft+2, T_stft]        (already STFT of source signal)
- Output: magnitude [B, n_fft/2+1, T_aud_stft]
          phase     [B, n_fft/2+1, T_aud_stft]

iSTFT, conv_pre, STFT, conditional finalize trim — остаются в Python wrapper.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hyperpyyaml import load_hyperpyyaml


class HiFTDecodeCore(nn.Module):
    """Pure GPU compute middle of CausalHiFTGenerator.decode (finalize=True path)."""

    def __init__(self, hift):
        super().__init__()
        self.ups = hift.ups
        self.reflection_pad = hift.reflection_pad
        self.source_downs = hift.source_downs
        self.source_resblocks = hift.source_resblocks
        self.resblocks = hift.resblocks
        self.conv_post = hift.conv_post
        self.num_upsamples = hift.num_upsamples
        self.num_kernels = hift.num_kernels
        self.lrelu_slope = hift.lrelu_slope
        self.n_fft_half_plus_one = hift.istft_params["n_fft"] // 2 + 1

    def forward(self, x_pre: torch.Tensor, s_stft: torch.Tensor):
        x = x_pre
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, self.lrelu_slope)
            x = self.ups[i](x)
            if i == self.num_upsamples - 1:
                x = self.reflection_pad(x)
            si = self.source_downs[i](s_stft)
            si = self.source_resblocks[i](si)
            x = x + si
            xs = None
            for j in range(self.num_kernels):
                rb = self.resblocks[i * self.num_kernels + j](x)
                xs = rb if xs is None else xs + rb
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        magnitude = torch.exp(x[:, :self.n_fft_half_plus_one, :])
        phase = torch.sin(x[:, self.n_fft_half_plus_one:, :])
        return magnitude, phase


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True,
                    help="Path to Fun-CosyVoice3-0.5B-2512/ (contains hift.pt + cosyvoice3.yaml)")
    ap.add_argument("--onnx-out", default=None,
                    help="Output ONNX path (default: <model-dir>/hift_decode_core.onnx)")
    ap.add_argument("--cosyvoice-root", default=None,
                    help="Path to CosyVoice/ root (for sys.path). If unset, assumes cosyvoice/ "
                         "is already importable.")
    ap.add_argument("--verify", action="store_true", help="run numeric verification vs eager")
    args = ap.parse_args()

    if args.onnx_out is None:
        args.onnx_out = os.path.join(args.model_dir, "hift_decode_core.onnx")
    if args.cosyvoice_root:
        sys.path.insert(0, args.cosyvoice_root)

    # Load HiFT
    print(f"Loading hift from {args.model_dir} ...")
    with open(os.path.join(args.model_dir, "cosyvoice3.yaml"), "r") as f:
        configs = load_hyperpyyaml(f, overrides={
            "qwen_pretrain_path": os.path.join(args.model_dir, "CosyVoice-BlankEN")
        })
    hift = configs["hift"]
    state = {k.replace("generator.", ""): v
             for k, v in torch.load(os.path.join(args.model_dir, "hift.pt"),
                                     map_location="cpu", weights_only=True).items()}
    hift.load_state_dict(state, strict=True)
    hift = hift.cuda().eval()
    # weight_norm to plain weight for cleaner ONNX export
    for mod in hift.modules():
        if hasattr(mod, "weight_v"):
            try:
                torch.nn.utils.remove_weight_norm(mod)
            except Exception:
                pass

    core = HiFTDecodeCore(hift).cuda().eval()

    # Build sample input. token_hop_len=8 produces mel chunks of ~8 frames first chunk
    # (or 9 with lookahead). conv_pre output is [B, base_channels=512, T_mel]. T_mel grows
    # over streaming (no shrink): first chunk 9, then accumulates each step.
    # We support dynamic T_mel via ONNX dynamic_axes.
    n_fft = hift.istft_params["n_fft"]
    upsample_total = int(np.prod(hift.upsample_rates))
    hop_len = hift.istft_params["hop_len"]

    # Mel sample: 25 frames (typical mid-streaming chunk)
    T_mel = 25
    x_pre = torch.randn(1, 512, T_mel, device="cuda")
    # s_stft sample: source signal upsampled before STFT. After STFT with hop_len=4, T_stft ≈ ceil((T_audio - n_fft)/hop) + 1
    T_audio = T_mel * upsample_total * hop_len  # mel → audio frames before istft
    # STFT of source pre-istft produces [B, n_fft+2, T_stft] where T_stft = T_audio/hop_len
    T_stft = T_audio // hop_len + 1
    s_stft = torch.randn(1, n_fft + 2, T_stft, device="cuda")

    print(f"Sample shapes: x_pre={tuple(x_pre.shape)}, s_stft={tuple(s_stft.shape)} "
          f"(n_fft={n_fft}, upsample_total={upsample_total}, hop_len={hop_len})")

    with torch.no_grad():
        out_mag, out_pha = core(x_pre, s_stft)
    print(f"Output: magnitude={tuple(out_mag.shape)}, phase={tuple(out_pha.shape)}")

    Path(os.path.dirname(args.onnx_out)).mkdir(parents=True, exist_ok=True)
    print(f"\nExporting to {args.onnx_out} ...")
    torch.onnx.export(
        core,
        (x_pre, s_stft),
        args.onnx_out,
        input_names=["x_pre", "s_stft"],
        output_names=["magnitude", "phase"],
        dynamic_axes={
            "x_pre":     {0: "B", 2: "T_mel"},
            "s_stft":    {0: "B", 2: "T_stft"},
            "magnitude": {0: "B", 2: "T_aud_stft"},
            "phase":     {0: "B", 2: "T_aud_stft"},
        },
        opset_version=17,
        do_constant_folding=True,
        verbose=False,
    )
    sz = os.path.getsize(args.onnx_out) / 1024 / 1024
    print(f"ONNX saved: {args.onnx_out} ({sz:.1f} MB)")

    if args.verify:
        import onnxruntime
        sess = onnxruntime.InferenceSession(args.onnx_out, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        # Try different shape
        T_mel2 = 50
        x2 = torch.randn(1, 512, T_mel2, device="cuda")
        T_aud2 = T_mel2 * upsample_total * hop_len
        s2 = torch.randn(1, n_fft + 2, T_aud2 // hop_len + 1, device="cuda")
        with torch.no_grad():
            mag_eager, pha_eager = core(x2, s2)
        mag_ort, pha_ort = sess.run(None, {"x_pre": x2.cpu().numpy(), "s_stft": s2.cpu().numpy()})
        d_mag = (torch.from_numpy(mag_ort).cuda() - mag_eager).abs().max().item()
        d_pha = (torch.from_numpy(pha_ort).cuda() - pha_eager).abs().max().item()
        print(f"\nVerify (T_mel={T_mel2}): max|Δmag|={d_mag:.4e}, max|Δpha|={d_pha:.4e}")


if __name__ == "__main__":
    main()
