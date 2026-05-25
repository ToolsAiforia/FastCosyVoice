#!/usr/bin/env python3
"""CosyVoice3 round-9-stable synthesis CLI.

Zero-shot voice cloning via reference audio + transcription.
(round-9-stable does NOT use spk2info — uses reference_wav per request.)

Usage:
  # Single text
  python synth_round9.py --text "Привет мир." --ref-wav runtime/milena.ogg \
      --ref-text "Транскрипция референса" -o out.wav

  # Batch (one line per prompt)
  python synth_round9.py --batch prompts.txt --ref-wav ref.wav \
      --ref-text "..." --outdir ./out

  # Stdin
  echo "Test" | python synth_round9.py --ref-wav ref.wav --ref-text "..." --outdir ./out
"""
import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import tritonclient.grpc.aio as grpc_aio
from tritonclient.grpc import InferInput, InferRequestedOutput


MODEL = "cosyvoice3"


def load_reference(path, target_sr=16000):
    """Load reference audio, convert to mono 16kHz float32."""
    data, sr = sf.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32)
    if sr != target_sr:
        try:
            import librosa
            data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
        except ImportError:
            print(f"warning: librosa missing; reference sr={sr} != {target_sr}",
                  file=sys.stderr)
    return data


def build_inputs(ref_wav, ref_text, target_text):
    samples = ref_wav.reshape(1, -1).astype(np.float32)
    lengths = np.array([[samples.shape[1]]], dtype=np.int32)
    ref_np = np.array([[ref_text.encode("utf-8")]], dtype=object)
    tgt_np = np.array([[target_text.encode("utf-8")]], dtype=object)
    inputs = [
        InferInput("reference_wav", samples.shape, "FP32"),
        InferInput("reference_wav_len", lengths.shape, "INT32"),
        InferInput("reference_text", ref_np.shape, "BYTES"),
        InferInput("target_text", tgt_np.shape, "BYTES"),
    ]
    inputs[0].set_data_from_numpy(samples)
    inputs[1].set_data_from_numpy(lengths)
    inputs[2].set_data_from_numpy(ref_np)
    inputs[3].set_data_from_numpy(tgt_np)
    return inputs


async def synthesize(client, idx, ref_wav, ref_text, target_text, out_path):
    inputs = build_inputs(ref_wav, ref_text, target_text)
    outputs = [InferRequestedOutput("waveform")]
    chunks = []
    ttfa = None
    t0 = time.time()
    err = None

    async def gen():
        yield {"model_name": MODEL, "inputs": inputs, "outputs": outputs,
               "request_id": f"synth_{idx}_{int(t0*1000)}"}

    try:
        async for r, e in client.stream_infer(inputs_iterator=gen(), stream_timeout=120):
            if e is not None:
                err = str(e); break
            if ttfa is None:
                ttfa = (time.time() - t0) * 1000
            wav = r.as_numpy("waveform")
            if wav is not None and wav.size:
                chunks.append(wav.flatten())
    except Exception as e:
        err = str(e)

    total = (time.time() - t0) * 1000
    if err or not chunks:
        return {"err": err or "no audio", "total_ms": total}
    full = np.concatenate(chunks)
    sf.write(out_path, full, 24000)
    return {
        "out": out_path,
        "ttfa_ms": ttfa, "total_ms": total,
        "duration_s": len(full) / 24000,
        "peak": float(np.max(np.abs(full))),
        "clip_pct": float(np.mean(np.abs(full) > 0.99)) * 100,
        "rtf": total / 1000.0 / max(len(full) / 24000, 0.001),
    }


def slugify(text, maxlen=40):
    safe = "".join(c if c.isalnum() or c in " -_" else "_" for c in text)
    return "_".join(safe.split())[:maxlen]


def iter_prompts(args):
    outdir = Path(args.outdir) if args.outdir else None
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)
    def out_path(idx, text):
        if args.output and not args.batch:
            return args.output
        name = f"{idx:03d}_{slugify(text)}.wav"
        return str(outdir / name) if outdir else name
    if args.text:
        yield 0, args.text, args.output or out_path(0, args.text)
    elif args.batch:
        with open(args.batch) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if line and not line.startswith("#"):
                    yield i, line, out_path(i, line)
    else:
        print(f"Reading prompts from stdin (Ctrl-D to stop)...", file=sys.stderr)
        for i, line in enumerate(sys.stdin):
            line = line.strip()
            if line:
                yield i, line, out_path(i, line)


async def amain(args):
    print(f"Loading reference from {args.ref_wav}", file=sys.stderr)
    ref_wav = load_reference(args.ref_wav)
    print(f"  reference: {len(ref_wav)/16000:.2f}s @ 16kHz", file=sys.stderr)
    print(f"  reference_text: {args.ref_text[:60]!r}...", file=sys.stderr)

    client = grpc_aio.InferenceServerClient(url=args.url, verbose=False)
    try:
        results = []
        for idx, text, out_path in iter_prompts(args):
            print(f"[{idx}] synth: {text[:60]!r}{'...' if len(text)>60 else ''}")
            res = await synthesize(client, idx, ref_wav, args.ref_text, text, out_path)
            if res.get("err"):
                print(f"      ERROR: {res['err']}", file=sys.stderr)
            else:
                print(f"      → {res['out']}  TTFA={res['ttfa_ms']:.0f}ms "
                      f"total={res['total_ms']:.0f}ms dur={res['duration_s']:.2f}s "
                      f"RTF={res['rtf']:.3f} peak={res['peak']:.3f} clip%={res['clip_pct']:.2f}")
            results.append(res)
        ok = [r for r in results if not r.get("err")]
        if len(results) > 1 and ok:
            ttfas = [r["ttfa_ms"] for r in ok]
            print(f"\n=== Summary: {len(ok)}/{len(results)} ok ===")
            print(f"  TTFA  avg={np.mean(ttfas):.0f}ms min={min(ttfas):.0f}ms max={max(ttfas):.0f}ms")
            print(f"  Total audio: {sum(r['duration_s'] for r in ok):.1f}s")
    finally:
        await client.close()


def main():
    p = argparse.ArgumentParser(description="CosyVoice3 round-9 zero-shot synth CLI")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--text", "-t", help="Single text to synthesize")
    src.add_argument("--batch", "-b", help="File with prompts (one per line)")
    p.add_argument("--output", "-o", help="Output wav (for single --text)")
    p.add_argument("--outdir", "-d", default="./out", help="Output dir for batch/stdin")
    p.add_argument("--ref-wav", required=True, help="Reference audio (wav/ogg/mp3)")
    p.add_argument("--ref-text", required=True, help="Transcription of reference")
    p.add_argument("--url", default="127.0.0.1:18001", help="Triton gRPC endpoint")
    args = p.parse_args()
    if not (args.text or args.batch) and sys.stdin.isatty():
        p.print_help()
        sys.exit("\nNothing to synthesize. Pass --text or --batch, or pipe via stdin.")
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
