#!/usr/bin/env python3
"""CosyVoice3 streaming TTS synthesis CLI.

Sends one or many text prompts to the Triton-served CosyVoice3 pipeline and
saves the resulting audio. Supports:
  - Single text via --text "..."
  - Batch from a file (one prompt per line) via --batch prompts.txt
  - Stdin (default if neither --text nor --batch given)

Voice selection (priority order):
  - --speaker NAME      → uses cached speaker from spk2info.pt (fastest, no
                          audio_tokenizer/speaker_embedding compute)
  - --ref-wav FILE      → zero-shot clone from reference audio
  - default speaker     → if spk2info.pt has a `default_speaker` flag

Usage examples:
  # Single prompt with cached speaker 'ref'
  python synth.py --text "Hello world." --speaker ref -o /tmp/hello.wav

  # Batch synthesis, one file per line in prompts.txt
  python synth.py --batch prompts.txt --speaker ref --outdir ./out

  # Zero-shot clone from custom reference
  python synth.py --text "Hi" --ref-wav my_voice.wav \\
      --ref-text "Reference transcription" -o cloned.wav

  # Interactive stdin (Ctrl-D to exit), each line → one wav
  python synth.py --speaker ref --outdir ./out
"""
import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import tritonclient.grpc.aio as grpcclient_aio
from tritonclient.grpc import InferInput, InferRequestedOutput


MODEL = "cosyvoice3"


def make_inputs(text, *, speaker=None, ref_wav=None, ref_text=None, instruction=None):
    """Build Triton InferInput list. Voice selection: speaker > ref_wav.
    Optional `instruction` overrides the speaker's baked style prompt
    (e.g. "Speak with excitement", "Whisper softly").
    """
    tgt = np.array([[text.encode("utf-8")]], dtype=object)
    inputs = [InferInput("target_text", tgt.shape, "BYTES")]
    inputs[0].set_data_from_numpy(tgt)

    if instruction:
        instr_np = np.array([[instruction.encode("utf-8")]], dtype=object)
        inp_i = InferInput("instruction", instr_np.shape, "BYTES")
        inp_i.set_data_from_numpy(instr_np)
        inputs.append(inp_i)

    if speaker:
        spk_np = np.array([[speaker.encode("utf-8")]], dtype=object)
        inp = InferInput("speaker_name", spk_np.shape, "BYTES")
        inp.set_data_from_numpy(spk_np)
        inputs.append(inp)
    elif ref_wav is not None:
        wav, sr = sf.read(ref_wav)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != 16000:
            try:
                import librosa
                wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=16000)
            except ImportError:
                print(f"warning: ref wav sr={sr} != 16000, install librosa or pre-resample", file=sys.stderr)
        samples = wav.reshape(1, -1).astype(np.float32)
        lens = np.array([[samples.shape[1]]], dtype=np.int32)
        inp_w = InferInput("reference_wav", samples.shape, "FP32")
        inp_w.set_data_from_numpy(samples)
        inp_l = InferInput("reference_wav_len", lens.shape, "INT32")
        inp_l.set_data_from_numpy(lens)
        inputs += [inp_w, inp_l]
        if ref_text:
            rt_np = np.array([[ref_text.encode("utf-8")]], dtype=object)
            inp_t = InferInput("reference_text", rt_np.shape, "BYTES")
            inp_t.set_data_from_numpy(rt_np)
            inputs.append(inp_t)

    return inputs


async def synthesize(client, idx, text, out_path, *, speaker=None, ref_wav=None,
                     ref_text=None, instruction=None):
    """Stream one synthesis. Returns dict with timings & audio stats."""
    inputs = make_inputs(text, speaker=speaker, ref_wav=ref_wav, ref_text=ref_text,
                         instruction=instruction)
    outputs = [InferRequestedOutput("waveform")]
    chunks = []
    ttfa = None
    t0 = time.time()
    chunk_count = 0
    err = None

    async def gen():
        yield {
            "model_name": MODEL,
            "inputs": inputs,
            "outputs": outputs,
            "request_id": f"synth_{idx}_{int(t0*1000)}",
        }

    try:
        async for response in client.stream_infer(inputs_iterator=gen(), stream_timeout=120):
            r, e = response
            if e is not None:
                err = str(e); break
            if ttfa is None:
                ttfa = (time.time() - t0) * 1000.0
            wav = r.as_numpy("waveform")
            if wav is not None and wav.size:
                chunks.append(wav.flatten())
                chunk_count += 1
    except Exception as e:
        err = str(e)

    total_ms = (time.time() - t0) * 1000.0
    if err:
        return {"err": err, "total_ms": total_ms}

    full = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    if len(full) == 0:
        return {"err": "no audio chunks received", "total_ms": total_ms}

    sf.write(out_path, full, 24000)
    return {
        "out": out_path,
        "ttfa_ms": ttfa,
        "total_ms": total_ms,
        "duration_s": len(full) / 24000,
        "chunks": chunk_count,
        "peak": float(np.max(np.abs(full))),
        "clip_pct": float(np.mean(np.abs(full) > 0.99)) * 100,
        "rtf": total_ms / 1000.0 / (len(full) / 24000),
        "err": None,
    }


def slugify(text, maxlen=40):
    """Cheap filename from text."""
    safe = "".join(c if c.isalnum() or c in " -_" else "_" for c in text)
    safe = "_".join(safe.split())
    return safe[:maxlen]


def iter_prompts(args):
    """Yield (idx, text, out_path) for each prompt to synthesize."""
    outdir = Path(args.outdir) if args.outdir else None
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)

    def out_path(idx, text):
        if args.output and not args.batch and not outdir:
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
        # stdin mode
        print(f"Reading prompts from stdin (one per line, Ctrl-D to stop)...", file=sys.stderr)
        for i, line in enumerate(sys.stdin):
            line = line.strip()
            if line:
                yield i, line, out_path(i, line)


async def amain(args):
    client = grpcclient_aio.InferenceServerClient(url=args.url, verbose=False)
    try:
        results = []
        for idx, text, out_path in iter_prompts(args):
            print(f"[{idx}] synth: {text[:60]!r}{'...' if len(text)>60 else ''}")
            res = await synthesize(
                client, idx, text, out_path,
                speaker=args.speaker, ref_wav=args.ref_wav, ref_text=args.ref_text,
                instruction=args.instruction,
            )
            if res.get("err"):
                print(f"      ERROR: {res['err']}", file=sys.stderr)
            else:
                print(f"      → {res['out']}  TTFA={res['ttfa_ms']:.0f}ms "
                      f"total={res['total_ms']:.0f}ms dur={res['duration_s']:.2f}s "
                      f"RTF={res['rtf']:.3f} peak={res['peak']:.3f}")
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
    p = argparse.ArgumentParser(
        description="CosyVoice3 streaming TTS synthesizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("\n\n", 1)[1],
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument("--text", "-t", help="Single text to synthesize")
    src.add_argument("--batch", "-b", help="File with prompts (one per line)")
    p.add_argument("--output", "-o", help="Output wav (for --text only)")
    p.add_argument("--outdir", "-d", default="./out", help="Output directory for batch/stdin (default ./out)")

    p.add_argument("--speaker", "-s", help="Cached speaker name from spk2info.pt")
    p.add_argument("--ref-wav", help="Reference audio for zero-shot voice cloning")
    p.add_argument("--ref-text", help="Reference text matching --ref-wav")
    p.add_argument("--instruction", "-i",
                   help="Per-request speaking-style override (e.g. \"Speak with excitement\")")

    p.add_argument("--url", default="127.0.0.1:18001",
                   help="Triton gRPC endpoint (default 127.0.0.1:18001)")
    args = p.parse_args()

    if not (args.text or args.batch) and sys.stdin.isatty():
        p.print_help()
        print("\nNothing to synthesize. Pass --text, --batch, or pipe prompts via stdin.", file=sys.stderr)
        sys.exit(1)

    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
