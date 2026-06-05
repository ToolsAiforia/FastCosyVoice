#!/usr/bin/env python3
"""CosyVoice3 synthesis CLI — full feature set.

Supports:
  - zero-shot voice cloning: --ref-wav + --ref-text
  - cached speaker:          --speaker <name>  (from spk2info.pt, fastest)
  - instruction / style:     --instruction "Speak slowly and clearly"
  - text input modes:        --text / --batch <file> / stdin
  - remote server:           --url HOST:GRPC_PORT
  - per-synthesis metrics:   TTFA / total / RTF / peak / clip%

The BLS resolves the voice in this priority order:
  1. --speaker  → cached prompt in spk2info.pt (no audio_tokenizer cost)
  2. --ref-wav (+ --ref-text) → zero-shot (computes prompt per request)
The instruction is prepended to reference_text as
  "<instruction><|endofprompt|><ref_text>" — overriding the server default.

Examples:
  # Cached default speaker, single text
  python synth.py --speaker default --text "Hello, how can I help you?" -o out.wav

  # Zero-shot from a reference clip with an instruction
  python synth.py --ref-wav voice.wav --ref-text "transcript of voice.wav" \
      --instruction "Speak calmly and slowly" \
      --text "Your payment is due tomorrow." -o out.wav

  # Batch against a remote RunPod server
  python synth.py --speaker default --batch prompts.txt --outdir ./out \
      --url 103.207.149.56:13814

  # Stdin
  echo "Quick test." | python synth.py --speaker default --outdir ./out
"""
import argparse
import asyncio
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import tritonclient.grpc.aio as grpc_aio
from tritonclient.grpc import InferInput, InferRequestedOutput

MODEL = "cosyvoice3"
ENDOFPROMPT = "<|endofprompt|>"


def load_reference(path, target_sr=16000):
    """Load reference audio → mono float32 @ target_sr."""
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


def compose_reference_text(ref_text, instruction):
    """Build the reference_text the BLS expects.

    If an instruction is given, prepend "<instruction><|endofprompt|>".
    The BLS only injects its default instruction when no <|endofprompt|>
    marker is present, so supplying our own here overrides it.
    """
    ref_text = ref_text or ""
    if instruction:
        # Strip any marker the user accidentally included, then add one.
        instr = instruction.split(ENDOFPROMPT)[0].rstrip()
        body = ref_text.split(ENDOFPROMPT)[-1]
        return f"{instr}{ENDOFPROMPT}{body}"
    return ref_text


def build_inputs(target_text, *, speaker=None, ref_wav=None, ref_text=None):
    """Assemble Triton inputs. Either speaker OR ref_wav must be set."""
    tgt_np = np.array([[target_text.encode("utf-8")]], dtype=object)
    inputs = [InferInput("target_text", tgt_np.shape, "BYTES")]
    inputs[0].set_data_from_numpy(tgt_np)

    if speaker is not None:
        spk_np = np.array([[speaker.encode("utf-8")]], dtype=object)
        spk_in = InferInput("speaker_name", spk_np.shape, "BYTES")
        spk_in.set_data_from_numpy(spk_np)
        inputs.append(spk_in)

    # reference_text is sent whenever we have one (carries the instruction
    # even on the cached-speaker path, where the BLS matches on this string).
    if ref_text is not None and ref_text != "":
        ref_np = np.array([[ref_text.encode("utf-8")]], dtype=object)
        rt_in = InferInput("reference_text", ref_np.shape, "BYTES")
        rt_in.set_data_from_numpy(ref_np)
        inputs.append(rt_in)

    if ref_wav is not None:
        samples = ref_wav.reshape(1, -1).astype(np.float32)
        lengths = np.array([[samples.shape[1]]], dtype=np.int32)
        w_in = InferInput("reference_wav", samples.shape, "FP32")
        w_in.set_data_from_numpy(samples)
        l_in = InferInput("reference_wav_len", lengths.shape, "INT32")
        l_in.set_data_from_numpy(lengths)
        inputs += [w_in, l_in]

    return inputs


async def synthesize(client, idx, target_text, out_path, *,
                     speaker=None, ref_wav=None, ref_text=None):
    inputs = build_inputs(target_text, speaker=speaker,
                          ref_wav=ref_wav, ref_text=ref_text)
    outputs = [InferRequestedOutput("waveform")]
    chunks, ttfa, err = [], None, None
    t0 = time.time()

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
        "out": out_path, "ttfa_ms": ttfa, "total_ms": total,
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
        print("Reading prompts from stdin (Ctrl-D to stop)...", file=sys.stderr)
        for i, line in enumerate(sys.stdin):
            line = line.strip()
            if line:
                yield i, line, out_path(i, line)


async def amain(args):
    ref_wav = None
    ref_text = None

    if args.ref_wav:
        print(f"Loading reference from {args.ref_wav}", file=sys.stderr)
        ref_wav = load_reference(args.ref_wav)
        print(f"  reference: {len(ref_wav)/16000:.2f}s @ 16kHz", file=sys.stderr)

    # reference_text: combine the supplied ref-text with the instruction.
    ref_text = compose_reference_text(args.ref_text, args.instruction)
    if args.speaker:
        print(f"Speaker: cached '{args.speaker}'", file=sys.stderr)
    if args.instruction:
        print(f"Instruction: {args.instruction!r}", file=sys.stderr)
    if ref_text:
        print(f"  reference_text: {ref_text[:80]!r}{'...' if len(ref_text)>80 else ''}",
              file=sys.stderr)
    print(f"Server: {args.url}", file=sys.stderr)

    client = grpc_aio.InferenceServerClient(url=args.url, verbose=False)
    try:
        results = []
        for idx, text, out_path in iter_prompts(args):
            print(f"[{idx}] synth: {text[:60]!r}{'...' if len(text)>60 else ''}")
            res = await synthesize(
                client, idx, text, out_path,
                speaker=args.speaker, ref_wav=ref_wav,
                ref_text=(ref_text if ref_text else None))
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
    p = argparse.ArgumentParser(
        description="CosyVoice3 synthesis CLI (speaker / zero-shot / instruction / remote)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Examples:")[-1] if "Examples:" in __doc__ else None)

    src = p.add_mutually_exclusive_group()
    src.add_argument("--text", "-t", help="Single text to synthesize")
    src.add_argument("--batch", "-b", help="File with prompts (one per line; # = comment)")

    p.add_argument("--speaker", "-s", help="Cached speaker name from spk2info.pt (e.g. 'default')")
    p.add_argument("--ref-wav", help="Reference audio for zero-shot (wav/ogg/mp3)")
    p.add_argument("--ref-text", help="Transcription of the reference audio")
    p.add_argument("--instruction", "-i",
                   help="Style/prosody instruction, e.g. 'Speak slowly and clearly'")

    p.add_argument("--output", "-o", help="Output wav path (single --text mode)")
    p.add_argument("--outdir", "-d", default="./out", help="Output dir for batch/stdin")
    p.add_argument("--url", default="127.0.0.1:18001",
                   help="Triton gRPC endpoint HOST:PORT (default local 127.0.0.1:18001)")
    args = p.parse_args()

    # Validation: need a voice source.
    if not args.speaker and not args.ref_wav:
        p.print_help()
        sys.exit("\nERROR: provide a voice — either --speaker NAME or --ref-wav FILE.")
    if args.ref_wav and not args.ref_text and not args.instruction:
        print("warning: --ref-wav without --ref-text — zero-shot quality is best "
              "with an accurate transcription.", file=sys.stderr)
    if args.speaker and args.instruction:
        print("warning: --instruction is IGNORED with --speaker. A cached speaker "
              "carries its own baked LLM prompt; the BLS resolves speaker_name "
              "first and never reads the instruction. Use --ref-wav (zero-shot) "
              "for a per-request instruction to take effect.", file=sys.stderr)
    if not (args.text or args.batch) and sys.stdin.isatty():
        p.print_help()
        sys.exit("\nNothing to synthesize. Pass --text or --batch, or pipe via stdin.")

    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
