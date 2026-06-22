#!/usr/bin/env python3
"""Per-step synth harness for the re-optimization ladder.

Two modes against the live reopt Triton (gRPC :18001, decoupled streaming):
  --mode dump   : no replay -> generate fresh, save wav + the speech_tokens.
                  Writes gold_tokens.json {f"{ref}|{uid}": [ids]} + manifest + wavs.
                  Use ONCE on the S0 server to fix the gold LLM token sequence.
  --mode replay : feed gold_tokens for each (ref,uid) as replay_tokens, save wav.
                  Audio deltas vs the gold render are attributable to the step only.

Smoke set: SMOKE_UIDS (10 texts) x REFS (3). Output: <outdir>/<ref>/<uid>.wav + manifest.json.
"""
import asyncio, json, os, sys, argparse, time
import numpy as np
import soundfile as sf
import librosa
import tritonclient.grpc.aio as grpc_aio
from tritonclient.grpc import InferInput, InferRequestedOutput

URL, MODEL = "127.0.0.1:18001", "cosyvoice3"
REFDIR, TEXTS_JSON = "reference_samples", "english_basket.json"
INSTR, EOP = "You are a helpful assistant.", "<|endofprompt|>"
SMOKE_UIDS = [6, 13, 46, 3, 8, 32, 5, 77, 63, 95]   # short..long mix
REFS = ["neutral_2", "friendly_2", "angry_2"]
CONC = 4


def load_ref(name):
    w, sr = sf.read(f"{REFDIR}/{name}_16k.wav")
    if w.ndim > 1: w = w.mean(1)
    w = w.astype(np.float32)
    if sr != 16000: w = librosa.resample(w, orig_sr=sr, target_sr=16000)
    return w, open(f"{REFDIR}/{name}.txt").read().strip()


def build(target_text, ref_wav, ref_text, replay=None):
    s = ref_wav.reshape(1, -1).astype(np.float32)
    L = np.array([[s.shape[1]]], dtype=np.int32)
    ins = [InferInput("reference_wav", s.shape, "FP32"),
           InferInput("reference_wav_len", L.shape, "INT32"),
           InferInput("reference_text", (1, 1), "BYTES"),
           InferInput("target_text", (1, 1), "BYTES")]
    ins[0].set_data_from_numpy(s); ins[1].set_data_from_numpy(L)
    ins[2].set_data_from_numpy(np.array([[ref_text.encode()]], dtype=object))
    ins[3].set_data_from_numpy(np.array([[target_text.encode()]], dtype=object))
    if replay is not None:
        a = np.asarray(replay, dtype=np.int32).reshape(1, -1)
        ri = InferInput("replay_tokens", list(a.shape), "INT32")
        ri.set_data_from_numpy(a); ins.append(ri)
    return ins


async def synth(client, key, target_text, ref_wav, ref_text, out_path, replay, want_tokens):
    outs = ["waveform"] + (["speech_tokens"] if want_tokens else [])
    chunks, tokens, err = [], None, None
    t0 = time.time(); ttfa = None
    async def gen():
        yield {"model_name": MODEL, "inputs": build(target_text, ref_wav, ref_text, replay),
               "outputs": [InferRequestedOutput(o) for o in outs],
               "request_id": f"{key}_{int(t0*1000)}"}
    try:
        async for r, e in client.stream_infer(inputs_iterator=gen(), stream_timeout=180):
            if e is not None: err = str(e); break
            if want_tokens:
                t = r.as_numpy("speech_tokens")
                if t is not None and t.size: tokens = t.flatten().tolist()
            x = r.as_numpy("waveform")
            if x is not None and x.size:
                if ttfa is None: ttfa = (time.time() - t0) * 1000
                chunks.append(x.flatten())
    except Exception as e:
        err = str(e)
    if err or not chunks:
        return {"err": err or "no audio"}
    full = np.concatenate(chunks)
    sf.write(out_path, full, 24000)
    return {"dur": len(full)/24000, "peak": float(np.max(np.abs(full))), "ttfa": ttfa,
            "tokens": tokens}


async def amain(args):
    data = json.load(open(TEXTS_JSON))
    texts = {i: d["normalized"] for i, d in enumerate(data)}
    raw = {i: d["text"] for i, d in enumerate(data)}
    gold = json.load(open(args.gold_tokens)) if args.mode == "replay" else {}
    os.makedirs(args.outdir, exist_ok=True)
    client = grpc_aio.InferenceServerClient(url=URL, verbose=False)
    manifest, dumped, ttfas = [], {}, []
    sem = asyncio.Semaphore(CONC)

    async def one(ref, uid, rw, rt_send):
        async with sem:
            key = f"{ref}|{uid}"
            replay = gold.get(key) if args.mode == "replay" else None
            if args.mode == "replay" and replay is None:
                return ref, uid, {"err": "no gold tokens"}
            od = f"{args.outdir}/{ref}"; os.makedirs(od, exist_ok=True)
            r = await synth(client, f"{ref}_{uid}", texts[uid], rw, rt_send,
                            f"{od}/{uid:04d}.wav", replay, want_tokens=(args.mode == "dump"))
            return ref, uid, r

    try:
        tasks = []
        for ref in REFS:
            rw, rt = load_ref(ref)
            send = f"{INSTR}{EOP}{rt}"
            # warm this ref serially first (computes prompt features once)
            ref0 = await one(ref, SMOKE_UIDS[0], rw, send)
            for r in [ref0]:
                _ref, _uid, _res = r
                if not _res.get("err"):
                    if _res.get("ttfa"): ttfas.append(_res["ttfa"])
                    if args.mode == "dump" and _res.get("tokens"):
                        dumped[f"{_ref}|{_uid}"] = _res["tokens"]
                    manifest.append({"uid": _uid, "text": texts[_uid], "raw_text": raw[_uid],
                                     "audio_path": f"./{_ref}/{_uid:04d}.wav", "system": _ref})
                else:
                    print(f"  ERR {_ref}|{_uid}: {_res['err'][:80]}", flush=True)
            for uid in SMOKE_UIDS[1:]:
                tasks.append(one(ref, uid, rw, send))
        for fut in asyncio.as_completed(tasks):
            ref, uid, res = await fut
            if res.get("err"):
                print(f"  ERR {ref}|{uid}: {res['err'][:80]}", flush=True); continue
            if res.get("ttfa"): ttfas.append(res["ttfa"])
            if args.mode == "dump" and res.get("tokens"):
                dumped[f"{ref}|{uid}"] = res["tokens"]
            manifest.append({"uid": uid, "text": texts[uid], "raw_text": raw[uid],
                             "audio_path": f"./{ref}/{uid:04d}.wav", "system": ref})
    finally:
        await client.close()

    json.dump(manifest, open(f"{args.outdir}/manifest.json", "w"), indent=2, ensure_ascii=False)
    if args.mode == "dump":
        json.dump(dumped, open(args.gold_tokens, "w"))
        print(f"DUMP: {len(dumped)} token seqs -> {args.gold_tokens}")
    ttfas = np.array(ttfas) if ttfas else np.array([0.0])
    print(f"{args.outdir}: {len(manifest)} wavs | TTFA p50={np.percentile(ttfas,50):.0f}ms "
          f"p95={np.percentile(ttfas,95):.0f}ms")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dump", "replay"], required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--gold-tokens", default="reopt_work/gold_tokens.json")
    asyncio.run(amain(ap.parse_args()))
