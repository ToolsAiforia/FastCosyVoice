#!/usr/bin/env python3
"""TTFA-vs-concurrency benchmark for the live S12 server.

For each N in LEVELS: fire N concurrent streaming requests (varied texts, one ref),
measure per-request TTFA (wall time to first audio chunk). Repeats ROUNDS times per
N for stable percentiles. Warms up first.
"""
import asyncio, json, time
import numpy as np, soundfile as sf, librosa
import tritonclient.grpc.aio as grpc_aio
from tritonclient.grpc import InferInput, InferRequestedOutput

URL, MODEL = "127.0.0.1:18001", "cosyvoice3"
REF = "neutral_2"
LEVELS = [4, 6, 8, 12]
ROUNDS = 4
INSTR, EOP = "You are a helpful assistant.", "<|endofprompt|>"


def load_ref():
    w, sr = sf.read(f"reference_samples/{REF}_16k.wav")
    if w.ndim > 1: w = w.mean(1)
    w = w.astype(np.float32)
    if sr != 16000: w = librosa.resample(w, orig_sr=sr, target_sr=16000)
    return w, INSTR + EOP + open(f"reference_samples/{REF}.txt").read().strip()


def build(text, w, rt):
    s = w.reshape(1, -1); L = np.array([[s.shape[1]]], dtype=np.int32)
    a = [InferInput("reference_wav", s.shape, "FP32"), InferInput("reference_wav_len", L.shape, "INT32"),
         InferInput("reference_text", (1, 1), "BYTES"), InferInput("target_text", (1, 1), "BYTES")]
    a[0].set_data_from_numpy(s); a[1].set_data_from_numpy(L)
    a[2].set_data_from_numpy(np.array([[rt.encode()]], dtype=object))
    a[3].set_data_from_numpy(np.array([[text.encode()]], dtype=object))
    return a


async def one(client, text, w, rt, i):
    t0 = time.time(); ttfa = None
    async def gen():
        yield {"model_name": MODEL, "inputs": build(text, w, rt),
               "outputs": [InferRequestedOutput("waveform")], "request_id": f"b{i}_{int(t0*1000)}"}
    async for r, e in client.stream_infer(inputs_iterator=gen(), stream_timeout=180):
        if e is not None: return None
        x = r.as_numpy("waveform")
        if x is not None and x.size:
            ttfa = (time.time() - t0) * 1000; break
    return ttfa


async def main():
    data = json.load(open("english_basket.json"))
    texts = [d["normalized"] for d in data if 6 <= len(d["normalized"].split()) <= 20][:40]
    w, rt = load_ref()
    client = grpc_aio.InferenceServerClient(url=URL, verbose=False)
    # warmup
    await asyncio.gather(*[one(client, texts[i % len(texts)], w, rt, 9000 + i) for i in range(8)])
    print(f"S12 server, ref={REF}, BLS=16 | TTFA (ms) to first audio chunk\n")
    print(f"{'N':>3} {'mean':>7} {'p50':>7} {'p90':>7} {'p95':>7} {'max':>7}  (samples)")
    rows = []
    ti = 0
    for N in LEVELS:
        samples = []
        for _ in range(ROUNDS):
            res = await asyncio.gather(*[one(client, texts[(ti := ti + 1) % len(texts)], w, rt, i) for i in range(N)])
            samples += [x for x in res if x is not None]
        s = np.array(samples)
        print(f"{N:>3} {s.mean():>7.0f} {np.percentile(s,50):>7.0f} {np.percentile(s,90):>7.0f} "
              f"{np.percentile(s,95):>7.0f} {s.max():>7.0f}  ({len(s)})")
        rows.append((N, float(s.mean()), float(np.percentile(s,50)), float(np.percentile(s,95))))
    await client.close()
    json.dump(rows, open("reopt_work/ttfa_s12.json", "w"))


if __name__ == "__main__":
    asyncio.run(main())
