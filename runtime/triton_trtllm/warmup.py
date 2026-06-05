#!/usr/bin/env python3
"""CosyVoice3 warm-up burst — prime every BLS instance + CUDA graph batch sizes.

Triton's built-in model_warmup primes the DiT/HiFT TRT kernels per instance at
load, but the LLM warm-up in the BLS is lock-file gated (only ONE of the N BLS
instances actually warms trtllm-serve; the rest stay cold until first traffic).
This script fixes that from the client side after the server is READY:

  1. Wait for Triton HTTP /v2/health/ready.
  2. Fire several waves of CONCURRENT dummy requests so Triton's round-robin
     dispatch touches every BLS instance, and so trtllm-serve captures CUDA
     graphs for the common batch sizes (1, 2, 4, 8).
  3. Report warm TTFA so you can confirm everything is hot.

After this, all instances are warm and stay warm until the process restarts
(TRT kernels, CUDA graphs and cuBLAS/cuDNN workspaces do not decay while the
process lives). KV prefix cache can still LRU-evict under many distinct
speakers, but the default-speaker prefix used here is re-warmed cheaply.

Usage (standalone):
  python warmup.py --url 127.0.0.1:18001                  # local, default speaker
  python warmup.py --url 103.207.149.56:17594 --waves 4 --concurrency 16
  python warmup.py --http 127.0.0.1:18000 --grpc 127.0.0.1:18001  # explicit ports

Usage (entrypoint, fire-and-forget — non-fatal if it fails):
  python warmup.py --url 127.0.0.1:18001 --quiet || true
"""
import argparse
import asyncio
import sys
import time

import numpy as np
import tritonclient.grpc.aio as grpc_aio
from tritonclient.grpc import InferInput, InferRequestedOutput

try:
    import requests
except Exception:
    requests = None

MODEL = "cosyvoice3"
WARMUP_TEXTS = [
    "Hello, this is a warm up request.",
    "Just priming the model for production traffic.",
    "Confirming the pipeline is ready to serve.",
    "Thank you for your patience while we warm up.",
]


def _build_inputs(speaker, text):
    spk = InferInput("speaker_name", (1, 1), "BYTES")
    spk.set_data_from_numpy(np.array([[speaker.encode()]], dtype=object))
    txt = InferInput("target_text", (1, 1), "BYTES")
    txt.set_data_from_numpy(np.array([[text.encode()]], dtype=object))
    return [spk, txt]


async def _one(client, speaker, text, idx):
    t0 = time.time()
    ttfa, got = None, False
    inputs = _build_inputs(speaker, text)

    async def gen():
        yield {"model_name": MODEL, "inputs": inputs,
               "outputs": [InferRequestedOutput("waveform")],
               "request_id": f"warmup_{idx}_{int(t0*1000)}"}
    try:
        async for r, e in client.stream_infer(inputs_iterator=gen(), stream_timeout=120):
            if e is not None:
                return None
            if ttfa is None:
                ttfa = (time.time() - t0) * 1000
            w = r.as_numpy("waveform")
            if w is not None and w.size:
                got = True
    except Exception:
        return None
    return ttfa if got else None


def _wait_ready(http_url, timeout_s, quiet):
    """Block until Triton reports ready (or timeout)."""
    if requests is None:
        return  # best-effort; gRPC calls will retry anyway
    deadline = time.time() + timeout_s
    url = f"http://{http_url}/v2/health/ready"
    while time.time() < deadline:
        try:
            if requests.get(url, timeout=3).status_code == 200:
                if not quiet:
                    print(f"Triton ready at {http_url}", file=sys.stderr)
                return
        except Exception:
            pass
        time.sleep(2)
    if not quiet:
        print(f"warning: Triton not ready after {timeout_s}s — proceeding anyway",
              file=sys.stderr)


async def amain(args):
    if args.http:
        _wait_ready(args.http, args.wait_timeout, args.quiet)

    client = grpc_aio.InferenceServerClient(url=args.grpc, verbose=False)
    try:
        all_ttfa = []
        for wave in range(args.waves):
            # Concurrency ramps 1,2,4,8,... up to --concurrency so trtllm-serve
            # captures CUDA graphs for each batch size, then full waves to cover
            # all BLS instances via round-robin.
            conc = args.concurrency
            tasks = [
                _one(client, args.speaker, WARMUP_TEXTS[i % len(WARMUP_TEXTS)], i)
                for i in range(conc)
            ]
            results = await asyncio.gather(*tasks)
            ok = [t for t in results if t is not None]
            all_ttfa += ok
            if not args.quiet:
                if ok:
                    print(f"wave {wave+1}/{args.waves}: {len(ok)}/{conc} ok  "
                          f"TTFA avg={np.mean(ok):.0f}ms min={min(ok):.0f} max={max(ok):.0f}")
                else:
                    print(f"wave {wave+1}/{args.waves}: 0/{conc} ok (server not answering?)")
        if not args.quiet:
            if all_ttfa:
                warm = all_ttfa[len(all_ttfa)//2:]  # second half = fully warm
                print(f"\n=== warm-up done: {len(all_ttfa)} reqs, "
                      f"final-half TTFA avg={np.mean(warm):.0f}ms p95={np.percentile(warm,95):.0f}ms ===")
            else:
                print("\n=== warm-up: NO successful requests — check server ===")
                return 1
    finally:
        await client.close()
    return 0


def main():
    p = argparse.ArgumentParser(description="CosyVoice3 warm-up burst")
    p.add_argument("--url", help="Triton gRPC HOST:PORT (shorthand; sets --grpc)")
    p.add_argument("--grpc", help="Triton gRPC HOST:PORT (default 127.0.0.1:18001)")
    p.add_argument("--http", help="Triton HTTP HOST:PORT for readiness poll "
                                  "(default derived from --grpc with port-1)")
    p.add_argument("--speaker", default="default", help="Cached speaker to warm (default: default)")
    p.add_argument("--waves", type=int, default=3, help="Number of warm-up waves (default 3)")
    p.add_argument("--concurrency", type=int, default=16,
                   help="Concurrent requests per wave — set >= BLS_INSTANCE_NUM (default 16)")
    p.add_argument("--wait-timeout", type=int, default=180,
                   help="Seconds to wait for Triton readiness (default 180)")
    p.add_argument("--quiet", action="store_true", help="Suppress progress (for entrypoint)")
    args = p.parse_args()

    # Resolve grpc/http defaults.
    args.grpc = args.grpc or args.url or "127.0.0.1:18001"
    if not args.http:
        host, _, port = args.grpc.rpartition(":")
        try:
            args.http = f"{host}:{int(port)-1}"   # gRPC 8001 -> HTTP 8000 (Triton convention)
        except ValueError:
            args.http = None

    rc = asyncio.run(amain(args))
    sys.exit(rc)


if __name__ == "__main__":
    main()
