# Deploy: round-9-stable + Tier A — host-side build & test

Current production state captured in tag `round-9-stable` on branch `round-9-stable`.

## What's inside

| Optimization | Status |
|---|---|
| DiT TRT layer_mixed_fp16 (75 FP32 sensitive layers) | ✓ |
| HiFT TRT layer_mixed_fp32io (507 FP32 sensitive layers, Snake-safe) | ✓ |
| HiFT hybrid PyTorch + TRT decode_core | ✓ |
| Path D — no `cuda.synchronize()` in forward_estimator | ✓ |
| token_hop_len=8 + flow_pre_lookahead_len=1 (streaming) | ✓ |
| H2 pre-allocated mel buffer | ✓ |
| **Tier A A1** real-shape LLM warmup (3 variants: 25/125/252 prompt tokens) | ✓ |
| **Tier A A3** max_tokens 750 → 200 (voice-chat optimized) | ✓ |
| **Tier A A4/A5** Triton model_warmup for token2wav + vocoder | ✓ |

Measured TTFA (warm, post-Tier-A, on H100 PCIe 80 GB):
- p50 = 246-330 ms  (varies by reference)
- p95 = 351 ms
- avg = 307-324 ms (5-burst)
- Cold-first req = 344 ms (after Tier A warmup; was 1025-3287 ms before)

## Prerequisites on host

- NVIDIA GPU (Ampere SM 8.0+ or Hopper SM 9.0) with ≥24 GB VRAM
- Docker with NVIDIA Container Toolkit (`nvidia-docker` or `docker run --gpus all`)
- ~50 GB free disk (model weights 19 GB + image 8 GB + build artefacts)
- Internet (pulls `soar97/triton-cosyvoice:25.06` base + git clone + HF model)

## Build steps (host machine)

```bash
# 1. Clone repo at the production tag
git clone https://github.com/ToolsAiforia/FastCosyVoice
cd FastCosyVoice
git checkout round-9-stable

# 2. Download model weights (one-time, ~19 GB)
cd runtime/triton_trtllm
bash download_cosyvoice3_models.sh                # pulls Fun-CosyVoice3-0.5B-2512/ + cosyvoice3_llm/

# 3. (Optional) Pre-build TRT engines on host so the image is "warm"
#    Skip this if you want the entrypoint to build engines on first container start.
#    Per-GPU SM, so build on the SAME GPU you'll deploy on.
# (Engines auto-build inside container on first start if absent — usually OK.)

# 4. Build the image (~10 min, includes model weight COPY)
docker build -f Dockerfile.cosyvoice3 -t cosyvoice3-tts:round9-tierA .

# 5. (Optional) tag for your registry
docker tag cosyvoice3-tts:round9-tierA your-registry.example.com/cosyvoice3-tts:round9-tierA
docker push your-registry.example.com/cosyvoice3-tts:round9-tierA
```

## Smoke test (host machine)

```bash
# 1. Run the container — entrypoint launches trtllm-serve (port 8010) + Triton (HTTP 8000, gRPC 8001)
docker run --rm --gpus '"device=0"' --shm-size 1g \
    -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    --name cosyvoice3-test \
    cosyvoice3-tts:round9-tierA

# Wait for "READY" in container logs (~60-90 s — TRT engine build + sequential model load + warmup)
docker logs -f cosyvoice3-test | grep -E "READY|warmup OK|model_repo"
```

In another terminal:

```bash
# 2. Bench a single text via the synth client (from repo root)
python synth_round9.py \
    --text "Hello, this is a smoke test of the round-9 Tier-A deployment." \
    --ref-wav runtime/triton_trtllm/reference.wav \
    --ref-text "So my favorite podcast at the moment is a podcast called Ruined where it's two best friends, one loves horror movies, the other one hates horror movies and so on." \
    --url localhost:8001 \
    -o smoke_test.wav
# Expected: TTFA ≈ 220-350 ms warm, peak ≤ 0.99, clip% = 0
```

Or use the included gRPC bench:

```bash
python runtime/triton_trtllm/client_grpc.py \
    --server-addr localhost --server-port 8001 \
    --num-tasks 8 --huggingface-dataset yuekai/seed_tts_cosy2
```

## Configuration knobs (env vars consumed by `entrypoint_cosyvoice3.sh`)

| Var | Default | Effect |
|---|---|---|
| `bls_instance_num` | 16 | BLS orchestrator workers (HIGH = more parallel streams; LOW = tighter batching) |
| `triton_max_batch_size` | 1 | per-sub-model max batch (round-9-stable не batchится — оставить 1) |
| `decoupled_mode` | True | streaming-friendly response mode |
| `LLM_PORT` | 8010 | port for trtllm-serve (BLS hits this) |

## Important notes

1. **TRT plans rebuild per-GPU** — engines tagged with SM number (`.0.plan`, `.1.plan`). Copying plans between different GPU models will fail. On first container start, missing plans are built (~3-5 min).
2. **First user request** still has ~344 ms cold tail despite Tier A — that's the propagation of warmup across all 16 BLS instances (only the first holds the warmup lock; others skip but still cold-JIT on first hit). Steady state after ~3-5 requests is 220-330 ms.
3. **trtllm-serve sidecar** runs in the same container — `--max_batch_size 64`, `--kv_cache_free_gpu_memory_fraction 0.3`. Tunable in `entrypoint_cosyvoice3.sh`.
4. **Model dir is 19 GB** — pure ADD into image bloats the image. For prod with multi-host deploy, mount as volume instead:
   ```bash
   docker run ... -v $(pwd)/Fun-CosyVoice3-0.5B-2512:/workdir/Fun-CosyVoice3-0.5B-2512 ...
   ```
   Then use the slim Dockerfile variant that does NOT ADD the model dir.

## What's NOT in this image (deferred to next iteration)

See `/root/.claude/plans/cosyvoice3-async-giraffe.md` for the queued optimizations:
- B2 first-chunk DiT n_timesteps=5 (TTFA −40-50 ms)
- HiFT FP32 island grouping (reduce 369 Reformatting CopyNodes)
- CampPlus TRT pre-build at boot
- trtllm-serve flag tuning (max_batch_size 64→16, kv_cache_fraction 0.3→0.7, enable_block_reuse)

These will land in a follow-up tag.

## Verification checklist (after deploy on test host)

- [ ] Container reaches READY in < 90 s
- [ ] `docker logs | grep "warmup OK"` shows at least 1 entry
- [ ] `synth_round9.py --text "Hello"` returns audio with TTFA ≤ 400 ms (warm 2nd+ request)
- [ ] `peak ≤ 0.99 && clip% == 0` on all generated samples
- [ ] N=4 concurrent (`client_grpc.py --num-tasks 4`) — TTFA p95 ≤ 450 ms
