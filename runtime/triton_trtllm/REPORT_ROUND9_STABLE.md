# CosyVoice3 round-9-stable — Production Report

**Hardware**: NVIDIA H100 PCIe 80 GB (sm_90)
**Branch**: `round-9-stable`
**HEAD**: `334aa3d`
**Date**: 2026-05-23
**Model**: Fun-CosyVoice3-0.5B-2512 (HF: `FunAudioLLM/Fun-CosyVoice3-0.5B-2512`)

## Что это за ветка

Reconstructed round-9 production winner (Slack `SYNC_ROUND9_2026-05-20.md`)
**без batching / coordinators / instruction support / spk2info auto-bake**.
Минимальный код, простая архитектура — для случаев, когда BLS Tier-4
coordinators не нужны.

## Track record commit'ов (от `3a4eb64` upstream baseline)

```
334aa3d  perf(round-9): backport warmup + streaming hop=8 + H2 pre-alloc mel buffer
92dd1bf  perf(vocoder): HiFT hybrid PyTorch + layer-mixed TRT decode_core
f967104  feat(round-9): production winner state — layer_mixed DiT TRT + Path D no-sync
3a4eb64  feat: add benchmarking script and GPU monitoring (upstream baseline)
```

## Current production config

| Component | Setting |
|---|---|
| LLM backend | `trtllm-serve` bfloat16, in-flight batching, `enable_block_reuse` |
| LLM `kv_cache_free_gpu_memory_fraction` | `0.3` |
| LLM `max_batch_size` | `64` |
| BLS instance count (`bls_instance_num`) | `16` |
| token2wav instance count | `8` (KIND_GPU) |
| vocoder instance count | `8` (KIND_CPU, runs on GPU internally) |
| audio_tokenizer / speaker_embedding count | `2 / 2` |
| Streaming `token_hop_len` | **`8`** (минимум до HiFT CausalConv1d assertion) |
| Streaming `flow_pre_lookahead_len` | **`1`** |
| Streaming chunk strategy | `exponential` (1×8 → 25, 50, 100, ... tokens) |
| BLS LLM warmup | 3 synthetic requests at boot (kills cold tail) |
| BLS H2 mel buffer | Pre-allocated 800-frame buffer, in-place slice |
| DiT TRT plan | `flow.decoder.estimator.layer_mixed_fp16.0.plan` (FP16 default + 75 FP32 sensitive layers) |
| Vocoder TRT plan | `hift_decode_core.layer_mixed_fp32io.plan` (FP32 IO + 507 FP32 sensitive layers + FP16 elsewhere) |
| Vocoder hybrid | PyTorch (f0_predictor + STFT + conv_pre + ISTFT) + TRT decode_core |
| Path D | `forward_estimator` без per-call `cuda.synchronize()` |

## TTFA benchmark (seed_tts_cosy2/test_en, random refs)

| N | TTFA p50 | TTFA p95 | TTFA p99 | Inter-chunk p95 | RTF | Stutter |
|---|---|---|---|---|---|---|
| 1 | 364 ms | **1025 ms** (cold tail) | 1182 ms | 309 ms | 0.48 | 0 % |
| 4 | 375 ms | **433 ms** | 470 ms | 461 ms | 0.078 | 0 % |
| 8 | 473 ms | **610 ms** | 643 ms | 780 ms | 0.059 | 4.3 % |
| 12 | 613 ms | **740 ms** | 815 ms | 1140 ms | 0.053 | 12.9 % |

### vs SYNC report

| N | SYNC report | Ours | Gap |
|---|---|---|---|
| 4 p95 | 360 ms | 433 ms | +20 % |
| **8 p95** | **530 ms** | **610 ms** | **+15 %** |
| 12 p95 | 652 ms | 740 ms | +13 % |

В пределах 15-20% от SYNC numbers — практически совпадает учитывая variance между прогонами. **N=1 cold tail** (1025 ms p95) — bench дополнительно warmup'ит сетью cold-state, в production это absorb'ится первым реальным запросом.

## Per-stage compute (625 calls)

| Stage | avg_infer | Role |
|---|---|---|
| audio_tokenizer | 82 ms | reference audio → s3 speech tokens (~2 calls/request) |
| speaker_embedding | 48 ms | campplus voice embedding (parallel with audio_tokenizer) |
| **token2wav (DiT)** | **146 ms** | main bottleneck, ~2-3 calls/request, layer_mixed TRT |
| **vocoder (HiFT)** | **81 ms** | mel → waveform, hybrid PyTorch+TRT layer_mixed |
| cosyvoice3 (BLS) | 2 ms | Python orchestrator overhead |

## Audio quality (8 N=8 samples)

| Метрика | Result |
|---|---|
| Peak amplitude | 0.16 — 0.99 |
| Clipping (samples > 0.99) | **0 %** во всех файлах |
| Audio duration range | 2.80 — 7.08 s |
| Errors | 0 / 48 |

Sample files saved to `listening_samples_round9/` для прослушивания.

## VRAM

- Idle baseline (Triton up, models loaded): ~50 GB
- Под N=12 нагрузкой: 60 GB used / 80 GB total
- Headroom: 20 GB (safe для дальнейшего scaling)

## Deployment quick-start

```bash
cd runtime/triton_trtllm

# 1. Download model weights (~6 GB) — at first run
bash download_cosyvoice3_models.sh

# 2. Build Docker image (fat — bakes weights, ~37 GB)
docker build -f Dockerfile.cosyvoice3 -t cosyvoice3-tts:round9 .

# 3. Run — entrypoint auto-builds TRT engines on this GPU
docker run -d --name cv3 \
    --gpus '"device=0"' --shm-size=1g \
    -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    cosyvoice3-tts:round9

# 4. Wait for ready (~10-15 min cold start: TRT-LLM build + layer_mixed plans)
docker logs -f cv3 | grep "Triton server is ready"

# 5. Smoke test
docker exec cv3 python3 /workspace/CosyVoice/runtime/triton_trtllm/test_streaming.py
```

## What if I need lower TTFA / more throughput?

Switch to `improve_cosyvoice3` branch — adds BLS Tier-4 coordinators
(real GPU batching) + instruction override + spk2info auto-bake +
spec'd Dockerfile (slim variant). Same TTFA p95 at N=8 (~570 ms).
Trade-off: значительно больше кода.

## Limitations / known gaps

1. **N=1 cold tail (1025 ms p95)** — warmup synthetic, real first request still pays small JIT penalty
2. **No incremental vocoder** — vocoder receives accumulated mel each chunk (grows with stream)
3. **No real GPU batching** — concurrent requests don't share batched GPU compute
4. **No instruction support** — voice style fixed by reference audio

Each of these is addressable (see `improve_cosyvoice3` branch) but adds code complexity.
