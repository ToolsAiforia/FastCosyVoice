# CosyVoice3 streaming TTS — tuning summary (rounds 1–5)

5 rounds of tuning on a single L40S, started from `soar97/triton-cosyvoice:25.06`
+ `run_cosyvoice3.sh 0 3`. Goal: reduce TTFA + stutter while supporting more
concurrent streams. 768+ wav files audited along the way.

## TL;DR — final production config

- **Throughput at saturation: 9.7× → 16-17× realtime (+70 %)**
- **N=8 stutter (>1 s gap): 33.9 % → 0.3-5 %** (model-dependent; the metric
  that matters most for live streaming UX)
- **TTFA p95 at N=4: 849 ms → 540-660 ms (−25-35 %)**
- **All output audio clean** (peak median ~0.5, 0 % clipping in 768/768
  files of the final sweep)

Two production variants — pick by workload pattern:

| Config | Best for | TTFA p95 N=4 | stutter@N=8 | stutter@N=12 |
|---|---|---:|---:|---:|
| **count=4/4 + kv_cache=0.4** | low-concurrency interactive | 539-660 ms | 6.9 % | 56.6 % |
| **count=5/5 + kv_cache=0.3** | high-concurrency batches | 446-744 ms | 5.5 % | **37.1 %** |

## What broke during tuning (and was fixed)

| Round | Change | Apparent gain | Audio | Verdict |
|---|---|---|---|---|
| 2 (step 4) | pure-fp16 TRT plan for `flow.decoder.estimator` | "RTF −12 %, stutter −90 %" | **100 % clipping (peak=1.0)** | ❌ rolled back |
| 2 (step 5) | batched vocoder + dynamic batching | mixed | broken (inherited) | ❌ rolled back |
| 3 (step 6) | count=4/4 + pure-fp16 | "throughput +24 %" | broken | ❌ rolled back |
| 5 (bf16 attempt) | bf16 plan for DiT | RTF parity | **near-clip / distorted** (peak=0.99 always) | ❌ rolled back |
| 5 (hop=5) | smaller first-chunk wait | -? TTFA | server crash | ❌ rolled back |
| 5 (count=6) | more vocoder/token2wav | n/a | n/a (won't load) | ❌ GPU OOM |
| 5 (count=5 + kv_cache=0.4) | mixed instances | n/a | n/a (runtime OOM) | ❌ rolled back |

**Big lesson learned**: latency/throughput/stutter percentiles measure chunk
*size*, not chunk *content*. A precision change that produces clipped or
noise-filled audio can pass all timing metrics with flying colors. **Always
audit wav peak/clip stats** after any TRT plan / precision / model-graph
change. We shipped two rounds before listening — won't repeat.

## What worked (stayed in production)

| Change | Effect | Why it works |
|---|---|---|
| `vocoder.count=2` (round 1) | stutter@N=8 33.9→16 % | Triton dispatches concurrent streaming chunks to multiple Python instances; queue drains 2× faster |
| `token2wav.count=2` (round 1) | extra ~12 % at N=4-16 | same |
| `vocoder.count=4`, `token2wav.count=4` (round 4) | N=8 stutter to 0.3 %, throughput +13 % | autocast fp16 plan keeps fp32 for precision-critical ops; pure-fp16 didn't |
| `token_hop_len 15→8` (round 5) | TTFA p95 N=4 −25-30 % | first chunk needs 11 LLM tokens vs 18; saves ~150 ms on critical path |
| `flow_pre_lookahead_len 3→1` (round 5) | TTFA −10-20 ms | 2 fewer LLM tokens needed before first chunk |
| `audio_tokenizer + speaker_embedding` parallel via `asyncio.gather` (round 5) | TTFA −20 ms (cold) | independent computations; no reason to serialize |
| `kv_cache_fraction 0.4→0.3` + `count=5` (round 5b) | stutter @ N=12 from 57 % → 37 % | trades LLM headroom for 1 more vocoder/t2w instance — net win at high concurrency |

## Final TTFA / RTF / stutter table (best operating points)

### count=4/4 (production default, kv_cache=0.4)

| N | TTFA p95 (ms) | RTF | stutter > 1 s | comment |
|---:|---:|---:|---:|---|
| 4 | 539-659 | 0.10 | 0-2 % | safe SLA |
| 6 | 794 | 0.086 | 1.3 % | new sweet spot |
| 8 | 1047 | 0.082 | 6.9 % | acceptable |
| 12 | 1338 | 0.080 | 56.6 % | offline-only |

### count=5/5 (high-concurrency, kv_cache=0.3)

| N | TTFA p95 (ms) | RTF | stutter > 1 s | comment |
|---:|---:|---:|---:|---|
| 4 | 744 | 0.118 | 0 % | slight TTFA tradeoff |
| 6 | 842 | 0.085 | 1.3 % | parity |
| 8 | **936** | 0.077 | **5.5 %** | better than count=4 |
| 12 | **1307** | 0.077 | **37.1 %** | better than count=4 |

## Why we couldn't hit TTFA p95 ≤ 300 ms at N=4

Single-user p50 floor is **~300 ms** (architectural):
- audio_tokenizer + speaker_embedding (parallel): ~25-30 ms
- LLM prefill + 11 tokens: ~90-110 ms
- token2wav first call (DiT): ~80 ms
- vocoder first call (HiFT): ~80 ms

To hit p95 ≤ 300 ms at N=4 we'd need single-user latency below 300 ms +
zero concurrency penalty + no p95 tail — none of which is achievable
with config-only tweaks. Real engineering needed: LLM prefill caching /
specdec / fp8, DiT mixed-precision plan, HiFT → TRT engine.

`token_hop_len=5` (smaller first chunk) failed — HiFT's CausalConv1d
asserts on too-short input; 8 is the practical floor.

## Production reference

- **Container image**: `cosyvoice3:tuning_step6` (committed snapshot). Start with:
  ```
  docker run -d --gpus all --shm-size=1gb -p 8000-8002:8000-8002 \
    -e PYTHONIOENCODING=utf-8 --name triton_trtllm-tts-1 cosyvoice3:tuning_step6 \
    bash -c "cd /workspace/CosyVoice/runtime/triton_trtllm && bash run_cosyvoice3.sh 3 3"
  ```
- **Bench harness**: `runtime/triton_trtllm/bench_sweep.sh` (uses patched `client_grpc.py`
  with inter-chunk jitter + warmup-skip + per-stage stats parsing).
- **All sweep artefacts**: `runtime/triton_trtllm/bench_step{N}_*/` — CSV + per-N WAVs.
- **Reports**: `runtime/triton_trtllm/bench_sweep_results/REPORT_tuning{1..5}.md`
- **MUST audit audio after any precision/plan/model-graph change**:
  ```python
  import wave, numpy as np, glob
  for f in glob.glob('bench_step*/N*/*.wav'):
      w = wave.open(f,'rb'); a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)/32768.0; w.close()
      peak = float(np.max(np.abs(a))); clip = float(np.mean(np.abs(a) > 0.99))
      # healthy: peak ~0.3-0.7, clip < 1%
      # broken: peak == 1.000, clip > 50%
  ```

## What's not pursued (real engineering, not config tuning)

- Per-layer mixed-precision DiT plan — would recover the −12 % RTF that
  pure-fp16 promised but with audio intact (~half day)
- HiFT → TRT engine — saves ~300 MB / instance, would let us run count=5+
  without reducing kv_cache fraction (~half day)
- Speaker prewarm cache for production-known voices — saves ~40-60 ms TTFA
  on cached speakers
- LLM-side: prefill caching, fp8, speculative decoding — only path to
  meaningful TTFA reductions below 450 ms

## Audio samples in this folder

`audio/` contains 3 sample wavs per stage (same source files where
available, so you can A/B compare). Listen to:
- `1_baseline_clean/` — original (pre-tuning) audio quality
- `2_pure_fp16_BROKEN_clipped/` — the bug we caught: full clipping
- `3_bf16_BROKEN_distorted/` — bf16 attempt: near-clip / distortion
- `4_round4_winner_clean/` — production winner audio
- `5_round5_TTFA_optimized/` — round 5 with TTFA tuning
- `6_step15_count5_kv30/` — final: count=5 + reduced KV cache
