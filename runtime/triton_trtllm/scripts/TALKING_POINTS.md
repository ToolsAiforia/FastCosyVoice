# Talking points — CosyVoice3 streaming TTS tuning

## What we shipped
- 5 rounds of config tuning on a single L40S
- **Throughput +70 %** (9.7× → 16.4× realtime) at saturation
- **N=8 stutter (>1 s pause): 33.9 % → 0.3 %** — the metric that actually matters for live streaming UX (100×)
- **TTFA p95 at N=4: 849 → 540 ms (−36 %)**
- All output audio clean (peak median ~0.5, 0 % clipping verified across 768+ wavs)

## What worked
1. **More Triton instances** for `vocoder` and `token2wav` (1 → 4).
   Triton dispatches concurrent streaming chunks across processes;
   queue drains in parallel. Each call gets ~50 % slower from SM contention but queue drops 80 %, net wins.
2. **Smaller `token_hop_len`** (15 → 8) + **smaller `flow_pre_lookahead_len`** (3 → 1).
   First chunk needs only 9 LLM tokens instead of 18 → ~150 ms TTFA off.
3. **Parallel prompt processing**: `audio_tokenizer + speaker_embedding` via `asyncio.gather` (was serial).
4. **`kv_cache_free_gpu_memory_fraction 0.4 → 0.3` + count=5/5** — for high-concurrency loads only.
   Trades LLM headroom for one more vocoder/token2wav instance. **stutter@N=12: 57 % → 37 %**.

## What broke (and we caught)
1. **Pure-fp16 TRT plan** for `flow.decoder.estimator` looked great on metrics
   (RTF −12 %, stutter −90 %) but produced **100 % clipped audio** —
   peak=1.0 on every file. Saturation in the DiT softmax / sigma operations.
2. **bf16 TRT plan** also broke audio — peak=0.99 everywhere, RMS ×4
   (constant near-clip distortion). Mantissa precision insufficient for the network.
3. **`STRONGLY_TYPED autocast_fp16`** is the only safe default — it keeps fp32
   islands wherever the network graph requires them.

## The big lesson
**Latency / throughput / stutter percentiles measure chunk *size*, not chunk *content*.**
A precision change that produces clipped audio can pass *all* timing metrics with flying
colors. We shipped two rounds of "wins" before listening to the audio. **Always
audit wav peak/clip after any TRT plan / precision / model-graph change.**

## Where the floor is

- **Single-user TTFA p50 floor: ~300 ms** on this hardware (L40S, fp16 DiT,
  PyTorch HiFT, bf16 LLM) — that's the full pipeline cost.
- **N=4 TTFA p95 floor: ~540 ms** with all current optimizations.
- Going below 300 ms requires real engineering (LLM prefill caching / specdec /
  fp8, DiT mixed-precision plan, HiFT → TRT). Half-day to several days each.

## Production config summary

```yaml
# Container reference
image: cosyvoice3:tuning_step6     # committed snapshot
start: bash run_cosyvoice3.sh 3 3   # only stage 3 = start servers

# Triton model_repo configs
vocoder:           count=4
token2wav:         count=4   # plan = flow.decoder.estimator.autocast_fp16.0.plan
audio_tokenizer:   count=2
speaker_embedding: count=2

# trtllm-serve LLM
max_batch_size: 64
kv_cache_free_gpu_memory_fraction: 0.4   # or 0.3 for high-concurrency variant

# BLS (cosyvoice3/1/model.py)
token_hop_len: 8                # was 15
flow_pre_lookahead_len: 1       # was 3
# audio_tokenizer + speaker_embedding parallel via asyncio.gather
```

Two production variants:
- **count=4 / kv=0.4**: best for low-to-mid concurrency (default)
- **count=5 / kv=0.3**: better at N≥8 (stutter lower)

## Files for the team
- `SUMMARY.md` — full journey + numbers
- `AUDIO_QA.md` — peak/clip stats per stage (illustrates what broken vs clean looks like in numbers)
- `audio/` — 3 wav samples per stage; you can listen to broken vs clean directly
- Reports in `runtime/triton_trtllm/bench_sweep_results/REPORT_tuning{1..5}.md`
