# CosyVoice3 — Production Report (Final)

**Hardware**: NVIDIA H100 PCIe 80 GB (sm_90)
**Date**: 2026-05-22
**Model**: Fun-CosyVoice3-0.5B-2512 (FunAudioLLM/Fun-CosyVoice3-0.5B-2512)

## Production config

| Component | Setting |
|---|---|
| LLM backend | `trtllm-serve` bfloat16, in-flight batching, `enable_block_reuse` |
| LLM `kv_cache_free_gpu_memory_fraction` | `0.4` |
| LLM `max_batch_size` | `64` |
| BLS `instance_group.count` | `2` |
| BLS `bls_max_concurrent` | `8` per instance → 16 cluster cap |
| BLS `t2w_dispatch_wait_ms` | `0` (opportunistic coordinator) |
| BLS `voc_dispatch_wait_ms` | `0` |
| token2wav `instance_group.count` | `4` (KIND_GPU) |
| token2wav `max_batch_size` | `8`, preferred [2,4,8] |
| token2wav TRT plan | `flow.decoder.estimator.layer_mixed_B8_fp16.{device_id}.plan` (B=2..16, opt=B=8, per-layer mixed precision: fp16 baseline + 75 fp32 layers for Norm/Softmax/time_embed/proj_out) |
| vocoder `instance_group.count` | `4` (KIND_CPU, GPU кому в коде) |
| vocoder `max_batch_size` | `8`, preferred [2,4,8] |
| vocoder TRT plans | `hift_decode_core.fp32.plan` (B=1 fastpath) + `hift_decode_core.fp32_B8.plan` (B=1..8, opt=B=4) — dual-plan switching по `mel.shape[0]` |
| audio_tokenizer count | `2` |
| speaker_embedding count | `2` (TRT fp32 для CampPlus) |
| Streaming `token_hop_len` | `8` (минимум до HiFT CausalConv1d assertion) |
| Streaming `flow_pre_lookahead_len` | `1` |

## Default speaker

`spk2info.pt` содержит one speaker: **`ref`** (English).
- Reference audio: `/workdir/reference.wav` (10s, 16 kHz, podcast extract)
- Reference text (with baked instruction):
  > `Speak in a calm, friendly podcast host tone.<|endofprompt|>So my favorite podcast at the moment is a podcast called Ruined, where it's two best friends. One loves horror movies, the other one hates horror movies, and so on.`
- prompt_speech_tokens: 252 tokens
- prompt_speech_feat: [504, 80]
- prompt_spk_embedding: [192]

Per-request `instruction` input может переопределить baked instruction (см. `synth.py --instruction "..."`).

## TTFA Benchmark

### Workload A — Cached speaker `'ref'` (voice-chat production scenario)

| N | TTFA p50 | TTFA p95 | TTFA p99 | RTF | Errors |
|---|---|---|---|---|---|
| 1 | **209 ms** (cold) | 209 ms | 209 ms | 0.20 | 0 |
| 4 | **435 ms** | **480 ms** | 480 ms | 0.12 | 0 |
| 8 | **861 ms** | **1032 ms** | 1032 ms | 0.10 | 0 |
| 12 | 1493 ms | 1782 ms | 1782 ms | 0.10 | 0 |

### Workload B — seed_tts random refs (varied per-request prompts)

| N | TTFA p50 | TTFA p95 | RTF | Stutter |
|---|---|---|---|---|
| 1 | 290 ms | 304 ms | 0.27 | 0 % |
| 4 | **324 ms** | **411 ms** | 0.082 | 0 % |
| 8 | **449 ms** | **569 ms** | 0.065 | 2.2 % |

## BLS Coordinator engagement (across full sweep)

| Stage | Total batched events | B=2 | B=3 | B=4 |
|---|---|---|---|---|
| token2wav | 5 | 4 | 1 | 0 |
| vocoder | 9 | 8 | 1 | 0 (1×B=4 via Triton dynamic batch) |

Opportunistic batching engages когда concurrent streams в одном BLS event-loop'е push'ат запросы с совпадающими shape_key. Default `WAIT_MS=0` без TTFA-стоимости.

## Per-stage compute (cumulative)

| Stage | Calls | B=1 avg | B=2 avg | B=3 avg | Weighted avg |
|---|---|---|---|---|---|
| token2wav | 462 | 128.1 ms | 292.8 ms | 449.4 ms | **130.9 ms** |
| vocoder | 459 | 67.0 ms | 104.0 ms | — | **67.8 ms** (B=4: 163 ms) |
| audio_tokenizer | 48 | 111.0 ms | — | — | 111.0 ms |
| speaker_embedding | 48 | 48.7 ms | — | — | 48.7 ms |

`audio_tokenizer` + `speaker_embedding` вызываются **0 раз** для cached speakers (Workload A) — ~160 ms экономии на каждом первом chunk'е stream'а.

## Audio quality

- Все sample'ы N=1/4/8/12: peak ≤ 0.99, clipping = 0 %
- Voice cloning preserved (cached speaker идентичен через все N)
- Audio sample peaks: 0.42–0.99 (зависит от текста и instruction'а)

## VRAM

- Idle baseline (Triton up, models loaded): ~76 GiB
- Под N=12 нагрузкой: 78.4 GiB used / 80 GiB total (headroom 1.6 GiB)
- Стабильно — pre-allocated buffers не растут под нагрузкой

## Saturation point

- **N≤8**: stutter 0 % (cached), p95 ≤ 1032 ms
- **N=12**: 0 errors но p95 1.8 s — на грани, рекомендую второй H100 для N>8 sustained traffic

## Deployment quick-start

```bash
cd runtime/triton_trtllm

# 1. Download model weights (~6 GB) — на первом запуске
bash download_cosyvoice3_models.sh

# 2. Build Docker image (вкл. reference.wav для default speaker)
docker build -f Dockerfile.cosyvoice3 -t cosyvoice3-tts:h100 .

# 3. Run — entrypoint авто-собирает TRT engines + spk2info при cold start
docker run -d --name cv3 \
    --gpus '"device=0"' --shm-size=1g \
    -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    cosyvoice3-tts:h100

# 4. Wait for ready (TRT-LLM build ~5-10 min + HiFT B-dyn ~3-4 min на cold start)
docker logs -f cv3 | grep -E "Triton server is ready|LLM server is ready"

# 5. Smoke test
docker exec cv3 python3 /workspace/CosyVoice/runtime/triton_trtllm/test_streaming.py ref

# Or use synth.py from host:
python synth.py --text "Hello world." --speaker ref -o hello.wav
python synth.py --text "Speak excited!" --speaker ref -i "Speak with excitement" -o excited.wav
```

## Optional tuning

Increase batching engagement (production with cached speakers):
```yaml
# In cosyvoice3/config.pbtxt parameters:
key: "t2w_dispatch_wait_ms"  value: "10"   # +10ms TTFA → engages 30-50% batching
key: "voc_dispatch_wait_ms"  value: "10"
```

Bake additional speakers via `generate_spk2info.py`:
```bash
docker exec cv3 python3 /workdir/generate_spk2info.py \
    --model-dir /workdir/Fun-CosyVoice3-0.5B-2512 \
    --audio /path/to/voice.wav \
    --reference-text "<|endofprompt|>Transcription with period." \
    --speaker-name myspeaker \
    --output /workdir/Fun-CosyVoice3-0.5B-2512/spk2info.pt
```
Then restart Triton (or send `POST /v2/repository/models/cosyvoice3/load`).
