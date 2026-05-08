# CosyVoice3 streaming TTS — tuning summary

> Cтатистика замеров, изменения, эффект каждого изменения, и текущий
> production winner. Платформа: 1× NVIDIA L40S (46 GB), Triton 25.06,
> TensorRT-LLM, образ `soar97/triton-cosyvoice:25.06`. Датасет —
> `yuekai/seed_tts_cosy2`, split `test_en`.

## TL;DR

- Throughput при насыщении: **9.7× → 16.4× realtime (+70 %)**
- Stutter (паузы > 1 с в стриме) на N=8: **33.9 % → 0.3-5.5 %** (до 100×)
- TTFA p95 на N=4: **849 → 539-744 ms (−25-36 %)**
- Все 768+ wav-файлов финального sweep'а — clean (peak median ~0.5, 0 % clipping)

## Базовая точка измерений (baseline, count=1/1, autocast plan)

| N | RTF | TTFA p50 (ms) | TTFA p95 (ms) | TTFA p99 (ms) | inter-chunk p95 (ms) | stutter > 1 с |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.142 | 280 | 350 | 420 | 220 | 0.3 % |
| 2 | 0.112 | 460 | 561 | 692 | 358 | 0.3 % |
| 4 | 0.104 | 720 | 849 | 1020 | 751 | 0.3 % |
| 8 | 0.100 | 1180 | 1406 | 1654 | 1210 | **33.9 %** |
| 16 | 0.096 | 2350 | 2708 | 2925 | 2477 | 94.6 % |
| 32 | 0.099 | 2900 | 3367 | 3750 | 3250 | 92.2 % |

> Throughput: 9.7× realtime при насыщении. N=8 уже невозможен для интерактива
> (1 из 3 chunk'ов опаздывает > 1 с).

## Что меняли пошагово и какой эффект

| # | Раунд | Изменение | Эффект | Audio | Status |
|---:|---|---|---|---|---|
| 1 | 1 | `vocoder.count: 1 → 2` | stutter@N=8: **33.9 % → 16.2 %**; TTFA p95@N=8 −14 % | clean | ✅ оставили |
| 2 | 1 | `vocoder.count: 2 → 4` (с t2w=1) | регресс (SM contention pre-fp16); stutter@N=8 16 → 32 % | clean | ❌ откат |
| 3 | 1 | `token2wav.count: 1 → 2` | total p95@N=4: 2901 → 1952 ms (−33 %) | clean | ✅ оставили |
| 4 | 2 | **pure-fp16 TRT plan** для DiT | RTF −12 %, TTFA −20 % | **❌ 100 % clipping** (peak=1.0) | ❌ откат |
| 5 | 2 | + batched vocoder + dynamic batching | mixed | broken (унаследовано) | ❌ откат |
| 6 | 3 | count=4/4 + pure-fp16 | "throughput +24 %" | broken | ❌ откат |
| 7 | 4 | откат на autocast_fp16 + count=4/4 | TTFA p95@N=8: 1406 → **1020 ms**, stutter **0.3 %** | clean | ✅ **производственный baseline** |
| 8 | 5 | `token_hop_len: 15 → 8` | TTFA p95@N=4: 767 → 587 ms (−23 %) | clean | ✅ оставили |
| 9 | 5 | + `flow_pre_lookahead_len: 3 → 1` | TTFA −15-25 ms | clean | ✅ оставили |
| 10 | 5 | + parallel `audio_tokenizer + speaker_embedding` через `asyncio.gather` | TTFA −20 ms на cold | clean | ✅ оставили |
| 11 | 5 | `audio_tokenizer.count + speaker_embedding.count: 1 → 2` | нейтрально (queue per-call < 5 ms) | clean | ✅ оставили (insurance) |
| 12 | 5 | `token_hop_len: 8 → 5` | crash: HiFT CausalConv1d AssertionError | n/a | ❌ откат (минимум) |
| 13 | 5 | bf16 TRT plan для DiT | "RTF parity" | **❌ near-clip distortion** (peak=0.99 везде, RMS×4) | ❌ откат |
| 14 | 5 | count=6 vocoder/token2wav | crash при загрузке (Myelin OOM) | n/a | ❌ откат (GPU memory) |
| 15 | 5 | count=5 + kv_cache=0.4 | runtime OOM | n/a | ❌ откат |
| 16 | 5 | count=5 + **`kv_cache_fraction: 0.4 → 0.3`** | stutter@N=12: **57 % → 37 %** | clean | ✅ оставили (Variant B) |

## Production winner — два варианта

### Variant A — count=4/4 + kv_cache=0.4 (default)

**Когда использовать:** N ≤ 8 (интерактивный TTS, low-to-mid concurrency).
Лучше TTFA, ниже RTF на малых N.

| N | RTF | TTFA p50 | TTFA p95 | TTFA p99 | inter p95 (ms) | stutter > 1 с |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.159 | 321 | 440 | 469 | 266 | 0.6 % |
| 2 | 0.114 | 373 | 432 | 474 | 439 | 0.0 % |
| 4 | 0.089 | **456** | **608** | 655 | 704 | 0.6 % |
| 6 | 0.086 | 628 | 794 | 920 | 833 | 1.3 % |
| 8 | 0.076 | **815** | **1020** | 1104 | 850 | **0.33 %** |
| 16 | 0.074 | 1511 | 1744 | 1822 | 1533 | 92 % |
| 32 | 0.074 | 1879 | 2092 | 2150 | 1912 | 89 % |

### Variant B — count=5/5 + kv_cache=0.3 (high-concurrency)

**Когда использовать:** N ≥ 8 (когда важнее throughput и stutter под
большой нагрузкой). Жертвуем LLM-headroom-ом ради лишнего
vocoder/token2wav instance.

| N | RTF | TTFA p50 | TTFA p95 | TTFA p99 | inter p95 (ms) | stutter > 1 с |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0.118 | 446 | 744 | 2080 | 689 | 0.0 % |
| 6 | 0.085 | 624 | 842 | 893 | 886 | 1.3 % |
| 8 | 0.077 | **794** | **936** | 1018 | 1010 | **5.5 %** |
| 12 | 0.077 | 1092 | **1307** | 1536 | 1275 | **37 %** |

> Для **N ≥ 8** Variant B стабильно лучше Variant A (лучше p95 + лучше stutter).
> Для **N ≤ 4** Variant A немного лучше.

## Сводное сравнение baseline → final

| N | RTF baseline → final | TTFA p95 baseline → final | stutter baseline → final |
|---:|---|---|---|
| 1 | 0.142 → 0.159 | 350 → 440 | 0.3 % / 0.6 % |
| 2 | 0.112 → 0.114 | 561 → **432** (−23 %) | 0.3 % / 0.0 % |
| 4 | 0.104 → **0.089** (−14 %) | 849 → **608** (−28 %) | 0.3 % / 0.6 % |
| 6 | n/a | n/a → 794 | n/a → 1.3 % |
| 8 | 0.100 → **0.076** (−24 %) | 1406 → **1020** (−27 %) | **33.9 % → 0.33 %** (−100×) |
| 12 | n/a | n/a → 1307 (Variant B) | n/a → 37 % |
| 16 | 0.096 → **0.074** (−23 %) | 2708 → 1744 (−36 %) | 94.6 % → 92.2 % |
| 32 | 0.099 → **0.074** (−25 %) | 3367 → 2092 (−38 %) | 92.2 % → 89.3 % |

## Финальная конфигурация (что лежит в production-image)

### Triton model_repo (`model_repo_cosyvoice3_copy/*/config.pbtxt`)

| Модель | count | kind | max_batch_size | Замечания |
|---|---:|---|---:|---|
| `cosyvoice3` (BLS) | 1 | KIND_CPU | 1 | `bls_instance_num=10`, decoupled |
| `audio_tokenizer` | 2 | KIND_GPU | 1 | round 5 step 11 |
| `speaker_embedding` | 2 | KIND_GPU | 1 | round 5 step 11 |
| `token2wav` | **4 или 5** | KIND_GPU | 1 | Variant A=4, Variant B=5 |
| `vocoder` | **4 или 5** | KIND_CPU* | 1 | (model.py использует CUDA) |

### TRT plan для `flow.decoder.estimator`

`flow.decoder.estimator.autocast_fp16.0.plan` (STRONGLY_TYPED autocast).
**НЕ pure-fp16, НЕ bf16** — оба ломают audio quality.

### `trtllm-serve` LLM (Qwen-0.5B fine-tuned)

```bash
trtllm-serve serve \
  --tokenizer hf_cosyvoice3_llm \
  trt_engines_bfloat16 \
  --max_batch_size 64 \
  --kv_cache_free_gpu_memory_fraction 0.4   # Variant A
  # --kv_cache_free_gpu_memory_fraction 0.3 # Variant B
```

### BLS streaming params (`cosyvoice3/1/model.py`)

```python
self.token_frame_rate         = 25
self.flow_pre_lookahead_len   = 1   # было 3
self.token_hop_len            = 8   # было 15
self.token_mel_ratio          = 2

# audio_tokenizer + speaker_embedding запускаются параллельно через asyncio.gather
async def _prepare_prompt(self, request):
    ...
    prompt_speech_tokens, prompt_spk_embedding = await asyncio.gather(
        self.forward_audio_tokenizer(wav, wav_len),
        self.forward_speaker_embedding(wav_tensor),
    )
```

### Container

```bash
# Production reference image (committed snapshot после round 3)
image: cosyvoice3:tuning_step6

docker run -d --gpus all --shm-size=1gb \
  -p 8000-8002:8000-8002 \
  -e PYTHONIOENCODING=utf-8 \
  --name triton_trtllm-tts-1 \
  cosyvoice3:tuning_step6 \
  bash -c "cd /workspace/CosyVoice/runtime/triton_trtllm && bash run_cosyvoice3.sh 3 3"
```

## GPU memory budget на L40S 46 GB

| Компонент | Memory |
|---|---:|
| trtllm-serve LLM (kv_cache=0.4) | ~14.6 GB |
| trtllm-serve LLM (kv_cache=0.3) | ~12.0 GB |
| 4× token2wav (TRT engines + workspace) | ~14.8 GB |
| 5× token2wav | ~18.5 GB |
| 4-5× vocoder (PyTorch HiFT) | ~2.2-2.8 GB |
| 2× audio_tokenizer + 2× speaker_embedding | ~1.5 GB |
| Triton overhead + cudaIpc handles + temp | ~3-4 GB |
| Variant A total (idle) | ~38 GB (82 %) |
| Variant B total (idle) | ~34 GB (73 %) |
| count=6 (любой kv_cache) | не помещается |

## Архитектурные лимиты

| Лимит | Значение | Почему |
|---|---|---|
| TTFA p50 floor | ~300 ms на N=1 | Сумма prompt + LLM prefill + первый DiT + первый vocoder |
| TTFA p95 floor на N=4 | ~540 ms | Concurrency penalty в LLM batch |
| `token_hop_len` минимум | 8 | Меньше → CausalConv1d.forward AssertionError в HiFT |
| `vocoder.count` максимум | 5 | На kv=0.3; count=6 — Myelin OOM |
| TRT plan precision | autocast_fp16 only | pure-fp16 / bf16 ломают amplitude scaling в DiT |
| Stutter @ N≥16 | ≥ 84 % | GPU SMs полностью насыщены |

## Главные ошибки и опасения

### Что чуть не отгрузили в production (поймали аудитом аудио)

1. **Pure-fp16 TRT plan** для DiT (round 2 step 4)
    - Метрики: RTF −12 %, TTFA −20 %, stutter −80 %
    - Реальность: peak=1.0, **100 % сэмплов клиппиновано**
    - Причина: переполнение fp16 в softmax / sigma в DiT
2. **bf16 plan** (round 5)
    - Метрики: парально с autocast
    - Реальность: peak=0.99 везде, RMS в 4 раза выше нормы (constant near-clip distortion)
    - Причина: 7-битная mantissa bf16 недостаточна для precision-sensitive DiT operations

> **Ключевой урок:** RTF/TTFA/stutter меряют **размер** chunk'а, не его
> **содержимое**. Любая precision-замена должна сопровождаться audio audit
> (peak/clip check на ~10 wav-файлах).

### Производственные риски

1. Cold-start tail после рестарта — первые 2-4 запроса в 5-7× медленнее.
   Mitigation: warmup-endpoint при старте сервиса.
2. Speaker cache miss на diverse-prompts → +25 ms на request.
   В production с фиксированными голосами не критично.
3. `kv_cache=0.3` (Variant B) — урезанный LLM headroom. Для длинных
   prompts (>1k токенов) или batch=64 одновременно может не хватить.
   Сейчас тестировали короткие тексты.
4. **NVML breaks on host driver upgrade.** Если хост-драйвер апдейтят при
   запущенном контейнере — новые процессы внутри не могут init CUDA.
   Workaround: `docker stop && start` (через `docker commit` если нет
   bootstrap volumes).

## Что ещё можно попробовать (ранжировано по value/cost)

| Идея | Эффект | Стоимость | Риск |
|---|---|---|---|
| Speaker prewarm cache (precompute для known voices) | TTFA −40-60 ms на cached | 1-2 ч | низкий |
| Warmup endpoint при старте | убрать cold-tail | 0.5 ч | низкий |
| **Per-layer mixed-precision DiT plan** (fp32 для softmax/sigma, fp16 остальное) | RTF −10-12 % без поломки звука | пол-дня + audio audit | средний |
| **HiFT → TRT engine** | −300 MB / vocoder instance, count=6 становится возможным; +возможно −20-30 ms vocoder infer | 4-8 ч | средний |
| LLM fp8 (TRT-LLM поддерживает) | TTFA −80-150 ms | 1-2 дня + calibration | средний |
| LLM prefill caching (если trtllm-serve реализует) | TTFA −30-50 ms | 4 ч + tests | низкий |
| 2-я GPU | удваивает capacity, развязывает kv_cache | hardware budget | низкий |
| Меньшая LLM (Qwen 0.4B / 0.3B / distill) | TTFA −50-100 ms | вернуться к training | высокий (quality) |
| `BLS_INSTANCE_NUM 10 → 16` | помощь на N≥12 | 5 мин | низкий |

> **Highest value/cost: per-layer mixed-precision DiT plan + HiFT→TRT.**
> Вместе могут дать 16× → 20-22× realtime + улучшить TTFA и не требуют
> training.

## Артефакты

| Файл / папка | Что внутри |
|---|---|
| `runtime/triton_trtllm/slack_handoff/SUMMARY.md` | Англ. журнал по раундам |
| `runtime/triton_trtllm/slack_handoff/SYNC_REPORT_RU.md` | Полный отчёт на русском |
| `runtime/triton_trtllm/slack_handoff/ARCHITECTURE_RU.md` | Описание pipeline'а + глоссарий |
| `runtime/triton_trtllm/slack_handoff/TALKING_POINTS.md` | Talking points для устного синка |
| `runtime/triton_trtllm/slack_handoff/AUDIO_QA.md` | Peak/clip stats per stage |
| `runtime/triton_trtllm/slack_handoff/audio/{1..6}_*/` | 18 wav-сэмплов (3 текста × 6 стадий) для прямого сравнения |
| `runtime/triton_trtllm/bench_sweep_results/REPORT_tuning{1..5}.md` | Детальные отчёты per round |
| `runtime/triton_trtllm/bench_step{N}*/summary.md` | Sweep-логи и WAV-файлы каждой итерации |
| `runtime/triton_trtllm/bench_sweep.sh` | Bench harness с patched `client_grpc.py` + `parse_stats.py` |
