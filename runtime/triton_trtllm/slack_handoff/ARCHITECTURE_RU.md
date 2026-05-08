# CosyVoice3 streaming TTS — архитектура, что улучшили, что осталось

Дополнение к `SYNC_REPORT_RU.md`. Подробное описание каждого компонента
pipeline'а: что он делает, на чём построен, что мы тюнили в раундах 1-5,
какие там ограничения и опасения.

В конце — глоссарий терминов, которые встречались в отчётах
(BLS, DiT, HiFT, KV cache, autocast plan, flow matching, и т.п.).

## Общая схема

```
Клиентский gRPC-запрос
        │
        ▼
┌──────────────────────────────────────┐
│  Triton Inference Server  (port 18001)│
│                                       │
│  ┌──────────────────────────────────┐ │
│  │ cosyvoice3 (BLS)  ≪orchestrator≫│ │  ← Python, async-pipeline
│  │  - принимает text + reference   │ │
│  │  - раскидывает по моделям       │ │
│  │  - decoupled streaming output   │ │
│  └──────────────────────────────────┘ │
│      │                                │
│      ├──→ ┌──────────────────────┐    │
│      │   │ audio_tokenizer      │    │  ← S3Tokenizer (PyTorch GPU)
│      │   │ ref-wav → speech-toks│    │     извлекает фонетические токены
│      │   └──────────────────────┘    │     из reference аудио
│      │                                │
│      ├──→ ┌──────────────────────┐    │
│      │   │ speaker_embedding    │    │  ← CAMPPlus (PyTorch GPU)
│      │   │ ref-wav → spk-vector │    │     embedding голоса для conditioning
│      │   └──────────────────────┘    │
│      │                                │
│      ├──→ ┌──────────────────────┐    │
│      │   │ token2wav (flow)     │    │  ← Flow matching DiT в TRT
│      │   │ tokens → mel         │    │     генерит mel-спектрограмму
│      │   └──────────────────────┘    │
│      │                                │
│      └──→ ┌──────────────────────┐    │
│          │ vocoder (HiFT)       │    │  ← CausalHiFTGenerator (PyTorch)
│          │ mel → waveform       │    │     mel → 24 kHz audio
│          └──────────────────────┘    │
└──────────────────────────────────────┘
        │ HTTP
        ▼
┌──────────────────────────────────────┐
│  trtllm-serve  (port 8000)           │  ← TensorRT-LLM сервер
│  Qwen-0.5B fine-tuned                │     генерит target speech tokens
│  text+ref-toks → target-toks         │     по reference & target text
│  (autoregressive, streaming)         │
└──────────────────────────────────────┘
```

Streaming-протокол (внутри BLS, для одного запроса):
1. BLS вызывает `audio_tokenizer` и `speaker_embedding` параллельно
   на reference аудио (cached если speaker уже видели).
2. BLS формирует prompt для LLM: `"<reference text>" + ref-tokens`.
3. BLS открывает streaming-соединение с trtllm-serve, получает токены
   по одному.
4. Когда накопилось `token_hop_len + flow_pre_lookahead_len = 9` LLM-токенов
   → BLS вызывает `token2wav` с накопленными токенами + speaker_embedding +
   `streaming=True, finalize=False, token_offset=N`.
5. token2wav возвращает кусок mel-спектрограммы.
6. BLS аккумулирует mel и вызывает `vocoder` → получает audio chunk.
7. Audio chunk отправляется клиенту через decoupled stream.
8. Goto 4 со следующими LLM-токенами (chunk size растёт по экспоненте:
   25, 50, 100, ... до finalize).
9. Когда LLM закончил — BLS делает финальный chunk с `finalize=True`.

## Компонент 1 — `audio_tokenizer`

**Что делает.** Принимает reference wav (16 kHz, обычно 3-10 сек —
короткий пример голоса спикера), выдаёт последовательность дискретных
**speech tokens** — фонетических единиц, которые понимает LLM.

**Реализация.** `s3tokenizer` библиотека (S3-токенайзер из CosyVoice2).
PyTorch модель на CUDA. Один forward на запрос, токены ~25/сек.
Чистый GPU compute, ~25 ms на запрос.

**Что трогали в раундах:**
- *Round 1*: ничего, работало с count=1.
- *Round 5 step 11*: `count=2` для страховки. Эффект на TTFA ≈ ноль
  (per-call queue был 3-6 ms всегда), но не вредит. Оставил.
- *Round 5 step 10*: переписал `forward_audio_tokenizer` в
  `async def` + `inference_request.async_exec()`, чтобы запускать
  параллельно со `speaker_embedding` (см. ниже).

**Опасения / лимиты:**
- Если когда-то перейдём на длинные reference samples (>30 сек), время
  растёт линейно. Сейчас 25 ms — приемлемо.
- `s3tokenizer` v0.3.0 — публичный, но если изменят формат токенов в
  будущей версии CosyVoice3 — нужна будет пересборка.

## Компонент 2 — `speaker_embedding`

**Что делает.** Принимает тот же reference wav, выдаёт **embedding
голоса** — fixed-size вектор, который характеризует голос (тембр,
интонационный профиль). Используется как conditioning для DiT.

**Реализация.** CAMPPlus (Context-Aware Multi-granularity speaker
embedding network). PyTorch на CUDA. ~20 ms на запрос.

**Что трогали:**
- *Round 5 step 11*: `count=2` (страховка, как у audio_tokenizer).
- *Round 5 step 10*: `async def` + параллельный запуск с audio_tokenizer:
  ```python
  prompt_speech_tokens, prompt_spk_embedding = await asyncio.gather(
      self.forward_audio_tokenizer(wav, wav_len),
      self.forward_speaker_embedding(wav_tensor),
  )
  ```
  Раньше шли серийно: ~25 ms (tokenizer) + ~20 ms (embedding) = 45 ms.
  Теперь параллельно: max(25, 20) = 25 ms. Win ~20 ms на cold request.

**Опасения / лимиты:**
- Speaker cache хеш — по reference text. Для diverse-prompts (как наш
  тестовый датасет) cache miss каждый запрос. В production с фиксированным
  набором голосов будет ~99 % cache hits.
- Если перейдём на cross-lingual / multi-speaker production, нужно
  будет thinking о cache invalidation.

## Компонент 3 — `token2wav` (Flow Matching DiT)

**Что делает.** Самый тяжёлый компонент. Принимает на вход:
- target speech tokens от LLM (накопленный набор)
- prompt speech tokens от audio_tokenizer
- prompt mel features (вычисленные из reference wav через STFT)
- speaker embedding
- `token_offset` — с какой позиции slice'ить выход
- `streaming` / `finalize` — флаги стриминг-режима

Выдаёт **mel-спектрограмму** для нового куска.

**Реализация.**
- `cosyvoice/flow/flow.py` — обёртка с PyTorch encoder + матрица prompt features
- `cosyvoice/flow/decoder.py` (`CausalMaskedDiffWithDiT`) — диффузионный
  трансформер с causal masking для streaming.
- `flow.decoder.estimator` = DiT-network (это TensorRT-engine), вызывается
  в цикле с разными timesteps (Euler integration over noise schedule).

**Метод**: **flow matching** — диффузионная модель, обученная на ODE между
шумом и mel'ом. На inference решаем ODE Эйлером за ~10 шагов. Каждый шаг
DiT с conditioning на токены + speaker + prompt mel.

**Что трогали (это тут больше всего боли):**
- *Round 1*: `count=2` → потом `count=4` (round 4) → попытка `count=5`
  (round 5b). count=4 — production, count=5 при kv_cache=0.3.
- *Round 2 step 4*: пересобрали TRT plan из `STRONGLY_TYPED autocast_fp16`
  в **pure fp16** (`BuilderFlag.FP16` + `tensor.dtype=HALF` на всех I/O).
  Метрики выглядели как победа (RTF −12 %), но **выходной mel
  переполнился в softmax/sigma операциях** → vocoder выдал полный clipping
  (peak=1.0 на 100 % сэмплов). **Откатили в round 4.**
- *Round 5 эксперимент*: попробовали **bf16 plan** (BF16 имеет fp32 exponent
  range, не должен переполняться). Не клипает на 100 %, но peak=0.99
  везде, RMS в 4 раза выше нормы — **mantissa precision (7 бит у bf16
  против 10 у fp16) не хватает в DiT**. Откатили.
- *Production*: только `STRONGLY_TYPED autocast_fp16` — TensorRT сам
  решает, какие layer'ы оставить в fp32 (softmax, sigma scales).
  **Не трогать без careful per-layer precision sweep + audio audit.**

**Опасения / лимиты:**
- Каждый instance держит ~3.7 GB GPU memory (TRT engine 666 MB + workspace).
  count=5 — потолок при kv_cache=0.3, count=6 не помещается на L40S.
- `flow.encoder` (PyTorch) с `self.flow.half()` — работает в fp16. Если
  когда-то понадобится больше precision (для редких языков?), может
  потребоваться вернуть fp32 здесь.
- Streaming mode: каждый вызов token2wav пересчитывает mel **с нуля** для
  всего префикса (не stateful). Это означает: при увеличении длительности
  стрима затраты на каждый chunk растут. Для коротких текстов (~5-10 сек)
  это OK, для очень длинных (>1 мин) — может стать узким местом.

## Компонент 4 — `vocoder` (CausalHiFTGenerator)

**Что делает.** Принимает mel-спектрограмму, выдаёт waveform 24 kHz.
**Самое затратное по wallclock на N=4-12** (queue копится здесь).

**Реализация.** Causal HiFi-GAN с iSTFT-головой. PyTorch fp32 (не половинной
точности!).
- `cosyvoice/hifigan/generator.py:CausalHiFTGenerator`
- Внутри: `f0_predictor` (CausalConvRNN с causal-1d свёртками,
  принципиально работает в **float64** для precision — комментарий в коде:
  "f0_predictor precision is crucial for causal inference"), `m_source`
  (источник синусоид), upsampling блоки с causal-conv, snake-net,
  `iSTFT` голова.
- Не TRT (пока) — PyTorch pure.

**Что трогали:**
- *Round 1*: `vocoder.count: 1 → 2 → 4`. Step 1 (1→2): stutter@N=8 33→16 %.
  Step 2 (2→4): SM contention pre-fp16, регресс. Откатили.
- *Round 4*: `count=4` (после pure-fp16 для DiT освободились SM —
  теперь 4 копии vocoder помещаются в pipeline без contention).
- *Round 2 step 5*: попробовали сделать **batched vocoder** (`max_batch_size=4 +
  dynamic_batching`). Переписал `model.py` чтобы принимать `[B, 80, T]`
  вход, верифицировал что HiFT stateless (caches передаются как аргументы,
  не self.). Win at N=4, регресс при N=16 (batched HiFT forward
  sublinear, SM contention при количестве 2 instance'ов × batch 4).
  Откатили.
- *Round 5 step 12*: пробовали `token_hop_len: 8 → 5`, vocoder упал с
  `AssertionError` в `CausalConv1d.forward(x, cache)`: `assert x.shape[2] ==
  input_timestep`. Минимальный размер первого chunk'а в HiFT — захардкожен.
  Откатили.

**Опасения / лимиты:**
- **Vocoder — главный bottleneck** в производстве. queue@N=12 ≈ 60-90 s
  суммарно. Дальнейшие win'ы упираются именно сюда.
- Каждый instance ~500-650 MB GPU. На count=4: 2.2 GB.
- В fp32 → не самый быстрый. **Перевод в TRT-engine** (HiFT → ONNX → TRT)
  — заявленный `TODO`-проект (4-8 часов). Освободит память + может дать
  −30-40 ms на vocoder infer.
- `f0_predictor.to(torch.float64)` — бутылочное горлышко, fp64 на L40S
  плохо ускоряется (Tensor Cores не работают на fp64). Не трогать без
  переобучения f0_predictor с care.

## Компонент 5 — LLM (`trtllm-serve` Qwen-0.5B fine-tuned)

**Что делает.** Главный generative-компонент. Принимает текст + reference
tokens, **autoregressive** генерит target speech tokens (по одному, ~25
токенов в секунду аудио). Streaming mode — выдаёт токены по мере
генерации.

**Реализация.**
- TensorRT-LLM, отдельный процесс `trtllm-serve serve` на отдельном порту
  (8000 default).
- TRT engine собирается из HF checkpoint через `trtllm-build`.
- bf16 weights, `--gemm_plugin bfloat16`.
- KV cache pre-allocated: `--kv_cache_free_gpu_memory_fraction 0.4` (или
  0.3 в варианте B).
- Доступ через OpenAI-compatible HTTP API (chat/completions).
- BLS дёргает через `httpx.AsyncClient` с stream=True.

**Что трогали:**
- *Round 5b*: `kv_cache_free_gpu_memory_fraction: 0.4 → 0.3`. Это
  единственный config-knob у LLM. Уменьшил pre-allocated KV cache на
  ~3.5 GB → освободил место для +1 token2wav instance. Net win при
  N≥8.
- В остальном LLM не трогали.

**Опасения / лимиты:**
- LLM **не виден в Triton stats** — он отдельный процесс. Per-call latency
  оценивается косвенно (TTFA минус остальные стадии).
- `--max_batch_size 64` — щедро для 0.5B на L40S, в реальности при N=4-12
  занят ~10-20 % batch capacity. Можно было бы сжать до 32 или 16, но
  выгода маленькая.
- **kv_cache=0.3** — при длинных prompts (>1k tokens) или batch=64
  одновременно может не хватить. Сейчас тестировали короткие тексты —
  риск отложенный.
- LLM TTFB (prefill) — самая большая невидимая часть TTFA. Оценочно
  ~30-50 ms prefill + 7-10 ms на каждый из ~9 первых токенов = ~100-130 ms.
  Чтобы ускорить — нужен **fp8** (TRT-LLM supports), **specdec** (но 0.5B
  обычно не оправдывает), **prefill caching** (если trtllm-serve
  реализует — стоит проверить).

## Компонент 6 — BLS orchestrator (`cosyvoice3` Python model)

**Что делает.** Координирует все остальные модели, реализует streaming-протокол,
хранит speaker cache, управляет dynamic chunk strategy.

**Реализация.** Triton Python backend, decoupled mode (модель может
эмитить response chunks по мере готовности, а не один блок). `async`-код
(asyncio internally в Triton's Python backend).

**Что трогали:**
- *Round 5 step 9*: `self.token_hop_len: 15 → 8`. Это сколько LLM-токенов
  нужно накопить перед первым chunk'ом. Меньше → меньше TTFA. **−150 ms
  TTFA p95 на N=4.**
- *Round 5 step 10*: `self.flow_pre_lookahead_len: 3 → 1`. 2 токена
  поменьше нужны для DiT lookahead. **−15-25 ms TTFA.**
- *Round 5 step 10*: `_prepare_prompt` стал `async def`, audio_tokenizer
  + speaker_embedding теперь через `asyncio.gather`. **−20 ms на cold
  request.**
- *Round 5 step 12*: пробовали `token_hop_len: 8 → 5` — упирались в
  HiFT minimum chunk → откатили.

**Опасения / лимиты:**
- `bls_instance_num=10` — сейчас 10 параллельных Python "слотов" для BLS.
  При N>10 они queue'ятся. На N=12 cosyvoice3 BLS infer = 0.4 ms,
  queue ~0 → 10 хватает. Если когда-то будем масштабировать N>20, может
  стоит увеличить.
- Speaker cache — простой Python dict, не expire'ится. На длительной
  работе с разнообразными speakers может сожрать RAM. Сейчас не критично,
  но стоит мониторить.
- Streaming-протокол: **каждый новый chunk требует полного pre-fix re-run**
  в token2wav. Состояние (cache) не передаётся между вызовами. На длинных
  текстах per-chunk латентность растёт. Это архитектурное (CosyVoice3-design),
  не наш баг.

## Компонент 7 — Triton Inference Server

**Что делает.** Хост для всех моделей. Управляет gRPC, dispatching между
instance'ами, queue, statistics.

**Что трогали:**
- `instance_group.count` для всех моделей (см. выше).
- НЕ трогали: `max_batch_size`, `dynamic_batching` (после провала round
  2 step 5), backend version.

**Опасения / лимиты:**
- Triton `25.06` — версия старая, но стабильная. Обновление до 25.10+
  может дать улучшения (новый Python backend, dynamic_batching changes),
  но это полная пересборка контейнера + risk regression.
- Decoupled mode добавляет overhead per-chunk (response_sender). На
  очень коротких чанках (token_hop_len=5 если бы работал) это могло бы
  стать заметным.

## Компонент 8 — Контейнер / GPU runtime

**Что делает.** Изолирует pipeline. NVIDIA Container Runtime инжектит
драйвер-libs.

**Что трогали:**
- `docker commit` контейнера в image `cosyvoice3:tuning_step6` после
  round 3 — чтобы все наши patches (configs, plans, model.py) сохранились
  в production-ready snapshot.
- Контейнер запускается с `bash run_cosyvoice3.sh 3 3` (только stage 3,
  без re-bootstrap stages 0-2).

**Опасения / лимиты:**
- **Host driver upgrade в running container ломает NVML.** Уже наступали:
  trtllm-serve работал на старом CUDA-контексте, новые процессы не могли
  init'ировать. Lesson — после host-driver upgrade всегда `docker stop &&
  start` (но это требует `commit` если не хочется перезагружать
  bootstrap'ом). См. `feedback_nvml_breaks_on_host_driver_upgrade.md` в
  memory.
- L40S 46 GB — потолок. На больше instance'ов (count=6) или больше
  моделей в pipeline нужна 2-я GPU.

---

## Глоссарий

### TTS-специфичные

**TTFA** (Time To First Audio). Время от gRPC-запроса до прихода первого
audio chunk'а клиенту. Ключевая интерактивная метрика. Состоит из:
prompt processing + LLM prefill + N первых токенов + первый token2wav +
первый vocoder.

**RTF** (Real-Time Factor). `processing_seconds / audio_duration_seconds`.
RTF=0.1 значит: 1 секунда аудио синтезируется за 100 ms (10× faster than
realtime). Меньше = лучше (для одного потока).

**Stutter (>1s)**. Доля inter-chunk-интервалов в стриминге, где gap между
соседними audio chunks > 1 sec. Если stutter > 0, юзер слышит паузу при
воспроизведении (буфер пуст).

**inter-chunk jitter**. Распределение интервалов между chunk'ами клиенту.
Метрика "плавности" стрима.

**token_hop_len**. Сколько новых LLM-токенов накопить перед очередным
вызовом token2wav (значит — каким будет следующий audio chunk). Меньше =
меньше TTFA, но больше overhead per chunk.

**flow_pre_lookahead_len**. Сколько лишних "будущих" токенов передать в
DiT для smoother boundary. Прямо влияет на TTFA первого chunk'а
(ждём `hop + lookahead` токенов перед началом).

**token_mel_ratio**. Сколько mel-frames соответствует одному speech-токену.
Здесь = 2.

**token_frame_rate**. Сколько speech-токенов в секунде аудио. Здесь = 25.

### Архитектурные

**BLS** (Business Logic Scripting). Triton-механизм где Python-модель
может вызывать другие модели как sub-инференсы. Используется как
orchestrator для multi-stage pipeline'ов вроде нашего CosyVoice3.

**Decoupled mode**. Triton-режим где одна inference-request может породить
**несколько** response chunks (а не строго один). Нужен для streaming
TTS / LLM, где аудио выдаётся по частям.

**DiT** (Diffusion Transformer). Трансформер, обученный как генератор для
диффузионного процесса (или, в нашем случае, flow matching). Принимает
шум + conditioning и итеративно восстанавливает целевое распределение.

**Flow Matching**. Метод обучения генеративных моделей: учим прямой ODE
из простого распределения (Гаусс) в данные. На inference решаем ODE Эйлером
за 5-15 шагов. Альтернатива классической диффузии (DDPM), часто быстрее
и стабильнее.

**HiFiGAN / HiFT**. Vocoder на основе GAN, преобразует mel-спектрограмму
в waveform. CausalHiFTGenerator — caually-masked версия для streaming.

**f0 predictor**. Подмодель vocoder'а, предсказывает fundamental frequency
(основную тональную частоту) из mel-спектрограммы. Используется как
условие для синтеза harmonic source signal.

**STFT/iSTFT**. Short-Time Fourier Transform / inverse. Магниту-фазовое
представление сигнала. Vocoder использует iSTFT как голову (mel→
magnitude+phase → time-domain).

**causal convolution**. Свёртка где output[t] зависит только от input[≤t]
(не "видит будущее"). Нужно для streaming-режима, где будущие фреймы
ещё не прибыли.

### TensorRT / precision

**TensorRT plan / engine**. Скомпилированная и оптимизированная версия
ONNX-графа для TRT runtime'а. Привязана к GPU compute capability и
версии TRT.

**STRONGLY_TYPED autocast_fp16 plan**. Тип TRT engine'а где TensorRT сам
решает какие layer'ы fp16, какие fp32 (на основе precision-сохранности).
Безопасный default для DiT.

**Pure fp16 plan** (`BuilderFlag.FP16` + tensor.dtype=HALF). Все layer'ы
жёстко в fp16. Быстрее но **может переполнять** в softmax / sigma /
exp операциях. Нужен per-layer precision sweep чтобы заработало без
поломки.

**bf16**. Brain-float16: 1 sign + 8 exponent + 7 mantissa бит. fp32-range
exponent (не переполняется как fp16), но **меньше precision** (mantissa 7
бит против 10 у fp16). Поддерживается TensorRT 10+ на Ada/Hopper. У нас
сломал DiT mel-amplitude.

**KV cache**. В LLM-инференсе: накопленные keys/values предыдущих токенов,
чтобы не пересчитывать attention для всего prefix'а. Pre-allocated buffer
размером `max_batch * max_seq_len * num_layers * 2 * dim`. Доминирует
GPU memory у LLM.

**`kv_cache_free_gpu_memory_fraction`**. Параметр trtllm-serve: какую
долю **свободной** GPU memory зарезервировать под KV cache при старте.
0.4 = 40 % free GPU. Чем больше — тем больше параллельных streams могут
держаться, но меньше памяти доступно для других моделей.

**Triton `instance_group.count`**. Количество параллельных копий модели
в Triton. Каждая копия — отдельный Python-процесс / CUDA context.
Concurrent-запросы дижатятся round-robin между копиями.

### GPU

**SM** (Streaming Multiprocessor). Compute-юнит GPU. На L40S — 142 SM.
"SM contention" = когда несколько параллельных вычислений соревнуются
за SMs и каждое идёт медленнее.

**NVML** (NVIDIA Management Library). API для query'а GPU state
(memory, utilization, etc.). Внутри контейнера ломается если host driver
upgrade'нули в процессе работы — нужен restart container.

**cudaIpcHandle**. Inter-process CUDA-handle, нужен для shared GPU memory
между разными процессами (Triton stub'ы общаются с tritonserver через
IPC). При OOM открытие handle падает первым.

**Myelin**. Внутренний JIT-компилятор TensorRT для некоторых kernel'ов.
Если TRT не может выделить память для Myelin tactic'а, выдаёт `Error
Code 1: Myelin: CUDA error 2 loading a module`.

---

## Что в каждом компоненте можно ещё улучшить (резюме)

| Компонент | Идея | Эффект | Стоимость |
|---|---|---|---|
| audio_tokenizer | speaker prewarm cache (precompute для known voices) | −25 ms TTFA на cached | 1-2 ч |
| speaker_embedding | то же | −20 ms TTFA на cached | (включено выше) |
| token2wav | per-layer mixed-precision DiT plan (fp32 для softmax/sigma, fp16 остальное) | −10-12 % RTF без поломки звука | пол-дня + audio audit |
| vocoder | HiFT → TRT engine | −300 MB / instance, −20-30 ms infer | 4-8 ч |
| LLM | fp8 weights via TRT-LLM (нужен calibration) | −80-150 ms TTFA | 1-2 дня |
| LLM | prefill caching (если trtllm-serve поддерживает) | −30-50 ms TTFA | 4 ч + tests |
| BLS | warmup endpoint при старте сервиса | убрать cold-tail | 30 мин |
| Triton | upgrade 25.06 → latest | unknown gains, regression risk | пол-дня |
| Hardware | 2-я GPU | удваивает capacity, развязывает kv_cache | hardware budget |

Из этого — **самый высокий value/cost: per-layer mixed precision + HiFT→TRT**.
Они вместе могут дать 13-17× → 18-22× realtime + улучшить TTFA. Остальное
дороже.
