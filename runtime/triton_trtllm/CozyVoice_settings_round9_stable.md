# CosyVoice3 — Actual Runtime Parameters (`round-9-stable` HEAD `8780042`)

> **Diff vs `CozyVoice_settings.md`**: фиксирует разрывы между документацией и
> кодом текущей `round-9-stable` ветки. Колонка **Doc** — из изначального
> файла; колонка **Actual** — что реально в коде на HEAD `8780042`; **Δ** —
> mismatch flag.

## Pipeline

| Area | Parameter | Doc | Actual (round-9-stable) | Δ |
|------|-----------|-----|-------------------------|---|
| Pipeline | Orchestrator | Triton BLS `cosyvoice3` → audio_tokenizer, speaker_embedding, token2wav, vocoder | ✓ same | — |
| Pipeline | LLM backend | `trtllm-serve` (HTTP `/v1/chat/completions`, port **8010** default) | BLS default port **8000** (`api_base` line 72), entrypoint exports `LLM_PORT=8010` and patches via `llm_api_base` param | ⚠ doc описывает entrypoint default; BLS code default = 8000 |
| Pipeline | Model weights | `Fun-CosyVoice3-0.5B-2512/` + HF `cosyvoice3_llm/` | ✓ same | — |

## Streaming (BLS)

| Parameter | Doc | Actual | Δ |
|-----------|-----|--------|---|
| `token_hop_len` | **8** | 8 | — |
| `flow_pre_lookahead_len` | **1** | 1 | — |
| `token_mel_ratio` | **2** | 2 | — |
| `token_frame_rate` | **25** | 25 | — |
| LLM tokens before 1st chunk | **9** (= 8+1) | 9 | — |
| Hop growth | Exponential `25 × 2^chunk_index` after 1st | ✓ same | — |
| `dynamic_chunk_strategy` | `exponential` | ✓ same | — |
| Mel per stable hop | ~16 frames | ✓ same | — |
| Audio output | Incremental `speech[:, speech_offset:]`; no server crossfade | ✓ same | — |

## Flow / DiT

| Parameter | Doc | Actual | Δ |
|-----------|-----|--------|---|
| Solver | Euler, `sigma_min=1e-6`, `cosine t_scheduler` | ✓ same | — |
| `n_timesteps` | **10** default; **5** first streaming chunk only | **10** default; **NO first-chunk override** | **⚠ MISMATCH** — first-chunk fast path (B2) лежит на `improve_cosyvoice3`, не в round-9-stable |
| `inference_cfg_rate` | 0.7 | 0.7 | — |
| `training_cfg_rate` | 0.2 | 0.2 | — |
| DiT dim/depth/heads/dim_head | 1024 / 22 / 16 / 64 | ✓ same | — |
| `ff_mult` / I/O dims | 2; 80 | ✓ same | — |
| `static_chunk_size` | 50 mel | ✓ same | — |
| `num_decoding_left_chunks` | -1 | ✓ same | — |
| Pre-lookahead layer | `PreLookaheadLayer(80→1024, len=3)`; runtime ctx 1 | ✓ same | — |
| CFM `in_channels` | 240 | ✓ same | — |
| Flow input per chunk | `tokens[:, :token_offset+hop+lookahead]` + prompt | ✓ same | — |
| DiT TRT seq min/opt/max | 4 / 500 / 3000 | 4 / 500 / 3000 (per `_build_layer_mixed_trt`) | — |
| TRT inputs | `x, mask, mu, cond` `[B,80,T]`; `t, spks` fixed | ✓ same | — |
| TRT I/O dtype | **FP16** IO; mixed internal | **FP16 IO**, FP32 default + FP16 на ~75 sensitive layers | ⚠ inverted description in doc (default is FP16 with FP32 sensitive override; doc says "mixed internal") |
| TRT plan file | `flow.decoder.estimator.layer_mixed_B8_fp16.{device_id}.plan` | `flow.decoder.estimator.layer_mixed_fp16.{device_id}.plan` (NO `B8` suffix) | **⚠ MISMATCH** — `B8` plan лежит на `improve_cosyvoice3` (с B-dynamic profile для batching) |
| TRT build batch (ONNX) | 2 (CFG pair) | 2 (B=2 hardcoded в opt profile shapes) | — |
| CFG | Always: cond + uncond; `(1+0.7)·cond − 0.7·uncond` | ✓ same | — |
| Estimator batch per step | `2 × B_flow` (B_flow=1 → TRT B=2) | ✓ same (round-9 не batches) | — |
| FP32 layers | ~75 by rule (Norm/Softmax/time_embed/proj_out/sensitive) | 75 — экземпляр precision constraints в `_build_layer_mixed_trt` | — |
| token2wav `max_batch_size` | **8**; instances **4**; BLS batch up to 8 | **`max_batch_size: 1`**; **`count: 8`** | **⚠ MISMATCH** обоих полей — batching нет в round-9-stable |
| Warmup chunk0 shape | 9 target tok, 25 prompt tok, 50×80 prompt mel | Triton `model_warmup`: **9 / 125 / 250** (typical 5s ref) | ⚠ prompt size: doc 25, actual 125 (after Tier-A A4) |

## HiFT

| Parameter | Doc | Actual | Δ |
|-----------|-----|--------|---|
| Class | `CausalHiFTGenerator` | ✓ same | — |
| `in_channels` / `base_channels` | 80 / 512 | ✓ same | — |
| `upsample_rates` / kernels | [8,5,3] / [16,11,7] → 512→256→128→64 | ✓ same | — |
| `upsample_total` | 120 | ✓ same | — |
| Resblocks | 3/stage k=[3,7,11] d=[1,3,5] | ✓ same | — |
| Source resblocks | k=[7,7,11] same dilations | ✓ same | — |
| `conv_pre` | Causal 80→512 k=5, `conv_pre_look_right=4` | ✓ same | — |
| `conv_post` | Causal 64→18 k=7 | ✓ same | — |
| ISTFT | n_fft=16, hop_len=4 → 24 kHz | ✓ same | — |
| TRT subgraph `decode_core` | ups + resblocks + conv_post + exp/sin | ✓ same | — |
| TRT inputs | `x_pre [B,512,T_pre]`, `s_stft [B,18,T_stft]` fp32 | ✓ same | — |
| TRT outputs | `magnitude, phase [B,9,T_stft]` fp32 | ✓ same | — |
| B=1 profile `x_pre` T | min/opt/max **1/16/2500** | 1/16/2500 (per `_build_hift_fp32_trt`) | — |
| B=1 profile `s_stft` T | min/opt/max **121/1921/300001** | 121/1921/300001 | — |
| **B-dynamic plan** | `hift_decode_core.fp32_B8.plan`; batch **1/4/8** opt **4** | **NOT loaded** by round-9 code; plan exists on disk only as artifact from improve_cosyvoice3 | **⚠ MISMATCH** — round-9 loads `hift_decode_core.layer_mixed_fp32io.plan` (layer-mixed precision) с B=1 only |
| B=1 plan | `hift_decode_core.fp32.plan` | Fallback only (primary = layer_mixed_fp32io) | ⚠ partial |
| **Layer-mixed plan** | NOT in doc | **`hift_decode_core.layer_mixed_fp32io.plan`** primary (FP16 default + 507 FP32 sensitive layers via NAME-based matching) | **⚠ DOC LACKS** entry для layer-mixed precision (commit `92dd1bf`) |
| PyTorch (not TRT) | f0_predictor (CPU fp64), m_source, STFT, conv_pre, iSTFT, clamp | ✓ same | — |
| Streaming trim (`finalize=False`) | mel/s_stft/audio causal trim; 480 samples audio | ✓ same | — |
| Chunk overlap/crossfade | No on server; optional client | ✓ same | — |
| vocoder `max_batch_size` | **8**; instances **4** (CPU group, GPU in code) | **`max_batch_size: 1`**; **`count: 8`** | **⚠ MISMATCH** обоих полей |

## LLM

| Parameter | Doc | Actual | Δ |
|-----------|-----|--------|---|
| Base model | Qwen2 ~0.5B | ✓ same | — |
| Engine dtype | **bfloat16** | ✓ same | — |
| `trtllm-build` | max_batch_size 64, max_num_tokens 32768 | ✓ same | — |
| Serve `max_batch_size` | 64 | ✓ same | — |
| Serve `kv_cache_free_gpu_memory_fraction` | **0.4** | Entrypoint default **0.4**; **actual running 0.3** (external override на этой машине) | ⚠ runtime drift |
| Batching | TRT-LLM continuous / in-flight (up to 64) | ✓ same | — |
| Vocab | Text specials + `<\|s_i\|>`; 6561 + 200 speech | ✓ same | — |
| `padded_vocab_size` | Multiple of 128 | ✓ same | — |
| Logits | Full lm_head; speech band at `speech_token_offset` | ✓ same | — |
| Chat template | User: `<\|sos\|>text<\|task_id\|>`; Assistant: prompt tokens | ✓ same | — |
| Prefill | Full chat (ref text + target + assistant tokens) | ✓ same | — |
| Decode | Autoregressive SSE; parse `<\|s_N\|>` | ✓ same | — |
| `max_tokens` | **200** streaming, **400** offline | **200** streaming, **200** offline (changed in Tier-A A3, both paths) | **⚠ MISMATCH** offline теперь тоже 200 |
| Sampling | T=0.8, top_p=0.95, top_k=50, rep_penalty=1.1 | ✓ same | — |
| Stop tokens | `<\|eos1\|>`, `<\|eos\|>` | ✓ same | — |
| BLS concurrency | `bls_max_concurrent: 8/instance`; Triton `max_batch_size: 1` | **NO `bls_max_concurrent` semaphore** в round-9-stable; cosyvoice3 `max_batch_size: 1` ✓ | **⚠ MISMATCH** — semaphore это `improve_cosyvoice3` feature (commit с Tier-2 A5), не в round-9-stable |

## Speaker

| Parameter | Doc | Actual | Δ |
|-----------|-----|--------|---|
| Embedding | CAMPPlus → 192-dim; TRT fp32 | ✓ same | — |
| Default cached speaker | `ref` in `spk2info.pt` (252 prompt tok, 504×80 mel) | spk2info loaded if file exists; **no auto-bake в round-9-stable** (это improve_cosyvoice3 feature) | **⚠ MISMATCH** — default install не имеет spk2info; нужно баковать вручную через `generate_spk2info.py` |

## Prod tuning

| Parameter | Doc | Actual | Δ |
|-----------|-----|--------|---|
| `t2w_dispatch_wait_ms` / `voc_dispatch_wait_ms` | **0** (optional 10 ms для batching) | **NO coordinator** в round-9-stable; параметры не существуют в config | **⚠ MISMATCH** — BLS coordinators это `improve_cosyvoice3` feature (commit `5492327`), не в round-9-stable |
| BLS instances | **10** default entrypoint (**2** в H100 report) | Entrypoint `BLS_INSTANCE_NUM=10` default ✓; H100 SYNC fact использовал **16** (не 2) | ⚠ H100 report number — `16`, не `2` (см. `SYNC_ROUND9_2026-05-20.md`) |

---

## Summary — что добавить в основную доку

8 расхождений с реальным кодом `round-9-stable`:

1. **DiT TRT plan filename** — `layer_mixed_fp16` (NO `_B8` suffix; B-dynamic это improve_cosyvoice3)
2. **token2wav** — `max_batch_size: 1`, `count: 8` (doc says 8/4)
3. **vocoder** — `max_batch_size: 1`, `count: 8` (doc says 8/4)
4. **DiT first-chunk `n_timesteps=5`** — NOT в round-9-stable (Tier-B B2 лежит на improve_cosyvoice3)
5. **HiFT primary plan** — `hift_decode_core.layer_mixed_fp32io.plan` (layer-mixed FP16 + 507 FP32 sensitive layers via NAME match); `fp32.plan` fallback; **`fp32_B8.plan` НЕ загружается** round-9 кодом
6. **LLM `max_tokens` offline** — 200 (не 400; изменено в Tier-A A3)
7. **`bls_max_concurrent` semaphore** — НЕТ в round-9-stable (Tier-2 A5 only в improve_cosyvoice3)
8. **`t2w_dispatch_wait_ms` / `voc_dispatch_wait_ms`** — НЕТ в round-9-stable (Tier-4 coordinators only в improve_cosyvoice3)
9. **spk2info auto-bake** — НЕТ в round-9-stable (entrypoint hardening only в improve_cosyvoice3)
10. **H100 BLS instances в SYNC report** — **16**, не **2**



Audio peak ≤0.99, clip 0% across 12 EN/RU samples.
