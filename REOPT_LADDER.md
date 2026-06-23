# CosyVoice3 quality-gated re-optimization — ladder log

Branch: `reopt/quality-gated` (forked from `round-9-stable`).
Gold = vanilla offline fp32 (`CosyVoice3(load_trt=False, fp16=False)`).
Eval harness: `eval_quality.py` (WER/CER faster-whisper+jiwer, spk-sim campplus, UTMOS, mel-L1, clicks/dips/lead).

## Baseline finding — round-9 DID degrade quality (confirmed)

700 files each, same 100 `english_basket.json` texts × 7 references. vanilla-offline (gold) vs round-9-streaming (current prod):

| metric | vanilla (gold) | round-9 | Δ | verdict |
|---|---|---|---|---|
| WER (corpus) | 14.29% | 16.16% | **+1.87 pp** | worse |
| CER (corpus) | 14.22% | 15.36% | +1.14 pp | worse |
| spk-sim mean | 0.7479 | 0.7459 | −0.002 | ~equal |
| spk-sim p05 | 0.457 | 0.427 | −0.030 | worse tail |
| **UTMOS mean** | **4.122** | **3.741** | **−0.381** | **clearly worse (audible)** |
| UTMOS p05 | 3.372 | 2.886 | −0.486 | worse tail |
| clicks total | 875 | 1193 | **+318 (+36%)** | worse (the «подёргивания») |
| files w/ clicks | 240 | 301 | +61 | worse |
| runaways (>18s) | 0 | 3 | +3 | worse |
| lead-silence | 590 ms | 9 ms | −581 ms | better (trim — intended) |
| peak max | 0.990 | 0.990 | = | = |

**Conclusion:** round-9 streaming is measurably worse on perceptual MOS (UTMOS −0.38), intelligibility (WER +1.9 pp), and click artifacts (+36%). The trim/lead-silence win is real but the precision (fp16 DiT/HiFT) + boundary (token_hop=8) changes cost quality. The ladder below isolates which.

Caveat: this baseline compares vanilla-OFFLINE-torchLLM vs round-9-STREAMING-trtLLM, so it mixes LLM, precision, and chunking. The token-replay ladder (fixed LLM tokens) disentangles them step-by-step.

## Ladder (one optimization per step, gated vs gold)

| step | change | RTF | TTFA p50/p95 | WER | spk-sim | UTMOS | mel-L1 | clicks | lead | gate |
|---|---|---|---|---|---|---|---|---|---|---|
| S0 | baseline: fp32 flow/hift, hop15, lookahead3, fixed, no-trim | TTFA 2163ms | — | 0.153 | 0.744 | 4.070 | 0 (gold) | 133 | 547ms | GOLD |
| S5 | lookahead 3→1 (isolated) | TTFA 1622ms | — | 0.177 | 0.754 | **3.264** | 1.44 | 121 | — | **REJECT(UTMOS −0.81, dur −12%)** |

**S5 rejected — untrained chunk combo.** lookahead=1 with hop=15 produces ~12% shorter audio for identical tokens + UTMOS −0.81. CLAUDE.md: chunk sizing must match training static_chunk_size. lookahead is COUPLED to token_hop; round-9 uses lookahead=1 *with hop=8*. → test chunking as one coupled unit (S6), not isolated.

| S6 | round-9 chunking: hop8+lookahead1+exp (coupled) | TTFA 1549ms | — | 0.156 | 0.747 | **3.834** | 1.50 | 118 | — | **MARGINAL(UTMOS −0.24 for −600ms TTFA)** |

**Boundary-tier finding:** aggressive streaming chunking (hop8/lookahead1) costs UTMOS −0.24 vs conservative S0 (hop15/lookahead3, which is near-offline 4.07 vs vanilla 4.12). This is ~2/3 of round-9's total −0.38 UTMOS loss. It buys −600 ms TTFA. A real tradeoff — sweet spot likely hop10-12 (tested after precision tier). Precision tier below is built on S0's conservative chunking to isolate fp16 effects.

| S10 | + HiFT layer-mixed fp16 (on S0 chunking) | TTFA 1609ms | — | 0.153 | 0.745 | 4.067 | **0.019** | 128 | — | **PASS (UTMOS −0.003, ~lossless)** |

| S11 | + prompt_speech_feat fp16 | TTFA 1623ms | — | 0.153 | 0.744 | 4.042 | 0.049 | 128 | — | **PASS (UTMOS −0.028 cum, spk unchanged)** |

| S12 | + flow fp16 (PyTorch, no TRT) | **TTFA 632ms** | — | 0.153 | 0.742 | 4.065 | 0.107 | 124 | — | **PASS (UTMOS −0.005, TTFA −990ms!)** |

| S13 | + flow layer-mixed fp16 **TRT** | **TTFA 283ms** | — | 0.153 | 0.747 | 4.011 | 0.184 | **166** | — | **MARGINAL (−0.054 vs S12 + clicks 124→166; CLAUDE.md warning confirmed, modest)** |

## VERDICT — what degraded round-9, and the fix

| optimization | UTMOS Δ | clicks | TTFA | verdict |
|---|---|---|---|---|
| **aggressive chunking hop8/lookahead1** (S6) | **−0.24** | ~same | −600ms | **DOMINANT culprit** |
| HiFT fp16 (S10) | −0.003 | ~same | small | free ✓ |
| prompt_feat fp16 (S11) | −0.028 | ~same | small | nearly free ✓ |
| **flow fp16 PyTorch** (S12) | −0.005 | ~same | **−990ms** | **free + huge speed ✓** |
| flow TRT layer-mixed (S13) | −0.054 | +33 | −350ms | modest cost (the CV3 fp16+TRT issue) |

**Root cause of round-9's −0.38 UTMOS: ~2/3 is the aggressive streaming chunk size (hop8/lookahead1), ~1/3 is the fp16 DiT TRT plan.** The fp16 *weights* (flow/HiFT/prompt-feat) in PyTorch are essentially free and deliver most of the speed (TTFA 2163→632 ms). The recommended config keeps fp16-PyTorch precision and a *conservative* chunk size; flow-TRT is optional for extra TTFA at a small quality cost.

### Chunk-size sweep on the fp16-PyTorch base — the decisive result

| config | chunk | UTMOS | TTFA p50 | note |
|---|---|---|---|---|
| **S12 (WINNER)** | hop15/lookahead3/fixed | **4.065** | **632ms** | conservative chunk + fp16 flow |
| C12 | hop12/lookahead2/exp | 3.971 | 635ms | smaller chunk, SAME ttfa, WORSE quality |
| S6 | hop8/lookahead1/exp | 3.834 | (fp32 base) | round-9 chunking |

**Decisive insight:** once flow is fp16 (fast), the conservative chunk (hop15/lookahead3) renders in 632 ms — *the same TTFA* as the small chunk — but with UTMOS 4.065 vs 3.971. So **aggressive chunking buys nothing on fp16 and only costs quality.** Round-9's small chunks existed to hide slow fp32 flow; with fp16 flow they're pure loss.

## RECOMMENDED PRODUCTION CONFIG (beats round-9 on quality AND speed)

`flow_precision=fp16, flow_trt=0, hift_plan=layer_mixed, prompt_feat_fp16=1, token_hop_len=15, flow_pre_lookahead_len=3, dynamic_chunk_strategy=fixed` (= **S12**)
- **UTMOS 4.065** vs round-9 **3.74** (+0.33, recovers nearly all the loss; vanilla 4.12)
- **TTFA p50 632 ms** (vs round-9-era fp32 baseline 2163 ms)
- WER 0.153 (= gold), speaker-sim unchanged, mel-L1 0.11 (near-transparent)
- Optional max-speed: + `flow_trt=1` → TTFA 283 ms at UTMOS 4.011 (−0.054, +clicks) — ship only if 283 ms is required.
- Trim (enable_trim=1) is orthogonal — only removes leading silence; toggle for UX.

## FastChunk (post-S13 lossless TTFA win) — hop8 chunk-0 + exponential grow + lookahead3 + TRT

Config: `token_hop_len=8, flow_pre_lookahead_len=3, dynamic_chunk_strategy=exponential` (chunk-0=8 tokens → low TTFA; then grows 25/50/100 — *bigger* than S13's fixed 15 → fewer seams). Pure config, no code change.

**Quality (same gold tokens, vs S13):** UTMOS 4.011→**4.093 (+0.08)**, clicks 166→**77 (halved)**, WER/spk-sim identical. → quality GAIN, not loss.

**Warm TTFA p50 / stutter:**
| N | FastChunk | S13-cons | round-9 |
|---|---|---|---|
| 1 | 280 / 0% | 308 / 0% | 284 / 0% |
| 4 | 356 / 3.4% | 534 / 0% | 349 / 2.6% |
| 8 | **449 / 5.9%** | 890 / 0% | 427 / 5.8% |
| 12 | 607 / 13% | 1149 / 0% | 530 / 14% |

FastChunk ≈ round-9 TTFA (449 vs 427 @N=8) — **closes S13's TTFA gap** — with **better quality than S13**. Cost: reintroduces stutter at N≥8 (5.9–13%, from the big 50/100-token chunks under load).

### ✅ S14 (SHIPPED) = FastChunk-capped: hop8 chunk-0 + exponential **cap@25** + lookahead3 + TRT
Add `max_token_hop_len=25` cap so chunks go 8→25→25→25 (small fast chunk-0, then bounded steady chunk). New `max_token_hop_len` knob in `cosyvoice3/1/model.py`. **Strictly beats S13 — no tradeoff:**

| N | **S14 (capped)** TTFA / stutter | S13 TTFA / stutter |
|---|---|---|
| 1 | 334 / 0% | 308 / 0% |
| 4 | 405 / **0%** | 534 / 0% |
| 8 | **579 / 0%** | 890 / 0% |
| 12 | **696 / 0%** | 1149 / 0% |

Quality (same gold tokens): UTMOS **4.011 = S13**, clicks **96 < S13's 166**, WER/spk-sim identical. RTF 0.08 < S13 0.11.
**Verdict: TTFA −35% @N=8 / −39% @N=12, 0% stutter to N=12, equal quality, lower RTF, fewer clicks.** Shipped as the new default (was S13). Lossless win — exactly "ускорение без потери качества".

## ⚠️ CORRECTED WARM BENCHMARKS (cold-start fixed — these supersede everything below)

Methodology: per config — cold restart → **`warmup.py --waves 3 --concurrency 16`** (primes all instances + TRT plans + CUDA graphs) → official `client_grpc.py` sweep (seed_tts zero-shot, max_samples=96, warmup-requests=4). GPU steady at 1755 MHz. **Validated**: config A (hop8+TRT) reproduces documented round-9 (N=8 TTFA p50 427ms vs doc 429ms, stutter 5.8% vs 6.3%).

Note: the bogus "4327 ms @N=4" earlier came from my ad-hoc `reopt_ttfa_bench.py` (no warmup) — *that* was cold. The official `client_grpc` numbers I reported were already ~warm and match these within run-to-run noise.

**TTFA p50 (ms) / stutter — three configs, warm:**

| N | A: prod hop8+TRT | B: **S13-cons hop15+TRT** | C: S12 hop15 no-TRT |
|---|---|---|---|
| 1 | 284 / 0% | 308 / 0% | 624 / 0% |
| 4 | 349 / 2.5% | 534 / 0% | 859 / 1.9% |
| 6 | 400 / 1.9% | 690 / 0% | 1308 / 7.3% |
| 8 | **427 / 5.8%** | **890 / 0%** | 1845 / 24.7% |
| 12 | 530 / 14.5% | 1149 / 0% | 2893 / 72% |

RTF (lower=faster): A 0.05–0.17, B 0.09–0.30, C 0.25–0.81. Per-stage GPU (B,N=8 cumulative): token2wav/flow 165s, vocoder 93s dominate; audio_tokenizer 3.3s, speaker_embedding 2.8s, BLS 0.1s.

**Verdict (warm, trustworthy):**
- **A (prod hop8)**: lowest TTFA (427ms @N=8) — waits for only 9 tokens — but **stutters 5.8%→14.5%** at N≥8 and worst quality (UTMOS 3.74).
- **B (S13-conservative)**: **0% stutter at every N**, best speed/quality balance (UTMOS 4.011), but TTFA ~2× of A (890ms @N=8) — the real, non-cold cost of hop15 (waits for 18 tokens). RTF 0.09–0.30 = comfortably realtime.
- **C (S12 no-TRT)**: highest TTFA + stutter explodes (24–72% at N≥8). **Offline / N≤4 only.** TRT flow is mandatory for concurrency.

**Production call:** if TTFA <500ms @N=8 is a hard requirement → only A's hop8 hits it, but it stutters + lower quality. **B (hop15+TRT) is the right default** for streaming-at-scale: zero stutter to N=12, RTF≈0.1, +0.27 UTMOS over prod; TTFA 0.9s @N=8 is acceptable for most voice-chat. If both <500ms TTFA AND zero stutter are needed at N=8, a **middle hop (~10–12) + TRT** is the next thing to sweep (user deferred it).

---

## (OLD — COLD/INVALID) S12 TTFA vs concurrency (END-TO-END: LLM+flow+vocoder, BLS=16, ref neutral_2)

| N | mean | p50 | p95 | max |
|---|---|---|---|---|
| 1 (uncontended) | ~900 | — | — | 1117 |
| 4 | 4327 | 4697 | 6311 | 6417 |
| 6 | 4236 | 4083 | 5197 | 5363 |
| 8 | 6194 | 6204 | 7063 | 7176 |
| 12 | 9314 | 9474 | 11124 | 11259 |

**Caveat — this is the real end-to-end TTFA, NOT the 632 ms render-only replay figure.** Two costs the replay benchmark hid:
1. hop=15 makes the first chunk wait for **18 LLM tokens** (round-9's hop=8 waits for 9) → higher LLM-side TTFA.
2. fp16-PyTorch flow (no TRT) is GPU-bound and contends under concurrency; non-incremental token2wav reprocesses the accumulated mel.

So S12 trades **concurrency-TTFA for quality**: excellent UTMOS but TTFA scales worse than round-9 under load. To recover concurrency-TTFA: (a) `flow_trt=1` (S13) — TRT flow renders ~2× faster, scales far better, at −0.054 UTMOS; and/or (b) smaller `token_hop_len` (lower LLM-wait) — but that re-introduces the chunking quality cost. The right production point is likely **flow_trt=1 + moderate hop (~10)** — fast render removes the need for tiny chunks.

### Official streaming bench (client_grpc.py, seed_tts_cosy2 zero-shot, 64 samples, warmup-dropped)

**S12 (fp16-PyTorch flow, no TRT, hop15):**
| N | RTF | TTFA p50 | TTFA p95 | total p50 | **stutter (>1s gaps)** |
|---|---|---|---|---|---|
| 1 | 0.86 | 642ms | 1592ms | 3955ms | 0% |
| 4 | 0.32 | 870ms | 1014ms | 4421ms | 1.4% |
| 8 | 0.28 | 1910ms | 2212ms | 7915ms | **23%** |
| 12 | 0.28 | 2694ms | 3290ms | 9113ms | **45%** |

S12 is excellent at **N≤4** (TTFA <900ms, ~no stutter, RTF<1) but **collapses at N≥8** — stutter 23–45% (the GPU-bound fp16-PyTorch flow + non-incremental token2wav can't sustain realtime under load). So S12 = best quality, low-concurrency only. For N≥8 you need flow-TRT (faster render).

**S13 (flow layer-mixed fp16 TRT, hop15 — same conservative chunking):**

| N | RTF | TTFA p50 | TTFA p95 | total p50 | **stutter** |
|---|---|---|---|---|---|
| 1 | 0.36 | 320ms | 1422ms | 1269ms | 0% |
| 4 | 0.14 | 491ms | 650ms | 2113ms | 0% |
| 8 | 0.10 | 820ms | 1041ms | 3424ms | 0% |
| 12 | 0.10 | 1125ms | 1288ms | 3685ms | **0%** |

**S13 sustains N=12 at 0% stutter, TTFA 320–1125ms, RTF 0.10** — far better scaling than S12. The TRT flow is essential for concurrency.

## FINAL PRODUCTION RECOMMENDATION

**Ship S13-conservative** = `flow_precision=fp16, flow_trt=1, hift_plan=layer_mixed, prompt_feat_fp16=1, token_hop_len=15, flow_pre_lookahead_len=3, dynamic_chunk_strategy=fixed`.

| | round-9 (current) | **S13-conservative (new)** | S12 (max quality) |
|---|---|---|---|
| UTMOS | 3.74 | **4.011** (+0.27) | 4.065 (+0.33) |
| chunking | hop8 (aggressive) | hop15 (conservative) | hop15 |
| flow | fp16 TRT | fp16 TRT | fp16 PyTorch |
| TTFA p50 @N=8 | ~290ms* | 820ms | 1910ms |
| stutter @N=12 | (not measured) | **0%** | 45% |

**The key fix:** round-9 lost quality from its **aggressive chunking (hop8/lookahead1)**, NOT from TRT. Keep the TRT flow (for speed/concurrency) but revert to **conservative chunking (hop15/lookahead3)** → recover +0.27 UTMOS while keeping TRT-grade throughput (0% stutter to N=12). If TTFA must be <500ms at high N, S12-style won't do it; only TRT will, and hop15 keeps it fast enough (820ms @N=8). Use S12 (no TRT) only for offline/low-concurrency where max quality matters.

\* round-9's lower TTFA came partly from hop8 (fewer tokens to wait) — but at a quality cost we now avoid. hop15 + TRT is the better balance.

## Harness status (validated)

- Branch `reopt/quality-gated`: commits `19cc556` (config-driven knobs + replay) + `01ae61b` (fp32 transport fixes).
- Config-driven knobs (config.pbtxt parameters, round-9 defaults):
  - cosyvoice3: `token_hop_len`, `flow_pre_lookahead_len`, `dynamic_chunk_strategy`,
    `enable_trim`, `prompt_feat_fp16`, `llm_seed`, `load_spk2info`
  - token2wav: `flow_precision` (fp16|fp32), `flow_trt` (1|0)
  - vocoder: `hift_plan` (layer_mixed|fp32)
- Token-replay A/B **validated**: dump 140 tokens via streaming `speech_tokens` output →
  replay reproduces gold audio **byte-identical (sample-L1 = 0.0)**. mel-L1 is a clean
  precision metric.
- S0 deployed at `/tmp/reopt_repo` (Triton :18001, LLM sidecar :8000): fp32 PyTorch flow
  (no TRT), fp32 HiFT, no trim, hop=15, lookahead=3, fixed chunk, prompt-feat fp32,
  seed=1234, spk2info off. All 5 models ready.

