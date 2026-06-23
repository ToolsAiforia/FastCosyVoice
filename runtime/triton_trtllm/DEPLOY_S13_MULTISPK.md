# Deploy: CosyVoice3 S13-conservative + multi-speaker image

Self-contained handoff to **build and push the production image** on a Docker host.
Everything below runs in `runtime/triton_trtllm/` unless noted.

- **Source:** repo `git@github.com:acclaim-ai/FastCosyVoice.git`, branch **`reopt/quality-gated`** (build/verify against commit `c1ee355` or later).
- **What this image is:** the quality-gated re-optimization result — **S13-conservative** serving config + **18 baked speakers** (`emily` default + `spk01..spk17`).
- **Image name:tag (default):** `cosyvoice3-tts:s13-multispk`.

---

## 0. TL;DR
```bash
REGISTRY=<REGISTRY>            # <-- SET THIS, e.g. ghcr.io/acclaim-ai  (no trailing slash)
IMAGE=cosyvoice3-tts:s13-multispk

git clone git@github.com:acclaim-ai/FastCosyVoice.git
cd FastCosyVoice && git checkout reopt/quality-gated
cd runtime/triton_trtllm

pip install -U "huggingface_hub[cli]"
bash download_cosyvoice3_models.sh          # fetches the 2 weight dirs (see §2)

# .dockerignore so the 19GB model + bench junk don't bloat the build context wrongly
printf 'bench_*/\n*.log\n__pycache__/\n*.bak*\n*.plan\nspk2info.pt\n' > .dockerignore

DOCKER_BUILDKIT=1 docker build --ssh default -f Dockerfile.cosyvoice3 -t "$IMAGE" .
docker login "${REGISTRY%%/*}"
docker tag  "$IMAGE" "$REGISTRY/$IMAGE"
docker push "$REGISTRY/$IMAGE"
```
> The `--ssh default` + BuildKit is needed because the Dockerfile's `pip install
> git+...` pulls from the **private** `acclaim-ai` repo at build time. See §3 for
> alternatives (HTTPS PAT). If you switch to a token, you can drop `--ssh default`.

---

## 1. Prerequisites (build host)
- **Docker** with **BuildKit** (`DOCKER_BUILDKIT=1`, Docker ≥ 20.10).
- **~60 GB free disk** (base ~10 GB + weights ~20 GB + layers → final image ~25–30 GB).
- **Network**: pulls base `soar97/triton-cosyvoice:25.06` (public) + pip git clone (private acclaim-ai) + HuggingFace weights.
- **Registry access** (`docker login`).
- **GPU is NOT needed to build** (build = copy + pip). GPU is needed only at **container run** — the entrypoint builds per-GPU TRT plans and bakes `spk2info.pt` on first start.

## 2. Weights (not in git — download on the host)
`download_cosyvoice3_models.sh` fetches into `runtime/triton_trtllm/`:
- `Fun-CosyVoice3-0.5B-2512/` (~19 GB) ← `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` + `yuekai/Fun-CosyVoice3-0.5B-2512-FP16-ONNX`
- `cosyvoice3_llm/` (~1.3 GB) ← `yuekai/Fun-CosyVoice3-0.5B-2512-LLM-HF`

Already in git (come with the checkout): `model_repo_cosyvoice3/` (the S13 BLS), `speakers/` (18 voices), `Dockerfile.cosyvoice3`, `entrypoint_cosyvoice3.sh`, scripts.

⚠️ **Do NOT copy `*.plan` files from another machine** — they are per-GPU (built for a specific SM). Download weights fresh (no plans) → the entrypoint builds them for the target GPU. The `.dockerignore` above also excludes them.

## 3. Private-repo pip auth (build time)
The Dockerfile line:
```
pip install --no-cache-dir --no-deps "git+https://github.com/acclaim-ai/FastCosyVoice@reopt/quality-gated"
```
`acclaim-ai/FastCosyVoice` is private, so the build needs git auth. Pick one:

**(a) BuildKit SSH (recommended)** — change the URL to SSH and pass your agent:
```dockerfile
# in Dockerfile.cosyvoice3, swap the pip line URL to:
"git+ssh://git@github.com/acclaim-ai/FastCosyVoice.git@reopt/quality-gated"
```
```bash
eval "$(ssh-agent)"; ssh-add ~/.ssh/<key_with_repo_access>
DOCKER_BUILDKIT=1 docker build --ssh default -f Dockerfile.cosyvoice3 -t "$IMAGE" .
```
**(b) HTTPS PAT** — keep the https URL but inject a token (use a BuildKit secret, don't bake it):
```bash
# simplest (token visible in build history — use only in trusted CI):
sed -i 's#github.com/acclaim-ai#<PAT>@github.com/acclaim-ai#' Dockerfile.cosyvoice3
```
> Note: the `cosyvoice` python lib is unchanged vs `round-9-stable`; the ref bump is
> only for source-of-truth. If private-auth is painful, pointing the pip line back to
> a public mirror at `@reopt/quality-gated`-equivalent code is functionally identical.

## 4. Build / tag / push
```bash
REGISTRY=<REGISTRY>
IMAGE=cosyvoice3-tts:s13-multispk
DOCKER_BUILDKIT=1 docker build --ssh default -f Dockerfile.cosyvoice3 -t "$IMAGE" .
docker login "${REGISTRY%%/*}"
docker tag  "$IMAGE" "$REGISTRY/$IMAGE"
docker push "$REGISTRY/$IMAGE"
```

## 5. Run + smoke verify (on a GPU host)
First start does: build TRT-LLM engine (if absent) → build per-GPU flow/HiFT TRT plans → bake `spk2info.pt` from `speakers/` (~1–2 min) → launch Triton + trtllm-serve.
```bash
docker run --gpus all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  "$REGISTRY/$IMAGE"
# wait for "ALL models ready" / health 200, then a streaming request with
# inputs: target_text (BYTES) + speaker_name (BYTES, e.g. "emily" / "spk07").
# No speaker_name -> falls back to emily (first baked).
```
Sanity expectations (warm, H100, zero-shot): RTF ~0.1, TTFA p50 ~0.3–0.9 s up to N≈8, **0% stutter to N=12**. Audio peak ≤ 0.99.

## 6. Shipped config (S13-conservative) — defaults baked in `model_repo_cosyvoice3`
| knob | value | where |
|---|---|---|
| `token_hop_len` | **15** | cosyvoice3/1/model.py |
| `flow_pre_lookahead_len` | **3** | cosyvoice3/1/model.py |
| `dynamic_chunk_strategy` | **fixed** | cosyvoice3/1/model.py |
| `enable_trim` | **0** (off) | cosyvoice3/1/model.py |
| `prompt_feat_fp16` | 1 | cosyvoice3/1/model.py |
| `flow_precision` / `flow_trt` | **fp16 / 1 (TRT)** | token2wav/1/model.py |
| `hift_plan` | **layer_mixed** | vocoder/1/model.py |
| `load_spk2info` | 1 | cosyvoice3/1/model.py |

Quality vs old prod: **UTMOS 4.01 vs round-9 3.74** (+0.27), 0% stutter to N=12. To switch back to round-9's lowest-TTFA-but-lower-quality chunking, set via `config.pbtxt` parameters: `token_hop_len=8, flow_pre_lookahead_len=1, dynamic_chunk_strategy=exponential`.

## 7. Speakers (API `speaker_name`)
`emily` (default, neutral_2 business voice) + `spk01`..`spk17` (US/Canada, UTMOS≥4.1, SECS≥0.95). Full mapping in `speakers/INDEX.md`. Re-baked at container start from `speakers/<name>.{wav,txt}`. A request with no `speaker_name` uses `emily`.

## 8. Gotchas
- **Rare LLM runaway** (~0.3%): a short text occasionally generates to `max_tokens`. Handle with a **client-side retry** on empty/over-long output — one retry almost always fixes it.
- **First-start latency**: TRT plan build + spk2info bake take a few minutes on first boot (per GPU). Subsequent boots reuse them (mount the model dir as a volume to persist).
- **Registry value** is a placeholder `<REGISTRY>` — set it (e.g. `ghcr.io/acclaim-ai`).
