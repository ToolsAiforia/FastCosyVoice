# Build Plan: Triton TRT-LLM CosyVoice3 Docker Image

## Overview

Build `Dockerfile.cosyvoice3` which packages CosyVoice3 TTS into a production Docker image with Triton Inference Server + TensorRT-LLM.

## Prerequisites

- GPU with NVIDIA drivers + Docker with `--gpus` support
- `huggingface-cli` (from `pip install huggingface_hub`)
- `trtllm-build` (available inside `soar97/triton-cosyvoice:25.06` base image)

## Required Artifacts

The Dockerfile expects 4 directories in `runtime/triton_trtllm/`:

| Directory | Source | Size |
|-----------|--------|------|
| `Fun-CosyVoice3-0.5B-2512/` | HuggingFace: `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` + `yuekai/Fun-CosyVoice3-0.5B-2512-FP16-ONNX` | ~4.5 GB |
| `cosyvoice3_llm/` | HuggingFace: `yuekai/Fun-CosyVoice3-0.5B-2512-LLM-HF` | ~1.3 GB |
| `cosyvoice3_trt_engines/` | Built from `cosyvoice3_llm/` via `trtllm-build` (GPU-specific!) | ~1.3 GB |
| `model_repo_cosyvoice3/` | Already in git | small |

## Option A: All-in-one (if `trtllm-build` available on host)

```sh
cd runtime/triton_trtllm
bash download_cosyvoice3_models.sh
docker build -f Dockerfile.cosyvoice3 -t cosyvoice3-tts:v2 .
```

## Option B: Step by step (build engine inside base image)

### Step 1: Download models (no GPU needed)

```sh
cd runtime/triton_trtllm
bash download_cosyvoice3_models.sh --skip-build
```

This runs:
```sh
huggingface-cli download --local-dir ./Fun-CosyVoice3-0.5B-2512 FunAudioLLM/Fun-CosyVoice3-0.5B-2512
huggingface-cli download --local-dir ./Fun-CosyVoice3-0.5B-2512 yuekai/Fun-CosyVoice3-0.5B-2512-FP16-ONNX
huggingface-cli download --local-dir ./cosyvoice3_llm yuekai/Fun-CosyVoice3-0.5B-2512-LLM-HF
```

### Step 2: Build TRT-LLM engine (needs GPU)

Start the base image interactively with the download directory mounted:

```sh
docker run --rm -it --gpus '"device=0"' \
    -v $(pwd):/workspace/build \
    soar97/triton-cosyvoice:25.06 bash
```

Inside the container:

```sh
cd /workspace/build
TRT_DTYPE=bfloat16

python3 scripts/convert_checkpoint.py \
    --model_dir ./cosyvoice3_llm \
    --output_dir /tmp/trt_weights \
    --dtype $TRT_DTYPE

trtllm-build \
    --checkpoint_dir /tmp/trt_weights \
    --output_dir ./cosyvoice3_trt_engines \
    --max_batch_size 64 \
    --max_num_tokens 32768 \
    --gemm_plugin $TRT_DTYPE

exit
```

### Step 3: Build Docker image

```sh
docker build -f Dockerfile.cosyvoice3 -t cosyvoice3-tts:v2 .
```

### Step 4: Run

```sh
docker run -d --name cosyvoice3 \
    --gpus '"device=0"' --shm-size=1g \
    -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    cosyvoice3-tts:v2
```

## Important Notes

- **TRT-LLM engines are GPU-architecture-specific.** An engine built on sm86 (RTX 3090) won't work on sm120 (Blackwell). Build on the target GPU.
- **Base image** (`soar97/triton-cosyvoice:25.06`): If you need to rebuild it: `docker build -f Dockerfile.server -t soar97/triton-cosyvoice:25.06 .` — it's based on `nvcr.io/nvidia/tritonserver:25.06-trtllm-python-py3` + torchaudio from source + `requirements.txt`.

## What happens at container startup

The entrypoint (`entrypoint_cosyvoice3.sh`):

1. Fills Triton config templates in `model_repo_cosyvoice3/*/config.pbtxt` with paths and env var settings
2. Starts `trtllm-serve` on port 8010 (LLM inference, called over HTTP by the BLS model)
3. Starts `tritonserver` on ports 8000/8001/8002 with 5 Triton models:
   - `cosyvoice3` — BLS orchestrator (routes requests through pipeline)
   - `audio_tokenizer` — reference audio → speech tokens (s3tokenizer v3)
   - `speaker_embedding` — 192-dim speaker embedding (CAMPPlus)
   - `token2wav` — speech tokens → mel spectrogram (flow matching DiT)
   - `vocoder` — mel → waveform (HiFT)

## Configuration (env vars for `docker run`)

| Variable | Default | Description |
|----------|---------|-------------|
| `BLS_INSTANCE_NUM` | 10 | Number of BLS model instances |
| `DECOUPLED_MODE` | True | Enable streaming mode |
| `LLM_PORT` | 8010 | Internal trtllm-serve port |
| `LLM_MAX_BATCH_SIZE` | 64 | LLM max batch size |
| `LLM_KV_CACHE_FRACTION` | 0.4 | GPU memory fraction for KV cache |

## Verification

```sh
# Check all 5 models are READY
curl -s -X POST http://localhost:8000/v2/repository/index | python3 -m json.tool

# Quick streaming test
python3 test_streaming.py

# Test with a specific pre-computed speaker
python3 test_streaming.py emily
```
