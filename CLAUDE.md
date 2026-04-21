# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FastCosyVoice is an optimized implementation of CosyVoice3 (Fun-CosyVoice3-0.5B), an LLM-based multilingual text-to-speech system. It adds parallel pipeline processing, TensorRT acceleration, FP16 support, and streaming on top of the original FunAudioLLM/CosyVoice.

## Build & Development Commands

```sh
# Install dependencies (uses uv as package manager)
uv sync

# Install with dev dependencies (includes tensorrt-llm, ruff, flake8, mypy)
uv sync --group dev

# Run example scripts
uv run python run_basic.py      # Basic streaming inference
uv run python run_fast.py       # FP16 + TRT Flow + TRT-LLM
uv run python run_offline.py    # Offline mode for long texts
uv run python run_instruct.py   # Instruction-based generation
uv run python benchmark_llm.py  # LLM performance benchmarking

# Linting (CI workflow exists but is currently commented out)
uv run ruff check .
uv run flake8 --max-line-length 180 --exclude ./third_party/,./runtime/python/grpc/cosyvoice_pb2*py
uv run mypy .

# Web UI
uv run python webui.py
```

## Architecture

### Two-Package Structure

- **`cosyvoice/`** - Core TTS components from original FunAudioLLM/CosyVoice:
  - `cli/` - High-level interfaces (`CosyVoice3`, `CosyVoice2`, `AutoModel`), frontend (tokenization, speaker embedding), and model classes with TRT support
  - `llm/` - Qwen2-0.5B based language model for speech token generation
  - `flow/` - Flow matching acoustic model (mel-spectrogram generation)
  - `hifigan/` - HiFi-GAN neural vocoder (mel-to-audio)
  - `tokenizer/` - Speech tokenizer (semantic tokens)
  - `transformer/` - Transformer building blocks
  - `utils/` - File I/O, ONNX/TRT conversion, common utilities
  - `vllm/` - vLLM integration for LLM inference

- **`fastcosyvoice/`** - Optimized wrapper adding parallel pipeline:
  - `cosyvoice.py` - `FastCosyVoice3` class: main entry point, handles model loading, TRT conversion, and provides `inference_streaming()`/`inference_offline()` methods
  - `model.py` - `FastCosyVoice3Model`: parallel pipeline with threaded LLM/Flow/Hift stages
  - `frontend.py` - Re-exports `CosyVoiceFrontEnd` from `cosyvoice.cli`

### Parallel Pipeline (the key optimization)

```
[LLM Thread + dedicated CUDA stream]
  -> token_queue (25-token chunks) ->
[Flow+Hift Thread]
  -> audio_queue ->
[Main Thread: yields audio chunks]
```

LLM runs in its own thread with a dedicated CUDA stream. Flow+Hift run in a separate thread so their blocking operations (TRT sync, CPU f0_predictor) never stall LLM generation. This achieves 80-90% of isolated LLM throughput.

### Acceleration Layers

- **TensorRT Flow**: ONNX export then TRT conversion (~2.5x speedup). Controlled by `load_trt=True`.
- **TensorRT-LLM**: LLM acceleration (~3x speedup). Controlled by `load_trt_llm=True`. Engine build artifacts cached per GPU SM version (e.g., `sm86/`, `sm120/`).
- **FP16**: Half-precision for LLM module. Controlled by `fp16=True`.

### Runtime / Deployment

- `runtime/python/fastapi/` - FastAPI REST server
- `runtime/python/grpc/` - gRPC server with protobuf definitions
- `runtime/triton_trtllm/` - Triton Inference Server with TensorRT-LLM backend, Docker Compose configs for CosyVoice2 and CosyVoice3

## Key Technical Details

- **Python**: >=3.11, <3.13
- **CUDA**: PyTorch uses cu128 wheels; TRT-LLM from NVIDIA PyPI index
- **Token chunk size**: 25 tokens per chunk (must match training `static_chunk_size`), producing 50 mel frames (2:1 ratio)
- **TRT dynamic shapes**: Min 4, Opt 1000, Max 6000 frames (FP16) or 3000 (FP32)
- **Model download**: Uses ModelScope (`snapshot_download`) or HuggingFace Hub
- **Config system**: Hydra/HyperPyYAML for model configs (e.g., `cosyvoice3.yaml`)
- **Stress marks**: Russian stress via `+` before stressed letter (auto-converted to combining accent). `auto_stress` parameter uses silero-stress but should only be used sparingly.
- **Languages**: Chinese (18+ dialects), English, Japanese, Korean, German, Spanish, French, Italian, Russian
