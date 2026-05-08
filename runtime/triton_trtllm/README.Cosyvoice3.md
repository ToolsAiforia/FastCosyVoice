## CosyVoice3 with NVIDIA Triton Inference Server and TensorRT-LLM

CosyVoice3 TTS pipeline accelerated with TensorRT-LLM for the LLM and Triton Inference Server for orchestration.

### Architecture

The pipeline consists of 5 Triton models orchestrated by the BLS (Business Logic Scripting) model:

| Model | Role |
|-------|------|
| `cosyvoice3` | BLS orchestrator — routes requests through the pipeline |
| `audio_tokenizer` | Converts reference audio to speech tokens (s3tokenizer v3) |
| `speaker_embedding` | Extracts 192-dim speaker embedding (CAMPPlus) |
| `token2wav` | Converts speech tokens to mel spectrogram (flow matching DiT) |
| `vocoder` | Converts mel to waveform (HiFT) |

The LLM runs separately via `trtllm-serve` on port 8010, called by the BLS model over HTTP.

### Production Docker Image

The production image is based on `soar97/triton-cosyvoice:25.06` which includes Triton Server, TensorRT-LLM, and most Python dependencies.

**Build:**
```sh
cd runtime/triton_trtllm

docker build -f Dockerfile.cosyvoice3 -t cosyvoice3-tts:v2 .
```

**Run:**
```sh
docker run -d --name cosyvoice3 \
    --gpus '"device=0"' \
    --shm-size=1g \
    -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    cosyvoice3-tts:v2
```

**Pre-requisites in the build context:**
- `Fun-CosyVoice3-0.5B-2512/` — CosyVoice3 model files (~4.5GB)
- `cosyvoice3_llm/` — HuggingFace LLM tokenizer + weights (~1.3GB)
- `model_repo_cosyvoice3/` — Triton model repository configs

Use the download script to fetch all model files:

```sh
cd runtime/triton_trtllm
bash download_cosyvoice3_models.sh
```

The script downloads from HuggingFace:
| Directory | HuggingFace Repository |
|-----------|----------------------|
| `Fun-CosyVoice3-0.5B-2512/` | `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` + `yuekai/Fun-CosyVoice3-0.5B-2512-FP16-ONNX` |
| `cosyvoice3_llm/` | `yuekai/Fun-CosyVoice3-0.5B-2512-LLM-HF` |

All TRT engines are built automatically inside the container on first startup:
- **TRT-LLM engine** (`cosyvoice3_trt_engines/rank0.engine`) — built from `cosyvoice3_llm/` via `convert_checkpoint.py` + `trtllm-build`
- **TRT plans** (`campplus.*.trt`, `flow.decoder.estimator.*.plan`) — built from ONNX files by Triton model initialization

This ensures the engines match the target GPU architecture.

### Configuration

The entrypoint (`entrypoint_cosyvoice3.sh`) supports environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `BLS_INSTANCE_NUM` | 10 | Number of BLS model instances |
| `DECOUPLED_MODE` | True | Enable streaming (decoupled) mode |
| `LLM_PORT` | 8010 | Internal port for trtllm-serve |
| `LLM_MAX_BATCH_SIZE` | 64 | LLM max batch size |
| `LLM_KV_CACHE_FRACTION` | 0.4 | GPU memory fraction for KV cache |
| `TRITON_HTTP_PORT` | 8000 | Triton HTTP port |
| `TRITON_GRPC_PORT` | 8001 | Triton gRPC port |
| `TRITON_METRICS_PORT` | 8002 | Triton metrics port |

Example with custom settings:
```sh
docker run -d --name cosyvoice3 \
    --gpus '"device=0"' --shm-size=1g \
    -e BLS_INSTANCE_NUM=5 \
    -e LLM_KV_CACHE_FRACTION=0.6 \
    -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    cosyvoice3-tts:v2
```

### Speaker Management

#### Pre-computing speakers (offline)

Generate `spk2info.pt` with pre-computed speaker embeddings:

```sh
# Inside the container or with the model files available
python3 generate_spk2info.py \
    --model-dir ./Fun-CosyVoice3-0.5B-2512 \
    --audio ./Emily.wav \
    --reference-text "So my favorite podcast at the moment..." \
    --speaker-name emily \
    --output ./Fun-CosyVoice3-0.5B-2512/spk2info.pt

# Add more speakers (appends to existing file):
python3 generate_spk2info.py \
    --model-dir ./Fun-CosyVoice3-0.5B-2512 \
    --audio ./Bob.wav \
    --reference-text "Hello this is Bob speaking..." \
    --speaker-name bob \
    --output ./Fun-CosyVoice3-0.5B-2512/spk2info.pt
```

#### Selecting a speaker by name

Clients can select a pre-computed speaker by passing `speaker_name` as input:

```sh
# Streaming test with speaker selection
python3 test_streaming.py emily

# Benchmark client with speaker selection
python3 client_grpc.py \
    --server-addr localhost --server-port 8001 \
    --model-name cosyvoice3 \
    --mode streaming \
    --speaker-name emily \
    --target-text "Hello, this is a test of speaker selection."
```

#### Speaker resolution priority

When multiple inputs are provided, the server resolves the speaker in this order:

1. `speaker_name` — lookup pre-computed speaker from `spk2info.pt`
2. `reference_text` — match against cached speakers by transcript
3. `reference_wav` — compute speaker data from audio (zero-shot cloning)
4. Default — use the first speaker loaded from `spk2info.pt`

### API Reference

#### Inputs

| Name | Type | Dims | Optional | Description |
|------|------|------|----------|-------------|
| `target_text` | STRING | [1] | No | Text to synthesize |
| `speaker_name` | STRING | [1] | Yes | Speaker name from spk2info.pt |
| `reference_wav` | FP32 | [-1] | Yes | Reference audio (16kHz, mono) |
| `reference_wav_len` | INT32 | [1] | Yes | Length of reference audio in samples |
| `reference_text` | STRING | [1] | Yes | Transcript of reference audio |

#### Outputs

| Name | Type | Dims | Description |
|------|------|------|-------------|
| `waveform` | FP32 | [-1] | Generated audio (24kHz) |

In streaming (decoupled) mode, multiple `waveform` responses are sent as chunks.

### Testing

**Quick streaming test:**
```sh
# Default speaker
python3 test_streaming.py

# Specific speaker
python3 test_streaming.py emily
```

**Verify all models are loaded:**
```sh
curl -s -X POST http://localhost:8000/v2/repository/index | python3 -m json.tool
```

All 5 models should show `"state": "READY"`.

### Development Setup

Build the base image from scratch:
```sh
docker build . -f Dockerfile.server -t soar97/triton-cosyvoice:25.06
```

Run all stages inside the container:
```sh
bash run_cosyvoice3.sh 0 3
```

**Stages:**
- **Stage -1**: Clones the `CosyVoice` repository.
- **Stage 0**: Downloads the `Fun-CosyVoice3-0.5B-2512` model and its HuggingFace LLM checkpoint.
- **Stage 1**: Converts the HuggingFace checkpoint for the LLM to the TensorRT-LLM format and builds the TensorRT engines.
- **Stage 2**: Creates the Triton model repository, including configurations for all 5 models.
- **Stage 3**: Launches the Triton Inference Server and trtllm-serve.
- **Stage 4**: Runs the gRPC benchmark client for performance testing.
- **Stage 5**: Runs the offline TTS inference benchmark test.

### Benchmarks

The following results were obtained by decoding on a single L20 GPU.

#### Streaming TTS (Concurrent Tasks = 4)

**First Chunk Latency**

| Concurrent Tasks | Average (ms) | 50th Percentile (ms) | 90th Percentile (ms) | 95th Percentile (ms) | 99th Percentile (ms) |
| ---------------- | ------------ | -------------------- | -------------------- | -------------------- | -------------------- |
| 4                | 750.42       | 740.31               | 941.05               | 977.55               | 1002.37              |

#### Offline TTS (CosyVoice3 0.5B LLM + Token2Wav with TensorRT)

| Backend | LLM Batch Size | llm_time (s) | token2wav_time (s) | pipeline_time (s) | RTF    |
|---------|------------|--------------|--------------------|--------------------|--------|
| TRTLLM  | 1          | 13.21        | 5.72               | 19.48              | 0.1091 |
| TRTLLM  | 2          | 8.46         | 6.02               | 14.91              | 0.0822 |
| TRTLLM  | 4          | 5.07         | 5.95               | 11.43              | 0.0630 |
| TRTLLM  | 8          | 2.98         | 6.11               | 9.53               | 0.0562 |
| TRTLLM  | 16         | 2.12         | 6.27               | 8.83               | 0.0501 |
