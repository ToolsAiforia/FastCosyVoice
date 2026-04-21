#!/bin/bash
# Download CosyVoice3 model files and build TRT-LLM engine.
#
# Usage:
#   bash download_cosyvoice3_models.sh              # download all + build engine
#   bash download_cosyvoice3_models.sh --skip-build  # download only (no trtllm-build)
#
# Requirements:
#   - huggingface-cli (pip install huggingface_hub)
#   - trtllm-build (only for engine build, available in soar97/triton-cosyvoice:25.06)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

SKIP_BUILD=false
for arg in "$@"; do
    case "$arg" in
        --skip-build) SKIP_BUILD=true ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

MODEL_DIR="$SCRIPT_DIR/Fun-CosyVoice3-0.5B-2512"
LLM_DIR="$SCRIPT_DIR/cosyvoice3_llm"
TRT_ENGINES_DIR="$SCRIPT_DIR/cosyvoice3_trt_engines"

# --- Step 1: Download Fun-CosyVoice3-0.5B-2512 ---
echo "=== [1/3] Downloading Fun-CosyVoice3-0.5B-2512 (model weights + ONNX) ==="

huggingface-cli download --local-dir "$MODEL_DIR" \
    FunAudioLLM/Fun-CosyVoice3-0.5B-2512

huggingface-cli download --local-dir "$MODEL_DIR" \
    yuekai/Fun-CosyVoice3-0.5B-2512-FP16-ONNX

echo "[1/3] Done: $MODEL_DIR"

# --- Step 2: Download cosyvoice3_llm (HuggingFace LLM for trtllm-serve) ---
echo "=== [2/3] Downloading cosyvoice3_llm (LLM tokenizer + weights) ==="

huggingface-cli download --local-dir "$LLM_DIR" \
    yuekai/Fun-CosyVoice3-0.5B-2512-LLM-HF

echo "[2/3] Done: $LLM_DIR"

# --- Step 3: Build TRT-LLM engine ---
if [ "$SKIP_BUILD" = true ]; then
    echo "=== [3/3] Skipped (--skip-build) ==="
    echo ""
    echo "To build TRT-LLM engines manually (must run on target GPU):"
    echo "  TRT_DTYPE=bfloat16"
    echo "  python3 scripts/convert_checkpoint.py \\"
    echo "      --model_dir $LLM_DIR \\"
    echo "      --output_dir /tmp/trt_weights \\"
    echo "      --dtype \$TRT_DTYPE"
    echo ""
    echo "  trtllm-build \\"
    echo "      --checkpoint_dir /tmp/trt_weights \\"
    echo "      --output_dir $TRT_ENGINES_DIR \\"
    echo "      --max_batch_size 64 \\"
    echo "      --max_num_tokens 32768 \\"
    echo "      --gemm_plugin \$TRT_DTYPE"
else
    echo "=== [3/3] Building TRT-LLM engine (bfloat16) ==="

    TRT_DTYPE=bfloat16
    TRT_WEIGHTS_DIR="$(mktemp -d)"

    echo "Converting checkpoint to TensorRT format..."
    python3 scripts/convert_checkpoint.py \
        --model_dir "$LLM_DIR" \
        --output_dir "$TRT_WEIGHTS_DIR" \
        --dtype "$TRT_DTYPE"

    echo "Building TensorRT engine..."
    mkdir -p "$TRT_ENGINES_DIR"
    trtllm-build \
        --checkpoint_dir "$TRT_WEIGHTS_DIR" \
        --output_dir "$TRT_ENGINES_DIR" \
        --max_batch_size 64 \
        --max_num_tokens 32768 \
        --gemm_plugin "$TRT_DTYPE"

    rm -rf "$TRT_WEIGHTS_DIR"
    echo "[3/3] Done: $TRT_ENGINES_DIR"
fi

echo ""
echo "============================================"
echo "  All models downloaded successfully!"
echo "============================================"
echo "  $MODEL_DIR"
echo "  $LLM_DIR"
echo "  $TRT_ENGINES_DIR"
echo ""
echo "Next steps:"
echo "  1. Build Docker image:"
echo "     docker build -f Dockerfile.cosyvoice3 -t cosyvoice3-tts:v2 ."
echo "  2. Run container:"
echo "     docker run -d --gpus '\"device=0\"' --shm-size=1g \\"
echo "         -p 8000:8000 -p 8001:8001 -p 8002:8002 \\"
echo "         cosyvoice3-tts:v2"
echo "============================================"
