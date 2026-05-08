#!/bin/bash
# Download CosyVoice3 model files from HuggingFace.
#
# TRT-LLM engine and TRT plans are built automatically inside the
# container at first startup (entrypoint_cosyvoice3.sh).
#
# Usage:
#   bash download_cosyvoice3_models.sh
#
# Requirements:
#   - huggingface-cli (pip install huggingface_hub)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_DIR="$SCRIPT_DIR/Fun-CosyVoice3-0.5B-2512"
LLM_DIR="$SCRIPT_DIR/cosyvoice3_llm"

# --- Step 1: Download Fun-CosyVoice3-0.5B-2512 ---
echo "=== [1/2] Downloading Fun-CosyVoice3-0.5B-2512 (model weights + ONNX) ==="

huggingface-cli download --local-dir "$MODEL_DIR" \
    FunAudioLLM/Fun-CosyVoice3-0.5B-2512

huggingface-cli download --local-dir "$MODEL_DIR" \
    yuekai/Fun-CosyVoice3-0.5B-2512-FP16-ONNX

echo "[1/2] Done: $MODEL_DIR"

# --- Step 2: Download cosyvoice3_llm (HuggingFace LLM for trtllm-serve) ---
echo "=== [2/2] Downloading cosyvoice3_llm (LLM tokenizer + weights) ==="

huggingface-cli download --local-dir "$LLM_DIR" \
    yuekai/Fun-CosyVoice3-0.5B-2512-LLM-HF

echo "[2/2] Done: $LLM_DIR"

echo ""
echo "============================================"
echo "  All models downloaded successfully!"
echo "============================================"
echo "  $MODEL_DIR"
echo "  $LLM_DIR"
echo ""
echo "Next steps:"
echo "  1. Build Docker image:"
echo "     docker build -f Dockerfile.cosyvoice3 -t cosyvoice3-tts:v2 ."
echo "  2. Run container (TRT engines build on first start):"
echo "     docker run -d --gpus '\"device=0\"' --shm-size=1g \\"
echo "         -p 8000:8000 -p 8001:8001 -p 8002:8002 \\"
echo "         cosyvoice3-tts:v2"
echo "============================================"
