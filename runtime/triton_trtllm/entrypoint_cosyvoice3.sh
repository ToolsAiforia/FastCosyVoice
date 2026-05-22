#!/bin/bash
set -e

MODEL_DIR="/workdir/Fun-CosyVoice3-0.5B-2512"
LLM_TOKENIZER_DIR="/workdir/cosyvoice3_llm"
TRT_ENGINES_DIR="/workdir/cosyvoice3_trt_engines"
MODEL_REPO_DIR="/model_repo"
TRT_DTYPE="${TRT_DTYPE:-bfloat16}"

# --- Configuration (override via environment variables) ---
BLS_INSTANCE_NUM="${BLS_INSTANCE_NUM:-10}"
DECOUPLED_MODE="${DECOUPLED_MODE:-True}"
TRITON_MAX_BATCH_SIZE="${TRITON_MAX_BATCH_SIZE:-1}"
MAX_QUEUE_DELAY="${MAX_QUEUE_DELAY:-0}"
LLM_MAX_BATCH_SIZE="${LLM_MAX_BATCH_SIZE:-64}"
LLM_KV_CACHE_FRACTION="${LLM_KV_CACHE_FRACTION:-0.4}"

TRITON_HTTP_PORT="${TRITON_HTTP_PORT:-8000}"
TRITON_GRPC_PORT="${TRITON_GRPC_PORT:-8001}"
TRITON_METRICS_PORT="${TRITON_METRICS_PORT:-8002}"
LLM_PORT="${LLM_PORT:-8010}"

echo "============================================"
echo "  CosyVoice3 TTS Server"
echo "============================================"
echo "  Triton gRPC port:  ${TRITON_GRPC_PORT}"
echo "  Triton HTTP port:  ${TRITON_HTTP_PORT}"
echo "  LLM API port:      ${LLM_PORT}"
echo "  Decoupled mode:    ${DECOUPLED_MODE}"
echo "  BLS instances:     ${BLS_INSTANCE_NUM}"
echo "============================================"

# --- Step 0: Build TRT-LLM engine if not present ---
if [ ! -f "${TRT_ENGINES_DIR}/rank0.engine" ]; then
    echo "[0/4] Building TRT-LLM engine (${TRT_DTYPE})..."
    TRT_WEIGHTS_DIR="$(mktemp -d)"

    python3 /workdir/scripts/convert_checkpoint.py \
        --model_dir "${LLM_TOKENIZER_DIR}" \
        --output_dir "${TRT_WEIGHTS_DIR}" \
        --dtype "${TRT_DTYPE}"

    mkdir -p "${TRT_ENGINES_DIR}"
    trtllm-build \
        --checkpoint_dir "${TRT_WEIGHTS_DIR}" \
        --output_dir "${TRT_ENGINES_DIR}" \
        --max_batch_size "${LLM_MAX_BATCH_SIZE}" \
        --max_num_tokens 32768 \
        --gemm_plugin "${TRT_DTYPE}"

    rm -rf "${TRT_WEIGHTS_DIR}"
    echo "[0/4] Done: TRT-LLM engine built."
else
    echo "[0/4] Skipped: TRT-LLM engine already exists."
fi

# Remove pre-built TRT plans so they rebuild for this GPU.
# Includes HiFT plans (built per-GPU sm tag, not portable across GPUs).
rm -f "${MODEL_DIR}"/campplus.*.fp32.trt \
      "${MODEL_DIR}"/campplus.*.fp32.plan \
      "${MODEL_DIR}"/flow.decoder.estimator.*.plan \
      "${MODEL_DIR}"/hift_decode_core.fp32.plan \
      "${MODEL_DIR}"/hift_decode_core.fp32_B8.plan

# --- Step 1: Fill model_repo templates ---
echo "[1/4] Filling model repository templates..."

python3 /workdir/scripts/fill_template.py \
    -i "${MODEL_REPO_DIR}/cosyvoice3/config.pbtxt" \
    "model_dir:${MODEL_DIR},bls_instance_num:${BLS_INSTANCE_NUM},llm_tokenizer_dir:${LLM_TOKENIZER_DIR},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},decoupled_mode:${DECOUPLED_MODE},max_queue_delay_microseconds:${MAX_QUEUE_DELAY}"

# Set LLM API base URL (contains colons, can't use fill_template)
LLM_API_BASE="http://localhost:${LLM_PORT}/v1/chat/completions"
sed -i "s|LLM_API_BASE_PLACEHOLDER|${LLM_API_BASE}|g" "${MODEL_REPO_DIR}/cosyvoice3/config.pbtxt"

python3 /workdir/scripts/fill_template.py \
    -i "${MODEL_REPO_DIR}/token2wav/config.pbtxt" \
    "model_dir:${MODEL_DIR},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},max_queue_delay_microseconds:${MAX_QUEUE_DELAY}"

python3 /workdir/scripts/fill_template.py \
    -i "${MODEL_REPO_DIR}/vocoder/config.pbtxt" \
    "model_dir:${MODEL_DIR},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},max_queue_delay_microseconds:${MAX_QUEUE_DELAY}"

python3 /workdir/scripts/fill_template.py \
    -i "${MODEL_REPO_DIR}/audio_tokenizer/config.pbtxt" \
    "model_dir:${MODEL_DIR},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},max_queue_delay_microseconds:${MAX_QUEUE_DELAY}"

python3 /workdir/scripts/fill_template.py \
    -i "${MODEL_REPO_DIR}/speaker_embedding/config.pbtxt" \
    "model_dir:${MODEL_DIR},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},max_queue_delay_microseconds:${MAX_QUEUE_DELAY}"

echo "[1/4] Done."

# --- Step 1.5: Build HiFT B-dynamic TRT plan (per-GPU, ~3-4 min cold) ---
HIFT_B8_PLAN="${MODEL_DIR}/hift_decode_core.fp32_B8.plan"
if [ ! -f "${HIFT_B8_PLAN}" ] || [ ! -s "${HIFT_B8_PLAN}" ]; then
    echo "[1.5/4] Building HiFT B-dynamic TRT plan (this takes ~3-4 minutes on cold start)..."
    python3 /workdir/scripts/build_hift_batched_trt.py --model-dir "${MODEL_DIR}"
    echo "[1.5/4] Done."
else
    echo "[1.5/4] Skipped: HiFT B-dynamic plan already exists."
fi

# --- Step 1.6: Auto-bake spk2info.pt for default 'ref' speaker ---
# Allows docker `up` to deliver a working stack out-of-box (test_streaming.py ref works).
SPK_INFO="${MODEL_DIR}/spk2info.pt"
REF_WAV="/workdir/reference.wav"
REF_TEXT_FILE="/workdir/reference_text.txt"
if [ ! -f "${SPK_INFO}" ] && [ -f "${REF_WAV}" ] && [ -f "${REF_TEXT_FILE}" ]; then
    echo "[1.6/4] Baking spk2info.pt with default 'ref' speaker..."
    python3 /workdir/generate_spk2info.py \
        --model-dir "${MODEL_DIR}" \
        --audio "${REF_WAV}" \
        --reference-text "$(cat ${REF_TEXT_FILE})" \
        --speaker-name ref \
        --output "${SPK_INFO}"
    echo "[1.6/4] Done."
elif [ -f "${SPK_INFO}" ]; then
    echo "[1.6/4] Skipped: spk2info.pt already exists."
else
    echo "[1.6/4] Skipped: no reference.wav bundled (zero-shot path only)."
fi

# --- Step 2: Start TensorRT-LLM inference server ---
echo "[2/4] Starting TensorRT-LLM server on port ${LLM_PORT}..."

CUDA_VISIBLE_DEVICES=0 mpirun -np 1 --allow-run-as-root --oversubscribe \
    trtllm-serve serve \
        --tokenizer "${LLM_TOKENIZER_DIR}" \
        "${TRT_ENGINES_DIR}" \
        --max_batch_size "${LLM_MAX_BATCH_SIZE}" \
        --kv_cache_free_gpu_memory_fraction "${LLM_KV_CACHE_FRACTION}" \
        --port "${LLM_PORT}" &
LLM_PID=$!

# Wait for LLM server to be ready
echo "Waiting for LLM server to start..."
for i in $(seq 1 120); do
    if curl -s "http://localhost:${LLM_PORT}/health" > /dev/null 2>&1; then
        echo "LLM server is ready (took ${i}s)"
        break
    fi
    if ! kill -0 $LLM_PID 2>/dev/null; then
        echo "ERROR: LLM server process died"
        exit 1
    fi
    sleep 1
done

# --- Step 3: Start Triton Inference Server ---
echo "[3/4] Starting Triton Inference Server..."

CUDA_VISIBLE_DEVICES=0 tritonserver \
    --model-repository "${MODEL_REPO_DIR}" \
    --http-port "${TRITON_HTTP_PORT}" \
    --grpc-port "${TRITON_GRPC_PORT}" \
    --metrics-port "${TRITON_METRICS_PORT}" &
TRITON_PID=$!

# Wait for Triton to be ready
echo "Waiting for Triton server to start..."
for i in $(seq 1 120); do
    if curl -s "http://localhost:${TRITON_HTTP_PORT}/v2/health/ready" > /dev/null 2>&1; then
        echo "Triton server is ready (took ${i}s)"
        break
    fi
    if ! kill -0 $TRITON_PID 2>/dev/null; then
        echo "ERROR: Triton server process died"
        exit 1
    fi
    sleep 1
done

echo "============================================"
echo "  CosyVoice3 TTS Server is running!"
echo "  gRPC: localhost:${TRITON_GRPC_PORT}"
echo "  HTTP: localhost:${TRITON_HTTP_PORT}"
echo "============================================"

# Keep container alive - wait for either process to exit
wait -n $LLM_PID $TRITON_PID
echo "A server process exited unexpectedly. Shutting down..."
kill $LLM_PID $TRITON_PID 2>/dev/null
wait
exit 1
