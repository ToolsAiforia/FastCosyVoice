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

# --- Thread-pool caps (CRITICAL for pids cgroup limit) ---
# Each Triton python_backend stub otherwise spawns ~64 OpenBLAS threads; with
# BLS_INSTANCE_NUM at 10-16 the stack overruns the container's pids cgroup cap
# and tritonserver dies on pthread_create ("OpenBLAS blas_thread_init failed" ->
# SIGSEGV). Capping these drops a stub from ~132 to ~8 threads. Inherited by
# tritonserver and all backend stubs. Override per-var if the host has headroom.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-2}"

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
# Includes HiFT plans — otherwise a corrupted hift_decode_core.*.plan from
# a prior parallel-build race (pre-66a18ac) would persist on the volume
# mount, slip past Step 0.6's "skip if exists" check, and load → NaN audio.
rm -f "${MODEL_DIR}"/campplus.*.fp32.trt \
      "${MODEL_DIR}"/campplus.*.fp32.plan \
      "${MODEL_DIR}"/flow.decoder.estimator.*.plan \
      "${MODEL_DIR}"/hift_decode_core.*.plan

# --- Step 0.5: Export hift_decode_core.onnx if missing ---
# HF model repo does NOT ship this ONNX (subgraph extracted from hift.pt).
# Export takes ~5 s, idempotent (skipped if file present).
if [ ! -f "${MODEL_DIR}/hift_decode_core.onnx" ]; then
    echo "[0.5/4] Exporting hift_decode_core.onnx from hift.pt..."
    # cosyvoice package is pip-installed in site-packages — no --cosyvoice-root needed.
    python3 /workdir/scripts/export_hift_trt.py \
        --model-dir "${MODEL_DIR}"
    echo "[0.5/4] Done."
else
    echo "[0.5/4] Skipped: hift_decode_core.onnx already exists."
fi

# round-9-stable token2wav/vocoder config.pbtxt hardcode model_dir as
# /workspace/CosyVoice/runtime/triton_trtllm/Fun-CosyVoice3-0.5B-2512 (legacy from
# git-clone-bootstrap container). Symlink, чтобы model.py нашёл yaml/checkpoints.
mkdir -p /workspace/CosyVoice/runtime/triton_trtllm
ln -sfn "${MODEL_DIR}" /workspace/CosyVoice/runtime/triton_trtllm/Fun-CosyVoice3-0.5B-2512

# --- Step 0.6: Pre-build TRT plans (DiT + HiFT) single-process ---
# Without this, multiple token2wav/vocoder instances (count=6+) race when
# building plans on first start — partial writes corrupt the .plan file →
# "Serialization assertion plan.header.size == blobSize failed" → load fails.
# Idempotent: skips any plan that already exists. Takes ~3-5 min on first
# start, ~0 s on subsequent starts.
if [ ! -f "${MODEL_DIR}/flow.decoder.estimator.layer_mixed_fp16.0.plan" ] || \
   [ ! -f "${MODEL_DIR}/hift_decode_core.layer_mixed_fp32io.plan" ] || \
   [ ! -f "${MODEL_DIR}/hift_decode_core.fp32.plan" ]; then
    echo "[0.6/4] Pre-building TRT plans (single-process, ~3-5 min)..."
    python3 /workdir/scripts/prebuild_trt_plans.py --model-dir "${MODEL_DIR}"
    echo "[0.6/4] Done."
else
    echo "[0.6/4] Skipped: all TRT plans already exist."
fi

# --- Step 0.7: Bake all baked speakers into spk2info.pt ---
# Bakes /workdir/speakers/<name>.{wav,txt} (emily + spk01..spk17). emily is
# baked FIRST so it becomes the default-fallback speaker (no speaker_name).
# generate_spk2info auto-prepends the base instruction prefix
# ("You are a helpful assistant.<|endofprompt|>") — matches the BLS cache key.
#
# If speakers/ dir exists with wavs, ALWAYS rebake — the upstream HF download
# ships its own spk2info.pt with a single 'default' entry, which would otherwise
# silence our 18-speaker fleet. Wipe + bake fresh.
SPEAKERS_DIR="/workdir/speakers"
SPK2INFO_PATH="${MODEL_DIR}/spk2info.pt"
if [ -d "${SPEAKERS_DIR}" ] && ls "${SPEAKERS_DIR}"/*.wav >/dev/null 2>&1; then
    echo "[0.7/4] Baking speakers into spk2info.pt (emily + spk01..spk17)..."
    rm -f "${SPK2INFO_PATH}"   # drop any pre-baked HF default
    # emily first (-> default), then the rest sorted
    SPK_ORDER="emily $(ls "${SPEAKERS_DIR}"/*.wav 2>/dev/null | xargs -n1 basename | sed 's/\.wav$//' | grep -v '^emily$' | sort)"
    for name in ${SPK_ORDER}; do
        wav="${SPEAKERS_DIR}/${name}.wav"; txt="${SPEAKERS_DIR}/${name}.txt"
        [ -f "$wav" ] && [ -f "$txt" ] || { echo "  skip ${name} (missing wav/txt)"; continue; }
        python3 /workdir/scripts/generate_spk2info.py \
            --model-dir "${MODEL_DIR}" \
            --audio "${wav}" \
            --reference-text "$(cat ${txt})" \
            --speaker-name "${name}" \
            --output "${SPK2INFO_PATH}" >/dev/null 2>&1 && echo "  baked ${name}" || echo "  FAILED ${name}"
    done
    echo "[0.7/4] Done."
elif [ -f "${SPK2INFO_PATH}" ]; then
    echo "[0.7/4] No speakers/ dir — using existing spk2info.pt as-is."
else
    echo "[0.7/4] WARN: no spk2info.pt and no speakers/ dir — multi-speaker API won't work."
fi

# --- Step 1: Fill model_repo templates ---
echo "[1/4] Filling model repository templates..."

python3 /workdir/scripts/fill_template.py \
    -i "${MODEL_REPO_DIR}/cosyvoice3/config.pbtxt" \
    "model_dir:${MODEL_DIR},bls_instance_num:${BLS_INSTANCE_NUM},llm_tokenizer_dir:${LLM_TOKENIZER_DIR},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},decoupled_mode:${DECOUPLED_MODE},max_queue_delay_microseconds:${MAX_QUEUE_DELAY}"

# Set LLM API base URL (contains colons, can't use fill_template)
LLM_API_BASE="http://localhost:${LLM_PORT}/v1/chat/completions"
sed -i "s|LLM_API_BASE_PLACEHOLDER|${LLM_API_BASE}|g" "${MODEL_REPO_DIR}/cosyvoice3/config.pbtxt"

# round-9-stable: token2wav и vocoder config.pbtxt уже hardcoded
# (max_batch_size: 1, без placeholder'ов) — fill_template для них skip'аем.

python3 /workdir/scripts/fill_template.py \
    -i "${MODEL_REPO_DIR}/audio_tokenizer/config.pbtxt" \
    "model_dir:${MODEL_DIR},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},max_queue_delay_microseconds:${MAX_QUEUE_DELAY}"

python3 /workdir/scripts/fill_template.py \
    -i "${MODEL_REPO_DIR}/speaker_embedding/config.pbtxt" \
    "model_dir:${MODEL_DIR},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},max_queue_delay_microseconds:${MAX_QUEUE_DELAY}"

echo "[1/4] Done."

# --- Step 2: Start TensorRT-LLM inference server ---
echo "[2/4] Starting TensorRT-LLM server on port ${LLM_PORT}..."

# --extra_llm_api_options is REQUIRED — without it the runtime yaml
# (enable_block_reuse / cuda_graph_mode / enable_chunked_prefill) is ignored
# and the server runs on trtllm-serve defaults. Path resolves to the file
# baked into the image at /workdir/trtllm_runtime_options.yaml (Dockerfile COPY).
LLM_API_OPTIONS="/workdir/trtllm_runtime_options.yaml"
EXTRA_OPTS=""
if [ -f "${LLM_API_OPTIONS}" ]; then
    EXTRA_OPTS="--extra_llm_api_options ${LLM_API_OPTIONS}"
    echo "  using extra LLM api options: ${LLM_API_OPTIONS}"
else
    echo "  WARNING: ${LLM_API_OPTIONS} missing — running on trtllm-serve defaults"
fi

CUDA_VISIBLE_DEVICES=0 mpirun -np 1 --allow-run-as-root --oversubscribe \
    trtllm-serve serve \
        --tokenizer "${LLM_TOKENIZER_DIR}" \
        "${TRT_ENGINES_DIR}" \
        --max_batch_size "${LLM_MAX_BATCH_SIZE}" \
        --kv_cache_free_gpu_memory_fraction "${LLM_KV_CACHE_FRACTION}" \
        ${EXTRA_OPTS} \
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

# --- Step 4: Warm-up burst (all BLS instances + CUDA graph batch sizes) ---
# Triton model_warmup primes DiT/HiFT per instance, but the BLS LLM warm-up is
# lock-file gated (only one BLS instance warms trtllm-serve). This burst hits
# every instance via round-robin + captures CUDA graphs for batch sizes 1..N,
# so the first real user request is already hot. Non-fatal: never blocks serving.
echo "[4/4] Warm-up burst..."
WARMUP_CONC="${WARMUP_CONCURRENCY:-${BLS_INSTANCE_NUM}}"
python3 /workdir/scripts/warmup.py \
    --grpc "localhost:${TRITON_GRPC_PORT}" \
    --http "localhost:${TRITON_HTTP_PORT}" \
    --speaker "${WARMUP_SPEAKER:-emily}" \
    --concurrency "${WARMUP_CONC}" \
    --waves "${WARMUP_WAVES:-3}" || echo "  warm-up burst failed (non-fatal, continuing)"
echo "[4/4] Done — pipeline hot."

# Keep container alive - wait for either process to exit
wait -n $LLM_PID $TRITON_PID
echo "A server process exited unexpectedly. Shutting down..."
kill $LLM_PID $TRITON_PID 2>/dev/null
wait
exit 1
