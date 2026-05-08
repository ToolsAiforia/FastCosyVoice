#!/bin/bash
# Concurrency sweep for CosyVoice3 streaming TTS via the running Triton container.
#
# Pushes the host's patched client_grpc.py + parse_stats.py into the container,
# runs client_grpc.py for each num-tasks N in TASKS_LIST against the seed_tts_cosy2
# split, captures GPU util via nvidia-smi dmon on the host, and aggregates
# per-N rtf-*.txt summaries into a single CSV + markdown report.
#
# Override via env:
#   CONTAINER     (default triton_trtllm-tts-1)
#   SERVER_PORT   (default 18001 — Triton gRPC inside the container)
#   DATASET       (default yuekai/seed_tts_cosy2)
#   SPLIT         (default test_en)
#   MAX_SAMPLES   (default 128 — items pulled from dataset before split_data)
#   WARMUP        (default 2 — warmup requests dropped per task)
#   TASKS_LIST    (default "1 2 4 8 16 32")
#   HOST_OUT_DIR  (default ./bench_sweep_results)
set -euo pipefail

CONTAINER="${CONTAINER:-triton_trtllm-tts-1}"
SERVER_PORT="${SERVER_PORT:-18001}"
DATASET="${DATASET:-yuekai/seed_tts_cosy2}"
SPLIT="${SPLIT:-test_en}"
MAX_SAMPLES="${MAX_SAMPLES:-128}"
WARMUP="${WARMUP:-2}"
TASKS_LIST="${TASKS_LIST:-1 2 4 8 16 32}"
HOST_OUT_DIR="${HOST_OUT_DIR:-./bench_sweep_results}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$HOST_OUT_DIR"
HOST_OUT_DIR="$(cd "$HOST_OUT_DIR" && pwd)"

echo "=== CosyVoice3 streaming concurrency sweep ==="
echo "  container=$CONTAINER, port=$SERVER_PORT"
echo "  dataset=$DATASET, split=$SPLIT, max_samples=$MAX_SAMPLES, warmup=$WARMUP"
echo "  tasks_list=$TASKS_LIST"
echo "  out=$HOST_OUT_DIR"
echo

if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
    echo "ERROR: container '$CONTAINER' is not running." >&2
    exit 1
fi

echo ">>> Pushing patched client_grpc.py / parse_stats.py into container"
docker cp "$SCRIPT_DIR/client_grpc.py" "$CONTAINER:/workspace/CosyVoice/runtime/triton_trtllm/client_grpc.py"
docker cp "$SCRIPT_DIR/parse_stats.py" "$CONTAINER:/workspace/CosyVoice/runtime/triton_trtllm/parse_stats.py"

SUMMARY_CSV="$HOST_OUT_DIR/summary.csv"
SUMMARY_MD="$HOST_OUT_DIR/summary.md"
echo "num_tasks,total_audio_s,processing_s,rtf,first_chunk_p50_ms,first_chunk_p95_ms,first_chunk_p99_ms,total_request_p50_ms,total_request_p95_ms,total_request_p99_ms,inter_chunk_p50_ms,inter_chunk_p95_ms,inter_chunk_p99_ms,stutter_frac" > "$SUMMARY_CSV"

extract() {
    local file="$1" key="$2" col="${3:-2}"
    grep -E "^${key}" "$file" 2>/dev/null | head -1 | awk -v c="$col" '{print $c}'
}

for N in $TASKS_LIST; do
    LOG_NAME="N${N}"
    REMOTE_LOG="/tmp/sweep_${LOG_NAME}"
    HOST_LOG="$HOST_OUT_DIR/$LOG_NAME"
    mkdir -p "$HOST_LOG"
    rm -f "$HOST_LOG"/* 2>/dev/null || true

    docker exec "$CONTAINER" bash -c "rm -rf $REMOTE_LOG && mkdir -p $REMOTE_LOG"

    GPU_CSV="$HOST_LOG/gpu_dmon.csv"
    "$SCRIPT_DIR/gpu_dmon.sh" start "$GPU_CSV"

    echo
    echo ">>> N=$N : starting client_grpc"
    set +e
    docker exec "$CONTAINER" bash -c "
        cd /workspace/CosyVoice/runtime/triton_trtllm && \
        python3 client_grpc.py \
            --server-addr 127.0.0.1 --server-port $SERVER_PORT \
            --model-name cosyvoice3 \
            --num-tasks $N --mode streaming \
            --huggingface-dataset $DATASET --split-name $SPLIT \
            --max-samples $MAX_SAMPLES --warmup-requests $WARMUP \
            --log-dir $REMOTE_LOG
    " 2>&1 | tee "$HOST_LOG/run.log"
    rc=$?
    set -e

    "$SCRIPT_DIR/gpu_dmon.sh" stop "$GPU_CSV" || true

    docker cp "$CONTAINER:$REMOTE_LOG/." "$HOST_LOG/" 2>/dev/null || true

    if [ "$rc" -ne 0 ]; then
        echo "WARN: client_grpc exited rc=$rc for N=$N"
    fi

    RTF_TXT="$HOST_LOG/rtf-${SPLIT}.txt"
    if [ -f "$RTF_TXT" ]; then
        TOTAL_AUDIO=$(extract "$RTF_TXT" 'total_duration:')
        PROC=$(extract "$RTF_TXT" 'processing time:' 3)
        RTF_VAL=$(extract "$RTF_TXT" 'RTF:')
        FC_P50=$(extract "$RTF_TXT" 'first_chunk_latency_50_percentile_ms:')
        FC_P95=$(extract "$RTF_TXT" 'first_chunk_latency_95_percentile_ms:')
        FC_P99=$(extract "$RTF_TXT" 'first_chunk_latency_99_percentile_ms:')
        TR_P50=$(extract "$RTF_TXT" 'total_request_latency_50_percentile_ms:')
        TR_P95=$(extract "$RTF_TXT" 'total_request_latency_95_percentile_ms:')
        TR_P99=$(extract "$RTF_TXT" 'total_request_latency_99_percentile_ms:')
        IC_P50=$(extract "$RTF_TXT" 'inter_chunk_interval_50_percentile_ms:')
        IC_P95=$(extract "$RTF_TXT" 'inter_chunk_interval_95_percentile_ms:')
        IC_P99=$(extract "$RTF_TXT" 'inter_chunk_interval_99_percentile_ms:')
        STUTTER=$(extract "$RTF_TXT" 'stutter_fraction')
        echo "$N,${TOTAL_AUDIO:-},${PROC:-},${RTF_VAL:-},${FC_P50:-},${FC_P95:-},${FC_P99:-},${TR_P50:-},${TR_P95:-},${TR_P99:-},${IC_P50:-},${IC_P95:-},${IC_P99:-},${STUTTER:-}" >> "$SUMMARY_CSV"
    else
        echo "WARN: rtf file missing for N=$N"
        echo "$N,,,,,,,,,,,,,," >> "$SUMMARY_CSV"
    fi
done

{
    echo "## CosyVoice3 streaming concurrency sweep"
    echo
    echo "- dataset: \`$DATASET\` split \`$SPLIT\`, max_samples=$MAX_SAMPLES, warmup=$WARMUP"
    echo "- container: \`$CONTAINER\`"
    echo
    echo "| N | total_audio (s) | processing (s) | RTF | TTFA p50 | p95 | p99 | total p50 | p95 | p99 | inter p50 | p95 | p99 | stutter>1s |"
    echo "|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"
    tail -n +2 "$SUMMARY_CSV" | awk -F, '{
        printf("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n",
            $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14)
    }'
    echo
    echo "### Per-stage breakdown"
    echo
    for d in "$HOST_OUT_DIR"/N*/; do
        f="${d}stats_summary-${SPLIT}.txt"
        [ -f "$f" ] && python3 "$SCRIPT_DIR/parse_stats.py" "$f" | sed "s|^### .*|### $(basename "$d") (stats_summary-${SPLIT}.txt)|"
        echo
    done
} > "$SUMMARY_MD"

echo
echo "=== Sweep complete ==="
echo "  CSV: $SUMMARY_CSV"
echo "  MD:  $SUMMARY_MD"
column -t -s, "$SUMMARY_CSV"
