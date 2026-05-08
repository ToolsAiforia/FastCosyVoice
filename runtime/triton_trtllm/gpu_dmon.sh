#!/bin/bash
# Stand-alone nvidia-smi dmon launcher for benchmark profiling.
# Usage:
#   ./gpu_dmon.sh start <out.csv>     -> writes PID to out.csv.pid
#   ./gpu_dmon.sh stop  <out.csv>     -> kills the recorded PID
set -euo pipefail

cmd="${1:-}"
out="${2:-}"
if [ -z "$cmd" ] || [ -z "$out" ]; then
    echo "usage: $0 {start|stop} <out.csv>" >&2
    exit 2
fi

pid_file="${out}.pid"

case "$cmd" in
    start)
        : > "$out"
        nvidia-smi dmon -s pucm -d 1 -o T > "$out" 2>&1 &
        echo $! > "$pid_file"
        echo "dmon started pid=$(cat "$pid_file") -> $out"
        ;;
    stop)
        if [ -f "$pid_file" ]; then
            pid=$(cat "$pid_file")
            kill "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
            rm -f "$pid_file"
            echo "dmon stopped"
        else
            echo "no pid file at $pid_file" >&2
        fi
        ;;
    *)
        echo "unknown cmd: $cmd" >&2
        exit 2
        ;;
esac
