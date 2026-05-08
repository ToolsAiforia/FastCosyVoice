#!/usr/bin/env python3
"""Parse a Triton stats_summary-*.txt produced by client_grpc.py into a
per-stage markdown table. Usage: parse_stats.py <stats_summary.txt> [...]"""
import re
import sys
from pathlib import Path


HEADER_RE = re.compile(r"^model name is\s+(\S+)")
TIMES_RE = re.compile(
    r"queue time\s+([\d.]+)\s+s,\s+compute infer time\s+([\d.]+)\s+s,\s+"
    r"compute input time\s+([\d.]+)\s+s,\s+compute output time\s+([\d.]+)\s+s"
)
BATCH_RE = re.compile(
    r"execuate inference with batch_size\s+(\d+)\s+total\s+(\d+)\s+times,\s+"
    r"total_infer_time\s+([\d.]+)\s+ms.*?=\s*([\d.]+)\s+ms"
)


def parse(path: Path):
    rows = []
    current = None
    for line in path.read_text().splitlines():
        m = HEADER_RE.match(line)
        if m:
            if current is not None:
                rows.append(current)
            current = {"model": m.group(1)}
            continue
        if current is None:
            continue
        m = TIMES_RE.search(line)
        if m:
            current["queue_s"] = float(m.group(1))
            current["infer_s"] = float(m.group(2))
            current["input_s"] = float(m.group(3))
            current["output_s"] = float(m.group(4))
            continue
        m = BATCH_RE.search(line)
        if m:
            current.setdefault("batch_size", int(m.group(1)))
            current["count"] = current.get("count", 0) + int(m.group(2))
            current["total_infer_ms"] = current.get("total_infer_ms", 0.0) + float(m.group(3))
            current["avg_infer_ms"] = float(m.group(4))
    if current is not None:
        rows.append(current)
    return rows


def render_md(rows, source_label):
    lines = [f"### {source_label}", ""]
    lines.append(
        "| Model | Count | Avg infer (ms) | Total infer (s) | Queue (s) | Input (s) | Output (s) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    sums = {"infer_s": 0.0, "queue_s": 0.0}
    for r in rows:
        lines.append(
            "| {model} | {count} | {avg:.2f} | {infer:.2f} | {queue:.2f} | {inp:.2f} | {out:.2f} |".format(
                model=r["model"],
                count=r.get("count", 0),
                avg=r.get("avg_infer_ms", 0.0),
                infer=r.get("infer_s", 0.0),
                queue=r.get("queue_s", 0.0),
                inp=r.get("input_s", 0.0),
                out=r.get("output_s", 0.0),
            )
        )
        sums["infer_s"] += r.get("infer_s", 0.0)
        sums["queue_s"] += r.get("queue_s", 0.0)
    lines.append("")
    lines.append(
        f"_Sum across stages: infer={sums['infer_s']:.2f}s, queue={sums['queue_s']:.2f}s_"
    )
    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        print("usage: parse_stats.py <stats_summary.txt> [...]", file=sys.stderr)
        sys.exit(2)
    out = []
    for p in sys.argv[1:]:
        path = Path(p)
        rows = parse(path)
        out.append(render_md(rows, path.name))
    print("\n\n".join(out))


if __name__ == "__main__":
    main()
