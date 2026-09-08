#!/usr/bin/env bash
# Sweep the simulator across a few rates and ring sizes.
#
# Correctness gate: every run must report zero packet-id gaps and must observe
# exactly as many packets as the producer published.

set -euo pipefail

BIN="${1:-./build/sim/gnp}"
if [[ ! -x "$BIN" ]]; then
    echo "usage: $0 [path-to-gnp]   (built binary not found at '$BIN')" >&2
    exit 1
fi

fail=0

run() {
    local desc="$1"; shift
    local out
    out="$("$BIN" "$@" 2>&1)"

    local published observed gaps
    published=$(awk '/descriptors published/ {print $3}' <<<"$out")
    observed=$(awk  '/packets observed/      {print $3}' <<<"$out")
    gaps=$(awk      '/packet-id gaps/        {print $3}' <<<"$out")
    mean=$(awk      '/^  mean/               {print $2}' <<<"$out")

    if [[ "$published" == "$observed" && "$gaps" == "0" ]]; then
        printf 'ok    %-34s %10s pkts  mean %8s us\n' "$desc" "$observed" "$mean"
    else
        printf 'FAIL  %-34s published=%s observed=%s gaps=%s\n' \
               "$desc" "$published" "$observed" "$gaps"
        fail=1
    fi
}

run "100 kpps, ring 1024"    --pps 100000  --duration 1000
run "1 Mpps, ring 1024"      --pps 1000000 --duration 1000
run "unpaced, ring 1024"     --pps 0       --duration 1000
run "unpaced, ring 64"       --pps 0       --duration 1000 --ring 64
run "bursty, 16 per burst"   --pps 500000  --duration 1000 --burst 16
run "fixed 200k packets"     --pps 0       --duration 3000 --packets 200000
run "jumbo payload 9000 B"   --pps 200000  --duration 1000 --size 9000
run "near-idle, 1 pps"       --pps 1       --duration 500

echo
if (( fail )); then
    echo "sweep FAILED"
    exit 1
fi
echo "sweep passed"
