#!/usr/bin/env bash
# Scaling sweep: run the miniapp at increasing mesh sizes until it fails, and
# record throughput plus the real device footprint at each size.
#
#   tools/scaling_sweep.sh [--bin bin/bench_gpu] [--steps 20] [--case dam]
#                          [--csv build/scaling.csv] [nx1 nx2 ...] [-- extra...]
#
# Everything after `--` is passed to the benchmark verbatim, e.g.
#   tools/scaling_sweep.sh 1000 2000 -- --scheme ader2 --flux scatter
#
# Device memory is sampled from nvidia-smi while each run is in flight, because
# ANUGA's own gpu_query_device_memory() only reports real numbers when the
# kernels are built with -DUSE_CUDA, and no ANUGA build sets it.
set -u

BIN=bin/bench_gpu
STEPS=20
WARMUP=5
CASE=dam
CSV=build/scaling.csv
SIZES=()
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bin)    BIN=$2;    shift 2 ;;
        --steps)  STEPS=$2;  shift 2 ;;
        --warmup) WARMUP=$2; shift 2 ;;
        --case)   CASE=$2;   shift 2 ;;
        --csv)    CSV=$2;    shift 2 ;;
        -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
        --)       shift; EXTRA=("$@"); break ;;
        *)        SIZES+=("$1"); shift ;;
    esac
done

if [[ ${#SIZES[@]} -eq 0 ]]; then
    SIZES=(100 200 300 500 700 1000 1300 1600 2000 2400 2800 3200 3600 4000 4400 4800)
fi

WORK=$(dirname "$CSV")
mkdir -p "$WORK"
rm -f "$CSV"
LOG="${CSV%.csv}.log"
: > "$LOG"
PEAKFILE="$WORK/.devmem.$$"

printf '%8s %12s %10s %10s %12s %10s %s\n' \
       nx triangles ms/step Mcell/s GFLOP/s devGiB status

for nx in "${SIZES[@]}"; do
    tris=$((4 * nx * nx))

    # Sample device memory every 200 ms for the duration of the run.
    : > "$PEAKFILE"
    ( while :; do
          nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
              >> "$PEAKFILE"
          sleep 0.05
      done ) &
    poller=$!

    out=$("$BIN" --nx "$nx" --ny "$nx" --steps "$STEPS" --warmup "$WARMUP" \
                 --case "$CASE" --csv "$CSV" "${EXTRA[@]}" 2>&1)
    rc=$?

    kill "$poller" 2>/dev/null; wait "$poller" 2>/dev/null
    peak=$(sort -n "$PEAKFILE" | tail -1)
    [[ -z "$peak" ]] && peak=0
    devgib=$(awk -v m="$peak" 'BEGIN{printf "%.2f", m/1024}')

    { echo "=== nx=$nx (rc=$rc, peak device MiB=$peak) ==="; echo "$out"; echo; } >> "$LOG"

    if [[ $rc -ne 0 ]]; then
        reason=$(echo "$out" | grep -Ei "fail|error|out of memory|cannot|abort" | head -1)
        printf '%8s %12s %10s %10s %12s %10s %s\n' \
               "$nx" "$tris" - - - "$devgib" "FAILED (rc=$rc) ${reason:-see $LOG}"
        echo
        echo "stopped at nx=$nx ($tris triangles); full output in $LOG"
        break
    fi

    read -r ms mcell gflops < <(tail -1 "$CSV" | awk -F, '{print $6, $7, $8}')
    printf '%8s %12s %10.3f %10.1f %12.1f %10.2f %s\n' \
           "$nx" "$tris" "$ms" "$mcell" "$gflops" "$devgib" ok
done

rm -f "$PEAKFILE"
echo
echo "csv: $CSV   log: $LOG"
