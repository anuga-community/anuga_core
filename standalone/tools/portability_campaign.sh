#!/usr/bin/env bash
# Portability campaign: everything the SCA27 paper needs from one device, in
# one queue slot.  Vendor-neutral -- no nvidia-smi, no PBS, no MPI.  Run it on
# an NVIDIA, AMD or Intel node with whichever binary that node's compiler
# produced.
#
#   tools/portability_campaign.sh --label mi250x --bin bin/bench_gpu
#   tools/portability_campaign.sh --label h200 --stages phases,variants
#   tools/portability_campaign.sh --label mi300-atomics --bin bin/bench_gpu_unsafe
#
# Stages (default: all five, in this order)
#
#   gates      the correctness ladder -- lake-at-rest, cell-vs-scatter,
#              active-set-vs-full, and cross-build golden if bin/bench_cpu
#              exists.  Everything downstream is void if this fails.  [paper S4]
#   sweep      the 19-size throughput sweep, via scaling_sweep.sh.   [Fig 2]
#   phases     per-kernel breakdown at ~4M / 24M / 100M triangles,
#              parsed into one tidy CSV per device.                  [E1, Fig 3]
#   variants   flux cell|scatter x order row|morton at a fixed size:
#              the gather-vs-scatter A/B the AMD case study turns on. [E3, Fig 7]
#   ceiling    walk up until the device runs out of memory, then run the
#              largest size that fits as the hero dam break.         [E6, Fig 8]
#
# Results land in build/<label>/ as CSVs meant to be concatenated across
# devices -- every row carries the label, so `cat build/*/phases.csv` is the
# cross-vendor figure's input directly.
#
# Options
#   --label NAME    device tag, becomes build/<label>/ and the csv column
#   --bin PATH      benchmark binary            (default bin/bench_gpu)
#   --stages LIST   comma-separated subset of the five stages
#   --sizes "..."   override the sweep sizes
#   --quick         short version for smoke-testing the script itself
#   --steps N       timed steps per measurement (default 20; 50 for phases)
set -u

LABEL=""
BIN=bin/bench_gpu
STAGES=gates,sweep,phases,variants,ceiling
SIZES=""
STEPS=20
QUICK=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --label)  LABEL=$2;  shift 2 ;;
        --bin)    BIN=$2;    shift 2 ;;
        --stages) STAGES=$2; shift 2 ;;
        --sizes)  SIZES=$2;  shift 2 ;;
        --steps)  STEPS=$2;  shift 2 ;;
        --quick)  QUICK=1;   shift ;;
        -h|--help) sed -n '2,45p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

[[ -n "$LABEL" ]] || { echo "--label is required (e.g. --label mi250x)" >&2; exit 2; }
[[ -x "$BIN"   ]] || { echo "no such binary: $BIN (make gpu / amdgpu / intelgpu first)" >&2; exit 2; }

OUT="build/$LABEL"
mkdir -p "$OUT"
has_stage() { [[ ",$STAGES," == *",$1,"* ]]; }

# ---- sizes -----------------------------------------------------------------
# nx -> 4*nx*nx triangles.  The three phase points are ~4M / 24M / 100M, the
# sizes the existing NVIDIA numbers were taken at, so the new rows line up
# against the banked ones.
PHASE_NX=(1000 2450 5000)
VARIANT_NX=2450
if [[ $QUICK -eq 1 ]]; then
    PHASE_NX=(200 300)
    VARIANT_NX=300
    STEPS=5
    [[ -n "$SIZES" ]] || SIZES="100 200 300"
fi

log()  { printf '\n=== %s ===\n' "$*" | tee -a "$OUT/campaign.log"; }
note() { printf '%s\n' "$*" | tee -a "$OUT/campaign.log"; }

: > "$OUT/campaign.log"
note "campaign  : $LABEL"
note "binary    : $BIN"
note "stages    : $STAGES"
note "started   : $(date -Is)"
note "host      : $(hostname)"

# ============================================================================
# gates -- the correctness ladder.  Records PASS/FAIL per rung; a FAIL here is
# reported and the campaign continues (the timings are still worth having, but
# the paper cannot use them until the rung is green).
# ============================================================================
gate_fails=0
gate() {  # gate <name> <cmd...>
    local name=$1; shift
    local out rc
    out=$("$@" 2>&1); rc=$?
    { echo "--- $name"; echo "\$ $*"; echo "$out"; echo; } >> "$OUT/gates.txt"
    if [[ $rc -eq 0 ]]; then
        printf '  %-34s PASS\n' "$name" | tee -a "$OUT/campaign.log"
    else
        printf '  %-34s FAIL (rc=%d)\n' "$name" "$rc" | tee -a "$OUT/campaign.log"
        gate_fails=$((gate_fails + 1))
    fi
    echo "$out"
}

# The active set is bit-exact against full stepping on the GPU builds, but the
# host build's compaction reorders a reduction and lands at ~5e-16 relative.
# Both are correct results and the paper wants to report which one this device
# gave, so: try bitwise first, fall back to the roundoff gate, and record the
# distinction instead of failing the campaign over it.
gate_exact() {  # gate_exact <name> <cmd...>   -- cmd must end before tolerances
    local name=$1; shift
    local out rc
    out=$("$@" --atol 0 --rtol 0 2>&1); rc=$?
    { echo "--- $name (bitwise attempt)"; echo "$out"; echo; } >> "$OUT/gates.txt"
    if [[ $rc -eq 0 ]]; then
        printf '  %-34s PASS (bit-exact)\n' "$name" | tee -a "$OUT/campaign.log"
        echo "$LABEL,$name,bit-exact" >> "$OUT/gates.csv"
        return 0
    fi
    out=$("$@" 2>&1); rc=$?   # default atol 1e-10 / rtol 1e-8
    { echo "--- $name (roundoff fallback)"; echo "$out"; echo; } >> "$OUT/gates.txt"
    local worst
    worst=$(echo "$out" | awk '$2 ~ /^[0-9.e+-]+$/ && NF >= 4 {print $3}' | sort -g | tail -1)
    if [[ $rc -eq 0 ]]; then
        printf '  %-34s PASS (roundoff, %s)\n' "$name" "${worst:-?}" | tee -a "$OUT/campaign.log"
        echo "$LABEL,$name,roundoff ${worst:-?}" >> "$OUT/gates.csv"
    else
        printf '  %-34s FAIL\n' "$name" | tee -a "$OUT/campaign.log"
        echo "$LABEL,$name,FAIL" >> "$OUT/gates.csv"
        gate_fails=$((gate_fails + 1))
    fi
}

if has_stage gates; then
    log "gates"
    : > "$OUT/gates.txt"
    echo "label,gate,result" > "$OUT/gates.csv"
    G_NX=${PHASE_NX[0]}
    [[ $QUICK -eq 1 ]] && G_NX=200

    # 1. well-balanced: still water over bumps must hold max|momentum| at roundoff.
    lake_out=$(gate "lake-at-rest" \
        "$BIN" --nx 300 --ny 300 --steps 50 --case lake --scheme ader2 --flux scatter)
    lake_mom=$(echo "$lake_out" | awk '/max \|momentum\|/ {print $NF}' | tail -1)
    note "    max|momentum| = ${lake_mom:-n/a}   (want <= 1e-10)"

    # 2. the two flux algorithms must agree to roundoff.  This is also the
    #    gate that has to be re-run after any AMD atomics change, since
    #    relaxed fp64 atomics reorder the scatter's summation.
    "$BIN" --nx 300 --ny 300 --steps 30 --case dambumps --scheme ader2 \
           --flux cell --save "$OUT/.gate_cell.bin" > /dev/null 2>&1
    flux_out=$(gate "scatter == cell (ftol 1e-6)" \
        "$BIN" --nx 300 --ny 300 --steps 30 --case dambumps --scheme ader2 \
               --flux scatter --check "$OUT/.gate_cell.bin" --ftol 1e-6)
    flux_worst=$(echo "$flux_out" | awk '$2 ~ /^[0-9.e+-]+$/ && NF >= 4 {print $3}' \
                 | sort -g | tail -1)
    note "    worst field diff rel to scale = ${flux_worst:-n/a}   (order-of-summation only)"

    # 3. the active set must be bit-exact against full stepping.
    "$BIN" --nx 300 --ny 300 --steps 30 --case river --scheme ader2 \
           --flux scatter --save "$OUT/.gate_full.bin" > /dev/null 2>&1
    gate_exact "active-set == full" \
        "$BIN" --nx 300 --ny 300 --steps 30 --case river --scheme ader2 \
               --flux scatter --active-set --check "$OUT/.gate_full.bin"

    # 4. cross-build: this device against the host CPU reference, if one is built.
    if [[ -x bin/bench_cpu ]]; then
        bin/bench_cpu --nx 200 --ny 200 --steps 30 --case dambumps --scheme ader2 \
                      --flux cell --save "$OUT/.gate_cpu.bin" > /dev/null 2>&1
        gate "device == cpu golden" \
            "$BIN" --nx 200 --ny 200 --steps 30 --case dambumps --scheme ader2 \
                   --flux cell --check "$OUT/.gate_cpu.bin" > /dev/null
    else
        note "  (bin/bench_cpu absent -- skipping the cross-build golden)"
    fi

    rm -f "$OUT"/.gate_*.bin
    note "  gate failures: $gate_fails"
fi

# ============================================================================
# sweep -- 19 sizes, the throughput-vs-size figure.  Delegates to the existing
# sweep driver so the CSV schema stays identical to the banked NVIDIA runs.
# ============================================================================
if has_stage sweep; then
    log "sweep"
    # shellcheck disable=SC2086
    tools/scaling_sweep.sh --bin "$BIN" --steps "$STEPS" \
        --csv "$OUT/sweep.csv" $SIZES -- --scheme ader2 --flux scatter \
        2>&1 | tee -a "$OUT/campaign.log"
fi

# ============================================================================
# phases -- E1.  Parses the --phases block into label,nx,triangles,kernel,
# ms_per_step,pct so the five devices concatenate into one stacked-bar figure.
# ============================================================================
parse_phases() {  # parse_phases <label> <nx> <text>
    local lbl=$1 nx=$2 txt=$3
    local tris=$((4 * nx * nx))
    # the block is "    <name>   <ms> ms   <pct>%" until the first blank line
    echo "$txt" | awk -v lbl="$lbl" -v nx="$nx" -v tris="$tris" '
        /per-kernel breakdown/ { inblk = 1; next }
        inblk && NF == 0       { inblk = 0 }
        inblk && $1 == "sum" { inblk = 0; next }
        inblk && $2 ~ /^[0-9.]+$/ && $3 == "ms" {
            pct = $4; sub(/%$/, "", pct)
            printf "%s,%d,%d,%s,%s,%s\n", lbl, nx, tris, $1, $2, pct
        }'
}

if has_stage phases; then
    log "phases"
    echo "label,nx,triangles,kernel,ms_per_step,pct" > "$OUT/phases.csv"
    P_STEPS=$(( QUICK == 1 ? 5 : 50 ))
    for nx in "${PHASE_NX[@]}"; do
        note "  nx=$nx ($((4 * nx * nx)) triangles)"
        out=$("$BIN" --nx "$nx" --ny "$nx" --steps "$P_STEPS" --warmup 5 \
                     --case dam --scheme ader2 --flux scatter --phases 2>&1)
        rc=$?
        { echo "=== phases nx=$nx (rc=$rc) ==="; echo "$out"; echo; } >> "$OUT/phases.log"
        if [[ $rc -ne 0 ]]; then
            note "    FAILED (rc=$rc) -- see $OUT/phases.log"
            continue
        fi
        parse_phases "$LABEL" "$nx" "$out" >> "$OUT/phases.csv"
        echo "$out" | sed -n '/per-kernel breakdown/,/^$/p' | tee -a "$OUT/campaign.log"
    done
fi

# ============================================================================
# variants -- E3/E5.  flux x order at one size.  On AMD this is the whole case
# study: `cell` touches no atomics, `scatter` does six fp64 atomic adds per
# edge, so the cell/scatter ratio here is the direct read on atomic cost.
# ============================================================================
if has_stage variants; then
    log "variants"
    echo "label,nx,triangles,flux,order,ms_per_step,mcellsteps_per_s,gflops,max_momentum" \
        > "$OUT/variants.csv"
    tris=$((4 * VARIANT_NX * VARIANT_NX))
    printf '  %-9s %-8s %12s %12s\n' flux order ms/step Mcell/s | tee -a "$OUT/campaign.log"
    for flux in cell scatter; do
        for order in row morton; do
            ROW="$OUT/.variant_row.csv"; rm -f "$ROW"
            out=$("$BIN" --nx "$VARIANT_NX" --ny "$VARIANT_NX" --steps "$STEPS" \
                         --warmup 5 --case dam --scheme ader2 \
                         --flux "$flux" --order "$order" --csv "$ROW" 2>&1)
            rc=$?
            { echo "=== flux=$flux order=$order (rc=$rc) ==="; echo "$out"; echo; } \
                >> "$OUT/variants.log"
            if [[ $rc -ne 0 ]]; then
                printf '  %-9s %-8s %12s %12s  FAILED\n' "$flux" "$order" - - \
                    | tee -a "$OUT/campaign.log"
                continue
            fi
            # read the machine-readable row rather than re-parsing the report
            read -r ms mcell gflops mom < <(tail -1 "$ROW" | awk -F, '{print $6, $7, $8, $14}')
            echo "$LABEL,$VARIANT_NX,$tris,$flux,$order,$ms,$mcell,$gflops,$mom" \
                >> "$OUT/variants.csv"
            printf '  %-9s %-8s %12.3f %12.1f\n' "$flux" "$order" "$ms" "$mcell" \
                | tee -a "$OUT/campaign.log"
        done
    done
    rm -f "$OUT/.variant_row.csv"
fi

# ============================================================================
# ceiling -- E6.  Doubling walk to the first failure, bisect to the largest
# size that fits, then run that as the hero dam break.  Confirms the
# 512 B/triangle model on this device and produces the closing figure.
# ============================================================================
if has_stage ceiling; then
    log "ceiling"
    fits() {  # fits <nx>
        "$BIN" --nx "$1" --ny "$1" --steps 2 --warmup 0 --case dam \
               --scheme ader2 --flux scatter > /dev/null 2>&1
    }
    lo=1000; hi=0
    probe=1000
    walk_max=20000
    [[ $QUICK -eq 1 ]] && probe=100 && lo=100 && walk_max=400
    while [[ $probe -le $walk_max ]]; do
        if fits "$probe"; then
            note "  nx=$probe ($((4 * probe * probe)) tris) fits"
            lo=$probe
            probe=$((probe * 2))
        else
            note "  nx=$probe ($((4 * probe * probe)) tris) does not fit"
            hi=$probe
            break
        fi
    done

    if [[ $hi -gt 0 ]]; then
        while [[ $((hi - lo)) -gt 50 ]]; do
            mid=$(((lo + hi) / 2))
            if fits "$mid"; then lo=$mid; else hi=$mid; fi
        done
    fi
    note "  largest nx that fits: $lo -> $((4 * lo * lo)) triangles"
    note "  512 B/tri predicts $(awk -v t=$((4 * lo * lo)) 'BEGIN{printf "%.1f", t*512/1073741824}') GiB"

    note "  hero run at nx=$lo"
    "$BIN" --nx "$lo" --ny "$lo" --steps "$STEPS" --warmup 5 --case dam \
           --scheme ader2 --flux scatter --order morton --phases \
           --csv "$OUT/hero.csv" 2>&1 | tee "$OUT/hero.txt" | tee -a "$OUT/campaign.log"
    echo "label,nx,triangles,kernel,ms_per_step,pct" > "$OUT/hero_phases.csv"
    parse_phases "$LABEL" "$lo" "$(cat "$OUT/hero.txt")" >> "$OUT/hero_phases.csv"
fi

# ============================================================================
log "manifest"
{
    echo "label     : $LABEL"
    echo "binary    : $BIN"
    echo "host      : $(hostname)"
    echo "finished  : $(date -Is)"
    echo "gates     : $gate_fails failure(s)"
    echo
    echo "files:"
    ls -la "$OUT" | sed 's/^/  /'
} | tee "$OUT/MANIFEST.txt"

note ""
note "collect across devices with:"
note "  head -1 build/*/phases.csv | head -1 > build/phases_all.csv"
note "  tail -q -n +2 build/*/phases.csv >> build/phases_all.csv"

exit $(( gate_fails > 0 ? 1 : 0 ))
