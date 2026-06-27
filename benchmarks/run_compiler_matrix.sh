#!/usr/bin/env bash
# run_compiler_matrix.sh — build ANUGA with GCC / ICX / NVHPC-CPU and run
# per-kernel + fast-suite benchmarks for each, tagging outputs by compiler.
#
# Usage:
#   bash benchmarks/run_compiler_matrix.sh [--compilers gcc,icx,nvhpc] [--skip-build] [--skip-tests]
#
# Prerequisites (HPC/NCI environment):
#   conda envs:  anuga_phase0_gcc   (Python 3.13, deps; CC overridden per run)
#                anuga_phase0_icx   (Python 3.13, Intel deps + MKL)
#   Modules available: gcc/11.2.0, hpc_sdk/nvhpc/24.11
#   ICX:  ~/intel/oneapi/compiler/2025.1/bin/icx  (source setvars.sh first)
#
# Outputs (benchmarks/results/):
#   kernels_matrix_gcc_<commit>_<ts>.json
#   kernels_matrix_icx_<commit>_<ts>.json
#   kernels_matrix_nvhpc_cpu_<commit>_<ts>.json
#   fastsuite_<compiler>_<commit>_<ts>.txt
#
# Compare two compiler runs afterwards:
#   python benchmarks/compare_kernel_benchmarks.py \
#       benchmarks/results/kernels_matrix_gcc_*.json \
#       benchmarks/results/kernels_matrix_icx_*.json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RESULTS_DIR="${REPO_ROOT}/benchmarks/results"
mkdir -p "${RESULTS_DIR}"

COMPILERS="gcc,icx,nvhpc"
SKIP_BUILD=0
SKIP_TESTS=0
KERNEL_ARGS="--size medium --modes 1,2 --reps 50"

for arg in "$@"; do
    case "$arg" in
        --compilers=*) COMPILERS="${arg#*=}" ;;
        --skip-build)  SKIP_BUILD=1 ;;
        --skip-tests)  SKIP_TESTS=1 ;;
    esac
done

COMMIT="$(git -C "${REPO_ROOT}" rev-parse --short HEAD)"
BRANCH="$(git -C "${REPO_ROOT}" rev-parse --abbrev-ref HEAD)"
TS="$(date +%Y%m%d_%H%M%S)"

log() { echo "[matrix] $*" >&2; }

# ---------------------------------------------------------------------------
# Compiler configurations
# ---------------------------------------------------------------------------

build_gcc() {
    log "=== GCC build ==="
    module load gcc/11.2.0 2>/dev/null || log "gcc module not available, using PATH gcc"
    source ~/miniforge3/etc/profile.d/conda.sh && conda activate anuga_phase0_gcc
    CC=gcc pip install --no-build-isolation -q -e "${REPO_ROOT}" \
        -Cbuild-dir=build/cp313 >/dev/null 2>&1
    log "GCC build done"
}

run_gcc() {
    log "=== GCC benchmarks ==="
    source ~/miniforge3/etc/profile.d/conda.sh && conda activate anuga_phase0_gcc
    OUT="${RESULTS_DIR}/kernels_matrix_gcc_${COMMIT}_${TS}.json"
    OMP_NUM_THREADS=4 python "${REPO_ROOT}/benchmarks/run_kernel_benchmarks.py" \
        ${KERNEL_ARGS} --output "${OUT}"
    log "Kernel results: ${OUT}"
    if [[ "${SKIP_TESTS}" -eq 0 ]]; then
        TESTOUT="${RESULTS_DIR}/fastsuite_gcc_${COMMIT}_${TS}.txt"
        pytest --pyargs anuga --run-fast -q 2>&1 | tail -5 | tee "${TESTOUT}"
        log "Fast-suite: ${TESTOUT}"
    fi
}

build_icx() {
    log "=== ICX build ==="
    # Source Intel oneAPI environment (adjusts PATH/LD_LIBRARY_PATH for icx)
    SETVARS="${HOME}/intel/oneapi/setvars.sh"
    if [[ -f "${SETVARS}" ]]; then
        # shellcheck disable=SC1090
        source "${SETVARS}" --force >/dev/null 2>&1
    else
        log "WARNING: ${SETVARS} not found — relying on icx already in PATH"
    fi
    source ~/miniforge3/etc/profile.d/conda.sh && conda activate anuga_phase0_icx
    CC=icx pip install --no-build-isolation -q -e "${REPO_ROOT}" \
        -Cbuild-dir=build/cp313-icx >/dev/null 2>&1
    log "ICX build done"
}

run_icx() {
    log "=== ICX benchmarks ==="
    SETVARS="${HOME}/intel/oneapi/setvars.sh"
    [[ -f "${SETVARS}" ]] && source "${SETVARS}" --force >/dev/null 2>&1
    source ~/miniforge3/etc/profile.d/conda.sh && conda activate anuga_phase0_icx
    OUT="${RESULTS_DIR}/kernels_matrix_icx_${COMMIT}_${TS}.json"
    OMP_NUM_THREADS=4 python "${REPO_ROOT}/benchmarks/run_kernel_benchmarks.py" \
        ${KERNEL_ARGS} --output "${OUT}"
    log "Kernel results: ${OUT}"
    if [[ "${SKIP_TESTS}" -eq 0 ]]; then
        TESTOUT="${RESULTS_DIR}/fastsuite_icx_${COMMIT}_${TS}.txt"
        pytest --pyargs anuga --run-fast -q 2>&1 | tail -5 | tee "${TESTOUT}"
        log "Fast-suite: ${TESTOUT}"
    fi
}

build_nvhpc() {
    log "=== NVHPC-CPU build ==="
    module load hpc_sdk/nvhpc/24.11 2>/dev/null || log "nvhpc module not available"
    source ~/miniforge3/etc/profile.d/conda.sh && conda activate anuga_phase0_gcc
    CC=nvc pip install --no-build-isolation -q -e "${REPO_ROOT}" \
        -Cbuild-dir=build/cp313-nvc-cpu >/dev/null 2>&1
    log "NVHPC build done"
}

run_nvhpc() {
    log "=== NVHPC-CPU benchmarks ==="
    module load hpc_sdk/nvhpc/24.11 2>/dev/null || true
    source ~/miniforge3/etc/profile.d/conda.sh && conda activate anuga_phase0_gcc
    OUT="${RESULTS_DIR}/kernels_matrix_nvhpc_cpu_${COMMIT}_${TS}.json"
    OMP_NUM_THREADS=4 python "${REPO_ROOT}/benchmarks/run_kernel_benchmarks.py" \
        ${KERNEL_ARGS} --output "${OUT}"
    log "Kernel results: ${OUT}"
    if [[ "${SKIP_TESTS}" -eq 0 ]]; then
        TESTOUT="${RESULTS_DIR}/fastsuite_nvhpc_${COMMIT}_${TS}.txt"
        pytest --pyargs anuga --run-fast -q 2>&1 | tail -5 | tee "${TESTOUT}"
        log "Fast-suite: ${TESTOUT}"
    fi
}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

IFS=',' read -ra COMPILER_LIST <<< "${COMPILERS}"

for compiler in "${COMPILER_LIST[@]}"; do
    case "${compiler}" in
        gcc)
            [[ "${SKIP_BUILD}" -eq 0 ]] && build_gcc
            run_gcc
            ;;
        icx)
            [[ "${SKIP_BUILD}" -eq 0 ]] && build_icx
            run_icx
            ;;
        nvhpc)
            [[ "${SKIP_BUILD}" -eq 0 ]] && build_nvhpc
            run_nvhpc
            ;;
        *)
            log "Unknown compiler: ${compiler} (choices: gcc, icx, nvhpc)"
            exit 1
            ;;
    esac
done

log ""
log "All done. Compare results with:"
log "  python benchmarks/compare_kernel_benchmarks.py \\"
log "    ${RESULTS_DIR}/kernels_matrix_gcc_${COMMIT}_${TS}.json \\"
log "    ${RESULTS_DIR}/kernels_matrix_icx_${COMMIT}_${TS}.json"
