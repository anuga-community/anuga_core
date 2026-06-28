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
#   Modules available: gcc/11.2.0
#   ICX:  ~/intel/oneapi/compiler/2025.1/bin/icx  (source setvars.sh first)
#   NVHPC: loaded via spack (see SPACK_SETUP / NVHPC_SPACK_HASH below)
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

# -u (nounset) is intentionally omitted: Intel setvars.sh and conda reference
# unbound variables in clean subshells, which -u would turn into fatal errors.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RESULTS_DIR="${REPO_ROOT}/benchmarks/results"
mkdir -p "${RESULTS_DIR}"

SPACK_SETUP="${HOME}/source/samir/spack/share/spack/setup-env.sh"
NVHPC_SPACK_HASH="4jo2stp"  # nvhpc@26.3

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
    # GCC 11 has OpenMP 4.5 only; gpu_culvert_operator.c requires OpenMP 5.0
    # (#pragma omp target teams loop).  Building only sw_domain_openmp_ext via
    # ninja avoids triggering the gpu_ext target.  The build dir must already
    # be configured (from a prior pip-install / Phase 0 setup).
    ninja -C "${REPO_ROOT}/build/cp313" \
        anuga/shallow_water/sw_domain_openmp_ext.cpython-313-x86_64-linux-gnu.so \
        || { log "ERROR: GCC ninja build failed"; exit 1; }
    log "GCC build done (openmp_ext only; gpu_ext skipped — GCC 11 lacks OMP 5.0)"
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
    SETVARS="${HOME}/intel/oneapi/setvars.sh"
    if [[ -f "${SETVARS}" ]]; then
        # shellcheck disable=SC1090
        source "${SETVARS}" --force >/dev/null 2>&1 || true
    else
        log "WARNING: ${SETVARS} not found — relying on icx already in PATH"
    fi
    source ~/miniforge3/etc/profile.d/conda.sh && conda activate anuga_phase0_icx
    # Build via ninja — pip install -Cbuild-dir= can fail on older pip versions.
    # ICX can compile all targets (no OMP-version restriction like GCC 11).
    ninja -C "${REPO_ROOT}/build/cp313-icx" \
        || { log "ERROR: ICX ninja build failed"; exit 1; }
    log "ICX build done"
}

run_icx() {
    log "=== ICX benchmarks ==="
    SETVARS="${HOME}/intel/oneapi/setvars.sh"
    [[ -f "${SETVARS}" ]] && source "${SETVARS}" --force >/dev/null 2>&1 || true
    source ~/miniforge3/etc/profile.d/conda.sh && conda activate anuga_phase0_icx
    export CC=icx  # ensure benchmark output is tagged with the right compiler
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

_gcc_loader() {
    # Path to the anuga_phase0_gcc editable-install loader file.
    source ~/miniforge3/etc/profile.d/conda.sh && conda activate anuga_phase0_gcc
    python -c "import site; print(site.getsitepackages()[0])"
}

_load_spack_nvhpc() {
    if [[ -f "${SPACK_SETUP}" ]]; then
        # shellcheck disable=SC1090
        source "${SPACK_SETUP}"
        spack load nvhpc /"${NVHPC_SPACK_HASH}" 2>/dev/null || {
            log "WARNING: spack load nvhpc /${NVHPC_SPACK_HASH} failed — relying on nvc already in PATH"
        }
    else
        log "WARNING: ${SPACK_SETUP} not found — relying on nvc already in PATH"
    fi
}

build_nvhpc() {
    log "=== NVHPC-CPU build (spack nvhpc@26.3 / ${NVHPC_SPACK_HASH}) ==="
    _load_spack_nvhpc
    source ~/miniforge3/etc/profile.d/conda.sh && conda activate anuga_phase0_gcc
    # If the build dir was configured with a different nvc version, reconfigure it so
    # the new compiler's flags and version string are recorded in build.ninja.
    _cfg_ver=$(python3 -c "
import json, sys
try:
    d = json.load(open('${REPO_ROOT}/build/cp313-nvc-cpu/meson-info/intro-compilers.json'))
    print(d['host']['c']['version'])
except Exception: print('')
" 2>/dev/null)
    _cur_ver=$(nvc --version 2>&1 | grep -oP '\d+\.\d+-\d+' | head -1)
    if [[ -d "${REPO_ROOT}/build/cp313-nvc-cpu" && -n "${_cfg_ver}" && "${_cfg_ver}" != "${_cur_ver}" ]]; then
        log "nvc version changed (${_cfg_ver} → ${_cur_ver}): reconfiguring build dir..."
        CC=nvc meson setup --reconfigure "${REPO_ROOT}/build/cp313-nvc-cpu" \
            -Dgpu_offload=false \
            || { log "ERROR: NVHPC meson reconfigure failed"; exit 1; }
    fi
    # Build via ninja (pip install -Cbuild-dir= can be unreliable on older pip).
    # NVHPC can compile all targets (no OMP-version restriction).
    ninja -C "${REPO_ROOT}/build/cp313-nvc-cpu" \
        || { log "ERROR: NVHPC ninja build failed"; exit 1; }
    # Switch anuga_phase0_gcc editable pointer from build/cp313 (gcc) to the nvc dir.
    _loader="$(_gcc_loader)/_anuga_editable_loader.py"
    sed -i "s|/build/cp313'|/build/cp313-nvc-cpu'|g" "${_loader}"
    log "NVHPC build done (nvc ${_cur_ver}); editable loader → build/cp313-nvc-cpu"
}

run_nvhpc() {
    log "=== NVHPC-CPU benchmarks ==="
    _load_spack_nvhpc
    source ~/miniforge3/etc/profile.d/conda.sh && conda activate anuga_phase0_gcc
    export CC=nvc  # ensure benchmark output is tagged with the right compiler
    OUT="${RESULTS_DIR}/kernels_matrix_nvhpc_cpu_${COMMIT}_${TS}.json"
    OMP_NUM_THREADS=4 python "${REPO_ROOT}/benchmarks/run_kernel_benchmarks.py" \
        ${KERNEL_ARGS} --output "${OUT}"
    log "Kernel results: ${OUT}"
    if [[ "${SKIP_TESTS}" -eq 0 ]]; then
        TESTOUT="${RESULTS_DIR}/fastsuite_nvhpc_${COMMIT}_${TS}.txt"
        pytest --pyargs anuga --run-fast -q 2>&1 | tail -5 | tee "${TESTOUT}"
        log "Fast-suite: ${TESTOUT}"
    fi
    # Restore anuga_phase0_gcc editable pointer back to the gcc build dir.
    _loader="$(_gcc_loader)/_anuga_editable_loader.py"
    sed -i "s|/build/cp313-nvc-cpu'|/build/cp313'|g" "${_loader}"
    log "Editable loader restored → build/cp313"
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
