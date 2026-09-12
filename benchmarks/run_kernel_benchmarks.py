#!/usr/bin/env python3
"""
ANUGA Per-Kernel Benchmark Suite
---------------------------------
Times individual hot kernels in isolation (compute_fluxes, extrapolate,
distribute, protect, manning friction) rather than a whole evolve loop, so
that a regression in one kernel is attributable instead of hiding inside an
aggregate cells/s figure. Implements the "Per-kernel microbenchmarks" TODO
in claude/C_EXTENSION_AUDIT_TODOS.md (P3), and is the prerequisite for the
cross-compiler tuning loop in claude/PLAN_compiler_tuning.md (Phase 1).

Each kernel is called repeatedly on a *fixed* domain state (the domain is
primed with a short evolve to reach a non-trivial wet state, then the timed
loop calls the kernel directly without advancing time) — this isolates the
kernel's own cost from the timestep controller and keeps every repetition's
input identical.

Usage
-----
    # Default: mode 1 (legacy openmp_ext) and mode 2 (unified gpu_ext, CPU
    # multicore unless built with gpu_offload=true), medium mesh
    python benchmarks/run_kernel_benchmarks.py

    # Just mode 2, small mesh, more repetitions
    python benchmarks/run_kernel_benchmarks.py --modes 2 --size small --reps 100

    # Compare against a saved baseline
    python benchmarks/compare_benchmarks.py results/kernels_a.json results/kernels_b.json

Metrics
-------
- mean_us / median_us / std_us : per-call wall time in microseconds
- cells_per_s                  : n_triangles / mean_time_s — comparable across
                                  mesh sizes for the same kernel
"""

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime


# ---------------------------------------------------------------------------
# Mesh sizes (triangle counts, not the small/medium/large evolve scenarios in
# run_benchmarks.py — kernels are timed per-call, not per-evolve, so a single
# representative size per run is enough; --size still lets you compare kernel
# cost scaling).
# ---------------------------------------------------------------------------

SIZES = {
    'small':  dict(nx=50,  ny=50),
    'medium': dict(nx=150, ny=150),
    'large':  dict(nx=300, ny=300),
}

# Short prime: enough CFL steps to reach a non-trivial wet/dry mixed state
# (dam-break IC) without spending benchmark time evolving.
PRIME_FINALTIME = 5.0
PRIME_YIELDSTEP = 5.0


def _create_domain(nx, ny, mode, tmpdir):
    import numpy as np
    import anuga
    from anuga import rectangular_cross_domain, Reflective_boundary

    domain = rectangular_cross_domain(nx, ny, len1=1000.0, len2=1000.0)
    domain.set_flow_algorithm('DE0')
    domain.set_low_froude(0)
    domain.set_name('kernel_bench')
    domain.set_datadir(tmpdir)
    domain.store = False

    domain.set_quantity('elevation', 0.0)
    domain.set_quantity('stage', lambda x, y: np.where(x < 500.0, 2.0, 0.5))
    domain.set_quantity('xmomentum', 0.0)
    domain.set_quantity('ymomentum', 0.0)
    domain.set_boundary({t: Reflective_boundary(domain)
                         for t in domain.get_boundary_tags()})

    domain.set_multiprocessor_mode(mode)

    # Prime: reach a mixed wet/dry, non-trivial-velocity state so kernels
    # exercise their real branches (not a degenerate all-dry / all-still IC).
    for _ in domain.evolve(yieldstep=PRIME_YIELDSTEP, finaltime=PRIME_FINALTIME):
        pass

    return domain


# ---------------------------------------------------------------------------
# Kernel registry
#
# Each entry: callable(domain) -> None, plus which multiprocessor_mode(s)
# it is meaningful to benchmark under.
#
# extrapolate_edge_only and manning_friction_flat currently always dispatch
# to sw_domain_openmp_ext regardless of domain.multiprocessor_mode (see
# friction.manning_friction_semi_implicit and the direct pyx import below) —
# they are still useful per-compiler timings of that extension, just not a
# mode-2-vs-mode-1 comparison. Marked 'always_openmp_ext': True so the report
# can say so instead of implying a mode-2 kernel was exercised.
# ---------------------------------------------------------------------------

def _compute_fluxes(domain):
    domain.compute_fluxes()


def _protect(domain):
    domain.protect_against_infinitesimal_and_negative_heights()


def _distribute(domain):
    domain.distribute_to_vertices_and_edges()


def _extrapolate_edge_only(domain):
    from anuga.shallow_water.sw_domain_openmp_ext import extrapolate_second_order_edge_sw
    extrapolate_second_order_edge_sw(domain, distribute_to_vertices=False)


def _manning_friction_flat(domain):
    from anuga.shallow_water.friction import manning_friction_semi_implicit
    manning_friction_semi_implicit(domain)


KERNELS = {
    'compute_fluxes':          dict(fn=_compute_fluxes,          modes=(1, 2), always_openmp_ext=False),
    'protect':                 dict(fn=_protect,                 modes=(1, 2), always_openmp_ext=False),
    'distribute':              dict(fn=_distribute,              modes=(1, 2), always_openmp_ext=False),
    'extrapolate_edge_only':   dict(fn=_extrapolate_edge_only,   modes=(1,),   always_openmp_ext=True),
    'manning_friction_flat':   dict(fn=_manning_friction_flat,   modes=(1,),   always_openmp_ext=True),
}


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def _time_kernel(fn, domain, reps, warmup):
    for _ in range(warmup):
        fn(domain)

    samples_s = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(domain)
        samples_s.append(time.perf_counter() - t0)

    mean_s = statistics.mean(samples_s)
    return {
        'mean_us':   round(mean_s * 1e6, 2),
        'median_us': round(statistics.median(samples_s) * 1e6, 2),
        'std_us':    round((statistics.stdev(samples_s) if len(samples_s) > 1 else 0.0) * 1e6, 2),
        'min_us':    round(min(samples_s) * 1e6, 2),
        'max_us':    round(max(samples_s) * 1e6, 2),
        'n_reps':    reps,
    }


def run_one(kernel_name, size, mode, reps, warmup):
    cfg = SIZES[size]
    tmpdir = tempfile.mkdtemp()
    try:
        domain = _create_domain(cfg['nx'], cfg['ny'], mode, tmpdir)
        n_tris = domain.number_of_triangles
        fn = KERNELS[kernel_name]['fn']
        timing = _time_kernel(fn, domain, reps, warmup)
        cells_per_s = (n_tris / (timing['mean_us'] * 1e-6)) if timing['mean_us'] > 0 else 0.0
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    return {
        'kernel': kernel_name,
        'size': size,
        'n_triangles': n_tris,
        'mode': mode,
        'always_openmp_ext': KERNELS[kernel_name]['always_openmp_ext'],
        'cells_per_s': round(cells_per_s, 0),
        **timing,
    }


# ---------------------------------------------------------------------------
# Metadata helpers (mirrors run_benchmarks.py for compare_benchmarks.py reuse)
# ---------------------------------------------------------------------------

def _git_info():
    def _run(args):
        try:
            return subprocess.check_output(
                args, cwd=os.path.dirname(__file__),
                stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return 'unknown'

    return {
        'commit': _run(['git', 'rev-parse', '--short', 'HEAD']),
        'branch': _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD']),
        'commit_long': _run(['git', 'rev-parse', 'HEAD']),
    }


def _env_info():
    omp = os.environ.get('OMP_NUM_THREADS', 'unset')
    try:
        import anuga
        anuga_version = anuga.__version__
    except Exception:
        anuga_version = 'unknown'

    cc = os.environ.get('CC', 'unset')

    return {
        'python_version': sys.version.split()[0],
        'platform': platform.system(),
        'hostname': platform.node().split('.')[0],
        'omp_num_threads_env': omp,
        'anuga_version': anuga_version,
        'cc_env': cc,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='ANUGA per-kernel benchmark suite.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--size', default='medium', choices=list(SIZES),
                         help='Mesh size (default: medium)')
    parser.add_argument('--modes', default='1,2',
                         help='Comma-separated multiprocessor_modes to test (default: 1,2)')
    parser.add_argument('--kernels', default=','.join(KERNELS),
                         help=f'Comma-separated kernel names (default: all = {",".join(KERNELS)})')
    parser.add_argument('--reps', type=int, default=50,
                         help='Timed repetitions per kernel (default: 50)')
    parser.add_argument('--warmup', type=int, default=5,
                         help='Untimed warmup calls per kernel (default: 5)')
    parser.add_argument('--output', default=None,
                         help='Output JSON path. Default: benchmarks/results/kernels_<branch>_<commit>_<timestamp>.json')
    args = parser.parse_args()

    modes = [int(m.strip()) for m in args.modes.split(',')]
    kernel_names = [k.strip() for k in args.kernels.split(',')]
    for k in kernel_names:
        if k not in KERNELS:
            parser.error(f'Unknown kernel {k!r}. Choose from: {list(KERNELS)}')

    git = _git_info()
    env = _env_info()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    omp_threads = int(os.environ.get('OMP_NUM_THREADS', 1))

    if args.output:
        outpath = args.output
    else:
        outdir = os.path.join(os.path.dirname(__file__), 'results')
        os.makedirs(outdir, exist_ok=True)
        fname = f"kernels_{git['branch'].replace('/', '_')}_{git['commit']}_{timestamp}.json"
        outpath = os.path.join(outdir, fname)

    print(f'ANUGA kernel benchmark  commit={git["commit"]}  branch={git["branch"]}  CC={env["cc_env"]}')
    print(f'Python {env["python_version"]}  OMP_NUM_THREADS={omp_threads}  size={args.size}  modes={modes}')
    print(f'Output: {outpath}')
    print()

    results = []
    header = f"{'Kernel':<24} {'mode':>4}  {'tris':>8}  {'mean(us)':>10}  {'median(us)':>11}  {'std(us)':>9}  {'cells/s':>12}"
    rule = '-' * len(header)
    print(header)
    print(rule)

    for mode in modes:
        for kname in kernel_names:
            kinfo = KERNELS[kname]
            if mode not in kinfo['modes']:
                continue
            label = f'{kname}{" [openmp_ext always]" if kinfo["always_openmp_ext"] else ""}'
            sys.stdout.write(f'  Running {label} mode={mode} ... ')
            sys.stdout.flush()
            try:
                rec = run_one(kname, args.size, mode, args.reps, args.warmup)
                results.append(rec)
                print(
                    f"\r  {kname:<24} {rec['mode']:>4}  {rec['n_triangles']:>8}  "
                    f"{rec['mean_us']:>10.2f}  {rec['median_us']:>11.2f}  "
                    f"{rec['std_us']:>9.2f}  {rec['cells_per_s']:>12,.0f}"
                )
            except Exception as exc:
                print(f'\r  {kname:<24} mode={mode} FAILED: {exc}')

    print(rule)
    print()

    payload = {
        'timestamp': timestamp,
        'git': git,
        'env': env,
        'omp_threads': omp_threads,
        'size': args.size,
        'results': results,
    }
    with open(outpath, 'w') as fh:
        json.dump(payload, fh, indent=2)
    print(f'Saved {len(results)} results -> {outpath}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
