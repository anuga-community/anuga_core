#!/usr/bin/env python3
"""Benchmark: HLLC vs Central-Upwind Riemann solver

Compares performance and accuracy of the HLLC Riemann solver against
ANUGA's default central-upwind scheme on three test cases:

  1. Single-edge flux throughput  — measures raw flux call throughput
  2. 1D dam-break accuracy        — compares final stage profiles against the
                                    Ritter analytical solution
  3. Full-domain timing           — wall-time per timestep for both solvers
     on a fine mesh

Usage::

    python benchmarks/benchmark_hllc.py

Optional arguments::

    --nx N         Grid divisions per side (default 40); total ~N*N*2 triangles
    --finaltime T  Simulation end time in seconds (default 0.5)
    --nflux N      Number of single-edge flux calls in throughput test (default 1e6)
    --threads N    OMP_NUM_THREADS (default: use current env value)

Notes
-----
* Both solvers use identical meshes, quantities, boundaries and time-steps
  so wall-time differences reflect only the flux computation.
* The analytical Ritter solution for a dam-break on a dry bed is used as
  the accuracy reference.  Neither solver is expected to match it exactly on
  a 2-D mesh, but HLLC typically resolves the bore front more crisply.
"""

import sys
import os
import time
import argparse
import numpy as np
from math import sqrt

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--nx', type=int, default=40,
                   help='Grid divisions per side (default 40)')
    p.add_argument('--finaltime', type=float, default=0.5,
                   help='Simulation end time in seconds (default 0.5)')
    p.add_argument('--nflux', type=int, default=1_000_000,
                   help='Number of single-edge flux calls in throughput test '
                        '(default 1000000)')
    p.add_argument('--threads', type=int, default=None,
                   help='OMP_NUM_THREADS override (default: env value)')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _header(title):
    print()
    print('=' * 62)
    print(title)
    print('=' * 62)


def _row(label, val_a, val_b, fmt='.4f', label_a='central', label_b='hllc'):
    print(f'  {label:<36s}  {label_a}: {val_a:{fmt}}   {label_b}: {val_b:{fmt}}')


def _timeit(fn, n=3):
    """Return minimum wall-time over n calls."""
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


# ---------------------------------------------------------------------------
# 1. Single-edge flux throughput
# ---------------------------------------------------------------------------

def benchmark_flux_throughput(n_calls):
    _header('1. Single-edge flux throughput')

    from anuga.shallow_water.sw_domain_openmp_ext import (
        flux_function_central,
        flux_function_hllc,
    )

    g = 9.8
    eps = 1.0e-6
    rng = np.random.default_rng(42)

    # Build random (but physically valid) edge states
    n = n_calls
    h_l = rng.uniform(0.1, 3.0, n)
    h_r = rng.uniform(0.1, 3.0, n)
    u_l = rng.uniform(-2.0, 2.0, n)
    u_r = rng.uniform(-2.0, 2.0, n)
    v_l = rng.uniform(-1.0, 1.0, n)
    v_r = rng.uniform(-1.0, 1.0, n)
    normals = np.tile([1.0, 0.0], (n, 1))

    def _run(fn):
        ef = np.zeros(3, dtype=float)
        for i in range(n):
            ql = np.array([h_l[i], h_l[i] * u_l[i], h_l[i] * v_l[i]])
            qr = np.array([h_r[i], h_r[i] * u_r[i], h_r[i] * v_r[i]])
            fn(normals[i], ql, qr,
               h_l[i], h_r[i], h_l[i], h_r[i],
               ef, eps, 0.0, g, 1.0, h_l[i], h_r[i], 0)

    # Warm-up
    _run(flux_function_central)
    _run(flux_function_hllc)

    t_c = _timeit(lambda: _run(flux_function_central), n=2)
    t_h = _timeit(lambda: _run(flux_function_hllc), n=2)

    mflux_c = n_calls / t_c / 1e6
    mflux_h = n_calls / t_h / 1e6

    print(f'  {"Throughput (Mflux/s)":<36s}  central: {mflux_c:.2f}   hllc: {mflux_h:.2f}')
    print(f'  {"HLLC / central speed ratio":<36s}  {t_c / t_h:.3f}x')
    return t_c, t_h


# ---------------------------------------------------------------------------
# 2. 1D dam-break accuracy vs Ritter analytical solution
# ---------------------------------------------------------------------------

def ritter_stage(x, x_dam, h0, t, g=9.8):
    """Ritter (1892) analytical solution for 1D dam-break into a dry bed.

    Parameters
    ----------
    x      : array of positions
    x_dam  : dam location
    h0     : initial water depth on the wet side
    t      : time after dam break
    g      : gravity

    Returns
    -------
    h : array of water depths
    """
    c0 = sqrt(g * h0)
    xi = (x - x_dam) / (c0 * t)    # dimensionless position
    h = np.where(xi <= -1.0,
                 h0,
                 np.where(xi <= 2.0,
                          h0 / 9.0 * (2.0 - xi) ** 2,
                          0.0))
    return h


def benchmark_dam_break_accuracy(nx, finaltime):
    _header('2. 1D dam-break accuracy (vs. Ritter analytical)')

    import anuga
    from anuga import Reflective_boundary

    Lx = 20.0      # domain length
    Ly = 2.0       # domain width (thin strip)
    x_dam = Lx / 2.0
    h0 = 1.0       # upstream depth

    def _make_domain(solver, name):
        domain = anuga.rectangular_cross_domain(nx, max(nx // 10, 2),
                                                len1=Lx, len2=Ly,
                                                origin=(0, 0))
        domain.set_name(name)
        domain.set_store(False)
        domain.set_flow_algorithm('DE0')
        domain.set_quantity('elevation', 0.0)
        domain.set_quantity('stage', lambda x, y: np.where(x < x_dam, h0, 0.05))
        domain.set_quantity('xmomentum', 0.0)
        domain.set_quantity('ymomentum', 0.0)
        domain.set_boundary({'left':   Reflective_boundary(domain),
                             'right':  Reflective_boundary(domain),
                             'top':    Reflective_boundary(domain),
                             'bottom': Reflective_boundary(domain)})
        domain.set_flux_solver(solver)
        return domain

    stages = {}
    xc_final = {}
    for solver in ['central', 'hllc']:
        domain = _make_domain(solver, f'tmp_dam_{solver}')
        t0 = time.perf_counter()
        for _ in domain.evolve(yieldstep=finaltime, finaltime=finaltime):
            pass
        elapsed = time.perf_counter() - t0

        xc = domain.get_quantity('stage').domain.centroid_coordinates[:, 0]
        h  = domain.get_quantity('stage').get_values(location='centroids')
        stages[solver] = h.copy()
        xc_final[solver] = xc.copy()

        print(f'  {solver:8s}: evolved to t={finaltime:.2f}s in {elapsed:.3f}s')

    # Sort by x for comparison
    idx_c = np.argsort(xc_final['central'])
    x_num = xc_final['central'][idx_c]
    h_central = stages['central'][idx_c]
    h_hllc = stages['hllc'][idx_c]
    h_exact = ritter_stage(x_num, x_dam, h0, finaltime)

    # Error metrics (L1 relative to h0)
    dx = np.diff(x_num, prepend=x_num[0])
    Lx_total = x_num[-1] - x_num[0]
    l1_c = np.sum(np.abs(h_central - h_exact) * np.abs(dx)) / Lx_total
    l1_h = np.sum(np.abs(h_hllc - h_exact) * np.abs(dx)) / Lx_total

    print()
    print(f'  L1 error vs Ritter solution (mean |h_num - h_exact| / L):')
    _row('L1 error', l1_c, l1_h, fmt='.5f')

    # Bore-front detection: position where h first drops below 0.1
    def bore_pos(x, h, threshold=0.1):
        mask = h > threshold
        if mask.any():
            return x[mask].max()
        return np.nan

    bp_exact = x_dam + 2.0 * sqrt(9.8 * h0) * finaltime
    bp_c = bore_pos(x_num, h_central)
    bp_h = bore_pos(x_num, h_hllc)
    print()
    print(f'  Bore front position (analytical: {bp_exact:.3f} m):')
    _row('  bore_front (m)', bp_c, bp_h, fmt='.3f')

    return l1_c, l1_h


# ---------------------------------------------------------------------------
# 3. Full-domain timing
# ---------------------------------------------------------------------------

def benchmark_domain_timing(nx, finaltime):
    _header('3. Full-domain timing (wall time per timestep)')

    import anuga
    from anuga import Reflective_boundary

    Lx, Ly = 10.0, 10.0

    def _make_domain(solver):
        domain = anuga.rectangular_cross_domain(nx, nx,
                                                len1=Lx, len2=Ly,
                                                origin=(0, 0))
        domain.set_name('tmp_timing')
        domain.set_store(False)
        domain.set_flow_algorithm('DE0')
        domain.set_quantity('elevation', 0.0)
        domain.set_quantity('stage', lambda x, y: np.where(x < Lx/2, 2.0, 1.0))
        domain.set_quantity('xmomentum', 0.0)
        domain.set_quantity('ymomentum', 0.0)
        domain.set_boundary({'left':   Reflective_boundary(domain),
                             'right':  Reflective_boundary(domain),
                             'top':    Reflective_boundary(domain),
                             'bottom': Reflective_boundary(domain)})
        domain.set_flux_solver(solver)
        return domain

    ntri = nx * nx * 4   # approximate (rectangular_cross gives ~4*nx^2 triangles)
    print(f'  Mesh: ~{ntri} triangles  (nx={nx})')

    results = {}
    for solver in ['central', 'hllc']:
        domain = _make_domain(solver)
        n_steps = 0
        t0 = time.perf_counter()
        for _ in domain.evolve(yieldstep=finaltime, finaltime=finaltime):
            n_steps += 1
        elapsed = time.perf_counter() - t0
        actual_steps = domain.number_of_steps
        ms_per_step = elapsed / max(actual_steps, 1) * 1e3
        results[solver] = (elapsed, ms_per_step, actual_steps)
        print(f'  {solver:8s}: {elapsed:.3f}s total, '
              f'{actual_steps} steps, {ms_per_step:.2f} ms/step')

    t_c = results['central'][1]
    t_h = results['hllc'][1]
    print(f'\n  HLLC overhead vs central: {(t_h/t_c - 1)*100:.1f}%')
    return t_c, t_h


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = _parse()

    if args.threads is not None:
        os.environ['OMP_NUM_THREADS'] = str(args.threads)
        print(f'OMP_NUM_THREADS set to {args.threads}')

    print()
    print('HLLC Riemann solver benchmark')
    print(f'  nx={args.nx}  finaltime={args.finaltime}  nflux={args.nflux}')
    threads = os.environ.get('OMP_NUM_THREADS', '(env default)')
    print(f'  OMP_NUM_THREADS={threads}')

    benchmark_flux_throughput(args.nflux)
    benchmark_dam_break_accuracy(args.nx, args.finaltime)
    benchmark_domain_timing(args.nx, args.finaltime)

    print()
    print('Done.')


if __name__ == '__main__':
    main()
