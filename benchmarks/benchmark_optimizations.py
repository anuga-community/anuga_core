#!/usr/bin/env python3
"""Benchmark: Algorithmic Optimizations for ANUGA

Measures performance of new algorithmic additions vs. existing implementations:

  1. CG solver variants: plain, Jacobi-preconditioned, SSOR-preconditioned,
     and persistent-thread
  2. Point-in-polygon: sequential vs. parallel (prefix-sum) partitioning
  3. Quantity extrapolation: multi-pass vs. single-pass fused kernel

Run with::

    python benchmarks/benchmark_optimizations.py

Optional arguments::

    --ntri N     Number of triangles for quantity benchmark (default: 50000)
    --npoints N  Number of query points for polygon benchmark (default: 200000)
    --cg-size N  CG matrix size (default: 5000)
    --threads N  OMP_NUM_THREADS to use (default: 1, or current env value)

Results are printed as a comparison table.

Performance notes
-----------------
* OMP_NUM_THREADS=1 (default in CI): parallel algorithms show little benefit
  because OpenMP overheads dominate at small sizes. Use --threads 4 (or more)
  with --ntri 500000 --npoints 2000000 to observe the expected speedups.

* SSOR preconditioner: the tridiagonal test matrix used here already has a
  low condition number, so SSOR does not reduce iterations vs. Jacobi/plain.
  SSOR shows the most benefit on ill-conditioned FEM stiffness matrices
  (condition number > 1e4) arising from irregular meshes and bathymetry fitting.

* Persistent-thread CG: small matrices are dominated by Python overhead, not
  fork/join cost. The benefit appears for N > 50K with multiple threads.

* Fused extrapolation: memory-bandwidth bound, so benefits are greatest on
  large meshes (> 200K triangles) with full 5-quantity DE domain runs.
"""

import sys
import os
import time
import argparse
import numpy as np

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _timeit(fn, n_repeat=3):
    """Return minimum wall-time (seconds) over ``n_repeat`` calls."""
    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


def _header(title):
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def _row(label, t_old, t_new, unit="ms"):
    scale = 1e3 if unit == "ms" else 1.0
    speedup = t_old / t_new if t_new > 0 else float("inf")
    print(f"  {label:<38s}  {t_old*scale:8.2f} {unit}  ->  {t_new*scale:8.2f} {unit}"
          f"   ({speedup:.2f}x speedup)")


# ---------------------------------------------------------------------------
# 1. CG Solver benchmark
# ---------------------------------------------------------------------------

def bench_cg(n=5000, n_repeat=5):
    """Compare CG solver variants on a synthetic SPD system."""
    from anuga.utilities.sparse import Sparse, Sparse_CSR
    from anuga.utilities.cg_solve import conjugate_gradient

    _header("1. CG Solver Variants")

    rng = np.random.default_rng(42)

    # Build a sparse SPD matrix: diagonal dominant, bandwidth-3
    A_sparse = Sparse(n, n)
    for i in range(n):
        A_sparse[i, i] = 4.0
        if i > 0:
            A_sparse[i, i - 1] = -1.0
            A_sparse[i - 1, i] = -1.0
    A_csr = Sparse_CSR(A_sparse)

    b = rng.standard_normal(n)
    x0 = np.zeros(n)

    # Reference: plain C CG (no preconditioner)
    t_plain = _timeit(
        lambda: conjugate_gradient(A_csr, b, x0.copy(),
                                   use_c_cg=True, precon='None'),
        n_repeat)

    # Jacobi-preconditioned C CG
    t_jacobi = _timeit(
        lambda: conjugate_gradient(A_csr, b, x0.copy(),
                                   use_c_cg=True, precon='Jacobi'),
        n_repeat)

    # SSOR-preconditioned C CG  (new)
    t_ssor = _timeit(
        lambda: conjugate_gradient(A_csr, b, x0.copy(),
                                   use_c_cg=True, precon='SSOR', omega=1.2),
        n_repeat)

    # Persistent-thread C CG  (new)
    t_persistent = _timeit(
        lambda: conjugate_gradient(A_csr, b, x0.copy(),
                                   use_c_cg=True, solver='persistent'),
        n_repeat)

    print(f"  Matrix size: {n} x {n}  (bandwidth-3 SPD tridiagonal)")
    print()
    _row("Plain C CG  (baseline)", t_plain, t_plain)
    _row("Jacobi-preconditioned C CG", t_plain, t_jacobi)
    _row("SSOR-preconditioned C CG  (new)", t_plain, t_ssor)
    _row("Persistent-thread C CG  (new)", t_plain, t_persistent)

    # Verify correctness: all solvers should give the same solution
    x_ref  = conjugate_gradient(A_csr, b, x0.copy(), use_c_cg=True)
    x_ssor = conjugate_gradient(A_csr, b, x0.copy(),
                                use_c_cg=True, precon='SSOR', omega=1.2)
    x_pers = conjugate_gradient(A_csr, b, x0.copy(),
                                use_c_cg=True, solver='persistent')

    err_ssor = np.linalg.norm(x_ssor - x_ref) / (np.linalg.norm(x_ref) + 1e-14)
    err_pers = np.linalg.norm(x_pers - x_ref) / (np.linalg.norm(x_ref) + 1e-14)
    print()
    print(f"  Correctness check (relative error vs. plain CG):")
    print(f"    SSOR-preconditioned : {err_ssor:.2e}  {'OK' if err_ssor < 1e-6 else 'FAIL'}")
    print(f"    Persistent-thread   : {err_pers:.2e}  {'OK' if err_pers < 1e-6 else 'FAIL'}")


# ---------------------------------------------------------------------------
# 2. Point-in-polygon benchmark
# ---------------------------------------------------------------------------

def bench_polygon(n_points=200_000, n_repeat=5):
    """Compare sequential vs. parallel prefix-sum point partitioning."""
    from anuga.geometry.polygon import separate_points_by_polygon

    _header("2. Point-in-Polygon Partitioning")

    rng = np.random.default_rng(123)

    # A simple closed polygon (unit square)
    polygon = np.array([[0.0, 0.0], [1.0, 0.0],
                        [1.0, 1.0], [0.0, 1.0]], dtype=float)

    # Spread points over [-0.5, 1.5] x [-0.5, 1.5] so ~25% fall inside
    points = rng.uniform(-0.5, 1.5, size=(n_points, 2))

    t_seq = _timeit(
        lambda: separate_points_by_polygon(points, polygon,
                                           check_input=False,
                                           use_parallel=False),
        n_repeat)

    t_par = _timeit(
        lambda: separate_points_by_polygon(points, polygon,
                                           check_input=False,
                                           use_parallel=True),
        n_repeat)

    print(f"  Points: {n_points:,}   Polygon vertices: {len(polygon)}")
    print()
    _row("Sequential partitioning  (baseline)", t_seq, t_seq)
    _row("Parallel prefix-sum      (new)", t_seq, t_par)

    # Correctness: counts must match and the first `count` indices must
    # refer to points that are actually inside the polygon.
    idx_seq, cnt_seq = separate_points_by_polygon(points, polygon,
                                                   check_input=False,
                                                   use_parallel=False)
    idx_par, cnt_par = separate_points_by_polygon(points, polygon,
                                                   check_input=False,
                                                   use_parallel=True)

    inside_seq = set(idx_seq[:cnt_seq].tolist())
    inside_par = set(idx_par[:cnt_par].tolist())
    count_match = cnt_seq == cnt_par
    set_match   = inside_seq == inside_par
    print()
    print("  Correctness check:")
    print(f"    Count match  : {cnt_seq} == {cnt_par}  "
          f"{'OK' if count_match else 'FAIL'}")
    print(f"    Index sets   : {'OK' if set_match else 'FAIL'}")


# ---------------------------------------------------------------------------
# 3. Quantity extrapolation benchmark
# ---------------------------------------------------------------------------

def bench_extrapolate(n_tri=50_000, n_repeat=5):
    """Compare multi-pass vs. single-pass fused extrapolation."""
    from anuga.abstract_2d_finite_volumes.quantity_openmp_ext import (
        extrapolate_second_order_and_limit_by_edge,
        extrapolate_second_order_and_limit_by_vertex,
        extrapolate_second_order_and_limit_by_edge_fused,
        extrapolate_second_order_and_limit_by_vertex_fused,
    )

    _header("3. Quantity Extrapolation: Multi-Pass vs. Fused")

    # Build a minimal fake domain/quantity so the Cython wrappers work
    # without starting a full ANUGA simulation.
    class FakeDomain:
        pass

    class FakeQuantity:
        pass

    rng = np.random.default_rng(7)
    N = n_tri

    domain = FakeDomain()
    domain.centroid_coordinates    = rng.uniform(0, 10, size=(N, 2))
    domain.vertex_coordinates      = rng.uniform(0, 10, size=(N, 6))
    domain.number_of_boundaries    = rng.integers(0, 3, size=N).astype(np.int64)
    # surrogate_neighbours: all point to self or first neighbour
    sn = np.zeros((N, 3), dtype=np.int64)
    sn[:, 0] = np.arange(N)
    sn[:, 1] = np.roll(np.arange(N), 1)
    sn[:, 2] = np.roll(np.arange(N), 2)
    domain.surrogate_neighbours = sn
    # neighbours: -1 means boundary
    nb = np.full((N, 3), -1, dtype=np.int64)
    mask = domain.number_of_boundaries < 3
    nb[mask, 0] = np.roll(np.arange(N), 1)[mask]
    domain.neighbours = nb

    qty = FakeQuantity()
    qty.domain            = domain
    qty.object            = domain  # needed by legacy multi-pass wrappers
    qty.beta              = 0.9
    qty.centroid_values   = rng.uniform(0, 1, size=N)
    qty.vertex_values     = np.zeros((N, 3))
    qty.edge_values       = np.zeros((N, 3))
    qty.phi               = np.zeros(N)
    qty.x_gradient        = np.zeros(N)
    qty.y_gradient        = np.zeros(N)

    def _reset():
        qty.vertex_values[:] = 0.0
        qty.edge_values[:]   = 0.0
        qty.phi[:]           = 0.0
        qty.x_gradient[:]    = 0.0
        qty.y_gradient[:]    = 0.0

    # Baseline: existing multi-pass (edge limiting)
    def _multipass_edge():
        _reset()
        extrapolate_second_order_and_limit_by_edge(qty)

    t_mp_edge = _timeit(_multipass_edge, n_repeat)

    # New fused (edge limiting)
    def _fused_edge():
        _reset()
        extrapolate_second_order_and_limit_by_edge_fused(qty)

    t_fused_edge = _timeit(_fused_edge, n_repeat)

    # Baseline: existing multi-pass (vertex limiting)
    def _multipass_vert():
        _reset()
        extrapolate_second_order_and_limit_by_vertex(qty)

    t_mp_vert = _timeit(_multipass_vert, n_repeat)

    # New fused (vertex limiting)
    def _fused_vert():
        _reset()
        extrapolate_second_order_and_limit_by_vertex_fused(qty)

    t_fused_vert = _timeit(_fused_vert, n_repeat)

    print(f"  Triangles: {N:,}")
    print()
    _row("Multi-pass edge-limit   (baseline)", t_mp_edge, t_mp_edge)
    _row("Fused edge-limit        (new)",      t_mp_edge, t_fused_edge)
    _row("Multi-pass vertex-limit (baseline)", t_mp_vert, t_mp_vert)
    _row("Fused vertex-limit      (new)",      t_mp_vert, t_fused_vert)

    # Correctness: both paths must produce the same edge and vertex values
    _reset()
    extrapolate_second_order_and_limit_by_edge(qty)
    ev_old = qty.edge_values.copy()
    vv_old = qty.vertex_values.copy()
    xg_old = qty.x_gradient.copy()

    _reset()
    extrapolate_second_order_and_limit_by_edge_fused(qty)
    ev_new = qty.edge_values.copy()
    vv_new = qty.vertex_values.copy()
    xg_new = qty.x_gradient.copy()

    max_ev_err = np.max(np.abs(ev_new - ev_old))
    max_vv_err = np.max(np.abs(vv_new - vv_old))
    max_xg_err = np.max(np.abs(xg_new - xg_old))

    print()
    print("  Correctness check (fused edge vs. multi-pass edge):")
    print(f"    Max |edge_values diff|   : {max_ev_err:.2e}  "
          f"{'OK' if max_ev_err < 1e-10 else 'FAIL'}")
    print(f"    Max |vertex_values diff| : {max_vv_err:.2e}  "
          f"{'OK' if max_vv_err < 1e-10 else 'FAIL'}")
    print(f"    Max |x_gradient diff|    : {max_xg_err:.2e}  "
          f"{'OK' if max_xg_err < 1e-10 else 'FAIL'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark algorithmic optimizations in ANUGA")
    parser.add_argument("--ntri",     type=int, default=50_000,
                        help="Number of triangles for quantity benchmark")
    parser.add_argument("--npoints",  type=int, default=200_000,
                        help="Number of query points for polygon benchmark")
    parser.add_argument("--cg-size",  type=int, default=5_000,
                        help="CG matrix size")
    parser.add_argument("--threads",  type=int, default=None,
                        help="OMP_NUM_THREADS (overrides env variable)")
    args = parser.parse_args()

    if args.threads is not None:
        os.environ["OMP_NUM_THREADS"] = str(args.threads)

    n_threads = int(os.environ.get("OMP_NUM_THREADS", "1"))
    print(f"\nANUGA Optimization Benchmark  (OMP_NUM_THREADS={n_threads})")

    bench_cg(n=args.cg_size)
    bench_polygon(n_points=args.npoints)
    bench_extrapolate(n_tri=args.ntri)

    print()
    print("Done.")


if __name__ == "__main__":
    main()
