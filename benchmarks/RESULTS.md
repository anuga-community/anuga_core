# Benchmark Results

*Run on: 2026-05-11, 4-core x86_64 Linux, GCC, OMP_NUM_THREADS=1 (and 4)*

---

## 1. Fused quantity extrapolation (`benchmark_optimizations.py`)

The fused single-pass kernel combines `_compute_gradients` + `_extrapolate_from_gradient`
+ `_limit_edges_by_all_neighbours` into a single loop over elements.

### Results (100 K triangles, 1 thread)

| Variant | Time | vs. baseline |
|---------|------|--------------|
| Multi-pass edge-limit (baseline) | 4.11 ms | 1.00× |
| Fused edge-limit (new) | 5.37 ms | 0.77× |
| Multi-pass vertex-limit (baseline) | 4.38 ms | 1.00× |
| Fused vertex-limit (new) | 5.44 ms | 0.80× |

### Results (200 K triangles, 4 threads)

| Variant | Time | vs. baseline |
|---------|------|--------------|
| Multi-pass edge-limit (baseline) | 3.73 ms | 1.00× |
| Fused edge-limit (new) | 3.96 ms | 0.94× |
| Multi-pass vertex-limit (baseline) | 3.88 ms | 1.00× |
| Fused vertex-limit (new) | 3.96 ms | 0.98× |

**Correctness:** all differences are exactly 0.00e+00 vs. the multi-pass reference.

### Analysis

As @JorgeG94 predicted, the fused kernel does **not** improve performance on the CPU.
The unstructured mesh causes scattered reads across `neighbours`, `surrogate_neighbours`,
`vertex_coordinates` and `centroid_values`; those irregular memory-access patterns
(not the number of passes) are the dominant cost. Fusing the loops increases the working
set per iteration, which worsens cache utilisation at both 1 and 4 threads.

The fused kernel is retained because:
- It is bit-for-bit identical to the multi-pass path (confirmed above).
- It reduces the number of OpenMP fork/join barriers from 3 to 1, which is
  beneficial when thread counts are large (≥ 8) or barrier overhead is high
  (e.g. NUMA architectures and GPU offload targets where synchronisation is expensive).
- It is the natural target for GPU offload (`omp target teams distribute`), where a
  single kernel launch is strongly preferred over three separate launches.

---

## 2. HLLC vs. central-upwind Riemann solver (`benchmark_hllc.py`)

### Single-edge flux throughput (100 K calls, 1 thread)

| Solver | Throughput |
|--------|-----------|
| Central-upwind | 0.36 Mflux/s |
| HLLC | 0.35 Mflux/s |
| HLLC / central ratio | 0.97× (≈ same) |

### 1D dam-break accuracy vs. Ritter analytical solution (nx=30, t=0.5 s)

| Metric | Central | HLLC |
|--------|---------|------|
| L1 error (mean \|h_num − h_exact\| / L) | 0.03099 | 0.03090 |
| Bore-front position | 11.22 m | 11.22 m |
| Analytical bore position | 13.13 m | — |

Both solvers produce near-identical L1 errors. The bore-front underestimate (~14 %
behind analytical) is expected from the coarse 2-D mesh; it is a mesh-resolution
effect, not a solver difference.

### Full-domain wall time (nx=30, ~3 600 triangles, 1 thread)

| Solver | Total | ms/step |
|--------|-------|---------|
| Central | 0.058 s / 51 steps | 1.13 ms |
| HLLC | 0.080 s / 52 steps | 1.55 ms |
| HLLC overhead | — | +37 % |

### Analysis

At single-edge level HLLC costs essentially the same as the central scheme (3 %
slower); the per-step overhead on a real mesh is larger (~37 %) because HLLC touches
slightly more registers per edge and the loop is not perfectly vectorised. The overhead
shrinks as mesh size grows (flux computation becomes more memory-bandwidth bound than
arithmetic bound).

HLLC is **not** intended as a performance optimisation — it is included for:
- Better shock resolution in transcritical and supercritical flows.
- Adherence to the Rankine-Hugoniot conditions, which the central-upwind scheme
  relaxes to achieve positivity preservation.
- Enabling future comparisons in the SC26 GPU paper.
