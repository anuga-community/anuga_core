# ANUGA Benchmarks

Tools to track wall-time and memory performance across commits.

## Quick start

```bash
# Activate your env
conda activate anuga_env_3.14

# Run small + medium scenarios, all modes (takes ~40 s)
python benchmarks/run_benchmarks.py

# Quick sanity check — small only (~5 s)
python benchmarks/run_benchmarks.py --sizes small

# List saved result files
python benchmarks/compare_benchmarks.py --list

# Compare two results
python benchmarks/compare_benchmarks.py benchmarks/results/before.json \
                                        benchmarks/results/after.json
```

## Scenarios

| Name   | Triangles | finaltime | Typical wall (mode 0) |
|--------|----------:|----------:|----------------------:|
| small  |     10 000 |     200 s |                  ~1 s |
| medium |     90 000 |     100 s |                 ~16 s |
| large  |    360 000 |      50 s |                 ~90 s |

`large` is not run by default. Use `--sizes small,medium,large` to include it.

## Modes

| Mode | Description |
|------|-------------|
| 0    | Python Euler (default) |
| 1    | Python RK2 |
| 2    | C RK2 / GPU (CPU_ONLY_MODE if no GPU present) |

## Metrics

| Metric      | Meaning |
|-------------|---------|
| `cells/s`   | n_triangles × n_steps / wall_time — primary performance figure |
| `setup MB`  | RSS after domain creation (before any evolve) |
| `peak MB`   | Peak RSS sampled at 100 ms intervals during evolve |
| `MB/Ktri`   | peak_MB / (n_triangles / 1000) — memory per 1 000 triangles |

## Workflow: before/after comparison

```bash
# 1. Capture baseline on current commit
python benchmarks/run_benchmarks.py --output /tmp/before.json

# 2. Make your code changes ...

# 3. Capture new result
python benchmarks/run_benchmarks.py --output /tmp/after.json

# 4. Compare
python benchmarks/compare_benchmarks.py /tmp/before.json /tmp/after.json

# Show only rows that changed by more than 5%
python benchmarks/compare_benchmarks.py /tmp/before.json /tmp/after.json --threshold 5
```

## Result files

Results are saved to `benchmarks/results/<branch>_<commit>_<timestamp>.json`.
The `results/` directory is git-ignored so committed baselines don't pollute
the repo. Copy important baselines elsewhere if you want to keep them.

---

## Per-kernel benchmarks

`run_benchmarks.py` above measures whole-evolve throughput only, so a regression in one
kernel hides inside an aggregate cells/s figure. `run_kernel_benchmarks.py` times
individual hot kernels (`compute_fluxes`, `protect`, `distribute`, `extrapolate_edge_only`,
`manning_friction_flat`) in isolation on a primed, fixed domain state:

```bash
# Default: modes 1 and 2, medium mesh, 50 reps per kernel
python benchmarks/run_kernel_benchmarks.py

# Compare two compilers (or before/after a tuning change)
python benchmarks/run_kernel_benchmarks.py --output benchmarks/results/kernels_gcc.json
python benchmarks/run_kernel_benchmarks.py --output benchmarks/results/kernels_icx.json
python benchmarks/compare_kernel_benchmarks.py benchmarks/results/kernels_gcc.json \
                                               benchmarks/results/kernels_icx.json
```

This is part of `claude/PLAN_compiler_tuning.md` (cross-compiler kernel tuning) — see
`claude/KNOWN_ISSUES.md` for the current baseline findings across GCC/ICX/NVHPC.

---

## Parallel distribution benchmarks (MPI)

`distribute_benchmarks.py` compares four approaches for distributing a mesh
across MPI ranks. Requires `mpi4py`.

```bash
# Run all four methods on a 500×500 mesh (~1M tris) with 8 ranks
mpirun -np 8 python benchmarks/distribute_benchmarks.py --size 500

# Morton scheme, 3 repetitions for stable medians
mpirun -np 8 python benchmarks/distribute_benchmarks.py --size 500 --scheme morton --reps 3
```

| Method | Description |
|--------|-------------|
| `distribute()` | Traditional: full Domain on rank 0, then distribute |
| `distribute_collaborative()` | Shared-memory cooperative version |
| `distribute_basic_mesh()` | Mesh-first: only Basic_mesh on rank 0, quantities set locally after |
| `dump()+load()` | Rank 0 partitions and writes files; all ranks read |

### Grid sweep (multiple np × scheme combinations)

```bash
# Run the full np=[10,20,30] × scheme=[metis,morton,hilbert] grid
python benchmarks/run_benchmark_grid.py --size 1000

# Custom grid
python benchmarks/run_benchmark_grid.py --size 500 --np 4,8,16 --schemes metis,morton

# Dry run (print commands without executing)
python benchmarks/run_benchmark_grid.py --dry-run
```

Results are saved to `benchmarks/results/dist/bench_np<N>_<scheme>.txt` and
a consolidated `summary.txt`.
