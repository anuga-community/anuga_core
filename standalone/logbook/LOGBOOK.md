# Scaling logbook — standalone ANUGA miniapp

A running record of the SC26 scaling campaign: what was run, where, what came
out, and what is still in flight. Newest entries at the bottom of each machine
section. Raw CSVs live under `standalone/benchmarks/results/` (gitignored), so
the numbers that matter are copied into the tables here.

Benchmark problem used throughout, unless noted: wet dam break, `--scheme ader2
--flux scatter`, 1-D slab decomposition along x, one rank per GPU/GCD/tile,
ny = 2048, dx = 1 m, `--warmup 10 --repeat 3 --phases`, 100 steps. Metric:
ms/step (the harness reports the best of the 3 repeats).
`halo_ms` / `dt_ms` are per-phase maxima over ranks; `dt_ms` includes waiting for
the slowest rank (no `--phases-sync`).

Sessions working on this (October 2026):

| session  | machine            | owns                                                       |
|----------|--------------------|------------------------------------------------------------|
| gadi     | Gadi (V100, H200)  | `benchmarks/scaling.py`, `analyze_scaling.py`              |
| frontier | OLCF Frontier      | AMD Makefile configs, `frontier` entry in `scaling.json`   |
| aurora   | ALCF Aurora        | Intel Makefile configs, `aurora` entry in `scaling.json`   |

---

## Frontier (OLCF, MI250X, 8 GCDs/node, project CSC693)

Budget: **25k node-hours** — spend carefully. Every point gets `--timeout 600`.
Do a cheap check before any large allocation.

### Environment (verified 2026-10-05)

```bash
module load PrgEnv-amd amd/10.0.0 rocm/10.0.0        # NOT craype-accel-amd-gfx90a
export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$(pkg-config --variable=libdir libfabric):$LD_LIBRARY_PATH
cd standalone && make CONFIG=amdgpumpi MPICC=cc AMD_ARCH=gfx90a     # -> bin/bench_amdgpumpi
```

- `craype-accel-amd-gfx90a` makes `cc` link `libmpi_gtl_hsa`, which needs ROCm 6's
  `libamdhip64.so.6`. Under rocm/10.0.0 the binary then fails to load. GTL is not
  needed: the halo is packed on device and staged through the host
  (`MPICH_GPU_SUPPORT_ENABLED=0`).
- `~/.bashrc` sets `LD_LIBRARY_PATH` outright (`=$ROCM_PATH/lib`). That drops
  libfabric's path even though the module stays loaded, so the job fails with
  `libfabric.so.1: not found`. Hence the `pkg-config` term above.
- Launch line: `srun -N N -n 8N --ntasks-per-node=8 -c 7 --threads-per-core=1
  --gpus-per-task=1 --gpu-bind=closest`. Checked: each rank gets its own GCD
  (`ROCR_VISIBLE_DEVICES` 0–7) and 7 disjoint cores (56 = the cores left by the
  default `-S 8`).
- Login nodes can compile but not run GPU jobs. The user runs interactive tests
  (`salloc -A CSC693 -N 1`); batch submission is allowed.
- In a non-interactive agent shell, `module load` does not persist between
  commands. Use `eval "$($LMOD_CMD bash load ...)"` in the same command.

### 2026-10-05 — first contact (commit 14e48919)

**`omp target teams loop` is ~11.5× slower than `teams distribute parallel for`
under amdclang.** One GCD, 1M triangles (`frontier_smoke.sh ab`):

| loop construct                     | ms/step | Mcell-steps/s | volume drift |
|------------------------------------|---------|---------------|--------------|
| `teams loop`                       | 28.01   | 36.0          | 0            |
| `teams distribute parallel for`    | 2.44    | 413.5         | 0            |

Fix: `-DANUGA_OMP_TEAMS_DPF` (set by `CONFIG=amdgpu|amdgpumpi`) switches
`OMP_PARALLEL_LOOP[_SIMD]` in `gpu_omp_macros.h` and `DI_GEOM_LOOP` in
`src/setup_device.c`. nvc builds keep `teams loop` unchanged.

Also fixed: `CONFIG=amdgpu` had `CC ?= amdclang`, which never takes effect
because make predefines `CC=cc`.

1-node smoke (8 GCDs, `frontier_smoke.sh mpi`):

| tris/GCD | ms/step | kernels | halo | dt  | Mcell-steps/s/node | GiB/GCD | drift   |
|----------|---------|---------|------|-----|--------------------|---------|---------|
| 1M       | 2.75    | 2.52    | 0.25 | 0.19| 2903               | 0.46    | 2.7e-14 |
| 40M      | 110.6   | 110.3   | 0.67 | 7.5 | 2895               | 18.2    | 1.7e-13 |

Open question: in the 8-rank runs each GCD delivers ~362 Mcell-steps/s, vs 413
for a lone GCD at 1M. The ~12% gap is size-independent, so it is not the halo.
A single-GCD 40M run would separate kernel slowdown from waiting at the dt
allreduce.

### 2026-10-05 — weak scaling, 40M triangles/GCD

Jobs 5624934 (1–8 nodes, 3m51s), 5624969 (8–256 nodes, 5m59s, ~26 node-h) and
5625186 (512 + 1024 nodes, 2m25s, ~41 node-h; ran 2026-10-06).
Efficiency = t(1 node) / t(N), with t(1) = 110.1 ms.

| nodes | GCDs | triangles | ms/step | halo ms | dt ms | efficiency |
|-------|------|-----------|---------|---------|-------|------------|
| 1     | 8    | 0.32G     | 110.1   | 0.80    | 8.3   | 1.000      |
| 2     | 16   | 0.64G     | 113.7   | 0.92    | 10.6  | 0.968      |
| 4     | 32   | 1.28G     | 110.8   | 1.08    | 11.4  | 0.993      |
| 8     | 64   | 2.56G     | 113.7 / 110.2 | 0.85 / 1.04 | 12.6 / 8.9 | 0.968 / 0.999 |
| 16    | 128  | 5.12G     | 114.2   | 1.18    | 12.2  | 0.964      |
| 32    | 256  | 10.2G     | 114.4   | 1.05    | 13.1  | 0.962      |
| 64    | 512  | 20.5G     | 113.8   | 0.98    | 13.3  | 0.968      |
| 128   | 1024 | 41.0G     | 113.9   | 1.32    | 12.6  | 0.967      |
| 256   | 2048 | 81.9G     | 114.4   | 1.25    | 14.8  | 0.962      |
| 512   | 4096 | 164G      | 113.9   | 1.21    | 14.0  | 0.967      |
| 1024  | 8192 | 328G      | 114.7   | 1.16    | 15.4  | **0.960**  |

- All points `ok`, no NaNs, volume drift ≤ 1.7e-13. 18.2 GiB/GCD.
- Efficiency settles near 96% by 16 nodes and stays flat from 16 to 1024 nodes.
  The halo stays at ~1 ms. Kernels are ~99% of the step.
- Run-to-run noise is about ±2%: the 8-node point was measured in both jobs.
- 256 nodes reach ~716 G cell-steps/s; 1024 nodes (8192 GCDs, 328G triangles)
  reach ~2.86 T cell-steps/s. The dt allreduce grows slowly (8 → 15 ms, log-like
  in node count); the halo does not grow.
- Plot: `python3 standalone/logbook/plot_weak_frontier.py out.png` (reads every
  `benchmarks/results/frontier_n*/scaling_frontier.csv`).

### 2026-10-06 — strong scaling (overnight jobs 5625208–5625215)

All 8 jobs and all 12 points `ok`, no NaNs, |drift| ≤ 2.4e-14. Cost of the
overnight batch including the 1024-node weak job: **~71 node-h** (estimate was
~130). Job scripts: `benchmarks/results/frontier_strong/jobs/strong_n*.sl`.
Efficiency = (t_ref · N_ref) / (t_N · N).

**Large / headline — 81.9G triangles (nx = 10000384), reference = 128 nodes:**

| nodes | GCDs | tris/GCD | ms/step | kernel ms | halo ms | dt ms | efficiency |
|-------|------|----------|---------|-----------|---------|-------|------------|
| 128   | 1024 | 80.0M    | 224.8   | 226.0     | 1.59    | 23.1  | 1.000      |
| 256   | 2048 | 40.0M    | 114.5   | 114.2     | 1.19    | 15.3  | 0.982      |
| 512   | 4096 | 20.0M    | 57.6    | 57.1      | 0.64    | 8.0   | 0.976      |
| 1024  | 8192 | 10.0M    | 28.8    | 28.3      | 0.46    | 4.4   | **0.977**  |

- 7.8× speed-up for 8× the nodes. The 256-node point reproduces the weak-scaling
  run of the same mesh (114.45 vs 114.4 ms).
- 128 nodes held 36.4 GiB/GCD without trouble.

**Small / failure point — 2.56G triangles (nx = 312512), reference = 8 nodes:**

| nodes | GCDs | tris/GCD | ms/step | kernel ms | halo ms | dt ms | efficiency |
|-------|------|----------|---------|-----------|---------|-------|------------|
| 8     | 64   | 40.0M    | 111.3   | 112.0     | 1.28    | 11.2  | 1.000      |
| 16    | 128  | 20.0M    | 56.3    | 56.1      | 0.74    | 5.95  | 0.988      |
| 32    | 256  | 10.0M    | 28.1    | 28.0      | 0.41    | 3.10  | 0.990      |
| 64    | 512  | 5.0M     | 13.81   | 13.49     | 0.23    | 1.54  | 1.007      |
| 128   | 1024 | 2.5M     | 6.97    | 6.50      | 0.22    | 0.96  | 0.999      |
| 256   | 2048 | 1.25M    | 3.42    | 3.18      | 0.24    | 0.39  | 1.016      |
| 512   | 4096 | 0.63M    | 2.00    | 1.73      | 0.27    | 0.32  | 0.869      |
| 1024  | 8192 | 0.31M    | 1.30    | 0.99      | 0.27    | 0.28  | **0.668**  |

- It holds ≥ 99% down to 1.25M triangles/GCD (256 nodes). That is much later than
  the predicted fall-off at 64–128 nodes. The ~5M/GCD rule of thumb was too pessimistic.
  The fixed costs are smaller at small sizes: halo ~0.25 ms (vs ~1 ms at 40M,
  where the message is larger) and dt ~0.3 ms (less imbalance to wait for).
- Slightly superlinear points (64, 256 nodes) are within the ±2% noise. Part of the cause is a lower
  per-triangle kernel cost at small sizes: 2.55 ms/M at 1.25M vs 2.8 ms/M at 40M.
- The fall-off starts below ~1M triangles/GCD: 87% at 0.63M, 67% at 0.31M. At
  1024 nodes the step is 1.30 ms, of which ~0.55 ms is halo + dt. That floor is
  set by latency.
- Plot: `python3 standalone/logbook/plot_strong_frontier.py out.png` (reads every
  `benchmarks/results/frontier_strong/n*/*/scaling_frontier.csv`). Current renders:
  `logbook/weak_frontier.png`, `logbook/strong_frontier.png`.

### 2026-10-06 — full-machine strong scaling (jobs 5629157–5629159)

Strong scaling on 1.263T triangles (nx = 154140672, ny = 2048), reference 2048 nodes.
All three jobs COMPLETED with exit 0 and status `ok`: 0 NaNs, volume drift <= 2.1e-16.
The 2048- and 4096-node jobs started at 19:46, ahead of the 2026-10-07 estimate.

| nodes | ranks | tris/GCD | GiB/GCD | ms/step | halo ms | dt ms | speedup | eff. | elapsed | node-h |
|-------|-------|----------|---------|---------|---------|-------|---------|------|---------|--------|
| 2048 | 16384 | 77.1M | 35.0 | 218.13 | 2.04 | 25.3 | 1.00x | — | 2:16 | 77 |
| 4096 | 32768 | 38.5M | 17.5 | 109.88 | 1.38 | 14.8 | 1.99x | 99.3% | 1:30 | 102 |
| 8192 | 65536 | 19.3M | 8.8  | 55.47  | 0.70 | 8.1  | 3.93x | 98.3% | 1:55 | 262 |

- Peak aggregate throughput: 22.8T triangle-steps/s on 8192 nodes (65536 GCDs).
- Per-GCD rate lines up with the smaller meshes: 77M tris/GCD runs at 218 ms, against
  224.8 ms for 80M/GCD on the 81.9G mesh.
- The batch cost about 440 node-h, well under the ~495–635 estimated.
- Still queued: `full_n9408.sl` (5629160; strong + weak 3.01T at 40M/GCD). On
  2026-10-06 20:16 Slurm estimated a 20:54 start. `full_n9216.sl` was never
  submitted and is now only a fallback.
- Scripts are in `benchmarks/results/frontier_full/jobs/`. Results are in
  `benchmarks/results/frontier_full/n*/{strong,weak}/scaling_frontier.csv`.
- Plots: `plot_strong_frontier.py` and `plot_weak_frontier.py` now also read
  `frontier_full/`. The new `plot_throughput_frontier.py out.png` draws every weak and strong
  run as aggregate triangle-steps/s against the 1-node-weak × N ideal.
  Renders: `logbook/strong_frontier.png` and `logbook/throughput_frontier.png`.

### Caveats for the paper

- Weak scaling is a best case by construction: 1-D slab, halo size independent
  of rank count, uniform all-wet work.
- `scaling.py` flags `BAD_PHYSICS` when volume drift exceeds 1e-9. At 1e11
  triangles, summation roundoff could trip that gate; check before trusting it.
  So far drift *decreases* with size (2.0e-16 at 512 and 1024 nodes).
- `scaling.py emit` does not pass `--timeout` / `--out` through to the run
  line; job scripts were edited by hand (feature request sent to the gadi
  session).

---

## Aurora (ALCF, PVC, 12 tiles/node)

Briefed 2026-10-05; work is in progress in the aurora session. As of the last
report, `CONFIG=intelgpumpi` builds, with an optional AOT build for PVC
(`INTEL_AOT=pvc`), and `benchmarks/aurora_smoke.sh` is written. GPU smoke tests
are waiting on the user's interactive session. Whether `ANUGA_OMP_TEAMS_DPF`
helps on Intel is still to be measured (1-tile A/B). That session owns this
section.

---

## Gadi (NCI, V100 / H200)

H200 strong scaling, 400M triangles total: efficiency 0.976 at 2 nodes (50M
triangles/GPU), 0.963 at 4, 0.904 at 8 (12.5M/GPU); the 15-node point was still
queued. The gadi session owns this section.
