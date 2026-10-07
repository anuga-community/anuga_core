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

### 2026-10-06 — full machine: strong to 9408 nodes, weak at 9408 (jobs 5629157–5629160)

Strong scaling on 1.263T triangles (nx = 154140672, ny = 2048), reference 2048 nodes.
All four jobs COMPLETED with exit 0 and status `ok`: 0 NaNs, volume drift <= 2.1e-16.
The 2048- and 4096-node jobs started at 19:46, ahead of the 2026-10-07 estimate.

| nodes | ranks | tris/GCD | GiB/GCD | ms/step | halo ms | dt ms | speedup | eff. | elapsed | node-h |
|-------|-------|----------|---------|---------|---------|-------|---------|------|---------|--------|
| 2048 | 16384 | 77.1M | 35.0 | 218.13 | 2.04 | 25.3 | 1.00x | — | 2:16 | 77 |
| 4096 | 32768 | 38.5M | 17.5 | 109.88 | 1.38 | 14.8 | 1.99x | 99.3% | 1:30 | 102 |
| 8192 | 65536 | 19.3M | 8.8  | 55.47  | 0.70 | 8.1  | 3.93x | 98.3% | 1:55 | 262 |
| 9408 | 75264 | 16.8M | 7.6  | 48.28  | 0.87 | 7.4  | 4.52x of 4.59x | 98.4% | 4:33 (incl. weak) | 713 |

Weak at 9408 nodes (75264 GCDs, 40M tris/GCD, 3.01T triangles): 114.87 ms/step.
That is **95.8%** of the 1-node 110.08 ms, the same as 1024 nodes (114.71 ms, 96.0%).
The dt allreduce grows to 16.2 ms and halo to 1.5 ms, while kernels stay at 114.8 ms.

- Peak aggregate throughput: 26.2T triangle-steps/s on 9408 nodes, from both the strong
  (1.26T) and the weak (3.01T) run.
- Per-GCD rate lines up with the smaller meshes: 77M tris/GCD runs at 218 ms, against
  224.8 ms for 80M/GCD on the 81.9G mesh.
- The whole batch cost about 1150 node-h. The 9408-node job started at 20:43.
  `full_n9216.sl` was never submitted and is no longer needed.
- Scripts are in `benchmarks/results/frontier_full/jobs/`. Results are in
  `benchmarks/results/frontier_full/n*/{strong,weak}/scaling_frontier.csv`.
- Plots: `plot_strong_frontier.py` and `plot_weak_frontier.py` now also read
  `frontier_full/`. The new `plot_throughput_frontier.py out.png` draws every weak and strong
  run as aggregate triangle-steps/s against the 1-node-weak × N ideal.
  Renders: `logbook/{strong,weak,throughput}_frontier.png`. Node-count tick labels are thinned
  so 8192 and 9408 don't collide. The strong and throughput plots leave out the
  2.56G mesh at 512 and 1024 nodes (0.63M and 0.31M tris/GCD; 86.9% and 66.8%), so that
  fall-off has to be covered in the paper text (`DROP` in each script).

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

## Aurora (ALCF, PVC, 12 tiles/node, project mom6_anuga_gpu)

Budget: **30k node-hours**, spend carefully. Every point gets `--timeout 600`.
Queues: `debug` (1–2 nodes, 1 h), `debug-scaling` (2–256 nodes, 1 h), `prod`
(256+ nodes).

### Environment (verified 2026-10-06)

```bash
# default Aurora PE: oneapi/release/2026.1.0 (icx 2026.1.0), mpich/prd/5.0.0, no extra modules
cd standalone && make intelgpumpi MPICC=mpicc INTEL_AOT=pvc      # -> bin/bench_intelgpumpi
```

- `INTEL_AOT=pvc` (new Makefile switch) builds the device code ahead of time
  (`-fopenmp-targets=spir64_gen -Xopenmp-target-backend "-device pvc"`). The
  backend flag only takes effect at link. Without the switch the build stays
  JIT `spir64`, and JIT costs ~2 s per process at startup (see the A/B below).
  The AOT link reports one IGC `RetryManager` recompile, in
  `core_compute_fluxes_central`, which `--flux scatter` does not run.
- `ldd` is clean: mpich 5.0.0, libomptarget and libze_loader from the PE,
  libfabric 2.3.1. The halo is staged through host, so the run uses
  `MPIR_CVAR_ENABLE_GPU=0`.
- Launch line: `mpiexec -n 12N -ppn 12
  --cpu-bind=list:1-8:9-16:17-24:25-32:33-40:41-48:53-60:61-68:69-76:77-84:85-92:93-100
  gpu_tile_compact.sh`, with `ZE_FLAT_DEVICE_HIERARCHY=FLAT` and
  `OMP_NUM_THREADS=8`. Checked (`aurora_smoke.sh bind`): ranks 0–11 get
  `ZE_AFFINITY_MASK` 0–11 and 8 disjoint cores each, cores 0 and 52 (OS) are
  skipped, and ranks 0–5 sit on socket 0. Each rank sees 1 device.
- Login nodes compile and can run a 1-rank CPU smoke test
  (`OMP_TARGET_OFFLOAD=DISABLED`), but `mpiexec` does not work there at all.
  The user runs GPU tests in an interactive session
  (`qsub -I -l select=1 -l walltime=1:00:00 -l filesystems=home:flare -A mom6_anuga_gpu -q debug`).
- `scaling.py emit` used to write `#PBS -l wd` (NCI-only). The gadi session
  moved it into the gadi presets; until that is pushed, strip the line from
  Aurora job scripts by hand.

### 2026-10-06 — first contact

**On icx the loop construct makes no difference** (unlike amdclang). One tile,
1M triangles (`aurora_smoke.sh ab`):

| build                                  | ms/step | Mcell-steps/s | process wall | volume drift |
|----------------------------------------|---------|---------------|--------------|--------------|
| AOT, `teams loop`                      | 1.2805  | 786.9         | 1.83 s       | 0            |
| AOT, `teams distribute parallel for`   | 1.2821  | 785.9         | 1.49 s       | 0            |
| JIT, `teams distribute parallel for`   | 1.2813  | 786.4         | 3.54 s       | 0            |

- The Intel configs stay without `ANUGA_OMP_TEAMS_DPF`, the same as nvc.
- The phase split is the same in all three: fluxes 46%, extrapolate 36%.
- One PVC tile is ~1.9× one MI250X GCD at this size (1.28 vs 2.44 ms/step).

1-node smoke (12 tiles, `aurora_smoke.sh mpi`):

| tris/tile | ms/step | kernels | halo | dt   | Mcell-steps/s/node | GiB/tile | drift   | wall/point |
|-----------|---------|---------|------|------|--------------------|----------|---------|------------|
| 1M        | 1.542   | 1.378   | 0.16 | 0.10 | 7775               | 0.46     | 1.8e-14 | 2 s        |
| 40M       | 65.13   | 65.72   | 0.23 | 3.65 | 7371               | 18.2     | 1.1e-13 | 37 s       |

- One Aurora node is **2.55× one Frontier node** at 40M/device (7371 vs 2895
  Mcell-steps/s).
- Each tile delivers 614 Mcell-steps/s, against 787 for a lone tile at 1M. That
  ~20% gap is wider than Frontier's ~12%. It is not the halo (0.2 ms). The
  same single-device 40M check applies here.
- A 40M point takes ~37 s including setup, so weak points cost about 1 node-min
  per node.

### 2026-10-06 — weak scaling, 40M triangles/tile

Jobs 8905901 (1–8 nodes, 2 min, ~0.3 node-h), 8905911 (16 nodes, 41 s) and
8905920 (32–256 nodes, 2m47s, ~12 node-h), all on `debug-scaling`. Efficiency
= t(1 node) / t(N), with t(1) = 65.32 ms.

| nodes | tiles | triangles | ms/step | kernels | halo ms | dt ms | efficiency |
|-------|-------|-----------|---------|---------|---------|-------|------------|
| 1     | 12    | 0.48G     | 65.32   | 65.77   | 0.23    | 2.1   | 1.000      |
| 2     | 24    | 0.96G     | 66.69   | 67.61   | 0.30    | 5.6   | 0.979      |
| 4     | 48    | 1.92G     | 66.50   | 67.95   | 0.27    | 5.8   | 0.982      |
| 8     | 96    | 3.84G     | 67.30   | 69.99   | 0.31    | 7.4   | 0.971      |
| 16    | 192   | 7.68G     | 68.94   | 72.27   | 0.26    | 10.5  | 0.948      |
| 32    | 384   | 15.4G     | 67.87   | 69.66   | 0.25    | 8.6   | 0.962      |
| 64    | 768   | 30.7G     | 68.93   | 72.07   | 0.28    | 10.8  | 0.948      |
| 128   | 1536  | 61.4G     | 70.53   | 75.04   | 0.36    | 12.9  | 0.926      |
| 256   | 3072  | 122.9G    | 70.61   | 75.24   | 0.38    | 17.1  | **0.925**  |

- All points `ok`, no NaNs, drift ≤ 1.1e-13 (2.6e-16 at 256 nodes). Each
  point takes 38–48 s.
- 256 nodes (3072 tiles, 123G triangles) reach ~1.74 T cell-steps/s, 2.4× the
  Frontier 256-node point at 1.5× the mesh.
- The efficiency loss is not the halo (≤ 0.4 ms throughout). `dt` (waiting at
  the allreduce) climbs from 2 to 17 ms, which fits tile-to-tile speed
  variation: the slowest of N tiles gets slower as N grows. This is still a
  hypothesis. The `kernels` column cannot settle it: the CSV `kernel_ms` is
  the sum over phases of the per-phase max over ranks. That sum exceeds the
  step time (75.2 > 70.6 ms) and grows with N from noise alone. The per-rank
  stderr line (`rank r: ... ms/step kernels`) includes the MPI waits, so every
  rank shows the wall time (70.609 ± 0.001 ms). To test it, make each rank
  print its compute-only time and rerun 256-node weak with `--phases-sync`.
  Rank 0 at 256 nodes: fluxes 30.7, extrapolate 19.2, forcing+update 6.6,
  prepare 5.3 (compute ≈ 61.9 ms), dt_allreduce 8.5, halo 0.2.
- The 1-node point reproduces the interactive smoke run (65.13 vs 65.32 ms).

`debug-scaling` allows only one queued job per user (dependency holds do not
get around it), so 32–256 nodes ran as one 256-node job. Prepared, not yet
submitted: 512, 1024 and 2048 nodes (`prod`).

### In flight (submitted 2026-10-06)

| job     | nodes | queue         | what                                        | results in                                     |
|---------|-------|---------------|---------------------------------------------|------------------------------------------------|
| 8905967 | 2048  | large         | weak, 512 + 1024 + 2048                     | `benchmarks/results/aurora_n512-2048/`         |
| 8907425 | 128   | debug-scaling | strong large @128; strong small @8–128      | `aurora_strong/n128/large`, `aurora_strong/n8-128/small` |
| 8907420 | 256   | small         | strong large + small                        | `aurora_strong/n256/{large,small}`             |
| 8907421 | 512   | small         | strong large + small                        | `aurora_strong/n512/{large,small}`             |
| 8907422 | 1024  | small         | strong large + small                        | `aurora_strong/n1024/{large,small}`            |
| 8907423 | 2048  | large         | strong large + small                        | `aurora_strong/n2048/{large,small}`            |
| 8907424 | 4096  | large         | weak (1.97T tris) + strong large + small    | `aurora_n4096`, `aurora_strong/n4096/{large,small}` |

All paths are under `benchmarks/results/`; job scripts are in
`benchmarks/results/aurora_strong/jobs/`. Walltime is 20 min (30 min for the
8–128 and 4096 jobs).

Strong-scaling design (mirrors Frontier, extended to 4096 nodes):
- **Large / headline:** 122.9G triangles (nx = 15000576, the 256-node weak
  mesh), 128 → 4096 nodes = 80M → 2.5M triangles per tile. The 128-node
  reference uses 38 GiB of each tile's 64. The 256-node point should reproduce
  the weak result (70.61 ms).
- **Small / failure point:** 3.84G triangles (nx = 468768, the 8-node weak
  mesh), 8 → 4096 nodes = 40M → 78k triangles per tile (9.5 columns per rank at
  4096).
- Cost: ~500 node-h expected and ~2.7k worst case at walltime, for everything
  in the table, including 8905967.

Queue rules learned: `prod` needs ≥ 256 nodes (it routes 256–1024 to `small`
and ≥ 2000 to `large`); `debug-scaling` caps at 256 nodes and one queued job
per user; `large` allows 10 queued jobs per project.

To collect:

```bash
qstat -xf 8905967 8907420 8907421 8907422 8907423 8907424 8907425 | grep -E 'Job Id|job_state|resources_used.walltime|Exit_status'
cd benchmarks/results
for f in aurora_n512-2048 aurora_n4096 aurora_strong/n*/*; do echo "== $f"; tail -n +2 $f/scaling_aurora.csv; done
```

Then add the weak rows (efficiency vs 65.32 ms), build the strong tables
(efficiency vs the 128-node large and the 8-node small reference), and re-plot.

### Full-machine run (submitted 2026-10-06, requested by the Frontier session)

Sized with the Frontier recipe. At submission ~10.1k of 10,624 nodes were up
and not reserved, so the largest point is 9600 nodes.

| job     | nodes | walltime | what                                         |
|---------|-------|----------|----------------------------------------------|
| 8907628 | 2048  | 15 min   | strong, huge mesh (reference)                |
| 8907629 | 4096  | 15 min   | strong, huge mesh                            |
| 8907630 | 8192  | 20 min   | strong, huge mesh                            |
| 8907631 | 9600  | 30 min   | weak 40M/tile (4.61T tris) + strong, huge mesh |

- Huge mesh: nx = 235,929,600 = 32 × lcm(12 × {2048, 4096, 8192, 9600}),
  ny = 2048, so 1.933T triangles. That is 78.6M/tile at 2048 nodes and 16.8M/tile
  at 9600, all above the ~10M level where efficiency stays near-ideal.
- Results go to `benchmarks/results/aurora_full/n<N>/{huge,weak}`. Scripts are in
  `aurora_full/jobs/`.
- Cost: ~2.8k node-h expected, ~10k worst case at walltime.

### 2026-10-07 — strong scaling to 1024 nodes

Jobs 8907425 (8–128 nodes), 8907420 (256), 8907421 (512) and 8907422 (1024),
all `ok`, drift ≤ 1.6e-14. These use the same binary as the weak runs:
src/ is unchanged since 14e48919.

Large, 122.9G triangles (nx = 15000576). Efficiency is relative to 128 nodes.

| nodes | tiles | tris/tile | ms/step | halo ms | dt ms | efficiency |
|-------|-------|-----------|---------|---------|-------|------------|
| 128   | 1536  | 80M       | 135.69  | 0.27    | 20.6  | 1.000      |
| 256   | 3072  | 40M       | 69.53   | 0.26    | 12.8  | 0.976      |
| 512   | 6144  | 20M       | 34.61   | 0.26    | 8.4   | 0.980      |
| 1024  | 12288 | 10M       | 16.89   | 0.27    | 3.4   | **1.004**  |

- The 256-node point reproduces the weak result (69.53 vs 70.61 ms).
- 1024 nodes reach ~7.28 T cell-steps/s.

Small, 3.84G triangles (nx = 468768). Efficiency is relative to 8 nodes.

| nodes | tiles | tris/tile | ms/step | halo ms | dt ms | efficiency |
|-------|-------|-----------|---------|---------|-------|------------|
| 8     | 96    | 40M       | 68.51   | 0.26    | 9.9   | 1.000      |
| 16    | 192   | 20M       | 33.88   | 0.25    | 5.7   | 1.011      |
| 32    | 384   | 10M       | 16.40   | 0.28    | 2.2   | 1.044      |
| 64    | 768   | 5M        | 8.06    | 0.22    | 1.1   | 1.063      |
| 128   | 1536  | 2.5M      | 3.86    | 0.18    | 0.38  | **1.111**  |
| 256   | 3072  | 1.25M     | 2.07    | 0.18    | 0.27  | 1.036      |
| 512   | 6144  | 625k      | 1.13    | 0.18    | 0.21  | 0.944      |
| 1024  | 12288 | 312k      | 0.75    | 0.18    | 0.18  | 0.717      |

- Scaling is superlinear from 20M down to 1.25M triangles per tile. The cause
  has not been diagnosed: no cache counters were collected. Part of it is
  `dt` (the slowest-tile wait) shrinking faster than the work does.
- Efficiency falls below ~1M triangles per tile. At 312k, the fixed ~0.18 ms
  halo and ~0.18 ms dt are about half the step.
- All rows and the weak CSV were sent to the gadi session for the paper
  (2026-10-07). Still queued: 8905967, 8907423, 8907424 and 8907628–31.

---

## Gadi (NCI, V100 / H200)

H200 strong scaling, 400M triangles total: efficiency 0.976 at 2 nodes (50M
triangles/GPU), 0.963 at 4, 0.904 at 8 (12.5M/GPU); the 15-node point was still
queued. The gadi session owns this section.
