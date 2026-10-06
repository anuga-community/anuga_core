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

Jobs 5624934 (1–8 nodes, 3m51s) and 5624969 (8–256 nodes, 5m59s, ~26 node-h).
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
| 256   | 2048 | 81.9G     | 114.4   | 1.25    | 14.8  | **0.962**  |

- All points `ok`, no NaNs, volume drift ≤ 1.7e-13. 18.2 GiB/GCD.
- Efficiency settles near 96% by 16 nodes and stays flat from 16 to 256 nodes.
  The halo stays at ~1 ms. Kernels are ~99% of the step.
- Run-to-run noise is about ±2%: the 8-node point was measured in both jobs.
- 256 nodes reach ~716 G cell-steps/s.
- Plot: `python3 standalone/logbook/plot_weak_frontier.py out.png` (reads every
  `benchmarks/results/frontier_n*/scaling_frontier.csv`).

### In flight (submitted 2026-10-05, not yet run)

| job      | nodes | what                                       | results in                                     |
|----------|-------|--------------------------------------------|------------------------------------------------|
| 5625186  | 1024  | weak, 512 + 1024 nodes                     | `benchmarks/results/frontier_n1024/`           |
| 5625208  | 8     | strong small (2.56G)                       | `benchmarks/results/frontier_strong/n8/small/` |
| 5625209  | 16    | strong small                               | `.../frontier_strong/n16/small/`               |
| 5625210  | 32    | strong small                               | `.../frontier_strong/n32/small/`               |
| 5625211  | 64    | strong small                               | `.../frontier_strong/n64/small/`               |
| 5625212  | 128   | strong large (81.9G) + small               | `.../frontier_strong/n128/{large,small}/`      |
| 5625213  | 256   | strong large + small                       | `.../frontier_strong/n256/{large,small}/`      |
| 5625214  | 512   | strong large + small                       | `.../frontier_strong/n512/{large,small}/`      |
| 5625215  | 1024  | strong large + small                       | `.../frontier_strong/n1024/{large,small}/`     |

Job scripts: `benchmarks/results/frontier_strong/jobs/strong_n*.sl` and
`benchmarks/results/frontier_n1024/jobs/`.

Strong-scaling design:
- **Large / headline:** 81.9G triangles (nx = 10000384, the same mesh as the
  256-node weak point) from 128 to 1024 nodes, i.e. 80M down to 10M triangles
  per GCD. The reference is 128 nodes (38 GiB/GCD; host ~380 GB/node, which is
  tight but fits). The 256-node point should reproduce the weak result (114.4 ms).
  Expect ≳ 90% at 1024 nodes.
- **Small / failure point:** 2.56G triangles (nx = 312512, the same mesh as the
  8-node weak point) from 8 to 1024 nodes, i.e. 40M down to 0.31M per GCD. Expect
  it to fall off around 64–128 nodes.
- Rule of thumb from the weak data: kernels cost ~2.76 ms per M triangles per
  GCD; halo (~1 ms) and dt (a few ms) are roughly fixed. So efficiency holds
  while each GCD has well over ~5M triangles.
- Estimated cost: ~130 node-h expected, ~480 node-h worst case.

To collect when done:

```bash
cd standalone/benchmarks/results
sacct -j 5625186,5625208,5625209,5625210,5625211,5625212,5625213,5625214,5625215 -X -o JobID,JobName%14,State,Elapsed,NNodes
cat frontier_n1024/scaling_frontier.csv
for f in frontier_strong/n*/*/scaling_frontier.csv; do echo "== $f"; tail -n +2 $f; done
```

Then add the rows to the tables above and re-plot. The weak plot picks up
`frontier_n1024` automatically; the strong plot still needs writing.

### Caveats for the paper

- Weak scaling is a best case by construction: 1-D slab, halo size independent
  of rank count, uniform all-wet work.
- `scaling.py` flags `BAD_PHYSICS` when volume drift exceeds 1e-9. At 1e11
  triangles, summation roundoff could trip that gate; check before trusting it.
  So far drift *decreases* with size (7.9e-16 at 256 nodes).
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
