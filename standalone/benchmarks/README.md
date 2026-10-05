# Portable benchmark suite

Analytic cases only (no data files), so every machine runs the identical problem.
Definitions live in `suite.json`; `suite.py` builds, gates, runs and tabulates.

```bash
source ../env.sh                                   # site modules (compiler, CUDA/ROCm)
python3 benchmarks/suite.py --label h200           # autodetects nvidia/amd/intel/cpu, builds fresh
python3 benchmarks/suite.py --label mi250x --target amdgpu --sizes S,M,L
python3 benchmarks/suite.py --label h200 --bin bin/bench_gpu --benchmarks river,beach
python3 benchmarks/suite.py --label x --quick      # smoke test, ~10 s
python3 benchmarks/suite.py --label h200 --emit-job pbs   # job-script skeleton (or slurm)
```

Output in `benchmarks/results/<label>/`: `results.csv` (concatenate across devices),
`gates.csv`, `env.json`, `summary.md`, `logs/`.

| benchmark    | regime                                   | what it shows |
|--------------|------------------------------------------|---------------|
| `dam`        | every cell wet, fixed 200 steps          | peak throughput, hero scaling |
| `river`      | reservoir into dry valley, run to t=20 s | active set wins (~20% active) |
| `beach`      | tsunami run-up, headland coast, t=60 s   | moving wet/dry front, mid-range active set |
| `river_rain` | uniform rain, t=20 s                     | active set worst case (100% active) |

Sizes: S=0.16M, M=2M, L=7.8M, XL=24M triangles (or any integer nx).

Design notes
- **Fixed work, not fixed steps**: dry/wet cases run to a fixed simulated time (`--tend`),
  so full vs active-set runs do identical physics and the active fraction is averaged
  over the same event. Compare `wall_s`, not `ms_per_step`.
- **Gates are physical**, not bitwise: scatter fluxes use atomics, so even two identical
  runs differ at ~1e-6 after thousands of steps. Gates: lake-at-rest momentum, mass drift,
  NaN count, and full-vs-active state agreement in `summary.md`.
- The suite builds into a private object dir per label. Reusing `build/gpu` once gave a
  stale `setup.o` that silently ran the wrong case.

## Scaling studies (Frontier / Aurora / Perlmutter / gadi)

`scaling.py` + `scaling.json` (machine presets) + `analyze_scaling.py`. One rank per GPU,
1-D slab decomposition, wet dam break, dx = 1 m, fixed step count (so ms/step is the metric).

```bash
python3 benchmarks/scaling.py plan --machine frontier --study both --max-nodes 1000   # sizes + memory fit
python3 benchmarks/scaling.py emit --machine frontier --study both --max-nodes 1000 --walltime 04:00:00
python3 benchmarks/scaling.py emit --machine aurora --study weak --max-nodes 1000 --per-size
# build: make CONFIG=amdgpumpi MPICC=cc AMD_ARCH=gfx90a   |  CONFIG=intelgpumpi MPICC=mpicc  |  CONFIG=gpumpi
python3 benchmarks/analyze_scaling.py benchmarks/results/scaling_*.csv --out figs/
```

- **Weak**: ~40M triangles/rank on Frontier/Aurora (`weak_tris_per_rank`), columns = k*ranks.
- **Strong**: fixed global mesh sized so the largest run keeps 64 columns/rank (ghost fraction ~3%);
  smallest node count comes from device/host memory (Frontier: 8 nodes, 4.2G triangles).
- Only `gadi_h200` / `generic` launch lines have been exercised. Frontier, Aurora and
  Perlmutter presets (launcher, binding, modules, make args) are best-effort: expect to
  tune them on first contact with the machine, ideally on 1-2 nodes.
