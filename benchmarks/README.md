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
