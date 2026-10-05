#!/usr/bin/env python3
"""Portable benchmark suite driver for the ANUGA shallow-water miniapp.

One command per machine: build the right binary, run the correctness gates,
sweep every benchmark in suite.json over a size ladder, and leave behind tidy
CSVs that concatenate across devices.  Python >= 3.6, standard library only,
no PBS/Slurm/MPI assumptions -- wrap it in whatever scheduler the site has
(see --emit-job).

    # on any node, after loading that site's compiler/CUDA/ROCm modules:
    python3 benchmarks/suite.py --label h200

    python3 benchmarks/suite.py --label mi250x --target amdgpu --sizes S,M,L
    python3 benchmarks/suite.py --label laptop  --target cpu    --quick
    python3 benchmarks/suite.py --label h200 --bin bin/bench_gpu --benchmarks river,beach

Outputs (benchmarks/results/<label>/):
    results.csv   one row per run          -- cat results/*/results.csv across devices
    gates.csv     correctness gates
    env.json      host, device, compiler, git revision, command line
    summary.md    wall-time table, active-set speedups, validity verdict
    logs/         full stdout of every run
"""

import argparse
import csv
import datetime
import json
import os
import platform
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                      # .../standalone

RESULT_FIELDS = [
    "label", "benchmark", "variant", "size", "nx", "triangles", "status",
    "wall_s", "ms_per_step", "steps", "sim_time_s", "sim_rate",
    "mcellsteps_per_s", "active_pct", "gflops", "dev_gib",
    "volume_drift", "max_momentum", "nans", "note",
]

TARGET_BIN = {
    "gpu": "bench_gpu", "cpu": "bench_cpu", "ompcpu": "bench_ompcpu",
    "amdgpu": "bench_amdgpu", "intelgpu": "bench_intelgpu",
    "clanggpu": "bench_clanggpu",
}


# ---------------------------------------------------------------- helpers
def sh(cmd, timeout=None, cwd=None):
    """Run a command, return (rc, combined output).  Never raises."""
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           universal_newlines=True, timeout=timeout, cwd=cwd)
        return p.returncode, p.stdout
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        return 124, out + "\n[suite] TIMEOUT"
    except OSError as e:
        return 127, str(e)


def have(tool):
    return shutil.which(tool) is not None


def detect_target():
    if have("nvidia-smi") and sh(["nvidia-smi", "-L"])[0] == 0:
        return "gpu"
    if have("rocm-smi") or have("rocminfo"):
        return "amdgpu"
    if have("xpu-smi") or have("sycl-ls"):
        return "intelgpu"
    return "cpu"


def parse_sizes(spec, table):
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok in table:
            out.append((tok, int(table[tok])))
        else:
            try:
                out.append((tok, int(tok)))
            except ValueError:
                sys.exit("suite: unknown size '%s' (use %s or an integer nx)"
                         % (tok, "/".join(table)))
    return out


def device_info():
    info = {}
    if have("nvidia-smi"):
        rc, o = sh(["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap",
                    "--format=csv,noheader"])
        if rc == 0:
            info["nvidia"] = o.strip().splitlines()
    if have("rocm-smi"):
        rc, o = sh(["rocm-smi", "--showproductname"])
        if rc == 0:
            info["rocm"] = o.strip().splitlines()[:12]
    if have("sycl-ls"):
        rc, o = sh(["sycl-ls"])
        if rc == 0:
            info["sycl"] = o.strip().splitlines()
    if have("lscpu"):
        rc, o = sh(["lscpu"])
        if rc == 0:
            for line in o.splitlines():
                if line.startswith(("Model name", "Socket(s)", "Core(s) per socket", "CPU(s):")):
                    k, v = line.split(":", 1)
                    info.setdefault("cpu", {})[k.strip()] = v.strip()
    return info


def git_rev():
    rc, o = sh(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT)
    rev = o.strip() if rc == 0 else "unknown"
    rc, o = sh(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT)
    return rev + ("+dirty" if rc == 0 and o.strip() else "")


def read_last_csv_row(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else None


def fnum(row, key, default=float("nan")):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return default


# ---------------------------------------------------------------- build
def build(target, label, out_dir, extra_make):
    objdir = os.path.join(out_dir, "obj")
    binp = os.path.join(out_dir, TARGET_BIN.get(target, "bench_" + target))
    # A private object dir per label: a stale object from another checkout,
    # compiler or arch silently gives the wrong binary (this has bitten us).
    shutil.rmtree(objdir, ignore_errors=True)
    cmd = ["make", "--no-print-directory", "-C", ROOT, "CONFIG=" + target,
           "OBJDIR=" + objdir, "BIN=" + binp, "build"] + extra_make
    print("[suite] building:", " ".join(cmd), flush=True)
    rc, out = sh(cmd)
    with open(os.path.join(out_dir, "build.log"), "w") as f:
        f.write(out)
    if rc != 0 or not os.path.exists(binp):
        print(out[-2000:])
        sys.exit("suite: build failed (see %s/build.log)" % out_dir)
    return binp


# ---------------------------------------------------------------- one run
def run_bench(binp, launcher, args, max_wall, log_path, csv_path):
    if os.path.exists(csv_path):
        os.remove(csv_path)
    cmd = launcher + [binp] + args + ["--csv", csv_path]
    t0 = time.time()
    rc, out = sh(cmd, timeout=max_wall + 600)
    with open(log_path, "w") as f:
        f.write("$ " + " ".join(cmd) + "\n" + out)
    row = read_last_csv_row(csv_path) if (rc == 0 and os.path.exists(csv_path)) else None
    note = ""
    if rc != 0:
        tail = [l for l in out.strip().splitlines() if l.strip()][-1:] or [""]
        note = "rc=%d %s" % (rc, tail[0][:120])
    return row, note, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", help="device tag (default: hostname)")
    ap.add_argument("--target", default="auto",
                    help="auto | gpu | cpu | amdgpu | intelgpu | clanggpu | ompcpu | none "
                         "(none: use --bin as given)")
    ap.add_argument("--bin", help="prebuilt benchmark binary (implies --target none)")
    ap.add_argument("--make-args", default="",
                    help="extra make arguments, e.g. 'GPU_ARCH=cc80 MPICC=cc'")
    ap.add_argument("--config", default=os.path.join(HERE, "suite.json"))
    ap.add_argument("--sizes", default="S,M,L", help="comma list of S/M/L/XL or integer nx")
    ap.add_argument("--benchmarks", default="", help="comma list (default: all in suite.json)")
    ap.add_argument("--variants", default="", help="restrict to e.g. 'full' or 'active'")
    ap.add_argument("--max-wall", type=float, default=600.0,
                    help="seconds of timed loop per run before it is cut short (default 600)")
    ap.add_argument("--launcher", default="",
                    help="prefix for every run, e.g. 'mpirun -np 4' or 'srun -n 4' "
                         "(use with a *mpi binary)")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--quick", action="store_true",
                    help="smoke test: size S only, short runs")
    ap.add_argument("--skip-gates", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print the commands, run nothing")
    ap.add_argument("--emit-job", choices=["pbs", "slurm"],
                    help="print a job-script skeleton wrapping this invocation and exit")
    a = ap.parse_args()

    label = a.label or platform.node().split(".")[0]

    if a.emit_job:
        emit_job(a.emit_job, label, sys.argv[1:])
        return 0

    with open(a.config) as f:
        cfg = json.load(f)
    common = cfg.get("common_args", [])
    sizes = parse_sizes("S" if a.quick else a.sizes, cfg["sizes"])
    wanted = [b for b in a.benchmarks.split(",") if b]
    benches = [b for b in cfg["benchmarks"] if not wanted or b["name"] in wanted]
    if wanted and len(benches) != len(wanted):
        sys.exit("suite: unknown benchmark in --benchmarks (have: %s)"
                 % ", ".join(b["name"] for b in cfg["benchmarks"]))
    only_variants = [v for v in a.variants.split(",") if v]
    launcher = a.launcher.split()

    out_dir = os.path.abspath(os.path.join(a.out, label))
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)

    # ---- binary --------------------------------------------------------
    target = "none" if a.bin else (detect_target() if a.target == "auto" else a.target)
    if a.dry_run and target != "none":
        binp = os.path.join(out_dir, TARGET_BIN.get(target, "bench_" + target))
        print("[suite] would build target '%s' -> %s" % (target, binp))
    elif target == "none":
        binp = os.path.abspath(a.bin) if a.bin else None
        if not binp or not os.access(binp, os.X_OK):
            sys.exit("suite: --target none needs an executable --bin")
    else:
        binp = build(target, label, out_dir, a.make_args.split())
    print("[suite] label=%s target=%s bin=%s" % (label, target, binp), flush=True)

    # ---- environment record -------------------------------------------
    env = {
        "label": label, "target": target, "binary": binp,
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "host": platform.node(), "platform": platform.platform(),
        "git": git_rev(), "argv": sys.argv, "launcher": a.launcher,
        "device": device_info(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "compiler": {c: sh([c, "--version"])[1].splitlines()[0]
                     for c in ("nvc", "gcc", "amdclang", "icx") if have(c)},
    }
    if not a.dry_run:
        with open(os.path.join(out_dir, "env.json"), "w") as f:
            json.dump(env, f, indent=2)

    tmp_csv = os.path.join(out_dir, ".row.csv")
    results = []

    def run(name, variant, size, nx, args, stop):
        stop_args = (["--steps", str(stop["steps"])] if "steps" in stop
                     else ["--steps", "1000000000", "--tend", str(stop["tend"])])
        full = ["--nx", str(nx), "--ny", str(nx)] + common + args + stop_args \
               + ["--max-wall", str(a.max_wall)]
        if a.quick and "steps" in stop:
            full[full.index("--steps") + 1] = "20"
        if a.dry_run:
            print("  " + " ".join(launcher + [str(binp)] + full))
            return None
        log = os.path.join(out_dir, "logs", "%s_%s_%s.log" % (name, variant, size))
        row, note, wall = run_bench(binp, launcher, full, a.max_wall, log, tmp_csv)
        return row, note

    # ---- gates ---------------------------------------------------------
    gate_ok = True
    gates = []
    if not a.skip_gates and not a.dry_run:
        print("\n=== gates ===", flush=True)
        g = cfg["gates"]
        row, note = run("gate", "lake", "S", 200,
                        ["--case", "lake"], {"steps": 100})
        if row is None:
            ok, val = False, note
        else:
            val = fnum(row, "max_momentum")
            ok = val <= g["lake_at_rest_max_momentum"] and fnum(row, "nans", 1) == 0
        gates.append(("lake-at-rest", val, g["lake_at_rest_max_momentum"], ok))
        print("  lake-at-rest   max|momentum| = %s  (want <= %g)  %s"
              % (val, g["lake_at_rest_max_momentum"], "PASS" if ok else "FAIL"))
        gate_ok = ok
        with open(os.path.join(out_dir, "gates.csv"), "w") as f:
            f.write("label,gate,value,limit,pass\n")
            for n, v, lim, ok in gates:
                f.write("%s,%s,%s,%g,%d\n" % (label, n, v, lim, ok))

    # ---- benchmarks ----------------------------------------------------
    print("\n=== benchmarks ===", flush=True)
    for b in benches:
        variants = [(v, args) for v, args in b["variants"].items()
                    if not only_variants or v in only_variants]
        dead = False          # ascending sizes: stop a benchmark at its first failure
        print("\n%s -- %s" % (b["name"], b["title"]), flush=True)
        for size, nx in sizes:
            if dead and not a.dry_run:
                break
            for vname, vargs in variants:
                r = run(b["name"], vname, size, nx, b["args"] + vargs, b["stop"])
                if a.dry_run:
                    continue
                row, note = r
                rec = {k: "" for k in RESULT_FIELDS}
                rec.update(label=label, benchmark=b["name"], variant=vname,
                           size=size, nx=nx, triangles=4 * nx * nx)
                if row is None:
                    rec.update(status="FAILED", note=note)
                    dead = True
                    print("  %-3s %-7s  FAILED  %s" % (size, vname, note))
                else:
                    steps = fnum(row, "steps")
                    ms = fnum(row, "ms_per_step")
                    sim = fnum(row, "t_end", fnum(row, "sim_time"))   # absolute time incl. warmup
                    status = "ok"
                    notes = []
                    if "tend" in b["stop"] and sim < 0.999 * b["stop"]["tend"]:
                        status = "TRUNCATED"
                        notes.append("hit --max-wall at t=%.2f of %.2f" % (sim, b["stop"]["tend"]))
                    if fnum(row, "nans", 1) != 0:
                        status = "NAN"
                    if b.get("conserves_volume") and abs(fnum(row, "volume_drift")) > cfg["gates"]["volume_drift"]:
                        status = "MASS"
                        notes.append("volume drift %.2e" % fnum(row, "volume_drift"))
                    rec.update(status=status, wall_s="%.4f" % (ms * steps / 1e3),
                               ms_per_step=row["ms_per_step"], steps=int(steps),
                               sim_time_s="%.4f" % fnum(row, "sim_time"), sim_rate=row.get("sim_rate", ""),
                               mcellsteps_per_s=row["mcellsteps_per_s"],
                               active_pct=row.get("active_pct", ""), gflops=row["gflops"],
                               dev_gib="%.3f" % (fnum(row, "dev_bytes") / 2**30),
                               volume_drift=row["volume_drift"],
                               max_momentum=row["max_momentum"], nans=row["nans"],
                               note="; ".join(notes))
                    print("  %-3s %-7s %10d tris  %9.3f ms/step  %6d steps  %8.2f s wall  active %5s%%  %s"
                          % (size, vname, 4 * nx * nx, ms, steps, ms * steps / 1e3,
                             rec["active_pct"], status), flush=True)
                results.append(rec)
                write_results(out_dir, results)

    if a.dry_run:
        return 0
    if os.path.exists(tmp_csv):
        os.remove(tmp_csv)
    write_summary(out_dir, label, results, gates, gate_ok, benches)
    print("\n[suite] results in", out_dir)
    return 0 if gate_ok and all(r["status"] == "ok" for r in results) else 1


def write_results(out_dir, results):
    with open(os.path.join(out_dir, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        w.writeheader()
        w.writerows(results)


def write_summary(out_dir, label, results, gates, gate_ok, benches):
    L = ["# Suite summary: %s" % label, ""]
    L.append("Gates: %s" % ("all PASS" if gate_ok else "**FAILED -- timings below are INVALID**"))
    for n, v, lim, ok in gates:
        L.append("- %s: %s (limit %g) %s" % (n, v, lim, "PASS" if ok else "FAIL"))
    L.append("")
    bad = [r for r in results if r["status"] != "ok"]
    if bad:
        L.append("Non-ok runs: " + ", ".join("%s/%s/%s=%s" % (r["benchmark"], r["variant"], r["size"], r["status"])
                                              for r in bad))
        L.append("")
    for b in benches:
        rs = [r for r in results if r["benchmark"] == b["name"]]
        if not rs:
            continue
        L.append("## %s -- %s" % (b["name"], b["title"]))
        L.append("_%s_" % b["regime"])
        L.append("")
        has_active = "active" in b["variants"] and "full" in b["variants"]
        hdr = "| size | triangles | full wall (s) | full ms/step | steps |"
        sep = "|---|---|---|---|---|"
        if has_active:
            hdr += " active wall (s) | active % | speedup |"
            sep += "---|---|---|"
        L += [hdr, sep]
        for size in dict.fromkeys(r["size"] for r in rs):
            f_ = next((r for r in rs if r["size"] == size and r["variant"] == "full"), None)
            a_ = next((r for r in rs if r["size"] == size and r["variant"] == "active"), None)
            if f_ is None:
                continue
            line = "| %s | %s | %s | %s | %s |" % (size, f_["triangles"], f_["wall_s"] or "-",
                                                 f_["ms_per_step"] or "-", f_["steps"] or "-")
            if has_active:
                if a_ and a_["wall_s"] and f_["wall_s"]:
                    sp = float(f_["wall_s"]) / float(a_["wall_s"])
                    agree = ""
                    try:
                        m1, m2 = float(f_["max_momentum"]), float(a_["max_momentum"])
                        if abs(m1 - m2) > 1e-3 * max(abs(m1), 1e-30):
                            agree = " **(state differs!)**"
                    except ValueError:
                        pass
                    line += " %s | %s | %.2fx%s |" % (a_["wall_s"], a_["active_pct"], sp, agree)
                else:
                    line += " - | - | - |"
            L.append(line)
        L.append("")
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n" + "\n".join(L))


def emit_job(kind, label, argv):
    inner = " ".join(x for x in argv if x not in ("--emit-job", kind))
    if kind == "pbs":
        print("""#!/bin/bash
#PBS -N anuga_suite_%(label)s
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=64GB
#PBS -l walltime=02:00:00
#PBS -l wd
#PBS -j oe
# Fill in -q / -P / storage for your site, and load compiler/CUDA modules below.
set -x
cd %(root)s
# source env.sh          # site modules (nvhpc + cuda on gadi)
python3 benchmarks/suite.py %(inner)s""" % dict(label=label, root=ROOT, inner=inner))
    else:
        print("""#!/bin/bash
#SBATCH -J anuga_suite_%(label)s
#SBATCH -N 1 -n 1 --gpus-per-task=1 -c 16
#SBATCH -t 02:00:00
#SBATCH -o suite_%(label)s.%%j.log
# Fill in -A / -C / -q for your site, and load compiler/CUDA modules below.
set -x
cd %(root)s
# module load PrgEnv-nvidia cudatoolkit
python3 benchmarks/suite.py %(inner)s""" % dict(label=label, root=ROOT, inner=inner))


if __name__ == "__main__":
    sys.exit(main())
