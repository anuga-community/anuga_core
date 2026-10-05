#!/usr/bin/env python3
"""Weak / strong scaling driver for the ANUGA shallow-water miniapp.

The problem is the wet dam break on the 1-D slab decomposition (one rank per
GPU, one ghost column each side, neighbour-only halo + one dt allreduce per
step).  dx is 1 m everywhere, so a larger mesh is a longer domain, never a
different physics problem, and every rank does a fixed number of identical
steps: wall time per step is the whole metric.

    # what would run, and does it fit in memory?
    python3 benchmarks/scaling.py plan --machine frontier --study both --max-nodes 1000

    # write batch scripts (one 1000-node allocation, or one job per size)
    python3 benchmarks/scaling.py emit --machine frontier --study both --max-nodes 1000 --walltime 04:00:00
    python3 benchmarks/scaling.py emit --machine aurora   --study weak --max-nodes 1000 --per-size

    # inside an allocation (what the emitted scripts call)
    python3 benchmarks/scaling.py run  --machine frontier --study weak --nodes 1,2,4,8 --bin build/frontier/bench

    # laptop/gadi check of the harness itself: 2-4 local ranks, tiny meshes
    python3 benchmarks/scaling.py run --machine generic --study weak --nodes 1,2,4 --tpn 1 \\
            --tris-per-rank 400000 --ny 256 --bin bin/bench_gpumpi

Results append to benchmarks/results/scaling_<machine>.csv; analyze_scaling.py
turns them into figures.  Weak: ideal is flat ms/step.  Strong: ideal is
ms/step ~ 1/nodes; the halo fraction explains the departure.
"""
import argparse
import csv
import json
import math
import os
import shlex
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

DEV_B_PER_TRI = 512      # measured: flat across three decades (README)
HOST_B_PER_TRI = 600     # host_peak_bytes / triangles on the generated mesh

FIELDS = ["machine", "study", "nodes", "ranks", "nx", "ny", "triangles",
          "tris_per_rank", "steps", "status", "ms_per_step", "kernel_ms",
          "halo_ms", "dt_ms", "mcellsteps_per_s", "volume_drift", "nans",
          "dev_gib_per_rank", "build_s", "note"]


def load():
    with open(os.path.join(HERE, "scaling.json")) as f:
        return json.load(f)


def node_list(a):
    if a.nodes:
        return sorted({int(x) for x in a.nodes.split(",") if x})
    n, out = 1, []
    while n < a.max_nodes:
        out.append(n)
        n *= 2
    out.append(a.max_nodes)
    return out


def build_jobs(cfg, a):
    m = cfg["machines"][a.machine]
    tpn = a.tpn or m["gpus_per_node"]
    nodes = node_list(a)
    studies = ["weak", "strong"] if a.study == "both" else [a.study]
    jobs, notes = [], []
    for st in studies:
        sc = cfg["studies"][st]
        ny = a.ny or sc["ny"]
        steps = a.steps or sc["steps"]
        if st == "weak":
            tpr = a.tris_per_rank or m["weak_tris_per_rank"]
            k = max(2, round(tpr / (4 * ny)))
            for n in nodes:
                r = n * tpn
                jobs.append(dict(study=st, nodes=n, ranks=r, nx=k * r, ny=ny, steps=steps))
        else:
            maxr = max(nodes) * tpn
            gt = a.global_tris or m.get("strong_global_tris")
            if gt:
                nx = max(1, round(gt / (4 * ny)))
            else:
                nx = sc["min_cols_per_rank"] * maxr
            T = 4 * nx * ny
            cap_dev = 0.85 * m["gpu_mem_gib"] * 2**30 / DEV_B_PER_TRI
            cap_host = 0.80 * m["host_mem_gib"] * 2**30 / HOST_B_PER_TRI / tpn
            cap = min(cap_dev, cap_host)
            for n in nodes:
                r = n * tpn
                if T / r > cap:
                    notes.append("strong: %d nodes skipped (%.0fM tris/rank > %.0fM that fits)"
                                 % (n, T / r / 1e6, cap / 1e6))
                    continue
                if nx / r < 4:
                    notes.append("strong: %d nodes skipped (<4 columns/rank)" % n)
                    continue
                jobs.append(dict(study=st, nodes=n, ranks=r, nx=nx, ny=ny, steps=steps))
    return m, tpn, jobs, notes


def bench_args(cfg, j):
    return (["--nx", str(j["nx"]), "--ny", str(j["ny"]),
             "--lenx", str(j["nx"]), "--leny", str(j["ny"]),
             "--steps", str(j["steps"])] + cfg["common_args"])


def cmd_plan(cfg, a):
    m, tpn, jobs, notes = build_jobs(cfg, a)
    print("machine %s: %d ranks/node, %g GiB device, %g GiB host"
          % (a.machine, tpn, m["gpu_mem_gib"], m["host_mem_gib"]))
    print("\n%-7s %6s %7s %10s %7s %14s %12s %10s %10s"
          % ("study", "nodes", "ranks", "nx", "ny", "triangles", "tris/rank", "cols/rank", "dev GiB"))
    for j in jobs:
        T = 4 * j["nx"] * j["ny"]
        print("%-7s %6d %7d %10d %7d %14.4g %12.4g %10.1f %10.1f"
              % (j["study"], j["nodes"], j["ranks"], j["nx"], j["ny"], T, T / j["ranks"],
                 j["nx"] / j["ranks"], T / j["ranks"] * DEV_B_PER_TRI / 2**30))
    for n in notes:
        print("  note:", n)
    return 0


def cmd_run(cfg, a):
    m, tpn, jobs, notes = build_jobs(cfg, a)
    for n in notes:
        print("[scaling] note:", n)
    binp = os.path.abspath(a.bin)
    if not os.access(binp, os.X_OK):
        sys.exit("scaling: need an executable MPI binary (--bin), e.g. one built with "
                 "make CONFIG=gpumpi / amdgpumpi / intelgpumpi")
    launcher = a.launcher or m["launcher"]
    env = dict(os.environ)
    env.update(m.get("env", {}))
    out_csv = os.path.join(a.out, "scaling_%s.csv" % a.machine)
    os.makedirs(os.path.join(a.out, "scaling_logs"), exist_ok=True)
    fresh = not os.path.exists(out_csv)
    row_csv = os.path.join(a.out, ".scaling_row_%d.csv" % os.getpid())
    with open(out_csv, "a", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=FIELDS)
        if fresh:
            w.writeheader()
        for j in jobs:
            lau = shlex.split(launcher.format(nodes=j["nodes"], ranks=j["ranks"], tpn=tpn))
            cmd = lau + [binp] + bench_args(cfg, j) + ["--csv", row_csv]
            if os.path.exists(row_csv):
                os.remove(row_csv)
            tag = "%s_%s_n%d" % (a.machine, j["study"], j["nodes"])
            t0 = time.time()
            print("[scaling] %-6s nodes=%-5d ranks=%-6d nx=%d ny=%d ..."
                  % (j["study"], j["nodes"], j["ranks"], j["nx"], j["ny"]), flush=True)
            try:
                p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   universal_newlines=True, env=env, timeout=a.timeout)
                rc, out = p.returncode, p.stdout
            except subprocess.TimeoutExpired as e:
                rc, out = 124, "TIMEOUT\n" + (e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or ""))
            with open(os.path.join(a.out, "scaling_logs", tag + ".log"), "w") as f:
                f.write("$ " + " ".join(cmd) + "\n" + out)
            rec = {k: "" for k in FIELDS}
            T = 4 * j["nx"] * j["ny"]
            rec.update(machine=a.machine, study=j["study"], nodes=j["nodes"], ranks=j["ranks"],
                       nx=j["nx"], ny=j["ny"], triangles=T,
                       tris_per_rank=T // j["ranks"], steps=j["steps"])
            if rc == 0 and os.path.exists(row_csv):
                with open(row_csv, newline="") as f:
                    r = list(csv.DictReader(f))[-1]
                bad = float(r["nans"]) != 0 or abs(float(r["volume_drift"])) > 1e-9
                rec.update(status="BAD_PHYSICS" if bad else "ok",
                           ms_per_step=r["ms_per_step"], kernel_ms=r["kernel_ms"],
                           halo_ms=r["halo_ms"], dt_ms=r["dt_ms"],
                           mcellsteps_per_s=r["mcellsteps_per_s"],
                           volume_drift=r["volume_drift"], nans=r["nans"],
                           dev_gib_per_rank="%.2f" % (float(r["dev_bytes"]) / 2**30),
                           build_s=r["build_s"])
                print("          %.3f ms/step (kernels %.3f, halo %.3f, dt %.3f)  %s  [%.0f s]"
                      % (float(r["ms_per_step"]), float(r["kernel_ms"]), float(r["halo_ms"]),
                         float(r["dt_ms"]), rec["status"], time.time() - t0), flush=True)
            else:
                tail = [l for l in out.strip().splitlines() if l.strip()][-1:] or [""]
                rec.update(status="FAILED", note="rc=%d %s" % (rc, tail[0][:150]))
                print("          FAILED rc=%d  %s" % (rc, tail[0][:100]), flush=True)
            w.writerow(rec)
            fo.flush()
    if os.path.exists(row_csv):
        os.remove(row_csv)
    return 0


def cmd_emit(cfg, a):
    m, tpn, jobs, notes = build_jobs(cfg, a)
    for n in notes:
        print("# note:", n, file=sys.stderr)
    sched = m["scheduler"]
    outdir = os.path.join(a.out, "jobs")
    os.makedirs(outdir, exist_ok=True)
    binp = a.bin or os.path.join("build", a.machine, "bench")
    groups = []
    if a.per_size:
        for st in sorted({j["study"] for j in jobs}):
            for n in sorted({j["nodes"] for j in jobs if j["study"] == st}):
                groups.append((st, [n]))
    else:
        for st in sorted({j["study"] for j in jobs}):
            groups.append((st, sorted({j["nodes"] for j in jobs if j["study"] == st})))
    for st, ns in groups:
        N = max(ns)
        name = "scale_%s_%s_%s" % (a.machine, st, ("n%d" % N) if a.per_size else "all")
        L = ["#!/bin/bash"]
        gpus = N * tpn
        if sched == "slurm":
            L += ["#SBATCH -J %s" % name, "#SBATCH -N %d" % N,
                  "#SBATCH -t %s" % a.walltime, "#SBATCH -o %s.%%j.log" % name]
            L += ["#SBATCH " + x for x in m.get("sbatch", [])]
        elif sched == "pbs":
            res = m.get("pbs_resources", "-l select={nodes}").format(
                nodes=N, gpus=gpus, cpus=12 * gpus,
                mem=int(m["host_mem_gib"] / m["gpus_per_node"] * gpus))
            L += ["#PBS -N %s" % name[:15], "#PBS " + res,
                  "#PBS -l walltime=%s" % a.walltime, "#PBS -l wd", "#PBS -j oe"]
            L += ["#PBS " + x for x in m.get("pbs", [])]
        L += ["", "set -x", "cd %s" % ROOT] + m.get("setup", ["# " + m.get("modules_hint", "load compiler + MPI modules")]) + [""]
        if not a.bin:
            mk = " ".join(m["make"])
            L += ["mkdir -p build/%s" % a.machine,
                  "[ -x %s ] || make --no-print-directory %s OBJDIR=build/%s/obj BIN=%s build"
                  % (binp, mk, a.machine, binp), ""]
        for k, v in m.get("env", {}).items():
            L.append("export %s=%s" % (k, v))
        j0 = next(j for j in jobs if j["study"] == st)
        extra = ""
        if st == "strong":      # every job in a strong study shares ONE global mesh
            extra = " --global-tris %d" % (4 * j0["nx"] * j0["ny"])
        L.append("python3 benchmarks/scaling.py run --machine %s --study %s --nodes %s --bin %s"
                 " --ny %d --tpn %d%s%s"
                 % (a.machine, st, ",".join(map(str, ns)), binp, j0["ny"], tpn, extra,
                    " --tris-per-rank %d" % a.tris_per_rank if a.tris_per_rank else ""))
        path = os.path.join(outdir, name + (".sl" if sched == "slurm" else ".pbs"))
        with open(path, "w") as f:
            f.write("\n".join(L) + "\n")
        os.chmod(path, 0o755)
        print("wrote", path)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["plan", "emit", "run"])
    ap.add_argument("--machine", required=True)
    ap.add_argument("--study", choices=["weak", "strong", "both"], default="both")
    ap.add_argument("--nodes", help="explicit comma list (default: powers of 2 up to --max-nodes)")
    ap.add_argument("--max-nodes", type=int, default=1000)
    ap.add_argument("--tpn", type=int, help="ranks per node (default: GPUs per node)")
    ap.add_argument("--ny", type=int)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--tris-per-rank", type=int, help="weak: override the machine preset")
    ap.add_argument("--global-tris", type=int,
                    help="strong: fixed global size (default: min_cols_per_rank * largest rank count)")
    ap.add_argument("--bin")
    ap.add_argument("--launcher", help="override the machine's launch line template")
    ap.add_argument("--walltime", default="04:00:00")
    ap.add_argument("--per-size", action="store_true", help="emit: one job per node count")
    ap.add_argument("--timeout", type=float, default=3600, help="run: seconds per job")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    a = ap.parse_args()
    cfg = load()
    if a.machine not in cfg["machines"]:
        sys.exit("unknown machine; have: " + ", ".join(cfg["machines"]))
    if a.cmd == "run" and not a.bin:
        sys.exit("run needs --bin")
    return {"plan": cmd_plan, "emit": cmd_emit, "run": cmd_run}[a.cmd](cfg, a)


if __name__ == "__main__":
    sys.exit(main())
