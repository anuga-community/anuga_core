#!/usr/bin/env python3
"""Run benchmark_distribute.py over a grid of (np, scheme) values and
print a consolidated summary table.

Each (np, scheme) combination is run once and its stdout saved to
<outdir>/bench_np<N>_<scheme>.txt.  Existing files are reused unless
--force is given, so partial grids can be resumed cheaply.

Usage
-----
    python scripts/run_benchmark_grid.py [options]

Options
-------
--size M         Mesh grid size M (4*M*M triangles, default 1000)
--reps R         Repetitions per function (default 1)
--interval S     Ticker sample interval in seconds (default 5.0)
--np LIST        Comma-separated process counts (default 10,20,30)
--schemes LIST   Comma-separated schemes (default metis,morton,hilbert)
--outdir DIR     Directory for per-run output files (default bench_results)
--mpirun CMD     MPI launcher (default mpirun)
--script PATH    Path to benchmark_distribute.py (default: auto-detect)
--force          Re-run even if output file already exists
--dry-run        Print commands without executing them
--no-summary     Skip summary; just run and save
"""

import argparse
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


# ── Argument parsing ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument('--size',     type=int,   default=1000)
    p.add_argument('--reps',     type=int,   default=1)
    p.add_argument('--interval', type=float, default=5.0)
    p.add_argument('--np',       type=str,   default='10,20,30')
    p.add_argument('--schemes',  type=str,   default='metis,morton,hilbert')
    p.add_argument('--outdir',   type=str,   default='bench_results')
    p.add_argument('--mpirun',   type=str,   default='mpirun')
    p.add_argument('--script',   type=str,   default=None)
    p.add_argument('--force',    action='store_true')
    p.add_argument('--dry-run',  action='store_true')
    p.add_argument('--no-summary', action='store_true')
    a = p.parse_args()
    a.np_values = [int(x) for x in a.np.split(',')]
    a.scheme_list = [s.strip() for s in a.schemes.split(',')]
    return a


def find_benchmark_script(explicit=None):
    if explicit:
        return explicit
    # Look relative to this script's directory
    here = Path(__file__).resolve().parent
    candidate = here / 'benchmark_distribute.py'
    if candidate.exists():
        return str(candidate)
    raise FileNotFoundError(
        'Cannot find benchmark_distribute.py; use --script to specify it.')


# ── Running ────────────────────────────────────────────────────────────────────

def output_path(outdir, np_val, scheme):
    return Path(outdir) / f'bench_np{np_val}_{scheme}.txt'


def run_one(mpirun, script, np_val, scheme, size, reps, interval,
            outfile, dry_run=False):
    cmd = [mpirun, '-np', str(np_val), 'python', script,
           '--size', str(size),
           '--reps', str(reps),
           '--interval', str(interval),
           '--scheme', scheme]
    print(f'  $ {" ".join(cmd)}')
    print(f'    → {outfile}')
    if dry_run:
        return True
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=3600)
        text = result.stdout
        if result.returncode != 0:
            text += f'\n[STDERR]\n{result.stderr}'
        Path(outfile).write_text(text)
        if result.returncode != 0:
            print(f'    [FAILED — exit code {result.returncode}]')
            return False
        return True
    except subprocess.TimeoutExpired:
        print('    [TIMED OUT]')
        return False
    except Exception as e:
        print(f'    [ERROR: {e}]')
        return False


# ── Parsing ────────────────────────────────────────────────────────────────────

def _first_float(pattern, text, group=1):
    m = re.search(pattern, text)
    return float(m.group(group)) if m else None


def _first_int(pattern, text, group=1):
    m = re.search(pattern, text)
    return int(m.group(group).replace(',', '')) if m else None


def parse_output(text):
    """Extract benchmark metrics from one run's stdout."""
    r = {}

    # Header fields
    r['triangles'] = _first_int(r'Triangles:\s+([\d,]+)', text)
    r['nodes']     = _first_int(r'Nodes:\s+([\d,]+)', text)
    r['ranks']     = _first_int(r'MPI ranks:\s+(\d+)', text)
    r['scheme']    = (re.search(r'Scheme:\s+(\S+)', text) or
                      type('', (), {'group': lambda s, n: 'metis'})()).group(1)
    r['shmem']     = 'yes' in (re.search(r'Shared mem:\s+(\S+)', text)
                                or type('', (), {'group': lambda s, n: ''})()
                                ).group(1).lower()

    # Main table — three columns: distribute / collaborative / dump+load
    wall = re.search(
        r'Wall time[^\d]+([\d.]+)s\s+([\d.]+)s\s+([\d.]+)s', text)
    if wall:
        r['wall_dist']   = float(wall.group(1))
        r['wall_collab'] = float(wall.group(2))
        r['wall_dl']     = float(wall.group(3))

    pss = re.search(
        r'Peak PSS[^\d]+([\d.]+) GiB\s+([\d.]+) GiB\s+([\d.]+) GiB', text)
    if pss:
        r['pss_dist']   = float(pss.group(1))
        r['pss_collab'] = float(pss.group(2))
        r['pss_dl']     = float(pss.group(3))

    rss = re.search(
        r'Peak RSS[^\d]+([\d.]+) GiB\s+([\d.]+) GiB\s+([\d.]+) GiB', text)
    if rss:
        r['rss_dist']   = float(rss.group(1))
        r['rss_collab'] = float(rss.group(2))
        r['rss_dl']     = float(rss.group(3))

    # Dump/load breakdown
    dl = re.search(r'dump \(serial\) ([\d.]+)s.*?load \(parallel\) ([\d.]+)s',
                   text)
    if dl:
        r['dump_time'] = float(dl.group(1))
        r['load_time'] = float(dl.group(2))

    # Ghost stats (from distribute() column — all three columns are identical)
    g_tot = re.search(r'Ghost total[^(]+\(([\d.]+)%\)', text)
    if g_tot:
        r['ghost_pct'] = float(g_tot.group(1))

    g_avg = re.search(r'Ghost avg per rank\s+([\d,]+)', text)
    if g_avg:
        r['ghost_avg'] = int(g_avg.group(1).replace(',', ''))

    g_mm = re.search(r'Ghost min.*?([\d,]+)\s*[–\-]\s*([\d,]+)', text)
    if g_mm:
        r['ghost_min'] = int(g_mm.group(1).replace(',', ''))
        r['ghost_max'] = int(g_mm.group(2).replace(',', ''))

    return r


# ── Formatting helpers ─────────────────────────────────────────────────────────

def _t(val, bold=False):
    """Format a time value in seconds."""
    if val is None:
        return '  —   '
    s = f'{val:6.2f}s'
    return f'*{s}*' if bold else f' {s} '


def _g(val):
    """Format a memory value in GiB."""
    if val is None:
        return '   —  '
    return f'{val:6.2f}'


def _pct(val):
    if val is None:
        return '  —  '
    return f'{val:5.1f}%'


def _num(val):
    if val is None:
        return '    —  '
    return f'{val:>8,}'


# ── Summary printer ────────────────────────────────────────────────────────────

def print_summary(results, args, ntri, outdir):
    """Print consolidated summary tables to stdout and save to summary.txt."""
    lines = []

    def out(*a, **kw):
        s = ' '.join(str(x) for x in a)
        lines.append(s)
        print(s, **kw)

    W = 72
    out('=' * W)
    out('  ANUGA distribute() benchmark grid summary')
    out(f'  Mesh:      synthetic {args.size}×{args.size}'
        f'  ({ntri:,} triangles)')
    out(f'  np values: {args.np_values}')
    out(f'  Schemes:   {args.scheme_list}')
    out(f'  Reps:      {args.reps}')
    out(f'  Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    out('=' * W)

    # ── Wall time table ────────────────────────────────────────────────────
    out()
    out('  Wall time (seconds)')
    out(f'  {"np":>4}  {"scheme":<8}  '
        f'{"distribute()":>13}  {"collaborative()":>15}  '
        f'{"dump+load":>10}  {"dump":>7}  {"load":>6}  '
        f'{"collab/dist":>11}')
    out(f'  {"─"*4}  {"─"*8}  {"─"*13}  {"─"*15}  '
        f'{"─"*10}  {"─"*7}  {"─"*6}  {"─"*11}')
    for np_val in args.np_values:
        for scheme in args.scheme_list:
            key = (np_val, scheme)
            if key not in results:
                out(f'  {np_val:>4}  {scheme:<8}  (no data)')
                continue
            r = results[key]
            wd  = r.get('wall_dist')
            wc  = r.get('wall_collab')
            wdl = r.get('wall_dl')
            dt  = r.get('dump_time')
            lt  = r.get('load_time')
            spd = (f'{wd/wc:.2f}×'
                   if wd and wc and wc > 0 else '  —  ')
            out(f'  {np_val:>4}  {scheme:<8}  '
                f'{_t(wd):>13}  {_t(wc):>15}  '
                f'{_t(wdl):>10}  {_t(dt):>7}  {_t(lt):>6}  '
                f'{spd:>11}')

    # ── Memory table ──────────────────────────────────────────────────────
    out()
    out('  Peak PSS — physical memory total across all ranks (GiB)')
    out(f'  {"np":>4}  {"scheme":<8}  '
        f'{"distribute()":>13}  {"collaborative()":>15}  '
        f'{"dump+load load-phase":>20}')
    out(f'  {"─"*4}  {"─"*8}  {"─"*13}  {"─"*15}  {"─"*20}')
    for np_val in args.np_values:
        for scheme in args.scheme_list:
            key = (np_val, scheme)
            if key not in results:
                continue
            r = results[key]
            out(f'  {np_val:>4}  {scheme:<8}  '
                f'{_g(r.get("pss_dist")):>13}  '
                f'{_g(r.get("pss_collab")):>15}  '
                f'{_g(r.get("pss_dl")):>20}')

    # ── Ghost triangle table ───────────────────────────────────────────────
    has_ghost = any('ghost_pct' in r
                    for r in results.values())
    if has_ghost:
        out()
        out('  Partition quality — ghost triangles')
        out(f'  {"np":>4}  {"scheme":<8}  '
            f'{"ghost %":>8}  {"avg/rank":>10}  {"min/rank":>10}  '
            f'{"max/rank":>10}  {"min–max ratio":>13}')
        out(f'  {"─"*4}  {"─"*8}  {"─"*8}  {"─"*10}  {"─"*10}  '
            f'{"─"*10}  {"─"*13}')
        for np_val in args.np_values:
            for scheme in args.scheme_list:
                key = (np_val, scheme)
                if key not in results:
                    continue
                r = results[key]
                pct  = r.get('ghost_pct')
                avg  = r.get('ghost_avg')
                gmin = r.get('ghost_min')
                gmax = r.get('ghost_max')
                ratio = (f'{gmax/gmin:.2f}×'
                         if gmin and gmax and gmin > 0 else '  —  ')
                out(f'  {np_val:>4}  {scheme:<8}  '
                    f'{_pct(pct):>8}  {_num(avg):>10}  '
                    f'{_num(gmin):>10}  {_num(gmax):>10}  '
                    f'{ratio:>13}')
    else:
        out()
        out('  (Ghost triangle stats not available — '
            'run with updated benchmark_distribute.py)')

    out()
    out('=' * W)

    # Save to file
    summary_path = Path(outdir) / 'summary.txt'
    summary_path.write_text('\n'.join(lines) + '\n')
    print(f'\n  Summary saved to {summary_path}')


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    script = find_benchmark_script(args.script)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f'\nBenchmark grid: np={args.np_values}  schemes={args.scheme_list}')
    print(f'Output directory: {outdir.resolve()}')
    print(f'Script: {script}\n')

    # Run all combinations
    ntri = 4 * args.size * args.size
    failed = []
    for np_val in args.np_values:
        for scheme in args.scheme_list:
            outfile = output_path(outdir, np_val, scheme)
            if outfile.exists() and not args.force:
                print(f'[SKIP] np={np_val} scheme={scheme}  '
                      f'(file exists; use --force to re-run)')
                continue
            print(f'\n[RUN] np={np_val}  scheme={scheme}')
            ok = run_one(args.mpirun, script, np_val, scheme,
                         args.size, args.reps, args.interval,
                         outfile, dry_run=args.dry_run)
            if not ok:
                failed.append((np_val, scheme))

    if args.dry_run or args.no_summary:
        return

    # Parse all output files and build results dict
    results = {}
    for np_val in args.np_values:
        for scheme in args.scheme_list:
            outfile = output_path(outdir, np_val, scheme)
            if not outfile.exists():
                continue
            text = outfile.read_text()
            r = parse_output(text)
            if r.get('wall_dist') is not None:
                results[(np_val, scheme)] = r
            else:
                print(f'  [WARN] Could not parse {outfile}')

    if not results:
        print('\nNo parseable results found.')
        return

    if failed:
        print(f'\nWARNING: {len(failed)} run(s) failed: {failed}')

    print()
    print_summary(results, args, ntri, outdir)


if __name__ == '__main__':
    main()
