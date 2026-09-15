#!/usr/bin/env python3
"""
ANUGA Per-Kernel Benchmark Comparison Tool
--------------------------------------------
Compares two JSON results files produced by run_kernel_benchmarks.py and
prints a side-by-side delta table, keyed by (kernel, mode). Use this to
compare two compilers (e.g. GCC vs ICX) on the same mesh size, or the same
compiler before/after a tuning change — see claude/PLAN_compiler_tuning.md
Phase 3.

Usage
-----
    python benchmarks/compare_kernel_benchmarks.py gcc.json icx.json

    # Show only changes > 5%
    python benchmarks/compare_kernel_benchmarks.py gcc.json icx.json --threshold 5

    # List available result files
    python benchmarks/compare_kernel_benchmarks.py --list
"""

import argparse
import json
import os
import sys


def _load(path):
    with open(path) as fh:
        return json.load(fh)


def _pct(old, new):
    if old == 0:
        return None
    return (new - old) / old * 100.0


def _fmt_pct(pct):
    if pct is None:
        return '   n/a'
    return f'{pct:+.1f}%'


def _arrow(pct):
    """Lower mean_us is better."""
    if pct is None or abs(pct) < 0.5:
        return ' '
    return '↑' if pct < 0 else '↓'


def _key(r):
    return (r['kernel'], r['mode'])


def compare(before, after, threshold=0.0):
    b_results = {_key(r): r for r in before['results']}
    a_results = {_key(r): r for r in after['results']}

    all_keys = sorted(set(b_results) | set(a_results))

    print('\nKernel benchmark comparison')
    print(f"  Before : {before['git']['commit']}  branch={before['git']['branch']}  "
          f"CC={before['env'].get('cc_env', 'unset')}  t={before['timestamp']}")
    print(f"  After  : {after['git']['commit']}  branch={after['git']['branch']}  "
          f"CC={after['env'].get('cc_env', 'unset')}  t={after['timestamp']}")
    print()

    col = '{:<24}  {:>4}  {:>10}  {:>10}  {:>9}  {:>13}  {:>9}'
    hdr = col.format('Kernel', 'mode', 'before(us)', 'after(us)', 'Δtime',
                      'cells/s after', 'Δspeed')
    rule = '-' * len(hdr)
    print(hdr)
    print(rule)

    any_printed = False
    for key in all_keys:
        b = b_results.get(key)
        a = a_results.get(key)
        name = f'{key[0]} (mode {key[1]})'

        if b is None:
            print(f'  {name:<24}  (only in after)')
            any_printed = True
            continue
        if a is None:
            print(f'  {name:<24}  (only in before)')
            any_printed = True
            continue

        d_time = _pct(b['mean_us'], a['mean_us'])
        d_speed = _pct(b['cells_per_s'], a['cells_per_s'])

        if threshold > 0 and (d_time is None or abs(d_time) < threshold):
            continue

        arrow = _arrow(d_time)

        print(col.format(
            key[0], key[1],
            f"{b['mean_us']:.2f}",
            f"{a['mean_us']:.2f}",
            f'{_fmt_pct(d_time)}{arrow}',
            f"{a['cells_per_s']:,.0f}",
            _fmt_pct(d_speed),
        ))
        any_printed = True

    if not any_printed:
        print(f'  (no changes exceed {threshold}% threshold)')

    print(rule)
    print()


def _list_results(results_dir):
    if not os.path.isdir(results_dir):
        print(f'No results directory at {results_dir}')
        return
    files = sorted(f for f in os.listdir(results_dir)
                    if f.startswith('kernels_') and f.endswith('.json'))
    if not files:
        print(f'No kernel result files in {results_dir}')
        return
    print(f'Kernel result files in {results_dir}:')
    for f in files:
        print(f'  {f}')


def main():
    parser = argparse.ArgumentParser(
        description='Compare two ANUGA per-kernel benchmark result files.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('before', nargs='?', help='Baseline JSON results file')
    parser.add_argument('after', nargs='?', help='Comparison JSON results file')
    parser.add_argument('--threshold', type=float, default=0.0,
                         help='Only show rows where |Δtime| exceeds this percent')
    parser.add_argument('--list', action='store_true',
                         help='List available kernel result files and exit')
    args = parser.parse_args()

    results_dir = os.path.join(os.path.dirname(__file__), 'results')

    if args.list:
        _list_results(results_dir)
        return 0

    if not args.before or not args.after:
        parser.error('before and after JSON paths are required (or use --list)')

    compare(_load(args.before), _load(args.after), threshold=args.threshold)
    return 0


if __name__ == '__main__':
    sys.exit(main())
