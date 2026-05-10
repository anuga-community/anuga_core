"""
run_auto_validation_tests.py

Runs two groups of validation tests:

1. validate_*.py  — simulation + accuracy checks (analytical / experimental)
   Run directly as  python validate_*.py -alg DE1 -np 1

2. test_regression_*.py  — pytest-regressions baseline checks (behaviour_only)
   Run via  pytest test_regression_*.py
   On first run use --force-regen to generate the baseline .csv files.
"""

import os
import sys
import time
import subprocess
import argparse

import anuga
anuga_args = anuga.get_args()


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

dirs_to_skip = ['.']       # avoid infinite recursion
dirs_to_skip += ['patong'] # requires downloaded data, takes many hours

# (dirpath, filename, runner)  where runner is 'python' or 'pytest'
all_tests = []

for dirpath, dirnames, filenames in os.walk('.'):

    if '.svn' in dirnames:
        dirnames.remove('.svn')

    dirname = os.path.split(dirpath)[-1]
    if dirname in dirs_to_skip:
        continue

    for filename in sorted(filenames):
        if filename.startswith('validate_') and filename.endswith('.py'):
            all_tests.append((dirpath, filename, 'python'))
        elif filename.startswith('test_regression_') and filename.endswith('.py'):
            all_tests.append((dirpath, filename, 'pytest'))

all_tests.sort()

# Separate the two groups for display
validate_tests   = [(p, f) for p, f, r in all_tests if r == 'python']
regression_tests = [(p, f) for p, f, r in all_tests if r == 'pytest']


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

print()
print(80*'=')
print('Running all validation tests - some may take many minutes')
print('and some may require memory in the order of 8-16GB       ')
print(80*'=')

print('Simulation + accuracy tests (validate_*.py):')
for path, filename in validate_tests:
    print('    ', os.path.join(path, filename))

if regression_tests:
    print()
    print('Regression baseline tests (test_regression_*.py):')
    for path, filename in regression_tests:
        print('    ', os.path.join(path, filename))

print()

t0 = time.time()
parentdir = os.getcwd()

# Results: list of (label, returncode, elapsed, runner)
results = []


# ---------------------------------------------------------------------------
# Run validate_*.py tests
# ---------------------------------------------------------------------------

for path, filename in validate_tests:
    label = os.path.join(path, filename)
    os.chdir(path)

    cmd = [sys.executable, filename, '-alg', str(anuga_args.alg), '-np', str(anuga_args.np)]
    if anuga_args.verbose:
        cmd.append('-v')

    print()
    print(80*'=')
    print(' '.join(cmd))
    print(80*'=')
    sys.stdout.flush()

    t_start = time.time()
    proc = subprocess.run(cmd)
    elapsed = time.time() - t_start

    results.append((label, proc.returncode, elapsed, 'python'))
    status = 'PASSED' if proc.returncode == 0 else f'FAILED (exit code {proc.returncode})'
    print(f'  --> {status}  ({elapsed:.1f} s)')

    os.chdir(parentdir)


# ---------------------------------------------------------------------------
# Run test_regression_*.py tests via pytest
# ---------------------------------------------------------------------------

for path, filename in regression_tests:
    label = os.path.join(path, filename)
    os.chdir(path)

    cmd = ['pytest', filename, '-v']
    if anuga_args.verbose:
        cmd.append('-s')

    print()
    print(80*'=')
    print(' '.join(cmd))
    print(80*'=')
    sys.stdout.flush()

    t_start = time.time()
    proc = subprocess.run(cmd)
    elapsed = time.time() - t_start

    results.append((label, proc.returncode, elapsed, 'pytest'))
    status = 'PASSED' if proc.returncode == 0 else f'FAILED (exit code {proc.returncode})'
    print(f'  --> {status}  ({elapsed:.1f} s)')

    os.chdir(parentdir)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

total_elapsed = time.time() - t0
n_passed = sum(1 for _, rc, _, _ in results if rc == 0)
n_failed = len(results) - n_passed

print()
print(80*'=')
print(f'VALIDATION SUMMARY  ({total_elapsed:.1f} s total)')
print(80*'=')

col = 56
print(f"  {'Test':<{col}} {'Runner':>6}  {'Result':>12}  {'Time':>8}")
print(f"  {'-'*col}  {'-'*6}  {'-'*12}  {'-'*8}")
for label, rc, elapsed, runner in results:
    status = 'PASSED' if rc == 0 else f'FAILED ({rc})'
    marker = '  ' if rc == 0 else '* '
    print(f"{marker}{label:<{col}} {runner:>6}  {status:>12}  {elapsed:>7.1f}s")

print()
if n_failed == 0:
    print(f'All {n_passed} validation tests PASSED.')
else:
    print(f'{n_passed} PASSED,  {n_failed} FAILED.')
    print()
    print('Failed tests:')
    for label, rc, _, runner in results:
        if rc != 0:
            print(f'  * [{runner}] {label}  (exit code {rc})')

print()
if regression_tests:
    print('To regenerate regression baselines after an intentional change:')
    print('  cd validation_tests/behaviour_only')
    print('  pytest test_regression_*.py --force-regen')

print(80*'=')

sys.exit(0 if n_failed == 0 else 1)
