#!/usr/bin/env python3

import pstats
import os
import csv
import shutil
import numpy as np
import subprocess
import sys
import socket
import argparse

from pathlib import Path

# This script runs the specified script, saves output to a timestamped file.
#
# Usage example:
# python anuga_run.py -omp 4 run_small_towradgi.py arg1 arg2 ...
#
# -omp: number of OpenMP threads

from time import localtime, strftime, gmtime, sleep
time = strftime('%Y%m%d_%H%M', localtime())


parser = argparse.ArgumentParser(
    description="Run given script."
)

parser.add_argument('-omp', '--openmp_threads', type=int, nargs=1, default=[4],
                    help="OpenMP threads used for the experiment")


parser.add_argument(
    "script_file",
    type=str,
    nargs="?",
    default="run_small_towradgi.py",
    help="The Python script to run (default: run_small_towradgi.py)"
)

parser.add_argument(
    "script_args",
    type=str, 
    default=" ",
    nargs=argparse.REMAINDER,
    help="Arguments for the experimental script")

args = parser.parse_args()
script_file = args.script_file
script_args = args.script_args
openmp_threads = args.openmp_threads[0]

print(f"Using OpenMP threads: {openmp_threads}")



if script_args == []:
    script_args_str = ""
else:
    script_args_str = " " + " ".join(script_args)

#print(f"Using script args: {script_args_str}")

print(f"Using script file: {script_file + script_args_str}")

script_file_base = script_file.rsplit('.', 1)[0]
print(f"Using script base: {script_file_base}")

abs_script_file = os.path.abspath(script_file)

#print(f"Using project root: {abs_script_file}")
parts = os.path.normpath(abs_script_file).split(os.sep)
abs_script_file = '_'.join(parts[-4:])
abs_script_file_base = abs_script_file.rsplit('.', 1)[0]
print(f"Using absolute script file: {abs_script_file}")


# Copy the current environment and set OMP_NUM_THREADS
env = os.environ.copy()
env["OMP_NUM_THREADS"] = str(openmp_threads)

# Define the Conda environment name
conda_prefix = os.environ.get("CONDA_PREFIX")
if conda_prefix:
    anuga_env = os.path.basename(conda_prefix)
    print(f"Conda environment name: {anuga_env}")
else:
    print("Not running inside a conda environment.")

output_filename = f'output_{script_file_base}_{time}.txt'

cmd = ['conda', 'run', '--no-capture-output', 'python', '-u', script_file] + script_args

print('')
print(80 * '=')
print(f'Running command: {" ".join(cmd)}')
print(80 * '=')
print('')

# Run the subprocess with the modified environment
with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, text=True, env=env) as process, \
      open(output_filename, 'w', buffering=1) as f:
    for line in process.stdout:
        print(line, end='')   # print to console immediately
        f.write(line)         # write each line to file immediately
        f.flush()             # flush the file buffer
    process.wait()









