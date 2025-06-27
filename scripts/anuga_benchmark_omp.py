#!/usr/bin/env python3

import pstats
import os
import csv
import numpy as np

# This script runs benchmark ifor specified script with different numbers of OpenMP threads

from time import localtime, strftime, gmtime, sleep
time = strftime('%Y%m%d_%H%M', localtime())

import subprocess
import sys
import socket

import argparse


parser = argparse.ArgumentParser(
    description="Run benchmark for given script."
)
parser.add_argument(
    "script_file",
    type=str,
    nargs="?",
    default="run_small_towradgi.py",
    help="The Python script to run for benchmarking (default: run_small_towradgi.py)"
)


args = parser.parse_args()
script_file = args.script_file
print(f"Using script file: {script_file}")

script = script_file.rsplit('.', 1)[0]



# Copy the current environment and set OMP_NUM_THREADS
env = os.environ.copy()

hostname = socket.gethostname()
hostname = hostname.split('.')[0]  # Get the hostname without the domain part
hostname = hostname.split('-')[0]  # Get the first part of the hostname if it contains a hyphen

print(f"On machine: {hostname}")

# Define the Conda environment name
conda_prefix = os.environ.get("CONDA_PREFIX")
if conda_prefix:
    anuga_env = os.path.basename(conda_prefix)
    print(f"Conda environment name: {anuga_env}")
else:
    print("Not running inside a conda environment.")


if 'PBS_QUEUE' in os.environ:
    PBS_QUEUE = os.environ['PBS_QUEUE']
    print(f"Using PBS queue: {PBS_QUEUE}")
    if PBS_QUEUE == 'normalsr-exec':
        queue = 'normalsr'
        openmp_threads = [1, 2, 4, 8, 16, 32, 48, 64, 80, 100]
    elif PBS_QUEUE == 'normal-exec':
        queue = 'normal'
        openmp_threads = [1, 2, 4, 6, 8, 12, 16, 24, 32, 48]
else:
    queue = 'local'
    openmp_threads = [1,2,4]

print(f"Using queue: {queue}")

for threads in openmp_threads:
    env["OMP_NUM_THREADS"] = str(threads)  # Set to your desired number of threads
    pstat_file = f'profile_{script}_{hostname}_{queue}_{anuga_env}_{time}_omp_{threads}.pstat'

    cmd = ['conda', 'run', '--no-capture-output', '-n', anuga_env, 'python', '-u', '-m', 'cProfile', '-o', pstat_file, script_file]
    
    print('')
    print(80 * '=')
    print(f'Running command: {" ".join(cmd)}')
    print(80 * '=')
    print('')

    # Run the subprocess with the modified environment
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, text=True, env=env) as process:
        for line in process.stdout:
            print(line, end='')  # Print each line as it arrives




#=================================
# Collect timings
#=================================
pstat_basename = f'profile_{script}_{hostname}_{queue}_{anuga_env}_{time}'

from anuga.utilities.create_benchmark_csvfile import create_benchmark_csvfile

create_benchmark_csvfile(pstat_basename, openmp_threads, verbose=True)


