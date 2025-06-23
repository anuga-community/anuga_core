

import pstats
import os
import csv
import numpy as np

# This script runs the Towradgi benchmark with different numbers of OpenMP threads

from time import localtime, strftime, gmtime, sleep
time = strftime('%Y%m%d_%H%M', localtime())

import subprocess
import sys
import socket


# Copy the current environment and set OMP_NUM_THREADS
env = os.environ.copy()

hostname = socket.gethostname()

# Define the Conda environment name
conda_prefix = os.environ.get("CONDA_PREFIX")
if conda_prefix:
    anuga_env = os.path.basename(conda_prefix)
    print(f"Conda environment name: {anuga_env}")
else:
    print("Not running inside a conda environment.")

PBS_QUEUE=normalsr-exec


if 'PBS_QUEUE' in os.environ:
    PBS_QUEUE = os.environ['PBS_QUEUE']
    print(f"Using PBS queue: {PBS_QUEUE}")
    if PBS_QUEUE == 'normalsr-exec':
        openmp_threads = [1, 2, 4, 8, 16, 32, 48, 64, 80, 100]
    elif PBS_QUEUE == 'normal-exec':
        openmp_threads = [1, 2, 4, 6, 8, 12, 16, 24, 32, 48]
else:
    openmp_threads = [2,4]

for threads in openmp_threads:
    env["OMP_NUM_THREADS"] = str(threads)  # Set to your desired number of threads
    pstat_file = f'profile_{hostname}_{anuga_env}_{time}_omp_{threads}.pstat'

    cmd = ['conda', 'run', '--no-capture-output', '-n', anuga_env, 'python', '-u', '-m', 'cProfile', '-o', pstat_file, 'run_small_towradgi.py']
    
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
pstat_basename = f'profile_{hostname}_{anuga_env}_{time}'

from create_benchmark_csvfile import create_benchmark_csvfile

create_benchmark_csvfile(pstat_basename, openmp_threads, verbose=True)


