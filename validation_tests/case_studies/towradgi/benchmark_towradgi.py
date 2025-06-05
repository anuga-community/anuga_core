

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
anuga_env = 'anuga_env_3.10'
openmp_threads = [1, 2, 4, 8, 16, 32, 48]
openmp_threads = [4,6]

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


