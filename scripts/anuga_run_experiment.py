#!/usr/bin/env python3

import pstats
import os
import csv
import shutil
import numpy as np

# This script runs benchmark ifor specified script with different numbers of OpenMP threads

from time import localtime, strftime, gmtime, sleep
time = strftime('%Y%m%d_%H%M', localtime())

import subprocess
import sys
import socket

import argparse


parser = argparse.ArgumentParser(
    description="Run experiment for given script."
)

parser.add_argument('-omp', '--openmp_threads', type=int, nargs=1, default=4,
                    help="OpenMP threads used for the experiment")

parser.add_argument(
    "script_file",
    type=str,
    nargs="?",
    default="run_small_towradgi.py",
    help="The Python script to run experiment (default: run_small_towradgi.py)"
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
openmp_threads = args.openmp_threads

if script_args == []:
    script_args_str = ""
else:
    script_args_str = " " + " ".join(script_args)

#print(f"Using script args: {script_args_str}")

print(f"Using script file: {script_file + script_args_str}")

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
        #openmp_threads = [1, 2, 4, 8, 16, 32, 48, 64, 80, 100]
    elif PBS_QUEUE == 'normal-exec':
        queue = 'normal'
        #openmp_threads = [1, 2, 4, 6, 8, 12, 16, 24, 32, 48]
else:
    queue = 'local'
    #openmp_threads = [1,2,4]

print(f"Using queue: {queue}")


env["OMP_NUM_THREADS"] = str(openmp_threads)  # Set to your desired number of threads
output_dir = f'../../OUTPUT/output_{script}_{hostname}_{queue}_{anuga_env}_{time}_omp_{openmp_threads}'

os.makedirs(output_dir, exist_ok=True)


destination_dir = output_dir

for item in os.listdir('.'):
    src_path = os.path.join('.', item)
    dst_path = os.path.join(destination_dir, item)
    if os.path.isdir(src_path):
        shutil.copytree(src_path, dst_path)
    else:
        shutil.copy2(src_path, dst_path)





#shutil.copy(script_file, output_dir)
os.chdir(output_dir)

cmd = ['conda', 'run', '--no-capture-output', 'python', '-u', script_file] + script_args

print('')
print(80 * '=')
print(f'Running command: {" ".join(cmd)}')
print(80 * '=')
print('')

# Run the subprocess with the modified environment
output_lines = []
with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, text=True, env=env) as process:
    for line in process.stdout:
        print(line, end='')  # Print each line as it arrives
        output_lines.append(line)

with open('output.txt', 'w') as f:
    f.writelines(output_lines)








