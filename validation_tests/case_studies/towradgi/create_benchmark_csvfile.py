

import pstats
import os
import csv
import numpy as np


#=================================
# Collect timings
#=================================

def create_benchmark_csvfile(pstat_basename, openmp_threads, verbose=True):

    """
    Create a CSV file summarizing the benchmark timings from pstat files.
    Args:
        pstat_basename (str): The base name for the pstat files.
        openmp_threads (list): List of OpenMP thread counts to analyze.
    """

    output_csv = pstat_basename+'.csv'

    table_contents = []

    myfuncs = ['OMP_NUM_THREADS', 
    'total_time', 
    'evolve', 
    'evolve_one_rk2_step', 
    'compute_fluxes', 
    'distribute_to_vertices_and_edges', 
    'update_conserved_quantities', 
    'compute_forcing_terms', 
    'update_boundary',
    'backup_conserved_quantities',
    'saxpy_conserved_quantities',
    'apply_fractional_steps'
    ]

    column_header = ['THREADS', 
    'total_time', 
    'evolve', 
    'rk2_step', 
    'fluxes', 
    'distribute', 
    'update', 
    'forcing',
    'boundary',
    'backup',
    'saxpy', 
    'operators'
    ]

    table_contents.append(column_header)

    for threads in openmp_threads:

        pstat_file = pstat_basename+f'_omp_{threads}.pstat'

        print(f'Analysing file {pstat_file}')

        stats = pstats.Stats(pstat_file)

        benchmark_dict = {}

        for func_key, stat in stats.stats.items():
            filename, line, funcname = func_key
            for myfuncname in myfuncs:
                if funcname.endswith(myfuncname):
                    #print(f'{funcname}, {stat[3]:.3g}')
                    benchmark_dict[funcname] = stat[3]


        benchmark_dict['total_time']=stats.total_tt
        benchmark_dict['OMP_NUM_THREADS'] = threads


        #for key in myfuncs:
        #    print(f'{key} {benchmark_dict[key]:.3g}')

        table_line =[]

        for key in myfuncs:
            table_line.append(f'{benchmark_dict[key]:.3g}')

        table_contents.append(table_line)



    # Write the cumulative times to a CSV file
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(table_contents[0])  # Write the header
        for row in table_contents[1:]:
            writer.writerow(row)  # Write the data rows


    if verbose:
        print(f'Wrote benchmark results to {output_csv}')
        import pandas as pd

        df = pd.read_csv(f'{pstat_basename}.csv')

        print('')
        print(80 * '=')
        print(f'Benchmark results for {pstat_basename}')
        print(80 * '=')
        print('')
        print(df.to_string(index=False))  # Prints the entire DataFrame without row numbers
