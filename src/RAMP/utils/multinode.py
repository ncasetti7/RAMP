"""Module for running functions in parallel on a cluster"""

import os
import time
import pickle
import torch
from RAMP.utils import multiproc

def pickle_batch(batch):
    '''
    Pickle a batch of input dictionaries
    
    Returns:
        None
    '''
    with open("input.pickle", "wb") as f:
        pickle.dump(batch, f)

def write_submit(func_name, path_to_multinode, num_workers, calc=None):
    '''
    Write a submit script for a function to be run on a cluster

    Returns:
        None
    '''
    if calc is None:
        calc = " "
    with open("submit.sh", "w", encoding='utf-8') as f:
        f.write("#!/bin/bash\n")
        f.write("#SBATCH --job-name=batch\n")
        f.write("#SBATCH --output " + func_name + ".out\n")
        f.write("#SBATCH -N 1\n")
        f.write("#SBATCH -n " + str(num_workers) +"\n")
        f.write("#SBATCH --time=24:00:00\n")
        f.write("#SBATCH --mem 32GB\n")
        f.write("\n")
        f.write("python " + path_to_multinode + " " + func_name + " " + calc)

def parallel_run(func, input_list, num_workers, num_nodes, path_to_mechanism_search, calc=None):
    '''
    Submit jobs to run a function on each batch of input dictionaries in parallel and wait for and return the results

    Returns:
        results (list): list of results from the function
    '''
    # Batch the input list
    batched_dicts = multiproc.batch_dicts(input_list, num_nodes)

    # Write the path to the multinode script
    path_to_multinode = path_to_mechanism_search + "/src/RAMP/run_func.py"

    # For each batch, make an input directory, pickle the batch, write a submit script, and submit the job
    results = []
    for i, batch in enumerate(batched_dicts):
        os.system("mkdir batch_" + str(i))
        os.chdir("batch_" + str(i))
        pickle_batch(batch)
        write_submit(func.__name__, path_to_multinode, num_workers, calc)
        os.system("sbatch submit.sh")
        os.chdir("..")

    # Wait for all the jobs to finish and collect the results
    while True:
        jobs = os.popen("squeue -u ncasetti").read()
        if "batch" not in jobs:
            break
        time.sleep(10)

    for i in range(len(batched_dicts)):
        os.chdir("batch_" + str(i))
        # Check whether the output file exists
        if not os.path.exists("output.pickle"):
            print("Output file not found for batch " + str(i))
            os.chdir("..")
            continue
        with open("output.pickle", "rb") as f:
            results.extend(pickle.load(f))
        os.chdir("..")
        os.system("rm -r batch_" + str(i))

    return results
