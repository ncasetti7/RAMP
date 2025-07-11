"""Module for running functions in parallel on a single node"""

import os
import functools
import torch.multiprocessing as mp
import torch
from RAMP.utils import multinode

def batch_dicts(dicts, num_workers):
    '''
    Batch a list of dictionaries into a list of lists of dictionaries, and add a batch number to each dictionary
    
    Args:
        dicts (list): list of dictionaries
        num_workers (int): number of workers

    Returns:
        batched_dicts (list): list of lists of dictionaries
    '''
    batch_size = int(len(dicts)/num_workers) + 1
    batched_dicts = []
    # Batch the dictionaries and add a batch number to each dictionary
    for i in range(num_workers):
        if (i+1)*batch_size > len(dicts):
            for d in dicts[i*batch_size:]:
                d['batch'] = i
            batched_dicts.append(dicts[i*batch_size:])
        else:
            for d in dicts[i*batch_size:(i+1)*batch_size]:
                d['batch'] = i
            batched_dicts.append(dicts[i*batch_size:(i+1)*batch_size])
    # Remove any empty lists
    batched_dicts = [x for x in batched_dicts if x != []]

    return batched_dicts

def run_func(func, input_list, queue):
    '''
    Run a function in parallel with a list of arguments and puts the results in a queue. Do this in a directory named from the batch number
    
    Args:
        func (function): function to run
        input_list (list): list of dictionaries with arguments for the function
        queue (mp.Queue): queue to put the results in

    Returns:
        None
    '''
    # Set the number of threads to 1 to avoid issues with torch multiprocessing
    torch.set_num_threads(1)

    # Make and cd into a batch folder to run calculations in
    batch = input_list[0]['batch']
    os.system("mkdir batch_" + str(batch))
    os.chdir("batch_" + str(batch))

    # Run the function on each input dictionary
    results = []
    for input_dict in input_list:
        try:
            result = func(input_dict)
        except Exception as e:
            print("Error in batch", batch, ":", e)
            continue
        results.append(result)

    # Change directory back to the original directory and remove the batch folder
    os.chdir("..")
    os.system("rm -r batch_" + str(batch))

    # Put the results in the queue
    final_dict = {input_list[0]['batch']: results}
    queue.put(final_dict)

def parallel_run(func, input_list, num_workers, num_nodes, path_to_mechanism_search, multi=False, calc=None):
    '''
    Run a function in parallel on several or one node(s)

    Args:
        func (function): function to run
        input_list (list): list of dictionaries with arguments for the function
        num_workers (int): number of workers
        num_nodes (int): number of nodes
        path_to_mechanism_search (str): path to the mechanism search directory
        multi (bool): whether to use multinode
        calc (str): path to the calculator

    Returns:
        results (list): list of results from the function
    '''
    if multi:
        return multinode.parallel_run(func, input_list, num_workers, num_nodes, path_to_mechanism_search, calc)
    return parallel_run_proc(func, input_list, num_workers, calc)

def parallel_run_proc(func, input_list, num_workers, calc=None):
    '''
    Run a function in parallel with a list of arguments
    
    Returns:
        results (list): list of results from the function
    '''
    # Batch the input list
    batched_dicts = batch_dicts(input_list, num_workers)

    # Check whether to use a calculator (see if it's None)
    if calc is not None:
        func = functools.partial(func, calc=calc)

    # Set up the queue and processes
    queue = mp.Queue()
    num_processes = len(batched_dicts)
    processes = []
    rets = []
    for i in range(num_processes):
        p = mp.Process(target=run_func, args=(func, batched_dicts[i], queue))
        p.start()
        processes.append(p)

    # Get the results from the queue
    for p in processes:
        ret = queue.get()
        rets.append(ret)
    for p in processes:
        p.join()

    # Sort the results
    new_rets = []
    for i in range(len(rets)):
        for j in range(len(rets)):
            if i == list(rets[j].keys())[0]:
                new_rets.append(rets[j])
                break

    results = []
    for ret in new_rets:
        results.extend(list(ret.values())[0])

    return results
