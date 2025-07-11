"""Module to call the QM calculations from the config file"""

import sys
import os
import time
import yaml
from RAMP.qm_calcs.orca_ts_opt import run_orca_calcs

def write_to_log(message, file, first=False):
    '''
    Write a message to a log file

    Args:
        message (str): message to write to the log file
        file (str): path to the log file
        first (bool): if True, write the message to the log file for the first time

    Returns:
        None
    '''
    if first:
        mode = 'w'
    else:
        mode = 'a'
    with open(file, mode, encoding='utf-8') as f:
        f.write(message + "\n")
        f.flush()

def qm_calcs(config_file):
    '''
    Runs QM calculations on the transition states predicted by the neural network

    Args:
        config_file (str): path to the config file

    Returns:
        None
    '''
    stream = open(config_file, 'r', encoding='utf-8')
    config_input = yaml.load(stream, Loader=yaml.Loader)

    # Make a folder to save transition states in the results folder
    folder_name = config_input['launcher_args']['path_to_mechanism_search'] + "/results/" + config_file[16:-5]
    ts_folder = folder_name+"/"+config_input['nnp']['ts_folder']
    orca_ts_folder = folder_name + "/" + config_input['qm_calcs']['ts_folder']
    if os.path.isdir(orca_ts_folder) is False:
        os.mkdir(orca_ts_folder)

    # Log some initial information
    log_file = folder_name+"/"+config_input['qm_calcs']['log_output']
    t = time.localtime()
    start = time.time()
    current_time = time.strftime("%H:%M:%S", t)
    write_to_log("Job started at " + str(current_time), log_file, first=True)

    # Include template file in slurm_args_dict
    config_input['qm_slurm_args']['template_slurm_script'] = config_input['launcher_args']['path_to_mechanism_search'] + '/' + config_input['slurm_args']['template_slurm_script']

    # Run QM calculations
    run_orca_calcs(orca_path=config_input['qm_calcs']['orca_path'],
                   infile=folder_name+"/"+config_input['nnp']['barrier_preds_output'],
                   outfile=folder_name+"/"+config_input['qm_calcs']['smiles_output'],
                   equilibrium_cache_file=folder_name+"/../"+config_input['qm_calcs']['equilibrium_cache_file'],
                   ts_folder=ts_folder,
                   orca_ts_folder=orca_ts_folder,
                   canon_reactant=config_input['data']['input_structure'],
                   slurm_args_dict=config_input['qm_slurm_args'],
                   use_barrier_cutoff=config_input['filters']['use_barrier_cutoff'],
                   barrier_cutoff=config_input['filters']['barrier_cutoff'],
                   use_top_k_cutoff=config_input['filters']['use_top_k_cutoff'],
                   top_k_cutoff=config_input['filters']['top_k'])

    # Log some final information
    t2 = time.localtime()
    current_time2 = time.strftime("%H:%M:%S", t2)
    write_to_log("Analysis complete! Job complete at " + str(current_time2), log_file)
    end = time.time()
    write_to_log("Elapsed Time: " + str(round((end - start)/60, 2)) + " minutes", log_file)

if __name__ == "__main__":
    config = sys.argv[1]
    qm_calcs(config)
