"""Module to submit QM calculations to slurm from a config file"""

import sys
import os
import yaml
from RAMP.utils.file_write import write_slurm_script

def submit_slurm(config_file, config_input):
    '''
    Submit the QM calculation manager to slurm

    Args:
        config_file (str): path to the config file
        config_input (dict): dictionary of input parameters

    Returns:
        None
    '''
    run_scripts_folder = config_input['launcher_args']['path_to_mechanism_search'] + "/run_scripts/"
    parent_folder = run_scripts_folder + config_file[10:-15]
    slurm_folder = run_scripts_folder + "slurm_scripts/" + config_file[10:-15]
    os.chdir(parent_folder)
    job_name = config_file[-14:-5] + "_qm"
    job_folder = parent_folder + "/" + job_name
    if os.path.isdir(job_folder) is False:
        os.mkdir(job_folder)
    os.chdir(job_folder)
    slurm_arg_dict = config_input['qm_slurm_args']
    slurm_arg_dict['job_name'] = job_name
    slurm_arg_dict['output'] = job_name
    slurm_arg_dict['num_cpus'] = 1
    slurm_arg_dict['command'] = "python " + config_input['launcher_args']['path_to_mechanism_search'] + "/src/RAMP/run_qm_calcs.py ../../" + config_file + " >> " + job_name + ".out"
    write_slurm_script(config_input['launcher_args']['path_to_mechanism_search'] + '/' + config_input['slurm_args']['template_slurm_script'], slurm_arg_dict, slurm_folder + "/" + job_name + ".sh")
    #with open(slurm_folder + "/" + job_name + ".sh", 'w') as f:
    #    f.write("#!/bin/bash\n")
    #    f.write("#SBATCH --job-name=" + job_name + "\n")
    #    f.write("#SBATCH --output " + job_name + ".out\n")
    #    f.write("#SBATCH -N 1\n")
    #    f.write("#SBATCH -n " + str(config_input['qm_slurm_args']['num_cpus']) + "\n")
    #    f.write("#SBATCH --time=" + str(config_input['qm_slurm_args']['time_hours']) +  ":" + str(config_input['qm_slurm_args']['time_minutes']) + ":00\n")
    #    f.write("#SBATCH --mem " + config_input['qm_slurm_args']['mem'] + "\n")
    #    f.write("\n")
    #    f.write("python " + config_input['launcher_args']['path_to_mechanism_search'] + "/src/mechanism_search/run_qm_calcs.py ../../" + config_file + " >> " + job_name + ".out")
    os.system("sbatch " + slurm_folder + "/" + job_name + ".sh")

def submit_experiment(config_file):
    '''
    Submits experiment from a config file

    Args:
        config_file (str): path to the config file

    Returns:
        None
    '''
    stream = open(config_file, 'r', encoding='utf-8')
    config_input = yaml.load(stream, Loader=yaml.Loader)
    if config_input['launcher_args']['with_slurm']:
        submit_slurm(config_file, config_input)
    else:
        raise ValueError("This script will simply submit QM calcuations with slurm. Please set with_slurm to True in the config file.")
        #os.system("python ../src/mechanism_search/run_qm_calcs.py " + config_file)

if __name__ == "__main__":
    config = sys.argv[1]
    submit_experiment(config)
