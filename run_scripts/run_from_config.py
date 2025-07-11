"""Module to run experiments from a config file"""

import sys
import os
import yaml
from RAMP.utils.file_write import write_slurm_script

def submit_slurm(config_file, config_input):
    '''
    Submit the mechanism search to slurm

    Args:
        config_file (str): path to the config file
        config_input (dict): dictionary of input parameters

    Returns:
        None
    '''
    run_scripts_folder = config_input['launcher_args']['path_to_mechanism_search'] + "/run_scripts/"
    slurm_folder = run_scripts_folder + "slurm_scripts/" + config_file[10:-15]
    if os.path.isdir(slurm_folder) is False:
        os.mkdir(slurm_folder)
    parent_folder = run_scripts_folder + config_file[10:-15]
    # Make a folder for the overall search if it doesn't exist
    if os.path.isdir(parent_folder) is False:
        os.mkdir(parent_folder)
    os.chdir(parent_folder)
    # Make a directory for the specific job if it doesn't exist
    job_name = config_file[-14:-5]
    job_folder = parent_folder + "/" + job_name
    if os.path.isdir(job_folder) is False:
        os.mkdir(job_folder)
    os.chdir(job_folder)
    if config_input['slurm_args']['multinode']:
        num_cpus = 1
    else:
        num_cpus = config_input['slurm_args']['num_cpus']

    slurm_arg_dict = config_input['slurm_args']
    slurm_arg_dict['job_name'] = job_name
    slurm_arg_dict['output'] = job_name
    slurm_arg_dict['num_cpus'] = num_cpus
    slurm_arg_dict['command'] = "python " + config_input['launcher_args']['path_to_mechanism_search'] + "/src/RAMP/run_search.py ../../" + config_file + " >> " + job_name + ".out"
    write_slurm_script(config_input['launcher_args']['path_to_mechanism_search'] + '/' + config_input['slurm_args']['template_slurm_script'], slurm_arg_dict, slurm_folder + "/" + job_name + ".sh")
    #with open(slurm_folder + "/" + job_name + ".sh", 'w') as f:
    #    f.write("#!/bin/bash\n")
    #    f.write("#SBATCH --job-name=" + job_name + "\n")
    #    f.write("#SBATCH --output " + job_name + ".out\n")
    #    f.write("#SBATCH -N 1\n")
    #    f.write("#SBATCH -n " + str(num_cpus) + "\n")
    #    f.write("#SBATCH --time=" + str(config_input['slurm_args']['time_hours']) +  ":" + str(config_input['slurm_args']['time_minutes']) + ":00\n")
    #    f.write("#SBATCH --mem " + config_input['slurm_args']['mem'] + "\n")
    #    f.write("\n")
    #    f.write("python " + config_input['launcher_args']['path_to_mechanism_search'] + "/src/mechanism_search/run_search.py ../../" + config_file + " >> " + job_name + ".out")
    os.system("sbatch " + slurm_folder + "/" + job_name + ".sh")

def submit_experiment(config_file):
    '''
    Submit experiment from config file

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
        os.system("python ../src/RAMP/run_search.py " + config_file)

if __name__ == "__main__":
    config = sys.argv[1]
    submit_experiment(config)
