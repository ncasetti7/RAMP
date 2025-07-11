"""Module for writing files"""

def write_slurm_script(template_script_file, slurm_args_dict, script_file_name):
    '''
    Write a SLURM script file based on a template script file and a dictionary of SLURM arguments.
    '''

    # Read the template script file
    with open(template_script_file, 'r', encoding='utf-8') as f:
        template_script = f.readlines()

    with open(script_file_name, 'w', encoding='utf-8') as f:
        for line in template_script:
            if "--job-name" in line:
                f.write(line[:-1] + slurm_args_dict['job_name'] + "\n")
            elif "--output" in line:
                f.write(line[:-1] + slurm_args_dict['output'] + ".out\n")
            elif "-error" in line:
                f.write(line[:-1] + slurm_args_dict['output'] + ".err\n")
            elif "-n" in line:
                f.write(line[:-1] + str(slurm_args_dict['num_cpus']) + "\n")
            elif "--time" in line:
                f.write(line[:-1] + str(slurm_args_dict['time_hours']) + ":" + str(slurm_args_dict['time_minutes']) + ":00\n")
            elif "--mem" in line:
                f.write(line[:-1] + " " + str(slurm_args_dict['mem']) + "\n")
            else:
                f.write(line)
        f.write("\n")
        f.write(slurm_args_dict['command'])


def write_to_log(message, file, first=False):
    '''
    Write a message to a log file

    Args:
        message (str): message to write to the log file
        file (str): path to the log file
        first (bool): if True, write the message to the log file for the first time
    
    Returns:

    '''
    if first:
        mode = 'w'
    else:
        mode = 'a'
    with open(file, mode, encoding='utf-8') as f:
        f.write(message + "\n")
        f.flush()
