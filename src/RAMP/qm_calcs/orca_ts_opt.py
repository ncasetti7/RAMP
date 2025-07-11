"""Module to run Orca QM Calculations on the transition states predicted by the neural network"""

import os
import time
from pysisyphus.tsoptimizers.RSPRFOptimizer import RSPRFOptimizer
from pysisyphus.run import run_irc
from pysisyphus.calculators.ORCA import ORCA
from pysisyphus.elem_data import INV_ATOMIC_NUMBERS
from pysisyphus.constants import ANG2BOHR
from pysisyphus.Geometry import Geometry
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import rdDetermineBonds
from rdkit.Chem import rdMolAlign
from openbabel import pybel
from RAMP.utils.converter import xyz2rdmol, pybel2geom
from RAMP.utils.file_write import write_slurm_script


def write_to_equilibrium_cache(equilibrium_cache_file, smiles, energy):
    '''
    Write the energy of a structure to the cache file if it isn't in there. 
    If it is there, check whether the energy is lower than the one in the cache file and update if it is.

    Args:
        equilibrium_cache_file (str): path to the equilibrium cache file
        smiles (str): SMILES of the structure
        energy (float): energy of the structure

    Returns:
        None
    '''
    if os.path.isfile(equilibrium_cache_file):
        df = pd.read_csv(equilibrium_cache_file)
        if smiles in df["smiles"].tolist():
            index = df["smiles"].tolist().index(smiles)
            if energy < df["energy"].tolist()[index]:
                df["energy"].tolist()[index] = energy
                df.to_csv(equilibrium_cache_file, index=False)
        else:
            with open(equilibrium_cache_file, "a", encoding='utf-8') as f:
                f.write(smiles + "," + str(energy) + "\n")
    else:
        with open(equilibrium_cache_file, "w", encoding='utf-8') as f:
            f.write("smiles,energy\n")
            f.write(smiles + "," + str(energy) + "\n")

def write_orca_ts_input(ts_file, orca_ts_folder, index, cores=12):
    '''
    Write the input file for the Orca transition state optimization calculation
    
    Args:
        ts_file (str): path to the transition state file
        orca_ts_folder (str): path to the folder where the input file will be written
        index (int): index of the transition state
        cores (int): number of cores to use for the calculation
        
    Returns:
        ts_input (str): path to the input file

    '''
    ts_input = orca_ts_folder + "/ts_" + str(index) + ".inp"
    with open(ts_input, "w", encoding='utf-8') as f:
        f.write("! wB97X 6-31G(d,p) D2 Grid6 OptTS freq\n")
        f.write("%pal nprocs " + str(cores) + " end\n")
        f.write("%geom\n")
        f.write("Calc_Hess true\n")
        f.write("NumHess false\n")
        f.write("Recalc_Hess 5\n")
        f.write("end\n")
        f.write("* XYZfile 0 1 " + ts_file + "\n")
    return ts_input

def write_orca_irc_input(ts_file, orca_ts_folder, index, cores=12):
    '''
    Write the input file for the Orca IRC calculation

    Args:
        ts_file (str): path to the transition state file
        orca_ts_folder (str): path to the folder where the input file will be written
        index (int): index of the transition state
        cores (int): number of cores to use for the calculation

    Returns:
        irc_input (str): path to the input file
    '''
    irc_input = orca_ts_folder + "/irc_" + str(index) + ".inp"
    hess_file = orca_ts_folder + "/ts_" + str(index) + ".hess"
    with open(irc_input, "w", encoding='utf-8') as f:
        f.write("! IRC wB97X 6-31G(d,p) D2 Grid6\n")
        f.write("%pal nprocs " + str(cores) + " end\n")
        f.write("%irc MaxIter 50 InitHess Read Hess_Filename \"" + hess_file  + "\" END\n")
        f.write("* XYZfile 0 1 " + ts_file + "\n")
    return irc_input

def write_orca_opt_input(reac_file, orca_ts_folder, index, reac, cores=12):
    '''
    Write the input file for the Orca optimization calculation

    Args:
        reac_file (str): path to the reactant or product file
        orca_ts_folder (str): path to the folder where the input file will be written
        index (int): index of the reactant or product
        reac (bool): True if the file is a reactant, False if it is a product
        cores (int): number of cores to use for the calculation

    Returns:
        opt_input (str): path to the input file
    '''
    if reac:
        name = "reac"
    else:
        name = "prod"
    opt_input = orca_ts_folder + "/" + name + "_" + str(index) + ".inp"
    with open(opt_input, "w", encoding='utf-8') as f:
        f.write("! wB97X 6-31G(d,p) D2 Grid6 Opt freq\n")
        f.write("%geom maxiter 1000 end\n")
        f.write("%pal nprocs " + str(cores) + " end\n")
        f.write("* XYZfile 0 1 " + reac_file + "\n")
    return opt_input

def write_orca_spe_freq_input(geom_file, spe_type, orca_ts_folder, index, cores=12):
    '''
    Write the input file for the Orca single point energy and frequency calculation

    Args:
        geom_file (str): path to the geometry file
        spe_type (str): type of calculation (reac, prod, ts)
        orca_ts_folder (str): path to the folder where the input file will be written
        index (int): index of the reactant, product, or transition state
        cores (int): number of cores to use for the calculation

    Returns:
        spe_input (str): path to the input file
    '''
    spe_input = orca_ts_folder + "/spe_" + spe_type + str(index) + ".inp"
    with open(spe_input, "w", encoding='utf-8') as f:
        f.write("! wB97X 6-311+G(d,p) D2 Grid6\n")
        f.write("%pal nprocs " + str(cores) + " end\n")
        f.write("* XYZfile 0 1 " + geom_file + "\n")
    return spe_input

def write_slurm_ts_script(ts_input, orca_ts_folder, index, orca_path, slurm_args_dict):
    '''
    Write the SLURM script for the Orca transition state optimization calculation

    Args:
        ts_input (str): path to the input file
        orca_ts_folder (str): path to the folder where the input file is written
        index (int): index of the transition state
        orca_path (str): path to the Orca executable
        slurm_args_dict (dict): dictionary of SLURM arguments
    
    Returns:
        slurm_script (str): path to the SLURM script
    '''
    slurm_script = orca_ts_folder + "/ts_" + str(index) + ".job"
    ts_output = orca_ts_folder + "/ts_" + str(index) + ".log"
    slurm_args_dict['command'] = orca_path + " " + ts_input + " > " + ts_output
    slurm_args_dict['job_name'] = "ts_" + str(index)
    slurm_args_dict['output'] = orca_ts_folder + "/ts_" + str(index)
    write_slurm_script(slurm_args_dict['template_slurm_script'], slurm_args_dict, slurm_script)
    #with open(slurm_script, "w") as f:
    #    f.write("#!/bin/bash\n")
    #    f.write("#SBATCH -J ts_" + str(index) + "\n")
    #    f.write("#SBATCH -o " + orca_ts_folder + "/ts_" + str(index) + ".out\n")
    #    f.write("#SBATCH -e " + orca_ts_folder + "/ts_" + str(index) + ".err\n")
    #    f.write("#SBATCH -N 1\n")
    #    f.write("#SBATCH -n " + str(cores) + "\n")
    #    f.write("#SBATCH -t 48:00:00\n")
    #    f.write("#SBATCH --mem=4000\n")
    #    f.write("\n")
    #    f.write(orca_path + " " + ts_input + " > " + ts_output + "\n")
    return slurm_script

def write_slurm_irc_script(irc_input, orca_ts_folder, index, orca_path, slurm_args_dict):
    '''
    Write the SLURM script for the Orca IRC calculation

    Args:
        irc_input (str): path to the input file
        orca_ts_folder (str): path to the folder where the input file is written
        index (int): index of the transition state
        orca_path (str): path to the Orca executable
        slurm_args_dict (dict): dictionary of SLURM arguments

    Returns:
        slurm_script (str): path to the SLURM script
    '''
    slurm_script = orca_ts_folder + "/irc_" + str(index) + ".job"
    irc_output = orca_ts_folder + "/irc_" + str(index) + ".log"
    slurm_args_dict['command'] = orca_path + " " + irc_input + " > " + irc_output
    slurm_args_dict['job_name'] = "irc_" + str(index)
    slurm_args_dict['output'] = orca_ts_folder + "/irc_" + str(index)
    write_slurm_script(slurm_args_dict['template_slurm_script'], slurm_args_dict, slurm_script)
    #with open(slurm_script, "w") as f:
    #    f.write("#!/bin/bash\n")
    #    f.write("#SBATCH -J irc_" + str(index) + "\n")
    #    f.write("#SBATCH -o " + orca_ts_folder + "/irc_" + str(index) + ".out\n")
    #    f.write("#SBATCH -e " + orca_ts_folder + "/irc_" + str(index) + ".err\n")
    #    f.write("#SBATCH -N 1\n")
    #    f.write("#SBATCH -n " + str(cores) + "\n")
    #    f.write("#SBATCH -t 24:00:00\n")
    #    f.write("#SBATCH --mem=4000\n")
    #    f.write("\n")
    #    f.write(orca_path + " " + irc_input + " > " + irc_output + "\n")
    return slurm_script

def write_slurm_opt_script(opt_input, orca_ts_folder, index, orca_path, reac, slurm_args_dict):
    '''
    Write the SLURM script for the Orca optimization calculation

    Args:
        opt_input (str): path to the input file
        orca_ts_folder (str): path to the folder where the input file is written
        index (int): index of the reactant or product
        orca_path (str): path to the Orca executable
        reac (bool): True if the file is a reactant, False if it is a product
        slurm_args_dict (dict): dictionary of SLURM arguments

    Returns:
        slurm_script (str): path to the SLURM script
    '''
    if reac:
        name = "reac"
    else:
        name = "prod"
    slurm_script = orca_ts_folder + "/" + name + "_" + str(index) + ".job"
    opt_output = orca_ts_folder + "/" + name + "_" + str(index) + ".log"
    slurm_args_dict['command'] = orca_path + " " + opt_input + " > " + opt_output
    slurm_args_dict['job_name'] = name + "_" + str(index)
    slurm_args_dict['output'] = orca_ts_folder + "/" + name + "_" + str(index)
    write_slurm_script(slurm_args_dict['template_slurm_script'], slurm_args_dict, slurm_script)
    #with open(slurm_script, "w") as f:
    #    f.write("#!/bin/bash\n")
    #    f.write("#SBATCH -J " + name + "_" + str(index) + "\n")
    #    f.write("#SBATCH -o " + orca_ts_folder + "/" + name + "_" + str(index) + ".out\n")
    #    f.write("#SBATCH -e " + orca_ts_folder + "/" + name+ "_" + str(index) + ".err\n")
    #    f.write("#SBATCH -N 1\n")
    #    f.write("#SBATCH -n " + str(cores) + "\n")
    #    f.write("#SBATCH -t 24:00:00\n")
    #    f.write("#SBATCH --mem=4000\n")
    #    f.write("\n")
    #    f.write(orca_path + " " + opt_input + " > " + opt_output + "\n")
    return slurm_script

def write_slurm_spe_freq_script(spe_input, spe_type, orca_ts_folder, index, orca_path, slurm_args_dict):
    '''
    Write the SLURM script for the Orca single point energy and frequency calculation

    Args:
        spe_input (str): path to the input file
        spe_type (str): type of calculation (reac, prod, ts)
        orca_ts_folder (str): path to the folder where the input file is written
        index (int): index of the reactant, product, or transition state
        orca_path (str): path to the Orca executable
        slurm_args_dict (dict): dictionary of SLURM arguments

    Returns:
        slurm_script (str): path to the SLURM script
    '''
    slurm_script = orca_ts_folder + "/spe_" + spe_type + str(index) + ".job"
    spe_output = orca_ts_folder + "/spe_" + spe_type +str(index) + ".log"
    slurm_args_dict['command'] = orca_path + " " + spe_input + " > " + spe_output
    slurm_args_dict['job_name'] = "spe_" + str(index)
    slurm_args_dict['output'] = orca_ts_folder + "/spe_" + spe_type + str(index)
    write_slurm_script(slurm_args_dict['template_slurm_script'], slurm_args_dict, slurm_script)
    #with open(slurm_script, "w") as f:
    #    f.write("#!/bin/bash\n")
    #    f.write("#SBATCH -J spe_" + str(index) + "\n")
    #    f.write("#SBATCH -o " + orca_ts_folder + "/spe_" + spe_type +str(index) + ".out\n")
    #    f.write("#SBATCH -e " + orca_ts_folder + "/spe_" + spe_type +str(index) + ".err\n")
    #    f.write("#SBATCH -N 1\n")
    #    f.write("#SBATCH -n " + str(cores) + "\n")
    #    f.write("#SBATCH -t 36:00:00\n")
    #    f.write("#SBATCH --mem=32000\n")
    #    f.write("\n")
    #    f.write(orca_path + " " + spe_input + " > " + spe_output + "\n")
    return slurm_script

def parse_opt_output(output):
    '''
    Parse the output file of an Orca optimization calculation and return the energy

    Args:
        output (str): path to the output file

    Returns:
        energy (float): energy of the optimized structure
    '''
    with open(output, "r", encoding='utf-8') as f:
        lines = f.readlines()
    converged = False
    for line in lines:
        if "HURRAY" in line:
            converged = True
        if converged and "G-E(el)" in line:
            # Get the energy from the line
            energy_index = line.split(" ").index("Eh") - 1
            energy = float(line.split(" ")[energy_index])
            for mol in pybel.readfile("orca", output):
                ts_geom = pybel2geom(mol)
                ts_geom.dump_xyz(output.replace(".log", ".xyz"))
            return energy
    return None

def parse_irc_output(irc_output, orca_ts_folder, canon_reactant):
    '''
    Parse the output files of an Orca IRC calculation and return the reactant and product structures

    Args:
        irc_output (str): name of the IRC output file
        orca_ts_folder (str): path to the folder where the output files are written
        canon_reactant (str): canonical SMILES of the reactant

    Returns:
        backward_xyz (str): path to the backward geometry file
        forward_xyz (str): path to the forward geometry file
        rxn (str): reaction SMILES
    '''
    backward_xyz = orca_ts_folder + "/" + irc_output + "_IRC_B.xyz"
    forward_xyz = orca_ts_folder + "/" + irc_output + "_IRC_F.xyz"
    try:
        backward_mol = xyz2rdmol(backward_xyz)
        forward_mol = xyz2rdmol(forward_xyz)
    except ValueError:
        return None, None, None
    backward_smiles = Chem.CanonSmiles(Chem.MolToSmiles(backward_mol))
    forward_smiles = Chem.CanonSmiles(Chem.MolToSmiles(forward_mol))
    if backward_smiles == canon_reactant:
        return backward_xyz, forward_xyz, backward_smiles + ">>" + forward_smiles
    if forward_smiles == canon_reactant:
        return forward_xyz, backward_xyz, forward_smiles + ">>" + backward_smiles
    return None, None, None

def parse_spe_freq_output(output):
    '''
    Parse the output file of an Orca single point energy and frequency calculation and return the energy

    Args:
        output (str): path to the output file

    Returns:
        energy (float): energy of the structure
    '''
    with open(output, "r", encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        if "FINAL SINGLE POINT ENERGY" in line:
            energy = float(line.split(" ")[-1])
            return energy
    return None

def orca_slurm(orca_path, outfile, equilibrium_cache_file, indices, ts_folder, orca_ts_folder, canon_reactant, slurm_args_dict):
    '''
    Perform Orca calculations on the transition states predicted by the neural network

    Args:
        orca_path (str): path to the Orca executable
        outfile (str): path to the output file
        equilibrium_cache_file (str): path to the equilibrium cache file
        indices (list): list of indices of the transition states
        ts_folder (str): path to the folder where the transition state files are stored
        orca_ts_folder (str): path to the folder where the Orca input files will be written
        canon_reactant (str): canonical SMILES of the reactant
        slurm_args_dict (dict): dictionary of SLURM arguments

    Returns:
        None
    '''
    cores = slurm_args_dict['num_cpus']
    # Run TS optimizations
    for index in indices:
        file = ts_folder + "/ts_" + str(index) + ".xyz"
        ts_input = write_orca_ts_input(file, orca_ts_folder, index, cores=cores)
        slurm_script = write_slurm_ts_script(ts_input, orca_ts_folder, index, orca_path, slurm_args_dict)
        os.system("sbatch " + slurm_script)

    # Wait for all jobs to finish
    while True:
        jobs = os.popen("squeue -u ncasetti").read()
        if "ts_" not in jobs:
            break
        time.sleep(600)

    # Parse output files
    correction_energies = []
    ts_files = []
    new_indices = []
    for index in indices:
        ts_output = orca_ts_folder + "/ts_" + str(index) + ".log"
        try:
            correction_energy = parse_opt_output(ts_output)
        except ValueError:
            continue
        if correction_energy is None:
            continue
        correction_energies.append(correction_energy)
        ts_files.append(ts_output.replace(".log", ".xyz"))
        new_indices.append(index)

    # Run IRCs
    for i, index in enumerate(new_indices):
        ts_file = ts_files[i]
        irc_input = write_orca_irc_input(ts_file, orca_ts_folder, index, cores=cores)
        slurm_script = write_slurm_irc_script(irc_input, orca_ts_folder, index, orca_path, slurm_args_dict)
        os.system("sbatch " + slurm_script)

    # Wait for all jobs to finish
    while True:
        jobs = os.popen("squeue -u ncasetti").read()
        if "irc_" not in jobs:
            break
        time.sleep(300)

    # Parse output files
    irc_outputs = []
    for i, index in enumerate(new_indices):
        irc_output = "irc_" + str(index)
        reac, prod, rxn = parse_irc_output(irc_output, orca_ts_folder, canon_reactant)
        if reac is None:
            continue
        irc_outputs.append((index, reac, prod, rxn, correction_energies[i]))

    # Optimize the reactant and product
    for data in irc_outputs:
        index = data[0]
        reac_file = data[1]
        prod_file = data[2]
        opt_input = write_orca_opt_input(reac_file, orca_ts_folder, index, reac=True, cores=cores)
        slurm_script = write_slurm_opt_script(opt_input, orca_ts_folder, index, orca_path, reac=True, slurm_args_dict=slurm_args_dict)
        os.system("sbatch " + slurm_script)
        opt_input = write_orca_opt_input(prod_file, orca_ts_folder, index, reac=False, cores=cores)
        slurm_script = write_slurm_opt_script(opt_input, orca_ts_folder, index, orca_path, reac=False, slurm_args_dict=slurm_args_dict)
        os.system("sbatch " + slurm_script)

    # Wait for all jobs to finish
    while True:
        jobs = os.popen("squeue -u ncasetti").read()
        if "prod_" not in jobs and "reac_" not in jobs:
            break
        time.sleep(300)

    # Parse output files (recheck the connectivity)
    opt_results = []
    for data in irc_outputs:
        index = data[0]
        rxn = data[3]
        ts_correction = data[4]
        # Check connectivity and extract correction energy
        try:
            reactant_correction = parse_opt_output(orca_ts_folder + "/reac_" + str(index) + ".log")
            product_correction = parse_opt_output(orca_ts_folder + "/prod_" + str(index) + ".log")
            reactant_mol = xyz2rdmol(orca_ts_folder + "/reac_" + str(index) + ".xyz")
            product_mol = xyz2rdmol(orca_ts_folder + "/prod_" + str(index) + ".xyz")
        except ValueError:
            continue
        if reactant_correction is None or product_correction is None:
            continue
        reactant_smiles = Chem.CanonSmiles(Chem.MolToSmiles(reactant_mol))
        product_smiles = Chem.CanonSmiles(Chem.MolToSmiles(product_mol))
        canon_reaction = reactant_smiles + ">>" + product_smiles
        if canon_reaction == rxn:
            print(rxn, index, reactant_smiles, product_smiles)
            opt_results.append((rxn, index, reactant_smiles, product_smiles, ts_correction, reactant_correction, product_correction))

    # Run single point energy and frequency calculations 
    for data in opt_results:
        index = data[1]
        reac_file = orca_ts_folder + "/reac_" + str(index) + ".xyz"
        prod_file = orca_ts_folder + "/prod_" + str(index) + ".xyz"
        ts_file = orca_ts_folder + "/ts_" + str(index) + ".xyz"
        spe_input = write_orca_spe_freq_input(reac_file, "reac", orca_ts_folder, index, cores=cores)
        slurm_script = write_slurm_spe_freq_script(spe_input, "reac", orca_ts_folder, index, orca_path, slurm_args_dict)
        os.system("sbatch " + slurm_script)
        spe_input = write_orca_spe_freq_input(prod_file, "prod", orca_ts_folder, index, cores=cores)
        slurm_script = write_slurm_spe_freq_script(spe_input, "prod", orca_ts_folder, index, orca_path, slurm_args_dict)
        os.system("sbatch " + slurm_script)
        spe_input = write_orca_spe_freq_input(ts_file, "ts", orca_ts_folder, index, cores=cores)
        slurm_script = write_slurm_spe_freq_script(spe_input, "ts", orca_ts_folder, index, orca_path, slurm_args_dict)
        os.system("sbatch " + slurm_script)
    
    # Wait for all jobs to finish
    while True:
        jobs = os.popen("squeue -u ncasetti").read()
        if "spe_" not in jobs:
            break
        time.sleep(300)
    
    # Parse output files
    final_results = []
    for data in opt_results:
        index = data[1]
        rxn = data[0]
        reactant_smiles = data[2]
        product_smiles = data[3]
        ts_correction = data[4]
        reactant_correction = data[5]
        product_correction = data[6]
        reac_file = orca_ts_folder + "/reac_" + str(index) + ".xyz"
        prod_file = orca_ts_folder + "/prod_" + str(index) + ".xyz"
        ts_file = ts_files[new_indices.index(index)]
        reac_energy = parse_spe_freq_output(orca_ts_folder + "/spe_reac" + str(index) + ".log")
        prod_energy = parse_spe_freq_output(orca_ts_folder + "/spe_prod" + str(index) + ".log")
        ts_energy = parse_spe_freq_output(orca_ts_folder + "/spe_ts" + str(index) + ".log")
        final_results.append((rxn, ts_energy, reactant_smiles, product_smiles, reac_energy, prod_energy, ts_correction, reactant_correction, product_correction))

    # Write final_results to outfile
    with open(outfile, "w", encoding='utf-8') as f:
        f.write("smiles,ts_energy\n")
        for data in final_results:
            rxn = data[0]
            ts_energy = data[1]
            ts_correction = data[6]
            if ts_energy is None:
                continue
            ts_gibbs = ts_energy + ts_correction
            f.write(rxn + "," + str(ts_gibbs) + "\n")
            f.flush()

    # Write equilibrium structures to equilibrium_cache_file
    for data in final_results:
        reactant_smiles = data[2]
        product_smiles = data[3]
        reac_energy = data[4]
        prod_energy = data[5]
        reactant_correction = data[7]
        product_correction = data[8]
        if reac_energy is not None:
            reac_gibbs = reac_energy + reactant_correction
            write_to_equilibrium_cache(equilibrium_cache_file, reactant_smiles, reac_gibbs)
        if prod_energy is not None:
            prod_gibbs = prod_energy + product_correction
            write_to_equilibrium_cache(equilibrium_cache_file, product_smiles, prod_gibbs)

def run_orca_calcs(orca_path, infile, outfile, equilibrium_cache_file, ts_folder, orca_ts_folder, canon_reactant, slurm_args_dict, use_barrier_cutoff, barrier_cutoff, use_top_k_cutoff, top_k_cutoff):
    '''
    Parse which transition states to run Orca calculations on and run them

    Args:
        orca_path (str): path to the Orca executable
        infile (str): path to the input file
        outfile (str): path to the output file
        equilibrium_cache_file (str): path to the equilibrium cache file
        ts_folder (str): path to the folder where the transition state files are stored
        orca_ts_folder (str): path to the folder where the Orca input files will be written
        canon_reactant (str): canonical SMILES of the reactant
        slurm_args_dict (dict): dictionary of SLURM arguments
        use_barrier_cutoff (bool): if True, use the barrier cutoff
        barrier_cutoff (float): barrier cutoff energy
        use_top_k_cutoff (bool): if True, use the top k cutoff
        top_k_cutoff (int): number of reactions to run Orca calculations on

    Returns:
        None
    '''
    # Shut up rdkit
    RDLogger.DisableLog('rdApp.*')

    # Read in reactions and energies from infile csv
    df = pd.read_csv(infile)
    rxns = df["smiles"].tolist()
    energies = df["ts_energy"].tolist()

    # Select only the reactions where the reactant or product is canon_reactant and take the lowest energy TS
    selected_rxns = {}
    for i, rxn in enumerate(rxns):
        reactant, product = rxn.split(">>")
        reactant_smiles = Chem.CanonSmiles(reactant)
        product_smiles = Chem.CanonSmiles(product)
        canon_reaction = reactant_smiles + ">>" + product_smiles
        if canon_reactant in (reactant_smiles, product_smiles):
            if reactant_smiles == product_smiles:
                continue
            if use_barrier_cutoff:
                if energies[i] > barrier_cutoff:
                    continue
            if product_smiles == canon_reactant:
                canon_reaction = product_smiles + ">>" + reactant_smiles
            if canon_reaction in selected_rxns.keys():
                ts = Chem.MolFromXYZFile(f"{ts_folder}/ts_{i}.xyz")
                # Make a list of tuples like [(1,1), (2,2)...] for the number of atoms (to use for mapping for rdkit)
                atom_nums = []
                for a in range(ts.GetNumAtoms()):
                    atom_nums.append((a, a))
                add_to_list = True
                for j, values in enumerate(selected_rxns[canon_reaction]):
                    index = values[0]
                    energy = values[1]
                    ts_compare = Chem.MolFromXYZFile(f"{ts_folder}/ts_{index}.xyz")
                    rmsd = rdMolAlign.AlignMol(ts, ts_compare, prbCid=0, refCid=0, atomMap=atom_nums)
                    if rmsd < 0.1:
                        if energies[i] < energy:
                            selected_rxns[canon_reaction][j] = (i, energies[i])
                        add_to_list = False
                        break
                if add_to_list:
                    selected_rxns[canon_reaction].append((i, energies[i]))
            else:
                selected_rxns[canon_reaction] = [(i, energies[i])]

    # Order the indices by energy
    final_rxn_list = []
    for value in selected_rxns.values():
        final_rxn_list += value
    final_rxn_list.sort(key=lambda x: x[1])
    # Extract the indices from final_rxn_list
    indices = [x[0] for x in final_rxn_list]
    # Grab the first 50 reactions (or less if there are less than 50)
    if use_top_k_cutoff:
        indices = indices[:min(top_k_cutoff, len(indices))]

    # Run orca calculations on selected reactions
    orca_slurm(orca_path, outfile, equilibrium_cache_file, indices, ts_folder, orca_ts_folder, canon_reactant, slurm_args_dict)
