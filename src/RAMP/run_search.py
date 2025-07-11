"""Module to call the search from the config file"""

import sys
import os
import time
import yaml
from rdkit import Chem
from rdkit import RDLogger  
import torch.multiprocessing as mp
import torch
from RAMP.nnp_calculations.aimnet2sph import AIMNet2Calculator
from RAMP.intermediate_enumeration.enumerate_intermediates import enumeration
from RAMP.intermediate_enumeration.perform_actions import write_smi_from_actions, remove_duplicates
from RAMP.model.predict import make_prediction
from RAMP.filters.filter import filter_mols
from RAMP.nnp_calculations.enthalpy_calculate import calculate_enthalpy_parallel
from RAMP.nnp_calculations.barrier_calculate import calculate_barrier_parallel_stereoafter
from RAMP.utils.file_write import write_to_log

def search(config_file):
    '''
    Run the search on the molecules enumerated from the input structure

    Args:
        config_file (str): path to the config file

    Returns:
        None
    '''
    stream = open(config_file, 'r')
    config_input = yaml.load(stream, Loader=yaml.Loader)

    # Make an parent results folder
    parent_folder_name = config_input['launcher_args']['path_to_mechanism_search'] + "/results/" + config_file[16:-15]
    if os.path.isdir(parent_folder_name) is False:
        os.mkdir(parent_folder_name)

    # Make a results folder to save results
    folder_name = config_input['launcher_args']['path_to_mechanism_search'] + "/results/" + config_file[16:-5]
    if os.path.isdir(folder_name) is False:
        os.mkdir(folder_name)

    # Make a folder to save transition states
    ts_folder = folder_name+"/"+config_input['nnp']['ts_folder']
    if os.path.isdir(ts_folder) is False:
        os.mkdir(ts_folder)

    # Log some initial information
    log_file = folder_name+"/"+config_input['results']['log_output']
    t = time.localtime()
    start = time.time()
    current_time = time.strftime("%H:%M:%S", t)
    write_to_log("Job started at " + str(current_time), log_file, first=True)
    write_to_log("Running search on " + config_input['data']['input_structure'], log_file)
    write_to_log("Number of bonds broken " + str(config_input['enumeration']['num_bonds_broken']), log_file)


    # First run intermediate enumeration and write all smiles to a csv for processing by the model
    reactant_smiles = config_input['data']['input_structure']
    actions, mol = enumeration(reactant_smiles, config_input['enumeration']['num_bonds_broken'], config_input['enumeration']['closed_shell'])
    mapped_reactant_smiles = Chem.MolToSmiles(mol.orig_molecule)
    total_mols = len(actions)
    write_smi_from_actions(mol, actions, workers=os.cpu_count(), batch_size=os.cpu_count()*1000, outfile=folder_name+"/"+config_input['enumeration']['smiles_output'])
    remove_duplicates(folder_name+"/"+config_input['enumeration']['smiles_output'])
    write_to_log("Total mols enumerated: " + str(total_mols), log_file)

    # Now runs SMILES through local minimum classifier and score each SMILES
    make_prediction(infile=folder_name+"/"+config_input['enumeration']['smiles_output'], 
                        model_file=config_input['launcher_args']['path_to_mechanism_search'] + "/" + config_input['classification_model']['model'],
                        outfile=folder_name+"/"+config_input['classification_model']['preds_output'])

    # Filter SMILES based on scores and save number/percent of molecules pruned 
    class_filtered_mols = filter_mols(infile=folder_name+"/"+config_input['classification_model']['preds_output'], 
                                        outfile=folder_name+"/"+config_input['classification_model']['smiles_output'],
                                        cutoff=config_input['filters']['classification_probability_cutoff'],
                                        reac_smi=mapped_reactant_smiles,
                                        greater=True,
                                        reaction=True)
    write_to_log("Mols made through classifier: " + str(class_filtered_mols), log_file)

    # Run remaining SMILES through enthalpy model
    make_prediction(infile=folder_name+"/"+config_input['classification_model']['smiles_output'],
                        model_file=config_input['launcher_args']['path_to_mechanism_search'] + "/" +config_input['enthalpy_model']['model'],
                        outfile=folder_name+"/"+config_input['enthalpy_model']['preds_output'])

    # Filter SMILES based on predicted enthalpy and save number/percent of molecules pruned
    regress_filtered_mols = filter_mols(infile=folder_name+"/"+config_input['enthalpy_model']['preds_output'], 
                                        outfile=folder_name+"/"+config_input['enthalpy_model']['smiles_output'],
                                        cutoff=config_input['filters']['enthalpy_model_cutoff'],
                                        reac_smi=mapped_reactant_smiles,
                                        greater=False,
                                        reaction=False)

    write_to_log("Mols made through regression: " + str(regress_filtered_mols), log_file)

    # Load model for barrier calculations
    if config_input['slurm_args']['multinode']:
        calc = config_input['launcher_args']['path_to_mechanism_search'] + "/" + config_input['nnp']['barrier_model_file']
        enthalpy_calc = config_input['launcher_args']['path_to_mechanism_search'] + "/" + config_input['nnp']['enthalpy_model_file']
        workers = config_input['slurm_args']['num_cpus']
    else:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model = torch.jit.load(config_input['launcher_args']['path_to_mechanism_search'] + "/" + config_input['nnp']['barrier_model_file'])
        model.share_memory()
        calc = AIMNet2Calculator(model)
        # Load mdoel for enthalpy calcuation
        enthalpy_model = torch.jit.load(config_input['launcher_args']['path_to_mechanism_search'] + "/" + config_input['nnp']['enthalpy_model_file'])
        enthalpy_model.share_memory()
        enthalpy_calc = AIMNet2Calculator(enthalpy_model)
        workers = int(os.cpu_count())

    # Calculate enthalpy for remaining molecules
    reactant_energy = calculate_enthalpy_parallel(infile=folder_name+"/"+config_input['enthalpy_model']['smiles_output'],
                       outfile=folder_name+"/"+config_input['nnp']['enthalpy_preds_output'],
                       path_to_mechanism_search=config_input['launcher_args']['path_to_mechanism_search'],
                       reactant_file=folder_name+"/"+config_input['nnp']['reactant_file'],
                       calc=enthalpy_calc,
                       reactant_smiles=mapped_reactant_smiles,
                       canon_reactant=config_input['data']['input_structure'],
                       num_confs=config_input['conformers']['enthalpy_conformers'],
                       workers=workers,
                       nodes=config_input['slurm_args']['num_nodes'],
                       log_file=log_file,
                       multinode=config_input['slurm_args']['multinode'])

    # Filter SMILES based on predicted enthalpy and save number/percent of molecules pruned
    enthalpy_filtered_mols = filter_mols(infile=folder_name+"/"+config_input['nnp']['enthalpy_preds_output'], 
                                        outfile=folder_name+"/"+config_input['nnp']['enthalpy_smiles_output'],
                                        cutoff=config_input['filters']['enthalpy_nnp_cutoff'],
                                        reac_smi=mapped_reactant_smiles,
                                        greater=False,
                                        reaction=False)
    write_to_log("Mols made through enthalpy: " + str(enthalpy_filtered_mols), log_file)

    # Calculate barrier for remaining molecules
    calculate_barrier_parallel_stereoafter(infile=folder_name+"/"+config_input['nnp']['enthalpy_smiles_output'],
                      outfile=folder_name+"/"+config_input['nnp']['barrier_preds_output'],
                      ts_folder=folder_name+"/"+config_input['nnp']['ts_folder'],
                      calc=calc,
                      enthalpy_calc=enthalpy_calc,
                      num_confs=config_input['conformers']['barrier_conformers'],
                      reactant_canon=reactant_smiles,
                      reactant_energy=reactant_energy,
                      workers=workers,
                      nodes=config_input['slurm_args']['num_nodes'],
                      log_file=log_file,
                      path_to_mechanism_search=config_input['launcher_args']['path_to_mechanism_search'],
                      multinode=config_input['slurm_args']['multinode'])

    # Clean up files
    os.system("rm *.log *.xyz")

    # Log some final information
    t2 = time.localtime()
    current_time2 = time.strftime("%H:%M:%S", t2)
    write_to_log("Analysis complete! Job complete at " + str(current_time2), log_file)
    end = time.time()
    write_to_log("Elapsed Time: " + str(round((end - start)/60, 2)) + " minutes", log_file)

if __name__ == "__main__":
    RDLogger.DisableLog('rdApp.*')
    mp.set_start_method('fork')
    #mp.set_start_method('spawn')
    config = sys.argv[1]
    search(config)
