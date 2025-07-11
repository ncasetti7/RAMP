"""Module for calculating reaction enthalpies with NNP"""

import os
import contextlib
from copy import copy
from tqdm import tqdm
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdDetermineBonds
from RAMP.nnp_calculations import aimnet2sph
from RAMP.utils import multiproc, calculations, converter, file_write

def generate_conformers_from_smiles_ff_opt(smi, n_confs=1, rms_threshold=0.5):
    '''
    Generate conformers for a molecule using the RDKit force field and optimize them

    Args:
        smi: str, SMILES string
        n_confs: int, number of conformers to generate
        rms_threshold: float, RMS threshold for conformer pruning

    Returns:
        conformers: list of pysisyphus geometry objects
    '''
    # Generate conformers with RDKit
    try:
        mol = Chem.AddHs(Chem.MolFromSmiles(smi))
        cids = AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, pruneRmsThresh=rms_threshold, useExpTorsionAnglePrefs=True, useBasicKnowledge=True)
        AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=100, ignoreInterfragInteractions=False)
    except ValueError:
        return None
    # Convert to pysisyphus geometry
    conformers = []
    for cid in cids:
        conformers.append(converter.rdmol2geom(mol, cid))

    return conformers

def load_calc(model_file):
    '''
    Load the AIMNet2 model and return a calculator object

    Args:
        model_file: str, path to the model file

    Returns:
        calc: AIMNet2Calculator object
    '''
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = torch.jit.load(model_file)
    model.share_memory()
    return aimnet2sph.AIMNet2Calculator(model)

def get_reactant_energy(reactant_smiles, num_confs, calc, reactant_file, multinode):
    '''
    Get the energy of the reactant

    Args:
        reactant_smiles (str): SMILES string of the reactant
        num_confs (int): number of conformers to generate
        calc (object): calculator object
        reactant_file (str): path to the reactant file
        multinode (bool): whether to use multinode

    Returns:
        reactant_energy (float): energy of the reactant
    '''
    reactant_list = generate_conformers_from_smiles_ff_opt(reactant_smiles, num_confs)
    reactant_energy = 9999
    for reactant in reactant_list:
        if multinode:
            reac_calc = load_calc(calc)
        else:
            reac_calc = calc
        reac_geom = calculations.optimize_geometry(reactant, reac_calc)
        if reac_geom is None:
            continue
        if reac_geom.energy < reactant_energy:
            reactant_energy = reac_geom.energy
            reac_geom.dump_xyz(reactant_file)

    if reactant_energy == 9999:
        raise ValueError("Reactant energy is 0!")

    return reactant_energy

def calculate_enthalpy_parallel(infile, outfile, path_to_mechanism_search, reactant_file, calc, reactant_smiles, canon_reactant, num_confs, workers, nodes, log_file, multinode):
    '''
    Evaluates the reaction enthalpies for reactions in an input file and writes the results to an output file

    Args:
        infile (str): path to the input file
        outfile (str): path to the output file
        path_to_mechanism_search (str): path to the mechanism search directory
        reactant_file (str): path to the reactant file
        calc (object): calculator object
        reactant_smiles (str): SMILES string of the reactant
        canon_reactant (str): canonicalized reactant SMILES
        num_confs (int): number of conformers to generate
        workers (int): number of workers to use
        nodes (int): number of nodes to use
        log_file (str): path to log file
        multinode (bool): whether to use multinode

    Returns:
        reactant_energy (float): energy of the reactant
    '''

    # Calculate the energy of the reactant
    with contextlib.redirect_stdout(open(os.devnull, 'w', encoding='utf-8')):
        #try:
        reactant_energy = get_reactant_energy(reactant_smiles, num_confs, calc, reactant_file, multinode)
        #except ValueError:
        #    # Hard code in the old optimized reactant geometry for now (find a way to extract this easily)
        #    # There are certain cases where the reactant geometry cannot be embedded, if you'd still like to run a search, find the optimized geometry from your previous run and hard code it here
        #    file_write.write_to_log('Generating the reactant failed, falling back to old reactant geometry', log_file)
        #    current_folder = "2_7_2024_tandem_oxy_1"
        #    search_with_reactant = "search_01"
        #    r_file = "HARD CODE THE REACTANT FILE HERE"
        #    reactant_file = path_to_mechanism_search + "/results/" + current_folder + "/" + search_with_reactant + "/orca_ts_folder/" + r_file
        #    raw_mol = Chem.MolFromXYZFile(reactant_file)
        #    mol = Chem.Mol(raw_mol)
        #    rdDetermineBonds.DetermineBonds(mol,charge=0)
        #   if Chem.CanonSmiles(Chem.MolToSmiles(mol)) != canon_reactant:
        #       raise ValueError("The reactant geometry is not the same as the reactant smiles!")
        #   reac_geom = converter.rdmol2geom(mol, 0)
        #    calculations.optimize_geometry(reac_geom, calc)
        #    reactant_energy = reac_geom.energy

    file_write.write_to_log("Reactant energy: " + str(reactant_energy), log_file)

    # Open regression_filtered_smiles.csv and put each line into a list
    with open(infile, "r", encoding='utf-8') as f:
        lines = f.readlines()
    lines = lines[1:]

    # Generate conformers for each line in the list
    smiles_input = []
    pool_input = []
    for line in tqdm(lines):
        line = line.split(",")
        product_smiles = line[0].split(">>")[1]
        canon_smiles = Chem.CanonSmiles(Chem.MolToSmiles(converter.remove_mapping(copy(converter.make_mol(product_smiles)))), useChiral=0)
        smiles_input.append({"canon_smiles": canon_smiles, "mapped_smiles": product_smiles})
        pool_input.append({"smi": product_smiles, "n_confs": num_confs})

    file_write.write_to_log("Generating initial conformers for " + str(len(pool_input)) + " conformers", log_file)

    # Run the conformer generation in parallel
    all_conformers = multiproc.parallel_run(calculations.ff_generate_conformers, pool_input, workers, nodes, path_to_mechanism_search, multinode)

    # Make a list of dictionaries with the conformers and smiles (removing any None conformers)
    opt_input = []
    for i, conformers in enumerate(all_conformers):
        if conformers is not None:
            opt_input.append({"conformers": conformers, "canon_smiles": smiles_input[i]["canon_smiles"], "mapped_smiles": smiles_input[i]["mapped_smiles"]})

    file_write.write_to_log("Number of conformers to optimize: " + str(len(opt_input)), log_file)
    # Run the conformer optimization in parallel
    energies = multiproc.parallel_run(calculations.calc_prod_energy, opt_input, workers, nodes, path_to_mechanism_search, multinode, calc)

    file_write.write_to_log("Optimization complete!", log_file)
    # Write to output file
    with open(outfile, "w", encoding='utf-8') as f:
        f.write("smiles,explosion\n")
        for i, energy in enumerate(energies):
            if energy is not None:
                f.write(reactant_smiles + ">>" + opt_input[i]["mapped_smiles"][:-1] + "," + str(energy - reactant_energy) + "\n")
                f.flush()

    return reactant_energy
