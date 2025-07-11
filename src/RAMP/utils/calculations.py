"""Module for performing NNP calculations"""

import os
import contextlib
import torch
from pysisyphus.cos.AdaptiveNEB import AdaptiveNEB
from pysisyphus.interpolate import interpolate
from pysisyphus.optimizers.LBFGS import LBFGS
from pysisyphus.tsoptimizers.RSPRFOptimizer import RSPRFOptimizer
from pysisyphus.run import run_irc
from pysisyphus.optimizers.FIRE import FIRE as Opt
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolAlign
from RAMP.utils import converter

def optimize_geometry(geom, calc):
    '''
    Optimize a geometry with pysisyphus

    Args:
        geom: pysisyphus.Geometry
        calc: pysisyphus.Calculator

    Returns:
        geom: pysisyphus.Geometry
    '''
    geom.set_calculator(calc)
    opt = Opt(geom, max_cycles=300)
    #print("AHHHH")
    #with contextlib.redirect_stdout(open(os.devnull, 'w', encoding='utf-8')):
    with open(os.devnull, "w", encoding='utf-8') as f, contextlib.redirect_stdout(f):
        opt.run()
    geom = opt.geometry
    return geom

def ff_generate_conformers(input_dict):
    '''
    Generate conformers and optimizes them with RDKit

    Args:
        input_dict: dict containing the SMILES string and number of conformers to generate

    Returns:
        conformers: list of pysisyphus.Geometry
    '''
    # Generate conformers with RDKit
    smi = input_dict['smi']
    n_confs = input_dict['n_confs']
    try:
        mol = Chem.AddHs(Chem.MolFromSmiles(smi))
        cids = AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, pruneRmsThresh=0.5, useExpTorsionAnglePrefs=True, useBasicKnowledge=True, maxAttempts=20)
        AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=100, ignoreInterfragInteractions=False)
    except:
        return None

    # Convert to pysisyphus geometry
    conformers = []
    for cid in cids:
        conformers.append(converter.rdmol2geom(mol, cid))

    return conformers

def ff_check_stereoisomers(input_dict):
    '''
    Given an input reaction, enumerate stereoisomers for the reactant constrained by the product and check whether the stereoisomer is a valid reactant

    Args:
        input_dict: dict containing the reaction and whether it is reversible

    Returns:
        valid_rxns: list of valid stereoisomers
    '''
    rxn = input_dict['rxn']
    reverse = input_dict['reverse']

    if reverse is False:
        return ([rxn], reverse)

    stereoisomers = converter.generate_stereoisomers(rxn.split(">>")[0], rxn.split(">>")[1])

    # Check where the stereoisomer is a valid reactant by attempting to embed
    valid_rxns = []
    for stereoisomer in stereoisomers:
        mol = Chem.AddHs(Chem.MolFromSmiles(stereoisomer))
        try:
            check = AllChem.EmbedMolecule(mol, useExpTorsionAnglePrefs=True, useBasicKnowledge=True, maxAttempts=10)
            if check == 0:
                valid_rxns.append(stereoisomer + ">>" + rxn.split(">>")[1])
        except:
            continue

    return (valid_rxns, reverse)

def ff_generate_rxn_conformers_no_chiral(input_dict):
    '''
    Generate and optimize reaction conformers with RDKit, without accepting more than one stereoisomer

    Args:
        input_dict: dict containing the reaction and whether it is reversible

    Returns:
        conformers: list of pysisyphus.Geometry
    '''
    rxn = input_dict['rxn']
    reverse = input_dict['reverse']
    reactant_canon = input_dict['reactant_canon']
    rxn_confs = input_dict['rxn_confs']
    num_confs = input_dict['n_confs']
    reactant_geom_mol = converter.make_mol(rxn.split(">>")[0])
    product_geom_mol = converter.make_mol(rxn.split(">>")[1])
    # Generate conformers for reactant with RDKit but remove any conformers that have an RMSD less than the cutoff
    try:
        AllChem.EmbedMultipleConfs(reactant_geom_mol, numConfs=num_confs, pruneRmsThresh=0.5, useExpTorsionAnglePrefs=False, useBasicKnowledge=False)
        AllChem.UFFOptimizeMoleculeConfs(reactant_geom_mol, maxIters=100, ignoreInterfragInteractions=False)
    except:
        return None

    # Assign coordinates to product molecule and ff optimize
    for i in range(reactant_geom_mol.GetNumConformers()):
        product_geom_mol.AddConformer(reactant_geom_mol.GetConformer(i), assignId=True)

    AllChem.UFFOptimizeMoleculeConfs(product_geom_mol, maxIters=1000, ignoreInterfragInteractions=False, numThreads=0)

    # Make a list of tuples like [(1,1), (2,2)...] for the number of atoms (to use for mapping for rdkit)
    atom_nums = []
    for i in range(reactant_geom_mol.GetNumAtoms()):
        atom_nums.append((i, i))

    # Evaluete the RMSD of each reactant/product conformer pair
    rmsd = []
    for i in range(reactant_geom_mol.GetNumConformers()):
        rmsd.append(rdMolAlign.AlignMol(product_geom_mol, reactant_geom_mol, prbCid=i, refCid=i, atomMap=atom_nums))

    # Order the rmsd list and grab the rxn_confs*10 lowest rmsd conformers
    rmsd, indices = zip(*sorted(zip(rmsd, range(len(rmsd)))))

    index_list = []
    if reverse:
        for index in indices:
            if converter.check_canon(product_geom_mol, index, reactant_canon):
                index_list.append(index)
            if len(index_list) >= rxn_confs*10:
                break
    else:
        for index in indices:
            index_list.append(index)
            if len(index_list) >= rxn_confs*10:
                break

    # Assemble a list of dicts to return
    ret_list = []
    ret_list.append({"mols": (reactant_geom_mol, product_geom_mol),"og_indices": index_list})

    return ret_list

def ff_generate_rxn_conformers(input_dict):
    '''
    Generate and optimize reaction conformers with RDKit

    Args:
        input_dict: dict containing the reaction and whether it is reversible

    Returns:
        conformers: list of pysisyphus.Geometry
    '''
    rxn = input_dict['rxn']
    reverse = input_dict['reverse']
    reactant_canon = input_dict['reactant_canon']
    rxn_confs = input_dict['rxn_confs']
    num_confs = input_dict['n_confs']
    reactant_geom_mol = converter.make_mol(rxn.split(">>")[0])
    product_geom_mol = converter.make_mol(rxn.split(">>")[1])
    # Generate conformers for reactant with RDKit but remove any conformers that have an RMSD less than the cutoff
    try:
        AllChem.EmbedMultipleConfs(reactant_geom_mol, numConfs=num_confs, pruneRmsThresh=0.5, useExpTorsionAnglePrefs=False, useBasicKnowledge=False)
        AllChem.UFFOptimizeMoleculeConfs(reactant_geom_mol, maxIters=100, ignoreInterfragInteractions=False)
    except:
        return None

    # Assign coordinates to product molecule and ff optimize
    for i in range(reactant_geom_mol.GetNumConformers()):
        product_geom_mol.AddConformer(reactant_geom_mol.GetConformer(i), assignId=True)

    AllChem.UFFOptimizeMoleculeConfs(product_geom_mol, maxIters=1000, ignoreInterfragInteractions=False, numThreads=0)

    # Make a list of tuples like [(1,1), (2,2)...] for the number of atoms (to use for mapping for rdkit)
    atom_nums = []
    for i in range(reactant_geom_mol.GetNumAtoms()):
        atom_nums.append((i, i))

    # Evaluete the RMSD of each reactant/product conformer pair
    rmsd = []
    for i in range(reactant_geom_mol.GetNumConformers()):
        rmsd.append(rdMolAlign.AlignMol(product_geom_mol, reactant_geom_mol, prbCid=i, refCid=i, atomMap=atom_nums))

    # Order the rmsd list and grab the rxn_confs*10 lowest rmsd conformers
    rmsd, indices = zip(*sorted(zip(rmsd, range(len(rmsd)))))

    index_dict = {}
    if reverse:
        for index in indices:
            if converter.check_canon(product_geom_mol, index, reactant_canon):
                reactant_canon = converter.get_canon(reactant_geom_mol, index)
                if reactant_canon in index_dict.keys():
                    if len(index_dict[reactant_canon]) < rxn_confs*10:
                        index_dict[reactant_canon].append(index)
                else:
                    index_dict[reactant_canon] = [index]
    else:
        for index in indices:
            product_canon = converter.get_canon(product_geom_mol, index)
            if product_canon in index_dict.keys():
                if len(index_dict[product_canon]) < rxn_confs*10:
                    index_dict[product_canon].append(index)
            else:
                index_dict[product_canon] = [index]

    # Assemble a list of dicts to return
    ret_list = []
    for key in index_dict.keys():
        ret_list.append({"mols": (reactant_geom_mol, product_geom_mol),"og_indices": index_dict[key]})

    return ret_list

def calc_prod_energy(input_dict, calc):
    '''
    Calculate the energy of a product with pysisyphus

    Args:
        input_dict: dict containing the product conformers
        calc: pysisyphus.Calculator

    Returns:
        energy: float
    '''
    conformers = input_dict['conformers']
    if conformers is None:
        return None
    canon_smiles = input_dict['canon_smiles']
    # Optimize each conformer and take the lowest energy conformer
    energies = []
    for geom in conformers:
        opt_geom = optimize_geometry(geom, calc)
        if opt_geom is None:
            continue
        try:
            canon_mol = converter.geom2rdmol(opt_geom)
        except:
            continue
        canon_check = Chem.CanonSmiles(Chem.MolToSmiles(canon_mol), useChiral=0)
        if canon_check == canon_smiles:
            energies.append(opt_geom.energy)

    if len(energies) == 0:
        return None

    return min(energies)

def calc_rxn_conformer(input_dict, calc):
    '''
    Optimizes a set of reaction conformers and returns the conformers in order of RMSD

    Args:
        input_dict: dict containing the reactant and product conformers
        calc: pysisyphus.Calculator

    Returns:
        conformers: list of pysisyphus.Geometry
    '''
    mols = input_dict['mols']
    og_indices = input_dict['og_indices']
    reactant_geom_mol = mols[0]
    product_geom_mol = mols[1]

    # Optimize the conformers
    for i in og_indices:
        reactant_geom = converter.rdmol2geom(reactant_geom_mol, i)
        product_geom = converter.rdmol2geom(product_geom_mol, i)
        reactant_geom = optimize_geometry(reactant_geom, calc)
        product_geom = optimize_geometry(product_geom, calc)
        reactant_geom_mol = converter.add_geom_rdmol(reactant_geom, reactant_geom_mol, i)
        product_geom_mol = converter.add_geom_rdmol(product_geom, product_geom_mol, i)

    # Make a list of tuples like [(1,1), (2,2)...] for the number of atoms (to use for mapping for rdkit)
    atom_nums = []
    for i in range(reactant_geom_mol.GetNumAtoms()):
        atom_nums.append((i, i))

    # Calculate the RMSD of each conformer (while aligning the product to the reactant)
    rmsd = []
    conformers = []
    for i in og_indices:
        rmsd.append(rdMolAlign.AlignMol(product_geom_mol, reactant_geom_mol, prbCid=i, refCid=i, atomMap=atom_nums))
        conformers.append((converter.rdmol2geom(reactant_geom_mol, i), converter.rdmol2geom(product_geom_mol, i)))

    # Order the the conformers by RMSD
    rmsd, conformers = zip(*sorted(zip(rmsd, conformers)))
    return conformers

def calc_neb(input_dict, calc):
    '''
    Run a NEB calculation with pysisyphus

    Args:
        input_dict: dict containing the reactant and product conformers
        calc: pysisyphus.Calculator

    Returns:
        ts_guess: pysisyphus.Geometry
    '''
    geoms = input_dict["geoms"]
    for geom in geoms:
        geom.set_calculator(calc)

    # Run interpolation
    try:
        #with contextlib.redirect_stdout(open(os.devnull, 'w', encoding='utf-8')):
        with open(os.devnull, "w", encoding='utf-8') as f, contextlib.redirect_stdout(f):
            all_geoms = interpolate(geoms[0], geoms[1], 20, kind= "idpp")
    except:
        return None

    for geom in all_geoms:
        geom.set_calculator(calc)

    # Establish NEB parameters
    cos = AdaptiveNEB(all_geoms, keep_hei=True, adapt=False, k_min=0.5, k_max=1.5)

    # Run NEB
    #with contextlib.redirect_stdout(open(os.devnull, 'w', encoding='utf-8')):
    with open(os.devnull, "w", encoding='utf-8') as f, contextlib.redirect_stdout(f): 
        opt_kwargs = {
            "max_step": 0.04,
            "rms_force": 0.01,
            "coord_diff_thresh": 0.0001,
            "max_cycles": 300,
        }
        opt = LBFGS(cos, **opt_kwargs)

    with torch.jit.optimized_execution(False):
        try:
            #with contextlib.redirect_stdout(open(os.devnull, 'w', encoding='utf-8')):
            with open(os.devnull, "w", encoding='utf-8') as f, contextlib.redirect_stdout(f):
                opt.run()
        except:
            return None

    # Check whether results converged
    if opt.is_converged is False:
        return None

    # Return highest energy image
    hei_geom = cos.images[cos.get_hei_index()]
    hei_geom.set_calculator(None)
    return hei_geom

def calc_ts(input_dict, calc):
    '''
    Optimize a transition state geometry with pysisyphus

    Args:
        input_dict: dict containing the reactant and product conformers
        calc: pysisyphus.Calculator

    Returns:
        ts_geom: pysisyphus.Geometry
        ts_energy: float
    '''
    ts_guess = input_dict["ts_guess"]
    ts_guess.set_calculator(calc)
    tsopt = RSPRFOptimizer(ts_guess, 
                           thresh="gau_vtight",
                           max_cycles=300, 
                           hessian_recalc=5, 
                           trust_max=0.075, 
                           trust_min=0.00001, 
                           assert_neg_eigval=True)

    with torch.jit.optimized_execution(False):
        try:
            #with contextlib.redirect_stdout(open(os.devnull, 'w', encoding='utf-8')):
            with open(os.devnull, "w", encoding='utf-8') as f, contextlib.redirect_stdout(f): 
                tsopt.run()
        except:
            return None

    # Check whether results converged
    if tsopt.is_converged is False:
        return None

    # Return optimized transition state
    ts_geom = tsopt.geometry
    ts_energy = ts_geom.energy
    ts_geom.set_calculator(None)
    return ts_geom, ts_energy

def calc_irc(input_dict, calc):
    '''
    Run an IRC calculation with pysisyphus

    Args:
        input_dict: dict containing the transition state geometry
        calc: pysisyphus.Calculator

    Returns:
        irc_geoms: list of pysisyphus.Geometry
    '''
    ts_geom = input_dict["ts_geom"]
    max_cycles = 300
    irc_kwargs = {
        "type": "eulerpc",
        "max_cycles": max_cycles,
    }
    irc_key = irc_kwargs.pop("type")
    with torch.jit.optimized_execution(False):
        #with contextlib.redirect_stdout(open(os.devnull, 'w', encoding='utf-8')):
        with open(os.devnull, "w", encoding='utf-8') as f, contextlib.redirect_stdout(f):
            try:
                irc = run_irc(ts_geom, irc_key, irc_kwargs, lambda: calc)
            except:
                return None

    # Check whether results converged
    if irc.converged is False:
        return None
    irc_geoms = irc.get_endpoint_and_ts_geoms()
    for geom in irc_geoms:
        geom.set_calculator(None)
    irc_energy = (input_dict["energy"])
    irc_data = (irc_geoms, irc_energy)
    return irc_data

def calc_opt_irc_ends(input_dict, calc):
    '''
    Optimize the endpoints of an IRC calculation with pysisyphus

    Args:
        input_dict: dict containing the reactant and product conformers
        calc: pysisyphus.Calculator

    Returns:
        irc_geoms: list of pysisyphus.Geometry
    '''
    # Load geometry
    reactant_geom = input_dict['reactant_geom']
    product_geom = input_dict['product_geom']
    ts_geom = input_dict['ts_geom']
    reactant_canon = input_dict['reactant_canon']
    product_canon = input_dict['product_canon']
    ts_energy = input_dict['ts_energy']

    # Run optimization
    try:
        opt_reactant_geom = optimize_geometry(reactant_geom, calc)
        opt_product_geom = optimize_geometry(product_geom, calc)
        test_reactant_smiles = Chem.CanonSmiles(Chem.MolToSmiles(converter.geom2rdmol(opt_reactant_geom)))
        test_product_smiles = Chem.CanonSmiles(Chem.MolToSmiles(converter.geom2rdmol(opt_product_geom)))
    except:
        return None
    if test_reactant_smiles == reactant_canon and test_product_smiles == product_canon:
        return {"reactant_canon": reactant_canon, "product_canon": product_canon, "ts_energy": ts_energy, "ts_geom": ts_geom}
    return None
