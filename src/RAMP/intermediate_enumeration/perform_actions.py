"""Module to perform actions on a molecule"""

import os
from enum import IntEnum
from copy import deepcopy
import multiprocessing as mp
from functools import partial
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from RAMP.intermediate_enumeration.enumerate_intermediates import closed_shell_enumeration



sanitizeOps = Chem.rdmolops.SanitizeFlags.SANITIZE_ALL ^ \
    Chem.rdmolops.SanitizeFlags.SANITIZE_SETAROMATICITY ^ \
    Chem.rdmolops.SanitizeFlags.SANITIZE_FINDRADICALS ^ \
    Chem.rdmolops.SanitizeFlags.SANITIZE_ADJUSTHS

class Action(IntEnum):
    """The possible action space"""
    BREAK_BOND_WITH_0_ELECTRONS = -1
    BREAK_BOND_WITH_1_ELECTRON = -2
    BREAK_BOND_WITH_2_ELECTRONS = -3
    PAIR_ORBITALS = -4
    INTERACT_ORBITAL_WITH_BOND = -10
    SINGLE_ELECTRON_TRANSFER = -5
    STOP = -11


def decrement_num_radicals(mol, atom):
    '''
    Decrements the number of radicals on an atom by one

    Args:
        mol (rdkit.Chem.Mol): the molecule
        atom (int): the atom index

    Returns:
        None
    '''
    num_radicals = mol.GetAtomWithIdx(atom).GetNumRadicalElectrons()
    if num_radicals <= 0:
        raise ValueError("Invalid number of radicals!")
    mol.GetAtomWithIdx(atom).SetNumRadicalElectrons(num_radicals - 1)

def increment_num_radicals(mol, atom):
    '''
    Increments the number of radicals on an atom by one

    Args:
        mol (rdkit.Chem.Mol): the molecule
        atom (int): the atom index

    Returns:
        None
    '''
    num_radicals = mol.GetAtomWithIdx(atom).GetNumRadicalElectrons()
    mol.GetAtomWithIdx(atom).SetNumRadicalElectrons(num_radicals + 1)

def decrement_formal_charge(mol, atom):
    '''
    Decrements the formal charge on an atom by one

    Args:
        mol (rdkit.Chem.Mol): the molecule
        atom (int): the atom index

    Returns:
        None
    '''
    charge = mol.GetAtomWithIdx(atom).GetFormalCharge()
    mol.GetAtomWithIdx(atom).SetFormalCharge(charge - 1)

def increment_formal_charge(mol, atom):
    '''
    Increments the formal charge on an atom by one

    Args:
        mol (rdkit.Chem.Mol): the molecule
        atom (int): the atom index

    Returns:
        None
    '''
    charge = mol.GetAtomWithIdx(atom).GetFormalCharge()
    mol.GetAtomWithIdx(atom).SetFormalCharge(charge + 1)

def increment_bond_order(mol, first_atom, second_atom):
    '''
    Increments the bond order between two atoms

    Args:
        mol (rdkit.Chem.Mol): the molecule
        first_atom (int): the index of the first atom
        second_atom (int): the index of the second atom

    Returns:
        None
    '''
    current_bond = mol.GetBondBetweenAtoms(first_atom, second_atom)
    if current_bond is None:
        mol.AddBond(first_atom, second_atom, Chem.rdchem.BondType.SINGLE)
    elif current_bond.GetBondType() is Chem.rdchem.BondType.TRIPLE:
        raise ValueError("Cannot increment bond order beyond triple!")
    elif current_bond.GetBondType() is Chem.rdchem.BondType.DOUBLE:
        mol.RemoveBond(first_atom, second_atom)
        mol.AddBond(first_atom, second_atom, Chem.rdchem.BondType.TRIPLE)

    elif current_bond.GetBondType() is Chem.rdchem.BondType.SINGLE:
        mol.RemoveBond(first_atom, second_atom)
        mol.AddBond(first_atom, second_atom, Chem.rdchem.BondType.DOUBLE)
    else:
        raise ValueError("Unknown Bond Type!")

def decrement_bond_order(mol, first_atom, second_atom):
    '''
    Decrements the bond order between two atoms

    Args:
        mol (rdkit.Chem.Mol): the molecule
        first_atom (int): the index of the first atom
        second_atom (int): the index of the second atom

    Returns:
        None
    '''
    current_bond = mol.GetBondBetweenAtoms(first_atom, second_atom)
    if current_bond is None:
        raise ValueError("No bond exists to unpair!")
    if current_bond.GetBondType() is Chem.rdchem.BondType.TRIPLE:
        mol.RemoveBond(first_atom, second_atom)
        mol.AddBond(first_atom, second_atom, Chem.rdchem.BondType.DOUBLE)
    elif current_bond.GetBondType() is Chem.rdchem.BondType.DOUBLE:
        mol.RemoveBond(first_atom, second_atom)
        mol.AddBond(first_atom, second_atom, Chem.rdchem.BondType.SINGLE)
    elif current_bond.GetBondType() is Chem.rdchem.BondType.SINGLE:
        mol.RemoveBond(first_atom, second_atom)
    else:
        raise ValueError("Unknown Bond Type!")


def perform_actions_on_mol(actions, mol, in_place=False):
    '''
    Perform a list of actions on a molecule

    Args:
        actions (list): list of actions to perform
        mol (rdkit.Chem.Mol): the molecule
        in_place (bool): if True, perform the actions in place

    Returns:
        str: the SMILES string of the modified molecule
    '''
    rwmol = mol.molecule
    if not in_place:
        rwmol = deepcopy(rwmol)
    elec_dict = {}
    for i, action in enumerate(actions):
        first_atom = mol.get_vo_by_idx(action[1]).atom.idx
        second_atom = mol.get_vo_by_idx(action[2]).atom.idx
        if action[0] == Action.PAIR_ORBITALS:
            increment_bond_order(rwmol, first_atom, second_atom)
            if i == 0:
                num_electrons = mol.get_vo_by_idx(action[1]).num_electrons
            else:
                num_electrons = elec_dict[action[1]]
            if num_electrons == 0: # Atom one removes cation, atom two removes anion
                decrement_formal_charge(rwmol, first_atom)
                increment_formal_charge(rwmol, second_atom)
            elif num_electrons == 1: # Both atoms lose radical character
                decrement_num_radicals(rwmol, first_atom)
                decrement_num_radicals(rwmol, second_atom)
            elif num_electrons == 2: # Atom one removes anion, atom two removes cation
                increment_formal_charge(rwmol, first_atom)
                decrement_formal_charge(rwmol, second_atom)
            else:
                raise ValueError("Not valid number of electrons!")
        else:
            decrement_bond_order(rwmol, first_atom, second_atom)
            if action[0] == Action.BREAK_BOND_WITH_0_ELECTRONS: # Atom one becomes cation, atom two becomes anion
                increment_formal_charge(rwmol, first_atom)
                decrement_formal_charge(rwmol, second_atom)
                elec_dict[action[1]] = 0
                elec_dict[action[2]] = 2
            elif action[0] == Action.BREAK_BOND_WITH_1_ELECTRON: # Both atoms become radical
                increment_num_radicals(rwmol, first_atom)
                increment_num_radicals(rwmol, second_atom)
                elec_dict[action[1]] = 1
                elec_dict[action[2]] = 1
            elif action[0] == Action.BREAK_BOND_WITH_2_ELECTRONS: # Atom one becomes anion, atom two becomes cation
                decrement_formal_charge(rwmol, first_atom)
                increment_formal_charge(rwmol, second_atom)
                elec_dict[action[1]] = 2
                elec_dict[action[2]] = 0
            else:
                raise ValueError("Action not supported!")
    # Sanitize molecule and return smiles string
    Chem.SanitizeMol(rwmol, sanitizeOps=sanitizeOps)
    return Chem.MolToSmiles(rwmol)

def process_actions(mol, subset_actions, workers):
    '''
    Process a subset of actions on a molecule in parallel

    Args:
        mol (rdkit.Chem.Mol): the molecule
        subset_actions (list): list of actions to perform
        workers (int): number of workers to use

    Returns:
        list: list of SMILES strings of the modified molecules
    '''
    chunksize = int(len(subset_actions)/workers)
    if chunksize == 0:
        chunksize = 1
    perform_actions_with_mol = partial(perform_actions_on_mol, mol=mol)
    with mp.Pool(workers) as pool:
        result = pool.map(perform_actions_with_mol, subset_actions, chunksize=chunksize)
    return result

def write_smi_from_actions(mol, all_actions, workers, batch_size, outfile):
    '''
    Write the SMILES strings of the molecules resulting from a list of actions to a file

    Args:
        mol (rdkit.Chem.Mol): the molecule
        all_actions (list): list of actions to perform
        workers (int): number of workers to use
        batch_size (int): size of the batch to process
        outfile (str): path to the output file

    Returns:
        None
    '''
    smiles = []
    batch = []
    for i in range(int(len(all_actions)/batch_size) + 1):
        if (i+1)*batch_size > len(all_actions):
            batch.append(all_actions[i*batch_size:])
        else:
            batch.append(all_actions[i*batch_size:(i+1)*batch_size])
    first_batch = True
    for subset_actions in tqdm(batch):
        smiles = process_actions(mol, subset_actions, workers)
        df = pd.DataFrame({"smiles": smiles})
        if first_batch:
            df.to_csv("tmp.csv", mode='w', index=False)
        else:
            df.to_csv("tmp.csv", mode='a', index=False, header=False)
        first_batch = False

    # Now remove reactions where the reactant and product are the same
    reactant_mol = mol.molecule
    reactant_canon = Chem.CanonSmiles(Chem.MolToSmiles(reactant_mol), useChiral=0)
    with open("tmp.csv", 'r', encoding='utf-8') as f:
        lines = f.readlines()
    os.system("rm tmp.csv")
    lines = lines[1:]
    products = []
    for line in tqdm(lines):
        if line == "\"\"\n":
            break
        product = line
        product_mol = Chem.MolFromSmiles(product)
        if Chem.CanonSmiles(Chem.MolToSmiles(product_mol), useChiral=0) != reactant_canon:
            products.append(line)
    with open(outfile, 'w', encoding='utf-8') as f:
        f.write("smiles\n")
        for product in products:
            f.write(product)

def remove_duplicates(file):
    '''
    Removes smiles from the file that result in the same canonical smiles

    Args:
        file (str): path to the file

    Returns:
        None
    '''
    with open(file, 'r', encoding='utf-8') as f:
        mapped_smiles = f.read().split("\n")[1:-1]
    # Make a new list to store the canonical smiles for all the intermediates
    canonical_smiles = []
    for smi in mapped_smiles:
        canonical_smiles.append(Chem.CanonSmiles(smi))

    # Remove any duplicates from the canonical_smiles list with a set
    canonical_smiles_new = list(set(canonical_smiles))

    # Write the intermediates associated with the non-duplicate canonical smiles to a the file
    with open(file, 'w', encoding='utf-8') as f:
        f.write("smiles\n")
        for smi in canonical_smiles_new:
            f.write(mapped_smiles[canonical_smiles.index(smi)] + "\n")


if __name__ == '__main__':
    REAC_SMI = 'OCC1OC(O)C(O)C(O)C1O'
    actions_init, mol_init = closed_shell_enumeration(REAC_SMI, most_bonds_broken=4)
    write_smi_from_actions(mol_init, actions_init, workers=mp.cpu_count(), batch_size=mp.cpu_count()*1000, outfile="temp_smiles.csv")
