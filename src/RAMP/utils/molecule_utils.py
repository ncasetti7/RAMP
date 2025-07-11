"""Module for various molecule utilities"""

import os
import contextlib
from pathlib import Path
import logging
import subprocess
import shutil
from typing import List, Tuple, Optional
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit import Chem
import numpy as np
from scipy.spatial import distance_matrix
import networkx as nx

HARTREE_TO_EV = 27.2114

@contextlib.contextmanager
def make_tmp_directory():
    '''
    Makes a temporary directory to do things in, and then reverts back on exit.
    
    Args:
        None

    Returns:
        None
    '''
    prev_cwd = Path.cwd()
    if not os.path.exists('tmp_{}'.format(os.getpid())):
        os.mkdir('tmp_{}'.format(os.getpid()))
    os.chdir('tmp_{}'.format(os.getpid()))
    try:
        yield
    finally:
        os.chdir(prev_cwd)
        shutil.rmtree('tmp_{}/'.format(os.getpid()))


def one_hot_encode(index: int, n_dimensions: int, offset: int = 0) -> np.ndarray:
    '''
    Given an input index, returns a one-hot encoded array of length n_dimensions
    Subtracts offset from index, so use offset = min(values) to encode negative values.

    Args:
        index: int, index to encode
        n_dimensions: int, length of one-hot encoded array
        offset: int, offset to subtract from index

    Returns:
        one_hot: np.ndarray, one-hot encoded array
    '''
    if index < offset:
        raise ValueError("Cannot one-hot encode this value, as it is smaller than offset!")
    if index - offset >= n_dimensions:
        raise ValueError("Cannot one-hot encode this value, as it is greater than n_dimensions!")
    one_hot = np.zeros(n_dimensions)
    one_hot[index - offset] = 1
    return one_hot


def atom_to_num_vos(atom_symbol: str) -> int:
    '''
    Returns the number of VOs the atom should be initialized with.
    
    Args:
        atom_symbol: str, atom symbol

    Returns:
        int, number of VOs
    '''
    atom_to_vo_map = {'H': 1, 'He': 1, 'S': 6, 'P': 5}
    return atom_to_vo_map.get(atom_symbol, 4)


def output_3d_coords(atoms: List[str], atom_coords: List[Tuple[float, float, float]], output_format: str = 'xyz') -> str:
    '''
    Returns the coordinates of a molecule in the specified output format.

    Supported Output Formats: 'xyz', 'turbo'

    Args:
        atoms: List of atom symbols
        atom_coords: List of atom coordinates
        output_format: str, output format

    Returns:
        str, formatted coordinates
    '''
    if output_format not in {'xyz', 'turbo'}:
        raise ValueError('Invalid output format. Supported formats: xyz, turbo')

    logging.info('Outputting coordinates in {} format'.format(output_format))
    if output_format == 'xyz':
        output = str(len(atoms)) + '\n\n'
        for atom, (x, y, z) in zip(atoms, atom_coords):
            output += ' '.join([atom, str(x), str(y), str(z)]) + '\n'

    elif output_format == 'turbo':
        output = '$coord'
        for atom, (x, y, z) in zip(atoms, atom_coords):
            output += '\n' + ' '.join([str(x), str(y), str(z), atom.lower()])
        output += '\n$end\n'

    return output


def canonicalize(smi: str) -> str:
    '''
    Returns the canonical smiles string of a molecule.

    Args:
        smi: str, smiles string

    Returns:
        str, canonical smiles string
    '''
    output_smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi), canonical=True, isomericSmiles=False)
    if output_smi is None:
        raise ValueError('Failed to canonicalize SMILES string')
    return output_smi

def smi_to_coords(smi: str, optimizer: str = 'rdkit') -> Tuple[Optional[List[str]], Optional[List[Tuple[float, float, float]]]]:
    '''
    Returns the atoms and coordinates of a molecule (given as a smiles string) as arrays.
    Returns None if the optimized geometry differs from the original geometry too much.

    Supported Geometry Optimizers: 'rdkit' (ETKDG method), 'xtb' (GFN2-xTB method)

    Args:
        smi: str, smiles string
        optimizer: str, geometry optimizer

    Returns:
        atoms: List of atom symbols
        atom_coords: List of atom coordinates
    '''
    if optimizer not in {'rdkit', 'xtb'}:
        raise ValueError('Invalid optimizer. Supported optimizers: rdkit, xtb')

    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=0xf00d)  # for reproducibility
    lines = Chem.MolToMolBlock(mol).split('\n')
    atoms = []
    atom_coords = []

    logging.info('Converting SMILES {} to coordinates'.format(smi))
    # string parsing RDKit mol block
    for idx, line in enumerate(lines):
        if idx < 3:
            continue
        elif idx == 3:
            num_atoms = mol.GetNumAtoms()
        elif idx <= 3 + num_atoms:
            x, y, z, atom = [c for c in line.split(' ') if len(c) > 0][:4]
            x, y, z = map(float, [x, y, z])
            atom_coords.append((x, y, z))
            atoms.append(atom)

    # geometry optimization beyond RDKit with XTB if specified
    if optimizer == 'xtb':
        if len(atoms) == 1:
            # no need for optimization on single atom
            return atoms, atom_coords

        logging.info('Optimizing geometry with xTB')
        charge = Chem.rdmolops.GetFormalCharge(mol)
        num_unpaired_electrons = Descriptors.NumRadicalElectrons(mol)
        xyzfile = output_3d_coords(atoms, atom_coords, output_format='xyz')

        with make_tmp_directory():
            with open("tmp.xyz", "w", encoding='utf-8') as f:
                f.write(xyzfile)

            command = "xtb tmp.xyz --opt loose --gfn 2 --chrg {} --uhf {}".format(charge, num_unpaired_electrons)
            subprocess.check_call(command.split(), stdout=open('xtblog.txt', 'w', encoding='utf-8'), stderr=open(os.devnull, 'w', encoding='utf-8'))

            with open("xtbopt.xyz", "r", encoding='utf-8') as f:
                lines = f.readlines()
                atoms = []
                atom_coords = []
                for line in lines[2:]:
                    atom, x, y, z = line.split()
                    x, y, z = map(float, [x, y, z])
                    atom_coords.append((x, y, z))
                    atoms.append(atom)

            for output_file in ['tmp.xyz', 'xtbopt.xyz', 'xtbopt.log', 'xtbtopo.mol', 'xtbrestart', 'wbo', 'tmp.ges', 'charges', 'xtblog.txt']:
                if os.path.exists(output_file):
                    os.remove(output_file)

    # check that geometry actually matches input line diagram
    computed_adj_matrix = np.zeros((len(atom_coords), len(atom_coords)))
    dist_matrix = distance_matrix(atom_coords, atom_coords)
    actual_adj_matrix = Chem.rdmolops.GetAdjacencyMatrix(mol)
    for idx1, atom1 in enumerate(atoms):
        for idx2, atom2 in enumerate(atoms):
            if idx1 == idx2:
                continue
            # expected bond length between the two atoms
            #bond_threshold = COVALENT_RADII[atom1] + COVALENT_RADII[atom2]
            bond_threshold = 0
            distance = dist_matrix[idx1][idx2]
            if distance < bond_threshold * 1.3: # slight tolerance
                computed_adj_matrix[idx1][idx2] = 1

    if np.any(computed_adj_matrix != actual_adj_matrix):
        return None, None

    return atoms, atom_coords


#cache.memoize(name='get_molecule_energy', tag='get_molecule_energy', expire=None, typed=True)
def get_molecule_energy(molecule: str, optimizer: str = 'rdkit', n_attempts=10) -> Optional[float]:
    '''
    Returns the (potential) energy of a single molecule (given as a smiles string) in hartree.
    Uses xTB (GFN2-xTB) calculations for energy calculation.
    Geometry optimization is done either by RDKit (ETKDG method) or xTB (GFN2-xTB method).

    Energy calculations are attempted for n_attempts times. If they all fail, then
    this function returns None.

    Args:
        molecule: str, smiles string
        optimizer: str, geometry optimizer
        n_attempts: int, number of attempts for energy calculation

    Returns:
        float, energy in hartree
    '''
    mol = Chem.MolFromSmiles(molecule)
    canonical_smi = Chem.MolToSmiles(mol, canonical=True)
    if canonical_smi != molecule:
        return get_molecule_energy(canonical_smi, optimizer=optimizer)

    if canonical_smi == '[H+]':
        return 0.0

    try:
        atoms, atom_coords = smi_to_coords(molecule, optimizer=optimizer)
        if not atoms or not atom_coords:
            return None
    except Exception as e:
        logging.warning('Error {} in xTB geometry opt for {}'.format(e, molecule))
        return None

    charge = Chem.rdmolops.GetFormalCharge(mol)
    num_unpaired_electrons = Descriptors.NumRadicalElectrons(mol)
    xyzfile = output_3d_coords(atoms, atom_coords, output_format='xyz')

    with make_tmp_directory():
        with open("tmp.xyz", "w", encoding='utf-8') as f:
            f.write(xyzfile)

        logging.info('Getting energy with xTB')
        energy = None
        command = "xtb tmp.xyz --gfn 2 --chrg {} --uhf {}".format(charge, num_unpaired_electrons)
        for _ in range(n_attempts):
            try:
                subprocess.check_call(command.split(), stdout=open('xtblog.txt', 'w', encoding='utf-8'), stderr=open(os.devnull, 'w', encoding='utf-8'))
                with open("xtblog.txt", "r", encoding='utf-8') as f:
                    lines = f.readlines()
                    for line in lines:
                        if 'TOTAL ENERGY' in line:
                            energy = float(line.split()[-3])

            except Exception as e:
                logging.warning('Error: {} in xTB energy calculation for {}'.format(e, molecule))
                continue

        for output_file in ['tmp.xyz', 'xtbopt.xyz', 'xtbopt.log', 'xtbtopo.mol', 'xtbrestart', 'wbo', 'tmp.ges', 'charges', 'xtblog.txt']:
            if os.path.exists(output_file):
                os.remove(output_file)

        if energy is None:
            return None

        logging.info('Energy of {} is {} hartree'.format(molecule, energy))
    return energy

def mol_to_nx(mol):
    '''
    Converts and rdkit mol into a networkx graph

    Args:
        mol: rdkit mol object

    Returns:
        G: networkx graph
    '''
    g = nx.Graph()

    for atom in mol.GetAtoms():
        g.add_node(atom.GetIdx(), label=(atom.GetAtomicNum(), atom.GetIdx()))

    for bond in mol.GetBonds():
        g.add_edge(bond.GetBeginAtomIdx(),
                   bond.GetEndAtomIdx())
    return g

def check_rd_nodes(node1, node2):
    '''
    Checks for node equivalency to speed up graph edit calculation

    Args:
        node1: node 1
        node2: node 2

    Returns:
        bool: True if nodes are equivalent, False otherwise
    '''
    if node1['label'][0] != node2['label'][0]:
        return False
    else:
        if node1['label'][0] == 1:
            return True
        else:
            return node1['label'][1] == node2['label'][1]

def get_system_graph_edit(reac, prod) -> Optional[float]:
    '''
    Returns the number of graph edits between the reactant and product of a system

    Args:
        reac: rdkit mol object, reactant
        prod: rdkit mol object, product

    Returns:
        dist: float, number of graph edits
    '''
    reactant_rdgraph = mol_to_nx(reac)
    product_rdgraph = mol_to_nx(prod)
    a1 = nx.adjacency_matrix(reactant_rdgraph)
    a2 = nx.adjacency_matrix(product_rdgraph)
    dist = np.abs((a1-a2)).sum() / 2
    return dist

def map_mol(smi: str):
    '''
    Maps the atoms of a molecule to a canonical ordering

    Args:
        smi: str, smiles string
    
    Returns:
        mol: rdkit mol object
    '''
    ps = Chem.SmilesParserParams()
    ps.removeHs = False
    og = Chem.MolFromSmiles(smi, ps)
    fake_map = []
    for atom in og.GetAtoms():
        fake_map.append(atom.GetAtomMapNum() - 1)
    real_map = [0] * og.GetNumAtoms()
    for i, atom in enumerate(fake_map):
        real_map[atom] = i
    return Chem.RenumberAtoms(og, real_map)

#@cache.memoize(name='get_system_energy', tag='get_system_energy', expire=None, typed=True)
def get_system_energy(smi: str, optimizer: str = 'rdkit', n_attempts: int = 10) -> Optional[float]:
    '''
    Returns the (potential) energy of the system (given as a smiles string) in eV.
    Uses xTB (GFN2-xTB) calculations for energy calculation.
    Geometry optimization is done either by RDKit (ETKDG method) or xTB (GFN2-xTB method).

    Energy calculations are attempted for n_attempts times. If they all fail, then
    this function returns None.

    Args:
        smi: str, smiles string
        optimizer: str, geometry optimizer
        n_attempts: int, number of attempts for energy calculation

    Returns:
        float, energy in eV
    '''
    molecules = smi.split('.')
    total_energy = 0.0

    for molecule in molecules:
        molecule_energy = get_molecule_energy(molecule, optimizer=optimizer, n_attempts=n_attempts)
        if molecule_energy is None:
            return None
        else:
            total_energy += molecule_energy

    return total_energy * HARTREE_TO_EV


def get_orca_energy(smi: str, optimizer: str = 'dft') -> float:
    '''
    Returns the (potential) energy of the system (given as a smiles string) in eV.

    Uses ORCA calculations for energy calculation. For geometry optimization,
    either RDKit (ETKDG method), xTB (GFN2-xTB method), or DFT can be used.

    Use arguments 'rdkit', 'xtb', and 'dft' to call such optimization routines.

    Args:
        smi: str, smiles string
        optimizer: str, geometry optimizer

    Returns:
        float, energy in eV
    '''
    molecules = smi.split('.')
    total_energy = 0.0

    for molecule in molecules:
        mol = Chem.MolFromSmiles(molecule)
        charge = Chem.rdmolops.GetFormalCharge(mol)
        num_unpaired_electrons = Descriptors.NumRadicalElectrons(mol)
        atoms, atom_coords = smi_to_coords(molecule, optimizer='xtb' if optimizer == 'xtb' else 'rdkit')
        xyzfile = output_3d_coords(atoms, atom_coords, output_format='xyz')

        energy = None
        with make_tmp_directory():
            with open("tmp.xyz", "w", encoding='utf-8') as f:
                f.write('! PBE0 ma-def2-SVP {}\n'.format('opt' if optimizer == 'dft' else ''))
                f.write('* xyz {} {}\n'.format(charge, num_unpaired_electrons + 1))
                for idx, line in enumerate(xyzfile.split('\n')):
                    if idx > 1:
                        f.write(line + '\n')
                f.write('*')

            command = "orca tmp.xyz"
            subprocess.check_call(command.split(), stdout=open('orca.log', 'w', encoding='utf-8'), stderr=open(os.devnull, 'w', encoding='utf-8'))

            with open("orca.log", "r", encoding='utf-8') as f:
                for line in f.readlines():
                    if "FINAL SINGLE POINT ENERGY" in line:
                        energy = float(line.split()[4])

            for output_file in ['orca.log', 'tmp.xyz', 'tmp_trj.xyz', 'tmp_property.txt', 'tmp.engrad', 'tmp.opt', 'tmp.gbw', 'tmp.ges', 'tmp.densities']:
                if os.path.exists(output_file):
                    os.remove(output_file)

        logging.info('Energy of {} is {} hartree'.format(molecule, energy))
        if energy is None:
            logging.warning('Energy calculation for {} failed, not adding to total energy'.format(molecule))
        total_energy += energy * HARTREE_TO_EV

    return total_energy
