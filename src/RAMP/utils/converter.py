"""Module for converting between various molecule representations"""

from copy import copy
from rdkit import Chem
from rdkit.Chem import EnumerateStereoisomers, rdDetermineBonds
from pysisyphus.Geometry import Geometry
from pysisyphus.elem_data import INV_ATOMIC_NUMBERS
from pysisyphus.constants import ANG2BOHR
import numpy as np

def xyz2rdmol(xyz_file):
    '''
    Convert an xyz file to a rdkit molecule

    Args:
        xyz_file: path to the xyz file

    Returns:
        mol: rdkit.Chem.Mol
    '''
    raw_mol = Chem.MolFromXYZFile(xyz_file)
    mol = Chem.Mol(raw_mol)
    rdDetermineBonds.DetermineBonds(mol,charge=0)
    return mol

def pybel2geom(mol, coord_type='cart'):
    '''
    Convert a pybel molecule to a pysisyphus geometry

    Args:
        mol: pybel.Molecule
        coord_type: 'cart' currently only option

    Returns:
        geom: pysisyphus.Geometry
    '''
    coord = np.array([a.coords for a in mol.atoms]).flatten() * ANG2BOHR
    atoms = [INV_ATOMIC_NUMBERS[a.atomicnum].lower() for a in mol.atoms]
    geom = Geometry(atoms, coord, coord_type=coord_type)
    return geom

def rdmol2geom(mol, conf_id, coord_type='cart'):
    '''
    Convert a rdkit molecule to a pysisyphus geometry

    Args:
        mol: rdkit.Chem.Mol
        confId: int, conformer ID
        coord_type: 'cart' currently only option

    Returns:
        geom: pysisyphus.Geometry
    '''
    conf = mol.GetConformer(conf_id)
    coord = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())]).flatten() * ANG2BOHR
    atoms = [INV_ATOMIC_NUMBERS[a.GetAtomicNum()].lower() for a in mol.GetAtoms()]
    geom = Geometry(atoms, coord, coord_type=coord_type)
    return geom

def geom2rdmol(geom):
    '''
    Convert a pysisyphus geometry to a rdkit molecule

    Args:
        geom: pysisyphus.Geometry

    Returns:
        mol: rdkit.Chem.Mol
    '''
    geom.dump_xyz("temp.xyz")
    raw_mol = Chem.MolFromXYZFile('temp.xyz')
    mol = Chem.Mol(raw_mol)
    rdDetermineBonds.DetermineBonds(mol,charge=0)
    return mol

def remove_mapping(mol, in_place=True):
    '''
    Remove atom mapping from a rdkit molecule

    Args:
        mol: rdkit.Chem.Mol
        in_place: bool, whether to modify the input molecule

    Returns:
        mol: rdkit.Chem.Mol
    '''
    if in_place:
        test_mol = mol
    else:
        test_mol = copy(mol)
    for atom in test_mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return test_mol


def assign_mapping_from_mapped_mol(mol, mapped_mol):
    '''
    Assign atom mapping from a mapped molecule to a rdkit molecule

    Args:
        mol: rdkit.Chem.Mol
        mapped_mol: rdkit.Chem.Mol

    Returns:
        mol: rdkit.Chem.Mol
    '''
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(mapped_mol.GetAtomWithIdx(atom.GetIdx()).GetAtomMapNum())
    return mol

def add_geom_rdmol(geom, mol, conf_id):
    '''
    Add geometry to a rdkit molecule

    Args:
        geom: pysisyphus.Geometry
        mol: rdkit.Chem.Mol
        conf_id: int, conformer ID

    Returns:
        mol: rdkit.Chem.Mol
    '''
    for i in range(mol.GetNumAtoms()):
        mol.GetConformer(conf_id).SetAtomPosition(i, geom.coords[3*i:3*i+3]/ANG2BOHR)
    return mol

def make_mol(smi):
    '''
    Initialize a rdkit molecule from a SMILES string while preserving atom mapping

    Args:
        smi: str, SMILES string

    Returns:
        mol: rdkit.Chem.Mol
    '''
    ps = Chem.SmilesParserParams()
    ps.removeHs = False
    og = Chem.MolFromSmiles(smi, ps)
    fake_map = []
    for atom in og.GetAtoms():
        if atom.GetAtomMapNum() == 0:
            fake_map.append(og.GetNumAtoms() - 1)
        else:
            fake_map.append(atom.GetAtomMapNum() - 1)
    indices_order = sorted(range(len(fake_map)), key=lambda x: fake_map[x])
    mol = Chem.RenumberAtoms(og, indices_order)
    return mol

def get_canon(mol_check, conf_id):
    '''
    Get the canonical SMILES string for a molecule

    Args:
        mol_check: rdkit.Chem.Mol
        conf_id: int, conformer ID
    
    Returns:
        canon: str, canonical SMILES string
    '''
    block = Chem.MolToXYZBlock(mol_check, conf_id)
    raw_mol = Chem.MolFromXYZBlock(block)
    mol = Chem.Mol(raw_mol)
    try:
        rdDetermineBonds.DetermineBonds(mol,charge=0)
    except ValueError:
        return None
    return Chem.CanonSmiles(Chem.MolToSmiles(mol))

def check_canon(mol_check, conf_id, canon):
    '''
    Check if the canonical SMILES string of a molecule is the same as a given canonical SMILES string

    Args:
        mol_check: rdkit.Chem.Mol
        conf_id: int, conformer ID
        canon: str, canonical SMILES string

    Returns:
        bool
    '''
    block = Chem.MolToXYZBlock(mol_check, conf_id)
    raw_mol = Chem.MolFromXYZBlock(block)
    mol = Chem.Mol(raw_mol)
    try:
        rdDetermineBonds.DetermineBonds(mol,charge=0)
    except ValueError:
        return False
    return Chem.CanonSmiles(Chem.MolToSmiles(mol)) == canon

def generate_stereoisomers(prod_smi, reac_smi):
    '''
    Generate stereoisomers for a product while constraining stereochemistry based on the reactant

    Args:
        prod_smi: str, product SMILES string
        reac_smi: str, reactant SMILES string

    Returns:
        list of stereoisomers
    '''
    # Generate reactant and product moleculesS
    reactant = make_mol(reac_smi)
    product = make_mol(prod_smi)
    # Assign chiral tags from reactant to product
    for chiral_atom in Chem.FindMolChiralCenters(product, includeUnassigned=True):
        # Find the atoms that the chiral atom is connected to
        neighbors = [x.GetIdx() for x in product.GetAtomWithIdx(chiral_atom[0]).GetNeighbors()]
        reac_neighbors = [x.GetIdx() for x in reactant.GetAtomWithIdx(chiral_atom[0]).GetNeighbors()]
        # If connectivity is the same, assign the same chiral tag to the product (if it exists)
        if neighbors == reac_neighbors:
            reac_tag = reactant.GetAtomWithIdx(chiral_atom[0]).GetChiralTag()
            if str(reac_tag) != "CHI_UNSPECIFIED":
                product.GetAtomWithIdx(chiral_atom[0]).SetChiralTag(reac_tag)

    # Assign double bond stereochemistry from reactant to product
    for bond in product.GetBonds():
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            # Check if the same bond exists in the reactant
            reac_bond = reactant.GetBondBetweenAtoms(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            if reac_bond is not None:
                if reac_bond.GetBondType() == Chem.BondType.DOUBLE:
                    # Check if the bond is E/Z in the reactant. If so, check the neighbors of the atoms in the bond
                    reac_bond_stereo = reac_bond.GetStereo()
                    if reac_bond_stereo in [Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ]:
                        # Check if the neighbors are the same in the product
                        reac_bond_neighbors_start = [x.GetIdx() for x in bond.GetBeginAtom().GetNeighbors()]
                        reac_bond_neighbors_end = [x.GetIdx() for x in bond.GetEndAtom().GetNeighbors()]
                        prod_bond_neighbors_start = [x.GetIdx() for x in reac_bond.GetBeginAtom().GetNeighbors()]
                        prod_bond_neighbors_end = [x.GetIdx() for x in reac_bond.GetEndAtom().GetNeighbors()]
                        if reac_bond_neighbors_start == prod_bond_neighbors_start and reac_bond_neighbors_end == prod_bond_neighbors_end:
                            # If the neighbors are the same, set the stereochemistry of the bond to be the same as the reactant
                            bond.SetStereo(reac_bond_stereo)
                            atoms = []
                            for atom in reac_bond.GetStereoAtoms():
                                atoms.append(atom)
                            bond.SetStereoAtoms(atoms[0], atoms[1])

    # Generate stereoisomers
    test_prod_mol = remove_mapping(product, in_place=False)
    opts = EnumerateStereoisomers.StereoEnumerationOptions(unique=True, onlyUnassigned=True)
    stereo_enumerate = tuple(EnumerateStereoisomers.EnumerateStereoisomers(test_prod_mol, options=opts))
    stereoisomers = []
    for mol in stereo_enumerate:
        stereoisomers.append(Chem.MolToSmiles(assign_mapping_from_mapped_mol(mol, product)))

    return stereoisomers
