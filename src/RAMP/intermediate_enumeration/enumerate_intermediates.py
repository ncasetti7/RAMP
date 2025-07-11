"""Module for enumerating intermediates of a reactant"""

from enum import IntEnum
from copy import deepcopy
from itertools import permutations, product, combinations
from rdkit import Chem
import numpy as np
import networkx as nx
from tqdm import tqdm
from RAMP.utils.molecule import Molecule

class Action(IntEnum):
    """The possible action space"""
    BREAK_BOND_WITH_0_ELECTRONS = -1
    BREAK_BOND_WITH_1_ELECTRON = -2
    BREAK_BOND_WITH_2_ELECTRONS = -3
    PAIR_ORBITALS = -4


class MoleculeLeaf:
    '''
    A class to represent a molecule leaf in the search tree
    '''
    def __init__(self, mol: 'Molecule', prev_actions: list) -> None:
        self.mol = mol
        bond_list = []
        bond_set = set()
        vo_0 = []
        vo_1 = []
        vo_2 = []
        for vo in mol.get_all_valence_orbitals():
            if vo.neighbor is not None:
                set_length = len(bond_set)
                bond_set.add(frozenset((vo.index, vo.neighbor.index)))
                if len(bond_set) != set_length:
                    bond_list.append((vo.index, vo.neighbor.index))
            else:
                if vo.num_electrons == 0:
                    vo_0.append(vo.index)
                elif vo.num_electrons == 1:
                    vo_1.append(vo.index)
                else:
                    vo_2.append(vo.index)
        self.bond_list = bond_list
        self.vo_0 = vo_0
        self.vo_1 = vo_1
        self.vo_2 = vo_2
        self.prev_actions = prev_actions
        self.legal_actions = None
        self.children = []

    def remove_action(self, action):
        '''Remove the given action from legal actions'''
        self.legal_actions.remove(action)

    def get_leaf_from_action(self, action) -> 'MoleculeLeaf':
        '''Get a new MoleculeLeaf by performing action on current leaf'''
        vo = self.mol.get_vo_by_idx(action[1])
        vo_other = self.mol.get_vo_by_idx(action[2])
        new_mol = vo.get_molec_result_of_action(vo_other, action[0])[0]
        if self.prev_actions is None:
            prev_actions = [action]
        else:
            prev_actions = deepcopy(self.prev_actions)
            prev_actions.append(action)
        return MoleculeLeaf(new_mol, prev_actions)

def edit_distance(mol1,mol2):
    """The edit distance between graphs, defined as the number of changes one
    needs to make to put the edge lists in correspondence.

    Args:
        mol1 : Molecule
        mol2 : Molecule

    Returns:
        edit_distance : distance between the two molecules
    """
    g1 = mol1.to_networkx()
    g2 = mol2.to_networkx()
    a1 = nx.adjacency_matrix(g1)
    a2 = nx.adjacency_matrix(g2)
    dist = np.abs((a1-a2)).sum() / 2
    return dist

def generate_bnfn_actions(unused_vo, unused_elec, illegal_vos, current_vo0, current_vo1, current_vo2):
    '''
    Generate all possible bnfn actions given the unused valence orbitals and electrons

    Args:
        unused_vo : list of unused valence orbitals
        unused_elec : list of unused electrons
        illegal_vos : list of valence orbitals that should not be used
        current_vo0 : list of current valence orbitals with 0 electrons
        current_vo1 : list of current valence orbitals with 1 electron
        current_vo2 : list of current valence orbitals with 2 electrons

    Returns:
        actions : list of all possible bnfn actions
    '''
    # Remove illegal VOs from current VO lists as to not make bonds to these VOs
    if unused_elec[0] == 1:
        temp_vo1 = set(deepcopy(current_vo1))
        current_vo1 = list(temp_vo1.difference(set(illegal_vos)))
    else:
        temp_vo0 = set(deepcopy(current_vo0))
        current_vo0 = list(temp_vo0.difference(set(illegal_vos)))
        temp_vo2 = set(deepcopy(current_vo2))
        current_vo2 = list(temp_vo2.difference(set(illegal_vos)))

    # Build set of final legal actions from updated lists
    actions = []
    for idx, vo in enumerate(unused_vo):
        num_elec = unused_elec[idx]
        if num_elec == 0:
            for vo2 in current_vo2:
                actions.append([(Action.PAIR_ORBITALS, vo, vo2)])
        elif num_elec == 1:
            for vo1 in current_vo1:
                actions.append([(Action.PAIR_ORBITALS, vo, vo1)])
        else:
            for vo0 in current_vo0:
                actions.append([(Action.PAIR_ORBITALS, vo, vo0)])

    return actions

def get_same_atom_vos(vo, leaf):
    '''
    Get all valence orbitals that are on the same atom as the given valence orbital

    Args:
        vo : index of valence orbital
        leaf : MoleculeLeaf object

    Returns:
        vos : list of valence orbitals on the same atom
    '''
    vos = []
    for v in leaf.mol.get_vo_by_idx(vo).atom.valence_orbitals:
        vos.append(v.index)
    return vos

def swap_positions(list_swap, pos1, pos2):
    '''
    Swap the positions of two elements in a list

    Args:
        list_swap : list to swap elements in
        pos1 : index of first element
        pos2 : index of second element

    Returns:
        list : list with elements swapped
    '''
    list_swap[pos1], list_swap[pos2] = list_swap[pos2], list_swap[pos1]
    return list_swap

def closed_shell_bnfn_actions(bond_list):
    '''
    Enumerate all possible bnfn actions for a closed shell molecule

    Args:
        bond_list : list of bonds in the molecule

    Returns:
        actions : list of all possible bnfn actions
    '''
    actions = []
    vo_combos = []
    breaking_actions = []
    for bond in bond_list:
        breaking_actions.append((Action.BREAK_BOND_WITH_1_ELECTRON, bond[0], bond[1]))
    # Grab all the sets of bonds that result in a bnfn-1 action
    all_bond_converses = []
    for bond in bond_list[1:]:
        all_bond_converses.append([(bond[0], bond[1]), (bond[1], bond[0])])
    original_combinations = list(product(*all_bond_converses))
    for combination in original_combinations:
        for bonds in permutations(combination):
            final_vo_list = [bond_list[0][0]]
            for bond in bonds:
                final_vo_list.append(bond[0])
                final_vo_list.append(bond[1])
            final_vo_list.append(bond_list[0][1])
            it = iter(final_vo_list)
            combo = list(zip(it,it))
            vo_combos.append(combo)
            #for i in range(1, len(combo)):
            #    vo_combos.append(swap_positions(deepcopy(combo), 0, i))
    for vo_combo in vo_combos:
        forming_actions = []
        for bond in vo_combo:
            forming_actions.append((Action.PAIR_ORBITALS, bond[0], bond[1]))
        actions.append(breaking_actions + forming_actions)
    return actions

def generate_bnfn_1_actions(bond_list, number_of_bonds_broken):
    '''
    Generate all possible bnfn-1 actions given the bond list and number of bonds broken

    Args:
        bond_list : list of bonds in the molecule
        number_of_bonds_broken : number of bonds broken

    Returns:
        actions : list of all possible bnfn-1 actions
    '''
    actions = []
    vo_combos = []
    # Grab all the sets of bonds that result in a bnfn-1 action
    all_bond_converses = []
    for bond in bond_list[1:]:
        all_bond_converses.append([(bond[0], bond[1]), (bond[1], bond[0])])
    original_combinations = list(product(*all_bond_converses))
    for combination in original_combinations:
        for bonds in permutations(combination):
            final_vo_list = [bond_list[0][0]]
            for bond in bonds:
                final_vo_list.append(bond[0])
                final_vo_list.append(bond[1])
            final_vo_list.append(bond_list[0][1])
            it = iter(final_vo_list)
            combo = list(zip(it,it))
            vo_combos.append(combo)
            for i in range(1, len(combo)):
                vo_combos.append(swap_positions(deepcopy(combo), 0, i))

    for vo_combo in vo_combos:
        # Assign number of electrons for ionic bond breakages
        elec_dict = {}
        elec_dict[vo_combo[0][0]] = 0
        elec_dict[vo_combo[0][1]] = 2
        for a in range(max(1,number_of_bonds_broken-2)):
            for bond in bond_list:
                if bond[0] in elec_dict.keys():
                    elec_dict[bond[1]] = abs(elec_dict[bond[0]] - 2)
                elif bond[1] in elec_dict.keys():
                    elec_dict[bond[0]] = abs(elec_dict[bond[1]] - 2)
            for bond in vo_combo:
                if bond[0] in elec_dict.keys():
                    elec_dict[bond[1]] = abs(elec_dict[bond[0]] - 2)
                elif bond[1] in elec_dict.keys():
                    elec_dict[bond[0]] = abs(elec_dict[bond[1]] - 2)
        # Assign 3 ways to break/form bonds and add it to actions
        concerted_ionic_action = []
        concerted_inverse_ionic_action = []
        concerted_radical_action = []
        for bond in bond_list:
            concerted_radical_action.append((Action.BREAK_BOND_WITH_1_ELECTRON, bond[0], bond[1]))
            if elec_dict[bond[0]] == 0:
                concerted_ionic_action.append((Action.BREAK_BOND_WITH_0_ELECTRONS, bond[0], bond[1]))
                concerted_inverse_ionic_action.append((Action.BREAK_BOND_WITH_2_ELECTRONS, bond[0], bond[1]))
            else:
                concerted_ionic_action.append((Action.BREAK_BOND_WITH_2_ELECTRONS, bond[0], bond[1]))
                concerted_inverse_ionic_action.append((Action.BREAK_BOND_WITH_0_ELECTRONS, bond[0], bond[1]))
        for bond in vo_combo[1:]:
            concerted_radical_action.append((Action.PAIR_ORBITALS, bond[0], bond[1]))
            concerted_ionic_action.append((Action.PAIR_ORBITALS, bond[0], bond[1]))
            concerted_inverse_ionic_action.append((Action.PAIR_ORBITALS, bond[0], bond[1]))
        actions.append((concerted_radical_action, (vo_combo[0][0], vo_combo[0][1]), (1,1)))
        actions.append((concerted_ionic_action, (vo_combo[0][0], vo_combo[0][1]), (0,2)))
        actions.append((concerted_inverse_ionic_action, (vo_combo[0][0], vo_combo[0][1]), (2,0)))
    return actions

def identify_double_bonds(mol, bond_list):
    '''
    Identify all double bonds in the 
    
    Args:
        mol : Molecule object
        bond_list : list of bonds in the molecule

    Returns:
        double_bond_indices : list of indices of double bonds
    '''
    double_bond_indices = []
    for i in range(len(bond_list)):
        if i == len(bond_list) - 1:
            break
        for j in range(i+1, len(bond_list)):
            if mol.get_vo_by_idx(bond_list[i][0]).atom.idx == mol.get_vo_by_idx(bond_list[j][0]).atom.idx and mol.get_vo_by_idx(bond_list[i][1]).atom.idx == mol.get_vo_by_idx(bond_list[j][1]).atom.idx:
                double_bond_indices.append((i,j))
    return double_bond_indices


def initialize_reaction(smi):
    '''
    Initialize the reaction by creating the reactant and product molecules and assigning VO indices

    Args:
        smi : SMILES string of the reaction

    Returns:
        reac_leaf : MoleculeLeaf object of the reactant
        prod_leaf : MoleculeLeaf object of the product
        bonds_to_form : list of bonds to form
        bonds_to_break : list of bonds to break
        new_vo0 : list of new valence orbitals with 0 electrons
        new_vo2 : list of new valence orbitals with 2 electrons
        edit_dist : graph edit distance between reactant and product
    '''
    reac = Molecule(smi.split('>>')[0])
    prod = Molecule(smi.split('>>')[1])
    reac.assign_vo_indices()
    prod.match_vo_indices(reac)
    reac_leaf = MoleculeLeaf(reac, None)
    prod_leaf = MoleculeLeaf(prod, None)
    edit_dist = edit_distance(reac, prod)

    # Grab the differences between the reactant and the product
    new_vo0 = set(prod_leaf.vo_0).difference(set(reac_leaf.vo_0))
    new_vo2 = set(prod_leaf.vo_2).difference(set(reac_leaf.vo_2))
    bonds_to_break = list(set(reac_leaf.bond_list).difference(set(prod_leaf.bond_list)))
    bonds_to_form = list(set(prod_leaf.bond_list).difference(set(reac_leaf.bond_list)))
    edit_dist = edit_distance(reac_leaf.mol, prod_leaf.mol)
    print(bonds_to_break)
    print(bonds_to_form)
    print(new_vo0)
    print(new_vo2)
    return reac_leaf, prod_leaf, bonds_to_form, bonds_to_break, new_vo0, new_vo2, edit_dist


def intermediate_enumeration(leaf, reac_leaf, bonds_to_form, bonds_to_break, new_vo0, new_vo2, initial_edit_dist, current_edit_dist):
    '''
    Perform intermediate enumeration on a given leaf

    Args:
        leaf : MoleculeLeaf object
        reac_leaf : MoleculeLeaf object of the reactant
        bonds_to_form : list of bonds to form
        bonds_to_break : list of bonds to break
        new_vo0 : list of new valence orbitals with 0 electrons
        new_vo2 : list of new valence orbitals with 2 electrons
        initial_edit_dist : graph edit distance between reactant and product
        current_edit_dist : current graph edit distance
    '''
    # Get all the lists from the current leaf and compare them to the reactant
    current_vo0 = leaf.vo_0
    current_vo1 = leaf.vo_1
    current_vo2 = leaf.vo_2

    current_vo0_diff = len(current_vo0) - len(reac_leaf.vo_0)
    current_vo1_total = len(current_vo1)
    current_edit_dist_diff = current_edit_dist - initial_edit_dist

    # Catalog all actions that will result in a reduction in graph edit distance
    distance_reducing_actions = []
    for bond in bonds_to_form:
        for vo in get_same_atom_vos(bond[0], leaf):
            for other_vo in get_same_atom_vos(bond[1], leaf):
                distance_reducing_actions.append((Action.PAIR_ORBITALS, vo, other_vo))
                distance_reducing_actions.append((Action.PAIR_ORBITALS, other_vo, vo))

    for bond in bonds_to_break:
        if bond[0] in new_vo0 or bond[1] in new_vo2:
            distance_reducing_actions.append((Action.BREAK_BOND_WITH_0_ELECTRONS, bond[0], bond[1]))
        elif bond[0] in new_vo2 or bond[1] in new_vo0:
            distance_reducing_actions.append((Action.BREAK_BOND_WITH_2_ELECTRONS, bond[0], bond[1]))
        else:
            distance_reducing_actions.append((Action.BREAK_BOND_WITH_0_ELECTRONS, bond[0], bond[1]))
            distance_reducing_actions.append((Action.BREAK_BOND_WITH_1_ELECTRON, bond[0], bond[1]))
            distance_reducing_actions.append((Action.BREAK_BOND_WITH_2_ELECTRONS, bond[0], bond[1]))

    # Make a vo to atom dict to make skipping bond breaking patterns quicker
    vo_atom_dict = {}
    for vo in leaf.mol.get_valence_orbitals_by_index():
        vo_atom_dict[vo.index] = vo.atom.idx

    # Start building all possible bond breaking patterns
    most_bonds_broken = min(3, int(current_edit_dist))
    bond_break_dict = {}
    #print(leaf.bond_list)
    for i in range(1,most_bonds_broken+1):
        bond_break_dict[i] = list(combinations(leaf.bond_list, i))

    all_actions = []

    # First grab all the f1 actions because that'll be easy
    for vo in current_vo0:
        for other_vo in current_vo2:
            if leaf.mol.get_vo_by_idx(vo).atom.idx != leaf.mol.get_vo_by_idx(other_vo).atom.idx:
                all_actions.append([(Action.PAIR_ORBITALS, vo, other_vo)])

    # For every number of bond breakages, identify all moves with n and n-1 number of formations
    for bond_info in bond_break_dict.items():
        number_of_bonds_broken = bond_info[0]
        if number_of_bonds_broken == 1:
            for bond_list in bond_info[1]:
                for bond in bond_list:
                    # Grab all b1 actions
                    potential_actions = [[(Action.BREAK_BOND_WITH_0_ELECTRONS, bond[0], bond[1])],
                                        [(Action.BREAK_BOND_WITH_1_ELECTRON, bond[0], bond[1])],
                                        [(Action.BREAK_BOND_WITH_2_ELECTRONS, bond[0], bond[1])]]

                    # Add all b1 actions because that'll be easy
                    for actions in potential_actions:
                        all_actions.append(actions)
                        # Grab all b1f1 actions from each b1 action and add them if they are valid
                        if actions[0][0] == Action.BREAK_BOND_WITH_0_ELECTRONS:
                            unused_elec = [0,2]
                        elif actions[0][0] == Action.BREAK_BOND_WITH_1_ELECTRON:
                            unused_elec = [1,1]
                        else:
                            unused_elec = [2,0]
                        illegal_vos = get_same_atom_vos(bond[0], leaf)
                        illegal_vos += get_same_atom_vos(bond[1], leaf)
                        b1f1_actions = generate_bnfn_actions([bond[0], bond[1]], unused_elec, illegal_vos, current_vo0, current_vo1, current_vo2)
                        for final_action in b1f1_actions:
                            all_actions.append(actions + final_action)
        else:
            for bond_list in tqdm(bond_info[1]):
                # Determine whether to skip this set of bonds by checking whether any of them are in bonds_to_break
                skip_counter = 0
                for bond in bond_list:
                    if bond in bonds_to_break:
                        skip_counter += 1
                if skip_counter < number_of_bonds_broken - 2:
                    continue

                # Determine whether to skip this set of bonds by checking which atoms they are apart of
                skip = False
                atoms_involved = []
                for bond in bond_list:
                    for vo in bond:
                        atom_idx = leaf.mol.get_vo_by_idx(vo).atom.idx
                        if atom_idx in atoms_involved:
                            skip = True
                            break
                        else:
                            atoms_involved.append(atom_idx)
                if skip:
                    continue

                # Grab all VOs that shouldn't be used when forming final bond (for bnfn)
                illegal_vos = set()
                for bond in bond_list:
                    for vo in bond:
                        for illegal_vo in get_same_atom_vos(vo, leaf):
                            illegal_vos.add(illegal_vo)
                illegal_vos = list(illegal_vos)

                # Generate bnfn-1 moves and check them for validity
                for actions in generate_bnfn_1_actions(bond_list, number_of_bonds_broken):
                    all_actions.append(actions[0])
                    # With every bnfn-1 move, generate the possible bnfn moves and check them for validity
                    bnfn_actions = generate_bnfn_actions(actions[1], actions[2], illegal_vos, current_vo0, current_vo1, current_vo2)
                    for final_action in bnfn_actions:
                        all_actions.append(actions[0] + final_action)
                    # Also check the option where all the bonds broken get formed (self-contained)
                    all_actions.append(actions[0] + [(Action.PAIR_ORBITALS, actions[1][0], actions[1][1])])
        print(len(all_actions))

    return all_actions

def single_ended_enumeration(smi, most_bonds_broken):
    '''
    Perform single-ended enumeration on a given molecule

    Args:
        smi : SMILES string of the molecule
        most_bonds_broken : maximum number of bonds to break

    Returns:
        all_actions : list of all possible actions
    '''
    init_mol = Chem.MolFromSmiles(smi)
    init_mol = Chem.AddHs(init_mol)
    for atom in init_mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)
    mol = Molecule(Chem.MolToSmiles(init_mol))
    mol.assign_vo_indices()
    leaf = MoleculeLeaf(mol, None)

    # Get all the lists from the current leaf
    current_vo0 = leaf.vo_0
    current_vo1 = leaf.vo_1
    current_vo2 = leaf.vo_2

    # Grab all double bonds and remove one duplicate from each pair from the bond list to avoid duplicate actions
    double_bonds = identify_double_bonds(mol, leaf.bond_list)
    for double_bond in double_bonds:
        leaf.bond_list.remove(leaf.bond_list[double_bond[0]])

    # Start building all possible bond breaking patterns
    bond_break_dict = {}
    #print(leaf.bond_list)
    for i in range(1,most_bonds_broken+1):
        bond_break_dict[i] = list(combinations(leaf.bond_list, i))

    all_actions = []

    # First grab all the f1 actions because that'll be easy
    for vo in current_vo0:
        for other_vo in current_vo2:
            if leaf.mol.get_vo_by_idx(vo).atom.idx != leaf.mol.get_vo_by_idx(other_vo).atom.idx:
                all_actions.append([(Action.PAIR_ORBITALS, vo, other_vo)])

    # For every number of bond breakages, identify all moves with n and n-1 number of formations
    for bond_info in bond_break_dict.items():
        number_of_bonds_broken = bond_info[0]
        if number_of_bonds_broken == 1:
            for bond_list in bond_info[1]:
                for bond in bond_list:
                    # Grab all b1 actions
                    potential_actions = [[(Action.BREAK_BOND_WITH_0_ELECTRONS, bond[0], bond[1])],
                                        [(Action.BREAK_BOND_WITH_1_ELECTRON, bond[0], bond[1])],
                                        [(Action.BREAK_BOND_WITH_2_ELECTRONS, bond[0], bond[1])]]

                    # Add all b1 actions because that'll be easy
                    for actions in potential_actions:
                        all_actions.append(actions)
                        # Grab all b1f1 actions from each b1 action and add them if they are valid
                        if actions[0][0] == Action.BREAK_BOND_WITH_0_ELECTRONS:
                            unused_elec = [0,2]
                        elif actions[0][0] == Action.BREAK_BOND_WITH_1_ELECTRON:
                            unused_elec = [1,1]
                        else:
                            unused_elec = [2,0]
                        illegal_vos = get_same_atom_vos(bond[0], leaf)
                        illegal_vos += get_same_atom_vos(bond[1], leaf)
                        b1f1_actions = generate_bnfn_actions([bond[0], bond[1]], unused_elec, illegal_vos, current_vo0, current_vo1, current_vo2)
                        for final_action in b1f1_actions:
                            all_actions.append(actions + final_action)
        else:
            for bond_list in tqdm(bond_info[1]):

                # Determine whether to skip this set of bonds by checking which atoms they are apart of
                skip = False
                atoms_involved = []
                for bond in bond_list:
                    for vo in bond:
                        atom_idx = leaf.mol.get_vo_by_idx(vo).atom.idx
                        if atom_idx in atoms_involved:
                            skip = True
                            break
                        else:
                            atoms_involved.append(atom_idx)
                if skip:
                    continue

                # Grab all VOs that shouldn't be used when forming final bond (for bnfn)
                illegal_vos = set()
                for bond in bond_list:
                    for vo in bond:
                        for illegal_vo in get_same_atom_vos(vo, leaf):
                            illegal_vos.add(illegal_vo)
                illegal_vos = list(illegal_vos)

                # Generate bnfn-1 moves and check them for validity
                for actions in generate_bnfn_1_actions(bond_list, number_of_bonds_broken):
                    all_actions.append(actions[0])
                    # With every bnfn-1 move, generate the possible bnfn moves
                    bnfn_actions = generate_bnfn_actions(actions[1], actions[2], illegal_vos, current_vo0, current_vo1, current_vo2)
                    for final_action in bnfn_actions:
                        all_actions.append(actions[0] + final_action)
                    # Also check the option where all the bonds broken get formed (self-contained)
                    all_actions.append(actions[0] + [(Action.PAIR_ORBITALS, actions[1][0], actions[1][1])])
        print(len(all_actions))
    
    return all_actions, mol

def closed_shell_enumeration(smi, most_bonds_broken):
    '''
    Perform closed shell enumeration on a given molecule

    Args:
        smi : SMILES string of the molecule
        most_bonds_broken : maximum number of bonds to break

    Returns:
        all_actions : list of all possible actions
    '''
    init_mol = Chem.MolFromSmiles(smi)
    init_mol = Chem.AddHs(init_mol)
    for atom in init_mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)
    mol = Molecule(Chem.MolToSmiles(init_mol))
    mol.assign_vo_indices()
    leaf = MoleculeLeaf(mol, None)

    # Grab all double bonds and remove one duplicate from each pair from the bond list to avoid duplicate actions
    double_bonds = identify_double_bonds(mol, leaf.bond_list)
    #print(double_bonds)
    double_bond_vos = []
    #print(leaf.bond_list)
    for double_bond in sorted(double_bonds, reverse=True):
        double_bond_vos.append(leaf.bond_list[double_bond[1]])
        leaf.bond_list.remove(leaf.bond_list[double_bond[0]])
    #print(leaf.bond_list)
    #print(double_bond_vos)
    # Start building all possible bond breaking patterns
    bond_break_dict = {}
    #print(leaf.bond_list)
    for i in range(2,most_bonds_broken+1):
        bond_break_dict[i] = list(combinations(leaf.bond_list, i))

    # Iterate over all bond breaking scenarios
    all_actions = []
    for bond_info in bond_break_dict.items():
        for bond_list in tqdm(bond_info[1]):
            skip = False
            atoms_involved = []
            # Skip elementary steps with more than n (where n > 2) bonds broken that don't include n - 1 double bonds
            if len(bond_list) > 2 and len(set(bond_list).intersection(set(double_bond_vos))) < len(bond_list) - 1:
                continue
            # Determine whether to skip this set of bonds by checking which atoms they are apart of
            for bond in bond_list:
                for vo in bond:
                    atom_idx = leaf.mol.get_vo_by_idx(vo).atom.idx
                    if atom_idx in atoms_involved:
                        skip = True
                        break
                    atoms_involved.append(atom_idx)
            if skip:
                continue
            all_actions += closed_shell_bnfn_actions(bond_list)

    return all_actions, mol

def enumeration(smi, most_bonds_broken, closed_shell=True):
    '''
    Perform enumeration on a given molecule

    Args:
        smi : SMILES string of the molecule
        most_bonds_broken : maximum number of bonds to break
        closed_shell : whether the enumeration is closed shell

    Returns:
        all_actions : list of all possible actions
    '''
    if closed_shell:
        return closed_shell_enumeration(smi, most_bonds_broken)
    return single_ended_enumeration(smi, most_bonds_broken)

if __name__ == "__main__":
    SMI = '[C@:5]([C@:6]([O:7][C:8](=[O:9])[c:10]1[c:11]([H:34])[n:12][c:13]2[c:14]([H:35])[c:15]([H:36])[c:16]([H:37])[c:17]([H:38])[c:18]2[c:19]1[O:20][H:39])([H:32])[H:33])([H:29])([H:30])[H:31].[H][N:4]([C@@:3]([C@@:2]([C@:1]([H:21])([H:22])[H:23])([H:24])[H:25])([H:26])[H:27])[H:28]>>[C@:1]([C@:2]([C@:3]([N:4]([C:8](=[O:9])[c:10]1[c:11]([H:34])[n:12][c:13]2[c:14]([H:35])[c:15]([H:36])[c:16]([H:37])[c:17]([H:38])[c:18]2[c:19]1[O:20][H:39])[H:28])([H:26])[H:27])([H:24])[H:25])([H:21])([H:22])[H:23].[C@:5]([C@:6]([O-:7])([H:32])[H:33])([H:29])([H:30])[H:31].[H+:40]'
    all_actions_init, mol_init = closed_shell_enumeration(SMI.split(">>", maxsplit=1)[0], 3)
    print(len(all_actions_init))
