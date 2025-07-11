"""Module for housing the Molecule class"""

import logging
import copy
from typing import Optional, List, Dict, Tuple
from enum import IntEnum
import numpy as np
import rdkit
from rdkit import Chem
import torch
import networkx as nx
from RAMP.utils.molecule_utils import atom_to_num_vos


sanitizeOps = rdkit.Chem.rdmolops.SanitizeFlags.SANITIZE_ALL ^ \
    rdkit.Chem.rdmolops.SanitizeFlags.SANITIZE_SETAROMATICITY ^ \
    rdkit.Chem.rdmolops.SanitizeFlags.SANITIZE_FINDRADICALS ^ \
    rdkit.Chem.rdmolops.SanitizeFlags.SANITIZE_ADJUSTHS


class Action(IntEnum):
    """The possible action space"""
    BREAK_BOND_WITH_0_ELECTRONS = -1
    BREAK_BOND_WITH_1_ELECTRON = -2
    BREAK_BOND_WITH_2_ELECTRONS = -3
    PAIR_ORBITALS = -4
    INTERACT_ORBITAL_WITH_BOND = -10
    SINGLE_ELECTRON_TRANSFER = -5
    STOP = -11


class ValenceOrbital:
    """Representation of a Valence Orbital. When paired, the VO has self.num_electrons = -1; otherwise, it can range between 0-2."""
    def __init__(self, num_electrons: int, atom: 'Atom', neighbor: Optional['ValenceOrbital'] = None):
        self.num_electrons = num_electrons
        self.atom = atom
        self.neighbor = neighbor
        self.index = None

    def pair(self, other: 'ValenceOrbital') -> None:
        '''
        Pairs two valence orbitals, setting each neighbor to their respective neighbors
           and num_electrons to -1. Also updates RDKit Mol Objects to reflect new pairings.

        Args:
            other (ValenceOrbital): The other ValenceOrbital to pair with

        Returns:
            None
        '''
        if self.neighbor is not None or other.neighbor is not None:
            raise ValueError("Cannot pair already paired orbitals!")

        if self.num_electrons + other.num_electrons != 2:
            raise ValueError("Can only pair orbitals with two total electrons!")

        if self.atom is other.atom:
            raise ValueError("Cannot pair an orbital with an orbital on the same atom!")

        current_bond = self.atom.molecule.molecule.GetBondBetweenAtoms(self.atom.idx, other.atom.idx)

        if current_bond is not None and current_bond.GetBondType() is Chem.rdchem.BondType.TRIPLE:
            raise ValueError("Cannot make bond order above Triple!")

        self.neighbor = other
        self.neighbor.neighbor = self

        if self.num_electrons == 0:  # homolytic bond formation; electrons transferred from neighbor
            self.atom.decrement_formal_charge()
            self.neighbor.atom.increment_formal_charge()

        if self.num_electrons == 1:  # heterolytic bond formation; radicals removed
            self.atom.decrement_radical_electrons()
            self.neighbor.atom.decrement_radical_electrons()

        if self.num_electrons == 2:  # homolytic bond formation; electrons transferred to neighbor
            self.atom.increment_formal_charge()
            self.neighbor.atom.decrement_formal_charge()

        self.atom.increment_bond_order(other.atom)

        self.num_electrons = -1
        self.neighbor.num_electrons = -1

    def single_electron_transfer(self, other: 'ValenceOrbital') -> None:
        '''
        Transfers a single electron from self to other. Self must have at least one electron; other must have at most 1 electron.
        
        Args:
            other (ValenceOrbital): The other ValenceOrbital to transfer the electron to

        Returns:
            None
        '''
        if self.num_electrons < 1 or not 0 <= other.num_electrons <= 1:
            raise ValueError("Can only transfer electron from radical/filled to empty/radical orbital!")

        if self.num_electrons == 2:
            self.atom.increment_radical_electrons()
        else:
            self.atom.decrement_radical_electrons()
        self.atom.increment_formal_charge()

        if other.num_electrons == 0:
            other.atom.increment_radical_electrons()
        else:
            other.atom.decrement_radical_electrons()
        other.atom.decrement_formal_charge()

        self.num_electrons -= 1
        other.num_electrons += 1

        self.atom.molecule.smi = Chem.MolToSmiles(self.atom.molecule.molecule, kekuleSmiles=True, allHsExplicit=True)

    def unpair(self, num_electrons_remaining: int) -> None:
        '''
        Unpairs this valence orbital with its neighbor and sets the
        number of electrons in the current ValenceOrbital to num_electrons_remaining.
        Also updates RDKit Mol Objects to reflect new pairings.

        Args:
            num_electrons_remaining (int): The number of electrons to leave in the current ValenceOrbital
        
        Returns:
            None
        '''
        if self.neighbor is None:
            raise ValueError("Cannot unpair an unpaired orbital!")

        if not (0 <= num_electrons_remaining <= 2):
            raise ValueError("num_electrons_remaining must be between 0 and 2!")

        current_bond = self.atom.molecule.molecule.GetBondBetweenAtoms(self.atom.idx, self.neighbor.atom.idx)

        if current_bond is None:
            raise ValueError("Cannot remove non-existent bond!")

        self.num_electrons = num_electrons_remaining
        self.neighbor.num_electrons = 2 - num_electrons_remaining

        if self.num_electrons == 0:  # homolytic bond cleavage; electrons transferred to neighbor
            self.atom.increment_formal_charge()
            self.neighbor.atom.decrement_formal_charge()

        if self.num_electrons == 1:  # heterolytic bond cleavage; radicals formed
            self.atom.increment_radical_electrons()
            self.neighbor.atom.increment_radical_electrons()

        if self.num_electrons == 2:  # homolytic bond cleavage; electrons transferred to neighbor
            self.atom.decrement_formal_charge()
            self.neighbor.atom.increment_formal_charge()

        self.atom.decrement_bond_order(self.neighbor.atom)
        self.neighbor.neighbor = None
        self.neighbor = None

    def interact_empty_orbital_with_bond(self, other: 'ValenceOrbital') -> None:
        '''
        Interacts an empty orbital with a bond. This results in a pairing of self
        and other, and an unpairing between other and other's current neighbor. 
        This action can only be performed when other is paired and self is not.

        Args:
            other (ValenceOrbital): The ValenceOrbital to interact with

        Returns:
            None
        '''
        if self.atom is other.atom:
            raise ValueError("Cannot interact an orbital with an orbital on the same atom!")

        if self is other.neighbor:
            raise ValueError("Cannot interact an orbital with another orbital bonded to same atom!")

        if self.neighbor is None and other.neighbor is not None:
            other.unpair(2 - self.num_electrons)
            self.pair(other)

        else:
            raise ValueError("Invalid interaction!")

    def interact_single_electron_transfer(self, other: 'ValenceOrbital') -> None:
        '''
        Performs the single electron transfer from self to other.

        Args:
            other (ValenceOrbital): The ValenceOrbital to interact with

        Returns:
            None
        '''
        if self.atom is other.atom:
            raise ValueError("Cannot interact an orbital with an orbital on the same atom!")

        if self.neighbor is other:
            raise ValueError("Cannot interact an orbital with its neighbor!")

        if self.num_electrons >= 1 and 0 <= other.num_electrons <= 1:
            self.single_electron_transfer(other)

        else:
            raise ValueError("Invalid interaction!")

    def can_perform_vo_action(self, other_vo: 'ValenceOrbital', action: 'Action') -> bool:
        '''
        Returns True if this ValenceOrbital can perform the action on other_vo, and False otherwise.

        Args:
            other_vo (ValenceOrbital): The other ValenceOrbital to perform the action on
            action (Action): The action to perform

        Returns:
            bool: True if the action can be performed; False otherwise
        '''
        bond_order = self.atom.molecule.get_bond_order(self.atom, other_vo.atom)
        if self is other_vo or self.atom is other_vo.atom:
            return False

        if action in {Action.BREAK_BOND_WITH_0_ELECTRONS, Action.BREAK_BOND_WITH_1_ELECTRON, Action.BREAK_BOND_WITH_2_ELECTRONS}:
            return self.neighbor is other_vo

        if action == Action.PAIR_ORBITALS:
            return self.num_electrons + other_vo.num_electrons == 2 and bond_order < 3

        if action == Action.INTERACT_ORBITAL_WITH_BOND:
            return self.neighbor is None and other_vo.neighbor is not None and bond_order < 3 and other_vo.neighbor.atom is not self.atom

        if action == Action.SINGLE_ELECTRON_TRANSFER:
            return self.num_electrons >= 1 and 0 <= other_vo.num_electrons <= 1

        raise ValueError("Invalid Action {}".format(action))

    def vo_action(self, other_vo: 'ValenceOrbital', action: 'Action') -> bool:
        '''
        Args:
            other_vo: Other VO to perform action on
            action: Type of action to perform

        Returns:
            True if the action was successful; False otherwise

        Modifies:
            Current Molecule
        '''
        if not self.can_perform_vo_action(other_vo, action):
            raise ValueError("Cannot Perform action {} on VOs {} and {}".format(action, self, other_vo))

        if action in {Action.BREAK_BOND_WITH_0_ELECTRONS, Action.BREAK_BOND_WITH_1_ELECTRON, Action.BREAK_BOND_WITH_2_ELECTRONS}:
            num_electrons_remaining = 0 if action == Action.BREAK_BOND_WITH_0_ELECTRONS else 1 if action == Action.BREAK_BOND_WITH_1_ELECTRON else 2
            self.unpair(num_electrons_remaining=num_electrons_remaining)
            return True

        elif action == Action.PAIR_ORBITALS:
            self.pair(other_vo)
            return True

        elif action == Action.INTERACT_ORBITAL_WITH_BOND:
            self.interact_empty_orbital_with_bond(other_vo)
            return True

        elif action == Action.SINGLE_ELECTRON_TRANSFER:
            self.interact_single_electron_transfer(other_vo)
            return True

        return False

    def get_result_of_action(self, other_vo: 'ValenceOrbital', action: 'Action') -> Tuple[Optional[str], Optional[int], Optional[int]]:
        '''
        This method does not modify the molecule itself, but rather gives the result of the action specified as a SMILES string.

        Args:
            other_vo (ValenceOrbital): The other ValenceOrbital to perform the action on
            action (Action): The action to perform

        Returns:
            str: The molecule that would be gotten if the action was performed, and None if the action was bad.
            int: The number of charges in the molecule after the action was performed, or None if the action was bad.
            int: The number of radicals in the molecule after the action was performed, or None if the action was bad.
        '''
        if not self.can_perform_vo_action(other_vo, action):
            raise ValueError("Cannot Perform action {} on VOs {} and {}".format(action, self, other_vo))

        if action in {Action.BREAK_BOND_WITH_0_ELECTRONS, Action.BREAK_BOND_WITH_1_ELECTRON, Action.BREAK_BOND_WITH_2_ELECTRONS}:
            num_electrons_remaining = 0 if action == Action.BREAK_BOND_WITH_0_ELECTRONS else 1 if action == Action.BREAK_BOND_WITH_1_ELECTRON else 2
            self.unpair(num_electrons_remaining=num_electrons_remaining)
            output_smi = self.atom.molecule.smi
            charges, radicals = self.atom.molecule.get_total_charges_radicals()
            self.pair(other_vo)
            return output_smi, charges, radicals

        elif action == Action.PAIR_ORBITALS:
            original_num_electrons = self.num_electrons
            self.pair(other_vo)
            output_smi = self.atom.molecule.smi
            charges, radicals = self.atom.molecule.get_total_charges_radicals()
            self.unpair(num_electrons_remaining=original_num_electrons)
            return output_smi, charges, radicals

        elif action == Action.INTERACT_ORBITAL_WITH_BOND:
            vo_to_attack_with = other_vo.neighbor
            assert isinstance(vo_to_attack_with, ValenceOrbital)
            self.interact_empty_orbital_with_bond(other_vo)
            output_smi = self.atom.molecule.smi
            charges, radicals = self.atom.molecule.get_total_charges_radicals()
            vo_to_attack_with.interact_empty_orbital_with_bond(other_vo)
            return output_smi, charges, radicals

        elif action == Action.SINGLE_ELECTRON_TRANSFER:
            self.interact_single_electron_transfer(other_vo)
            output_smi = self.atom.molecule.smi
            charges, radicals = self.atom.molecule.get_total_charges_radicals()
            other_vo.interact_single_electron_transfer(self)
            return output_smi, charges, radicals

        return None, None, None
    
    def get_mol_result_of_action(self, other_vo: 'ValenceOrbital', action: 'Action') -> Tuple[Optional[Chem.Mol], Optional[int], Optional[int]]:
        '''
        This method does not modify the molecule itself, but rather gives the result of the action specified as a SMILES string.

        Args:
            other_vo (ValenceOrbital): The other ValenceOrbital to perform the action on
            action (Action): The action to perform

        Returns:
            str: The molecule that would be gotten if the action was performed, and None if the action was bad.
            int: The number of charges in the molecule after the action was performed, or None if the action was bad.
            int: The number of radicals in the molecule after the action was performed, or None if the action was bad.
        '''
        if not self.can_perform_vo_action(other_vo, action):
            raise ValueError("Cannot Perform action {} on VOs {} and {}".format(action, self, other_vo))

        if action in {Action.BREAK_BOND_WITH_0_ELECTRONS, Action.BREAK_BOND_WITH_1_ELECTRON, Action.BREAK_BOND_WITH_2_ELECTRONS}:
            num_electrons_remaining = 0 if action == Action.BREAK_BOND_WITH_0_ELECTRONS else 1 if action == Action.BREAK_BOND_WITH_1_ELECTRON else 2
            self.unpair(num_electrons_remaining=num_electrons_remaining)
            output_mol = copy.copy(self.atom.molecule.molecule)
            charges, radicals = self.atom.molecule.get_total_charges_radicals()
            self.pair(other_vo)
            return output_mol, charges, radicals

        elif action == Action.PAIR_ORBITALS:
            original_num_electrons = self.num_electrons
            self.pair(other_vo)
            output_mol = copy.copy(self.atom.molecule.molecule)
            charges, radicals = self.atom.molecule.get_total_charges_radicals()
            self.unpair(num_electrons_remaining=original_num_electrons)
            return output_mol, charges, radicals

        elif action == Action.INTERACT_ORBITAL_WITH_BOND:
            vo_to_attack_with = other_vo.neighbor
            assert isinstance(vo_to_attack_with, ValenceOrbital)
            self.interact_empty_orbital_with_bond(other_vo)
            output_mol = copy.copy(self.atom.molecule.molecule)
            charges, radicals = self.atom.molecule.get_total_charges_radicals()
            vo_to_attack_with.interact_empty_orbital_with_bond(other_vo)
            return output_mol, charges, radicals

        elif action == Action.SINGLE_ELECTRON_TRANSFER:
            self.interact_single_electron_transfer(other_vo)
            output_mol = copy.copy(self.atom.molecule.molecule)
            charges, radicals = self.atom.molecule.get_total_charges_radicals()
            other_vo.interact_single_electron_transfer(self)
            return output_mol, charges, radicals

        return None, None, None

    def get_molec_result_of_action(self, other_vo: 'ValenceOrbital', action: 'Action') -> Tuple[Optional[Chem.Mol], Optional[int], Optional[int]]:
        '''
        This method does not modify the molecule itself, but rather gives the result of the action specified as a SMILES string.

        Args:
            other_vo (ValenceOrbital): The other ValenceOrbital to perform the action on
            action (Action): The action to perform

        Returns:
            str: The molecule that would be gotten if the action was performed, and None if the action was bad.
            int: The number of charges in the molecule after the action was performed, or None if the action was bad.
            int: The number of radicals in the molecule after the action was performed, or None if the action was bad.
        '''
        if not self.can_perform_vo_action(other_vo, action):
            raise ValueError("Cannot Perform action {} on VOs {} and {}".format(action, self, other_vo))

        if action in {Action.BREAK_BOND_WITH_0_ELECTRONS, Action.BREAK_BOND_WITH_1_ELECTRON, Action.BREAK_BOND_WITH_2_ELECTRONS}:
            num_electrons_remaining = 0 if action == Action.BREAK_BOND_WITH_0_ELECTRONS else 1 if action == Action.BREAK_BOND_WITH_1_ELECTRON else 2
            self.unpair(num_electrons_remaining=num_electrons_remaining)
            output_mol = copy.deepcopy(self.atom.molecule)
            charges, radicals = self.atom.molecule.get_total_charges_radicals()
            self.pair(other_vo)
            return output_mol, charges, radicals

        elif action == Action.PAIR_ORBITALS:
            original_num_electrons = self.num_electrons
            self.pair(other_vo)
            output_mol = copy.deepcopy(self.atom.molecule)
            charges, radicals = self.atom.molecule.get_total_charges_radicals()
            self.unpair(num_electrons_remaining=original_num_electrons)
            return output_mol, charges, radicals

        elif action == Action.INTERACT_ORBITAL_WITH_BOND:
            vo_to_attack_with = other_vo.neighbor
            assert isinstance(vo_to_attack_with, ValenceOrbital)
            self.interact_empty_orbital_with_bond(other_vo)
            output_mol = copy.deepcopy(self.atom.molecule)
            charges, radicals = self.atom.molecule.get_total_charges_radicals()
            vo_to_attack_with.interact_empty_orbital_with_bond(other_vo)
            return output_mol, charges, radicals

        elif action == Action.SINGLE_ELECTRON_TRANSFER:
            self.interact_single_electron_transfer(other_vo)
            output_mol = copy.deepcopy(self.atom.molecule)
            charges, radicals = self.atom.molecule.get_total_charges_radicals()
            other_vo.interact_single_electron_transfer(self)
            return output_mol, charges, radicals

        return None, None, None


    def __str__(self) -> str:
        return f"self.num_electrons: {self.num_electrons}; self.atom: {str(self.atom)}; self.neighbor: {str(self.neighbor.atom) if self.neighbor else None}"

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, __value: object) -> bool:
        return self.__str__() == __value.__str__()


class Atom:
    """
    A representation of an Atom and its associated VOs. Atoms get the number of VOs specified in mech_pred.utils.atom_to_num_VOs.
    """
    def __init__(self, molecule: 'Molecule', atom_type: str, idx: int, num_valence_electrons: int):
        self.molecule = molecule
        self.atom_type = atom_type
        self.idx = idx
        self.valence_orbitals = []
        num_valence_orbitals = atom_to_num_vos(self.atom_type)

        # num_valence_electrons // num_valence_orbitals splits up all valence electrons that can be split
        # then have (num_valence_electrons % num_valence_orbitals) left over electrons, which would be assigned directly
        # TODO: support hypervalency. currently, just use hypervalence numbers from the atom_to_num_VOs function

        num_divisible_electrons = num_valence_electrons // num_valence_orbitals
        num_leftover_electrons = num_valence_electrons % num_valence_orbitals
        electrons_in_orbitals = [num_divisible_electrons + (1 if idx < num_leftover_electrons else 0) for idx in range(num_valence_orbitals)]

        for idx, num_electrons in enumerate(electrons_in_orbitals):
            self.valence_orbitals.append(ValenceOrbital(num_electrons=num_electrons, atom=self, neighbor=None))

    def make_vo_pairing(self, other_atom: 'Atom', radical_only: bool = False) -> bool:
        """
        Pairs one VO between this atom and other_atom. The VOs must both be
        currently unpaired and must have total number of electrons = 2.

        If radical_only is true, then only orbitals with one electron each are paired.
        This is used during initialization only.
        """
        for valence_orbital in self.valence_orbitals:
            if valence_orbital.neighbor is None:
                for other_valence_orbital in other_atom.valence_orbitals:
                    if other_valence_orbital.neighbor is None:
                        if valence_orbital.num_electrons + other_valence_orbital.num_electrons == 2:
                            if not radical_only or (valence_orbital.num_electrons == 1 and other_valence_orbital.num_electrons == 1):
                                valence_orbital.pair(other_valence_orbital)
                                return True

        raise ValueError("No possible bond between these atoms!")
        return False

    def combine_radical_vos(self) -> None:
        """
        If there are any pairs of radical VOs on the same atom, combine them into a single VO with 2 electrons.
        This means that a singlet state is always preferred over a triplet state.
        """
        radical_vos = []
        for vo in self.valence_orbitals:
            if vo.num_electrons == 1:
                radical_vos.append(vo)

        for vo1, vo2 in zip(radical_vos[::2], radical_vos[1::2]):
            # for every pair of radical VOs, replace them via single electron transfer
            vo1.single_electron_transfer(vo2)

        return None

    def get_rdkit_atom(self) -> 'rdkit.Chem.rdchem.Atom':
        """Returns the rdkit atom object corresponding to this Atom"""
        return self.molecule.molecule.GetAtomWithIdx(self.idx)

    def get_formal_charge(self) -> int:
        """Returns the Formal Charge on the RDKit Atom"""
        return self.get_rdkit_atom().GetFormalCharge()

    def set_formal_charge(self, charge: int) -> None:
        """Sets the Formal Charge on the RDKit Atom to the charge"""
        self.get_rdkit_atom().SetFormalCharge(charge)

    def get_num_radical_electrons(self) -> int:
        """Returns the number of radical electrons on the RDKit Atom"""
        return self.get_rdkit_atom().GetNumRadicalElectrons()

    def set_num_radical_electrons(self, num_radicals: int) -> None:
        """Sets the number of radical electrons on the RDKit Atom to num_radicals"""
        if num_radicals < 0:
            raise ValueError("err: num_radicals < 0")
        self.get_rdkit_atom().SetNumRadicalElectrons(num_radicals)

    def increment_formal_charge(self) -> None:
        """Increments the Formal Charge on the RDKit Atom by 1"""
        self.set_formal_charge(self.get_formal_charge() + 1)

    def decrement_formal_charge(self) -> None:
        """Decrements the Formal Charge on the RDKit Atom by 1"""
        self.set_formal_charge(self.get_formal_charge() - 1)

    def increment_radical_electrons(self) -> None:
        """Increments the number of radical electrons on the RDKit Atom by 1"""
        self.set_num_radical_electrons(self.get_num_radical_electrons() + 1)

    def decrement_radical_electrons(self) -> None:
        """Decrements the number of radical electrons on the RDKit Atom by 1"""
        self.set_num_radical_electrons(self.get_num_radical_electrons() - 1)

    def increment_bond_order(self, other_atom: 'Atom') -> None:
        """Increments the bond order of the bond on the RDKit Atom and other_atom"""
        rwmol = self.molecule.molecule
        current_bond = self.molecule.molecule.GetBondBetweenAtoms(self.idx, other_atom.idx)

        if current_bond is None:
            rwmol.AddBond(self.idx, other_atom.idx, Chem.rdchem.BondType.SINGLE)

        elif current_bond.GetBondType() is Chem.rdchem.BondType.TRIPLE:
            raise ValueError("Cannot increment bond order beyond triple!")
            return

        elif current_bond.GetBondType() is Chem.rdchem.BondType.DOUBLE:
            rwmol.RemoveBond(self.idx, other_atom.idx)
            rwmol.AddBond(self.idx, other_atom.idx, Chem.rdchem.BondType.TRIPLE)

        elif current_bond.GetBondType() is Chem.rdchem.BondType.SINGLE:
            rwmol.RemoveBond(self.idx, other_atom.idx)
            rwmol.AddBond(self.idx, other_atom.idx, Chem.rdchem.BondType.DOUBLE)

        else:
            logging.critical("Unknown bond type!")
            return

        self.molecule.molecule = rwmol

        Chem.SanitizeMol(self.molecule.molecule, sanitizeOps=sanitizeOps)
        self.molecule.smi = Chem.MolToSmiles(self.molecule.molecule, kekuleSmiles=True, allHsExplicit=True)

    def decrement_bond_order(self, other_atom: 'Atom') -> None:
        """Decrements the bond order of the bond on thes RDKit Atom and other_atom"""
        current_bond = self.molecule.molecule.GetBondBetweenAtoms(self.idx, other_atom.idx)
        if current_bond is None:
            raise ValueError("No bond between atoms!")

        rwmol = self.molecule.molecule
        if current_bond.GetBondType() is Chem.rdchem.BondType.TRIPLE:
            rwmol.RemoveBond(self.idx, other_atom.idx)
            rwmol.AddBond(self.idx, other_atom.idx, Chem.rdchem.BondType.DOUBLE)

        elif current_bond.GetBondType() is Chem.rdchem.BondType.DOUBLE:
            rwmol.RemoveBond(self.idx, other_atom.idx)
            rwmol.AddBond(self.idx, other_atom.idx, Chem.rdchem.BondType.SINGLE)

        elif current_bond.GetBondType() is Chem.rdchem.BondType.SINGLE:
            rwmol.RemoveBond(self.idx, other_atom.idx)

        else:
            logging.critical("Unknown bond type!")
            return

        self.molecule.molecule = rwmol
        Chem.SanitizeMol(self.molecule.molecule, sanitizeOps=sanitizeOps)
        self.molecule.smi = Chem.MolToSmiles(self.molecule.molecule, kekuleSmiles=True, allHsExplicit=True)

    def get_valence_orbitals_to_other_atom(self, atom: 'Atom') -> List['ValenceOrbital']:
        """
        Returns all of self's valence orbitals that are bonded to the specified atom.
        """
        bonded_valence_orbitals = []
        for self_vo in self.valence_orbitals:
            if self_vo.neighbor is not None and self_vo.neighbor.atom is atom:
                bonded_valence_orbitals.append(self_vo)

        return bonded_valence_orbitals

    def get_unpaired_valence_orbitals(self) -> List['ValenceOrbital']:
        """
        Returns all of self's valence orbitals that have no neighbor.
        """
        return [vo for vo in self.valence_orbitals if vo.neighbor is None]

    def __str__(self) -> str:
        return f"{self.idx}, {self.atom_type}"

    def __repr__(self) -> str:
        return self.__str__()


class Molecule:
    def __init__(self, smi: str):
        self.smi = smi
        self.molecule = Chem.RWMol()
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
        real_map = [0] * og.GetNumAtoms()
        for i, atom in enumerate(fake_map):
            real_map[atom] = i
        self.orig_molecule = Chem.RenumberAtoms(og, real_map)
        '''
        self.orig_molecule = Chem.MolFromSmiles(smi)
        self.orig_molecule = Chem.AddHs(self.orig_molecule)  # always add H's to make bonding correct
        Chem.Kekulize(self.orig_molecule)  # change to kekulized smiles to remove aromatic bonds
        self.num_atoms = self.orig_molecule.GetNumAtoms()

        # Create adjacency list representation for bonds. Initial_bonds is not symmetric.
        initial_bonds: Dict[int, List[int]] = dict()
        for bond in self.orig_molecule.GetBonds():
            bond.SetIsAromatic(False)  # remove aromaticity properties
            atom_1 = bond.GetBeginAtomIdx()
            atom_2 = bond.GetEndAtomIdx()
            num_bonds = round(bond.GetBondTypeAsDouble())
            initial_bonds[atom_1] = initial_bonds.get(atom_1, []) + [atom_2] * num_bonds

        # Create Atom objects
        self.atoms = []
        rd_periodic_table = Chem.GetPeriodicTable()
        for idx, atom in enumerate(self.orig_molecule.GetAtoms()):
            atom.SetIsAromatic(False)  # remove aromaticity properties
            num_valence_electrons = rd_periodic_table.GetNOuterElecs(atom.GetSymbol()) - atom.GetFormalCharge()
            rdkit_atom = Chem.Atom(atom.GetSymbol())
            rdkit_atom.SetFormalCharge(atom.GetFormalCharge())
            rdkit_atom.SetNumRadicalElectrons(atom_to_num_vos(atom.GetSymbol()) - abs(num_valence_electrons - atom_to_num_vos(atom.GetSymbol())))
            rdkit_atom.SetNoImplicit(True)  # prevent additional hydrogens from extraneously being added
            self.molecule.AddAtom(rdkit_atom)
            self.atoms.append(Atom(molecule=self, atom_type=atom.GetSymbol(), idx=idx, num_valence_electrons=num_valence_electrons))

        # Create all bonds as part of ValenceOrbital objects
        for atom_idx, neighbors in initial_bonds.items():
            atom = self.atoms[atom_idx]
            for neighbor_atom_idx in neighbors:
                atom.make_vo_pairing(self.atoms[neighbor_atom_idx], radical_only=True)

        # combine radical orbitals on each atom, if any exist
        for atom in self.atoms:
            atom.combine_radical_vos()
        
        for atom in self.molecule.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx() + 1)
        for atom in self.orig_molecule.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx() + 1)

        #Chem.SanitizeMol(self.molecule, sanitizeOps=sanitizeOps)S

    def get_atom_by_idx(self, idx: int) -> 'Atom':
        """Returns the Atom object at the specific index of the molecule"""
        if idx >= self.num_atoms or idx < 0:
            raise ValueError("get_atom_by_idx(): idx out of Range")

        return self.atoms[idx]

    def get_vos_between(self, atom1: 'Atom', atom2: 'Atom') -> List['ValenceOrbital']:
        """Returns all atom1-valence orbitals that are bonded to atom2"""
        return atom1.get_valence_orbitals_to_other_atom(atom2)

    def get_vos_between_by_idx(self, atom1_idx: int, atom2_idx: int) -> List['ValenceOrbital']:
        """Returns all atom1-valence orbitals that are bonded to atom2"""
        return self.get_vos_between(self.get_atom_by_idx(atom1_idx), self.get_atom_by_idx(atom2_idx))

    def get_all_valence_orbitals(self) -> List['ValenceOrbital']:
        """Returns all valence orbitals in the molecule"""
        return [vo for atom in self.atoms for vo in atom.valence_orbitals]
    
    def assign_vo_indices(self) -> None:
        """Assigns indices to valence orbitals (should be used for reactants)"""
        idx = 0
        for atom in self.atoms:
            for vo in atom.valence_orbitals:
                vo.index = idx
                idx += 1
    
    def match_vo_indices(self, other_mol: 'Molecule') -> None:
        """Assigns indices to valence orbitals based on another molecule's indices (should be used for products)"""
        for atom, other_atom in zip(self.atoms, other_mol.atoms):
            used_indices = []
            all_indices = []
            for vo in atom.valence_orbitals:
                for other_vo in other_atom.valence_orbitals:
                    all_indices.append(other_vo.index)
                    if vo == other_vo and other_vo.index not in used_indices:
                        vo.index = other_vo.index
                        used_indices.append(other_vo.index)
                        break
            unused_indices = list(set(all_indices).difference(set(used_indices)))
            for vo in atom.valence_orbitals:
                if vo.index == None:
                    vo.index = unused_indices[0]
                    unused_indices.remove(unused_indices[0])

    def get_valence_orbitals_by_index(self) -> List['ValenceOrbital']:
        """Returns all valence orbitals in the order of their indices (WILL NOT WORK IF INDICES HAVEN'T BEEN ASSIGNED)"""
        unordered_vos = self.get_all_valence_orbitals()
        vos = [0] * len(unordered_vos)
        for vo in unordered_vos:
            vos[vo.index] = vo
        return vos

    def get_vo_by_idx(self, idx: int) -> 'ValenceOrbital':
        """Returns the valence orbital at the given index"""
        return self.get_valence_orbitals_by_index()[idx]
    
    def get_bond_order(self, atom1: 'Atom', atom2: 'Atom') -> int:
        """Returns the bond order between two indices"""
        return len(atom1.get_valence_orbitals_to_other_atom(atom2))

    def get_bond_order_by_idx(self, atom1_idx: int, atom2_idx: int) -> int:
        """Calls get_bond_order() with atom indices rather than atom objects."""
        atom1, atom2 = self.get_atom_by_idx(atom1_idx), self.get_atom_by_idx(atom2_idx)
        return self.get_bond_order(atom1, atom2)

    def get_unpaired_vos(self, atom: 'Atom') -> List['ValenceOrbital']:
        return atom.get_unpaired_valence_orbitals()

    def get_unpaired_vos_by_idx(self, atom_idx: int) -> List['ValenceOrbital']:
        return self.get_unpaired_vos(self.get_atom_by_idx(atom_idx))

    def unpair_vo(self, atom1: 'Atom', atom2: 'Atom', num_electrons_remaining: int) -> None:
        """Unpairs one of the valence orbitals between the two specified atoms"""
        vos = self.get_vos_between(atom1, atom2)
        if len(vos) == 0:
            raise ValueError("unpair_vo(): No valence orbitals to unpair")
        vos[0].unpair(num_electrons_remaining=num_electrons_remaining)

    def unpair_vo_by_idx(self, atom1_idx: int, atom2_idx: int, num_electrons_remaining: int) -> None:
        """Calls unpair_vo() with atom indices rather than atom objects."""
        atom1, atom2 = self.get_atom_by_idx(atom1_idx), self.get_atom_by_idx(atom2_idx)
        return self.unpair_vo(atom1, atom2, num_electrons_remaining=num_electrons_remaining)

    def make_vo_pairing(self, atom1: Atom, atom2: Atom) -> None:
        """Makes a pairing between two atoms"""
        atom1.make_vo_pairing(atom2)

    def make_vo_pairing_by_idx(self, atom1_idx: int, atom2_idx: int) -> None:
        """Makes a pairing between two atoms"""
        atom1, atom2 = self.get_atom_by_idx(atom1_idx), self.get_atom_by_idx(atom2_idx)
        self.make_vo_pairing(atom1, atom2)

    def get_all_atomic_charges(self) -> List[int]:
        """Returns a list of all atomic charges in the molecule"""
        return [atom.get_formal_charge() for atom in self.atoms]

    def get_all_atomic_radicals(self) -> List[int]:
        """Returns a list of the number of radicals of each atom the molecule"""
        return [atom.get_num_radical_electrons() for atom in self.atoms]

    def get_total_charges_radicals(self) -> Tuple[int, int]:
        """Returns the total charge and radical electrons in the molecule"""
        return np.sum(np.abs(self.get_all_atomic_charges())).item(), np.sum(self.get_all_atomic_radicals()).item()
    
    def to_networkx(self) -> nx.Graph:
        """Convert a molecule to a networkx graph for visualization"""
        # Get VO to idx correspondences
        all_vos = self.get_valence_orbitals_by_index()
        G = nx.Graph()
        for idx, vo in enumerate(all_vos):
            G.add_node(vo.index)
        for idx, vo in enumerate(all_vos):
            for vo_same_atom in vo.atom.valence_orbitals:
                if vo_same_atom is vo:
                    continue
                G.add_edge(vo.index, vo_same_atom.index)
            if vo.neighbor:
                G.add_edge(vo.index, vo.neighbor.index)
        return G
    
    def get_atom_pooling(self) -> torch.tensor:
        atom_pool = torch.zeros((len(self.get_all_valence_orbitals()), 12))
        next_atom = None
        index_list = []
        count = 0
        first_vo = True
        for i, vo in enumerate(self.get_all_valence_orbitals()):
            if first_vo:
                next_atom = vo.atom
                first_vo = False
                index_list.append(i)
            elif vo.atom != next_atom:
                # Duplicate the index_tensor to fit in the atom pool (the smallest multiple of 4 and 6 is 12!)
                duplicates = int(12/(count+1))
                index_tensor = torch.cat([torch.Tensor(index_list)]*duplicates)
                # Assign mapping to all VOs that share an atom
                for c in range(count+1):
                    atom_pool[i-c-1,:] = index_tensor
                # Iterate the atom, reset the count and the index tensor and assign the first value of the index tensor
                next_atom = vo.atom
                count = 0
                index_list = []
                index_list.append(i)
            else:
                # Assign the next value in the index tensor
                count += 1
                index_list.append(i)

        # Run this part of the loop one more time to catch the last atom
        duplicates = int(12/(count+1))
        index_tensor = torch.cat([torch.Tensor(index_list)]*duplicates)
        for c in range(count+1):
            atom_pool[i-c,:] = index_tensor
        
        return atom_pool.long()

    def get_mask(self, num_unstable_charges: int) -> torch.tensor:
        '''
        Returns a N x N x 6 tensor mask corresponding to possible allowed actions in the molecule,
        where N is the number of valence orbitals in the system.

        Bond breakage and non-productive orbital/bond interactions are not allowed if they would make
        the molecule too unstable (i.e. too many charges or radicals), and the maximum number of
        charges/radicals is set by the parameter num_unstable_charges.
        '''
        vos = [(idx, vo) for idx, vo in enumerate(self.get_all_valence_orbitals())]

        # leave exactly one copy of isomorphic VOs
        symmetry_unbonded_vos = set()
        symmetry_ignored_vos = set()
        for idx, vo in vos:
            if (vo.atom.idx, vo.num_electrons) in symmetry_unbonded_vos:
                symmetry_ignored_vos.add((idx, vo))

            elif vo.num_electrons != -1:
                symmetry_unbonded_vos.add((vo.atom.idx, vo.num_electrons))


        vo_to_idx = {vo: idx for idx, vo in vos}
        mask = torch.zeros((len(vos), len(vos), 6))

        bonded_vos = [(idx, vo) for idx, vo in vos if vo.neighbor]
        unbonded_vos = [(idx, vo) for idx, vo in vos if not vo.neighbor and (idx, vo) not in symmetry_ignored_vos]
        curr_charges, curr_radicals = self.get_total_charges_radicals()

        # allow bond breakages if molecule not too unstable
        for idx, vo in bonded_vos:
            for action in {Action.BREAK_BOND_WITH_0_ELECTRONS, Action.BREAK_BOND_WITH_1_ELECTRON, Action.BREAK_BOND_WITH_2_ELECTRONS}:
                assert isinstance(vo.neighbor, ValenceOrbital)
                if vo.can_perform_vo_action(vo.neighbor, action):
                    new_smi, charges, radicals = vo.get_result_of_action(vo.neighbor, action)
                    assert isinstance(charges, int) and isinstance(radicals, int) and isinstance(new_smi, str)
                    if new_smi and charges <= num_unstable_charges and radicals <= num_unstable_charges:
                        mask[idx][vo_to_idx[vo.neighbor]][action] = 1

        # allow orbital / bond interactions
        for (idx1, vo1), (idx2, vo2) in itertools.product(unbonded_vos, bonded_vos):
            if vo1.can_perform_vo_action(vo2, Action.INTERACT_ORBITAL_WITH_BOND):
                n_e = vo1.num_electrons
                assert isinstance(vo2.neighbor, ValenceOrbital)
                if n_e == 0:
                    # charge decreases by 1, vo2-neighbor charge increases by 1
                    # this is good if either action preserves charges, i.e. if either VO1 atom charge is > 0 or if VO2 neighbor charge is < 0
                    # so, if we are already at limits, then only allow changes which help charge situation
                    if vo1.atom.get_formal_charge() > 0 or vo2.neighbor.atom.get_formal_charge() < 0 or curr_charges < num_unstable_charges - 1:
                        mask[idx1][idx2][Action.INTERACT_ORBITAL_WITH_BOND] = 1

                elif n_e == 1:
                    # conservation of radicals, so allow regardless
                    mask[idx1][idx2][Action.INTERACT_ORBITAL_WITH_BOND] = 1

                elif n_e == 2:
                    # similar logic to n_e = 0 case
                    if vo1.atom.get_formal_charge() < 0 or vo2.neighbor.atom.get_formal_charge() > 0 or curr_charges < num_unstable_charges - 1:
                        mask[idx1][idx2][Action.INTERACT_ORBITAL_WITH_BOND] = 1

                else:
                    raise ValueError("Invalid number of electrons in valence orbital")

        # allow bond formation and single electron transfer
        for (idx1, vo1), (idx2, vo2) in itertools.product(unbonded_vos, unbonded_vos):
            if vo1.can_perform_vo_action(vo2, Action.PAIR_ORBITALS):
                new_smi, charges, radicals = vo1.get_result_of_action(vo2, Action.PAIR_ORBITALS)
                assert isinstance(charges, int) and isinstance(radicals, int) and isinstance(new_smi, str)
                if new_smi and charges <= num_unstable_charges and radicals <= num_unstable_charges:
                    mask[idx1][idx2][Action.PAIR_ORBITALS] = 1

            if vo1.can_perform_vo_action(vo2, Action.SINGLE_ELECTRON_TRANSFER):
                new_smi, charges, radicals = vo1.get_result_of_action(vo2, Action.SINGLE_ELECTRON_TRANSFER)
                assert isinstance(charges, int) and isinstance(radicals, int) and isinstance(new_smi, str)
                if new_smi and charges <= num_unstable_charges and radicals <= num_unstable_charges:
                    mask[idx1][idx2][Action.SINGLE_ELECTRON_TRANSFER] = 1

        return mask.bool()

    def __str__(self) -> str:
        return self.smi

    def __repr__(self) -> str:
        return self.smi