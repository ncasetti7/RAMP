""""Module for testing NNP calculations"""

import unittest
import os
import sys
import torch
from rdkit import Chem
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src/RAMP")
from RAMP.nnp_calculations import aimnet2sph
from RAMP.utils import calculations, converter


class TestEnumeration(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestEnumeration, self).__init__(*args, **kwargs)
        self.model_file = "../src/RAMP/nnp_calculations/model_1.jpt"
        self.model = torch.jit.load(self.model_file)
        self.calc = aimnet2sph.AIMNet2Calculator(self.model)

    def test_optimization(self):
        print("\nTesting conformer generation...")
        smi = "O=CCCOO"

        # Generate conformers
        n_confs = 3
        input_dict = {'smi': smi, 'n_confs': n_confs}
        conformers = calculations.ff_generate_conformers(input_dict)

        print("Testing conformer optimization")
        # Optimize the conformers
        opt_input = {'conformers': conformers, 'canon_smiles': smi}
        calculations.calc_prod_energy(opt_input, self.calc)
        os.system("rm *.xyz *.log")

    def test_NEB_TS_IRC(self):
        print("\nTesting reaction conformer generation...")
        canon_reaction = "O=CCCOO>>O[C@@H]1CCOO1"
        rxn_smiles = "[O:1]=[C:2]([C:3]([C:4]([O:5][O:6][H:12])([H:10])[H:11])([H:8])[H:9])[H:7]>>[O:1]([C:2]1([H:7])[C:3]([H:8])([H:9])[C:4]([H:10])([H:11])[O:5][O:6]1)[H:12]"
        reactant_canon = canon_reaction.split(">>")[0]
        n_confs = 100
        rxn_confs = 4
        reverse = False
        input_dict = {'rxn': rxn_smiles, 'reactant_canon': reactant_canon, 'reverse': reverse, 'rxn_confs': rxn_confs, 'n_confs': n_confs}
        conformers = calculations.ff_generate_rxn_conformers(input_dict)
        
        print("Testing reaction conformer optimization...")
        opt_conformers = calculations.calc_rxn_conformer(conformers[0], self.calc)

        print("Testing NEB calculation...")
        neb_dict = {'geoms': [opt_conformers[0][0], opt_conformers[0][1]]}
        neb_conformer = calculations.calc_neb(neb_dict, self.calc)

        print("Testing TS search...")
        ts_dict = {'ts_guess': neb_conformer}
        ts_conformer, ts_energy = calculations.calc_ts(ts_dict, self.calc)

        print("Testing IRC calculation...")
        irc_dict = {'ts_geom': ts_conformer, "energy": ts_energy}
        irc_data = calculations.calc_irc(irc_dict, self.calc)
        first = irc_data[0][0]
        last = irc_data[0][2]
        first_mol = converter.geom2rdmol(first)
        last_mol = converter.geom2rdmol(last)
        first_smiles = Chem.CanonSmiles(Chem.MolToSmiles(first_mol))
        last_smiles = Chem.CanonSmiles(Chem.MolToSmiles(last_mol))
        assert first_smiles == canon_reaction.split(">>")[0] and last_smiles == canon_reaction.split(">>")[1] or first_smiles == canon_reaction.split(">>")[1] and last_smiles == canon_reaction.split(">>")[0]
        os.system("rm *.h5 *.xyz *.log *.trj")
        os.system("rm -r qm_calcs")


if __name__ == '__main__':
    unittest.main()
