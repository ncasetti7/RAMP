""""Module for testing intermediate enumeration"""

import unittest
import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src/RAMP")
from RAMP.intermediate_enumeration import enumerate_intermediates, perform_actions


class TestEnumeration(unittest.TestCase):

    def test_enumeration(self):
        print("Testing enumeration...")
        reactant_smiles = "O=CCCOO"
        actions, mol = enumerate_intermediates.enumeration(reactant_smiles, 4, True)
        self.assertEqual(len(actions), 76)
        perform_actions.write_smi_from_actions(mol, actions, workers=os.cpu_count(), batch_size=os.cpu_count()*1000, outfile="tmp.csv")
        with open("tmp.csv", encoding='utf-8') as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 64)
        os.system("rm tmp.csv")

if __name__ == '__main__':
    unittest.main()
