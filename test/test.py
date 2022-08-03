import ase
import ase.io
import numpy as np

from octahedral_rotation import (standardize_atoms)

test_structure = "CONTCAR_S_Pnma.vasp"
xtl = ase.io.read(test_structure)
emptyXtl = ase.Atoms()

def test_get_standardized_cell():
    out1 = standardize_atoms(xtl)
    out2 = standardize_atoms(xtl, False)

    print(out2)

    out3 = standardize_atoms(emptyXtl)
    print(out3)
