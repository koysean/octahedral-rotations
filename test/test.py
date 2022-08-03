import ase
import ase.io
import numpy as np
import os
import sys

sys.path.append("..")

from octahedral_rotation import (
        anions,
        standardize_atoms,
        find_MO_bonds
        )

# Read all test structures into list
xtls = []
struct_dir = "test_structures/"
for i, structure in enumerate(os.listdir(struct_dir)):
    if structure.endswith(".vasp"):
        xtls.append(ase.io.read(struct_dir + structure))
xtls.append(ase.Atoms()) # also add an empty Atoms to xtls

std_at = False
find_bond = True

def main():
    if std_at:
        for xtl in xtls:
            test_standardize_atoms(xtl)
            print(xtl)

    if find_bond:
        for xtl in xtls:
            print(xtl)
            test_find_MO_bonds(xtl)
            print()

def test_standardize_atoms(xtl):
    ''' Check if the lattice parameters are ordered correctly '''
    try:
        xtl_std = standardize_atoms(xtl, True)
    except ValueError:
        print("xtl is None")
        return 1

    cell = xtl_std.cell

    a, b, c = cell.lengths()

    if not ((a <= c) and (b <= c)):
        print("FAILED (c is not long axis)")
        print(xtl_std)
        return 1

    print("SUCCESS")
    return 0

def test_find_MO_bonds(xtl):
    xtl_std = standardize_atoms(xtl, True)
    i, j, D = find_MO_bonds(xtl_std)

    # find 6-coordinate atoms (i.e. B-site cations)
    coord = np.bincount(i)
    cation_indices = np.where(coord == 6)[0]
    print(cation_indices)
    print(xtl_std[cation_indices])

    anion_indices = np.where(
            [atom in anions for atom in xtl_std.get_chemical_symbols()]
            )[0]
    print(anion_indices)
    print(xtl_std[anion_indices])


    #print(i, j, D)

if __name__ == "__main__":
    main()
