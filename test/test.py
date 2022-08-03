import ase
import ase.io
import numpy as np
import os
import sys

sys.path.append("..")

from octahedral_rotation import (
        anions, cations,
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
        #xtl = ase.io.read("test_structures/Pnma7.vasp")
        #test_find_MO_bonds(xtl)

        for xtl in xtls:
            print(xtl.symbols)
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
    # bond pairs and bond distances (anion centers)
    bp, bd = find_MO_bonds(xtl_std)

    # count bonds
    bond_centers = bp[:,0]
    bond_counts = np.bincount(bond_centers)[np.unique(bond_centers)]

    # require anions to have coordination environment of 2 B-site cations
    if not ((bond_counts == 2).all()):
        print("FAILURE - anion has more than 2 bonds")
        return 1
    # XXX: bond distance vectors were not thoroughly checked - may require
    # further testing.
    else:
        print("SUCCESS")
        return 0

if __name__ == "__main__":
    main()
