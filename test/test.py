import ase
import ase.io
import numpy as np
import os
import sys

sys.path.append("..")

from octahedral_rotation import (
        anions, cations,
        standardize_atoms,
        find_MO_bonds,
        pseudocubic_lattice_vectors,
        vector_projection
        )

# Read all test structures into list
xtls = []
struct_dir = "test_structures/"
for i, structure in enumerate(os.listdir(struct_dir)):
    if structure.endswith(".vasp"):
        xtls.append(ase.io.read(struct_dir + structure))
xtls.append(ase.Atoms()) # also add an empty Atoms to xtls

std_at = False # standardize_atoms()
find_bond = False # find_MO_bonds()
pseudo = False # pseudocubic_lattice_vectors()
proj = True # vector_projection()

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

    if pseudo:
        print("No test implemented for pseudocubic_lattice_vectors() yet!")

    if proj:
        test_vector_projection()

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

def test_pseudocubic_lattice_vectors(xtl):
    return 0

def test_vector_projection():
    v1 = [1,0,0]
    v2 = [0.23, 2.4, 3.5]
    n1 = [0, 1, 0]
    n2 = [1/np.sqrt(2), -1/np.sqrt(2), 0]
    n3 = [0.3, 1.5089, -0.1231]

    proj11 = [1, 0, 0]
    proj12 = [0.5, 0.5, 0]
    proj13 = [0.962216, -0.190043, 0.0155042]

    proj21 = [0.23, 0, 3.5]
    proj22 = [1.315, 1.315, 3.5]
    proj23 = [-0.180529, 0.335175, 3.66845]

    if (not np.isclose(vector_projection(v1, n1), proj11).all()
            or (not np.isclose(vector_projection(v1, n2), proj12).all())
            or (not np.isclose(vector_projection(v1, n3), proj13).all())
            or (not np.isclose(vector_projection(v2, n1), proj21).all())
            or (not np.isclose(vector_projection(v2, n2), proj22).all())
            or (not np.isclose(vector_projection(v2, n3), proj23).all())
            ):
        print("FAILURE")
        return 1
    else:
        return 0

if __name__ == "__main__":
    main()
