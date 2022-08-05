import ase
import ase.io
import numpy as np
import os
import sys
import warnings

sys.path.append("..")

from octahedral_rotation import (
        OctahedralRotations,
        find_MO_bonds,
        pseudocubic_lattice_vectors,
        )
from utilities import(
        standardize_atoms,
        vector_projection,
        bond_angle
        )

### PREPARING STRUCTURES FOR TESTING
xtls = []
struct_dir = "test_structures/"
for i, structure in enumerate(os.listdir(struct_dir)):
    if structure.endswith(".vasp"):
        xtls.append(ase.io.read(struct_dir + structure))
#xtls.append(ase.Atoms()) # add an empty Atoms to xtls

# octs contains OctahedralRotations objects for each structure
oct_rots = []
for xtl in xtls:
    oct_rots.append(OctahedralRotations(xtl))


### FLAGS TO ACTIVATE UNIT TESTS
std_at = True # standardize_atoms()
find_bond = True # find_MO_bonds()
pseudo = True # pseudocubic_lattice_vectors()
proj = True # vector_projection()
angle = True # bond_angle()

def main():
    test_counter = 0
    success_counter = 0
    fail_counter = 0

    if std_at:
        print("Running standardize_atoms() tests")
        for xtl in xtls:
            print(xtl.symbols)
            test_counter += 1
            err = test_standardize_atoms(xtl)
            if err:
                fail_counter += 1
            else:
                success_counter += 1
            print()

        print("\n\n")

    if find_bond:
        print("Running find_MO_bonds() tests")
        for oct in oct_rots:
            print(oct.atoms.symbols)
            test_counter += 1
            err = test_find_MO_bonds(oct)
            if err:
                fail_counter += 1
            else:
                success_counter += 1
            print()

        print("\n\n")

    if pseudo:
        test_pseudocubic_lattice_vectors(oct_rots[0])
        print("\n\n")

    if proj:
        print("Running vector_projection() tests")
        test_counter += 1
        err = test_vector_projection()
        if err:
            fail_counter += 1
        else:
            success_counter += 1
        print("\n\n")

    if angle:
        print("Running bond_angle() tests")
        test_counter += 1
        err = test_bond_angle()
        if err:
            fail_counter += 1
        else:
            success_counter += 1
        print("\n\n")

    print("All tests complete!")
    print("{:d} TESTS RUN. {:d} SUCCESSES, {:d} FAILURES.".format(
        test_counter, success_counter, fail_counter))

def test_standardize_atoms(xtl):
    ''' Check if the lattice parameters are ordered correctly '''
    try:
        xtl_std = standardize_atoms(xtl, True)
    except ValueError:
        print("FAILURE: xtl is None")
        return 1

    cell = xtl_std.cell

    a, b, c = cell.lengths()

    if not ((a <= c) and (b <= c)):
        print("FAILED (c is not long axis)")
        print(xtl_std)
        return 1

    print("SUCCESS!")
    return 0

def test_find_MO_bonds(oct):
    #xtl_std = standardize_atoms(xtl, True)
    # bond pairs and bond distances (anion centers)
    bp, bd = oct._find_bonds()

    # count bonds
    bond_centers = bp[:,0]
    bond_counts = np.bincount(bond_centers)[np.unique(bond_centers)]

    # require anions to have coordination environment of 2 B-site cations
    if not ((bond_counts == 2).all()):
        print("FAILURE - anion has more than 2 bonds")
        return 1
    # XXX: bond distance vectors were not thoroughly checked - may require
    # further testing.

    print("SUCCESS!")
    return 0

def test_pseudocubic_lattice_vectors(oct):
    warnings.warn("Pseudocubic_lattice_vectors() test is not written!",
            RuntimeWarning)
    return 1

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

    print("SUCCESS!")
    return 0

def test_bond_angle():
    # value manually derived from Pnma7.vasp, O8 (idx 15)
    bond_angle_1 = 162.75

    # find bonds from O8 in Pnma7.vasp
    xtl = ase.io.read("test_structures/Pnma7.vasp")
    oct = OctahedralRotations(xtl)
    bp, bd = oct._find_bonds()
    
    o8_bond_idx = np.where(bp[:,0] == 15)[0]
    bond1 = bd[o8_bond_idx[0]]
    bond2 = bd[o8_bond_idx[1]]
    norm = [0, 0, 1]

    angle = bond_angle(bond1, bond2, norm)

    if not np.isclose(angle, bond_angle_1, atol=0.01):
        print("FAILURE")
        return 1

    print("SUCCESS!")
    return 0

if __name__ == "__main__":
    main()
