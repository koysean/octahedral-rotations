''' Compute the octahedral rotations and tilts of a corner-connected
orthorhombic perovskite structure.

Functions:
    standardize_atoms(xtl: ase.Atoms, to_primitive: bool = True)
'''
import ase
import ase.io
import ase.spacegroup as spacegroup
import numpy as np
import numpy.typing as npt
import spglib
import typing


def standardize_atoms(xtl: ase.Atoms, to_primitive: bool = True)  -> ase.Atoms:
    ''' Converts ASE Atoms object to standard cell via spglib

    Parameters:
        xtl (ase.Atoms)
        to_primitive (bool)

    Returns:
        cell, positions, chemical_numbers (tuple(ndarray[np.float64],
            ndarray[np.float64], ndarray.int32)):
        '''
    # Using spglib to convert xtl to standardized cell
    spglib_cell = (xtl.cell, xtl.get_scaled_positions(), xtl.numbers)
    standardized_cell = spglib.standardize_cell(spglib_cell, to_primitive=to_primitive)

    # If spglib.standardize_cell() doesn't return a value, catch the error
    atoms = ase.Atoms()
    if not standardized_cell:
        raise ValueError("spglib.standardize_cell() returned None; check that xtl is a valid ase.Atoms object.")

    atoms.set_cell(standardized_cell[0])
    atoms.set_scaled_positions(standardized_cell[1])
    atoms.set_chemical_symbols(standardized_cell[2])
    atoms.set_pbc(True)
    # 
    sg = spacegroup.get_spacegroup(atoms)

    return atoms

def get_octahedral_bonds(xtl: ase.Atoms):
    xtl_standard = standardize_atoms(xtl)
    return 0
