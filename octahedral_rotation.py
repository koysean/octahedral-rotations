''' Compute the octahedral rotations and tilts of a corner-connected
orthorhombic perovskite structure.

Functions:
    standardize_atoms(xtl: ase.Atoms, to_primitive: bool = True)
'''
import ase
import ase.geometry.geometry as geom
import ase.io
import ase.neighborlist
import ase.spacegroup as spacegroup
import numpy as np
import numpy.typing as npt
import spglib

anions = ["O", "S", "Se"] # anions to search for bonds

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
    if not standardized_cell:
        raise ValueError("spglib.standardize_cell() returned None; check that xtl is a valid ase.Atoms object.")

    xtl_std = ase.Atoms(
            cell=standardized_cell[0],
            scaled_positions=standardized_cell[1],
            symbols=standardized_cell[2],
            pbc=True
            )

    # permute axes (if necessary) so that c is the long axis.
    lp = xtl_std.cell.lengths() # lattice parameters

    if np.argmax(lp) == 2: # no permutation needed
        return xtl_std
    elif np.argmax(lp) == 0:
        permute = [1, 2, 0]
    else:
        permute = [2, 0, 1]

    xtl_std = geom.permute_axes(xtl_std, permute)

    return xtl_std

def find_MO_bonds(xtl: ase.Atoms):
    xtl_std = standardize_atoms(xtl)
    cutoff = ase.neighborlist.natural_cutoffs(xtl_std)
    i, j, D = ase.neighborlist.neighbor_list('ijD', xtl_std, cutoff=cutoff)
    return i, j, D
