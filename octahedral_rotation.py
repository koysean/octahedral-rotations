''' Compute the octahedral rotations and tilts of a corner-connected
orthorhombic perovskite structure.

Functions:
    standardize_atoms(xtl: ase.Atoms, to_primitive: bool = True)
    find_MO_bonds(xtl: ase.Atoms)
'''
import ase
import ase.geometry.geometry as geom
import ase.io
import ase.neighborlist
import ase.spacegroup as spacegroup
import numpy as np
import numpy.typing as npt
import spglib
from typing import Any

anions = ["O", "S", "Se"] # anions to search for bonds
cations = ["Ti", "Zr", "Hf"]

def standardize_atoms(xtl: ase.Atoms, to_primitive: bool = True)  -> ase.Atoms:
    ''' Converts ASE Atoms object to standard cell using spglib

    Parameters:
        xtl (ase.Atoms)
        to_primitive (bool)

    Returns:
        (cell, positions, chemical_numbers)
            cell (ndarray[np.float64])
            positions (ndarray[np.float64])
            chemical_numbers (ndarray[int32])
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

def find_MO_bonds(xtl: ase.Atoms) -> tuple[npt.NDArray[np.int32], npt.NDArray[Any]]:
    ''' Finds B-site cation-anion pairs and the vectors connecting them.

    Parameters:
        xtl (ase.Atoms)

    Returns:
        bonds (Tuple[ndarray[int], ndarray[float]])
    '''
    xtl_std = standardize_atoms(xtl)
    # For each nearest neighbor ("bonded") pair:
    #   i: ion 1 index
    #   j: ion 2 index
    #   D: distance vector
    cutoff = ase.neighborlist.natural_cutoffs(xtl_std)
    i, j, D = ase.neighborlist.neighbor_list('ijD', xtl_std, cutoff=cutoff)

    # find cation and anion indices
    cation_indices = np.where(
            [atom in cations for atom in xtl_std.get_chemical_symbols()]
            )[0]
    anion_indices = np.where(
            [atom in anions for atom in xtl_std.get_chemical_symbols()]
            )[0]

    # enumerate valid B-site cation-anion bonds
    bond_pairs = [] # populate with bond pairs [ion1, ion2]
    bond_distances = [] # populate with distance vectors [x, y, z]
    for idx, atom in enumerate(i):
        # only count bond pairs where central atom is an anion and bonded
        # atom is a B-site cation
        if (atom in anion_indices) and (j[idx] in cation_indices):
            bond_pairs.append([atom, j[idx]])
            bond_distances.append(D[idx])

    return (np.array(bond_pairs), np.array(bond_distances))
