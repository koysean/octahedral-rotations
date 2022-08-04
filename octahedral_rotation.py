''' Compute the octahedral rotations and tilts of a corner-connected
orthorhombic perovskite structure.

Functions:
    standardize_atoms(xtl: ase.Atoms, to_primitive: bool = True)
    find_MO_bonds(xtl: ase.Atoms)
    pseudocubic_lattice_vectors(xtl: ase.Atoms)
    vector_projection(vector: npt.ArrayLike, normal: npt.ArrayLike)
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

anions = ["O", "S", "Se"]
cations = ["Ti", "Zr", "Hf"] # B site cations

def standardize_atoms(xtl: ase.Atoms, to_primitive: bool = True) -> ase.Atoms:
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

def find_MO_bonds(xtl: ase.Atoms) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.float64]]:
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

def pseudocubic_lattice_vectors(xtl: ase.Atoms) -> tuple[npt.NDArray, npt.NDArray]:
    ''' Takes orthorhombic perovskite and returns its pseudocubic lattice
    vectors as unit vectors and lengths.
    Assumes that the long axis (c) is xtl.cell[2]

    Parameters:
    xtl: ase.Atoms

    Returns: (pseudo_unit_vectors, lengths)
    pseudo_unit_vectors: np.ndarray
            unit vectors of pseudocubic vectors
    lengths: np.ndarray
            lengths of each of the three pseudocubic vectors
    '''

    # compute pseudocubic lattice vectors a_p and b_p
    a_pseudo = (xtl.cell.array[0] + xtl.cell.array[1]) / np.sqrt(2)
    b_pseudo = (xtl.cell.array[0] - xtl.cell.array[1]) / np.sqrt(2)

    pseudo_unit_vectors = np.array([
        a_pseudo / np.linalg.norm(a_pseudo),
        b_pseudo / np.linalg.norm(b_pseudo),
        xtl.cell.array[2] / np.linalg.norm(xtl.cell.array[2])
        ])

    lengths = np.array([
        np.linalg.norm(a_pseudo),
        np.linalg.norm(b_pseudo),
        xtl.cell.lengths()[2]
        ])

    return (pseudo_unit_vectors, lengths)

def vector_projection(vector: npt.ArrayLike, normal: npt.ArrayLike) -> npt.NDArray:
    ''' Projects vector onto the plane defined by the normal vector '''
    vector = np.asarray(vector)
    normal = np.asarray(normal)

    if not np.isclose(np.linalg.norm(normal), 1):
        normal /= np.linalg.norm(normal)

    proj = vector - (np.dot(vector, normal)) * normal

    return proj
