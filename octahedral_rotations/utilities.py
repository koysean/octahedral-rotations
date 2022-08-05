'''
Utility functions for OctahedralRotations

Functions:
    standardize_atoms(atoms: ase.Atoms, to_primitive: bool = True)
    vector_projection(vector: npt.ArrayLike, normal: npt.ArrayLike)
    bond_angle(bond1: npt.ArrayLike, bond2: npt.ArrayLike, proj_normal: npt.ArrayLike)
'''
import ase
import ase.geometry.geometry as geom
import numpy as np
import numpy.typing as npt
import spglib

def standardize_atoms(atoms: ase.Atoms, to_primitive: bool = True) -> ase.Atoms:
    ''' Converts ASE Atoms object to standard cell using spglib

    Parameters:
        atoms (ase.Atoms)
        to_primitive (bool)

    Returns:
        (cell, positions, chemical_numbers)
            cell (ndarray[np.float64])
            positions (ndarray[np.float64])
            chemical_numbers (ndarray[int32])
    '''
    # Using spglib to convert atoms to standardized cell
    spglib_cell = (atoms.cell, atoms.get_scaled_positions(), atoms.numbers)
    standardized_cell = spglib.standardize_cell(spglib_cell, to_primitive=to_primitive)

    # If spglib.standardize_cell() doesn't return a value, catch the error
    if not standardized_cell:
        raise ValueError("spglib.standardize_cell() returned None; check that atoms is a valid ase.Atoms object.")

    atoms_std = ase.Atoms(
            cell=standardized_cell[0],
            scaled_positions=standardized_cell[1],
            symbols=standardized_cell[2],
            pbc=True
            )

    # permute axes (if necessary) so that c is the long axis.
    lp = atoms_std.cell.lengths() # lattice parameters

    if np.argmax(lp) == 2: # no permutation needed
        return atoms_std
    elif np.argmax(lp) == 0:
        permute = [1, 2, 0]
    else:
        permute = [2, 0, 1]

    atoms_std = geom.permute_axes(atoms_std, permute)

    return atoms_std


def vector_projection(vector: npt.ArrayLike, normal: npt.ArrayLike) -> npt.NDArray:
    ''' Projects vector onto the plane defined by the normal vector.

    The vector projection is given by:
        vector - DOT_PRODUCT(vector, normal) * normal

    Parameters:
        vector: npt.ArrayLike
        normal: npt.ArrayLike

    Returns:
        proj: npt.NDArray
    '''
    vector = np.asarray(vector)
    normal = np.asarray(normal)

    if not np.isclose(np.linalg.norm(normal), 1):
        normal /= np.linalg.norm(normal)

    proj = vector - (np.dot(vector, normal)) * normal

    return proj

def bond_angle(bond1: npt.ArrayLike, bond2: npt.ArrayLike,
        proj_normal: npt.ArrayLike) -> float:
    '''
    Computes bond angle projected onto a plane.

    Parameters:
        bond1: npt.ArrayLike
                Vector of first bond
        bond2: npt.ArrayLike
                Vector of second bond
        proj_normal: npt.ArrayLike
                Normal vector to the projection plane

    Returns:
        angle: float
                bond angle in degrees
    '''

    proj1 = vector_projection(bond1, proj_normal)
    proj2 = vector_projection(bond2, proj_normal)
    l1 = np.linalg.norm(proj1)
    l2 = np.linalg.norm(proj2)

    angle = np.arccos(
            np.dot(proj1, proj2) / (l1 * l2)
            )
    angle = np.rad2deg(angle)

    return angle
