''' Compute the octahedral rotations and tilts of a corner-connected
orthorhombic perovskite structure.

Functions:
    standardize_atoms(atoms: ase.Atoms, to_primitive: bool = True)
    find_MO_bonds(atoms: ase.Atoms)
    pseudocubic_lattice_vectors(atoms: ase.Atoms)
    vector_projection(vector: npt.ArrayLike, normal: npt.ArrayLike)
    bond_angle(bond1: npt.ArrayLike, bond2: npt.ArrayLike, proj_normal: npt.ArrayLike)
'''

import ase
import ase.geometry.geometry as geom
import ase.io
import ase.neighborlist
import numpy as np
import numpy.typing as npt
import spglib
from collections.abc import Iterable

# default anions/cations
anions = ["O", "S", "Se"]
cations = ["Ti", "Zr", "Hf"]

class OctahedralRotations():
    ''' 
    Class to contain information about octahedral rotations in orthorhombic
    corner-connected perovskites.

    Attributes:
        atoms: ase.Atoms
        bond_pairs: ndarray
        bond_distances: ndarray
        site_bond_angles: NDArray
                Array of projected bond angles at each individual anion site.
                Set when compute_angles() is called.
        rotation: float
                Rotation angle in degrees
        tilt_a: float
                Tilt angle (along a) in degrees
        tilt_b: float
                Tilt angle (along b) in degrees
    '''
    # TODO: implement octahedral rotation pattern recognition.
    def __init__(self, 
            atoms: ase.Atoms, 
            anions: Iterable[str] = anions,
            cations: Iterable[str] = cations
            ) -> None:
        '''
        Initialize OctahedralRotations object.
        Only requires an ase.Atoms object; however, anions and cations are
        optional arguments to specify the atomic species to treat as the anions
        and B-site cations, respectively.
        '''

        self.atoms = standardize_atoms(atoms)
        self._anions = anions
        self._cations = cations
        

        self.bond_pairs, self.bond_distances = self._find_bonds()

        self.site_bond_angles = None
        self.rotation = None
        self.tilt_a = None
        self.tilt_b = None

    def compute_angles(self):
        '''
        Computes projected bond angles.
        '''
        lat_vec, lat_param = self.get_pseudocubic_lattice()
        for site_idx in np.unique(self.bond_pairs[:,0]):
            site_bond_idx = np.where(self.bond_pairs == site_idx)[0]
            # TODO: at each anion site, compute projected bond angles for each
            # of the three projection planes in lat_vec

    def get_pseudocubic_lattice(self):
        ''' Returns pseudocubic lattice vectors as unit vectors and lengths.
        Assumes that the long axis (c) is atoms.cell[2]

        Returns: (pseudo_unit_vectors, lengths)
        pseudo_unit_vectors: np.ndarray
                unit vectors of pseudocubic vectors
        lengths: np.ndarray
                lengths of each of the three pseudocubic vectors
        '''

        # compute pseudocubic lattice vectors a_p and b_p
        a_pseudo = (self.atoms.cell.array[0] + self.atoms.cell.array[1]) / np.sqrt(2)
        b_pseudo = (self.atoms.cell.array[0] - self.atoms.cell.array[1]) / np.sqrt(2)

        pseudo_unit_vectors = np.array([
            a_pseudo / np.linalg.norm(a_pseudo),
            b_pseudo / np.linalg.norm(b_pseudo),
            self.atoms.cell.array[2] / np.linalg.norm(self.atoms.cell.array[2])
            ])

        lengths = np.array([
            np.linalg.norm(a_pseudo),
            np.linalg.norm(b_pseudo),
            self.atoms.cell.lengths()[2]
            ])

        return (pseudo_unit_vectors, lengths)

    def _find_bonds(self) -> tuple[npt.NDArray, npt.NDArray]:
        ''' Finds B-site cation-anion pairs and the vectors connecting them.
        Only bonds centered at the anions are returned.

        Parameters:
            atoms: ase.Atoms

        Returns:
            (bond_pairs, bond_distances)
            bond_pairs: npt.NDArray[int]
                    Each element contains a pair of ion indices [ion1, ion2],
                    where ion1 is the anion and ion2 is the B-site cation.
            bond_distances: npt.NDArray[float]
                    Each element is a 3-dim vector pointing from ion1 to ion2.
        '''
        # For each nearest neighbor ("bonded") pair:
        #   i: ion 1 index
        #   j: ion 2 index
        #   D: distance vector
        cutoff = ase.neighborlist.natural_cutoffs(self.atoms)
        i, j, D = ase.neighborlist.neighbor_list('ijD', self.atoms, cutoff=cutoff)

        # find cation and anion indices
        cation_indices = np.where(
                [atom in cations for atom in self.atoms.get_chemical_symbols()]
                )[0]
        anion_indices = np.where(
                [atom in anions for atom in self.atoms.get_chemical_symbols()]
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

def find_MO_bonds(atoms: ase.Atoms) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.float64]]:
    ''' Finds B-site cation-anion pairs and the vectors connecting them.
    Only bonds centered at the anions are returned.

    Parameters:
        atoms: ase.Atoms

    Returns:
        (bond_pairs, bond_distances)
        bond_pairs: npt.NDArray[int]
        bond_distances: npt.NDArray[float]
    '''
    atoms_std = standardize_atoms(atoms)
    # For each nearest neighbor ("bonded") pair:
    #   i: ion 1 index
    #   j: ion 2 index
    #   D: distance vector
    cutoff = ase.neighborlist.natural_cutoffs(atoms_std)
    i, j, D = ase.neighborlist.neighbor_list('ijD', atoms_std, cutoff=cutoff)

    # find cation and anion indices
    cation_indices = np.where(
            [atom in cations for atom in atoms_std.get_chemical_symbols()]
            )[0]
    anion_indices = np.where(
            [atom in anions for atom in atoms_std.get_chemical_symbols()]
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

def pseudocubic_lattice_vectors(atoms: ase.Atoms) -> tuple[npt.NDArray, npt.NDArray]:
    ''' Takes orthorhombic perovskite and returns its pseudocubic lattice
    vectors as unit vectors and lengths.
    Assumes that the long axis (c) is atoms.cell[2]

    Parameters:
    atoms: ase.Atoms

    Returns: (pseudo_unit_vectors, lengths)
    pseudo_unit_vectors: np.ndarray
            unit vectors of pseudocubic vectors
    lengths: np.ndarray
            lengths of each of the three pseudocubic vectors
    '''

    # compute pseudocubic lattice vectors a_p and b_p
    a_pseudo = (atoms.cell.array[0] + atoms.cell.array[1]) / np.sqrt(2)
    b_pseudo = (atoms.cell.array[0] - atoms.cell.array[1]) / np.sqrt(2)

    pseudo_unit_vectors = np.array([
        a_pseudo / np.linalg.norm(a_pseudo),
        b_pseudo / np.linalg.norm(b_pseudo),
        atoms.cell.array[2] / np.linalg.norm(atoms.cell.array[2])
        ])

    lengths = np.array([
        np.linalg.norm(a_pseudo),
        np.linalg.norm(b_pseudo),
        atoms.cell.lengths()[2]
        ])

    return (pseudo_unit_vectors, lengths)

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
