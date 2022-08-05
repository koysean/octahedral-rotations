''' Compute the octahedral rotations and tilts of a corner-connected
orthorhombic perovskite structure.

Classes:
    OctahedralRotations(atoms: ase.Atoms, 
            anions: Iterable[str] = anions,
            cations: Iterable[str] = cations)

Functions:
    (DEPRECATED) find_MO_bonds(atoms: ase.Atoms)
    (DEPRECATED) pseudocubic_lattice_vectors(atoms: ase.Atoms)
'''

import ase
import ase.io
import ase.neighborlist
from collections.abc import Iterable
import numpy as np
import numpy.typing as npt
import warnings
import utilities


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
                dim: [n_anions, 3]. Angles are ordered tilt_a, tilt_b, and
                rotation
        mean_rotation: float
                Rotation angle in degrees
        mean_tilt_a: float
                Tilt angle (along a) in degrees
        mean_tilt_b: float
                Tilt angle (along b) in degrees
    '''
    # TODO: implement octahedral rotation pattern recognition.
    # TODO: implement cations and anions as properties.
    '''
            cations and anions as properties implementation:
            when set: 
                    * the format should be checked (Iterable[str])
                    * self.bond_pairs, self.bond_distances, self.anion_sites
                        should all be recomputed.
                    * self.site_bond_angles, self.mean_* should all be unset.
    '''
    
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

        self.atoms = utilities.standardize_atoms(atoms)
        self._anions = anions
        self._cations = cations
        

        self.bond_pairs, self.bond_distances = self._find_bonds()

        self.anion_sites = np.unique(self.bond_pairs[:,0])
        self.site_bond_angles = []
        self.mean_rotation = None
        self.mean_tilt_a = None
        self.mean_tilt_b = None

    def compute_angles(self):
        '''
        Computes projected bond angles.
        '''
        site_bond_angles = []
        lat_vec, lat_param = self.get_pseudocubic_lattice()
        for site_idx in self.anion_sites:
            site_bond_angles.append([])
            # for each site, find the two bonds from that anion to B-site
            # cations
            site_bond_idx = np.where(self.bond_pairs == site_idx)[0]
            bond1 = self.bond_distances[site_bond_idx[0]]
            bond2 = self.bond_distances[site_bond_idx[1]]

            # at each anion site, compute projected bond angles for each of the
            # three projection planes in lat_vec
            for v in lat_vec:
                theta = np.nan_to_num(utilities.bond_angle(bond1, bond2, v))
                site_bond_angles[-1].append(theta)

        self.site_bond_angles = np.array(site_bond_angles)

        # use temp to reference each type of angle
        temp = self.site_bond_angles[:,0] #tilt_a
        self.mean_tilt_a = np.mean(temp[~np.isclose(temp, 0, atol=0.001)])
        temp = self.site_bond_angles[:,1] #tilt_b
        self.mean_tilt_b = np.mean(temp[~np.isclose(temp, 0, atol=0.001)])
        temp = self.site_bond_angles[:,2] #rotation
        self.mean_rotation = np.mean(temp[~np.isclose(temp, 0, atol=0.001)])

    def get_pseudocubic_lattice(self) -> tuple[npt.NDArray, npt.NDArray]:
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
                [atom in self._cations for atom in self.atoms.get_chemical_symbols()]
                )[0]
        anion_indices = np.where(
                [atom in self._anions for atom in self.atoms.get_chemical_symbols()]
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
    warnings.warn("This function has been moved to an OctahedralRotations class method!", DeprecationWarning)

    atoms_std = utilities.standardize_atoms(atoms)
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
    warnings.warn("This function has been moved to an OctahedralRotations class method!", DeprecationWarning)

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

