import ase
import ase.io

from octahedral_rotations import OctahedralRotations

# Test script to find rotations in CaTiS3 polymorphs
pnma: ase.Atoms = ase.io.read("structures/Pnma.vasp")
pna21: ase.Atoms = ase.io.read("structures/Pna21.vasp")
pmn21: ase.Atoms = ase.io.read("structures/Pmn21.vasp")

pnma_oct = OctahedralRotations(pnma)
pna21_oct = OctahedralRotations(pna21)
pmn21_oct = OctahedralRotations(pmn21)

pnma_oct.compute_angles()
pna21_oct.compute_angles()
pmn21_oct.compute_angles()

print(pnma_oct.atoms.symbols)
print("Octahedral rotations:", pnma_oct.mean_rotation)
print("Octahedral tilts (x):", pnma_oct.mean_tilt_a)
print("Octahedral tilts (y):", pnma_oct.mean_tilt_b)
print()

print(pna21_oct.atoms.symbols)
print("Octahedral rotations:", pna21_oct.mean_rotation)
print("Octahedral tilts (x):", pna21_oct.mean_tilt_a)
print("Octahedral tilts (y):", pna21_oct.mean_tilt_b)
print()

print(pmn21_oct.atoms.symbols)
print("Octahedral rotations:", pmn21_oct.mean_rotation)
print("Octahedral tilts (x):", pmn21_oct.mean_tilt_a)
print("Octahedral tilts (y):", pmn21_oct.mean_tilt_b)
print()
