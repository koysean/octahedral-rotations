import ase
import ase.io

from octahedral_rotations import OctahedralRotations

# Test script to find rotations in CaTiS3 polymorphs
pnma, pna21, pmn21 = ase.io.iread(
        "structures/Pnma.vasp",
        "structures/Pna21.vasp",
        "structures/Pmn21.vasp")

pnma_oct = OctahedralRotations(pnma)
pna21_oct = OctahedralRotations(pna21)
pmn21_oct = OctahedralRotations(pmn21)

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
