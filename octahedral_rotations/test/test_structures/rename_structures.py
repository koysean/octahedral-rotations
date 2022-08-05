import ase.io
import ase.spacegroup
import os
import re

counter = 3

for filename in os.listdir():
    if filename.endswith("3.vasp"):
        xtl = ase.io.read(filename)
        sg = ase.spacegroup.get_spacegroup(xtl)
        if sg.no == 62:
            os.rename(filename, "Pnma{:d}.vasp".format(counter))
            counter += 1
