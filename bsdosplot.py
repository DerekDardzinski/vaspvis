import matplotlib as mpl
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pymatgen.electronic_structure.core import Spin, Orbital
from pymatgen.io.vasp.inputs import Kpoints
from pymatgen.io.vasp.outputs import Procar, BSVasprun
import src.band as bs
import argparse


mpl.rcParams['font.family'] = "serif"

# =============================================
# ----------------- Inputs --------------------
# =============================================

# folder = './band'
# atoms = [0]
# orbitals = [0, 1, 2, 3, 4, 5, 6, 7, 8]


def parse_arguments():
    parser = argparse.ArgumentParser(description='param')
    parser.add_argument('--folder', dest='folder', type=str, default='band/')
    parser.add_argument('--atoms', dest='atoms', type=int, nargs='+', default=[-1])
    parser.add_argument('--orbitals', dest='orbitals', type=int, nargs='+', default=[-1])
    parser.add_argument('--fontsize', dest='fontsize', type=float, default=20)
    parser.add_argument('--ylim', dest='ylim', type=float,
                        nargs='+', default=[-6, 6])
    parser.add_argument('--figsize', dest='figsize',
                        type=float, nargs='+', default=[4, 3])
    parser.add_argument('--dos', dest='dos', type=str, default='True')
    parser.add_argument('--spin', dest='spin', type=str,
                        nargs='+', default=['up'])
    return parser.parse_args()


args = parse_arguments()
folder = args.folder
atoms = args.atoms
orbitals = args.orbitals
fontsize = args.fontsize
ylim = args.ylim
figsize = args.figsize
dos = args.dos.lower().strip()
spins = [i.lower().strip() for i in args.spin]

# =============================================

# dict_keys(['filename', 'occu_tol', 'efermi', 'eigenvalues', 'projected_eigenvalues', 'generator', 'incar', 'kpoints', 'actual_kpoints', 'actual_kpoints_weights', 'parameters', 'atomic_symbols', 'potcar_symbols', 'potcar_spec', 'final_structure', 'vasp_version'])


vasprun_file = f'{folder}/vasprun.xml'
kpoints_file = f'{folder}/KPOINTS'

spin_dict = {'up': Spin.up, 'down': Spin.down}

fontsize = 6
fig = plt.figure(figsize=(figsize[0], figsize[1]), dpi=300)
ax = fig.add_subplot(111)
plt.ylabel('$E - E_{F}$ (eV)', fontsize=fontsize)
plt.ylim(ylim[0], ylim[1])

if atoms != [-1] and orbitals != [-1]:
    vasprun = BSVasprun(vasprun_file, parse_projected_eigen=True)
    kpoints = Kpoints.from_file(kpoints_file)

    for spin in spins:
        bs.orbital_atom_band(
            vasprun=vasprun,
            kpoints=kpoints,
            orbitals=orbitals,
            atoms=atoms,
            ax=ax,
            spin=spin_dict[spin]
        )

elif atoms != [-1] and orbitals == [-1]:
    vasprun = BSVasprun(vasprun_file, parse_projected_eigen=True)

    kpoints = Kpoints.from_file(kpoints_file)

    for spin in spins:
        bs.atom_band(
            vasprun=vasprun,
            kpoints=kpoints,
            atoms=atoms,
            ax=ax,
            spin=spin_dict[spin]
        )

elif atoms == [-1] and orbitals != [-1]:
    vasprun = BSVasprun(vasprun_file, parse_projected_eigen=True)

    kpoints = Kpoints.from_file(kpoints_file)

    for spin in spins:
        bs.orbital_band(
            vasprun=vasprun,
            kpoints=kpoints,
            orbitals=orbitals,
            ax=ax,
            spin=spin_dict[spin]
        )

elif atoms == [-1] and orbitals == [-1]:
    vasprun = BSVasprun(vasprun_file, parse_projected_eigen=False)

    kpoints = Kpoints.from_file(kpoints_file)

    for spin in spins:
        bs.plain_band(
            vasprun=vasprun,
            kpoints=kpoints,
            ax=ax,
            spin=spin_dict[spin]
        )

plt.tick_params(labelsize=fontsize, length=1.5)
plt.tick_params(axis='x', length=0)
plt.tight_layout(pad=0.5)
plt.savefig('bs.png')
plt.show()
