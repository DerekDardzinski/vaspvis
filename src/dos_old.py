import matplotlib.pyplot as plt
from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.electronic_structure.core import Spin, Orbital
from pymatgen.electronic_structure.dos import Dos
import numpy as np
import argparse

folder='dos'
tp = 'pdos'

# Command line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='param')
    parser.add_argument('--folder', dest='folder', type=str, default='band/', help='Folder that contains VASP files')
    parser.add_argument('--output', dest='output', type=str, default='bs.png', help='Output file name')
    parser.add_argument('--fontsize', dest='fontsize', type=float, default=10, help='Font size of labels')
    parser.add_argument('--linewidth', dest='linewidth', type=float, default=2, help='Line width of bands')
    parser.add_argument('--erange', dest='erange', type=float, nargs='+', default=[-6, 6], help='y-axis range')
    parser.add_argument('--fill', dest='fill', type=str, default='True', help='Determimes wether to fill underneath the line or not')
    # parser.add_argument('--kpath', dest='kpath', type=str, default='GXWLGK', help='kpoints path used in the calculation')
    parser.add_argumesigmat('--sigma', dest='sigma', type=int, default=0.05, help='Sigma smearing parameter')
    parser.add_argument('--color', dest='color', type=str, default='black', help='Color of line used in plain density of states')
    parser.add_argument('--cmap', dest='cmap', type=str, default='plasma', help='Color gradient for atom/orbital projected density of states')
    parser.add_argument('--orbitals', dest='orbitals', type=int, nargs='+', default=None, help='Projected orbitals, s=0, px, py, pz=1 2 3, ...')
    parser.add_argument('--atoms', dest='atoms', type=int, nargs='+', default=None, help='Projected atoms: 1=first atoms in POSCAR, 1 2=first two atoms, ...')
    return parser.parse_args()

args = parse_arguments()
folder = args.folder.split('/')[0]
output = args.output
fontsize = args.fontsize
erange = args.erange
fill = args.fill.lower()
# kpath = [f'${k}$' if k !='G' else '$\\Gamma$' for k in args.kpath.upper().strip()]
sigma = args.sigma
color = args.color
cmap = args.cmap
linewidth = args.linewidth
orbitals = args.orbitals
atoms = args.atoms


vasprun_file = f'{folder}/vasprun.xml'
POSCAR = f'{folder}/POSCAR'

with open(POSCAR) as poscar:
    atom_info = poscar.readlines()[5:7]


element_symbol = atom_info[0].split()
element_number = atom_info[1].split()
atom_info_dict = {element:int(number) for (element,number) in zip(element_symbol, element_number)}

vasprun = Vasprun(vasprun_file, parse_dos=True, parse_eigen=False)

num2orbital = {
        0: Orbital.s,
        1: Orbital.px,
        2: Orbital.py,
        3: Orbital.pz,
        4: Orbital.dx2,
        5: Orbital.dxy,
        6: Orbital.dxz,
        7: Orbital.dyz,
        8: Orbital.dz2,
        }

orbital_dict = {
        Orbital.s: {},
        Orbital.px: {},
        Orbital.py: {},
        Orbital.pz: {},
        Orbital.dx2: {},
        Orbital.dxy: {},
        Orbital.dxz: {},
        Orbital.dyz: {},
        Orbital.dz2: {},
        }

if tp == 'pdos':
    pdos = vasprun.pdos[0]
    orbital_dict[num2orbital[0]][Spin.up] = pdos[num2orbital[0]][Spin.up]


print(orbital_dict)

if tp == 'tot':
    tdos = vasprun.tdos
    efermi = tdos.efermi
    energies = tdos.energies
    densities = tdos.densities
    densities_smeared = Dos(efermi, energies, densities).get_smeared_densities(0.5)
    spin_up = densities_smeared[Spin.up]
    spin_down = densities_smeared[Spin.down]
    # print(spin_up_smeared)

fig = plt.figure(dpi=300)
ax = fig.add_subplot(111)

# ax.plot(spin_up,energies - efermi)
# plt.savefig('dostest.png')

# pdos = vasprun.pdos[0][orbital_dict[2]]

# print(pdos)

