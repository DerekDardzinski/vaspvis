import matplotlib.pyplot as plt
from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.electronic_structure.core import Spin, Orbital
from pymatgen.electronic_structure.dos import Dos
from scipy.ndimage.filters import gaussian_filter1d
import numpy as np
import time


# PDOS structure = atom -> orbitals -> spin

# ===============================================================
# For next time, merge all p and d orbitals into one
# ===============================================================`


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


def smear(dos, energy, sigma):
    diff = np.diff(energy)
    avgdiff = np.mean(diff)
    smeared_dos = gaussian_filter1d(dos, sigma / avgdiff)

    return smeared_dos


def get_orbilal_atom_pdos(pdos, orbitals, atoms):

    orbital_atom_dict = {
        atom: {
            orbital: {
                'up': pdos[atom][num2orbital[orbital]][Spin.up],
                'down': pdos[atom][num2orbital[orbital]][Spin.down]
            } for orbital in orbitals} for atom in atoms
    }

    return orbital_atom_dict


def get_orbilal_pdos(pdos, orbitals):

    orbital_dict = {
        orbital: {'up': [], 'down': []} for orbital in orbitals
    }

    for orbital in orbitals:
        for (i, atom) in enumerate(pdos):
            orbital_dict[orbital]['up'].append(
                pdos[i][num2orbital[orbital]][Spin.up]
            )
            orbital_dict[orbital]['down'].append(
                pdos[i][num2orbital[orbital]][Spin.down]
            )

        orbital_dict[orbital]['up'] = \
            np.sum(np.matrix(orbital_dict[orbital]['up']), axis=0)
        orbital_dict[orbital]['down'] = \
            np.sum(np.matrix(orbital_dict[orbital]['down']), axis=0)

    return orbital_dict


def get_atom_pdos(pdos, atoms):

    atom_dict = {
        atom: {'up': [], 'down': []} for atom in atoms
    }

    for atom in atoms:
        for orbital in pdos[atom]:
            atom_dict[atom]['up'].append(
                pdos[atom][orbital][Spin.up]
            )
            atom_dict[atom]['down'].append(
                pdos[atom][orbital][Spin.down]
            )

        atom_dict[atom]['up'] = \
            np.squeeze(np.asarray(
                np.sum(np.matrix(atom_dict[atom]['up']), axis=0)))
        atom_dict[atom]['down'] = np.sum(
            np.matrix(atom_dict[atom]['down']), axis=0)

    return atom_dict


def get_tdos(tdos):

    tdos_dict = {
        'energy': tdos.energies - tdos.efermi,
        'up': tdos.densities[Spin.up],
        'down': tdos.densities[Spin.down]
    }

    return tdos_dict


def sum_spd(pdos_dict, orbitals, atoms):
    spd_orbitals = {'s': [0], 'p': [1, 2, 3], 'd': [4, 5, 6, 7, 8]}

    if orbitals != [-1] and atoms != [-1]:
        spd_dict = {atom: {} for atom in atoms}
        for atom in atoms:
            key = np.array(list(pdos_dict[atom].keys()))
            for spd in spd_orbitals:
                index = key[np.isin(key, spd_orbitals[spd])]
                if len(index) > 0:
                    spd_dict[atom][spd] = np.sum([
                        pdos_dict[atom][i]['up'] for i in index
                    ], axis=0)

    if orbitals != [-1] and atoms == [-1]:
        spd_dict = {}
        key = np.array(list(pdos_dict.keys()))
        print(key)
        for spd in spd_orbitals:
            index = key[np.isin(key, spd_orbitals[spd])]
            print(spd)
            print(index)
            if len(index) > 0:
                print(spd)
                spd_dict[spd] = np.sum([
                    pdos_dict[i]['up'] for i in index
                ], axis=0)

    return spd_dict


def plot_orbital_atom_pdos(tdos_dict, orbital_atom_dict, sigma, ax, fill=True):

    colors = {'s': 'red', 'p': 'blue', 'd': 'green'}

    for atom in orbital_atom_dict:
        for orbital in orbital_atom_dict[atom]:
            dos = orbital_atom_dict[atom][orbital]
            energy = tdos_dict['energy']
            if sigma != 0:
                dos = smear(dos, energy, sigma)

            ax.plot(
                dos,
                energy,
                color=colors[orbital],
            )

            if fill:
                ax.fill_betweenx(
                    energy,
                    dos,
                    0,
                    color=colors[orbital],
                    linewidth=1.5,
                    alpha=0.2,
                )


def plot_orbital_pdos(tdos_dict, orbital_dict, sigma, ax, fill=True):

    colors = {'s': 'red', 'p': 'blue', 'd': 'green'}

    for orbital in orbital_dict:
        dos = orbital_dict[orbital][0]
        print(len(dos))
        energy = tdos_dict['energy']
        if sigma != 0:
            dos = smear(dos, energy, sigma)

        print(len(dos))
        print(len(energy))
        ax.plot(
            dos,
            energy,
            color=colors[orbital],
        )

        if fill:
            ax.fill_betweenx(
                energy,
                dos,
                0,
                color=colors[orbital],
                linewidth=1.5,
                alpha=0.2,
            )


def plot_atom_pdos(tdos_dict, atom_dict, sigma, ax, fill=True):

    colors = {0: 'red', 1: 'blue', 2: 'green'}

    for atom in atom_dict:
        dos = atom_dict[atom]['up']
        energy = tdos_dict['energy']
        if sigma != 0:
            dos = smear(dos, energy, sigma)

        ax.plot(
            dos,
            energy,
            color=colors[atom],
        )

        if fill:
            ax.fill_betweenx(
                energy,
                dos,
                0,
                color=colors[atom],
                linewidth=1.5,
                alpha=0.2,
            )


def plot_tdos(tdos_dict, sigma, ax, fill=True):
    dos = tdos_dict['up']
    energy = tdos_dict['energy']
    if sigma != 0:
        dos = smear(dos, energy, sigma)

    ax.plot(
        dos,
        energy,
        color='black',
        linewidth=1.5,
    )

    if fill:
        ax.fill_betweenx(
            energy,
            dos,
            0,
            color='black',
            linewidth=1.5,
            alpha=0.2,
        )


def main():
    vasprun_file = '../dos/vasprun.xml'
    vasprun = Vasprun(
        vasprun_file,
        parse_dos=True,
        parse_eigen=False,
        parse_potcar_file=False
    )
    pdos = vasprun.pdos
    tdos = vasprun.tdos

    # orbitals = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    orbitals = [-1]
    atoms = [0, 1]
    sigma = 0.1

    # orbital_atom_dict = get_orbilal_atom_pdos(
    # pdos=pdos,
    # orbitals=orbitals,
    # atoms=atoms
    # )

    # orbital_dict = get_orbilal_pdos(
    # pdos=pdos,
    # orbitals=orbitals
    # )

    atom_dict = get_atom_pdos(
        pdos,
        atoms
    )

    # [print(li) for li in atom_dict[0]['up'][0]]

    tdos_dict = get_tdos(tdos)

    # spd_dict = sum_spd(
    # pdos_dict=orbital_dict,
    # orbitals=orbitals,
    # atoms=atoms,
    # )
    # print(spd_dict.keys())

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_tdos(tdos_dict, sigma, ax)
    # plot_orbital_atom_pdos(tdos_dict, spd_dict, sigma, ax)
    # plot_orbital_pdos(tdos_dict, spd_dict, sigma, ax)
    plot_atom_pdos(tdos_dict, atom_dict, sigma, ax)
    plt.ylim(-6, 6)
    plt.xlim(xmin=0)
    plt.show()


if __name__ == "__main__":
    main()
