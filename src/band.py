from pymatgen.electronic_structure.core import Spin, Orbital
import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd


def get_bands(eigenvalues, efermi):

    bands_dict = {f'band{i+1}': [] for i in range(len(eigenvalues[0]))}

    for kpoint in eigenvalues:
        for (i, eigenvalue) in enumerate(kpoint):
            bands_dict[f'band{i+1}'].append(eigenvalue[0] - efermi)

    return bands_dict


def get_orbital_atom_weights(projected_eigenvalues, orbitals, atoms):

    orbital_atom_dict = {}

    for i in range(len(projected_eigenvalues[0])):
        orbital_atom_dict[f'band{i+1}'] = {
            atom: {orbital: [] for orbital in orbitals} for atom in atoms
        }

    for kpoint in projected_eigenvalues:
        for (j, projected_eigenvalue) in enumerate(kpoint):
            band = f'band{j+1}'
            for atom in atoms:
                for orbital in orbitals:
                    orbital_atom_dict[band][atom][orbital].append(
                        projected_eigenvalue[atom][orbital]
                    )
    
    return orbital_atom_dict


def get_orbital_weights(projected_eigenvalues, orbitals):

    orbitals_dict = {}

    for i in range(len(projected_eigenvalues[0])):
        orbitals_dict[f'band{i+1}'] = {orbital: [] for orbital in orbitals}

    for kpoint in projected_eigenvalues:
        for (j, projected_eigenvalue) in enumerate(kpoint):
            band = f'band{j+1}'
            atom_sums = np.sum(projected_eigenvalue, axis=0).tolist()
            for orbital in orbitals:
                orbitals_dict[band][orbital].append(
                    atom_sums[orbital]
                )

    return orbitals_dict


def get_atom_weights(projected_eigenvalues, atomic_symbol):

    atoms_dict = {}

    for i in range(len(projected_eigenvalues[0])):
        atoms_dict[f'band{i+1}'] = {atom: [] for atom in atomic_symbol}

    for kpoint in projected_eigenvalues:
        for (j, projected_eigenvalue) in enumerate(kpoint):
            band = f'band{j+1}'
            orbital_sums = np.sum(projected_eigenvalue, axis=1).tolist()
            for atom in atomic_symbol:
                atoms_dict[band][atom].append(
                    orbital_sums[atom]
                )

    return atoms_dict


def plot_bands(wave_vector, bands_dict, ax, color='black', linewidth=1.5, style='-', alpha=1):
    for band in bands_dict:
        ax.plot(
            wave_vector,
            bands_dict[band],
            color=color,
            linestyle=style,
            linewidth=linewidth,
            alpha=alpha,
            zorder=0,
        )


def plot_orbital_atom(wave_vector, bands_dict, orbital_atom_dict, ax, alpha=0.6, scale_factor=30):
    colors_dict = {
        0: 'red',
        1: 'blue',
        2: 'blue',
        3: 'blue',
        4: 'green',
        5: 'green',
        6: 'green',
        7: 'green',
        8: 'green',
    }

    df = pd.DataFrame.from_dict(orbital_atom_dict['band10'][0])
    df.to_csv('test.txt', sep='\t')

    for band in bands_dict:
        for atom in orbital_atom_dict[band]:
            for orbital in orbital_atom_dict[band][atom]:
                ax.scatter(
                    wave_vector,
                    bands_dict[band],
                    c=colors_dict[orbital],
                    s=scale_factor *
                    np.array(orbital_atom_dict[band][atom][orbital]),
                    zorder=1,
                    alpha=alpha,
                    edgecolors=None,
                )


def plot_orbital(wave_vector, bands_dict, orbitals_dict, ax, alpha=0.6, scale_factor=30):
    colors_dict = {
        0: 'red',
        1: 'blue',
        2: 'blue',
        3: 'blue',
        4: 'green',
        5: 'green',
        6: 'green',
        7: 'green',
        8: 'green',
    }
    
    print(len(list(bands_dict.keys())))
    for band in bands_dict:
        for orbital in orbitals_dict[band]:
            ax.scatter(
                wave_vector,
                bands_dict[band],
                c=colors_dict[orbital],
                s=scale_factor*np.array(orbitals_dict[band][orbital]),
                zorder=1,
                alpha=alpha,
                edgecolors=None,
            )


def plot_atom(wave_vector, bands_dict, atoms_dict, ax, alpha=0.6, scale_factor=30):
    for band in bands_dict:
        for atom in atoms_dict[band]:
            ax.scatter(
                wave_vector,
                bands_dict[band],
                c='blue',
                s=scale_factor*np.array(atoms_dict[band][atom]),
                zorder=1,
                alpha=alpha,
                edgecolors=None,
            )


def get_kpoints(kpoints, all_kpoints, ax):
    high_sym_points = kpoints.kpts
    kpts_labels = np.array([f'${k}$' for k in kpoints.labels])

    index = [0]
    for i in range(len(high_sym_points) - 2):
        if high_sym_points[i + 2] != high_sym_points[i + 1]:
            index.append(i)
    index.append(len(high_sym_points) - 1)

    kpts_loc = np.isin(all_kpoints, high_sym_points).all(1)
    kpoints_index = np.where(kpts_loc == True)[0]

    kpts_labels = kpts_labels[index]
    kpoints_index = list(kpoints_index[index])

    for i in range(len(kpoints_index)):
        if 0 < i < len(kpoints_index) - 1:
            kpoints_index[i] = kpoints_index[i] + 0.5

    for k in kpoints_index:
        ax.axvline(x=k, color='black', alpha=0.7, linewidth=0.75)

    plt.xticks(kpoints_index, kpts_labels)


def orbital_atom_band(vasprun, kpoints, orbitals, atoms, ax, spin=Spin.up):
    eigenvalues = vasprun.eigenvalues[spin]

    projected_eigenvalues = vasprun.projected_eigenvalues[spin]

    efermi = vasprun.efermi

    atomic_symbol = vasprun.atomic_symbols

    all_kpoints = np.array(vasprun.actual_kpoints)

    bands_dict = get_bands(eigenvalues, efermi)

    orbital_atom_dict = get_orbital_atom_weights(
        projected_eigenvalues,
        orbitals,
        atoms
    )

    plt.xlim(0, len(all_kpoints) - 1)
    wave_vector = range(len(all_kpoints))

    plot_bands(
        wave_vector=wave_vector,
        bands_dict=bands_dict,
        ax=ax,
        alpha=0.5,
        linewidth=1
    )

    plot_orbital_atom(
        wave_vector=wave_vector,
        bands_dict=bands_dict,
        orbital_atom_dict=orbital_atom_dict,
        ax=ax,
        alpha=1,
        scale_factor=5
    )

    get_kpoints(
        kpoints=kpoints,
        all_kpoints=all_kpoints,
        ax=ax,
    )


def orbital_band(vasprun, kpoints, orbitals, ax, spin=Spin.up):
    eigenvalues = vasprun.eigenvalues[spin]

    projected_eigenvalues = vasprun.projected_eigenvalues[spin]

    efermi = vasprun.efermi

    atomic_symbol = vasprun.atomic_symbols

    all_kpoints = np.array(vasprun.actual_kpoints)

    bands_dict = get_bands(eigenvalues, efermi)

    orbitals_dict = get_orbital_weights(projected_eigenvalues, orbitals)

    plt.xlim(0, len(all_kpoints) - 1)
    wave_vector = range(len(all_kpoints))

    plot_bands(
        wave_vector=wave_vector,
        bands_dict=bands_dict,
        ax=ax,
        alpha=0.5,
        linewidth=1
    )

    plot_orbital(
        wave_vector=wave_vector,
        bands_dict=bands_dict,
        orbitals_dict=orbitals_dict,
        ax=ax,
        alpha=1,
        scale_factor=5
    )

    get_kpoints(
        kpoints=kpoints,
        all_kpoints=all_kpoints,
        ax=ax,
    )


def atom_band(vasprun, kpoints, atoms, ax, spin=Spin.up):
    eigenvalues = vasprun.eigenvalues[spin]

    projected_eigenvalues = vasprun.projected_eigenvalues[spin]

    efermi = vasprun.efermi

    atomic_symbol = vasprun.atomic_symbols

    all_kpoints = np.array(vasprun.actual_kpoints)

    bands_dict = get_bands(eigenvalues, efermi)

    atoms_dict = get_atom_weights(projected_eigenvalues, atoms)

    plt.xlim(0, len(all_kpoints) - 1)
    wave_vector = range(len(all_kpoints))

    plot_bands(
        wave_vector=wave_vector,
        bands_dict=bands_dict,
        ax=ax,
        alpha=0.5,
        linewidth=1
    )

    plot_atom(
        wave_vector=wave_vector,
        bands_dict=bands_dict,
        atoms_dict=atoms_dict,
        ax=ax,
        alpha=1,
        scale_factor=5
    )

    get_kpoints(kpoints, all_kpoints, ax)


def plain_band(vasprun, kpoints, ax, spin=Spin.up):
    eigenvalues = vasprun.eigenvalues[spin]

    efermi = vasprun.efermi

    atomic_symbol = vasprun.atomic_symbols

    all_kpoints = np.array(vasprun.actual_kpoints)

    bands_dict = get_bands(eigenvalues, efermi)

    plt.xlim(0, len(all_kpoints) - 1)
    wave_vector = range(len(all_kpoints))

    plot_bands(
        wave_vector=wave_vector,
        bands_dict=bands_dict,
        ax=ax,
    )

    get_kpoints(
        kpoints=kpoints,
        all_kpoints=all_kpoints,
        ax=ax,
    )
