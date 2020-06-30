from pymatgen.electronic_structure.core import Spin, Orbital
from pymatgen.io.vasp.outputs import BSVasprun
from pymatgen.io.vasp.inputs import Kpoints, Poscar
from functools import reduce
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time


class BandStructure:
    def __init__(self, folder, projected=False, hse=False, spin='up'):
        self.vasprun = BSVasprun(
            f'{folder}/vasprun.xml',
            parse_projected_eigen=projected
        )
        self.projected = projected
        self.hse = hse
        self.folder = folder
        self.spin = 'up'
        self.bands_dict = self.load_bands()

        if projected:
            self.projected_dict = self.load_projected_bands_matrix()

        if not hse:
            self.kpoints = Kpoints.from_file(f'{folder}/KPOINTS')

    def load_bands(self):
        eigenvalues = self.vasprun.eigenvalues
        efermi = self.vasprun.efermi
        spin = self.spin
        nkpoints = len(eigenvalues[Spin.up])
        nbands = len(eigenvalues[Spin.up][0])
        spin_dict = {'up': Spin.up, 'down': Spin.down}

        bands_dict = {f'band{i+1}': [] for i in range(nbands)}

        for i in range(nkpoints):
            for j in range(nbands):
                bands_dict[f'band{j+1}'].append(
                    eigenvalues[spin_dict[spin]][i][j][0] - efermi
                )

        return bands_dict

    def load_projected_bands(self):
        assert_message = "Projected eigenvalues not loaded pass projected=True to load them."
        assert self.projected is True, assert_message

        projected_eigenvalues = self.vasprun.projected_eigenvalues
        poscar = Poscar.from_file(
            f'{self.folder}/POSCAR',
            check_for_POTCAR=False,
            read_velocities=False
        )
        natoms = len(poscar.site_symbols)
        nkpoints = len(projected_eigenvalues[Spin.up])
        nbands = len(projected_eigenvalues[Spin.up][0])

        projected_dict = {}

        for i in range(nbands):
            projected_dict[f'band{i+1}'] = \
                {atom: {orbital: {}
                        for orbital in range(9)} for atom in range(natoms)}

        for i in range(nkpoints):
            for j in range(nbands):
                band = f'band{j+1}'
                for atom in range(natoms):
                    for orbital in range(9):
                        projected_dict[band][atom][orbital][i] = \
                            projected_eigenvalues[Spin.up][i][j][atom][orbital]

        return projected_dict

    def load_projected_bands_matrix(self):
        projected_eigenvalues = self.vasprun.projected_eigenvalues
        poscar = Poscar.from_file(
            f'{self.folder}/POSCAR',
            check_for_POTCAR=False,
            read_velocities=False
        )
        natoms = len(poscar.site_symbols)
        nkpoints = len(projected_eigenvalues[Spin.up])
        nbands = len(projected_eigenvalues[Spin.up][0])

        projected_dict = {f'band{i+1}':
                          {atom: np.zeros(9) for atom in range(natoms)}
                          for i in range(nbands)}

        for i in range(nkpoints):
            for j in range(nbands):
                band = f'band{j+1}'
                for atom in range(natoms):
                    orbital_weights = projected_eigenvalues[Spin.up][i][j][atom]
                    projected_dict[band][atom] = np.vstack([
                        projected_dict[band][atom],
                        orbital_weights
                    ])

        for band in projected_dict:
            for atom in projected_dict[band]:
                projected_dict[band][atom] = pd.DataFrame(
                    projected_dict[band][atom][1:]
                )

        return projected_dict

    def sum_spd(self):
        spd_orbitals = {'s': [0], 'p': [1, 2, 3], 'd': [4, 5, 6, 7, 8]}

        spd_dict = {band: np.nan for band in self.projected_dict}

        for band in self.projected_dict:
            atom_list = [
                self.projected_dict[band][atom] for atom in self.projected_dict[band]]
            spd_dict[band] = reduce(
                lambda x, y: x.add(y, fill_value=0), atom_list
            )

        for band in spd_dict:
            df = spd_dict[band]
            spd_dict[band]['s'] = df[0]
            spd_dict[band]['p'] = df[1] + df[2] + df[3]
            spd_dict[band]['d'] = df[4] + df[5] + df[6] + df[7] + df[8]
            spd_dict[band] = spd_dict[band].drop(columns=range(9))

        return spd_dict

    def sum_orbital(self, orbitals):

        orbital_dict = {band: np.nan for band in self.projected_dict}

        for band in self.projected_dict:
            atom_list = [
                self.projected_dict[band][atom] for atom in self.projected_dict[band]]
            orbital_dict[band] = reduce(
                lambda x, y: x.add(y, fill_value=0), atom_list
            )

        for band in orbital_dict:
            df = orbital_dict[band]
            for col in df.columns.tolist():
                if sum(np.isin(orbitals, col)) == 0:
                    orbital_dict[band] = orbital_dict[band].drop(columns=col)

        return orbital_dict

    def get_kticks(self, ax):
        high_sym_points = self.kpoints.kpts
        kpts_labels = np.array([f'${k}$' for k in self.kpoints.labels])
        all_kpoints = self.vasprun.actual_kpoints

        # index = []
        # for i in range(len(high_sym_points)):
        # if high_sym_points[i] != high_sym_points[i - 1]:
        # index.append(i)

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
            ax.axvline(x=k, color='black', alpha=0.7, linewidth=0.5)

        plt.xticks(kpoints_index, kpts_labels)

    def plot_plain(self, ax, color='black', linewidth=1.5):
        wave_vector = range(len(self.bands_dict['band1']))

        for band in self.bands_dict:
            band_values = self.bands_dict[band]
            ax.plot(
                wave_vector,
                band_values,
                color=color,
                linewidth=linewidth,
                zorder=0,
            )

        self.get_kticks(ax=ax)
        plt.xlim(0, len(wave_vector)-1)

    def plot_spd(self, ax, scale_factor=5, order=['s', 'p', 'd']):
        spd_dict = self.sum_spd()
        color_dict = {'s': 'red', 'p': 'blue', 'd': 'green'}

        self.plot_plain(ax, linewidth=0.5)

        plot_df = pd.DataFrame(columns=['s', 'p', 'd'])
        plot_band = []
        plot_wave_vec = []

        for band in spd_dict:
            plot_df = plot_df.append(spd_dict[band])
            plot_band.extend(self.bands_dict[band])
            plot_wave_vec.extend(range(len(spd_dict[band])))

        for col in order:
            ax.scatter(
                plot_wave_vec,
                plot_band,
                c=color_dict[col],
                s=scale_factor * plot_df[col],
                zorder=1,
            )

        pass

    def plot_atom_orbitals(self, atom_orbital_pairs, ax, scale_factor=5, colors=None):
        self.plot_plain(ax=ax, linewidth=0.75)
        self.get_kticks(ax=ax)

        projected_dict = self.projected_dict
        wave_vector = range(len(self.bands_dict['band1']))

        if colors is None:
            colors = ['red', 'blue', 'green', 'orange', 'purple']

        for band in projected_dict:
            for i, atom_orbital_pair in enumerate(atom_orbital_pairs):
                atom = atom_orbital_pair[0]
                orbital = atom_orbital_pair[1]

                ax.scatter(
                    wave_vector,
                    self.bands_dict[band],
                    c=colors[i],
                    s=scale_factor * projected_dict[band][atom][orbital]
                )

        pass

    def compare_orbitals(self, orbitals, ax, scale_factor=5, color_dict=None):
        self.plot_plain(ax=ax, linewidth=0.75)
        self.get_kticks(ax=ax)

        orbital_dict = self.sum_orbital(orbitals=orbitals)

        if color_dict is None:
            color_dict = {
                0: 'red',
                1: 'green',
                2: 'blue',
                3: 'orange',
                4: 'purple',
                5: 'gold',
                6: 'mediumturquoise',
                7: 'navy',
                8: 'springgreen',
            }

        plot_df = pd.DataFrame(columns=['s', 'p', 'd'])
        plot_band = []
        plot_wave_vec = []

        for band in orbital_dict:
            plot_df = plot_df.append(orbital_dict[band])
            plot_band.extend(self.bands_dict[band])
            plot_wave_vec.extend(range(len(orbital_dict[band])))

        for orbital in orbitals:
            ax.scatter(
                plot_wave_vec,
                plot_band,
                c=color_dict[orbital],
                s=scale_factor * plot_df[orbital],
                zorder=1,
            )

        pass


def main():
    bands = BandStructure(folder='../../vaspvis_data/band',
                          projected=True, spin='up')
    fig = plt.figure(figsize=(4, 3), dpi=300)
    ax = fig.add_subplot(111)
    # bands.plot_spd(ax=ax, order=['d', 'p', 's'])
    # bands.plot_plain(ax=ax)
    # bands.plot_atom_orbitals(ax=ax, atom_orbital_pairs=[[0, 0], [1, 3]])
    # bands.compare_orbitals(ax=ax, orbitals=[0, 1, 2, 3, 4, 5, 6, 7, 8])
    plt.ylim(-6, 6)
    plt.ylabel('$E - E_F$ $(eV)$', fontsize=6)
    plt.tick_params(labelsize=6, length=1.5)
    plt.tick_params(axis='x', length=0)
    plt.tight_layout(pad=0.5)
    plt.show()


if __name__ == "__main__":
    main()
