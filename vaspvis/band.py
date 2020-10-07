from pymatgen.electronic_structure.core import Spin, Orbital
from pymatgen.io.vasp.outputs import BSVasprun
from pymatgen.io.vasp.inputs import Kpoints, Poscar
from pymatgen.core.periodic_table import Element
from functools import reduce
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from copy import deepcopy
import os


class Band:
    """
    This class contains all the methods for constructing band structures
    from the outputs of VASP band structure calculations.

    Parameters:
        folder (str): This is the folder that contains the VASP files
        projected (bool): Determined wheter of not to parte the projected
            eigenvalues from the vasprun.xml file. Making this true
            increases the computational time, so only use if a projected
            band structure is required.
        hse (bool): Determines if the KPOINTS file is in the form of HSE
            or not. Only make true if the band structure was calculated
            using a hybrid functional.
        spin (str): Choose which spin direction to parse. ('up' or 'down')
        kpath (str): High symmetry k-point path of band structure calculation
            Due to the nature of the KPOINTS file for HSE calculations this
            information is a required input for proper labeling of the figure
            for HSE calculations. This information is extracted from the KPOINTS
            files for non-HSE calculations. (G is automaticall converted to \\Gamma)
        n (int): Number of points between each high symmetry points.
            This is also only required for HSE calculations. This number should be 
            known by the user, as it was used to generate the KPOINTS file.
    """

    def __init__(self, folder, projected=False, hse=False, spin='up', kpath=None, n=None):
        """
        Initialize parameters upon the generation of this class

        Parameters:
            folder (str): This is the folder that contains the VASP files
            projected (bool): Determined wheter of not to parte the projected
                eigenvalues from the vasprun.xml file. Making this true
                increases the computational time, so only use if a projected
                band structure is required.
            hse (bool): Determines if the KPOINTS file is in the form of HSE
                or not. Only make true if the band structure was calculated
                using a hybrid functional.
            spin (str): Choose which spin direction to parse. ('up' or 'down')
        """

        self.vasprun = BSVasprun(
            os.path.join(folder, 'vasprun.xml'),
            parse_projected_eigen=projected
        )
        self.poscar = Poscar.from_file(
            os.path.join(folder, 'POSCAR'),
            check_for_POTCAR=False,
            read_velocities=False
        )
        self.projected = projected
        self.forbitals = False
        self.hse = hse
        self.kpath = kpath
        self.n = n
        self.folder = folder
        self.spin = spin
        self.spin_dict = {'up': Spin.up, 'down': Spin.down}
        self.bands_dict = self._load_bands()
        self.color_dict = {
            0: '#FF0000',
            1: '#0000FF',
            2: '#008000',
            3: '#800080',
            4: '#E09200',
            5: '#FF5C77',
            6: '#778392',
            7: '#07C589',
            8: '#40BAF2',
        }
        self.orbital_labels = {
            0: '$s$',
            1: '$p_{y}$',
            2: '$p_{x}$',
            3: '$p_{z}$',
            4: '$d_{xy}$',
            5: '$d_{yz}$',
            6: '$d_{z^{2}}$',
            7: '$d_{xz}$',
            8: '$d_{x^{2}-y^{2}}$',
            9: '$f_{y^{3}x^{2}}$',
            10: '$f_{xyz}$',
            11: '$f_{yz^{2}}$',
            12: '$f_{z^{3}}$',
            13: '$f_{xz^{2}}$',
            14: '$f_{zx^{3}}$',
            15: '$f_{x^{3}}$',
        }

        if projected:
            self.projected_dict = self._load_projected_bands()

        if not hse:
            self.kpoints = Kpoints.from_file(os.path.join(folder, 'KPOINTS'))

    def _load_bands(self):
        """
        This function is used to load eigenvalues from the vasprun.xml
        file and into a dictionary which is in the form of
        band index --> eigenvalues

        Returns:
            bands_dict (dict[str][np.ndarray]): Dictionary which contains
                the eigenvalues for each band
        """

        spin = self.spin

        if self.hse:
            kpoints_band = self.n * (len(self.kpath) - 1)
            eigenvalues = self.vasprun.eigenvalues[
                self.spin_dict[spin]
            ][-1 * kpoints_band:]
        else:
            eigenvalues = self.vasprun.eigenvalues[self.spin_dict[spin]]

        efermi = self.vasprun.efermi
        nkpoints = len(eigenvalues)
        nbands = len(eigenvalues[0])

        bands_dict = {f'band{i+1}': [] for i in range(nbands)}

        for i in range(nkpoints):
            for j in range(nbands):
                bands_dict[f'band{j+1}'].append(
                    eigenvalues[i][j][0] - efermi
                )

        return bands_dict

    def _load_projected_bands(self):
        """
        This function loads the project weights of the orbitals in each band
        from vasprun.xml into a dictionary of the form:
        band index --> atom index --> weights of orbitals

        Returns:
            projected_dict (dict([str][int][pd.DataFrame])): Dictionary containing the projected weights of all orbitals on each atom for each band.
        """

        spin = self.spin

        if self.hse:
            kpoints_band = self.n * (len(self.kpath) - 1)
            projected_eigenvalues = self.vasprun.projected_eigenvalues[
                self.spin_dict[spin]
            ][-1 * kpoints_band:]
        else:
            projected_eigenvalues = self.vasprun.projected_eigenvalues[
                self.spin_dict[spin]
            ]

        poscar = Poscar.from_file(
            f'{self.folder}/POSCAR',
            check_for_POTCAR=False,
            read_velocities=False
        )
        spin = self.spin
        natoms = np.sum(poscar.natoms)
        nkpoints = len(projected_eigenvalues)
        nbands = len(projected_eigenvalues[0])
        norbitals = len(projected_eigenvalues[0][0][0])

        if norbitals == 16:
            self.forbitals = True

        projected_dict = {f'band{i+1}':
                          {atom: np.zeros(norbitals) for atom in range(natoms)}
                          for i in range(nbands)}

        for i in range(nkpoints):
            for j in range(nbands):
                band = f'band{j+1}'
                for atom in range(natoms):
                    orbital_weights = projected_eigenvalues[i][j][atom] ** 2
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

    def _sum_spd(self):
        """
        This function sums the weights of the s, p, and d orbitals for each atom
        and creates a dictionary of the form:
        band index --> s,p,d orbital weights

        Returns:
            spd_dict (dict([str][pd.DataFrame])): Dictionary that contains the summed weights for the s, p, and d orbitals for each band
        """

        # spd_orbitals = {'s': [0], 'p': [1, 2, 3], 'd': [4, 5, 6, 7, 8]}

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

            if self.forbitals:
                spd_dict[band]['f'] = df[9] + df[10] + \
                    df[11] + df[12] + df[13] + df[14] + df[15]
                spd_dict[band] = spd_dict[band].drop(columns=range(16))
            else:
                spd_dict[band] = spd_dict[band].drop(columns=range(9))

        return spd_dict

    def _sum_orbitals(self, orbitals):
        """
        This function finds the weights of desired orbitals for all atoms and
            returns a dictionary of the form:
            band index --> orbital index

        Parameters:
            orbitals (list): List of desired orbitals. 
                0 = s
                1 = py
                2 = pz
                3 = px
                4 = dxy
                5 = dyz
                6 = dz2
                7 = dxz
                8 = dx2-y2
                9 = fy3x2
                10 = fxyz
                11 = fyz2
                12 = fz3
                13 = fxz2
                14 = fzx3
                15 = fx3

        Returns:
            orbital_dict (dict[str][pd.DataFrame]): Dictionary that contains the projected weights of the selected orbitals.
        """

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

    def _sum_atoms(self, atoms):
        """
        This function finds the weights of desired atoms for all orbitals and
            returns a dictionary of the form:
            band index --> atom index

        Parameters:
            atoms (list): List of desired atoms where atom 0 is the first atom in
                the POSCAR file. 

        Returns:
            atom_dict (dict[str][pd.DataFrame]): Dictionary that contains the projected
                weights of the selected atoms.
        """

        projected_dict = self.projected_dict
        atoms_dict = {band: np.nan for band in projected_dict}

        for band in projected_dict:
            atom_dict = {atom: projected_dict[band][atom].sum(
                axis=1) for atom in atoms}
            atoms_dict[band] = pd.DataFrame.from_dict(atom_dict)

        return atoms_dict

    def _sum_elements(self, elements, orbitals=False, spd=False):
        """
        This function sums the weights of the orbitals of specific elements within the
        calculated structure and returns a dictionary of the form:
        band index --> element label --> orbital weights for orbitals = True
        band index --> element label for orbitals = False
        This is useful for structures with many elements because manually entering indicies is
        not practical for large structures.

        Parameters:
            elements (list): List of element symbols to sum the weights of.
            orbitals (bool): Determines whether or not to inclue orbitals or not
                (True = keep orbitals, False = sum orbitals together )
            spd (bool): Determines whether or not to sum the s, p, and d orbitals


        Returns:
            element_dict (dict([str][str][pd.DataFrame])): Dictionary that contains the summed weights for each orbital for a given element in the structure.
        """

        poscar = self.poscar
        natoms = poscar.natoms
        symbols = poscar.site_symbols
        projected_dict = self.projected_dict

        element_list = np.hstack(
            [[symbols[i] for j in range(natoms[i])]
             for i in range(len(symbols))]
        )

        element_dict = {
            band: {element: [] for element in elements} for band in projected_dict
        }

        for band in projected_dict:
            band_df = pd.DataFrame()
            for element in elements:
                element_index = np.where(np.isin(element_list, element))[0]
                nb_atoms = len(element_index)
                df = pd.concat(
                    [projected_dict[band][i] for i in element_index],
                    axis=1
                )

                if orbitals:
                    element_dict[band][element] = df.groupby(
                        by=df.columns,
                        axis=1
                    ).sum()
                    if spd:
                        df = element_dict[band][element]
                        element_dict[band][element]['s'] = df[0]
                        element_dict[band][element]['p'] = df[1] + \
                            df[2] + df[3]
                        element_dict[band][element]['d'] = df[4] + \
                            df[5] + df[6] + df[7] + df[8]

                        if self.forbitals:
                            element_dict[band][element]['f'] = df[9] + df[10] + \
                                df[11] + df[12] + df[13] + df[14] + df[15]
                            element_dict[band][element] = element_dict[band][element].drop(
                                columns=range(16))
                        else:
                            element_dict[band][element] = element_dict[band][element].drop(
                                columns=range(9))
                else:
                    norm_df = df.sum(axis=1)
                    element_dict[band][element] = norm_df.tolist()

        return element_dict

    @staticmethod
    def _calculate_distance(point1, point2):
        """
        This function calculates the distance between two points

        Parameters:
            point1 (list): First point
            point2 (list): Second point

        Returns:
            Absolute value of the distance
        """
        x_diff = (point2[0] - point1[0]) ** 2
        y_diff = (point2[1] - point1[1]) ** 2
        z_diff = (point2[2] - point1[2]) ** 2
        dist = np.abs(np.sqrt(x_diff + y_diff + z_diff))

        return dist

    def _get_k_distance_fraction(self):
        """
        This function find the relative distances in reciprocal space between the high
        symmetry k-points

        Parameters:
            high_sym_points (list): List of high symmetry kpoints

        Returns:
            Fractional distances of each kpath segment
        """
        if not self.hse:
            raw_high_sym_points = self.kpoints.kpts
            index = [0]
            for i in range(len(raw_high_sym_points) - 2):
                if raw_high_sym_points[i + 2] != raw_high_sym_points[i + 1]:
                    index.append(i)
            index.append(len(raw_high_sym_points) - 1)
            high_sym_points = [raw_high_sym_points[i] for i in index]
        else:
            with open(os.path.join(self.folder, 'KPOINTS')) as kpoints_file:
                kpoints = kpoints_file.read()
                kpoints_split = kpoints.split('\n')
                kpoints_split = kpoints_split[::-1]

            kpoints_split = [list(np.array(k.split()[:3], dtype=float)) for k in kpoints_split[1:int((len(self.kpath) -1) * self.n)+1]]
            kpoints_split = kpoints_split[::-1]

            kpoints_index = [(i*self.n) - 1 for i in range(len(self.kpath)) if 0 < i < len(self.kpath)-1]
            kpoints_index.append(self.n*(len(self.kpath) - 1)-1)
            kpoints_index.insert(0, 0)
            high_sym_points = np.array(kpoints_split)[kpoints_index]
            

        distances = [
            self._calculate_distance(high_sym_points[i], high_sym_points[i+1]) for i in range(len(high_sym_points)-1)
        ]
        fractions = [distance / np.sum(distances) for distance in distances]

        return fractions


    def _get_kticks(self, ax):
        """
        This function extracts the kpoint labels and index locations for a regular
        band structure calculation (non HSE).

        Parameters:
            ax (matplotlib.pyplot.axis): Axis to append the tick labels
        """

        high_sym_points = self.kpoints.kpts
        kpts_labels = np.array([f'${k}$' if k != 'G' else '$\\Gamma$' for k in self.kpoints.labels])
        all_kpoints = self.vasprun.actual_kpoints

        index = [0]
        for i in range(len(high_sym_points) - 2):
            if high_sym_points[i + 2] != high_sym_points[i + 1]:
                index.append(i)
        index.append(len(high_sym_points) - 1)

        kpts_loc = np.isin(all_kpoints, high_sym_points).all(1)
        kpoints_index = np.where(kpts_loc == True)[0]

        kpts_labels = kpts_labels[index]
        kpoints_index = list(kpoints_index[index])
        kpoints_index = ax.lines[0].get_xdata()[kpoints_index]

        for k in kpoints_index:
            ax.axvline(x=k, color='black', alpha=0.7, linewidth=0.5)
        
        ax.set_xticks(kpoints_index)
        ax.set_xticklabels(kpts_labels)

    def _get_kticks_hse(self, ax, kpath, n):

        kpoints_index = [(i*n) - 1 for i in range(len(kpath))
                         if 0 < i < len(kpath)-1]
        kpoints_index.append(n*(len(kpath) - 1)-1)
        kpoints_index.insert(0, 0)
        kpoints_index = ax.lines[0].get_xdata()[kpoints_index]

        kpath = [f'${k}$' if k != 'G' else '$\\Gamma$' for k in kpath.upper().strip()]

        for k in kpoints_index:
            ax.axvline(x=k, color='black', alpha=0.7, linewidth=0.5)

        plt.xticks(kpoints_index, kpath)

    def plot_plain(self, ax, color='black', linewidth=1.25, linestyle='-'):
        """
        This function plots a plain band structure.

        Parameters:
            ax (matplotlib.pyplot.axis): Axis to plot the data on
            color (str): Color of the band structure lines
            linewidth (float): Line width of the band structure lines
            linestyle (str): Line style of the bands
        """

        #  wave_vector = range(len(self.bands_dict['band1']))
        nb_kpoints = len(self.bands_dict['band1'])
        fractions = self._get_k_distance_fraction()
        spacing = int(nb_kpoints / len(fractions))

        wave_vector = np.array([])
        for i in range(len(fractions)):
            segment = np.linspace(np.sum(fractions[:i]) * nb_kpoints, np.sum(fractions[:i+1]) * nb_kpoints, spacing)
            wave_vector = np.append(wave_vector, segment)
        
        self.wave_vector = wave_vector


        for band in self.bands_dict:
            band_values = self.bands_dict[band]
            ax.plot(
                wave_vector,
                band_values,
                color=color,
                linewidth=linewidth,
                linestyle=linestyle,
                zorder=0,
            )

        if self.hse:
            self._get_kticks_hse(ax=ax, kpath=self.kpath, n=self.n)
        else:
            self._get_kticks(ax=ax)

        ax.set_xlim(0, np.max(wave_vector))

    def plot_spd(self, ax, scale_factor=5, order=['s', 'p', 'd'], color_dict=None, legend=True, linewidth=0.75, band_color='black'):
        """
        This function plots the s, p, d projected band structure onto a given axis

        Parameters:
            ax (matplotlib.pyplot.axis): Axis to plot the data on
            scale_factor (float): Factor to scale weights. This changes the size of the
                points in the scatter plot
            order (list): This determines the order in which the points are plotted on the
                graph. This is an option because sometimes certain orbitals can be hidden
                under others because they have a larger weight. For example, if the
                weights of the d orbitals are greater than that of the s orbitals, it
                might be smart to choose ['d', 'p', 's'] as the order so the s orbitals are
                plotted over the d orbitals.
            color_dict (dict[str][str]): This option allow the colors of the s, p, and d
                orbitals to be specified. Should be in the form of:
                {'s': <s color>, 'p': <p color>, 'd': <d color>}
            legend (bool): Determines if the legend should be included or not.
            linewidth (float): Line width of the plain band structure plotted in the background
            band_color (string): Color of the plain band structure
        """
        scale_factor = scale_factor ** 1.5

        self.plot_plain(ax=ax, linewidth=linewidth, color=band_color)

        spd_dict = self._sum_spd()

        if color_dict is None:
            color_dict = {
                's': self.color_dict[0],
                'p': self.color_dict[1],
                'd': self.color_dict[2],
                'f': self.color_dict[4],
            }

        plot_df = pd.DataFrame()

        if self.forbitals and 'f' not in order:
            order.append('f')

        plot_band = []
        plot_wave_vec = []

        for band in spd_dict:
            plot_df = plot_df.append(spd_dict[band])
            plot_band.extend(self.bands_dict[band])
            plot_wave_vec.extend(self.wave_vector)

        for col in order:
            ax.scatter(
                plot_wave_vec,
                plot_band,
                c=color_dict[col],
                s=scale_factor * plot_df[col],
                zorder=1,
            )

        if legend:
            legend_lines = []
            legend_labels = []
            for orbital in order:
                legend_lines.append(plt.Line2D(
                    [0],
                    [0],
                    marker='o',
                    markersize=2,
                    linestyle='',
                    color=color_dict[orbital]
                ))
                legend_labels.append(
                    f'${orbital}$'
                )

            leg = ax.get_legend()

            if leg is None:
                handles = legend_lines
                labels = legend_labels
            else:
                handles = [l._legmarker for l in leg.legendHandles]
                labels = [text._text for text in leg.texts]
                handles.extend(legend_lines)
                labels.extend(legend_labels)

            ax.legend(
                handles,
                labels,
                ncol=1,
                loc='upper left',
                fontsize=5,
                bbox_to_anchor=(1, 1),
                borderaxespad=0,
                frameon=False,
                handletextpad=0.1,
            )

    def plot_atom_orbitals(self, ax, atom_orbital_pairs, scale_factor=5, color_list=None, legend=True, linewidth=0.75, band_color='black'):
        """
        This function plots the projected band structure of individual orbitals on a given axis.

        Parameters:
            ax (matplotlib.pyplot.axis): Axis to plot the data on
            atom_orbital_pairs (list[list]): Selected orbitals on selected atoms to plot.
                This should take the form of [[atom index, orbital_index], ...]. 
                To plot the px orbital of the 1st atom and the pz orbital of the 2nd atom
                in the POSCAR file, the input would be [[0, 3], [1, 2]]
            scale_factor (float): Factor to scale weights. This changes the size of the
                points in the scatter plot
            color_list (list): List of colors of the same length as the atom_orbital_pairs
            legend (bool): Determines if the legend should be included or not.
            linewidth (float): Line width of the plain band structure plotted in the background
            band_color (string): Color of the plain band structure
        """
        scale_factor = scale_factor ** 1.5

        self.plot_plain(ax=ax, linewidth=linewidth, color=band_color)

        projected_dict = self.projected_dict
        wave_vector = self.wave_vector

        if color_list is None:
            color_dict = self.color_dict
        else:
            color_dict = {i: color for i, color in enumerate(color_list)}

        for band in projected_dict:
            for (i, atom_orbital_pair) in enumerate(atom_orbital_pairs):
                atom = atom_orbital_pair[0]
                orbital = atom_orbital_pair[1]

                ax.scatter(
                    wave_vector,
                    self.bands_dict[band],
                    c=color_dict[i],
                    s=scale_factor * projected_dict[band][atom][orbital],
                    zorder=1,
                )

        if legend:
            legend_lines = []
            legend_labels = []
            for (i, atom_orbital_pair) in enumerate(atom_orbital_pairs):
                atom = atom_orbital_pair[0]
                orbital = atom_orbital_pair[1]

                legend_lines.append(plt.Line2D(
                    [0],
                    [0],
                    marker='o',
                    markersize=2,
                    linestyle='',
                    color=color_dict[i])
                )
                legend_labels.append(
                    f'{atom}({self.orbital_labels[atom_orbital_pair[1]]})'
                )

            ax.legend(
                legend_lines,
                legend_labels,
                ncol=1,
                loc='upper left',
                fontsize=5,
                bbox_to_anchor=(1, 1),
                borderaxespad=0,
                frameon=False,
                handletextpad=0.1,
            )

    def plot_orbitals(self, ax, orbitals, scale_factor=5, color_dict=None, legend=True, linewidth=0.75, band_color='black'):
        """
        This function plots the projected band structure of given orbitals summed across all atoms on a given axis.

        Parameters:
            ax (matplotlib.pyplot.axis): Axis to plot the data on
            orbitals (list): List of orbits to compare

                | 0 = s
                | 1 = py
                | 2 = pz
                | 3 = px
                | 4 = dxy
                | 5 = dyz
                | 6 = dz2
                | 7 = dxz
                | 8 = dx2-y2
                | 9 = fy3x2
                | 10 = fxyz
                | 11 = fyz2
                | 12 = fz3
                | 13 = fxz2
                | 14 = fzx3
                | 15 = fx3

            scale_factor (float): Factor to scale weights. This changes the size of the
                points in the scatter plot
            color_dict (dict[str][str]): This option allow the colors of each orbital
                specified. Should be in the form of:
                {'orbital index': <color>, 'orbital index': <color>, ...}
            legend (bool): Determines if the legend should be included or not.
            linewidth (float): Line width of the plain band structure plotted in the background
            band_color (string): Color of the plain band structure
        """
        scale_factor = scale_factor ** 1.5

        self.plot_plain(ax=ax, linewidth=linewidth, color=band_color)

        orbital_dict = self._sum_orbitals(orbitals=orbitals)

        if color_dict is None:
            color_dict = self.color_dict

        plot_df = pd.DataFrame(columns=orbitals)
        plot_band = []
        plot_wave_vec = []

        for band in orbital_dict:
            plot_df = plot_df.append(orbital_dict[band])
            plot_band.extend(self.bands_dict[band])
            plot_wave_vec.extend(self.wave_vector)

        for orbital in orbitals:
            ax.scatter(
                plot_wave_vec,
                plot_band,
                c=color_dict[orbital],
                s=scale_factor * plot_df[orbital],
                zorder=1,
            )

        if legend:
            legend_lines = []
            legend_labels = []
            for orbital in orbitals:
                legend_lines.append(plt.Line2D(
                    [0],
                    [0],
                    marker='o',
                    markersize=2,
                    linestyle='',
                    color=color_dict[orbital])
                )
                legend_labels.append(
                    f'{self.orbital_labels[orbital]}'
                )

            leg = ax.get_legend()

            if leg is None:
                handles = legend_lines
                labels = legend_labels
            else:
                handles = [l._legmarker for l in leg.legendHandles]
                labels = [text._text for text in leg.texts]
                handles.extend(legend_lines)
                labels.extend(legend_labels)

            ax.legend(
                handles,
                labels,
                ncol=1,
                loc='upper left',
                fontsize=5,
                bbox_to_anchor=(1, 1),
                borderaxespad=0,
                frameon=False,
                handletextpad=0.1,
            )

    def plot_atoms(self, ax, atoms, scale_factor=5, color_list=None, legend=True, linewidth=0.75, band_color='black'):
        """
        This function plots the projected band structure of given atoms summed across all orbitals on a given axis.

        Parameters:
            ax (matplotlib.pyplot.axis): Axis to plot the data on
            atoms (list): List of atoms to project onto
            scale_factor (float): Factor to scale weights. This changes the size of the
                points in the scatter plot
            color_list (list): List of colors of the same length as the atoms list
            legend (bool): Determines if the legend should be included or not.
            linewidth (float): Line width of the plain band structure plotted in the background
            band_color (string): Color of the plain band structure
        """
        scale_factor = scale_factor ** 1.5

        self.plot_plain(ax=ax, linewidth=linewidth, color=band_color)

        atom_dict = self._sum_atoms(atoms=atoms)

        if color_list is None:
            color_dict = self.color_dict
        else:
            color_dict = {i: color for i, color in enumerate(color_list)}

        plot_df = pd.DataFrame(columns=atoms)
        plot_band = []
        plot_wave_vec = []

        for band in atom_dict:
            plot_df = plot_df.append(atom_dict[band], ignore_index=True)
            plot_band.extend(self.bands_dict[band])
            plot_wave_vec.extend(self.wave_vector)

        for (i, atom) in enumerate(atoms):
            ax.scatter(
                plot_wave_vec,
                plot_band,
                c=color_dict[i],
                s=scale_factor * plot_df[atom],
                zorder=1,
            )

        if legend:
            legend_lines = []
            legend_labels = []
            for (i, atom) in enumerate(atoms):
                legend_lines.append(plt.Line2D(
                    [0],
                    [0],
                    marker='o',
                    markersize=2,
                    linestyle='',
                    color=color_dict[i])
                )
                legend_labels.append(
                    f'{atom}'
                )

            leg = ax.get_legend()

            if leg is None:
                handles = legend_lines
                labels = legend_labels
            else:
                handles = [l._legmarker for l in leg.legendHandles]
                labels = [text._text for text in leg.texts]
                handles.extend(legend_lines)
                labels.extend(legend_labels)

            ax.legend(
                handles,
                labels,
                ncol=1,
                loc='upper left',
                fontsize=5,
                bbox_to_anchor=(1, 1),
                borderaxespad=0,
                frameon=False,
                handletextpad=0.1,
            )

    def plot_elements(self, ax, elements, scale_factor=5, color_list=None, legend=True, linewidth=0.75, band_color='black'):
        """
        This function plots the projected band structure on specified elements in the calculated structure

        Parameters:
            ax (matplotlib.pyplot.axis): Axis to plot the data on
            elements (list): List of element symbols to project onto
            scale_factor (float): Factor to scale weights. This changes the size of the
                points in the scatter plot
            color_list (list): List of colors of the same length as the elements list
            legend (bool): Determines if the legend should be included or not.
            linewidth (float): Line width of the plain band structure plotted in the background
            band_color (string): Color of the plain band structure
        """
        scale_factor = scale_factor ** 1.5

        self.plot_plain(ax=ax, linewidth=linewidth, color=band_color)

        element_dict = self._sum_elements(elements=elements, orbitals=False)

        if color_list is None:
            color_dict = self.color_dict
        else:
            color_dict = {i: color for i, color in enumerate(color_list)}

        plot_element = {element: [] for element in elements}
        plot_band = []
        plot_wave_vec = []

        for band in element_dict:
            plot_band.extend(self.bands_dict[band])
            plot_wave_vec.extend(self.wave_vector)
            for element in elements:
                plot_element[element].extend(element_dict[band][element])

        for (i, element) in enumerate(elements):
            ax.scatter(
                plot_wave_vec,
                plot_band,
                c=color_dict[i],
                s=scale_factor * np.array(plot_element[element]),
                zorder=1,
            )

        if legend:
            legend_lines = []
            legend_labels = []
            for (i, element) in enumerate(elements):
                legend_lines.append(plt.Line2D(
                    [0],
                    [0],
                    marker='o',
                    markersize=2,
                    linestyle='',
                    color=color_dict[i])
                )
                legend_labels.append(
                    f'{element}'
                )

            leg = ax.get_legend()

            if leg is None:
                handles = legend_lines
                labels = legend_labels
            else:
                handles = [l._legmarker for l in leg.legendHandles]
                labels = [text._text for text in leg.texts]
                handles.extend(legend_lines)
                labels.extend(legend_labels)

            ax.legend(
                handles,
                labels,
                ncol=1,
                loc='upper left',
                fontsize=5,
                bbox_to_anchor=(1, 1),
                borderaxespad=0,
                frameon=False,
                handletextpad=0.1,
            )

    def plot_element_orbitals(self, ax, element_orbital_pairs, scale_factor=5, color_list=None, legend=True, linewidth=0.75, band_color='black'):
        """
        this function plots the projected band structure on chosen orbitals for each specified element in the calculated structure.

        Parameters:
            ax (matplotlib.pyplot.axis): axis to plot the data on
            element_orbital_pairs (list[list]): List of list in the form of 
                [[element symbol, orbital index], [element symbol, orbital_index], ...]
            scale_factor (float): factor to scale weights. this changes the size of the
                points in the scatter plot
            color_list (list): List of colors of the same length as the element_orbital_pairs
            legend (bool): determines if the legend should be included or not.
            linewidth (float): line width of the plain band structure plotted in the background
            band_color (string): color of the plain band structure
        """
        scale_factor = scale_factor ** 1.5

        self.plot_plain(ax=ax, linewidth=linewidth, color=band_color)

        elements = [i[0] for i in element_orbital_pairs]

        element_dict = self._sum_elements(elements=elements, orbitals=True)

        if color_list is None:
            color_dict = self.color_dict
        else:
            color_dict = {i: color for i, color in enumerate(color_list)}

        plot_element = {element: pd.DataFrame(
            columns=[range(9)]) for element in elements}
        plot_band = []
        plot_wave_vec = []

        for band in element_dict:
            plot_band.extend(self.bands_dict[band])
            plot_wave_vec.extend(self.wave_vector)
            for element in elements:
                plot_element[element] = plot_element[element].append(
                    element_dict[band][element])

        for i, element_orbital_pair in enumerate(element_orbital_pairs):
            element = element_orbital_pair[0]
            orbital = element_orbital_pair[1]
            ax.scatter(
                plot_wave_vec,
                plot_band,
                c=color_dict[i],
                s=scale_factor * plot_element[element][orbital],
                zorder=1,
            )

        if legend:
            legend_lines = []
            legend_labels = []
            for i, element_orbital_pair in enumerate(element_orbital_pairs):
                element = element_orbital_pair[0]
                orbital = element_orbital_pair[1]
                legend_lines.append(plt.Line2D(
                    [0],
                    [0],
                    marker='o',
                    markersize=2,
                    linestyle='',
                    color=color_dict[i])
                )
                legend_labels.append(
                    f'{element}({self.orbital_labels[orbital]})'
                )

            leg = ax.get_legend()

            if leg is None:
                handles = legend_lines
                labels = legend_labels
            else:
                handles = [l._legmarker for l in leg.legendHandles]
                labels = [text._text for text in leg.texts]
                handles.extend(legend_lines)
                labels.extend(legend_labels)

            ax.legend(
                handles,
                labels,
                ncol=1,
                loc='upper left',
                fontsize=5,
                bbox_to_anchor=(1, 1),
                borderaxespad=0,
                frameon=False,
                handletextpad=0.1,
            )

    def plot_element_spd(self, elements, ax, order=['s', 'p', 'd'], scale_factor=5, color_dict=None, legend=True, linewidth=0.75, band_color='black'):
        """
        This function plots the projected band structure on the s, p, and d orbitals for each specified element in the calculated structure.

        Parameters:
            ax (matplotlib.pyplot.axis): Axis to plot the data on
            elements (list): List of element symbols to project onto
            order (list): This determines the order in which the points are plotted on the
                graph. This is an option because sometimes certain orbitals can be hidden
                under other orbitals because they have a larger weight. For example, if the
                signitures of the d orbitals are greater than that of the s orbitals, it
                might be smart to choose ['d', 'p', 's'] as the order so the s orbitals are
                plotted over the d orbitals.
            scale_factor (float): Factor to scale weights. This changes the size of the
                points in the scatter plot
            color_dict (dict[str][str]): This option allow the colors of the s, p, and d
                orbitals to be specified. Should be in the form of:
                {'s': <s color>, 'p': <p color>, 'd': <d color>}
            legend (bool): Determines if the legend should be included or not.
            linewidth (float): Line width of the plain band structure plotted in the background
            band_color (string): Color of the plain band structure
        """
        scale_factor = scale_factor ** 1.5

        self.plot_plain(ax=ax, linewidth=linewidth, color=band_color)

        element_dict = self._sum_elements(
            elements=elements, orbitals=True, spd=True)

        if color_dict is None:
            color_dict = {
                's': self.color_dict[0],
                'p': self.color_dict[1],
                'd': self.color_dict[2],
                'f': self.color_dict[4],
            }

        plot_element = {element: pd.DataFrame() for element in elements}

        if self.forbitals and 'f' not in order:
            order.append('f')

        plot_band = []
        plot_wave_vec = []

        for band in element_dict:
            plot_band.extend(self.bands_dict[band])
            plot_wave_vec.extend(self.wave_vector)
            for element in elements:
                plot_element[element] = plot_element[element].append(
                    element_dict[band][element])

        for (i, element) in enumerate(elements):
            if self.forbitals:
                electronic_structure = Element(
                    element).full_electronic_structure
                if not np.isin('f', electronic_structure):
                    order = order.remove('f')
            for orbital in order:
                ax.scatter(
                    plot_wave_vec,
                    plot_band,
                    c=color_dict[orbital],
                    s=scale_factor * plot_element[element][orbital],
                    zorder=1,
                )

        if legend:
            legend_lines = []
            legend_labels = []
            for element in elements:
                for orbital in order:
                    legend_lines.append(plt.Line2D(
                        [0],
                        [0],
                        marker='o',
                        markersize=2,
                        linestyle='',
                        color=color_dict[orbital])
                    )
                    legend_labels.append(
                        f'{element}(${orbital}$)'
                    )

            leg = ax.get_legend()

            if leg is None:
                handles = legend_lines
                labels = legend_labels
            else:
                handles = [l._legmarker for l in leg.legendHandles]
                labels = [text._text for text in leg.texts]
                handles.extend(legend_lines)
                labels.extend(legend_labels)

            ax.legend(
                handles,
                labels,
                ncol=1,
                loc='upper left',
                fontsize=5,
                bbox_to_anchor=(1, 1),
                borderaxespad=0,
                frameon=False,
                handletextpad=0.1,
            )


