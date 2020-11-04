from pymatgen.electronic_structure.core import Spin, Orbital
from pymatgen.io.vasp.outputs import BSVasprun, Eigenval
from pymatgen.io.vasp.inputs import Kpoints, Poscar, Incar
from pymatgen.core.periodic_table import Element
from pyprocar.utilsprocar import UtilsProcar
from pyprocar.procarparser import ProcarParser
from functools import reduce
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from copy import deepcopy
import os

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)


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

        self.eigenval = Eigenval(os.path.join(folder, 'EIGENVAL'))
        self.efermi = float(os.popen(f'grep E-fermi {os.path.join(folder, "OUTCAR")}').read().split()[2])
        self.poscar = Poscar.from_file(
            os.path.join(folder, 'POSCAR'),
            check_for_POTCAR=False,
            read_velocities=False
        )
        self.incar = Incar.from_file(
            os.path.join(folder, 'INCAR')
        )
        self.projected = projected
        self.forbitals = False
        self.hse = hse
        self.kpath = kpath
        self.n = n
        self.folder = folder
        self.spin = spin
        self.spin_dict = {'up': Spin.up, 'down': Spin.down}
        self.eigenvalues, self.kpoints = self._load_bands()
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
            9: '#FF0000',
            10: '#0000FF',
            11: '#008000',
            12: '#800080',
            13: '#E09200',
            14: '#FF5C77',
            15: '#778392',
        }
        self.orbital_labels = {
            0: 's',
            1: 'p_{y}',
            2: 'p_{x}',
            3: 'p_{z}',
            4: 'd_{xy}',
            5: 'd_{yz}',
            6: 'd_{z^{2}}',
            7: 'd_{xz}',
            8: 'd_{x^{2}-y^{2}}',
            9: 'f_{y^{3}x^{2}}',
            10: 'f_{xyz}',
            11: 'f_{yz^{2}}',
            12: 'f_{z^{3}}',
            13: 'f_{xz^{2}}',
            14: 'f_{zx^{3}}',
            15: 'f_{x^{3}}',
        }
        self.spd_relations = {
            's': 0,
            'p': 1,
            'd': 2,
            'f': 3,
        }

        if 'LORBIT' in self.incar:
            if self.incar['LORBIT']:
                self.lorbit = True
            else:
                self.lorbit = False
        else:
            self.lorbit = False

        if projected:
            self.pre_loaded_projections = os.path.isfile(os.path.join(folder, 'projected_eigenvalues.npy'))
            self.projected_eigenvalues = self._load_projected_bands()

        if not hse:
            self.kpoints_file = Kpoints.from_file(os.path.join(folder, 'KPOINTS'))

    def _load_bands(self):
        """
        This function is used to load eigenvalues from the vasprun.xml
        file and into a dictionary which is in the form of
        band index --> eigenvalues

        Returns:
            bands_dict (dict[str][np.ndarray]): Dictionary which contains
                the eigenvalues for each band
        """

        spin = self.spin_dict[self.spin]

        eigenvalues = np.transpose(self.eigenval.eigenvalues[spin][:,:,0]) - self.efermi
        kpoints = np.array(self.eigenval.kpoints)

        if self.hse:
            kpoints_band = self.n * (len(self.kpath) - 1)
            eigenvalues = eigenvalues[-kpoints_band:]
            kpoints = kpoints[-kpoints_band:]

        return eigenvalues, kpoints


    def _load_projected_bands(self):
        """
        This function loads the project weights of the orbitals in each band
        from vasprun.xml into a dictionary of the form:
        band index --> atom index --> weights of orbitals

        Returns:
            projected_dict (dict([str][int][pd.DataFrame])): Dictionary containing the projected weights of all orbitals on each atom for each band.
        """
        
        if self.lorbit:
            spin = 0
        elif self.spin == 'up':
            spin = 0
        elif self.spin == 'down':
            spin = 1

        if not os.path.isfile(os.path.join(self.folder, 'PROCAR_repaired')):
            UtilsProcar().ProcarRepair(
                os.path.join(self.folder, 'PROCAR'),
                os.path.join(self.folder, 'PROCAR_repaired'),
            )

        if self.pre_loaded_projections:
            with open(os.path.join(self.folder, 'projected_eigenvalues.npy'), 'rb') as projected_eigenvals:
                projected_eigenvalues = np.load(projected_eigenvals) 
        else:
            parser = ProcarParser()
            parser.readFile(os.path.join(self.folder, 'PROCAR_repaired'))
            projected_eigenvalues = np.transpose(parser.spd[:,:,spin,:-1, 1:-1], axes=(1,0,2,3))
            projected_eigenvalues = projected_eigenvalues / np.sum(np.sum(projected_eigenvalues, axis=3),axis=2)[:,:,np.newaxis,np.newaxis]
            np.save(os.path.join(self.folder, 'projected_eigenvalues.npy'), projected_eigenvalues)

        if self.hse:
            kpoints_band = self.n * (len(self.kpath) - 1)
            projected_eigenvalues = projected_eigenvalues[-kpoints_band:]

        if projected_eigenvalues.shape[-1] == 16:
            self.forbitals = True

        projected_eigenvalues = np.square(projected_eigenvalues)

        return projected_eigenvalues


    def _sum_spd(self, spd):
        """
        This function sums the weights of the s, p, and d orbitals for each atom
        and creates a dictionary of the form:
        band index --> s,p,d orbital weights

        Returns:
            spd_dict (dict([str][pd.DataFrame])): Dictionary that contains the summed weights for the s, p, and d orbitals for each band
        """

        if not self.forbitals:
            spd_indices = [np.array([False for _ in range(9)]) for i in range(3)]
            spd_indices[0][0] = True
            spd_indices[1][1:4] = True
            spd_indices[2][4:] = True
        else:
            spd_indices = [np.array([False for _ in range(16)]) for i in range(4)]
            spd_indices[0][0] = True
            spd_indices[1][1:4] = True
            spd_indices[2][4:9] = True
            spd_indices[3][9:] = True

        orbital_contributions = self.projected_eigenvalues.sum(axis=2)

        spd_contributions = np.transpose(
            np.array([
                np.sum(orbital_contributions[:,:,ind], axis=2) for ind in spd_indices
            ]), axes=[1,2,0]
        )

        spd_contributions = spd_contributions[:,:,[self.spd_relations[orb] for orb in spd]]

        print(spd_contributions.shape)

        return spd_contributions



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
        orbital_contributions = self.projected_eigenvalues.sum(axis=2)[:,:,[orbitals]]

        return orbital_contributions

    def _sum_atoms(self, atoms, spd=False):
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

        if spd:
            if not self.forbitals:
                spd_indices = [np.array([False for _ in range(9)]) for i in range(3)]
                spd_indices[0][0] = True
                spd_indices[1][1:4] = True
                spd_indices[2][4:] = True
            else:
                spd_indices = [np.array([False for _ in range(16)]) for i in range(4)]
                spd_indices[0][0] = True
                spd_indices[1][1:4] = True
                spd_indices[2][4:9] = True
                spd_indices[3][9:] = True

            atoms_spd = np.transpose(np.array([
                np.sum(self.projected_eigenvalues[:,:,:,ind], axis=3) for ind in spd_indices
            ]), axes=(1,2,3,0))

            return atoms_spd
        else:
            atoms_array = self.projected_eigenvalues.sum(axis=3)[:,:,[atoms]]

            return atoms_array

    def _sum_elements(self, elements, orbitals=False, spd=False, spd_options=None):
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
        projected_eigenvalues = self.projected_eigenvalues

        element_list = np.hstack(
            [[symbols[i] for j in range(natoms[i])] for i in range(len(symbols))]
        )

        element_indices = [np.where(np.isin(element_list, element))[0] for element in elements]

        element_orbitals = np.transpose(
            np.array([
                np.sum(projected_eigenvalues[:,:,ind,:], axis=2) for ind in element_indices
            ]), axes=(1,2,0,3)
        )

        if orbitals:
            return element_orbitals
        elif spd:
            if not self.forbitals:
                spd_indices = [np.array([False for _ in range(9)]) for i in range(3)]
                spd_indices[0][0] = True
                spd_indices[1][1:4] = True
                spd_indices[2][4:] = True
            else:
                spd_indices = [np.array([False for _ in range(16)]) for i in range(4)]
                spd_indices[0][0] = True
                spd_indices[1][1:4] = True
                spd_indices[2][4:9] = True
                spd_indices[3][9:] = True

            element_spd = np.transpose(np.array([
                np.sum(element_orbitals[:,:,:,ind], axis=3) for ind in spd_indices
            ]), axes=(1,2,3,0))

            #  element_spd = element_spd[:,:,[self.spd_relations[i] for i in spd_options]]

            return element_spd
        else:
            element_array = np.sum(element_orbitals, axis=3)
            return element_array



    def _get_k_distance(self):
        cell = self.poscar.structure.lattice.matrix
        kpt_c = np.dot(self.kpoints, np.linalg.inv(cell).T)
        kdist = np.r_[0, np.cumsum(np.linalg.norm( np.diff(kpt_c,axis=0), axis=1))]

        return kdist


    def _get_kticks(self, ax):
        """
        This function extracts the kpoint labels and index locations for a regular
        band structure calculation (non HSE).

        Parameters:
            ax (matplotlib.pyplot.axis): Axis to append the tick labels
        """

        high_sym_points = self.kpoints_file.kpts
        kpts_labels = np.array([f'${k}$' if k != 'G' else '$\\Gamma$' for k in self.kpoints_file.labels])
        all_kpoints = self.kpoints

        index = [0]
        for i in range(len(high_sym_points) - 2):
            if high_sym_points[i + 2] != high_sym_points[i + 1]:
                index.append(i)
        index.append(len(high_sym_points) - 1)

        kpts_loc = np.isin(np.round(all_kpoints, 3), np.round(high_sym_points, 3)).all(1)
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

    def _filter_bands(self, erange):
        eigenvalues = self.eigenvalues
        where = (eigenvalues >= np.min(erange)) & (eigenvalues <= np.max(erange))
        is_true = np.sum(np.isin(where, True), axis=1)
        bands_in_plot = is_true > 0

        return bands_in_plot

    def _add_legend(self, ax, names, colors):
        legend_lines = []
        legend_labels = []
        for name, color in zip(names, colors):
            legend_lines.append(plt.Line2D(
                [0],
                [0],
                marker='o',
                markersize=2,
                linestyle='',
                color=color
            ))
            legend_labels.append(
                f'${name}$'
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

    def plot_plain(self, ax, color='black', erange=[-6,6], linewidth=1.25, linestyle='-'):
        """
        This function plots a plain band structure.

        Parameters:
            ax (matplotlib.pyplot.axis): Axis to plot the data on
            color (str): Color of the band structure lines
            linewidth (float): Line width of the band structure lines
            linestyle (str): Line style of the bands
        """

        bands_in_plot = self._filter_bands(erange=erange)
        eigenvalues = self.eigenvalues[bands_in_plot]
        wave_vectors = self._get_k_distance()
        eigenvalues_ravel = np.ravel(np.c_[eigenvalues, np.empty(eigenvalues.shape[0]) * np.nan])
        wave_vectors_tile = np.tile(np.append(wave_vectors, np.nan), eigenvalues.shape[0])

        ax.plot(
            wave_vectors_tile,
            eigenvalues_ravel,
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            zorder=0,
        )

        if self.hse:
            self._get_kticks_hse(ax=ax, kpath=self.kpath, n=self.n)
        else:
            self._get_kticks(ax=ax)

        ax.set_xlim(0, np.max(wave_vectors))


    def _plot_projected_general(self, ax, projected_data, colors, scale_factor=5, erange=[-6,6], display_order=None, linewidth=0.75, band_color='black'):
        """
        This is a general method for plotting projected data

        Parameters:
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
        
        self.plot_plain(ax=ax, linewidth=linewidth, color=band_color, erange=erange)

        bands_in_plot = self._filter_bands(erange=erange)
        projected_data = projected_data[bands_in_plot]
        wave_vectors = self._get_k_distance()
        eigenvalues = self.eigenvalues[bands_in_plot]

        projected_data_ravel = np.ravel(projected_data)
        wave_vectors_tile = np.tile(
            np.repeat(wave_vectors, projected_data.shape[-1]), projected_data.shape[0]
        )
        eigenvalues_tile = np.repeat(np.ravel(eigenvalues), projected_data.shape[-1])
        colors_tile = np.tile(colors, np.prod(projected_data.shape[:-1]))

        if display_order is None:
            pass
        else:
            sort_index = np.argsort(projected_data_ravel)

            if display_order == 'all':
                sort_index = sort_index[::-1]

            wave_vectors_tile = wave_vectors_tile[sort_index]
            eigenvalues_tile = eigenvalues_tile[sort_index]
            colors_tile = colors_tile[sort_index]
            projected_data_ravel = projected_data_ravel[sort_index]

        ax.scatter(
            wave_vectors_tile,
            eigenvalues_tile,
            c=colors_tile,
            s=scale_factor * projected_data_ravel,
            zorder=100,
        )
        
    def plot_orbitals(self, ax, orbitals, scale_factor=5, erange=[-6,6], display_order=None, color_list=None, legend=True, linewidth=0.75, band_color='black'):
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

        if color_list is None:
            colors = np.array([self.color_dict[i] for i in orbitals])
        else:
            colors = color_list

        projected_data = self._sum_orbitals(orbitals=orbitals)

        self._plot_projected_general(
            ax=ax,
            projected_data=projected_data,
            colors=colors,
            scale_factor=scale_factor,
            erange=erange,
            display_order=display_order,
            linewidth=linewidth,
            band_color=band_color
        )

        self._add_legend(ax, names=[self.orbital_labels[i] for i in orbitals], colors=colors)


    def plot_spd(self, ax, scale_factor=5, orbitals='spd', erange=[-6,6], display_order=None, color_dict=None, legend=True, linewidth=0.75, band_color='black'):
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
        if color_dict is None:
            color_dict = {
                0: self.color_dict[0],
                1: self.color_dict[1],
                2: self.color_dict[2],
                3: self.color_dict[4],
            }

        if self.forbitals:
            orbitals = orbitals + 'f'

        colors = np.array([color_dict[self.spd_relations[i]] for i in orbitals])

        projected_data = self._sum_spd(spd=orbitals)

        self._plot_projected_general(
            ax=ax,
            projected_data=projected_data,
            colors=colors,
            scale_factor=scale_factor,
            erange=erange,
            display_order=display_order,
            linewidth=linewidth,
            band_color=band_color
        )

        self._add_legend(ax, names=[i for i in orbitals], colors=colors)



    def plot_atoms(self, ax, atoms, scale_factor=5, erange=[-6,6], display_order=None, color_list=None, legend=True, linewidth=0.75, band_color='black'):
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
        if color_list is None:
            colors = np.array([self.color_dict[i] for i in range(len(atoms))])
        else:
            colors = color_list

        projected_data = self._sum_atoms(atoms=atoms)

        self._plot_projected_general(
            ax=ax,
            projected_data=projected_data,
            colors=colors,
            scale_factor=scale_factor,
            erange=erange,
            display_order=display_order,
            linewidth=linewidth,
            band_color=band_color
        )

        self._add_legend(ax, names=atoms, colors=colors)


    def plot_atom_orbitals(self, ax, atom_orbital_dict, scale_factor=5, erange=[-6,6], display_order=None, color_list=None, legend=True, linewidth=0.75, band_color='black'):
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

        atom_indices = list(atom_orbital_dict.keys())
        orbital_indices = list(atom_orbital_dict.values())
        number_orbitals = [len(i) for i in orbital_indices]
        atom_indices = np.repeat(atom_indices, number_orbitals)
        orbital_symbols_long = np.hstack([
            [self.orbital_labels[o] for o in  orb] for orb in orbital_indices
        ])
        orbital_indices_long = np.hstack(orbital_indices)
        indices = np.vstack([atom_indices, orbital_indices_long]).T

        projected_data = self.projected_eigenvalues
        projected_data = np.transpose(np.array([
            projected_data[:,:,ind[0],ind[1]] for ind in indices
        ]), axes=(1,2,0))

        if color_list is None:
            colors = np.array([self.color_dict[i] for i in range(len(orbital_indices_long))])
        else:
            colors = color_list

        self._plot_projected_general(
            ax=ax,
            projected_data=projected_data,
            colors=colors,
            scale_factor=scale_factor,
            erange=erange,
            display_order=display_order,
            linewidth=linewidth,
            band_color=band_color
        )

        self._add_legend(
            ax,
            names=[f'{i[0]}, {i[1]}' for i in zip(atom_indices, orbital_symbols_long)],
            colors=colors
        )

    def plot_atom_spd(self, ax, atom_spd_dict, scale_factor=5, erange=[-6,6], display_order=None, color_list=None, legend=True, linewidth=0.75, band_color='black'):
        """
        This function plots the projected band structure on the s, p, and d orbitals for each specified atom in the calculated structure.

        Parameters:
            ax (matplotlib.pyplot.axis): Axis to plot the data on
            atom_spd_dict (dict): Dictionary to determine the atom and spd orbitals to project onto
                Format: {0: 'spd', 1: 'sp', 2: 's'} where 0,1,2 are atom indicies in the POSCAR
            display_order (None or str): The available options are None, 'all', 'dominant' where None
                plots the scatter points in the order presented in the atom_spd_dict, 'all' plots the 
                scatter points largest --> smallest to all points are visable, and 'dominant' plots
                the scatter points smallest --> largest so only the dominant color is visable.
            scale_factor (float): Factor to scale weights. This changes the size of the
                points in the scatter plot
            color_dict (dict[str][str]): This option allow the colors of the s, p, and d
                orbitals to be specified. Should be in the form of:
                {'s': <s color>, 'p': <p color>, 'd': <d color>}
            legend (bool): Determines if the legend should be included or not.
            linewidth (float): Line width of the plain band structure plotted in the background
            band_color (string): Color of the plain band structure
        """
        atom_indices = list(atom_spd_dict.keys())
        orbital_symbols = list(atom_spd_dict.values())
        number_orbitals = [len(i) for i in orbital_symbols]
        atom_indices = np.repeat(atom_indices, number_orbitals)
        orbital_symbols_long = np.hstack([[o for o in  orb] for orb in orbital_symbols])
        orbital_indices = np.hstack([[self.spd_relations[o] for o in  orb] for orb in orbital_symbols])
        indices = np.vstack([atom_indices, orbital_indices]).T

        projected_data = self._sum_atoms(atoms=atom_indices, spd=True)
        projected_data = np.transpose(np.array([
            projected_data[:,:,ind[0],ind[1]] for ind in indices
        ]), axes=(1,2,0))

        if color_list is None:
            colors = np.array([self.color_dict[i] for i in range(len(orbital_symbols_long))])
        else:
            colors = color_list

        self._plot_projected_general(
            ax=ax,
            projected_data=projected_data,
            colors=colors,
            scale_factor=scale_factor,
            erange=erange,
            display_order=display_order,
            linewidth=linewidth,
            band_color=band_color
        )

        self._add_legend(
            ax,
            names=[f'{i[0]}, {i[1]}' for i in zip(atom_indices, orbital_symbols_long)],
            colors=colors
        )



    def plot_elements(self, ax, elements, scale_factor=5, erange=[-6,6], display_order=None, color_list=None, legend=True, linewidth=0.75, band_color='black'):
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
        if color_list is None:
            colors = np.array([self.color_dict[i] for i in range(len(elements))])
        else:
            colors = color_list

        projected_data = self._sum_elements(elements=elements)

        self._plot_projected_general(
            ax=ax,
            projected_data=projected_data,
            colors=colors,
            scale_factor=scale_factor,
            erange=erange,
            display_order=display_order,
            linewidth=linewidth,
            band_color=band_color
        )

        self._add_legend(ax, names=elements, colors=colors)


    def plot_element_orbitals(self, ax, element_orbital_dict, scale_factor=5, erange=[-6,6], display_order=None, color_list=None, legend=True, linewidth=0.75, band_color='black'):
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
        element_symbols = list(element_orbital_dict.keys())
        orbital_indices = list(element_orbital_dict.values())
        number_orbitals = [len(i) for i in orbital_indices]
        element_symbols_long = np.repeat(element_symbols, number_orbitals)
        element_indices = np.repeat(range(len(element_symbols)), number_orbitals)
        orbital_symbols_long = np.hstack([[self.orbital_labels[o] for o in  orb] for orb in orbital_indices])
        orbital_indices_long = np.hstack(orbital_indices)
        indices = np.vstack([element_indices, orbital_indices_long]).T

        projected_data = self._sum_elements(elements=element_symbols, orbitals=True)
        projected_data = np.transpose(np.array([
            projected_data[:,:,ind[0],ind[1]] for ind in indices
        ]), axes=(1,2,0))

        if color_list is None:
            colors = np.array([self.color_dict[i] for i in range(len(orbital_indices_long))])
        else:
            colors = color_list

        self._plot_projected_general(
            ax=ax,
            projected_data=projected_data,
            colors=colors,
            scale_factor=scale_factor,
            erange=erange,
            display_order=display_order,
            linewidth=linewidth,
            band_color=band_color
        )

        self._add_legend(
            ax,
            names=[f'{i[0]}, {i[1]}' for i in zip(element_symbols_long, orbital_symbols_long)],
            colors=colors
        )

    def plot_element_spd(self, ax, element_spd_dict, scale_factor=5, erange=[-6,6], display_order=None, color_list=None, legend=True, linewidth=0.75, band_color='black'):
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
        element_symbols = list(element_spd_dict.keys())
        orbital_symbols = list(element_spd_dict.values())
        number_orbitals = [len(i) for i in orbital_symbols]
        element_symbols_long = np.repeat(element_symbols, number_orbitals)
        element_indices = np.repeat(range(len(element_symbols)), number_orbitals)
        orbital_symbols_long = np.hstack([[o for o in  orb] for orb in orbital_symbols])
        orbital_indices = np.hstack([[self.spd_relations[o] for o in  orb] for orb in orbital_symbols])
        indices = np.vstack([element_indices, orbital_indices]).T

        projected_data = self._sum_elements(elements=element_symbols, spd=True)
        projected_data = np.transpose(np.array([
            projected_data[:,:,ind[0],ind[1]] for ind in indices
        ]), axes=(1,2,0))

        if color_list is None:
            colors = np.array([self.color_dict[i] for i in range(len(orbital_symbols_long))])
        else:
            colors = color_list


        self._plot_projected_general(
            ax=ax,
            projected_data=projected_data,
            colors=colors,
            scale_factor=scale_factor,
            erange=erange,
            display_order=display_order,
            linewidth=linewidth,
            band_color=band_color
        )

        self._add_legend(
            ax,
            names=[f'{i[0]}, {i[1]}' for i in zip(element_symbols_long, orbital_symbols_long)],
            colors=colors
        )


    # =================================================================================================
    # =================================== Old Stuff ===================================================
    # =================================================================================================

    #  def plot_orbitals_old(self, ax, orbitals, scale_factor=5, color_dict=None, legend=True, linewidth=0.75, band_color='black'):
        #  """
        #  This function plots the projected band structure of given orbitals summed across all atoms on a given axis.
#
        #  Parameters:
            #  ax (matplotlib.pyplot.axis): Axis to plot the data on
            #  orbitals (list): List of orbits to compare
#
                #  | 0 = s
                #  | 1 = py
                #  | 2 = pz
                #  | 3 = px
                #  | 4 = dxy
                #  | 5 = dyz
                #  | 6 = dz2
                #  | 7 = dxz
                #  | 8 = dx2-y2
                #  | 9 = fy3x2
                #  | 10 = fxyz
                #  | 11 = fyz2
                #  | 12 = fz3
                #  | 13 = fxz2
                #  | 14 = fzx3
                #  | 15 = fx3
#
            #  scale_factor (float): Factor to scale weights. This changes the size of the
                #  points in the scatter plot
            #  color_dict (dict[str][str]): This option allow the colors of each orbital
                #  specified. Should be in the form of:
                #  {'orbital index': <color>, 'orbital index': <color>, ...}
            #  legend (bool): Determines if the legend should be included or not.
            #  linewidth (float): Line width of the plain band structure plotted in the background
            #  band_color (string): Color of the plain band structure
        #  """
        #  scale_factor = scale_factor ** 1.5
#
        #  self.plot_plain(ax=ax, linewidth=linewidth, color=band_color)
#
        #  orbital_dict = self._sum_orbitals(orbitals=orbitals)
#
        #  if color_dict is None:
            #  color_dict = self.color_dict
#
        #  plot_df = pd.DataFrame(columns=orbitals)
        #  plot_band = []
        #  plot_wave_vec = []
#
        #  for band in orbital_dict:
            #  plot_df = plot_df.append(orbital_dict[band])
            #  plot_band.extend(self.bands_dict[band])
            #  plot_wave_vec.extend(self.wave_vector)
#
        #  for orbital in orbitals:
            #  ax.scatter(
                #  plot_wave_vec,
                #  plot_band,
                #  c=color_dict[orbital],
                #  s=scale_factor * plot_df[orbital],
                #  zorder=1,
            #  )
#
        #  if legend:
            #  legend_lines = []
            #  legend_labels = []
            #  for orbital in orbitals:
                #  legend_lines.append(plt.Line2D(
                    #  [0],
                    #  [0],
                    #  marker='o',
                    #  markersize=2,
                    #  linestyle='',
                    #  color=color_dict[orbital])
                #  )
                #  legend_labels.append(
                    #  f'{self.orbital_labels[orbital]}'
                #  )
#
            #  leg = ax.get_legend()
#
            #  if leg is None:
                #  handles = legend_lines
                #  labels = legend_labels
            #  else:
                #  handles = [l._legmarker for l in leg.legendHandles]
                #  labels = [text._text for text in leg.texts]
                #  handles.extend(legend_lines)
                #  labels.extend(legend_labels)
#
            #  ax.legend(
                #  handles,
                #  labels,
                #  ncol=1,
                #  loc='upper left',
                #  fontsize=5,
                #  bbox_to_anchor=(1, 1),
                #  borderaxespad=0,
                #  frameon=False,
                #  handletextpad=0.1,
            #  )
#
    #  def _load_bands_old(self):
        #  """
        #  This function is used to load eigenvalues from the vasprun.xml
        #  file and into a dictionary which is in the form of
        #  band index --> eigenvalues
#
        #  Returns:
            #  bands_dict (dict[str][np.ndarray]): Dictionary which contains
                #  the eigenvalues for each band
        #  """
#
        #  spin = self.spin
#
        #  if self.hse:
            #  kpoints_band = self.n * (len(self.kpath) - 1)
            #  eigenvalues = self.vasprun.eigenvalues[
                #  self.spin_dict[spin]
            #  ][-1 * kpoints_band:]
        #  else:
            #  eigenvalues = self.vasprun.eigenvalues[self.spin_dict[spin]]
#
        #  efermi = self.vasprun.efermi
        #  nkpoints = len(eigenvalues)
        #  nbands = len(eigenvalues[0])
#
        #  bands_dict = {f'band{i+1}': [] for i in range(nbands)}
#
        #  for i in range(nkpoints):
            #  for j in range(nbands):
                #  bands_dict[f'band{j+1}'].append(
                    #  eigenvalues[i][j][0] - efermi
                #  )
#
        #  return bands_dict
#
    #  def _load_projected_bands_old(self):
        #  """
        #  This function loads the project weights of the orbitals in each band
        #  from vasprun.xml into a dictionary of the form:
        #  band index --> atom index --> weights of orbitals
#
        #  Returns:
            #  projected_dict (dict([str][int][pd.DataFrame])): Dictionary containing the projected weights of all orbitals on each atom for each band.
        #  """
#
        #  spin = self.spin
#
        #  if self.hse:
            #  kpoints_band = self.n * (len(self.kpath) - 1)
            #  projected_eigenvalues = self.vasprun.projected_eigenvalues[
                #  self.spin_dict[spin]
            #  ][-1 * kpoints_band:]
        #  else:
            #  projected_eigenvalues = self.vasprun.projected_eigenvalues[
                #  self.spin_dict[spin]
            #  ]
#
        #  poscar = Poscar.from_file(
            #  f'{self.folder}/POSCAR',
            #  check_for_POTCAR=False,
            #  read_velocities=False
        #  )
        #  spin = self.spin
        #  natoms = np.sum(poscar.natoms)
        #  nkpoints = len(projected_eigenvalues)
        #  nbands = len(projected_eigenvalues[0])
        #  norbitals = len(projected_eigenvalues[0][0][0])
#
        #  if norbitals == 16:
            #  self.forbitals = True
#
        #  projected_dict = {f'band{i+1}':
                          #  {atom: np.zeros(norbitals) for atom in range(natoms)}
                          #  for i in range(nbands)}
#
        #  for i in range(nkpoints):
            #  for j in range(nbands):
                #  band = f'band{j+1}'
                #  for atom in range(natoms):
                    #  orbital_weights = projected_eigenvalues[i][j][atom] ** 2
                    #  projected_dict[band][atom] = np.vstack([
                        #  projected_dict[band][atom],
                        #  orbital_weights
                    #  ])
#
        #  for band in projected_dict:
            #  for atom in projected_dict[band]:
                #  projected_dict[band][atom] = pd.DataFrame(
                    #  projected_dict[band][atom][1:]
                #  )
#
        #  return projected_dict
#
    #  def _sum_spd_old(self):
        #  """
        #  This function sums the weights of the s, p, and d orbitals for each atom
        #  and creates a dictionary of the form:
        #  band index --> s,p,d orbital weights
#
        #  Returns:
            #  spd_dict (dict([str][pd.DataFrame])): Dictionary that contains the summed weights for the s, p, and d orbitals for each band
        #  """
#
        #  # spd_orbitals = {'s': [0], 'p': [1, 2, 3], 'd': [4, 5, 6, 7, 8]}
#
        #  spd_dict = {band: np.nan for band in self.projected_dict}
#
        #  for band in self.projected_dict:
            #  atom_list = [
                #  self.projected_dict[band][atom] for atom in self.projected_dict[band]]
            #  spd_dict[band] = reduce(
                #  lambda x, y: x.add(y, fill_value=0), atom_list
            #  )
#
        #  for band in spd_dict:
            #  df = spd_dict[band]
            #  spd_dict[band]['s'] = df[0]
            #  spd_dict[band]['p'] = df[1] + df[2] + df[3]
            #  spd_dict[band]['d'] = df[4] + df[5] + df[6] + df[7] + df[8]
#
            #  if self.forbitals:
                #  spd_dict[band]['f'] = df[9] + df[10] + \
                    #  df[11] + df[12] + df[13] + df[14] + df[15]
                #  spd_dict[band] = spd_dict[band].drop(columns=range(16))
            #  else:
                #  spd_dict[band] = spd_dict[band].drop(columns=range(9))
#
        #  return spd_dict
#
    #  def _sum_atoms_old(self, atoms):
        #  """
        #  This function finds the weights of desired atoms for all orbitals and
            #  returns a dictionary of the form:
            #  band index --> atom index
#
        #  Parameters:
            #  atoms (list): List of desired atoms where atom 0 is the first atom in
                #  the POSCAR file.
#
        #  Returns:
            #  atom_dict (dict[str][pd.DataFrame]): Dictionary that contains the projected
                #  weights of the selected atoms.
        #  """
#
        #  projected_dict = self.projected_dict
        #  atoms_dict = {band: np.nan for band in projected_dict}
#
        #  for band in projected_dict:
            #  atom_dict = {atom: projected_dict[band][atom].sum(
                #  axis=1) for atom in atoms}
            #  atoms_dict[band] = pd.DataFrame.from_dict(atom_dict)
#
        #  return atoms_dict
#
    #  def plot_spd_old(self, ax, scale_factor=5, order=['s', 'p', 'd'], color_dict=None, legend=True, linewidth=0.75, band_color='black'):
        #  """
        #  This function plots the s, p, d projected band structure onto a given axis
#
        #  Parameters:
            #  ax (matplotlib.pyplot.axis): Axis to plot the data on
            #  scale_factor (float): Factor to scale weights. This changes the size of the
                #  points in the scatter plot
            #  order (list): This determines the order in which the points are plotted on the
                #  graph. This is an option because sometimes certain orbitals can be hidden
                #  under others because they have a larger weight. For example, if the
                #  weights of the d orbitals are greater than that of the s orbitals, it
                #  might be smart to choose ['d', 'p', 's'] as the order so the s orbitals are
                #  plotted over the d orbitals.
            #  color_dict (dict[str][str]): This option allow the colors of the s, p, and d
                #  orbitals to be specified. Should be in the form of:
                #  {'s': <s color>, 'p': <p color>, 'd': <d color>}
            #  legend (bool): Determines if the legend should be included or not.
            #  linewidth (float): Line width of the plain band structure plotted in the background
            #  band_color (string): Color of the plain band structure
        #  """
        #  scale_factor = scale_factor ** 1.5
#
        #  self.plot_plain(ax=ax, linewidth=linewidth, color=band_color)
#
        #  spd_dict = self._sum_spd()
#
        #  if color_dict is None:
            #  color_dict = {
                #  's': self.color_dict[0],
                #  'p': self.color_dict[1],
                #  'd': self.color_dict[2],
                #  'f': self.color_dict[4],
            #  }
#
        #  plot_df = pd.DataFrame()
#
        #  if self.forbitals and 'f' not in order:
            #  order.append('f')
#
        #  plot_band = []
        #  plot_wave_vec = []
#
        #  for band in spd_dict:
            #  plot_df = plot_df.append(spd_dict[band])
            #  plot_band.extend(self.bands_dict[band])
            #  plot_wave_vec.extend(self.wave_vector)
#
        #  for col in order:
            #  ax.scatter(
                #  plot_wave_vec,
                #  plot_band,
                #  c=color_dict[col],
                #  s=scale_factor * plot_df[col],
                #  zorder=1,
            #  )
#
        #  if legend:
            #  legend_lines = []
            #  legend_labels = []
            #  for orbital in order:
                #  legend_lines.append(plt.Line2D(
                    #  [0],
                    #  [0],
                    #  marker='o',
                    #  markersize=2,
                    #  linestyle='',
                    #  color=color_dict[orbital]
                #  ))
                #  legend_labels.append(
                    #  f'${orbital}$'
                #  )
#
            #  leg = ax.get_legend()
#
            #  if leg is None:
                #  handles = legend_lines
                #  labels = legend_labels
            #  else:
                #  handles = [l._legmarker for l in leg.legendHandles]
                #  labels = [text._text for text in leg.texts]
                #  handles.extend(legend_lines)
                #  labels.extend(legend_labels)
#
            #  ax.legend(
                #  handles,
                #  labels,
                #  ncol=1,
                #  loc='upper left',
                #  fontsize=5,
                #  bbox_to_anchor=(1, 1),
                #  borderaxespad=0,
                #  frameon=False,
                #  handletextpad=0.1,
            #  )
#
    #  def plot_atoms_old(self, ax, atoms, scale_factor=5, color_list=None, legend=True, linewidth=0.75, band_color='black'):
        #  """
        #  This function plots the projected band structure of given atoms summed across all orbitals on a given axis.
#
        #  Parameters:
            #  ax (matplotlib.pyplot.axis): Axis to plot the data on
            #  atoms (list): List of atoms to project onto
            #  scale_factor (float): Factor to scale weights. This changes the size of the
                #  points in the scatter plot
            #  color_list (list): List of colors of the same length as the atoms list
            #  legend (bool): Determines if the legend should be included or not.
            #  linewidth (float): Line width of the plain band structure plotted in the background
            #  band_color (string): Color of the plain band structure
        #  """
        #  scale_factor = scale_factor ** 1.5
#
        #  self.plot_plain(ax=ax, linewidth=linewidth, color=band_color)
#
        #  atom_dict = self._sum_atoms(atoms=atoms)
#
        #  if color_list is None:
            #  color_dict = self.color_dict
        #  else:
            #  color_dict = {i: color for i, color in enumerate(color_list)}
#
        #  plot_df = pd.DataFrame(columns=atoms)
        #  plot_band = []
        #  plot_wave_vec = []
#
        #  for band in atom_dict:
            #  plot_df = plot_df.append(atom_dict[band], ignore_index=True)
            #  plot_band.extend(self.bands_dict[band])
            #  plot_wave_vec.extend(self.wave_vector)
#
        #  for (i, atom) in enumerate(atoms):
            #  ax.scatter(
                #  plot_wave_vec,
                #  plot_band,
                #  c=color_dict[i],
                #  s=scale_factor * plot_df[atom],
                #  zorder=1,
            #  )
#
        #  if legend:
            #  legend_lines = []
            #  legend_labels = []
            #  for (i, atom) in enumerate(atoms):
                #  legend_lines.append(plt.Line2D(
                    #  [0],
                    #  [0],
                    #  marker='o',
                    #  markersize=2,
                    #  linestyle='',
                    #  color=color_dict[i])
                #  )
                #  legend_labels.append(
                    #  f'{atom}'
                #  )
#
            #  leg = ax.get_legend()
#
            #  if leg is None:
                #  handles = legend_lines
                #  labels = legend_labels
            #  else:
                #  handles = [l._legmarker for l in leg.legendHandles]
                #  labels = [text._text for text in leg.texts]
                #  handles.extend(legend_lines)
                #  labels.extend(legend_labels)
#
            #  ax.legend(
                #  handles,
                #  labels,
                #  ncol=1,
                #  loc='upper left',
                #  fontsize=5,
                #  bbox_to_anchor=(1, 1),
                #  borderaxespad=0,
                #  frameon=False,
                #  handletextpad=0.1,
            #  )
#
    #  def _sum_elements_old(self, elements, orbitals=False, spd=False):
        #  """
        #  This function sums the weights of the orbitals of specific elements within the
        #  calculated structure and returns a dictionary of the form:
        #  band index --> element label --> orbital weights for orbitals = True
        #  band index --> element label for orbitals = False
        #  This is useful for structures with many elements because manually entering indicies is
        #  not practical for large structures.
#
        #  Parameters:
            #  elements (list): List of element symbols to sum the weights of.
            #  orbitals (bool): Determines whether or not to inclue orbitals or not
                #  (True = keep orbitals, False = sum orbitals together )
            #  spd (bool): Determines whether or not to sum the s, p, and d orbitals
#
#
        #  Returns:
            #  element_dict (dict([str][str][pd.DataFrame])): Dictionary that contains the summed weights for each orbital for a given element in the structure.
        #  """
#
        #  poscar = self.poscar
        #  natoms = poscar.natoms
        #  symbols = poscar.site_symbols
        #  projected_dict = self.projected_dict
#
        #  element_list = np.hstack(
            #  [[symbols[i] for j in range(natoms[i])]
             #  for i in range(len(symbols))]
        #  )
#
        #  element_dict = {
            #  band: {element: [] for element in elements} for band in projected_dict
        #  }
#
        #  for band in projected_dict:
            #  band_df = pd.DataFrame()
            #  for element in elements:
                #  element_index = np.where(np.isin(element_list, element))[0]
                #  nb_atoms = len(element_index)
                #  df = pd.concat(
                    #  [projected_dict[band][i] for i in element_index],
                    #  axis=1
                #  )
#
                #  if orbitals:
                    #  element_dict[band][element] = df.groupby(
                        #  by=df.columns,
                        #  axis=1
                    #  ).sum()
                    #  if spd:
                        #  df = element_dict[band][element]
                        #  element_dict[band][element]['s'] = df[0]
                        #  element_dict[band][element]['p'] = df[1] + \
                            #  df[2] + df[3]
                        #  element_dict[band][element]['d'] = df[4] + \
                            #  df[5] + df[6] + df[7] + df[8]
#
                        #  if self.forbitals:
                            #  element_dict[band][element]['f'] = df[9] + df[10] + \
                                #  df[11] + df[12] + df[13] + df[14] + df[15]
                            #  element_dict[band][element] = element_dict[band][element].drop(
                                #  columns=range(16))
                        #  else:
                            #  element_dict[band][element] = element_dict[band][element].drop(
                                #  columns=range(9))
                #  else:
                    #  norm_df = df.sum(axis=1)
                    #  element_dict[band][element] = norm_df.tolist()
#
        #  return element_dict
#
    #  def plot_elements_old(self, ax, elements, scale_factor=5, color_list=None, legend=True, linewidth=0.75, band_color='black'):
        #  """
        #  This function plots the projected band structure on specified elements in the calculated structure
#
        #  Parameters:
            #  ax (matplotlib.pyplot.axis): Axis to plot the data on
            #  elements (list): List of element symbols to project onto
            #  scale_factor (float): Factor to scale weights. This changes the size of the
                #  points in the scatter plot
            #  color_list (list): List of colors of the same length as the elements list
            #  legend (bool): Determines if the legend should be included or not.
            #  linewidth (float): Line width of the plain band structure plotted in the background
            #  band_color (string): Color of the plain band structure
        #  """
        #  scale_factor = scale_factor ** 1.5
#
        #  self.plot_plain(ax=ax, linewidth=linewidth, color=band_color)
#
        #  element_dict = self._sum_elements(elements=elements, orbitals=False)
#
        #  if color_list is None:
            #  color_dict = self.color_dict
        #  else:
            #  color_dict = {i: color for i, color in enumerate(color_list)}
#
        #  plot_element = {element: [] for element in elements}
        #  plot_band = []
        #  plot_wave_vec = []
#
        #  for band in element_dict:
            #  plot_band.extend(self.bands_dict[band])
            #  plot_wave_vec.extend(self.wave_vector)
            #  for element in elements:
                #  plot_element[element].extend(element_dict[band][element])
#
        #  for (i, element) in enumerate(elements):
            #  ax.scatter(
                #  plot_wave_vec,
                #  plot_band,
                #  c=color_dict[i],
                #  s=scale_factor * np.array(plot_element[element]),
                #  zorder=1,
            #  )
#
        #  if legend:
            #  legend_lines = []
            #  legend_labels = []
            #  for (i, element) in enumerate(elements):
                #  legend_lines.append(plt.Line2D(
                    #  [0],
                    #  [0],
                    #  marker='o',
                    #  markersize=2,
                    #  linestyle='',
                    #  color=color_dict[i])
                #  )
                #  legend_labels.append(
                    #  f'{element}'
                #  )
#
            #  leg = ax.get_legend()
#
            #  if leg is None:
                #  handles = legend_lines
                #  labels = legend_labels
            #  else:
                #  handles = [l._legmarker for l in leg.legendHandles]
                #  labels = [text._text for text in leg.texts]
                #  handles.extend(legend_lines)
                #  labels.extend(legend_labels)
#
            #  ax.legend(
                #  handles,
                #  labels,
                #  ncol=1,
                #  loc='upper left',
                #  fontsize=5,
                #  bbox_to_anchor=(1, 1),
                #  borderaxespad=0,
                #  frameon=False,
                #  handletextpad=0.1,
            #  )
#
    #  def plot_element_spd_old(self, ax, atoms, scale_factor=5, color_list=None, legend=True, linewidth=0.75, band_color='black'):
        #  """
        #  This function plots the projected band structure on the s, p, and d orbitals for each specified element in the calculated structure.
#
        #  Parameters:
            #  ax (matplotlib.pyplot.axis): Axis to plot the data on
            #  elements (list): List of element symbols to project onto
            #  order (list): This determines the order in which the points are plotted on the
                #  graph. This is an option because sometimes certain orbitals can be hidden
                #  under other orbitals because they have a larger weight. For example, if the
                #  signitures of the d orbitals are greater than that of the s orbitals, it
                #  might be smart to choose ['d', 'p', 's'] as the order so the s orbitals are
                #  plotted over the d orbitals.
            #  scale_factor (float): Factor to scale weights. This changes the size of the
                #  points in the scatter plot
            #  color_dict (dict[str][str]): This option allow the colors of the s, p, and d
                #  orbitals to be specified. Should be in the form of:
                #  {'s': <s color>, 'p': <p color>, 'd': <d color>}
            #  legend (bool): Determines if the legend should be included or not.
            #  linewidth (float): Line width of the plain band structure plotted in the background
            #  band_color (string): Color of the plain band structure
        #  """
        #  scale_factor = scale_factor ** 1.5
#
        #  self.plot_plain(ax=ax, linewidth=linewidth, color=band_color)
#
        #  element_dict = self._sum_elements(
            #  elements=elements, orbitals=True, spd=True)
#
        #  if color_dict is None:
            #  color_dict = {
                #  's': self.color_dict[0],
                #  'p': self.color_dict[1],
                #  'd': self.color_dict[2],
                #  'f': self.color_dict[4],
            #  }
#
        #  plot_element = {element: pd.DataFrame() for element in elements}
#
        #  if self.forbitals and 'f' not in order:
            #  order.append('f')
#
        #  plot_band = []
        #  plot_wave_vec = []
#
        #  for band in element_dict:
            #  plot_band.extend(self.bands_dict[band])
            #  plot_wave_vec.extend(self.wave_vector)
            #  for element in elements:
                #  plot_element[element] = plot_element[element].append(
                    #  element_dict[band][element])
#
        #  for (i, element) in enumerate(elements):
            #  if self.forbitals:
                #  electronic_structure = Element(
                    #  element).full_electronic_structure
                #  if not np.isin('f', electronic_structure):
                    #  order = order.remove('f')
            #  for orbital in order:
                #  ax.scatter(
                    #  plot_wave_vec,
                    #  plot_band,
                    #  c=color_dict[orbital],
                    #  s=scale_factor * plot_element[element][orbital],
                    #  zorder=1,
                #  )
#
        #  if legend:
            #  legend_lines = []
            #  legend_labels = []
            #  for element in elements:
                #  for orbital in order:
                    #  legend_lines.append(plt.Line2D(
                        #  [0],
                        #  [0],
                        #  marker='o',
                        #  markersize=2,
                        #  linestyle='',
                        #  color=color_dict[orbital])
                    #  )
                    #  legend_labels.append(
                        #  f'{element}(${orbital}$)'
                    #  )
#
            #  leg = ax.get_legend()
#
            #  if leg is None:
                #  handles = legend_lines
                #  labels = legend_labels
            #  else:
                #  handles = [l._legmarker for l in leg.legendHandles]
                #  labels = [text._text for text in leg.texts]
                #  handles.extend(legend_lines)
                #  labels.extend(legend_labels)
#
            #  ax.legend(
                #  handles,
                #  labels,
                #  ncol=1,
                #  loc='upper left',
                #  fontsize=5,
                #  bbox_to_anchor=(1, 1),
                #  borderaxespad=0,
                #  frameon=False,
                #  handletextpad=0.1,
            #  )
#
    #  def plot_element_orbitals_old(self, ax, element_orbital_pairs, scale_factor=5, color_list=None, legend=True, linewidth=0.75, band_color='black'):
        #  """
        #  this function plots the projected band structure on chosen orbitals for each specified element in the calculated structure.
#
        #  Parameters:
            #  ax (matplotlib.pyplot.axis): axis to plot the data on
            #  element_orbital_pairs (list[list]): List of list in the form of
                #  [[element symbol, orbital index], [element symbol, orbital_index], ...]
            #  scale_factor (float): factor to scale weights. this changes the size of the
                #  points in the scatter plot
            #  color_list (list): List of colors of the same length as the element_orbital_pairs
            #  legend (bool): determines if the legend should be included or not.
            #  linewidth (float): line width of the plain band structure plotted in the background
            #  band_color (string): color of the plain band structure
        #  """
        #  scale_factor = scale_factor ** 1.5
#
        #  self.plot_plain(ax=ax, linewidth=linewidth, color=band_color)
#
        #  elements = [i[0] for i in element_orbital_pairs]
#
        #  element_dict = self._sum_elements(elements=elements, orbitals=True)
#
        #  if color_list is None:
            #  color_dict = self.color_dict
        #  else:
            #  color_dict = {i: color for i, color in enumerate(color_list)}
#
        #  plot_element = {element: pd.DataFrame(
            #  columns=[range(9)]) for element in elements}
        #  plot_band = []
        #  plot_wave_vec = []
#
        #  for band in element_dict:
            #  plot_band.extend(self.bands_dict[band])
            #  plot_wave_vec.extend(self.wave_vector)
            #  for element in elements:
                #  plot_element[element] = plot_element[element].append(
                    #  element_dict[band][element])
#
        #  for i, element_orbital_pair in enumerate(element_orbital_pairs):
            #  element = element_orbital_pair[0]
            #  orbital = element_orbital_pair[1]
            #  ax.scatter(
                #  plot_wave_vec,
                #  plot_band,
                #  c=color_dict[i],
                #  s=scale_factor * plot_element[element][orbital],
                #  zorder=1,
            #  )
#
        #  if legend:
            #  legend_lines = []
            #  legend_labels = []
            #  for i, element_orbital_pair in enumerate(element_orbital_pairs):
                #  element = element_orbital_pair[0]
                #  orbital = element_orbital_pair[1]
                #  legend_lines.append(plt.Line2D(
                    #  [0],
                    #  [0],
                    #  marker='o',
                    #  markersize=2,
                    #  linestyle='',
                    #  color=color_dict[i])
                #  )
                #  legend_labels.append(
                    #  f'{element}({self.orbital_labels[orbital]})'
                #  )
#
            #  leg = ax.get_legend()
#
            #  if leg is None:
                #  handles = legend_lines
                #  labels = legend_labels
            #  else:
                #  handles = [l._legmarker for l in leg.legendHandles]
                #  labels = [text._text for text in leg.texts]
                #  handles.extend(legend_lines)
                #  labels.extend(legend_labels)
#
            #  ax.legend(
                #  handles,
                #  labels,
                #  ncol=1,
                #  loc='upper left',
                #  fontsize=5,
                #  bbox_to_anchor=(1, 1),
                #  borderaxespad=0,
                #  frameon=False,
                #  handletextpad=0.1,
            #  )
#
    #  def plot_atom_orbitals_old(self, ax, atom_orbital_pairs, scale_factor=5, color_list=None, legend=True, linewidth=0.75, band_color='black'):
        #  """
        #  This function plots the projected band structure of individual orbitals on a given axis.
#
        #  Parameters:
            #  ax (matplotlib.pyplot.axis): Axis to plot the data on
            #  atom_orbital_pairs (list[list]): Selected orbitals on selected atoms to plot.
                #  This should take the form of [[atom index, orbital_index], ...].
                #  To plot the px orbital of the 1st atom and the pz orbital of the 2nd atom
                #  in the POSCAR file, the input would be [[0, 3], [1, 2]]
            #  scale_factor (float): Factor to scale weights. This changes the size of the
                #  points in the scatter plot
            #  color_list (list): List of colors of the same length as the atom_orbital_pairs
            #  legend (bool): Determines if the legend should be included or not.
            #  linewidth (float): Line width of the plain band structure plotted in the background
            #  band_color (string): Color of the plain band structure
        #  """
        #  scale_factor = scale_factor ** 1.5
#
        #  self.plot_plain(ax=ax, linewidth=linewidth, color=band_color)
#
        #  projected_dict = self.projected_dict
        #  wave_vector = self.wave_vector
#
        #  if color_list is None:
            #  color_dict = self.color_dict
        #  else:
            #  color_dict = {i: color for i, color in enumerate(color_list)}
#
        #  for band in projected_dict:
            #  for (i, atom_orbital_pair) in enumerate(atom_orbital_pairs):
                #  atom = atom_orbital_pair[0]
                #  orbital = atom_orbital_pair[1]
#
                #  ax.scatter(
                    #  wave_vector,
                    #  self.bands_dict[band],
                    #  c=color_dict[i],
                    #  s=scale_factor * projected_dict[band][atom][orbital],
                    #  zorder=1,
                #  )
#
        #  if legend:
            #  legend_lines = []
            #  legend_labels = []
            #  for (i, atom_orbital_pair) in enumerate(atom_orbital_pairs):
                #  atom = atom_orbital_pair[0]
                #  orbital = atom_orbital_pair[1]
#
                #  legend_lines.append(plt.Line2D(
                    #  [0],
                    #  [0],
                    #  marker='o',
                    #  markersize=2,
                    #  linestyle='',
                    #  color=color_dict[i])
                #  )
                #  legend_labels.append(
                    #  f'{atom}({self.orbital_labels[atom_orbital_pair[1]]})'
                #  )
#
            #  ax.legend(
                #  legend_lines,
                #  legend_labels,
                #  ncol=1,
                #  loc='upper left',
                #  fontsize=5,
                #  bbox_to_anchor=(1, 1),
                #  borderaxespad=0,
                #  frameon=False,
                #  handletextpad=0.1,
            #  )
