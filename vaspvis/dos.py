import matplotlib.pyplot as plt
from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.electronic_structure.core import Spin, Orbital
from pymatgen.electronic_structure.dos import Dos
from scipy.ndimage.filters import gaussian_filter1d
from scipy.ndimage import gaussian_filter
from functools import reduce
import numpy as np
import pandas as pd
import time


class Dos:

    def __init__(self, folder, spin='up'):
        self.folder = folder
        self.spin = spin
        self.forbitals = False
        self.vasprun = Vasprun(
            f'{folder}/vasprun.xml',
            parse_dos=True,
            parse_eigen=False,
            parse_potcar_file=False
        )
        self.poscar = Poscar.from_file(
            f'{folder}/POSCAR',
            check_for_POTCAR=False,
            read_velocities=False
        )
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
        self.spin_dict = {'up': Spin.up, 'down': Spin.down}
        self.tdos_dict = self._load_tdos()
        self.pdos_dict = self._load_pdos()

    def _load_tdos(self):
        """
        This function loads the total density of states into a dictionary

        Outputs:
        ----------
        tdos_dict: (dict[str][np.ndarray]) Dictionary that consists or the
            energies and densities of the system.
        """

        tdos = self.vasprun.tdos
        tdos_dict = {
            'energy': np.array(tdos.energies - tdos.efermi),
            'density': np.array(tdos.densities[self.spin_dict[self.spin]])
        }

        return tdos_dict

    def _load_pdos(self):
        """
        This function loads the projected density of states into a dictionary
        of the form:
        atom index --> orbital projections

        Outputs:
        ----------
        pdos_dict: (dict[int][pd.DataFrame]) Dictionary that contains a data frame
            with the orbital weights for each atom index.
        """

        pdos = self.vasprun.pdos
        pdos_dict = {i: [] for i in range(len(pdos))}
        spin = self.spin_dict[self.spin]

        for (i, atom) in enumerate(pdos):
            new_dict = {
                i: atom[orbital][spin] for (i, orbital) in enumerate(atom)
            }

            if len(list(new_dict.keys())) == 16:
                self.forbitals = True

            df = pd.DataFrame.from_dict(new_dict)
            pdos_dict[i] = df

        return pdos_dict

    def _smear(self, dos, sigma):
        """
        This function applied a 1D gaussian filter to the density of states

        Inputs:
        ----------
        dos: (np.ndarray) Array of densities.
        sigma: (float) Standard deviation used in the gaussian filter.


        Outputs:
        ----------
        _smeared_dos: (np.ndarray) Array of _smeared densities.
        """

        diff = np.diff(self.tdos_dict['energy'])
        avgdiff = np.mean(diff)
        _smeared_dos = gaussian_filter1d(dos, sigma / avgdiff)

        return _smeared_dos

    def _sum_orbitals(self):
        """
        This function sums the weights of all orbitals for each atom
        and creates a Dataframe containing the projected densities:

        Outputs:
        ----------
        orbital_df: (pd.DataFrame) Dataframe that has the summed densities of
            each orbital for all atoms
        """

        atom_list = [self.pdos_dict[atom] for atom in self.pdos_dict]
        orbital_df = reduce(
            lambda x, y: x.add(y, fill_value=0), atom_list
        )

        return orbital_df

    def _sum_atoms(self):
        """
        This function sums all the orbitals for each atom and returns a Dataframe
        of atoms with their projected densities.

        Outputs:
        atom_df: (pd.DataFrame) Dataframe containing the projected densities for
            each atom.
        """

        pdos_dict = self.pdos_dict
        atom_df = pd.concat(
            [pdos_dict[atom].sum(axis=1) for atom in pdos_dict],
            axis=1
        )

        return atom_df

    def _sum_elements(self, elements, orbitals=False, spd=False):
        """
        This function sums the weights of the orbitals of specific elements within the
        calculated structure and returns a dictionary of the form:
        element label --> orbital weights for orbitals = True
        element label for orbitals = False
        This is useful for structures with many elements because manually entering indicies is
        not practical for large structures.

        Inputs:
        ----------
        elements: (list) List of element symbols to sum the weights of.
        orbitals: (bool) Determines whether or not to inclue orbitals or not
            (True = keep orbitals, False = sum orbitals together )
        spd: (bool) Determines whether or not to sum the s, p, and d orbitals


        Outputs:
        ----------
        element_dict: (dict([str][str][pd.DataFrame])) Dictionary that contains the summed
            weights for each orbital for a given element in the structure.
        """

        poscar = self.poscar
        natoms = poscar.natoms
        symbols = poscar.site_symbols

        element_list = np.hstack(
            [[symbols[i] for j in range(natoms[i])]
             for i in range(len(symbols))]
        )

        element_dict = {element: [] for element in elements}

        for element in elements:
            element_index = np.where(element_list == element)[0]
            df = pd.concat(
                [self.pdos_dict[i] for i in element_index],
                axis=1
            )

            if orbitals:
                element_dict[element] = df.groupby(
                    by=df.columns,
                    axis=1
                ).sum()
                if spd:
                    df = element_dict[element]
                    element_dict[element]['s'] = df[0]
                    element_dict[element]['p'] = df[1] + \
                        df[2] + df[3]
                    element_dict[element]['d'] = df[4] + \
                        df[5] + df[6] + df[7] + df[8]

                    if self.forbitals:
                        element_dict[element]['f'] = df[9] + df[10] + \
                            df[11] + df[12] + df[13] + df[14] + df[15]
                        element_dict[element] = element_dict[element].drop(
                            columns=range(16))
                    else:
                        element_dict[element] = element_dict[element].drop(
                            columns=range(9))

            else:
                element_dict[element] = df.sum(axis=1).tolist()

        return element_dict

    def _sum_spd(self):
        """
        This function sums the weights of the s, p, and d orbitals for each atom
        and creates a Dataframe containing the projected densities:

        Outputs:
        ----------
        spd_df: (pd.DataFrame) Dataframe that has the summed densities of the
            s, p, and d orbitals across all atoms.
        """

        spd_df = self._sum_orbitals()

        spd_df['s'] = spd_df[0]
        spd_df['p'] = spd_df[1] + spd_df[2] + spd_df[3]
        spd_df['d'] = spd_df[4] + spd_df[5] + spd_df[6] + spd_df[7] + spd_df[8]

        if self.forbitals:
            spd_df['f'] = spd_df[9] + spd_df[10] + \
                spd_df[11] + spd_df[12] + spd_df[13] + spd_df[14] + spd_df[15]
            spd_df = spd_df.drop(columns=range(16))
        else:
            spd_df = spd_df.drop(columns=range(9))

        return spd_df

    def plot_plain(self, ax, linewidth=1.5, fill=True, alpha=0.3, sigma=0.05, energyaxis='y', color='black'):
        """
        This function plots the total density of states

        Inputs:
        -----------
        ax: (matplotlib.pyplot.axis) Axis to append the tick labels
        fill: (bool) Determines wether or not to fill underneath the plot
        alpha: (float) Alpha value for the fill
        linewidth: (float) Linewidth of lines
        sigma: (float) Standard deviation for gaussian filter
        energyaxis: (str) Determines the axis to plot the energy on ('x' or 'y')
        """

        tdos_dict = self.tdos_dict

        if sigma > 0:
            tdensity = self._smear(
                tdos_dict['density'],
                sigma=sigma
            )
        else:
            tdensity = tdos_dict['density']

        if energyaxis == 'y':
            ax.plot(
                tdensity,
                tdos_dict['energy'],
                linewidth=linewidth,
                color=color,
            )

            if fill:
                ax.fill_betweenx(
                    tdos_dict['energy'],
                    tdensity,
                    0,
                    alpha=alpha,
                    color=color,
                )

        if energyaxis == 'x':
            ax.plot(
                tdos_dict['energy'],
                tdensity,
                linewidth=linewidth,
                color=color,
            )

            if fill:
                ax.fill_between(
                    tdos_dict['energy'],
                    tdensity,
                    0,
                    color=color,
                    alpha=alpha,
                )

    def plot_spd(self, ax, order=['s', 'p', 'd'], fill=True, alpha=0.3, linewidth=1.5, sigma=0.05, energyaxis='y', color_dict=None, legend=True, total=True):
        """
        This function plots the total density of states with the projected
        density of states for the total projections of the s, p, and d orbitals.

        Inputs:
        ----------
        ax: (matplotlib.pyplot.axis) Axis to plot on
        order: (list) Order to plot the projected bands in. This feature helps to
            avoid situations where one projection completely convers the other.
        fill: (bool) Determines wether or not to fill underneath the plot
        alpha: (float) Alpha value for the fill
        linewidth: (float) Linewidth of lines
        sigma: (float) Standard deviation for gaussian filter
        energyaxis: (str) Determines the axis to plot the energy on ('x' or 'y')
        color_dict: (dict[str][str]) This option allow the colors of the s, p, and d
            orbitals to be specified. Should be in the form of:
            {'s': <s color>, 'p': <p color>, 'd': <d color>}
        """

        spd_df = self._sum_spd()
        tdos_dict = self.tdos_dict

        if color_dict is None:
            color_dict = {
                's': self.color_dict[0],
                'p': self.color_dict[1],
                'd': self.color_dict[2],
                'f': self.color_dict[4],
            }

        if self.forbitals and 'f' not in order:
            order.append('f')

        if total:
            self.plot_plain(
                ax=ax,
                linewidth=linewidth,
                fill=fill,
                alpha=alpha,
                sigma=sigma,
                energyaxis=energyaxis,
            )

        for orbital in order:
            if sigma > 0:
                pdensity = self._smear(
                    spd_df[orbital],
                    sigma=sigma
                )
            else:
                pdensity = spd_df[orbital]

            if energyaxis == 'y':
                ax.plot(
                    pdensity,
                    tdos_dict['energy'],
                    color=color_dict[orbital],
                    linewidth=linewidth,
                    label=f'${orbital}$',
                )

                if fill:
                    ax.fill_betweenx(
                        tdos_dict['energy'],
                        pdensity,
                        0,
                        color=color_dict[orbital],
                        alpha=alpha,
                    )

            if energyaxis == 'x':
                ax.plot(
                    tdos_dict['energy'],
                    pdensity,
                    color=color_dict[orbital],
                    linewidth=linewidth,
                    label=f'${orbital}$',
                )

                if fill:
                    ax.fill_between(
                        tdos_dict['energy'],
                        pdensity,
                        0,
                        color=color_dict[orbital],
                        alpha=alpha,
                    )

        if legend:
            legend_lines = []
            legend_labels = []
            for i, orbital in enumerate(order):
                legend_lines.append(plt.Line2D(
                    [0],
                    [0],
                    marker='o',
                    markersize=2,
                    linestyle='',
                    color=color_dict[orbital])
                )
                legend_labels.append(
                    f'${orbital}$'
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

    def plot_atom_orbitals(self, ax, atom_orbital_pairs, fill=True, alpha=0.3, linewidth=1.5, sigma=0.05, energyaxis='y', color_dict=None, legend=True, total=True):
        """
        This function plots the total density of states with the projected
        density of states for the total projections of the s, p, and d orbitals.

        Inputs:
        ----------
        ax: (matplotlib.pyplot.axis) Axis to plot on
        order: (list) Order to plot the projected bands in. This feature helps to
            avoid situations where one projection completely convers the other.
        fill: (bool) Determines wether or not to fill underneath the plot
        alpha: (float) Alpha value for the fill
        linewidth: (float) Linewidth of lines
        sigma: (float) Standard deviation for gaussian filter
        energyaxis: (str) Determines the axis to plot the energy on ('x' or 'y')
        color_dict: (dict[int][str]) Dictionary of colors for the atom-orbital pairs in       
            the order that the atom-orbital pairs were given.
        """

        tdos_dict = self.tdos_dict

        if color_dict is None:
            color_dict = self.color_dict

        if total:
            self.plot_plain(
                ax=ax,
                linewidth=linewidth,
                fill=fill,
                alpha=alpha,
                sigma=sigma,
                energyaxis=energyaxis,
            )

        for i, atom_orbital_pair in enumerate(atom_orbital_pairs):
            atom = atom_orbital_pair[0]
            orbital = atom_orbital_pair[1]

            if sigma > 0:
                pdensity = self._smear(
                    self.pdos_dict[atom][orbital],
                    sigma=sigma
                )
            else:
                pdensity = self.pdos_dict[atom][orbital]

            if energyaxis == 'y':
                ax.plot(
                    pdensity,
                    tdos_dict['energy'],
                    color=color_dict[i],
                    linewidth=linewidth,
                    label=f'${atom}({self.orbital_labels[orbital]})$',
                )

                if fill:
                    ax.fill_betweenx(
                        tdos_dict['energy'],
                        pdensity,
                        0,
                        color=color_dict[i],
                        alpha=alpha,
                    )

            if energyaxis == 'x':
                ax.plot(
                    tdos_dict['energy'],
                    pdensity,
                    color=color_dict[i],
                    linewidth=linewidth,
                    label=f'${atom}({self.orbital_labels[orbital]})$',
                )

                if fill:
                    ax.fill_between(
                        tdos_dict['energy'],
                        pdensity,
                        0,
                        color=color_dict[i],
                        alpha=alpha,
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

    def plot_orbitals(self, ax, orbitals, fill=True, alpha=0.3, linewidth=1.5, sigma=0.05, energyaxis='y', color_dict=None, legend=True, total=True):
        """
        This function plots the total density of states with the projected
        density of states for the total projections of the s, p, and d orbitals.

        Inputs:
        ----------
        ax: (matplotlib.pyplot.axis) Axis to plot on
        orbitals: (list) List of orbitals to project onto
        fill: (bool) Determines wether or not to fill underneath the plot
        alpha: (float) Alpha value for the fill
        linewidth: (float) Linewidth of lines
        sigma: (float) Standard deviation for gaussian filter
        energyaxis: (str) Determines the axis to plot the energy on ('x' or 'y')
        color_dict: (dict[str][str]) This option allow the colors of each orbital
            specified. Should be in the form of:
            {'orbital index': <color>, 'orbital index': <color>, ...}
        """

        orbital_df = self._sum_orbitals()
        tdos_dict = self.tdos_dict

        if color_dict is None:
            color_dict = self.color_dict

        if total:
            self.plot_plain(
                ax=ax,
                linewidth=linewidth,
                fill=fill,
                alpha=alpha,
                sigma=sigma,
                energyaxis=energyaxis,
            )

        for i, orbital in enumerate(orbitals):
            if sigma > 0:
                pdensity = self._smear(
                    orbital_df[orbital],
                    sigma=sigma
                )
            else:
                pdensity = orbital_df[orbital]

            if energyaxis == 'y':
                ax.plot(
                    pdensity,
                    tdos_dict['energy'],
                    color=color_dict[i],
                    linewidth=linewidth,
                )

                if fill:
                    ax.fill_betweenx(
                        tdos_dict['energy'],
                        pdensity,
                        0,
                        color=color_dict[i],
                        alpha=alpha,
                    )

            if energyaxis == 'x':
                ax.plot(
                    tdos_dict['energy'],
                    pdensity,
                    color=color_dict[i],
                    linewidth=linewidth,
                )

                if fill:
                    ax.fill_between(
                        tdos_dict['energy'],
                        pdensity,
                        0,
                        color=color_dict[i],
                        alpha=alpha,
                    )

        if legend:
            legend_lines = []
            legend_labels = []
            for i, orbital in enumerate(orbitals):
                legend_lines.append(plt.Line2D(
                    [0],
                    [0],
                    marker='o',
                    markersize=2,
                    linestyle='',
                    color=color_dict[i])
                )
                legend_labels.append(
                    f'{self.orbital_labels[orbital]}'
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

    def plot_atoms(self, ax, atoms, fill=True, alpha=0.3, linewidth=1.5, sigma=0.05, energyaxis='y', color_dict=None, legend=True, total=True):
        """
        This function plots the total density of states with the projected
        density of states on the given atoms.

        Inputs:
        ----------
        ax: (matplotlib.pyplot.axis) Axis to plot on
        atoms: (list) Index of atoms atom plot
        fill: (bool) Determines wether or not to fill underneath the plot
        alpha: (float) Alpha value for the fill
        color_dict: (dict[int][str]) Optional dictionary of colors for each atom
        linewidth: (float) Linewidth of lines
        sigma: (float) Standard deviation for gaussian filter
        energyaxis: (str) Determines the axis to plot the energy on ('x' or 'y')
        color_dict: (dict[str][str]) This option allow the colors of each atom
            specified. Should be in the form of:
            {'atom index': <color>, 'atom index': <color>, ...}
        """

        atom_df = self._sum_atoms()
        tdos_dict = self.tdos_dict

        if color_dict is None:
            color_dict = self.color_dict

        if total:
            self.plot_plain(
                ax=ax,
                linewidth=linewidth,
                fill=fill,
                alpha=alpha,
                sigma=sigma,
                energyaxis=energyaxis,
            )

        for i, atom in enumerate(atoms):
            if sigma > 0:
                pdensity = self._smear(
                    atom_df[atom],
                    sigma=sigma
                )
            else:
                pdensity = atom_df[atom]

            if energyaxis == 'y':
                ax.plot(
                    pdensity,
                    tdos_dict['energy'],
                    color=color_dict[i],
                    linewidth=linewidth,
                    label=atom,
                )

                if fill:
                    ax.fill_betweenx(
                        tdos_dict['energy'],
                        pdensity,
                        0,
                        color=color_dict[i],
                        alpha=alpha,
                    )

            elif energyaxis == 'x':
                ax.plot(
                    tdos_dict['energy'],
                    pdensity,
                    color=color_dict[i],
                    linewidth=linewidth,
                    label=atom,
                )

                if fill:
                    ax.fill_between(
                        tdos_dict['energy'],
                        pdensity,
                        0,
                        color=color_dict[i],
                        alpha=alpha,
                    )

        if legend:
            legend_lines = []
            legend_labels = []
            for i, atom in enumerate(atoms):
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

    def plot_elements(self, ax, elements, fill=True, alpha=0.3, linewidth=1.5, sigma=0.05, energyaxis='y', color_dict=None, legend=True, total=True):
        """
        This function plots the total density of states with the projected
        density of states for the projection onto specified elements. This is 
        useful for supercells where there are many atoms of the same element and
        it is inconvienient to manually list each index in the POSCAR.

        Inputs:
        ----------
        ax: (matplotlib.pyplot.axis) Axis to plot on
        elements: (list) List of element symbols to project onto
        fill: (bool) Determines wether or not to fill underneath the plot
        alpha: (float) Alpha value for the fill
        linewidth: (float) Linewidth of lines
        sigma: (float) Standard deviation for gaussian filter
        energyaxis: (str) Determines the axis to plot the energy on ('x' or 'y')
        color_dict: (dict[str][str]) This option allow the colors of each element
            specified. Should be in the form of:
            {'element index': <color>, 'element index': <color>, ...}
        """

        element_dict = self._sum_elements(
            elements=elements, orbitals=False, spd=False)
        tdos_dict = self.tdos_dict

        if color_dict is None:
            color_dict = self.color_dict

        if total:
            self.plot_plain(
                ax=ax,
                linewidth=linewidth,
                fill=fill,
                alpha=alpha,
                sigma=sigma,
                energyaxis=energyaxis,
            )

        for i, element in enumerate(elements):
            if sigma > 0:
                pdensity = self._smear(
                    element_dict[element],
                    sigma=sigma
                )
            else:
                pdensity = element_dict[element]

            if energyaxis == 'y':
                ax.plot(
                    pdensity,
                    tdos_dict['energy'],
                    color=color_dict[i],
                    linewidth=linewidth,
                    label=f'{element}',
                )

                if fill:
                    ax.fill_betweenx(
                        tdos_dict['energy'],
                        pdensity,
                        0,
                        color=color_dict[i],
                        alpha=alpha,
                    )

            if energyaxis == 'x':
                ax.plot(
                    tdos_dict['energy'],
                    pdensity,
                    color=color_dict[i],
                    linewidth=linewidth,
                    label=f'{element}',
                )

                if fill:
                    ax.fill_between(
                        tdos_dict['energy'],
                        pdensity,
                        0,
                        color=color_dict[i],
                        alpha=alpha,
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

    def plot_element_orbitals(self, ax, element_orbital_pairs, fill=True, alpha=0.3, linewidth=1.5, sigma=0.05, energyaxis='y', color_dict=None, legend=True, total=True):
        """
        This function plots the total density of states with the projected
        density of states onto the chosen orbitals of specified elements. This is 
        useful for supercells where there are many atoms of the same element and
        it is inconvienient to manually list each index in the POSCAR.

        Inputs:
        ----------
        ax: (matplotlib.pyplot.axis) Axis to plot on
        elements: (list) List of element symbols to project onto
        orbitals: (list) List of orbitals to project onto
        fill: (bool) Determines wether or not to fill underneath the plot
        alpha: (float) Alpha value for the fill
        linewidth: (float) Linewidth of lines
        sigma: (float) Standard deviation for gaussian filter
        energyaxis: (str) Determines the axis to plot the energy on ('x' or 'y')
        color_dict: (dict[str][str]) This option allow the colors of each element
            specified. Should be in the form of:
            {'element index': <color>, 'element index': <color>, ...}
        """

        elements = [i[0] for i in element_orbital_pairs]

        element_dict = self._sum_elements(
            elements=elements, orbitals=True, spd=False)
        tdos_dict = self.tdos_dict

        if color_dict is None:
            color_dict = self.color_dict

        if total:
            self.plot_plain(
                ax=ax,
                linewidth=linewidth,
                fill=fill,
                alpha=alpha,
                sigma=sigma,
                energyaxis=energyaxis,
            )

        for (i, element_orbital_pair) in enumerate(element_orbital_pairs):
            element = element_orbital_pair[0]
            orbital = element_orbital_pair[1]
            if sigma > 0:
                pdensity = self._smear(
                    element_dict[element][orbital],
                    sigma=sigma
                )
            else:
                pdensity = element_dict[element][orbital]

            if energyaxis == 'y':
                ax.plot(
                    pdensity,
                    tdos_dict['energy'],
                    color=color_dict[i],
                    linewidth=linewidth,
                    label=f'{element}({self.orbital_labels[orbital]})',
                )

                if fill:
                    ax.fill_betweenx(
                        tdos_dict['energy'],
                        pdensity,
                        0,
                        color=color_dict[i],
                        alpha=alpha,
                    )

            if energyaxis == 'x':
                ax.plot(
                    tdos_dict['energy'],
                    pdensity,
                    color=color_dict[i],
                    linewidth=linewidth,
                    label=f'{element}({self.orbital_labels[orbital]})',
                )

                if fill:
                    ax.fill_between(
                        tdos_dict['energy'],
                        pdensity,
                        0,
                        color=color_dict[i],
                        alpha=alpha,
                    )

        if legend:
            legend_lines = []
            legend_labels = []
            for (i, element_orbital_pair) in enumerate(element_orbital_pairs):
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

    def plot_element_spd(self, ax, elements, order=['s', 'p', 'd'], fill=True, alpha=0.3, linewidth=1.5, sigma=0.05, energyaxis='y', color_dict=None, legend=True, total=True):
        """
        This function plots the total density of states with the projected
        density of states onto the s, p, and d orbitals of specified elements. 
        This is useful for supercells where there are many atoms of the same 
        element and it is inconvienient to manually list each index in the POSCAR.

        Inputs:
        ----------
        ax: (matplotlib.pyplot.axis) Axis to plot on
        elements: (list) List of element symbols to project onto
        order: (list) Order to plot the projected bands in. This feature helps to
            avoid situations where one projection completely convers the other.
        fill: (bool) Determines wether or not to fill underneath the plot
        alpha: (float) Alpha value for the fill
        linewidth: (float) Linewidth of lines
        sigma: (float) Standard deviation for gaussian filter
        energyaxis: (str) Determines the axis to plot the energy on ('x' or 'y')
        color_dict: (dict[str][str]) This option allow the colors of each element
            specified. Should be in the form of:
            {'element index': <color>, 'element index': <color>, ...}
        """

        element_dict = self._sum_elements(
            elements=elements, orbitals=True, spd=True)
        tdos_dict = self.tdos_dict

        if color_dict is None:
            color_dict = {
                's': self.color_dict[0],
                'p': self.color_dict[1],
                'd': self.color_dict[2],
                'f': self.color_dict[4],
            }

        if self.forbitals and 'f' not in order:
            order.append('f')

        if total:
            self.plot_plain(
                ax=ax,
                linewidth=linewidth,
                fill=fill,
                alpha=alpha,
                sigma=sigma,
                energyaxis=energyaxis,
            )

        for element in elements:
            for i, orbital in enumerate(order):
                if sigma > 0:
                    pdensity = self._smear(
                        element_dict[element][orbital],
                        sigma=sigma
                    )
                else:
                    pdensity = element_dict[element][orbital]

                if energyaxis == 'y':
                    ax.plot(
                        pdensity,
                        tdos_dict['energy'],
                        color=color_dict[orbital],
                        linewidth=linewidth,
                        label=f'{element}({orbital})',
                    )

                    if fill:
                        ax.fill_betweenx(
                            tdos_dict['energy'],
                            pdensity,
                            0,
                            color=color_dict[orbital],
                            alpha=alpha,
                        )

                if energyaxis == 'x':
                    ax.plot(
                        tdos_dict['energy'],
                        pdensity,
                        color=color_dict[orbital],
                        linewidth=linewidth,
                        label=f'{element}({orbital})',
                    )

                    if fill:
                        ax.fill_between(
                            tdos_dict['energy'],
                            pdensity,
                            0,
                            color=color_dict[orbital],
                            alpha=alpha,
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

    def plot_layers(self, ax, ylim=[-6, 6], cmap='magma', sigma=5, energyaxis='y'):
        """
        This function plots a layer by layer heat map of the density
        of states.

        Inputs:
        ----------
        ax: (matplotlib.pyplot.axis) Axis to plot on
        ylim: (list) Upper and lower energy bounds for the plot.
        cmap: (str) Color map to use in the heat map
        sigma: (float) Sigma parameter for the _smearing of the heat map.
        energyaxis: (str) Axis to plot the energy on. ('x' or 'y')
        """

        poscar = self.poscar
        sites = poscar.structure.sites
        zvals = np.array([site.c for site in sites])
        zorder = np.argsort(zvals)
        energy = self.tdos_dict['energy']

        ind = np.where((ylim[0] - 0.1 <= energy) & (energy <= ylim[-1] + 0.1))
        atom_index = range(len(zorder))
        energies = energy[ind]
        densities = self._sum_atoms().to_numpy()[ind]
        densities = np.transpose(densities)
        densities = np.transpose(densities[zorder])
        densities = gaussian_filter(densities, sigma=sigma)

        if energyaxis == 'y':
            im = ax.pcolormesh(
                atom_index,
                energies,
                densities,
                cmap=cmap,
                shading='gouraud',
                vmax=0.6,
            )

        if energyaxis == 'x':
            im = ax.pcolormesh(
                energies,
                atom_index,
                np.transpose(densities),
                cmap=cmap,
                shading='gouraud',
                vmax=0.6,
            )

        fig = plt.gcf()
        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.tick_params(labelsize=6)
        cbar.set_label('Density of States', fontsize=6)


def main():
    fig = plt.figure(figsize=(4, 8), dpi=200)
    ax = fig.add_subplot(111)
    plt.ylim(-6, 6)
    ax.margins(x=0.005, y=0.005)
    dos = Dos(folder='../../vaspvis_data/dos')
    dos.plot_spd(ax=ax, energyaxis='y')
    # dos.plot_layers(ax=ax)
    # dos.plot_atoms(ax=ax, atoms=[0, 1], sigma=0.1, fill=True, energyaxis='y')
    plt.show()


if __name__ == '__main__':
    main()
