import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.electronic_structure.core import Spin, Orbital
from pymatgen.electronic_structure.dos import Dos
from scipy.ndimage.filters import gaussian_filter1d
from scipy.ndimage import gaussian_filter
from functools import reduce
import numpy as np
import pandas as pd
from ase.visualize.plot import plot_atoms
from pymatgen.io.ase import AseAtomsAdaptor
import copy
import time
import os


class Dos:
    """
    This class contains all the methods for contructing density of states plots from the outputs of VASP calculations.

    Parameters:
        folder (str): This is the folder that contains the VASP files.
        spin (str): Which spin direction to parse ('up' or 'down')
        combination_method (str): If the spin option is 'both', the combination method can either be additive or substractive
            by passing 'add' or 'sub'. It spin is passed as 'up' or 'down' this option is ignored.
    """

    def __init__(self, folder, spin='up', combination_method="add"):
        self.folder = folder
        self.spin = spin
        self.combination_method = combination_method
        self.forbitals = False
        self.vasprun = Vasprun(
            os.path.join(folder, 'vasprun.xml'),
            parse_dos=True,
            parse_eigen=False,
            parse_potcar_file=False
        )
        self.poscar = Poscar.from_file(
            os.path.join(folder, 'POSCAR'),
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

        Returns:
            tdos_dict (dict[str][np.ndarray]): Dictionary that consists or the
                energies and densities of the system.
        """

        if self.spin == 'up':
            spin_factor = 1
        elif self.spin == 'down':
            spin_factor = -1

        tdos = self.vasprun.tdos

        if self.spin == 'up' or self.spin == 'down':
            tdos_dict = {
                'energy': np.array(tdos.energies - tdos.efermi),
                'density': spin_factor * np.array(tdos.densities[self.spin_dict[self.spin]])
            }
        elif self.spin == 'both':
            if self.combination_method == "add":
                tdos_dict = {
                    'energy': np.array(tdos.energies - tdos.efermi),
                    'density': np.array(tdos.densities[Spin.up]) + np.array(tdos.densities[Spin.down])
                }
            if self.combination_method == "sub":
                tdos_dict = {
                    'energy': np.array(tdos.energies - tdos.efermi),
                    'density': np.array(tdos.densities[Spin.up]) - np.array(tdos.densities[Spin.down])
                }

        return tdos_dict

    def _load_pdos(self):
        """
        This function loads the projected density of states into a dictionary
        of the form:
        atom index --> orbital projections

        Returns:
            pdos_dict (dict[int][pd.DataFrame]): Dictionary that contains a data frame
                with the orbital weights for each atom index.
        """

        if self.spin == 'up':
            spin_factor = 1
        elif self.spin == 'down':
            spin_factor = -1

        pdos = self.vasprun.pdos
        pdos_dict = {i: [] for i in range(len(pdos))}

        if self.spin == 'up' or self.spin == 'down':
            spin = self.spin_dict[self.spin]
            for (i, atom) in enumerate(pdos):
                new_dict = {
                    i: spin_factor * atom[orbital][spin] for (i, orbital) in enumerate(atom)
                }

                if len(list(new_dict.keys())) == 16:
                    self.forbitals = True

                df = pd.DataFrame.from_dict(new_dict)
                pdos_dict[i] = df

        elif self.spin == 'both':
            for (i, atom) in enumerate(pdos):
                new_dict = {
                    i: np.array(atom[orbital][Spin.up]) + np.array(atom[orbital][Spin.down]) for (i, orbital) in enumerate(atom)
                }

                if len(list(new_dict.keys())) == 16:
                    self.forbitals = True

                df = pd.DataFrame.from_dict(new_dict)
                pdos_dict[i] = df

        return pdos_dict

    def _smear(self, dos, sigma):
        """
        This function applied a 1D gaussian filter to the density of states

        Parameters:
            dos (np.ndarray): Array of densities.
            sigma (float): Standard deviation used in the gaussian filter.


        Returns:
            _smeared_dos (np.ndarray): Array of _smeared densities.
        """

        diff = np.diff(self.tdos_dict['energy'])
        avgdiff = np.mean(diff)
        _smeared_dos = gaussian_filter1d(dos, sigma / avgdiff)

        return _smeared_dos

    def _sum_orbitals(self):
        """
        This function sums the weights of all orbitals for each atom
        and creates a Dataframe containing the projected densities:

        Returns:
            orbital_df (pd.DataFrame): Dataframe that has the summed densities of
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

        Returns:
            atom_df (pd.DataFrame): Dataframe containing the projected densities for
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

        Parameters:
            elements (list): List of element symbols to sum the weights of.
            orbitals (bool): Determines whether or not to inclue orbitals or not
                (True = keep orbitals, False = sum orbitals together )
            spd (bool): Determines whether or not to sum the s, p, and d orbitals


        Returns:
            element_dict (dict([str][str][pd.DataFrame])): Dictionary that contains the summed
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
            element_index = np.where(np.isin(element_list, element))[0]
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

        Returns:
            spd_df (pd.DataFrame): Dataframe that has the summed densities of the
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

    def _set_density_lims(self, ax, tdensity, tenergy, erange, energyaxis, spin, partial=False, is_dict=False, idx=None, multiple=False):
        energy_in_plot_index = np.where(
            (tenergy > erange[0]) & (tenergy < erange[1])
        )[0]

        if partial:
            if is_dict:
                if multiple:
                    total = []
                    for i in idx:
                        first = i[0]
                        second = i[1]
                        total.append(tdensity[first][second][energy_in_plot_index])

                    if spin == 'up' or spin == 'both':
                        density_in_plot = total[np.argmax(np.max(total, axis=1))]
                    elif spin == 'down':
                        density_in_plot = total[np.argmin(np.min(total, axis=1))]
                else:
                    columns = []
                    values = []
                    dfs = []
                    for i in idx:
                        selected_densities = tdensity[i].iloc[energy_in_plot_index]
                        dfs.append(selected_densities)
                        if spin == 'up' or spin == 'both':
                            max_value_column = selected_densities.max().idxmax() 
                            max_value = selected_densities[max_value_column].max().max()
                            columns.append(max_value_column)
                            values.append(max_value)
                        elif spin == 'down':
                            min_value_column = selected_densities.min().idxmin() 
                            min_value = selected_densities[min_value_column].min().min()
                            columns.append(min_value_column)
                            values.append(min_value)

                    if spin=='up' or spin=='both':
                        max_column = columns[np.argmax(values)]
                        max_df = dfs[np.argmax(values)]
                        density_in_plot = max_df[max_column].__array__()
                    elif spin=='down':
                        min_column = columns[np.argmin(values)]
                        min_df = dfs[np.argmin(values)]
                        density_in_plot = min_df[min_column].__array__()

            else:
                selected_densities = tdensity.iloc[energy_in_plot_index]
                if spin=='up' or spin=='both':
                    density_in_plot = selected_densities[selected_densities.max().idxmax()].__array__()
                elif spin=='down':
                    density_in_plot = selected_densities[selected_densities.min().idxmax()].__array__()

        else:
            density_in_plot = tdensity[energy_in_plot_index]

        if len(ax.lines) == 0:
            if energyaxis == 'y':
                ax.set_ylim(erange)
                if spin == 'up' or spin == 'both':
                    ax.set_xlim(0, np.max(density_in_plot) * 1.1)
                elif spin == 'down':
                    ax.set_xlim(np.min(density_in_plot) * 1.1, 0)
            elif energyaxis == 'x':
                ax.set_xlim(erange)
                if spin == 'up' or spin == 'both':
                    ax.set_ylim(0, np.max(density_in_plot) * 1.1)
                elif spin == 'down':
                    ax.set_ylim(np.min(density_in_plot) * 1.1, 0)
        elif len(ax.lines) > 0:
            if energyaxis == 'y':
                ax.set_ylim(erange)
                xlims = ax.get_xlim()
                if xlims[0] == 0:
                    if spin == 'up' or spin == 'both':
                        ax.set_xlim(0, np.max(density_in_plot) * 1.1)
                    elif spin == 'down':
                        ax.set_xlim(np.min(density_in_plot) * 1.1, xlims[1])
                if xlims[1] == 0:
                    if spin == 'up' or spin == 'both':
                        ax.set_xlim(xlims[0], np.max(density_in_plot) * 1.1)
                    elif spin == 'down':
                        ax.set_xlim(np.min(density_in_plot) * 1.1, 0)
            elif energyaxis == 'x':
                ax.set_xlim(erange)
                ylims = ax.get_ylim()
                if ylims[0] == 0:
                    if spin == 'up' or spin == 'both':
                        ax.set_ylim(0, np.max(density_in_plot) * 1.1)
                    elif spin == 'down':
                        ax.set_ylim(np.min(density_in_plot) * 1.1, ylims[1])
                if ylims[1] == 0:
                    if spin == 'up' or spin == 'both':
                        ax.set_ylim(ylims[0], np.max(density_in_plot) * 1.1)
                    elif spin == 'down':
                        ax.set_ylim(np.min(density_in_plot) * 1.1, 0)

    def _group_layers(self):
        poscar = self.poscar
        sites = poscar.structure.sites
        zvals = np.array([site.c for site in sites])
        unique_values = np.sort(np.unique(np.round(zvals, 3)))
        diff = np.mean(np.diff(unique_values)) * 0.2

        grouped = False
        groups = []
        group_heights = []
        zvals_copy = copy.deepcopy(zvals)
        while not grouped:
            if len(zvals_copy) > 0:
                group_index = np.where(
                    np.isclose(zvals, np.min(zvals_copy), atol=diff)
                )[0]
                group_heights.append(np.min(zvals_copy))
                zvals_copy = np.delete(zvals_copy, np.where(
                    np.isin(zvals_copy, zvals[group_index]))[0])
                groups.append(group_index)
            else:
                grouped = True

        return np.array(groups), np.array(group_heights)


    def plot_plain(self, ax, linewidth=1.5, fill=True, alpha=0.3, alpha_line=1.0, sigma=0.05, energyaxis='y', color='black', erange=[-6, 6]):
        """
        This function plots the total density of states

        Parameters:
            ax (matplotlib.pyplot.axis): Axis to append the tick labels
            linewidth (float): Linewidth of lines
            fill (bool): Determines wether or not to fill underneath the plot
            alpha (float): Alpha value for the fill
            alpha_line (float): Alpha value for the line
            sigma (float): Standard deviation for gaussian filter
            energyaxis (str): Determines the axis to plot the energy on ('x' or 'y')
            color (str): Color of line
            erange (list): Energy range for the DOS plot ([lower bound, upper bound])
        """

        tdos_dict = self.tdos_dict

        if sigma > 0:
            tdensity = self._smear(
                tdos_dict['density'],
                sigma=sigma
            )
        else:
            tdensity = tdos_dict['density']

        self._set_density_lims(
            ax=ax,
            tdensity=tdensity,
            tenergy=tdos_dict['energy'],
            erange=erange,
            energyaxis=energyaxis,
            spin=self.spin,
        )

        if energyaxis == 'y':
            ax.plot(
                tdensity,
                tdos_dict['energy'],
                linewidth=linewidth,
                color=color,
                alpha=alpha_line
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
                alpha=alpha_line
            )

            if fill:
                ax.fill_between(
                    tdos_dict['energy'],
                    tdensity,
                    0,
                    color=color,
                    alpha=alpha,
                )

    def plot_spd(self, ax, order=['s', 'p', 'd'], fill=True, alpha=0.3, alpha_line=1.0, linewidth=1.5, sigma=0.05, energyaxis='y', color_dict=None, legend=True, total=True, erange=[-6, 6]):
        """
        This function plots the total density of states with the projected
        density of states for the total projections of the s, p, and d orbitals.

        Parameters:
            ax (matplotlib.pyplot.axis): Axis to plot on
            order (list): Order to plot the projected bands in. This feature helps to
                avoid situations where one projection completely convers the other.
            fill (bool): Determines wether or not to fill underneath the plot
            alpha (float): Alpha value for the fill
            alpha_line (float): Alpha value for the line
            linewidth (float): Linewidth of lines
            sigma (float): Standard deviation for gaussian filter
            energyaxis (str): Determines the axis to plot the energy on ('x' or 'y')
            color_dict (dict[str][str]): This option allow the colors of the s, p, and d
                orbitals to be specified. Should be in the form of:
                {'s': <s color>, 'p': <p color>, 'd': <d color>}
            legend (bool): Determines whether to draw the legend or not
            total (bool): Determines wheth to draw the total density of states or not
            erange (list): Energy range for the DOS plot ([lower bound, upper bound])
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
                alpha_line=alpha_line,
                sigma=sigma,
                energyaxis=energyaxis,
                erange=erange,
            )
        else:
            self._set_density_lims(
                ax=ax,
                tdensity=spd_df,
                tenergy=tdos_dict['energy'],
                erange=erange,
                energyaxis=energyaxis,
                spin=self.spin,
                partial=True,
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
                    alpha=alpha_line
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
                    alpha=alpha_line
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
                fontsize=6,
                bbox_to_anchor=(1, 1),
                borderaxespad=0,
                frameon=False,
                handletextpad=0.1,
            )

    def plot_atom_orbitals(self, ax, atom_orbital_pairs, fill=True, alpha=0.3, alpha_line=1.0, linewidth=1.5, sigma=0.05, energyaxis='y', color_list=None, legend=True, total=True, erange=[-6, 6]):
        """
        This function plots the total density of states with the projected
        density of states for the projections or orbitals on individual atoms.

        Parameters:
            ax (matplotlib.pyplot.axis): Axis to plot on
            atom_orbital_pairs (list[list]): List of atoms orbitals pairs in the form of
                [[atom index, orbital index], [atom index, orbital index], ..]
            fill (bool): Determines wether or not to fill underneath the plot
            alpha (float): Alpha value for the fill
            alpha_line (float): Alpha value for the line
            linewidth (float): Linewidth of lines
            sigma (float): Standard deviation for gaussian filter
            energyaxis (str): Determines the axis to plot the energy on ('x' or 'y')
            color_list (list): List of colors that is the same length as the atom orbitals list
            legend (bool): Determines whether to draw the legend or not
            total (bool): Determines wheth to draw the total density of states or not
            erange (list): Energy range for the DOS plot ([lower bound, upper bound])
        """

        tdos_dict = self.tdos_dict

        if color_list is None:
            color_dict = self.color_dict
        else:
            color_dict = {i: color for i, color in enumerate(color_list)}

        if total:
            self.plot_plain(
                ax=ax,
                linewidth=linewidth,
                fill=fill,
                alpha=alpha,
                alpha_line=alpha_line,
                sigma=sigma,
                energyaxis=energyaxis,
                erange=erange,
            )
        else:
            self._set_density_lims(
                ax=ax,
                tdensity=self.pdos_dict,
                tenergy=tdos_dict['energy'],
                erange=erange,
                energyaxis=energyaxis,
                spin=self.spin,
                partial=True,
                is_dict=True,
                idx=atom_orbital_pairs,
                multiple=True,
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
                    alpha=alpha_line,
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
                    alpha=alpha_line,
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
                fontsize=6,
                bbox_to_anchor=(1, 1),
                borderaxespad=0,
                frameon=False,
                handletextpad=0.1,
            )

    def plot_orbitals(self, ax, orbitals, fill=True, alpha=0.3, alpha_line=1.0, linewidth=1.5, sigma=0.05, energyaxis='y', color_dict=None, legend=True, total=True, erange=[-6, 6]):
        """
        This function plots the total density of states with the projected
        density of states for the projections onto given orbitals

        Parameters:
            ax (matplotlib.pyplot.axis): Axis to plot on
            orbitals (list): List of orbitals to project onto
            fill (bool): Determines wether or not to fill underneath the plot
            alpha (float): Alpha value for the fill
            alpha_line (float): Alpha value for the line
            linewidth (float): Linewidth of lines
            sigma (float): Standard deviation for gaussian filter
            energyaxis (str): Determines the axis to plot the energy on ('x' or 'y')
            color_dict (dict[str][str]): This option allow the colors of each orbital
                specified. Should be in the form of:
                {'orbital index': <color>, 'orbital index': <color>, ...}
            legend (bool): Determines whether to draw the legend or not
            total (bool): Determines wheth to draw the total density of states or not
            erange (list): Energy range for the DOS plot ([lower bound, upper bound])
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
                alpha_line=alpha_line,
                sigma=sigma,
                energyaxis=energyaxis,
                erange=erange,
            )
        else:
            self._set_density_lims(
                ax=ax,
                tdensity=orbital_df,
                tenergy=tdos_dict['energy'],
                erange=erange,
                energyaxis=energyaxis,
                spin=self.spin,
                partial=True,
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
                    alpha=alpha_line,
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
                    alpha=alpha_line,
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
                fontsize=6,
                bbox_to_anchor=(1, 1),
                borderaxespad=0,
                frameon=False,
                handletextpad=0.1,
            )

    def plot_atoms(self, ax, atoms, fill=True, alpha=0.3, alpha_line=1.0, linewidth=1.5, sigma=0.05, energyaxis='y', color_list=None, legend=True, total=True, erange=[-6, 6]):
        """
        This function plots the total density of states with the projected density of states on the given atoms.

        Parameters:
            ax (matplotlib.pyplot.axis): Axis to plot on
            atoms (list): Index of atoms to plot
            fill (bool): Determines wether or not to fill underneath the plot
            alpha (float): Alpha value for the fill
            alpha_line (float): Alpha value for the line
            color_list (list): Optional list of colors for each atom
            linewidth (float): Linewidth of lines
            sigma (float): Standard deviation for gaussian filter
            energyaxis (str): Determines the axis to plot the energy on ('x' or 'y')
            color_list (list): List of colors that is the same length at the atoms list
            legend (bool): Determines whether to draw the legend or not
            total (bool): Determines wheth to draw the total density of states or not
            erange (list): Energy range for the DOS plot ([lower bound, upper bound])
        """

        atom_df = self._sum_atoms()
        tdos_dict = self.tdos_dict

        if color_list is None:
            color_dict = self.color_dict
        else:
            color_dict = {i: color for i, color in enumerate(color_list)}

        if total:
            self.plot_plain(
                ax=ax,
                linewidth=linewidth,
                fill=fill,
                alpha=alpha,
                alpha_line=alpha_line,
                sigma=sigma,
                energyaxis=energyaxis,
                erange=erange,
            )
        else:
            self._set_density_lims(
                ax=ax,
                tdensity=atom_df,
                tenergy=tdos_dict['energy'],
                erange=erange,
                energyaxis=energyaxis,
                spin=self.spin,
                partial=True,
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
                    alpha=alpha_line,
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
                    alpha=alpha_line,
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
                fontsize=6,
                bbox_to_anchor=(1, 1),
                borderaxespad=0,
                frameon=False,
                handletextpad=0.1,
            )

    def plot_elements(self, ax, elements, fill=True, alpha=0.3, alpha_line=1.0, linewidth=1.5, sigma=0.05, energyaxis='y', color_list=None, legend=True, total=True, erange=[-6, 6]):
        """
        This function plots the total density of states with the projected
        density of states for the projection onto specified elements. This is 
        useful for supercells where there are many atoms of the same element and
        it is inconvienient to manually list each index in the POSCAR.

        Parameters:
            ax (matplotlib.pyplot.axis): Axis to plot on
            elements (list): List of element symbols to project onto
            fill (bool): Determines wether or not to fill underneath the plot
            alpha (float): Alpha value for the fill
            alpha_line (float): Alpha value for the line
            linewidth (float): Linewidth of lines
            sigma (float): Standard deviation for gaussian filter
            energyaxis (str): Determines the axis to plot the energy on ('x' or 'y')
            color_list (list): List of colors that is the same length at the elements list
            legend (bool): Determines whether to draw the legend or not
            total (bool): Determines wheth to draw the total density of states or not
            erange (list): Energy range for the DOS plot ([lower bound, upper bound])
        """

        element_dict = self._sum_elements(
            elements=elements, orbitals=False, spd=False)
        tdos_dict = self.tdos_dict

        if color_list is None:
            color_dict = self.color_dict
        else:
            color_dict = {i: color for i, color in enumerate(color_list)}

        if total:
            self.plot_plain(
                ax=ax,
                linewidth=linewidth,
                fill=fill,
                alpha=alpha,
                alpha_line=alpha_line,
                sigma=sigma,
                energyaxis=energyaxis,
                erange=erange,
            )
        else:
            self._set_density_lims(
                ax=ax,
                tdensity=pd.DataFrame(element_dict),
                tenergy=tdos_dict['energy'],
                erange=erange,
                energyaxis=energyaxis,
                spin=self.spin,
                partial=True,
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
                    alpha=alpha_line,
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
                    alpha=alpha_line,
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
                fontsize=6,
                bbox_to_anchor=(1, 1),
                borderaxespad=0,
                frameon=False,
                handletextpad=0.1,
            )

    def plot_element_orbitals(self, ax, element_orbital_pairs, fill=True, alpha=0.3, alpha_line=1.0, linewidth=1.5, sigma=0.05, energyaxis='y', color_list=None, legend=True, total=True, erange=[-6, 6]):
        """
        This function plots the total density of states with the projected
        density of states onto the chosen orbitals of specified elements. This is 
        useful for supercells where there are many atoms of the same element and
        it is inconvienient to manually list each index in the POSCAR.

        Parameters:
            ax (matplotlib.pyplot.axis): Axis to plot on
            element_orbital_pairs (list[list]): List of element orbital pairs in the form of
                [[element symbol, orbital index], [element symbol, orbital index], ..]
            fill (bool): Determines wether or not to fill underneath the plot
            alpha (float): Alpha value for the fill
            alpha_line (float): Alpha value for the line
            linewidth (float): Linewidth of lines
            sigma (float): Standard deviation for gaussian filter
            energyaxis (str): Determines the axis to plot the energy on ('x' or 'y')
            color_list (list): List of colors that is the same length as the element orbitals list
            legend (bool): Determines whether to draw the legend or not
            total (bool): Determines wheth to draw the total density of states or not
            erange (list): Energy range for the DOS plot ([lower bound, upper bound])
        """

        elements = [i[0] for i in element_orbital_pairs]

        element_dict = self._sum_elements(
            elements=elements, orbitals=True, spd=False)
        tdos_dict = self.tdos_dict

        if color_list is None:
            color_dict = self.color_dict
        else:
            color_dict = {i: color for i, color in enumerate(color_list)}

        if total:
            self.plot_plain(
                ax=ax,
                linewidth=linewidth,
                fill=fill,
                alpha=alpha,
                alpha_line=alpha_line,
                sigma=sigma,
                energyaxis=energyaxis,
                erange=erange,
            )
        else:
            self._set_density_lims(
                ax=ax,
                tdensity=element_dict,
                tenergy=tdos_dict['energy'],
                erange=erange,
                energyaxis=energyaxis,
                spin=self.spin,
                partial=True,
                is_dict=True,
                idx=element_orbital_pairs,
                multiple=True,
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
                    alpha=alpha_line,
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
                    alpha=alpha_line,
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
                fontsize=6,
                bbox_to_anchor=(1, 1),
                borderaxespad=0,
                frameon=False,
                handletextpad=0.1,
            )

    def plot_element_spd(self, ax, elements, order=['s', 'p', 'd'], fill=True, alpha=0.3, alpha_line=1.0, linewidth=1.5, sigma=0.05, energyaxis='y', color_dict=None, legend=True, total=True, erange=[-6, 6]):
        """
        This function plots the total density of states with the projected
        density of states onto the s, p, and d orbitals of specified elements. 
        This is useful for supercells where there are many atoms of the same 
        element and it is inconvienient to manually list each index in the POSCAR.

        Parameters:
            ax (matplotlib.pyplot.axis): Axis to plot on
            elements (list): List of element symbols to project onto
            order (list): Order to plot the projected bands in. This feature helps to
                avoid situations where one projection completely convers the other.
            fill (bool): Determines wether or not to fill underneath the plot
            alpha (float): Alpha value for the fill
            alpha_line (float): Alpha value for the line
            linewidth (float): Linewidth of lines
            sigma (float): Standard deviation for gaussian filter
            energyaxis (str): Determines the axis to plot the energy on ('x' or 'y')
            color_dict (dict[str][str]): This option allow the colors of each element
                specified. Should be in the form of:
                {'element index': <color>, 'element index': <color>, ...}
            legend (bool): Determines whether to draw the legend or not
            total (bool): Determines wheth to draw the total density of states or not
            erange (list): Energy range for the DOS plot ([lower bound, upper bound])
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
                alpha_line=alpha_line,
                sigma=sigma,
                energyaxis=energyaxis,
                erange=erange,
            )
        else:
            self._set_density_lims(
                ax=ax,
                tdensity=element_dict,
                tenergy=tdos_dict['energy'],
                erange=erange,
                energyaxis=energyaxis,
                spin=self.spin,
                partial=True,
                is_dict=True,
                idx=elements,
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
                        alpha=alpha_line
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
                        alpha=alpha_line
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
                fontsize=6,
                bbox_to_anchor=(1, 1),
                borderaxespad=0,
                frameon=False,
                handletextpad=0.1,
            )

    def plot_layers(self, ax, cmap='magma', sigma=5, energyaxis='y', erange=[-6, 6], vmax=0.6, fontsize=6, interface_layer=None, interface_line_color='white', interface_line_width=2, interface_line_style='--'):
        """
        This function plots a layer by layer heat map of the density
        of states.

        Parameters:
            ax (matplotlib.pyplot.axis): Axis to plot on
            ylim (list): Upper and lower energy bounds for the plot.
            cmap (str): Color map to use in the heat map
            sigma (float): Sigma parameter for the _smearing of the heat map.
            energyaxis (str): Axis to plot the energy on. ('x' or 'y')
        """
        energy = self.tdos_dict['energy']

        ind = np.where(
                (erange[0] - 0.5 <= energy) & (energy <= erange[-1] + 0.5)
        )
        groups, group_heights = self._group_layers()
        atom_index = group_heights
        energies = energy[ind]
        atom_densities = self._sum_atoms().to_numpy()[ind]
        densities = np.vstack([np.sum(np.vstack(atom_densities[:,[group]]), axis=1) for group in groups])
        densities = np.transpose(densities)
        densities = gaussian_filter(densities, sigma=sigma)

        if energyaxis == 'y':
            im = ax.pcolormesh(
                atom_index,
                energies,
                densities,
                cmap=cmap,
                shading='gouraud',
                vmax=vmax,
            )

            if len(group_heights) <= 12:
                ax.set_xticks(group_heights)
                ax.set_xticklabels(range(len(group_heights)))
            else:
                idx = np.round(np.linspace(0, len(group_heights) - 1, 12)).astype(int)
                group_heights_new = group_heights[idx] 
                ax.set_xticks(group_heights_new)
                ax.set_xticklabels(idx)

            if interface_layer is not None:
                ax.axvline(
                    x=group_heights[interface_layer],
                    color=interface_line_color,
                    linestyle=interface_line_style,
                    linewidth=interface_line_width,
                )

        if energyaxis == 'x':
            im = ax.pcolormesh(
                energies,
                atom_index,
                np.transpose(densities),
                cmap=cmap,
                shading='gouraud',
                vmax=vmax,
            )
            if len(group_heights) <= 12:
                ax.set_yticks(group_heights)
                ax.set_yticklabels(range(len(group_heights)))
            else:
                idx = np.round(np.linspace(0, len(group_heights) - 1, 12)).astype(int)
                group_heights_new = group_heights[idx] 
                ax.set_yticks(group_heights_new)
                ax.set_yticklabels(idx)

            if interface_layer is not None:
                ax.axhline(
                    y=group_heights[interface_layer],
                    color=interface_line_color,
                    linestyle=interface_line_style,
                    linewidth=interface_line_width,
                )

        fig = plt.gcf()
        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.tick_params(labelsize=fontsize)
        cbar.set_label('Density of States', fontsize=fontsize)

    def plot_structure(self, ax, rotation=[90,90,90]):
        structure = self.poscar.structure
        atoms = AseAtomsAdaptor().get_atoms(structure)
        atoms = plot_atoms(
            atoms,
            ax,
            radii=0.5,
            rotation=(f'{rotation[0]}x,{rotation[1]}y,{rotation[2]}z'),
            show_unit_cell=0
        )


