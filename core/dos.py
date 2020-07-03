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


class DOSPlot:

    def __init__(self, folder, spin='up'):
        self.folder = folder
        self.spin = spin
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
            8: '$d_{x^{2}-y^{2}}$'
        }
        self.spin_dict = {'up': Spin.up, 'down': Spin.down}
        self.tdos_dict = self.load_tdos()
        self.pdos_dict = self.load_pdos()

    def load_tdos(self):
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

    def load_pdos(self):
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
            df = pd.DataFrame.from_dict(new_dict)
            pdos_dict[i] = df

        return pdos_dict

    def smear(self, dos, sigma):
        """
        This function applied a 1D gaussian filter to the density of states

        Inputs:
        ----------
        dos: (np.ndarray) Array of densities.
        sigma: (float) Standard deviation used in the gaussian filter.


        Outputs:
        ----------
        smeared_dos: (np.ndarray) Array of smeared densities.
        """

        diff = np.diff(self.tdos_dict['energy'])
        avgdiff = np.mean(diff)
        smeared_dos = gaussian_filter1d(dos, sigma / avgdiff)

        return smeared_dos

    def sum_orbitals(self):
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

    def sum_atoms(self):
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

    def sum_spd(self):
        """
        This function sums the weights of the s, p, and d orbitals for each atom
        and creates a Dataframe containing the projected densities:

        Outputs:
        ----------
        spd_df: (pd.DataFrame) Dataframe that has the summed densities of the
            s, p, and d orbitals across all atoms.
        """

        spd_df = self.sum_orbitals()

        spd_df['s'] = spd_df[0]
        spd_df['p'] = spd_df[1] + spd_df[2] + spd_df[3]
        spd_df['d'] = spd_df[4] + spd_df[5] + spd_df[6] + spd_df[7] + spd_df[8]
        spd_df = spd_df.drop(columns=range(9))

        return spd_df

    def plot_spd(self, ax, order=['s', 'p', 'd'], fill=True, alpha=0.3, linewidth=1.5, sigma=0.05, energyaxis='y'):
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
        """

        spd_df = self.sum_spd()
        tdos_dict = self.tdos_dict

        if sigma > 0:
            tdensity = self.smear(
                tdos_dict['density'],
                sigma=sigma
            )
        else:
            tdensity = tdos_dict['density']

        if energyaxis == 'y':
            ax.plot(
                tdensity,
                tdos_dict['energy'],
                color='black',
                linewidth=linewidth,
            )

            if fill:
                ax.fill_betweenx(
                    tdos_dict['energy'],
                    tdensity,
                    0,
                    color='black',
                    alpha=alpha,
                )

        if energyaxis == 'x':
            ax.plot(
                tdos_dict['energy'],
                tdensity,
                color='black',
                linewidth=linewidth,
            )

            if fill:
                ax.fill_between(
                    tdos_dict['energy'],
                    tdensity,
                    0,
                    color='black',
                    alpha=alpha,
                )

        for i, orbital in enumerate(order):
            if sigma > 0:
                pdensity = self.smear(
                    spd_df[orbital],
                    sigma=sigma
                )
            else:
                pdensity = spd_df[orbital]

            if energyaxis == 'y':
                ax.plot(
                    pdensity,
                    tdos_dict['energy'],
                    color=self.color_dict[i],
                    linewidth=linewidth,
                    label=f'${orbital}$',
                )

                if fill:
                    ax.fill_betweenx(
                        tdos_dict['energy'],
                        pdensity,
                        0,
                        color=self.color_dict[i],
                        alpha=alpha,
                    )

            if energyaxis == 'x':
                ax.plot(
                    tdos_dict['energy'],
                    pdensity,
                    color=self.color_dict[i],
                    linewidth=linewidth,
                    label=f'${orbital}$',
                )

                if fill:
                    ax.fill_between(
                        tdos_dict['energy'],
                        pdensity,
                        0,
                        color=self.color_dict[i],
                        alpha=alpha,
                    )

    def plot_atoms(self, ax, atoms, fill=True, alpha=0.3, color_dict=None, linewidth=1.5, sigma=0.05, energyaxis='y'):
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
        """

        atom_df = self.sum_atoms()
        tdos_dict = self.tdos_dict

        if color_dict is None:
            color_dict = self.color_dict

        if sigma > 0:
            tdensity = self.smear(
                tdos_dict['density'],
                sigma=sigma
            )
        else:
            tdensity = tdos_dict['density']

        if energyaxis == 'y':
            ax.plot(
                tdensity,
                tdos_dict['energy'],
                color='black',
                linewidth=linewidth,
            )

            if fill:
                ax.fill_betweenx(
                    tdos_dict['energy'],
                    tdensity,
                    0,
                    color='black',
                    alpha=alpha,
                )

        elif energyaxis == 'x':
            ax.plot(
                tdos_dict['energy'],
                tdensity,
                color='black',
                linewidth=linewidth,
            )

            if fill:
                ax.fill_between(
                    tdos_dict['energy'],
                    tdensity,
                    0,
                    color='black',
                    alpha=alpha,
                )

        for i, atom in enumerate(atoms):
            if sigma > 0:
                pdensity = self.smear(
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

    def plot_layers(self, ax, ylim=[-6, 6], cmap='magma', sigma=5):
        """
        This function plots a layer by layer heat map of the density
        of states.

        Inputs:
        ----------
        ax: (matplotlib.pyplot.axis) Axis to plot on
        ylim: (list) Upper and lower energy bounds for the plot.
        cmap: (str) Color map to use in the heat map
        sigma: (float) Sigma parameter for the smearing of the heat map.
        """

        poscar = self.poscar
        sites = poscar.structure.sites
        zvals = np.array([site.c for site in sites])
        zorder = np.argsort(zvals)
        energy = self.tdos_dict['energy']

        ind = np.where((ylim[0] - 0.1 <= energy) & (energy <= ylim[-1] + 0.1))
        atom_index = range(len(zorder))
        energies = energy[ind]
        densities = self.sum_atoms().to_numpy()[ind]
        densities = np.transpose(densities)
        densities = np.transpose(densities[zorder])
        densities = gaussian_filter(densities, sigma=sigma)

        ax.pcolormesh(
            atom_index,
            energies,
            densities,
            cmap=cmap,
            shading='gouraud'
        )


def main():
    fig = plt.figure(figsize=(4, 8), dpi=200)
    ax = fig.add_subplot(111)
    plt.ylim(-6, 6)
    ax.margins(x=0.005, y=0.005)
    dos = DOSPlot(folder='../../vaspvis_data/dos')
    dos.plot_spd(ax=ax, energyaxis='y')
    # dos.plot_layers(ax=ax)
    # dos.plot_atoms(ax=ax, atoms=[0, 1], sigma=0.1, fill=True, energyaxis='y')
    plt.show()


if __name__ == '__main__':
    main()
