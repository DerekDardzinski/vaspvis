from band import Band
from unfold import unfold, make_kpath, removeDuplicateKpoints, EBS_scatter
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


class Unfold:
    """
    This class contains methods for constructing unfolded band structures for
    supercell and slab calculations.
    """

    def __init__(self, folder, projected, kpath, high_symm_points, n, M, spin='up'):
        self.vasprun = BSVasprun(
            os.path.join(folder, 'vasprun.xml'),
            parse_projected_eigen=projected
        )
        self.poscar = Poscar.from_file(
            os.path.join(folder, 'POSCAR'),
            check_for_POTCAR=False,
            read_velocities=False
        )
        self.wavecar = os.path.join(folder, 'WAVECAR')
        self.projected = projected
        self.forbitals = False
        self.kpath = kpath
        self.n = n
        self.M = M
        self.high_symm_points = high_symm_points
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
            self.projected_array = self._load_projected_bands()


    def _load_bands(self):
        efermi = self.vasprun.efermi
        wavecar_data = unfold(
            M=self.M,
            wavecar=self.wavecar,
            lsorbit=True,
        )
        kpath = make_kpath(self.high_symm_points, nseg=self.n)
        spectral_weight = wavecar_data.spectral_weight(kpath)
        band_energies = []
        spectral_weights = []
        K_indices = []
        for i in range(len(spectral_weight[0][0])):
            band_energies.append(spectral_weight[0,:,i,0] - efermi)
            spectral_weights.append(spectral_weight[0,:,i,1])
            K_indices.append(spectral_weight[0,:,i,2])

        return [
            np.array(band_energies),
            np.array(spectral_weights),
            np.array(K_indices),
            np.array(kpath)
        ]


    def _load_projected_bands(self):
        """
        This function loads the project weights of the orbitals in each band
        from vasprun.xml into a dictionary of the form:
        band index --> atom index --> weights of orbitals

        Returns:
            projected_dict (dict([str][int][pd.DataFrame])): Dictionary containing the projected weights of all orbitals on each atom for each band.
        """

        spin = self.spin

        #  kpoints_band = self.n * (len(self.kpath) - 1)
        projected_eigenvalues = self.vasprun.projected_eigenvalues[
            self.spin_dict[spin]
        ]
        projected_eigenvalues = np.transpose(projected_eigenvalues, axes=(1,0,2,3))

        return projected_eigenvalues

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

        orbital_contributions = self.projected_array.sum(axis=2) / np.sum(self.projected_array.sum(axis=2), axis=2)[:,:,np.newaxis]

        return orbital_contributions

    def _sum_spd(self):
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
            spd_indices[2][9:] = True

        orbital_contributions = self.projected_array.sum(axis=2) / np.sum(self.projected_array.sum(axis=2), axis=2)[:,:,np.newaxis]

        spd_contributions = np.transpose(
            np.array([
                np.sum(orbital_contributions[:,:,:], axis=2, where=ind) for ind in spd_indices
            ]), axes=[1,2,0]
        )

        return spd_contributions


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

        atom_contributions = self.projected_array.sum(axis=3) / np.sum(self.projected_array.sum(axis=3), axis=2)[:,:,np.newaxis]

        atom_contributions = atom_contributions[:,:,atoms]

        return atom_contributions

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

        element_indices = [np.where(np.isin(element_list, symbol))[0] for symbol in symbols]

        
        if not orbitals and not spd:
            atom_contributions = self.projected_array.sum(axis=3) / np.sum(self.projected_array.sum(axis=3), axis=2)[:,:,np.newaxis]
            element_contributions = np.transpose(
                np.array([
                    np.sum(atom_contributions[:,:,:], axis=2, where=ind) for ind in element_indices
                ]), axes=[1,2,0]
            )

        return element_contributions

    def _get_kticks(self, kpath):
        cell = self.poscar.structure.lattice.matrix
        kpt_c = np.dot(kpath, np.linalg.inv(cell).T)
        kdist = np.r_[0, np.cumsum(np.linalg.norm( np.diff(kpt_c,axis=0), axis=1))]

        return kdist

    def _set_ktick_labels(self, kpoints):

        print(self.kpath)

        kpath = [
            f'${k}$' if k != 'G' else '$\\Gamma$' for k in self.kpath.upper().strip()
        ]

        kpoints_index = [0] + [(self.n * i) for i in range(1, len(self.kpath))]
        print(kpoints_index)

        for k in kpoints_index:
            ax.axvline(x=kpoints[k], color='black', alpha=0.7, linewidth=0.5)

        plt.xticks(np.array(kpoints)[kpoints_index], kpath)

    def plot_plain(self, ax, scale=1, color='black'):
        [band_energies, spectral_weights, K_indices, kpath] = self._load_bands()
        kticks = self._get_kticks(kpath)
        kticks = np.ravel(np.array([kticks for _ in range(len(band_energies))]))
        band_energies = np.ravel(band_energies)
        spectral_weights = np.ravel(spectral_weights)

        ax.scatter(
            kticks,
            band_energies,
            s=scale * spectral_weights,
            c=color,
        )

        return ax

    def _draw_pie(self, xs,ys,dist,size, colors, ax):
        for i in range(len(xs)):
            indv_dist = dist[i]
            indv_color = colors
            if not np.isclose(np.sum(indv_dist), 1, atol=0.01):
                indv_dist = np.append(indv_dist, 1 - np.sum(indv_dist))
                indv_color.append('black')

            cumsum = np.cumsum(indv_dist)
            cumsum = cumsum/ cumsum[-1]
            pie = [0] + cumsum.tolist()

            for r1, r2, j in zip(pie[:-1], pie[1:], range(len(indv_dist))):
                angles = np.linspace(2 * np.pi * r1, 2 * np.pi * r2)
                x = [0] + np.cos(angles).tolist()
                y = [0] + np.sin(angles).tolist()

                xy = np.column_stack([x, y])

                ax.plot(
                    [xs[i]],
                    [ys[i]],
                    marker=xy,
                    color=colors[j],
                    markersize=np.sqrt(size[i])
                )

        return ax

    def plot_spd(self, ax, scale_factor=1, colors=None, legend=True):
        [band_energies, spectral_weights, K_indices, kpath] = self._load_bands()
        new_shape = int(band_energies.shape[0] * band_energies.shape[1])
        kticks = self._get_kticks(kpath)
        kticks = np.reshape(np.array([kticks for _ in range(len(band_energies))]), newshape=(new_shape))
        band_energies = np.reshape(band_energies, newshape=(new_shape))
        spectral_weights = np.reshape(spectral_weights, newshape=(new_shape))
        projections = self._sum_spd()
        projections = projections[:,np.array(K_indices[0], dtype=int),:]
        projections = np.reshape(projections, newshape=(new_shape, projections.shape[2]))

        self._draw_pie(
            xs=kticks,
            ys=band_energies,
            dist=projections,
            size=scale_factor * spectral_weights,
            colors=["red", "blue", "green"],
            ax=ax
        )

        ax.set_xlim(np.min(kticks), np.max(kticks))

        self._set_ktick_labels(kpoints=kticks)

        if legend:
            legend_lines = []
            legend_labels = []
            for orbital in ["s", "p", "d"]:
                legend_lines.append(plt.Line2D(
                    [0],
                    [0],
                    marker='o',
                    markersize=2,
                    linestyle='',
                    color=colors[orbital]
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

        return ax
    
    def plot_atoms(self, ax, atoms, scale_factor=1, color_list=None, legend=True):
        [band_energies, spectral_weights, K_indices, kpath] = self._load_bands()
        new_shape = int(band_energies.shape[0] * band_energies.shape[1])
        kticks = self._get_kticks(kpath)
        kticks = np.reshape(np.array([kticks for _ in range(len(band_energies))]), newshape=(new_shape))
        band_energies = np.reshape(band_energies, newshape=(new_shape))
        spectral_weights = np.reshape(spectral_weights, newshape=(new_shape))
        projections = self._sum_atoms(atoms=atoms)
        projections = projections[:,np.array(K_indices[0], dtype=int),:]
        projections = np.reshape(projections, newshape=(new_shape, projections.shape[2]))

        if color_list is None:
            color_dict = self.color_dict
        else:
            color_dict = {i: color for i, color in enumerate(color_list)}

        self._draw_pie(
            xs=kticks,
            ys=band_energies,
            dist=projections,
            size=scale_factor * spectral_weights,
            colors=[color_dict[i] for i in atoms],
            ax=ax
        )

        ax.set_xlim(np.min(kticks), np.max(kticks))

        self._set_ktick_labels(kpoints=kticks)

        if legend:
            legend_lines = []
            legend_labels = []
            for atom in atoms:
                legend_lines.append(plt.Line2D(
                    [0],
                    [0],
                    marker='o',
                    markersize=2,
                    linestyle='',
                    color=color_dict[atom]
                ))
                legend_labels.append(
                    f'${atom}$'
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

        return ax


if __name__ == "__main__":
    M = [[-1,1,0],[-1,-1,1],[0,0,1]]
    high_symm_points = [
        [0.5, 0.5, 0],
        [0.0, 0.0, 0],
        [0.5, 0.5, 0]
    ] 
    uf = Unfold(
        folder="../../vaspvis_data/band-unfold",
        projected=True,
        kpath='XGX',
        high_symm_points=high_symm_points, 
        n=30,
        M=M,
    )
    fig, ax = plt.subplots(figsize=(3,4), dpi=300)
    start = time.time()
    uf.plot_atoms(
        ax=ax,
        scale_factor=10,
        atoms=[0,1,2,3],
    )
    end = time.time()
    print(end-start)
    ax.set_ylabel('$E - E_{F}$ $(eV)$', fontsize=8)
    ax.tick_params(labelsize=8, length=2.5)
    ax.tick_params(axis='x', length=0)
    ax.set_ylim(-5,0)
    plt.tight_layout()
    plt.savefig('test_atoms.png')
    #  plt.show()
        
        


