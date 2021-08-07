from pymatgen.electronic_structure.core import Spin, Orbital
from pymatgen.io.vasp.outputs import BSVasprun, Eigenval
from pymatgen.io.vasp.inputs import Kpoints, Poscar, Incar
from pymatgen.symmetry.bandstructure import HighSymmKpath
from pymatgen.core.periodic_table import Element
from vaspvis.unfold import unfold, make_kpath, removeDuplicateKpoints
from pymatgen.core.periodic_table import Element
from pyprocar.utilsprocar import UtilsProcar
from pyprocar.procarparser import ProcarParser
from functools import reduce
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from matplotlib.collections import PatchCollection
import matplotlib.colors as colors
from matplotlib.colors import Normalize, to_rgba, LinearSegmentedColormap
import matplotlib.transforms as transforms
import numpy as np
import pandas as pd
import time
from copy import deepcopy
import os
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)


class Band:
    """
    This class contains all the methods for constructing band structures
    from the outputs of VASP band structure calculations.

    Parameters:
        folder (str): This is the folder that contains the VASP files
        projected (bool): Determines whether of not to parse the projected
            eigenvalues from the PROCAR file. Making this true
            increases the computational time, so only use if a projected
            band structure is required.
        spin (str): Choose which spin direction to parse. ('up' or 'down')
        kpath (str): High symmetry k-point path of band structure calculation
            Due to the nature of the KPOINTS file for unfolded calculations this
            information is a required input for proper labeling of the figure
            for unfolded calculations. This information is extracted from the KPOINTS
            files for non-unfolded calculations. (G is automaticall converted to \\Gamma)
        n (int): Number of points between each high symmetry point.
            This is also only required for unfolded calculations. This number should be 
            known by the user, as it was used to generate the KPOINTS file.
    """

    def __init__(
        self,
        folder,
        projected=False,
        unfold=False,
        spin='up',
        kpath=None,
        n=None,
        M=None,
        high_symm_points=None,
        bandgap=False,
        printbg=True,
        shift_efermi=0,
        interpolate=True,
        new_n=200,
        custom_kpath=None,
    ):
        """
        Initialize parameters upon the generation of this class

        Parameters:
            folder (str): This is the folder that contains the VASP files
            projected (bool): Determines whether of not to parse the projected
                eigenvalues from the PROCAR file. Making this true
                increases the computational time, so only use if a projected
                band structure is required.
            unfold (bool): Determines if the band structure should be unfolded or not.
            spin (str): Choose which spin direction to parse. ('up' or 'down')
            kpath (str): High symmetry k-point path of band structure calculation
                Due to the nature of the KPOINTS file for unfolded calculations this
                information is a required input for proper labeling of the figure
                for unfolded calculations. This information is extracted from the KPOINTS
                files for non-unfolded calculations. (G is automatically converted to \\Gamma)
            n (int): Number of points between each high symmetry point.
                This is also only required for unfolded calculations. This number should be 
                known by the user, as it was used to generate the KPOINTS file.
            M (list[list]): Transformation matrix for unfolding calculations. Can be found using
                the conver_slab function in the utils module.
            high_symm_points (list[list]): Coordinates of the high symmetry points of the bulk 
                Brilloin zone for an unfolded calculation.
            bandgap (bool): Determines is the band gap should be determined or not.
            printbg (bool): Determines is the band gap value should be printed while plotting
            shift_efermi (float): Gives the option to shift the fermi energy by the specified value
            interpolate (bool): Determines is the data between each high symmetry point should be
                interpolated or not.
            new_n (int): New number of k-points in between each high symmetry point.
            custom_kpath (list): Custom kpath that can be selected is the user desires. 
                Given a path G-X-W-L-G-K then there are 5 segements to choose from
                [1 -> G-X, 2 -> X-W, 3 -> W-L, 4 -> L-G, 5 -> G-K]. If a user wanted to 
                plot only the path G-X-W they can set custom_kpath=[1,2]. If a user wanted
                to flip the k-path of a segment, then the index should be made negative, so
                if the desired path was G-X|L-W then custom_kpath=[1,-3]
        """

        self.interpolate = interpolate
        self.new_n = new_n
        self.bandgap = bandgap
        self.printbg = printbg
        self.eigenval = Eigenval(os.path.join(folder, 'EIGENVAL'))
        self.efermi = float(os.popen(f'grep E-fermi {os.path.join(folder, "OUTCAR")}').read().split()[2]) + shift_efermi
        self.poscar = Poscar.from_file(
            os.path.join(folder, 'POSCAR'),
            check_for_POTCAR=False,
            read_velocities=False
        )
        self.incar = Incar.from_file(
            os.path.join(folder, 'INCAR')
        )
        if 'LSORBIT' in self.incar:
            if self.incar['LSORBIT']:
                self.lsorbit = True
            else:
                self.lsorbit = False
        else:
            self.lsorbit = False

        if 'ISPIN' in self.incar:
            if self.incar['ISPIN'] == 2:
                self.ispin = True
            else:
                self.ispin = False
        else:
            self.ispin = False

        if 'LHFCALC' in self.incar:
            if self.incar['LHFCALC']:
                self.hse = True
            else:
                self.hse = False
        else:
            self.hse = False

        self.kpoints_file = Kpoints.from_file(os.path.join(folder, 'KPOINTS'))

        self.wavecar = os.path.join(folder, 'WAVECAR')
        self.projected = projected

        self.forbitals = self._check_f_orb()
        self.unfold = unfold

        if self.hse and self.unfold:
            self.hse = False

        self.kpath = kpath
        self.n = n
        self.M = M
        self.high_symm_points = high_symm_points
        self.folder = folder
        self.spin = spin
        self.spin_dict = {'up': Spin.up, 'down': Spin.down}
        if not self.unfold:
            self.pre_loaded_bands = os.path.isfile(os.path.join(folder, 'eigenvalues.npy'))
            self.eigenvalues, self.kpoints = self._load_bands()
        else:
            self.pre_loaded_bands = os.path.isfile(os.path.join(folder, 'unfolded_eigenvalues.npy'))
            self.eigenvalues, self.spectral_weights, self.K_indices, self.kpoints = self._load_bands_unfold()
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
            2: 'p_{z}',
            3: 'p_{x}',
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
        if self.bandgap:
            self.bg = self._get_bandgap()
        else:
            self.bg = None

        self.custom_kpath = custom_kpath
        if self.custom_kpath is not None:
            self.custom_kpath_inds, self.custom_kpath_flip = self._get_custom_kpath()
        #  else:
            #  self.custom_kpath_inds, self.custom_kpath_flip = None, None

        if projected:
            self.pre_loaded_projections = os.path.isfile(os.path.join(folder, 'projected_eigenvalues.npy'))
            self.projected_eigenvalues = self._load_projected_bands()

    def _get_custom_kpath(self):
        flip = (-np.sign(self.custom_kpath)+1).astype(bool)
        inds = (np.abs(self.custom_kpath) - 1).astype(int)

        return inds, flip

    def _get_bandgap(self):
        from vaspvis.utils import get_bandgap
        self.bg = get_bandgap(
            folder=self.folder,
            printbg=self.printbg,
        )

    def _check_f_orb(self):
        f_elements = [
            'La',
            'Ac',
            'Ce',
            'Tb',
            'Th',
            'Pr',
            'Dy',
            'Pa',
            'Nd',
            'Ho',
            'U',
            'Pm',
            'Er',
            'Np',
            'Sm',
            'Tm',
            'Pu',
            'Eu',
            'Yb',
            'Am',
            'Gd',
            'Lu',
        ]
        f = False
        for element in self.poscar.site_symbols:
            if element in f_elements:
                f = True
        
        return f
                    


    def _load_bands(self):
        """
        This function is used to load eigenvalues from the vasprun.xml
        file and into a dictionary which is in the form of
        band index --> eigenvalues

        Returns:
            bands_dict (dict[str][np.ndarray]): Dictionary which contains
                the eigenvalues for each band
        """

        if self.spin == 'up':
            spin = 0
        if self.spin == 'down':
            spin = 1

        if self.pre_loaded_bands:
            with open(os.path.join(self.folder, 'eigenvalues.npy'), 'rb') as eigenvals:
                band_data = np.load(eigenvals)

            if self.ispin and not self.lsorbit:
                eigenvalues = band_data[:,:,[0,2]]
                kpoints = band_data[0,:,4:]
            else:
                eigenvalues = band_data[:,:,0]
                kpoints = band_data[0,:,2:]
        else:
            if len(self.eigenval.eigenvalues.keys()) > 1:
                eigenvalues_up = np.transpose(self.eigenval.eigenvalues[Spin.up], axes=(1,0,2))
                eigenvalues_down = np.transpose(self.eigenval.eigenvalues[Spin.down], axes=(1,0,2))
                eigenvalues_up[:,:,0] = eigenvalues_up[:,:,0] - self.efermi
                eigenvalues_down[:,:,0] = eigenvalues_down[:,:,0] - self.efermi
                eigenvalues = np.concatenate(
                    [eigenvalues_up, eigenvalues_down],
                    axis=2
                )
            else:
                eigenvalues = np.transpose(self.eigenval.eigenvalues[Spin.up], axes=(1,0,2))
                eigenvalues[:,:,0] = eigenvalues[:,:,0] - self.efermi

            kpoints = np.array(self.eigenval.kpoints)

            if self.hse:
                kpoint_weights = np.array(self.eigenval.kpoints_weights)
                zero_weight = np.where(kpoint_weights == 0)[0]
                eigenvalues = eigenvalues[:,zero_weight]
                kpoints = kpoints[zero_weight]

            band_data = np.append(
                eigenvalues,
                np.tile(kpoints, (eigenvalues.shape[0],1,1)),
                axis=2,
            )

            np.save(os.path.join(self.folder, 'eigenvalues.npy'), band_data)

            if len(self.eigenval.eigenvalues.keys()) > 1:
                eigenvalues = eigenvalues[:,:,[0,2]]
            else:
                eigenvalues =  eigenvalues[:,:,0]

        if len(self.eigenval.eigenvalues.keys()) > 1:
            eigenvalues = eigenvalues[:,:,spin]

        return eigenvalues, kpoints

    
    def _load_bands_unfold(self):
        
        if self.spin == 'up':
            spin = 0
        if self.spin == 'down':
            spin = 1

        kpath = make_kpath(self.high_symm_points, nseg=self.n)

        if self.pre_loaded_bands:
            with open(os.path.join(self.folder, 'unfolded_eigenvalues.npy'), 'rb') as eigenvals:
                band_data = np.load(eigenvals) 
        else:
            wavecar_data = unfold(
                M=self.M,
                wavecar=self.wavecar,
                lsorbit=self.lsorbit,
            )
            band_data = wavecar_data.spectral_weight(kpath)
            np.save(os.path.join(self.folder, 'unfolded_eigenvalues.npy'), band_data)

        band_data = np.transpose(band_data[spin], axes=(2,1,0))
        eigenvalues, spectral_weights, K_indices = band_data
        eigenvalues = eigenvalues - self.efermi
        kpath = np.array(kpath)

        return eigenvalues, spectral_weights, K_indices, kpath


    def _load_projected_bands(self):
        """
        This function loads the project weights of the orbitals in each band
        from vasprun.xml into a dictionary of the form:
        band index --> atom index --> weights of orbitals

        Returns:
            projected_dict (dict([str][int][pd.DataFrame])): Dictionary containing the projected weights of all orbitals on each atom for each band.
        """
        
        if self.lsorbit:
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
            if self.ispin and not self.lsorbit and np.sum(self.poscar.natoms) == 1:
                shape = int(parser.spd.shape[1] / 2)
                projected_eigenvalues_up = np.transpose(parser.spd[:,:shape,0,:,1:-1], axes=(1,0,2,3))
                projected_eigenvalues_down = np.transpose(parser.spd[:,shape:,0,:,1:-1], axes=(1,0,2,3))
                projected_eigenvalues = np.concatenate(
                    [projected_eigenvalues_up[:,:,:,:,np.newaxis], projected_eigenvalues_down[:,:,:,:,np.newaxis]],
                    axis=4
                )
                projected_eigenvalues = np.transpose(projected_eigenvalues, axes=(0,1,4,2,3))
            elif self.ispin and not self.lsorbit and np.sum(self.poscar.natoms) != 1:
                shape = int(parser.spd.shape[1] / 2)
                projected_eigenvalues_up = np.transpose(parser.spd[:,:shape,0,:-1,1:-1], axes=(1,0,2,3))
                projected_eigenvalues_down = np.transpose(parser.spd[:,shape:,0,:-1,1:-1], axes=(1,0,2,3))
                projected_eigenvalues = np.concatenate(
                    [projected_eigenvalues_up[:,:,:,:,np.newaxis], projected_eigenvalues_down[:,:,:,:,np.newaxis]],
                    axis=4
                )
                projected_eigenvalues = np.transpose(projected_eigenvalues, axes=(0,1,4,2,3))
            else:
                if np.sum(self.poscar.natoms) == 1:
                    projected_eigenvalues = np.transpose(parser.spd[:,:,:,:, 1:-1], axes=(1,0,2,3,4))
                else:
                    projected_eigenvalues = np.transpose(parser.spd[:,:,:,:-1, 1:-1], axes=(1,0,2,3,4))

            np.save(os.path.join(self.folder, 'projected_eigenvalues.npy'), projected_eigenvalues)


        projected_eigenvalues = projected_eigenvalues[:,:,spin,:,:]

        if self.hse:
            kpoint_weights = np.array(self.eigenval.kpoints_weights)
            zero_weight = np.where(kpoint_weights == 0)[0]
            projected_eigenvalues = projected_eigenvalues[:,zero_weight]

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

        orbital_contributions = np.sum(self.projected_eigenvalues, axis=2)

        spd_contributions = np.transpose(
            np.array([
                np.sum(orbital_contributions[:,:,ind], axis=2) for ind in spd_indices
            ]), axes=[1,2,0]
        )

        #  norm_term = np.sum(spd_contributions, axis=2)[:,:,np.newaxis]
        #  spd_contributions = np.divide(spd_contributions, norm_term, out=np.zeros_like(spd_contributions), where=norm_term!=0)

        spd_contributions = spd_contributions[:,:,[self.spd_relations[orb] for orb in spd]]

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
        orbital_contributions = self.projected_eigenvalues.sum(axis=2)
        #  norm_term =  np.sum(orbital_contributions, axis=2)[:,:,np.newaxis]
        #  orbital_contributions = np.divide(orbital_contributions, norm_term, out=np.zeros_like(orbital_contributions), where=norm_term!=0)
        orbital_contributions = orbital_contributions[:,:,[orbitals]]

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

            #  atoms_spd = atoms_spd[:,:,[atoms], :]

            #  norm_term = np.sum(atoms_spd_to_norm, axis=(2,3))[:,:, np.newaxis]
            #  atoms_spd = np.divide(atoms_spd, norm_term, out=np.zeros_like(atoms_spd), where=norm_term!=0)

            return atoms_spd
        else:
            atoms_array = self.projected_eigenvalues.sum(axis=3)
            #  norm_term = np.sum(atoms_array, axis=2)[:,:,np.newaxis]
            #  atoms_array = np.divide(atoms_array, norm_term, out=np.zeros_like(atoms_array), where=norm_term!=0)
            atoms_array = atoms_array[:,:,[atoms]]

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

            #  norm_term = np.sum(element_spd, axis=(2,3))[:,:,np.newaxis, np.newaxis]
            #  element_spd = np.divide(element_spd, norm_term, out=np.zeros_like(element_spd), where=norm_term!=0)

            return element_spd
        else:
            element_array = np.sum(element_orbitals, axis=3)
            #  norm_term = np.sum(element_array, axis=2)[:,:,np.newaxis]
            #  element_array = np.divide(element_array, norm_term, out=np.zeros_like(element_array), where=norm_term!=0)

            return element_array



    def _get_k_distance_old(self):
        cell = self.poscar.structure.lattice.matrix
        kpt_c = np.dot(self.kpoints, np.linalg.inv(cell).T)
        kdist = np.r_[0, np.cumsum(np.linalg.norm(np.diff(kpt_c,axis=0), axis=1))]

        return kdist

    def _get_k_distance(self):
        slices = self._get_slices(unfold=self.unfold, hse=self.hse)
        kdists = []
        
        if self.custom_kpath is not None:
        #  if self.custom_kpath is None:
            index = self.custom_kpath_inds
        else:
            index = range(len(slices))

        for j, i in enumerate(index):
            cell = self.poscar.structure.lattice.matrix
            kpt_c = np.dot(self.kpoints[slices[i]], np.linalg.inv(cell).T)
            kdist = np.r_[0, np.cumsum(np.linalg.norm(np.diff(kpt_c, axis=0), axis=1))]
            if j == 0:
                kdists.append(kdist)
            else:
                kdists.append(kdist + kdists[-1][-1])

        kdists = np.array(kdists)

        return kdists

    def _get_kticks(self, ax, wave_vectors, vlinecolor):
        """
        This function extracts the kpoint labels and index locations for a regular
        band structure calculation (non unfolded).

        Parameters:
            ax (matplotlib.pyplot.axis): Axis to append the tick labels
        """

        high_sym_points = self.kpoints_file.kpts

        segements = []
        for i in range(0, len(high_sym_points) - 1):
            if not i % 2:
                segements.append([i, i+1])

        if self.custom_kpath is not None:
            high_sym_points_inds = []
            for i, b in zip(self.custom_kpath_inds, self.custom_kpath_flip):
                if b:
                    seg = list(reversed(segements[i]))
                else:
                    seg = segements[i]

                high_sym_points_inds.extend(seg)
        else:
            high_sym_points_inds = list(range(len(high_sym_points)))

        num_kpts = self.kpoints_file.num_kpts
        kpts_labels = np.array([f'${k}$' if k != 'G' else '$\\Gamma$' for k in self.kpoints_file.labels])
        all_kpoints = self.kpoints

        group_index = []
        for i, j in enumerate(high_sym_points_inds):
            if i == 0:
                group_index.append([j])
            if i % 2 and not i == len(high_sym_points_inds) - 1:
                group_index.append([j, high_sym_points_inds[i+1]])
            if i == len(high_sym_points_inds) - 1:
                group_index.append([j])

        labels = []
        index = []

        for i in group_index:
            if len(i) == 1:
                labels.append(kpts_labels[i[0]])
                index.append(i[0])
            else:
                if kpts_labels[i[0]] == kpts_labels[i[1]]:
                    labels.append(kpts_labels[i[0]])
                    index.append(i[0])
                else:
                    merged_label = '|'.join([
                        kpts_labels[i[0]],
                        kpts_labels[i[1]],
                    ]).replace('$|$', '|')
                    labels.append(merged_label)
                    index.append(i[0])

        kpoints_index = [0] + [(i+1)*num_kpts - 1 for i in range(int((len(high_sym_points_inds) + 1) / 2))]

        for k in kpoints_index:
            ax.axvline(x=wave_vectors[k], color=vlinecolor, alpha=0.7, linewidth=0.5)
        
        ax.set_xticks([wave_vectors[k] for k in kpoints_index])
        ax.set_xticklabels(labels)


    def _get_kticks_hse(self, wave_vectors, ax, kpath, vlinecolor):
        structure = self.poscar.structure
        kpath_obj = HighSymmKpath(structure)
        kpath_labels = np.array(list(kpath_obj._kpath['kpoints'].keys()))
        kpath_coords = np.array(list(kpath_obj._kpath['kpoints'].values()))
        index = np.where(
            np.isclose(
                self.kpoints[:, None],
                kpath_coords,
            ).all(-1).any(-1) == True
        )[0]

        segements = []
        for i in range(0, len(index) - 1):
            if not i % 2:
                segements.append([index[i], index[i+1]])

        if self.custom_kpath is not None:
            high_sym_points_inds = []
            for i, b in zip(self.custom_kpath_inds, self.custom_kpath_flip):
                if b:
                    seg = list(reversed(segements[i]))
                else:
                    seg = segements[i]

                high_sym_points_inds.extend(seg)
        else:
            high_sym_points_inds = list(np.concatenate(segements))

        full_segments = []
        for i in range(0, len(high_sym_points_inds) - 1):
            if not i % 2:
                full_segments.append([high_sym_points_inds[i], high_sym_points_inds[i+1]])

        segment_lengths = [np.abs(i[1]-i[0]) + 1 for i in full_segments]
        kpoints_index = [0] + [np.sum(segment_lengths[:i]) for i in range(1,len(segment_lengths)+1)]
        kpoints_index[-1] -= 1

        group_index = []
        for i, j in enumerate(high_sym_points_inds):
            if i == 0:
                group_index.append([j])
            if i % 2 and not i == len(high_sym_points_inds) - 1:
                group_index.append([j, high_sym_points_inds[i+1]])
            if i == len(high_sym_points_inds) - 1:
                group_index.append([j])

        kpoints_in_band = []
        for group in group_index:
            g = [self.kpoints[g] for g in group]
            kpoints_in_band.append(g)

        group_labels = []
        for kpoints in kpoints_in_band:
            group = []
            for k in kpoints:
                for i, coords in enumerate(kpath_coords):
                    if (np.round(k, 5) == np.round(coords, 5)).all():
                        group.append(kpath_labels[i])
            group_labels.append(group)

        labels = []
        index = []

        for label in group_labels:
            if len(label) == 1:
                labels.append(label[0])
            else:
                if label[0] == label[1]:
                    labels.append(label[0])
                else:
                    merged_label = '|'.join([
                        label[0],
                        label[1],
                    ]).replace('$|$', '|')
                    labels.append(merged_label)

        kpath = [f'${k}$' if k != 'G' else '$\\Gamma$' for k in labels]

        for k in kpoints_index:
            ax.axvline(x=wave_vectors[k], color=vlinecolor, alpha=0.7, linewidth=0.5)

        plt.xticks(
            [wave_vectors[k] for k in kpoints_index],
            kpath
        )

    def _get_kticks_unfold(self, ax, wave_vectors, vlinecolor):
        if self.custom_kpath is not None:
            kpath = []
            for i, b in zip(self.custom_kpath_inds, self.custom_kpath_flip):
                if b:
                    seg = list(reversed(self.kpath[i]))
                else:
                    seg = self.kpath[i]

                kpath.extend(seg)
        else:
            kpath = []
            for seg in self.kpath:
                kpath.extend(seg)

        kpath = [f'${k.strip()}$' if k.strip() != 'G' else '$\\Gamma$' for k in kpath]

        group_kpath = []
        for i, j in enumerate(kpath):
            if i == 0:
                group_kpath.append([j])
            if i % 2 and not i == len(kpath) - 1:
                group_kpath.append([j, kpath[i+1]])
            if i == len(kpath) - 1:
                group_kpath.append([j])

        labels = []

        for k in group_kpath:
            if len(k) == 1:
                labels.append(k[0])
            else:
                if k[0] == k[1]:
                    labels.append(k[0])
                else:
                    merged_label = '|'.join([k[0], k[1]]).replace('$|$', '|')
                    labels.append(merged_label)

        kpoints_index = [0] + [(self.n * i) for i in range(1, len(labels))]
        kpoints_index[-1] -= 1

        for k in kpoints_index:
            ax.axvline(x=wave_vectors[k], color=vlinecolor, alpha=0.7, linewidth=0.5)

        ax.set_xticks(wave_vectors[kpoints_index])
        ax.set_xticklabels(labels)
        #  plt.xticks(np.array(kpoints)[kpoints_index], kpath)

    def _get_kticks_unfold_old(self, ax, wave_vectors, vlinecolor):
        if type(self.kpath) == str:
            kpath = [
                f'${k}$' if k != 'G' else '$\\Gamma$' for k in self.kpath.upper().strip()
            ]
        elif type(self.kpath) == list:
            kpath = self.kpath

        kpoints_index = [0] + [(self.n * i) for i in range(1, len(self.kpath))]
        kpoints_index[-1] -= 1

        for k in kpoints_index:
            ax.axvline(x=wave_vectors[k], color=vlinecolor, alpha=0.7, linewidth=0.5)

        ax.set_xticks(wave_vectors[kpoints_index])
        ax.set_xticklabels(kpath)
        #  plt.xticks(np.array(kpoints)[kpoints_index], kpath)

    def _get_kticks_old(self, ax, wave_vectors, vlinecolor):
        """
        This function extracts the kpoint labels and index locations for a regular
        band structure calculation (non unfolded).

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
        #  kpoints_index = ax.lines[0].get_xdata()[kpoints_index]

        for k in kpoints_index:
            ax.axvline(x=wave_vectors[k], color=vlinecolor, alpha=0.7, linewidth=0.5)
        
        ax.set_xticks([wave_vectors[k] for k in kpoints_index])
        ax.set_xticklabels(kpts_labels)

    def _get_kticks_hse_old(self, wave_vectors, ax, kpath, vlinecolor):
        structure = self.poscar.structure
        kpath_obj = HighSymmKpath(structure)
        kpath_labels = np.array(list(kpath_obj._kpath['kpoints'].keys()))
        kpath_coords = np.array(list(kpath_obj._kpath['kpoints'].values()))
        index = np.where(
            np.isclose(
                self.kpoints[:, None],
                kpath_coords,
            ).all(-1).any(-1) == True
        )[0]
        #  index = np.where(np.isclose(self.kpoints[:, None], kpath_coords).all(-1).any(-1) == True)[0]
        #  index = np.where((self.kpoints[:, None] == kpath_coords).all(-1).any(-1) == True)[0]
        index = [index[0]] + [index[i] for i in range(1,len(index)-1) if i % 2] + [index[-1]]
        kpoints_in_band = self.kpoints[index]

        label_index = []
        for i in range(kpoints_in_band.shape[0]):
            for j in range(kpath_coords.shape[0]):
                if (np.round(kpoints_in_band[i], 5) == np.round(kpath_coords[j], 5)).all():
                    label_index.append(j)

        kpoints_index = index
        kpath = kpath_labels[label_index]
        #  kpoints_index = ax.lines[0].get_xdata()[kpoints_index]

        kpath = [f'${k}$' if k != 'G' else '$\\Gamma$' for k in kpath]

        for k in kpoints_index:
            ax.axvline(x=wave_vectors[k], color=vlinecolor, alpha=0.7, linewidth=0.5)

        plt.xticks(
            [wave_vectors[k] for k in kpoints_index],
            kpath
        )

    def _get_slices(self, unfold=False, hse=False):
        if not unfold and not hse:
            high_sym_points = self.kpoints_file.kpts
            all_kpoints = self.kpoints
            num_kpts = self.kpoints_file.num_kpts
            num_slices = int(len(high_sym_points) / 2)
            slices = [
                slice(i*num_kpts, (i+1)*num_kpts, None) for i in range(num_slices)
            ]

        if hse and not unfold:
            structure = self.poscar.structure
            kpath_obj = HighSymmKpath(structure)
            kpath_coords = np.array(list(kpath_obj._kpath['kpoints'].values()))
            index = np.where(
                np.isclose(
                    self.kpoints[:, None],
                    kpath_coords,
                ).all(-1).any(-1) == True
            )[0]

            num_kpts = int(len(self.kpoints) / (len(index) / 2))
            slices = [
                slice(i*num_kpts, (i+1)*num_kpts, None) for i in range(int(len(index)/2))
            ]

        if unfold and not hse:
            slices = [
                slice(i*self.n, (i+1)*self.n, None) for i in range(int(len(self.kpath)))
            ]

        return slices

    def _get_slices_old(self, unfold=False, hse=False):
        if not unfold and not hse:
            high_sym_points = self.kpoints_file.kpts
            all_kpoints = self.kpoints
            num_kpts = self.kpoints_file.num_kpts

            if self.custom_kpath is not None:
                num_slices = len(self.custom_kpath_inds)
            else:
                num_slices = int(len(high_sym_points) / 2)

            slices = [
                slice(i*num_kpts, (i+1)*num_kpts, None) for i in range(num_slices)
            ]

        if hse and not unfold:
            structure = self.poscar.structure
            kpath_obj = HighSymmKpath(structure)
            kpath_coords = np.array(list(kpath_obj._kpath['kpoints'].values()))
            index = np.where(
                np.isclose(
                    self.kpoints[:, None],
                    kpath_coords,
                ).all(-1).any(-1) == True
            )[0]

            num_kpts = int(len(self.kpoints) / (len(index) / 2))
            slices = [
                slice(i*num_kpts, (i+1)*num_kpts, None) for i in range(int(len(index)/2))
            ]

        if unfold and not hse:
            slices = [
                slice(i*self.n, (i+1)*self.n, None) for i in range(int(len(self.kpath)-1))
            ]

        return slices

    def _get_interpolated_data_segment(self, wave_vectors, data, crop_zero=False, kind='cubic'):
        data_shape = data.shape

        if len(data_shape) == 1:
            fs = interp1d(wave_vectors, data, kind=kind, axis=0)
        else:
            fs = interp1d(wave_vectors, data, kind=kind, axis=1)

        new_wave_vectors = np.linspace(wave_vectors.min(), wave_vectors.max(), self.new_n)
        data = fs(new_wave_vectors) 

        if crop_zero:
            data[np.where(data < 0)] = 0

        return new_wave_vectors, data

    def _get_interpolated_data(self, wave_vectors, data, crop_zero=False, kind='cubic'):
        slices = self._get_slices(unfold=self.unfold, hse=self.hse)
        data_shape = data.shape
        if len(data_shape) == 1:
            data = [data[i] for i in slices]
        else:
            data = [data[:,i] for i in slices]

        wave_vectors = [wave_vectors[i] for i in slices]

        if len(data_shape) == 1:
            fs = [interp1d(i, j, kind=kind, axis=0) for (i,j) in zip(wave_vectors, data)]
        else:
            fs = [interp1d(i, j, kind=kind, axis=1) for (i,j) in zip(wave_vectors, data)]

        new_wave_vectors = [np.linspace(wv.min(), wv.max(), self.new_n) for wv in wave_vectors]
        data = np.hstack([f(wv) for (f, wv) in zip(fs, new_wave_vectors)])
        wave_vectors = np.hstack(new_wave_vectors)

        if crop_zero:
            data[np.where(data < 0)] = 0

        return wave_vectors, data


    def _filter_bands(self, erange):
        eigenvalues = self.eigenvalues
        where = (eigenvalues >= np.min(erange) - 1) & (eigenvalues <= np.max(erange) + 1)
        is_true = np.sum(np.isin(where, True), axis=1)
        bands_in_plot = is_true > 0

        return bands_in_plot

    def _add_legend(self, ax, names, colors, fontsize=10, markersize=4):
        legend_lines = []
        legend_labels = []
        for name, color in zip(names, colors):
            legend_lines.append(plt.Line2D(
                [0],
                [0],
                marker='o',
                markersize=markersize,
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
            fontsize=fontsize,
            bbox_to_anchor=(1, 1),
            borderaxespad=0,
            frameon=False,
            handletextpad=0.1,
        )

    def _heatmap(
        self,
        ax,
        wave_vectors,
        eigenvalues,
        weights,
        sigma,
        cmap,
        bins,
        projection=None,
        powernorm=True,
        gamma=0.5
    ):
        eigenvalues_ravel = np.ravel(eigenvalues)
        wave_vectors_tile = np.tile(wave_vectors, eigenvalues.shape[0])

        if projection is not None:
            if len(np.squeeze(projection).shape) == 2:
                weights *= np.squeeze(projection)
            else:
                weights *= np.sum(np.squeeze(projection), axis=2)

        weights_ravel = np.ravel(weights)

        data = np.histogram2d(
            wave_vectors_tile,
            eigenvalues_ravel,
            bins=bins,
            weights=weights_ravel,
        )[0]

        data = gaussian_filter(data, sigma=sigma)
        if powernorm:
            norm = colors.PowerNorm(gamma=gamma, vmin=np.min(data), vmax=np.max(data))
        else:
            norm = colors.Normalize(vmin=np.min(data), vmax=np.max(data))

        ax.pcolormesh(
            np.linspace(np.min(wave_vectors), np.max(wave_vectors), bins),
            np.linspace(np.min(eigenvalues), np.max(eigenvalues), bins),
            data.T,
            shading='gouraud',
            cmap=cmap,
            norm=norm,
        )

    def _alpha_cmap(color):
        cmap = LinearSegmentedColormap.from_list(
            'custom_cmap',
            [
                (1,1,1,0),
                to_rgba(color),
                to_rgba(color),
            ],
            N=10000
        )
        return cmap
        
    def plot_plain(
        self,
        ax,
        color='black',
        erange=[-6,6],
        linewidth=1.25,
        scale_factor=20,
        linestyle='-',
        heatmap=False,
        bins=400,
        sigma=3,
        cmap='hot',
        vlinecolor='black',
        powernorm=False,
        gamma=0.5,
        projection=None,
        highlight_band=False,
        highlight_band_color='red',
        band_index=None,
    ):
        """
        This function plots a plain band structure.

        Parameters:
            ax (matplotlib.pyplot.axis): Axis to plot the data on
            color (str): Color of the band structure lines
            linewidth (float): Line width of the band structure lines
            linestyle (str): Line style of the bands
        """
        bands_in_plot = self._filter_bands(erange=erange)
        slices = self._get_slices(unfold=self.unfold, hse=self.hse)
        wave_vector_segments = self._get_k_distance()

        if self.custom_kpath is not None:
            kpath_inds = self.custom_kpath_inds
            kpath_flip = self.custom_kpath_flip
        else:
            kpath_inds = range(len(slices))
            kpath_flip = [False for _ in range(len(slices))]

        for i, f, wave_vectors in zip(kpath_inds, kpath_flip, wave_vector_segments):
            if f:
                eigenvalues = np.flip(self.eigenvalues[bands_in_plot, slices[i]], axis=1)
            else:
                eigenvalues = self.eigenvalues[bands_in_plot, slices[i]]

            if highlight_band:
                if band_index is not None:
                    highlight_eigenvalues = self.eigenvalues[int(band_index), slices[i]]

            wave_vectors_for_kpoints = wave_vectors

            if self.interpolate:
                wave_vectors, eigenvalues = self._get_interpolated_data_segment(
                    wave_vectors_for_kpoints,
                    eigenvalues,
                )

                if highlight_band:
                    if band_index is not None:
                        _, highlight_eigenvalues = self._get_interpolated_data_segment(
                            wave_vectors_for_kpoints,
                            highlight_eigenvalues,
                        )

            eigenvalues_ravel = np.ravel(np.c_[eigenvalues, np.empty(eigenvalues.shape[0]) * np.nan])
            wave_vectors_tile = np.tile(np.append(wave_vectors, np.nan), eigenvalues.shape[0])

            if self.unfold:
                spectral_weights = self.spectral_weights[bands_in_plot, slices[i]]
                if f:
                    spectral_weights = np.flip(spectral_weights, axis=1)
                #  spectral_weights = spectral_weights / np.max(spectral_weights)

                if highlight_band:
                    if band_index is not None:
                        highlight_spectral_weights = self.spectral_weights[int(band_index), slices[i]]

                if self.interpolate:
                    _, spectral_weights = self._get_interpolated_data_segment(
                        wave_vectors_for_kpoints,
                        spectral_weights,
                        crop_zero=True,
                        kind='linear',
                    )

                    if highlight_band:
                        if band_index is not None:
                            _, highlight_spectral_weights = self._get_interpolated_data_segment(
                                wave_vectors_for_kpoints,
                                highlight_spectral_weights,
                                crop_zero=True,
                                kind='linear',
                            )
                
                spectral_weights_ravel = np.ravel(np.c_[spectral_weights, np.empty(spectral_weights.shape[0]) * np.nan])

                if heatmap:
                    self._heatmap(
                        ax=ax,
                        wave_vectors=wave_vectors,
                        eigenvalues=eigenvalues,
                        weights=spectral_weights,
                        sigma=sigma,
                        cmap=cmap,
                        bins=bins,
                        projection=projection,
                        powernorm=powernorm,
                        gamma=gamma,
                    )
                else:
                    ax.scatter(
                        wave_vectors_tile,
                        eigenvalues_ravel,
                        c=color,
                        ec=None,
                        s=scale_factor * spectral_weights_ravel,
                        zorder=0,
                    )
                    if highlight_band:
                        if band_index is not None:
                            ax.scatter(
                                wave_vectors,
                                highlight_eigenvalues,
                                c=highlight_band_color,
                                ec=None,
                                s=scale_factor * highlight_spectral_weights,
                                zorder=100,
                            )
            else:
                if heatmap:
                    self._heatmap(
                        ax=ax,
                        wave_vectors=wave_vectors,
                        eigenvalues=eigenvalues,
                        weights=np.ones(eigenvalues.shape),
                        sigma=sigma,
                        cmap=cmap,
                        bins=bins,
                        projection=projection,
                        powernorm=powernorm,
                        gamma=gamma,
                    )
                else:
                    ax.plot(
                        wave_vectors_tile,
                        eigenvalues_ravel,
                        color=color,
                        linewidth=linewidth,
                        linestyle=linestyle,
                        zorder=0,
                    )
                    if highlight_band:
                        if band_index is not None:
                            ax.plot(
                                wave_vectors,
                                highlight_eigenvalues,
                                color=highlight_band_color,
                                linewidth=linewidth,
                                linestyle=linestyle,
                                zorder=100,
                            )

        if self.hse:
            self._get_kticks_hse(
                ax=ax,
                wave_vectors=np.concatenate(self._get_k_distance()),
                kpath=self.kpath,
                vlinecolor=vlinecolor,
            )
        elif self.unfold:
            self._get_kticks_unfold(
                ax=ax,
                wave_vectors=np.concatenate(self._get_k_distance()),
                vlinecolor=vlinecolor,
            )
        else:
            self._get_kticks(
                ax=ax,
                wave_vectors=np.concatenate(self._get_k_distance()),
                vlinecolor=vlinecolor,
            )

        ax.set_xlim(0, np.max(self._get_k_distance()))


    def _plot_projected_general(
        self,
        ax,
        projected_data,
        colors,
        scale_factor=5,
        erange=[-6,6],
        display_order=None,
        linewidth=0.75,
        band_color='black',
        heatmap=False,
        bins=400,
        sigma=3,
        cmap='hot',
        vlinecolor='black',
        powernorm=False,
        gamma=0.5,
        plain_scale_factor=10,
    ):
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
        if self.unfold:
            if band_color == 'black':
                band_color = 'darkgrey'
            scale_factor = scale_factor * 4

        bands_in_plot = self._filter_bands(erange=erange)
        slices = self._get_slices(unfold=self.unfold, hse=self.hse)

        if self.unfold:
            K_indices = np.array(self.K_indices[0], dtype=int)
            projected_data = projected_data[:, K_indices, :]

        self.plot_plain(
            ax=ax,
            linewidth=linewidth,
            color=band_color,
            erange=erange,
            heatmap=heatmap,
            sigma=sigma,
            cmap=cmap,
            bins=bins,
            vlinecolor=vlinecolor,
            projection=projected_data,
            scale_factor=plain_scale_factor,
        )

        wave_vector_segments = self._get_k_distance()

        if self.custom_kpath is not None:
            kpath_inds = self.custom_kpath_inds
            kpath_flip = self.custom_kpath_flip
        else:
            kpath_inds = range(len(slices))
            kpath_flip = [False for _ in range(len(slices))]

        for i, f, wave_vectors in zip(kpath_inds, kpath_flip, wave_vector_segments):
            projected_data_slice = projected_data[bands_in_plot, slices[i]]
            if f:
                eigenvalues = np.flip(self.eigenvalues[bands_in_plot, slices[i]], axis=1)
                projected_data_slice = np.flip(projected_data_slice, axis=1)
            else:
                eigenvalues = self.eigenvalues[bands_in_plot, slices[i]]

            unique_colors = np.unique(colors)
            shapes = (projected_data_slice.shape[0], projected_data_slice.shape[1], projected_data_slice.shape[-1])
            projected_data_slice = projected_data_slice.reshape(shapes)

            if len(unique_colors) == len(colors):
                pass
            else:
                unique_inds = [np.isin(colors, c) for c in unique_colors]
                projected_data_slice = np.squeeze(projected_data_slice)
                projected_data_slice = np.c_[
                    [np.sum(projected_data_slice[...,u], axis=2) for u in unique_inds]
                ].transpose((1,2,0))
                plot_colors = unique_colors

            wave_vectors_old = wave_vectors

            if self.interpolate:
                wave_vectors, eigenvalues = self._get_interpolated_data_segment(wave_vectors_old, eigenvalues)
                _, projected_data_slice = self._get_interpolated_data_segment(
                    wave_vectors_old,
                    projected_data_slice,
                    crop_zero=True,
                    kind='linear',
                )


            if not heatmap:
                if self.unfold:
                    spectral_weights = self.spectral_weights[bands_in_plot, slices[i]]
                    if f:
                        spectral_weights = np.flip(spectral_weights, axis=1)
                    #  spectral_weights = spectral_weights / np.max(spectral_weights)

                    if self.interpolate:
                        _, spectral_weights = self._get_interpolated_data_segment(
                            wave_vectors_old,
                            spectral_weights,
                            crop_zero=True,
                            kind='linear',
                        )

                    spectral_weights_ravel = np.repeat(np.ravel(spectral_weights), projected_data_slice.shape[-1])

                projected_data_ravel = np.ravel(projected_data_slice)
                wave_vectors_tile = np.tile(
                    np.repeat(wave_vectors, projected_data_slice.shape[-1]), projected_data_slice.shape[0]
                )
                eigenvalues_tile = np.repeat(np.ravel(eigenvalues), projected_data_slice.shape[-1])
                colors_tile = np.tile(plot_colors, np.prod(projected_data_slice.shape[:-1]))

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

                    if self.unfold:
                        spectral_weights_ravel = spectral_weights_ravel[sort_index]

                if self.unfold:
                    s = scale_factor * projected_data_ravel * spectral_weights_ravel
                    ec = None
                else:
                    s = scale_factor * projected_data_ravel
                    ec = colors_tile

                ax.scatter(
                    wave_vectors_tile,
                    eigenvalues_tile,
                    c=colors_tile,
                    ec=ec,
                    s=s,
                    zorder=100,
                )



    def plot_plain_old(
        self,
        ax,
        color='black',
        erange=[-6,6],
        linewidth=1.25,
        scale_factor=20,
        linestyle='-',
        heatmap=False,
        bins=400,
        sigma=3,
        cmap='hot',
        vlinecolor='black',
        powernorm=False,
        gamma=0.5,
        projection=None,
        highlight_band=False,
        highlight_band_color='red',
        band_index=None,
    ):
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

        if highlight_band:
            if band_index is not None:
                highlight_eigenvalues = self.eigenvalues[int(band_index)]

        wave_vectors = self._get_k_distance()
        wave_vectors_for_kpoints = wave_vectors

        if self.interpolate:
            wave_vectors, eigenvalues = self._get_interpolated_data_segment(wave_vectors_for_kpoints, eigenvalues)

            if highlight_band:
                if band_index is not None:
                    _, highlight_eigenvalues = self._get_interpolated_data_segment(
                        wave_vectors_for_kpoints,
                        highlight_eigenvalues,
                    )
            


        eigenvalues_ravel = np.ravel(np.c_[eigenvalues, np.empty(eigenvalues.shape[0]) * np.nan])
        wave_vectors_tile = np.tile(np.append(wave_vectors, np.nan), eigenvalues.shape[0])

        if self.unfold:
            spectral_weights = self.spectral_weights[bands_in_plot]
            #  spectral_weights = spectral_weights / np.max(spectral_weights)

            if highlight_band:
                if band_index is not None:
                    highlight_spectral_weights = self.spectral_weights[int(band_index)]

            if self.interpolate:
                _, spectral_weights = self._get_interpolated_data_segment(
                    wave_vectors_for_kpoints,
                    spectral_weights,
                    crop_zero=True,
                    kind='linear',
                )

                if highlight_band:
                    if band_index is not None:
                        _, highlight_spectral_weights = self._get_interpolated_data_segment(
                            wave_vectors_for_kpoints,
                            highlight_spectral_weights,
                            crop_zero=True,
                            kind='linear',
                        )
            
            spectral_weights_ravel = np.ravel(np.c_[spectral_weights, np.empty(spectral_weights.shape[0]) * np.nan])

            if heatmap:
                self._heatmap(
                    ax=ax,
                    wave_vectors=wave_vectors,
                    eigenvalues=eigenvalues,
                    weights=spectral_weights,
                    sigma=sigma,
                    cmap=cmap,
                    bins=bins,
                    projection=projection,
                    powernorm=powernorm,
                    gamma=gamma,
                )
            else:
                ax.scatter(
                    wave_vectors_tile,
                    eigenvalues_ravel,
                    c=color,
                    ec=None,
                    s=scale_factor * spectral_weights_ravel,
                    zorder=0,
                )
                if highlight_band:
                    if band_index is not None:
                        ax.scatter(
                            wave_vectors,
                            highlight_eigenvalues,
                            c=highlight_band_color,
                            ec=None,
                            s=scale_factor * highlight_spectral_weights,
                            zorder=100,
                        )
        else:
            if heatmap:
                self._heatmap(
                    ax=ax,
                    wave_vectors=wave_vectors,
                    eigenvalues=eigenvalues,
                    weights=np.ones(eigenvalues.shape),
                    sigma=sigma,
                    cmap=cmap,
                    bins=bins,
                    projection=projection,
                    powernorm=powernorm,
                    gamma=gamma,
                )
            else:
                ax.plot(
                    wave_vectors_tile,
                    eigenvalues_ravel,
                    color=color,
                    linewidth=linewidth,
                    linestyle=linestyle,
                    zorder=0,
                )
                if highlight_band:
                    if band_index is not None:
                        ax.plot(
                            wave_vectors,
                            highlight_eigenvalues,
                            color=highlight_band_color,
                            linewidth=linewidth,
                            linestyle=linestyle,
                            zorder=100,
                        )

        if self.hse:
            self._get_kticks_hse(ax=ax, wave_vectors=wave_vectors_for_kpoints, kpath=self.kpath, vlinecolor=vlinecolor)
        elif self.unfold:
            self._get_kticks_unfold(ax=ax, wave_vectors=wave_vectors_for_kpoints, vlinecolor=vlinecolor)
        else:
            self._get_kticks(ax=ax, wave_vectors=wave_vectors_for_kpoints, vlinecolor=vlinecolor)

        ax.set_xlim(0, np.max(wave_vectors))


    def _plot_projected_general_old(
        self,
        ax,
        projected_data,
        colors,
        scale_factor=5,
        erange=[-6,6],
        display_order=None,
        linewidth=0.75,
        band_color='black',
        heatmap=False,
        bins=400,
        sigma=3,
        cmap='hot',
        vlinecolor='black',
        powernorm=False,
        gamma=0.5,
        plain_scale_factor=10,
    ):
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
        if self.unfold:
            if band_color == 'black':
                band_color = 'darkgrey'
            scale_factor = scale_factor * 4

        bands_in_plot = self._filter_bands(erange=erange)
        projected_data = projected_data[bands_in_plot]
        unique_colors = np.unique(colors)
        shapes = (projected_data.shape[0], projected_data.shape[1], projected_data.shape[-1])
        projected_data = projected_data.reshape(shapes)

        if len(unique_colors) == len(colors):
            pass
        else:
            unique_inds = [np.isin(colors, c) for c in unique_colors]
            projected_data = np.squeeze(projected_data)
            projected_data = np.c_[
                [np.sum(projected_data[...,i], axis=2) for i in unique_inds]
            ].transpose((1,2,0))
            colors = unique_colors

        #  projected_data = projected_data / np.max(projected_data)
        wave_vectors = self._get_k_distance()
        wave_vectors_old = wave_vectors
        eigenvalues = self.eigenvalues[bands_in_plot]

        if self.unfold:
            K_indices = np.array(self.K_indices[0], dtype=int)
            projected_data = projected_data[:, K_indices, :]

        if self.interpolate:
            wave_vectors, eigenvalues = self._get_interpolated_data(wave_vectors_old, eigenvalues)
            _, projected_data = self._get_interpolated_data(
                wave_vectors_old,
                projected_data,
                crop_zero=True,
                kind='linear',
            )

        self.plot_plain(
            ax=ax,
            linewidth=linewidth,
            color=band_color,
            erange=erange,
            heatmap=heatmap,
            sigma=sigma,
            cmap=cmap,
            bins=bins,
            vlinecolor=vlinecolor,
            projection=projected_data,
            scale_factor=plain_scale_factor,
        )

        if not heatmap:
            if self.unfold:
                spectral_weights = self.spectral_weights[bands_in_plot]
                spectral_weights = spectral_weights / np.max(spectral_weights)

                if self.interpolate:
                    _, spectral_weights = self._get_interpolated_data(
                        wave_vectors_old,
                        spectral_weights,
                        crop_zero=True,
                        kind='linear',
                    )

                spectral_weights_ravel = np.repeat(np.ravel(spectral_weights), projected_data.shape[-1])

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

                if self.unfold:
                    spectral_weights_ravel = spectral_weights_ravel[sort_index]

            if self.unfold:
                s = scale_factor * projected_data_ravel * spectral_weights_ravel
                ec = None
            else:
                s = scale_factor * projected_data_ravel
                ec = colors_tile

            ax.scatter(
                wave_vectors_tile,
                eigenvalues_tile,
                c=colors_tile,
                ec=ec,
                s=s,
                zorder=100,
            )
        
    def plot_orbitals(
        self,
        ax,
        orbitals,
        scale_factor=5,
        erange=[-6,6],
        display_order=None,
        color_list=None,
        legend=True,
        linewidth=0.75,
        band_color='black',
        heatmap=False,
        bins=400,
        sigma=3,
        cmap='hot',
        vlinecolor='black',
        powernorm=False,
        gamma=0.5,
    ):
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
            band_color=band_color,
            heatmap=heatmap,
            bins=bins,
            sigma=sigma,
            cmap=cmap,
            vlinecolor=vlinecolor,
        )

        if legend:
            self._add_legend(ax, names=[self.orbital_labels[i] for i in orbitals], colors=colors)


    def plot_spd(
        self,
        ax,
        scale_factor=5,
        orbitals='spd',
        erange=[-6,6],
        display_order=None,
        color_list=None,
        legend=True,
        linewidth=0.75,
        band_color='black',
        heatmap=False,
        bins=400,
        sigma=3,
        cmap='hot',
        vlinecolor='black',
        powernorm=False,
        gamma=0.5,
    ):
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
        if color_list is None:
            color_list = [
                self.color_dict[0],
                self.color_dict[1],
                self.color_dict[2],
                self.color_dict[4]
            ]
            colors = np.array(
                [color_list[i] for i in range(len(orbitals))]
            )
        else:
            colors = color_list

        projected_data = self._sum_spd(spd=orbitals)

        self._plot_projected_general(
            ax=ax,
            projected_data=projected_data,
            colors=colors,
            scale_factor=scale_factor,
            erange=erange,
            display_order=display_order,
            linewidth=linewidth,
            band_color=band_color,
            heatmap=heatmap,
            bins=bins,
            sigma=sigma,
            cmap=cmap,
            vlinecolor=vlinecolor,
        )

        if legend:
            self._add_legend(ax, names=[i for i in orbitals], colors=colors)



    def plot_atoms(
        self,
        ax,
        atoms,
        scale_factor=5,
        erange=[-6,6],
        display_order=None,
        color_list=None,
        legend=True,
        linewidth=0.75,
        band_color='black',
        heatmap=False,
        bins=400,
        sigma=3,
        cmap='hot',
        vlinecolor='black',
        powernorm=False,
        gamma=0.5,
    ):
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
            band_color=band_color,
            heatmap=heatmap,
            bins=bins,
            sigma=sigma,
            cmap=cmap,
            vlinecolor=vlinecolor,
        )

        if legend:
            self._add_legend(ax, names=atoms, colors=colors)


    def plot_atom_orbitals(
        self,
        ax,
        atom_orbital_dict,
        scale_factor=5,
        erange=[-6,6],
        display_order=None,
        color_list=None,
        legend=True,
        linewidth=0.75,
        band_color='black',
        heatmap=False,
        bins=400,
        sigma=3,
        cmap='hot',
        vlinecolor='black',
        powernorm=False,
        gamma=0.5,
    ):
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
            band_color=band_color,
            heatmap=heatmap,
            bins=bins,
            sigma=sigma,
            cmap=cmap,
            vlinecolor=vlinecolor,
        )

        if legend:
            self._add_legend(
                ax,
                names=[f'{i[0]}({i[1]})' for i in zip(atom_indices, orbital_symbols_long)],
                colors=colors
            )

    def plot_atom_spd(
        self,
        ax,
        atom_spd_dict,
        scale_factor=5,
        erange=[-6,6],
        display_order=None,
        color_list=None,
        legend=True,
        linewidth=0.75,
        band_color='black',
        heatmap=False,
        bins=400,
        sigma=3,
        cmap='hot',
        vlinecolor='black',
        powernorm=False,
        gamma=0.5,
    ):
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
            band_color=band_color,
            heatmap=heatmap,
            bins=bins,
            sigma=sigma,
            cmap=cmap,
            vlinecolor=vlinecolor,
        )

        if legend:
            self._add_legend(
                ax,
                names=[f'{i[0]}({i[1]})' for i in zip(atom_indices, orbital_symbols_long)],
                colors=colors
            )



    def plot_elements(
        self,
        ax,
        elements,
        scale_factor=5,
        erange=[-6,6],
        display_order=None,
        color_list=None,
        legend=True,
        linewidth=0.75,
        band_color='black',
        heatmap=False,
        bins=400,
        sigma=3,
        cmap='hot',
        vlinecolor='black',
        powernorm=False,
        gamma=0.5,
    ):
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
            band_color=band_color,
            heatmap=heatmap,
            bins=bins,
            sigma=sigma,
            cmap=cmap,
            vlinecolor=vlinecolor,
        )

        if legend:
            self._add_legend(ax, names=elements, colors=colors)


    def plot_element_orbitals(
        self,
        ax,
        element_orbital_dict,
        scale_factor=5,
        erange=[-6,6],
        display_order=None,
        color_list=None,
        legend=True,
        linewidth=0.75,
        band_color='black',
        heatmap=False,
        bins=400,
        sigma=3,
        cmap='hot',
        vlinecolor='black',
        powernorm=False,
        gamma=0.5,
    ):
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
            band_color=band_color,
            heatmap=heatmap,
            bins=bins,
            sigma=sigma,
            cmap=cmap,
            vlinecolor=vlinecolor,
        )

        if legend:
            self._add_legend(
                ax,
                names=[f'{i[0]}({i[1]})' for i in zip(element_symbols_long, orbital_symbols_long)],
                colors=colors
            )

    def plot_element_spd(
        self,
        ax,
        element_spd_dict,
        scale_factor=5,
        erange=[-6,6],
        display_order=None,
        color_list=None,
        legend=True,
        linewidth=0.75,
        band_color='black',
        heatmap=False,
        bins=400,
        sigma=3,
        cmap='hot',
        vlinecolor='black',
        powernorm=False,
        gamma=0.5,
    ):
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
            linewidth (float):12 Line width of the plain band structure plotted in the background
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
            band_color=band_color,
            heatmap=heatmap,
            bins=bins,
            sigma=sigma,
            cmap=cmap,
            vlinecolor=vlinecolor,
        )

        if legend:
            self._add_legend(
                ax,
                names=[f'{i[0]}({i[1]})' for i in zip(element_symbols_long, orbital_symbols_long)],
                colors=colors
            )


if __name__ == "__main__":
    M = [
        [0,1,-1],
        [1,-1,0],
        [-14,-14,-14]
    ]
#
    #  high_symm_points = [
        #  [2/3, 1/3, 1/3],
        #  [0.0, 0.0, 0],
        #  [2/3, 1/3, 1/3],
    #  ]
#
    high_symm_points = [
        [0.1, 0.1, 0],
        [0.0, 0.0, 0],
        [0.1, 0.1, 0],
    ]

    band = Band(
        folder="../../vaspvis_data/bandAGA",
        projected=True,
        interpolate=False,
        unfold=True,
        M=M,
        high_symm_points=high_symm_points,
        n=40,
        kpath=[['A', 'G'], ['G', 'A']],
        custom_kpath=[1,2,2,-1],
    )
    fig, ax = plt.subplots(figsize=(4,3), dpi=400)
    band.plot_spd(ax=ax, scale_factor=100)
    ax.set_ylim(-2,2)
    fig.tight_layout()
    fig.savefig('test.png')
    #  band._get_bandgap()
    #  #  band.plot_spd(ax=ax, orbitals='sd', display_order='all', scale_factor=35, erange=[-5,0])
    #  #  band.plot_orbitals(ax=ax, scale_factor=35, orbitals=range(8), display_order=None)
    #  band.plot_plain(
        #  ax=ax,
        #  scale_factor=20,
        #  erange=[-4,0.5],
        #  heatmap=True,
        #  cmap='hot',
        #  bins=800,
        #  sigma=3,
        #  powernorm=False,
        #  gamma=0.5,
        #  vlinecolor='white'
    #  )
    #  #  ax.set_aspect(3, adjustable='datalim')
    #  end = time.time()
    #  ax.set_ylabel('$E - E_{F}$ $(eV)$', fontsize=6)
    #  ax.tick_params(labelsize=6, length=2.5)
    #  ax.tick_params(axis='x', length=0)
    #  ax.set_ylim(-4, 0.5)
    #  plt.tight_layout(pad=0.2)
    #  plt.savefig('heatmap_powernorm.png')
        
        



