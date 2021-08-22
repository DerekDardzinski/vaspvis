import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, LogLocator
from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.io.vasp.inputs import Poscar, Incar
from pymatgen.electronic_structure.core import Spin, Orbital
from pymatgen.core.periodic_table import Element
from pychemia.code.vasp.doscar import VaspDoscar
from scipy.ndimage.filters import gaussian_filter1d
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp2d
from functools import reduce
import numpy as np
import pandas as pd
from ase.visualize.plot import plot_atoms
from pymatgen.io.ase import AseAtomsAdaptor
import copy
import time
import os

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

class Dos:
    """
    This class contains all the methods for contructing density of states plots from the outputs of VASP calculations.

    Parameters:
        folder (str): This is the folder that contains the VASP files.
        spin (str): Which spin direction to parse ('up' or 'down')
        combination_method (str): If the spin option is 'both', the combination method can either be additive or substractive
            by passing 'add' or 'sub'. It spin is passed as 'up' or 'down' this option is ignored.
    """

    def __init__(
            self,
            folder,
            spin='up',
            soc_axis=None,
            combination_method="add",
            sp_method='percentage',
            shift_efermi=0,
    ):
        self.folder = folder
        self.spin = spin
        self.soc_axis = soc_axis
        self.combination_method = combination_method
        self.sp_method = sp_method
        self.incar = Incar.from_file(
            os.path.join(folder, 'INCAR')
        )
        if 'LORBIT' in self.incar:
            if self.incar['LORBIT'] >= 11:
                self.lorbit = True
            else:
                self.lorbit = False
        else:
            self.lorbit = False

        if self.lorbit:
            if os.path.isfile(os.path.join(folder, 'dos.npy')) and os.path.isfile(os.path.join(folder, 'projected_dos.npy')):
                with open(os.path.join(folder, 'dos.npy'), 'rb') as dos_file:
                    dos = np.load(dos_file)
                with open(os.path.join(folder, 'projected_dos.npy'), 'rb') as projected_dos_file:
                    projected_dos = np.load(projected_dos_file)

                self.doscar = {
                    'total': dos,
                    'projected': projected_dos,
                }
            else:
                if self._check_f_error():
                    self._fix_doscar()

                self.doscar = VaspDoscar.parse_doscar(os.path.join(folder, 'DOSCAR'))
                np.save(os.path.join(folder, 'dos.npy'), self.doscar['total'])
                np.save(os.path.join(folder, 'projected_dos.npy'), self.doscar['projected'])
        else:
            if os.path.isfile(os.path.join(folder, 'dos.npy')):
                with open(os.path.join(folder, 'dos.npy'), 'rb') as dos_file:
                    dos = np.load(dos_file)

                self.doscar = {
                    'total': dos,
                }
            else:
                self.doscar = VaspDoscar.parse_doscar(os.path.join(folder, 'DOSCAR'))
                np.save(os.path.join(folder, 'dos.npy'), self.doscar['total'])

        self.efermi = float(os.popen(f'grep E-fermi {os.path.join(folder, "OUTCAR")}').read().split()[2]) + shift_efermi
        self.poscar = Poscar.from_file(
            os.path.join(folder, 'POSCAR'),
            check_for_POTCAR=False,
            read_velocities=False
        )
        self.forbitals = self._check_f_orb()
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

        self.spin_dict = {'up': Spin.up, 'down': Spin.down}

        self.tdos_array = self._load_tdos()

        if self.lorbit:
            self.pdos_array = self._load_pdos()

            if self.lsorbit and self.soc_axis is not None:
                self.tdos_array[:,1] = self.pdos_array.sum(axis=1).sum(axis=1)


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

    def _check_f_error(self):
        with open(os.path.join(self.folder, 'DOSCAR'), 'rb') as f:
            f.seek(-2, os.SEEK_END)
            while f.read(1) != b'\n':
                f.seek(-2, os.SEEK_CUR)
            last_line = f.readline().decode()

        last_line_len = len(last_line.split())

        if last_line_len == 28:
            return True
        else:
            return False

    def _fix_doscar(self):
        doscar = []
        with open(os.path.join(self.folder, 'DOSCAR')) as f:
            for line in f:
                split_line = line.split()
                doscar.append(split_line)

        num_atoms = int(doscar[0][1])
        nedos = int(doscar[5][2])
        nedos_f = 2 * nedos
        start_inds = nedos + 7

        top_file = []

        with open(os.path.join(self.folder, 'DOSCAR')) as f:
            count = 0
            for line in f:
                top_file.append(line)
                count += 1
                if count == start_inds:
                    break

        a = np.c_[[np.arange(0,nedos_f-1,2),np.arange(1,nedos_f,2)]].T
        a = np.c_[[a for _ in range(num_atoms)]]
        b = np.array([1] + [nedos_f for _ in range(num_atoms-1)])
        c = np.arange(num_atoms)
        d = np.arange(num_atoms)
        d[0] = 1
        inds = a + (b*c)[:,None,None] + c[:,None,None]
        inds += start_inds

        new_list = []

        for i, ind in enumerate(inds):
            inbetween_ind = np.max(ind) + 1
            for j in ind:
                new_list.append('\t' + '  '.join(doscar[j[0]] + doscar[j[1]]))

            if i != inds.shape[0]-1:
                new_list.append('\t' + '    '.join(doscar[inbetween_ind]))

        new_doscar = ''.join([''.join(top_file), '\n'.join(new_list)])

        os.rename(os.path.join(self.folder, 'DOSCAR'), os.path.join(self.folder, 'DOSCAR_old'))

        with open(os.path.join(self.folder, 'DOSCAR'), 'w') as x:
            x.write(new_doscar)

    def _load_tdos(self):
        """
        This function loads the total density of states into a dictionary

        Returns:
            tdos_dict (dict[str][np.ndarray]): Dictionary that consists or the
                energies and densities of the system.
        """

        tdos = self.doscar['total']
        tdos[:,0] = tdos[:,0] - self.efermi

        if self.spin == 'up':
            tdos = tdos[:,:2]
        elif self.spin == 'down':
            tdos = tdos[:,[0,2]]
            tdos[:,1] = -tdos[:,1]
        elif self.spin == 'both':
            tdos_up = tdos[:,1]
            tdos_down = tdos[:,2]
            if self.combination_method == "add":
                tdos = np.c_[tdos[:,0], tdos_up + tdos_down]
            if self.combination_method == "sub":
                if self.sp_method == 'percentage':
                    tdos = np.c_[tdos[:,0], (tdos_up - tdos_down) / (tdos_up + tdos_down)]
                elif self.sp_method == 'absolute':
                    tdos = np.c_[tdos[:,0], tdos_up - tdos_down]

        return tdos


    def _load_pdos(self):
        """
        This function loads the projected density of states into a dictionary
        of the form:
        atom index --> orbital projections

        Returns:
            pdos_dict (dict[int][pd.DataFrame]): Dictionary that contains a data frame
                with the orbital weights for each atom index.
        """

        pdos = self.doscar['projected']
        pdos = np.transpose(pdos, axes=(1,0,2))

        if self.spin == 'up':
            if not self.forbitals:
                if self.lsorbit:
                    if self.soc_axis is None:
                        pdos = pdos[:,:,[(j*4) + 1 for j in range(9)]]
                    elif self.soc_axis == 'x':
                        pdos = pdos[:,:,[(j*4) + 2 for j in range(9)]]
                    elif self.soc_axis == 'y':
                        pdos = pdos[:,:,[(j*4) + 3 for j in range(9)]]
                    elif self.soc_axis == 'z':
                        pdos = pdos[:,:,[(j*4) + 4 for j in range(9)]]

                    if self.soc_axis is not None:
                        pdos_up = np.zeros(pdos.shape)
                        pdos_up[np.where(pdos > 0)] = pdos[np.where(pdos > 0)]
                        pdos = pdos_up

                elif self.ispin and not self.lsorbit:
                    pdos = pdos[:,:,[(j*2) + 1 for j in range(9)]]
                else:
                    pdos = pdos[:,:,1:]
            else:
                if self.lsorbit:
                    if self.soc_axis is None:
                        pdos = pdos[:,:,[(j*4) + 1 for j in range(16)]]
                    if self.soc_axis == 'x':
                        pdos = pdos[:,:,[(j*4) + 2 for j in range(16)]]
                    if self.soc_axis == 'y':
                        pdos = pdos[:,:,[(j*4) + 3 for j in range(16)]]
                    if self.soc_axis == 'z':
                        pdos = pdos[:,:,[(j*4) + 4 for j in range(16)]]

                    if self.soc_axis is not None:
                        pdos_up = np.zeros(pdos.shape)
                        pdos_up[np.where(pdos > 0)] = pdos[np.where(pdos > 0)]
                        pdos = pdos_up

                elif self.ispin and not self.lsorbit:
                    pdos = pdos[:,:,[(j*2) + 1 for j in range(16)]]
                else:
                    pdos = pdos[:,:,1:]
        if self.spin == 'down':
            if not self.forbitals:
                if self.lsorbit:
                    if self.soc_axis is None:
                        raise("You have selected spin='down' for a SOC calculation, but soc_axis has not been selected. Please set soc_axis to 'x', 'y', or 'z' for this function to work.")
                    elif self.soc_axis == 'x':
                        pdos = pdos[:,:,[(j*4) + 2 for j in range(9)]]
                    elif self.soc_axis == 'y':
                        pdos = pdos[:,:,[(j*4) + 3 for j in range(9)]]
                    elif self.soc_axis == 'z':
                        pdos = pdos[:,:,[(j*4) + 4 for j in range(9)]]

                    if self.soc_axis is not None:
                        pdos_down = np.zeros(pdos.shape)
                        pdos_down[np.where(pdos < 0)] = pdos[np.where(pdos < 0)]
                        pdos = pdos_down

                elif self.ispin and not self.lsorbit:
                    pdos = -pdos[:,:,[(j*2) + 2 for j in range(9)]]
            else:
                if self.lsorbit:
                    if self.soc_axis is None:
                        raise("You have selected spin='down' for a SOC calculation, but soc_axis has not been selected. Please set soc_axis to 'x', 'y', or 'z' for this function to work.")
                    if self.soc_axis == 'x':
                        pdos = pdos[:,:,[(j*4) + 2 for j in range(16)]]
                    if self.soc_axis == 'y':
                        pdos = pdos[:,:,[(j*4) + 3 for j in range(16)]]
                    if self.soc_axis == 'z':
                        pdos = pdos[:,:,[(j*4) + 4 for j in range(16)]]

                    if self.soc_axis is not None:
                        pdos_down = np.zeros(pdos.shape)
                        pdos_down[np.where(pdos < 0)] = pdos[np.where(pdos < 0)]
                        pdos = pdos_down

                elif self.ispin and not self.lsorbit:
                    pdos = -pdos[:,:,[(j*2) + 2 for j in range(16)]]
        if self.spin == 'both':
            if not self.forbitals:
                if self.lsorbit:
                    if self.soc_axis is None:
                        raise("You have selected spin='down' for a SOC calculation, but soc_axis has not been selected. Please set soc_axis to 'x', 'y', or 'z' for this function to work.")
                    elif self.soc_axis == 'x':
                        pdos = pdos[:,:,[(j*4) + 2 for j in range(9)]]
                    elif self.soc_axis == 'y':
                        pdos = pdos[:,:,[(j*4) + 3 for j in range(9)]]
                    elif self.soc_axis == 'z':
                        pdos = pdos[:,:,[(j*4) + 4 for j in range(9)]]

                    if self.soc_axis is not None:
                        pdos_up = np.zeros(pdos.shape)
                        pdos_up[np.where(pdos > 0)] = pdos[np.where(pdos > 0)]
                        pdos_down = np.zeros(pdos.shape)
                        pdos_down[np.where(pdos < 0)] = -pdos[np.where(pdos < 0)]

                elif self.ispin and not self.lsorbit:
                    pdos_up = pdos[:,:,[(j*2) + 1 for j in range(9)]]
                    pdos_down = pdos[:,:,[(j*2) + 2 for j in range(9)]]
            else:
                if self.lsorbit:
                    if self.soc_axis is None:
                        raise("You have selected spin='down' for a SOC calculation, but soc_axis has not been selected. Please set soc_axis to 'x', 'y', or 'z' for this function to work.")
                    if self.soc_axis == 'x':
                        pdos = pdos[:,:,[(j*4) + 2 for j in range(16)]]
                    if self.soc_axis == 'y':
                        pdos = pdos[:,:,[(j*4) + 3 for j in range(16)]]
                    if self.soc_axis == 'z':
                        pdos = pdos[:,:,[(j*4) + 4 for j in range(16)]]

                    if self.soc_axis is not None:
                        pdos_up = np.zeros(pdos.shape)
                        pdos_up[np.where(pdos > 0)] = pdos[np.where(pdos > 0)]
                        pdos_down = np.zeros(pdos.shape)
                        pdos_down[np.where(pdos < 0)] = -pdos[np.where(pdos < 0)]

                elif self.ispin and not self.lsorbit:
                    pdos_up = pdos[:,:,[(j*2) + 1 for j in range(16)]]
                    pdos_down = pdos[:,:,[(j*2) + 2 for j in range(16)]]

            if self.combination_method == 'add':
                pdos = pdos_up + pdos_down
            if self.combination_method == 'sub':
                if self.sp_method == 'percentage':
                    pdos = np.array([pdos_up, pdos_down])
                    #  pdos = (pdos_up - pdos_down) / (pdos_up + pdos_down)
                elif self.sp_method == 'absolute':
                    pdos = pdos_up - pdos_down

        return pdos


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


        if self.spin == 'both' and self.combination_method == 'sub' and self.sp_method == 'percentage':
            orbital_contributions_up = np.sum(self.pdos_array[0], axis=1)
            orbital_contributions_down = np.sum(self.pdos_array[1], axis=1)

            spd_contributions_up = np.transpose(
                np.array([
                    np.sum(orbital_contributions_up[:,ind], axis=1) for ind in spd_indices
                ]), axes=[1,0]
            )
            spd_contributions_down = np.transpose(
                np.array([
                    np.sum(orbital_contributions_down[:,ind], axis=1) for ind in spd_indices
                ]), axes=[1,0]
            )

            spd_contributions_up = spd_contributions_up[:,[self.spd_relations[orb] for orb in spd]]
            spd_contributions_down = spd_contributions_down[:,[self.spd_relations[orb] for orb in spd]]

            #  spd_contributions = (spd_contributions_up - spd_contributions_down) / (spd_contributions_up + spd_contributions_down)
            spd_contributions = np.array([spd_contributions_up, spd_contributions_down])

        else:
            orbital_contributions = np.sum(self.pdos_array, axis=1)

            spd_contributions = np.transpose(
                np.array([
                    np.sum(orbital_contributions[:,ind], axis=1) for ind in spd_indices
                ]), axes=[1,0]
            )

            spd_contributions = spd_contributions[:,[self.spd_relations[orb] for orb in spd]]

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
        if self.spin == 'both' and self.combination_method == 'sub' and self.sp_method == 'percentage':
            orbital_contributions_up = self.pdos_array[0].sum(axis=1)
            orbital_contributions_down = self.pdos_array[1].sum(axis=1)

            orbital_contributions_up = orbital_contributions_up[:,orbitals]
            orbital_contributions_down = orbital_contributions_down[:,orbitals]

            #  orbital_contributions = (orbital_contributions_up - orbital_contributions_down) / (orbital_contributions_up + orbital_contributions_down)
            orbital_contributions = np.array([orbital_contributions_up, orbital_contributions_down])
        else:
            orbital_contributions = self.pdos_array.sum(axis=1)
            orbital_contributions = orbital_contributions[:,orbitals]

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

            if self.spin == 'both' and self.combination_method == 'sub' and self.sp_method == 'percentage':
                atoms_spd_up = self.pdos_array[0]
                atoms_spd_down = self.pdos_array[1]
                atoms_spd_up = np.transpose(np.array([
                    np.sum(atoms_spd_up[:,:,ind], axis=2) for ind in spd_indices
                ]), axes=(1,2,0))
                atoms_spd_down = np.transpose(np.array([
                    np.sum(atoms_spd_down[:,:,ind], axis=2) for ind in spd_indices
                ]), axes=(1,2,0))
                #  atoms_spd = (atoms_spd_up - atoms_spd_down) / (atoms_spd_up + atoms_spd_down)
                atoms_spd = np.array([atoms_spd_up, atoms_spd_down])
            else:
                atoms_spd = np.transpose(np.array([
                    np.sum(self.pdos_array[:,:,ind], axis=2) for ind in spd_indices
                ]), axes=(1,2,0))

            return atoms_spd
        else:
            if self.spin == 'both' and self.combination_method == 'sub' and self.sp_method == 'percentage':
                atoms_array_up = self.pdos_array[0].sum(axis=2)
                atoms_array_down = self.pdos_array[1].sum(axis=2)
                #  atoms_array = (atoms_array_up - atoms_array_down) / (atoms_array_up + atoms_array_down)
                if atoms is not None:
                    atoms_array_up = atoms_array_up[:, atoms]
                    atoms_array_down = atoms_array_down[:, atoms]

                atoms_array = np.array([atoms_array_up, atoms_array_down])
            else:
                atoms_array = self.pdos_array.sum(axis=2)

                if atoms is not None:
                    atoms_array = atoms_array[:,atoms]

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


        if self.spin == 'both' and self.combination_method == 'sub' and self.sp_method == 'percentage':
            pdos_array_up = self.pdos_array[0]
            pdos_array_down = self.pdos_array[1]

            element_list = np.hstack(
                [[symbols[i] for j in range(natoms[i])] for i in range(len(symbols))]
            )

            element_indices = [np.where(np.isin(element_list, element))[0] for element in elements]

            element_orbitals_up = np.transpose(
                np.array([
                    np.sum(pdos_array_up[:,ind,:], axis=1) for ind in element_indices
                ]), axes=(1,0,2)
            )
            element_orbitals_down = np.transpose(
                np.array([
                    np.sum(pdos_array_down[:,ind,:], axis=1) for ind in element_indices
                ]), axes=(1,0,2)
            )

            if orbitals:
                #  element_orbitals = (element_orbitals_up - element_orbitals_down) / (element_orbitals_up + element_orbitals_down)
                element_orbitals = np.array([element_orbitals_up, element_orbitals_down])
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

                element_spd_up = np.transpose(np.array([
                    np.sum(element_orbitals_up[:,:,ind], axis=2) for ind in spd_indices
                ]), axes=(1,2,0))

                element_spd_down = np.transpose(np.array([
                    np.sum(element_orbitals_down[:,:,ind], axis=2) for ind in spd_indices
                ]), axes=(1,2,0))

                #  element_spd = (element_spd_up - element_spd_down) / (element_spd_up + element_spd_down)
                element_spd = np.array([element_spd_up, element_spd_down])

                return element_spd
            else:
                element_array_up = np.sum(element_orbitals_up, axis=2)
                element_array_down = np.sum(element_orbitals_down, axis=2)

                #  element_array = (element_array_up - element_array_down) / (element_array_up + element_array_down)
                element_array = np.array([element_array_up, element_array_down])

                return element_array
        else:
            pdos_array = self.pdos_array

            element_list = np.hstack(
                [[symbols[i] for j in range(natoms[i])] for i in range(len(symbols))]
            )

            element_indices = [np.where(np.isin(element_list, element))[0] for element in elements]

            element_orbitals = np.transpose(
                np.array([
                    np.sum(pdos_array[:,ind,:], axis=1) for ind in element_indices
                ]), axes=(1,0,2)
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
                    np.sum(element_orbitals[:,:,ind], axis=2) for ind in spd_indices
                ]), axes=(1,2,0))

                return element_spd
            else:
                element_array = np.sum(element_orbitals, axis=2)

                return element_array


    def _smear(self, dos, sigma):
        """
        This function applied a 1D gaussian filter to the density of states

        Parameters:
            dos (np.ndarray): Array of densities.
            sigma (float): Standard deviation used in the gaussian filter.


        Returns:
            _smeared_dos (np.ndarray): Array of _smeared densities.
        """

        diff = np.diff(self.tdos_array[:,0])
        avgdiff = np.mean(diff)
        _smeared_dos = gaussian_filter1d(dos, sigma / avgdiff)

        return _smeared_dos

    def _set_density_lims(
        self,
        ax,
        tdensity,
        tenergy,
        erange,
        energyaxis,
        spin,
        partial=False,
        is_dict=False,
        idx=None,
        multiple=False,
        log_scale=False
    ):
        energy_in_plot_index = np.where(
            (tenergy >= erange[0]) & (tenergy <= erange[1])
        )[0]

        tdensity = tdensity[energy_in_plot_index]

        if len(np.squeeze(tdensity).shape) == 1:
           density_in_plot = np.squeeze(tdensity) 
        else:
            if spin == 'up' or spin == 'both':
                max_index = np.argmax(np.max(tdensity, axis=0))
                density_in_plot = tdensity[:,max_index]
            else:
                min_index = np.argmin(np.min(tdensity, axis=0))
                density_in_plot = tdensity[:,min_index]

        if len(ax.lines) == 0:
            if energyaxis == 'y':
                ax.set_ylim(erange)
                if spin == 'up' or spin == 'both':
                    ax.set_xlim(0, np.max(density_in_plot) * 1.1)
                    if log_scale:
                         ax.set_xlim(np.min(density_in_plot), np.max(density_in_plot) + np.abs(np.max(density_in_plot) * 0.1))
                    else:
                        ax.set_xlim(0, np.max(density_in_plot) * 1.1)
                elif spin == 'down':
                    ax.set_xlim(np.min(density_in_plot) * 1.1, 0)
            elif energyaxis == 'x':
                ax.set_xlim(erange)
                if spin == 'up' or spin == 'both':
                    if log_scale:
                         ax.set_ylim(np.min(density_in_plot), np.max(density_in_plot) + np.abs(np.max(density_in_plot) * 0.1))
                    else:
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

    def _sum_layers(self, layers, atol=None, custom_layer_inds=None):
        from vaspvis.utils import group_layers
        if custom_layer_inds is None:
            groups, _ = group_layers(self.poscar.structure, atol=atol)
        else:
            groups = custom_layer_inds
        atom_densities = self._sum_atoms(atoms=None)

        if self.spin == 'both' and self.combination_method == 'sub' and self.sp_method == 'percentage':
            densities_up = np.vstack([np.sum(np.vstack(atom_densities[0, :,[group]]), axis=1) for group in groups])
            densities_down = np.vstack([np.sum(np.vstack(atom_densities[1, :,[group]]), axis=1) for group in groups])
            summed_layers_up = np.sum(densities_up[layers], axis=0)
            summed_layers_down = np.sum(densities_down[layers], axis=0)
            summed_layers = np.array([summed_layers_up, summed_layers_down])
        else:
            densities = np.vstack([np.sum(np.vstack(atom_densities[:,[group]]), axis=1) for group in groups])
            summed_layers = np.sum(densities[layers], axis=0)

        return summed_layers

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


    def _plot_projected_general(self, ax, energy, projected_data, colors, sigma, erange, linewidth, alpha_line, alpha, fill, energyaxis, total):
        energy_in_plot_index = np.where(
            (energy >= erange[0] - 0.5) & (energy <= erange[1] + 0.5)
        )[0]
        energy = energy[energy_in_plot_index]

        if self.spin == 'both' and self.combination_method == 'sub' and self.sp_method == 'percentage':
            projected_data_up = projected_data[0]
            projected_data_down = projected_data[1]

            projected_data_up = projected_data_up[energy_in_plot_index]
            projected_data_down = projected_data_down[energy_in_plot_index]

            unique_colors = np.unique(colors)

            if len(unique_colors) == len(colors):
                plot_colors = colors
            else:
                unique_inds = [np.isin(colors, c) for c in unique_colors]
                projected_data_up = np.c_[
                    [np.sum(projected_data_up[:,i], axis=1) for i in unique_inds]
                ].transpose()
                projected_data_down = np.c_[
                    [np.sum(projected_data_down[:,i], axis=1) for i in unique_inds]
                ].transpose()
                plot_colors = unique_colors

            projected_data = (projected_data_up - projected_data_down) / (projected_data_up + projected_data_down)

            if sigma > 0:
                for i in range(projected_data.shape[-1]):
                    projected_data[:,i] = self._smear(
                        projected_data[:,i],
                        sigma=sigma,
                    )
        else:
            projected_data = projected_data[energy_in_plot_index]

            unique_colors = np.unique(colors)

            if len(unique_colors) == len(colors):
                plot_colors = colors
            else:
                unique_inds = [np.isin(colors, c) for c in unique_colors]
                projected_data = np.c_[
                    [np.sum(projected_data[:,i], axis=1) for i in unique_inds]
                ].transpose()
                plot_colors = unique_colors

            if sigma > 0:
                for i in range(projected_data.shape[-1]):
                    projected_data[:,i] = self._smear(
                        projected_data[:,i],
                        sigma=sigma,
                    )

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
                tdensity=projected_data,
                tenergy=energy,
                erange=erange,
                energyaxis=energyaxis,
                spin=self.spin,
                partial=True,
            )

        for i in range(projected_data.shape[-1]):

            pdensity = projected_data[:,i]

            if energyaxis == 'y':
                ax.plot(
                    pdensity,
                    energy,
                    color=plot_colors[i],
                    linewidth=linewidth,
                    alpha=alpha_line
                )

                if fill:
                    ax.fill_betweenx(
                        energy,
                        pdensity,
                        0,
                        color=plot_colors[i],
                        alpha=alpha,
                    )

            if energyaxis == 'x':
                ax.plot(
                    energy,
                    pdensity,
                    color=plot_colors[i],
                    linewidth=linewidth,
                    alpha=alpha_line
                )

                if fill:
                    ax.fill_between(
                        energy,
                        pdensity,
                        0,
                        color=plot_colors[i],
                        alpha=alpha,
                    )


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

        tdos_array = self.tdos_array
        energy_in_plot_index = np.where(
            (tdos_array[:,0] >= erange[0] - 0.5) & (tdos_array[:,0] <= erange[1] + 0.5)
        )[0]
        tdos_array = tdos_array[energy_in_plot_index]

        if sigma > 0:
            tdensity = self._smear(
                tdos_array[:,1],
                sigma=sigma
            )
        else:
            tdensity = tdos_array[:,1]

        self._set_density_lims(
            ax=ax,
            tdensity=tdensity,
            tenergy=tdos_array[:,0],
            erange=erange,
            energyaxis=energyaxis,
            spin=self.spin,
        )

        if energyaxis == 'y':
            ax.plot(
                tdensity,
                tdos_array[:,0],
                linewidth=linewidth,
                color=color,
                alpha=alpha_line
            )

            if fill:
                ax.fill_betweenx(
                    tdos_array[:,0],
                    tdensity,
                    0,
                    alpha=alpha,
                    color=color,
                )

        if energyaxis == 'x':
            ax.plot(
                tdos_array[:,0],
                tdensity,
                linewidth=linewidth,
                color=color,
                alpha=alpha_line
            )

            if fill:
                ax.fill_between(
                    tdos_array[:,0],
                    tdensity,
                    0,
                    color=color,
                    alpha=alpha,
                )

    def plot_ldos(
        self,
        ax,
        layers,
        linewidth=1.5,
        fill=False,
        alpha=0.3,
        alpha_line=1.0,
        sigma=0.05,
        energyaxis='x',
        color='black',
        log_scale=False,
        erange=[-6, 6],
        atol=None,
        custom_layer_inds=None,
    ):
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

        #  tdos_array = self._sum_layers(layers=layers)
        tdos_array = self.tdos_array

        tdensity = self._sum_layers(
            layers=layers,
            atol=atol,
            custom_layer_inds=custom_layer_inds
        )

        if self.spin == 'both' and self.combination_method == 'sub' and self.sp_method == 'percentage':
            tdensity = (tdensity[0] - tdensity[1]) / (tdensity[0] + tdensity[1])

        if sigma > 0:
            tdensity = self._smear(tdensity, sigma=sigma)


        if log_scale:
            tdensity = np.log10(tdensity)
            neg_inf_loc = np.isin(tdensity, -np.inf)
            min_val = np.min(tdensity[np.logical_not(neg_inf_loc)])
            tdensity[neg_inf_loc] = min_val

        self._set_density_lims(
            ax=ax,
            tdensity=tdensity,
            tenergy=tdos_array[:,0],
            erange=erange,
            energyaxis=energyaxis,
            spin=self.spin,
            log_scale=log_scale,
        )

        if energyaxis == 'y':
            ax.plot(
                tdensity,
                tdos_array[:,0],
                linewidth=linewidth,
                color=color,
                alpha=alpha_line
            )

            if fill:
                ax.fill_betweenx(
                    tdos_array[:,0],
                    tdensity,
                    0,
                    alpha=alpha,
                    color=color,
                )

        if energyaxis == 'x':
            ax.plot(
                tdos_array[:,0],
                tdensity,
                linewidth=linewidth,
                color=color,
                alpha=alpha_line
            )

            if fill:
                ax.fill_between(
                    tdos_array[:,0],
                    tdensity,
                    0,
                    color=color,
                    alpha=alpha,
                )

    def plot_spd(self, ax, orbitals='spd', fill=True, alpha=0.3, alpha_line=1.0, linewidth=1.5, sigma=0.05, energyaxis='y', color_list=None, legend=True, total=True, erange=[-6, 6]):
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

        projected_data = self._sum_spd(spd=orbitals)

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

        self._plot_projected_general(
            ax=ax,
            energy=self.tdos_array[:,0],
            projected_data=projected_data,
            colors=colors,
            sigma=sigma,
            erange=erange,
            linewidth=linewidth,
            alpha_line=alpha_line,
            alpha=alpha,
            fill=fill,
            energyaxis=energyaxis,
            total=total,
        )

        if legend:
            self._add_legend(ax, names=[i for i in orbitals], colors=colors)

    def plot_atom_orbitals(self, ax, atom_orbital_dict, fill=True, alpha=0.3, alpha_line=1.0, linewidth=1.5, sigma=0.05, energyaxis='y', color_list=None, legend=True, total=True, erange=[-6, 6]):
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

        atom_indices = list(atom_orbital_dict.keys())
        orbital_indices = list(atom_orbital_dict.values())
        number_orbitals = [len(i) for i in orbital_indices]
        atom_indices = np.repeat(atom_indices, number_orbitals)
        orbital_symbols_long = np.hstack([
            [self.orbital_labels[o] for o in  orb] for orb in orbital_indices
        ])
        orbital_indices_long = np.hstack(orbital_indices)
        indices = np.vstack([atom_indices, orbital_indices_long]).T

        projected_data = self.pdos_array

        if self.spin == 'both' and self.combination_method == 'sub' and self.sp_method == 'percentage':
            projected_data_up = np.transpose(np.array([
                projected_data[0,:,ind[0],ind[1]] for ind in indices
            ]), axes=(1,0))
            projected_data_down = np.transpose(np.array([
                projected_data[1,:,ind[0],ind[1]] for ind in indices
            ]), axes=(1,0))
            projected_data = np.array([projected_data_up, projected_data_down])
        else:
            projected_data = np.transpose(np.array([
                projected_data[:,ind[0],ind[1]] for ind in indices
            ]), axes=(1,0))

        if color_list is None:
            colors = np.array([self.color_dict[i] for i in range(len(orbital_indices_long))])
        else:
            colors = color_list

        self._plot_projected_general(
            ax=ax,
            energy=self.tdos_array[:,0],
            projected_data=projected_data,
            colors=colors,
            sigma=sigma,
            erange=erange,
            linewidth=linewidth,
            alpha_line=alpha_line,
            alpha=alpha,
            fill=fill,
            energyaxis=energyaxis,
            total=total,
        )

        if legend:
            self._add_legend(
                ax,
                names=[f'{i[0]}({i[1]})' for i in zip(atom_indices, orbital_symbols_long)],
                colors=colors
            )


    def plot_orbitals(self, ax, orbitals, fill=True, alpha=0.3, alpha_line=1.0, linewidth=1.5, sigma=0.05, energyaxis='y', color_list=None, legend=True, total=True, erange=[-6, 6]):
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
        if color_list is None:
            colors = np.array([self.color_dict[i] for i in orbitals])
        else:
            colors = color_list

        projected_data = self._sum_orbitals(orbitals=orbitals)

        self._plot_projected_general(
            ax=ax,
            energy=self.tdos_array[:,0],
            projected_data=projected_data,
            colors=colors,
            sigma=sigma,
            erange=erange,
            linewidth=linewidth,
            alpha_line=alpha_line,
            alpha=alpha,
            fill=fill,
            energyaxis=energyaxis,
            total=total,
        )

        if legend:
            self._add_legend(ax, names=[self.orbital_labels[i] for i in orbitals], colors=colors)


    def plot_atoms(self, ax, atoms, fill=True, alpha=0.3, alpha_line=1.0, linewidth=1.5, sigma=0.05, energyaxis='y', color_list=None, legend=True, total=True, erange=[-6, 6], sum_atoms=False):
        """
        This function plots the total density of states with the projected density of states on the given atoms.

        Parameters:
            ax (matplotlib.pyplot.axis): Axis to plot on
            atoms (list): Index of atoms to plot
            fill (bool): Determines wether or not to fill underneath the plot
            alpha (float): Alpha value for the fill
            alpha_line (float): Alpha value for the line
            color_list (list): Optional list of colors of the same length as the atoms list.
            linewidth (float): Linewidth of lines
            sigma (float): Standard deviation for gaussian filter
            energyaxis (str): Determines the axis to plot the energy on ('x' or 'y')
            legend (bool): Determines whether to draw the legend or not
            total (bool): Determines wheth to draw the total density of states or not
            erange (list): Energy range for the DOS plot ([lower bound, upper bound])
        """

        if color_list is None:
            colors = np.array([self.color_dict[i] for i in range(len(atoms))])
        else:
            colors = color_list

        projected_data = self._sum_atoms(atoms=atoms)

        #  if sum_atoms:
            #  if self.spin == 'both' and self.combination_method == 'sub' and self.sp_method == 'percentage':
                #  projected_data_up = np.sum(projected_data[0], axis=1).reshape(-1,1)
                #  projected_data_up = np.sum(projected_data[1], axis=1).reshape(-1,1)
                #  colors = [colors[0]]
            #  else:
                #  projected_data = np.sum(projected_data, axis=1).reshape(-1,1)
                #  colors = [colors[0]]

        self._plot_projected_general(
            ax=ax,
            energy=self.tdos_array[:,0],
            projected_data=projected_data,
            colors=colors,
            sigma=sigma,
            erange=erange,
            linewidth=linewidth,
            alpha_line=alpha_line,
            alpha=alpha,
            fill=fill,
            energyaxis=energyaxis,
            total=total,
        )

        if legend:
            self._add_legend(ax, names=atoms, colors=colors)


    def plot_atom_spd(self, ax, atom_spd_dict, fill=True, alpha=0.3, alpha_line=1.0, linewidth=1.5, sigma=0.05, energyaxis='y', color_list=None, legend=True, total=True, erange=[-6, 6]):
        """
        This function plots the total density of states with the projected
        density of states onto the s, p, and d orbitals of specified atoms. 
        This is useful for supercells where there are many atoms of the same 
        atom and it is inconvienient to manually list each index in the POSCAR.

        Parameters:
            ax (matplotlib.pyplot.axis): Axis to plot on
            atoms (list): List of atom symbols to project onto
            order (list): Order to plot the projected bands in. This feature helps to
                avoid situations where one projection completely convers the other.
            fill (bool): Determines wether or not to fill underneath the plot
            alpha (float): Alpha value for the fill
            alpha_line (float): Alpha value for the line
            linewidth (float): Linewidth of lines
            sigma (float): Standard deviation for gaussian filter
            energyaxis (str): Determines the axis to plot the energy on ('x' or 'y')
            color_dict (dict[str][str]): This option allow the colors of each atom
                specified. Should be in the form of:
                {'atom index': <color>, 'atom index': <color>, ...}
            legend (bool): Determines whether to draw the legend or not
            total (bool): Determines wheth to draw the total density of states or not
            erange (list): Energy range for the DOS plot ([lower bound, upper bound])
        """

        atom_indices = list(atom_spd_dict.keys())
        orbital_symbols = list(atom_spd_dict.values())
        number_orbitals = [len(i) for i in orbital_symbols]
        atom_indices = np.repeat(atom_indices, number_orbitals)
        orbital_symbols_long = np.hstack([[o for o in  orb] for orb in orbital_symbols])
        orbital_indices = np.hstack([[self.spd_relations[o] for o in  orb] for orb in orbital_symbols])
        indices = np.vstack([atom_indices, orbital_indices]).T

        projected_data = self._sum_atoms(atoms=atom_indices, spd=True)

        if self.spin == 'both' and self.combination_method == 'sub' and self.sp_method == 'percentage':
            projected_data_up = np.transpose(np.array([
                projected_data[0,:,ind[0],ind[1]] for ind in indices
            ]), axes=(1,0))
            projected_data_down = np.transpose(np.array([
                projected_data[1,:,ind[0],ind[1]] for ind in indices
            ]), axes=(1,0))
            projected_data = np.array([projected_data_up, projected_data_down])
        else:
            projected_data = np.transpose(np.array([
                projected_data[:,ind[0],ind[1]] for ind in indices
            ]), axes=(1,0))

        if color_list is None:
            colors = np.array([self.color_dict[i] for i in range(len(orbital_symbols_long))])
        else:
            colors = color_list

        self._plot_projected_general(
            ax=ax,
            energy=self.tdos_array[:,0],
            projected_data=projected_data,
            colors=colors,
            sigma=sigma,
            erange=erange,
            linewidth=linewidth,
            alpha_line=alpha_line,
            alpha=alpha,
            fill=fill,
            energyaxis=energyaxis,
            total=total,
        )

        if legend:
            self._add_legend(
                ax,
                names=[f'{i[0]}({i[1]})' for i in zip(atom_indices, orbital_symbols_long)],
                colors=colors
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

        if color_list is None:
            colors = np.array([self.color_dict[i] for i in range(len(elements))])
        else:
            colors = color_list

        projected_data = self._sum_elements(elements=elements)

        self._plot_projected_general(
            ax=ax,
            energy=self.tdos_array[:,0],
            projected_data=projected_data,
            colors=colors,
            sigma=sigma,
            erange=erange,
            linewidth=linewidth,
            alpha_line=alpha_line,
            alpha=alpha,
            fill=fill,
            energyaxis=energyaxis,
            total=total,
        )

        if legend:
            self._add_legend(ax, names=elements, colors=colors)

    def plot_element_orbitals(self, ax, element_orbital_dict, fill=True, alpha=0.3, alpha_line=1.0, linewidth=1.5, sigma=0.05, energyaxis='y', color_list=None, legend=True, total=True, erange=[-6, 6]):
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

        element_symbols = list(element_orbital_dict.keys())
        orbital_indices = list(element_orbital_dict.values())
        number_orbitals = [len(i) for i in orbital_indices]
        element_symbols_long = np.repeat(element_symbols, number_orbitals)
        element_indices = np.repeat(range(len(element_symbols)), number_orbitals)
        orbital_symbols_long = np.hstack([[self.orbital_labels[o] for o in  orb] for orb in orbital_indices])
        orbital_indices_long = np.hstack(orbital_indices)
        indices = np.vstack([element_indices, orbital_indices_long]).T

        projected_data = self._sum_elements(elements=element_symbols, orbitals=True)

        if self.spin == 'both' and self.combination_method == 'sub' and self.sp_method == 'percentage':
            projected_data_up = np.transpose(np.array([
                projected_data[0, :,ind[0],ind[1]] for ind in indices
            ]), axes=(1,0))
            projected_data_down = np.transpose(np.array([
                projected_data[1, :,ind[0],ind[1]] for ind in indices
            ]), axes=(1,0))
            projected_data = np.array([projected_data_up, projected_data_down])
        else:
            projected_data = np.transpose(np.array([
                projected_data[:,ind[0],ind[1]] for ind in indices
            ]), axes=(1,0))

        if color_list is None:
            colors = np.array([self.color_dict[i] for i in range(len(orbital_indices_long))])
        else:
            colors = color_list

        self._plot_projected_general(
            ax=ax,
            energy=self.tdos_array[:,0],
            projected_data=projected_data,
            colors=colors,
            sigma=sigma,
            erange=erange,
            linewidth=linewidth,
            alpha_line=alpha_line,
            alpha=alpha,
            fill=fill,
            energyaxis=energyaxis,
            total=total,
        )

        if legend:
            self._add_legend(
                ax,
                names=[f'{i[0]}({i[1]})' for i in zip(element_symbols_long, orbital_symbols_long)],
                colors=colors
            )

    def plot_element_spd(self, ax, element_spd_dict, fill=True, alpha=0.3, alpha_line=1.0, linewidth=1.5, sigma=0.05, energyaxis='y', color_list=None, legend=True, total=True, erange=[-6, 6]):
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
        element_symbols = list(element_spd_dict.keys())
        orbital_symbols = list(element_spd_dict.values())
        number_orbitals = [len(i) for i in orbital_symbols]
        element_symbols_long = np.repeat(element_symbols, number_orbitals)
        element_indices = np.repeat(range(len(element_symbols)), number_orbitals)
        orbital_symbols_long = np.hstack([[o for o in  orb] for orb in orbital_symbols])
        orbital_indices = np.hstack([[self.spd_relations[o] for o in  orb] for orb in orbital_symbols])
        indices = np.vstack([element_indices, orbital_indices]).T

        projected_data = self._sum_elements(elements=element_symbols, spd=True)

        if self.spin == 'both' and self.combination_method == 'sub' and self.sp_method == 'percentage':
            projected_data_up = np.transpose(np.array([
                projected_data[0,:,ind[0],ind[1]] for ind in indices
            ]), axes=(1,0))
            projected_data_down = np.transpose(np.array([
                projected_data[1,:,ind[0],ind[1]] for ind in indices
            ]), axes=(1,0))
            projected_data = np.array([projected_data_up, projected_data_down])
        else:
            projected_data = np.transpose(np.array([
                projected_data[:,ind[0],ind[1]] for ind in indices
            ]), axes=(1,0))

        if color_list is None:
            colors = np.array([self.color_dict[i] for i in range(len(orbital_symbols_long))])
        else:
            colors = color_list

        self._plot_projected_general(
            ax=ax,
            energy=self.tdos_array[:,0],
            projected_data=projected_data,
            colors=colors,
            sigma=sigma,
            erange=erange,
            linewidth=linewidth,
            alpha_line=alpha_line,
            alpha=alpha,
            fill=fill,
            energyaxis=energyaxis,
            total=total,
        )

        if legend:
            self._add_legend(
                ax,
                names=[f'{i[0]}({i[1]})' for i in zip(element_symbols_long, orbital_symbols_long)],
                colors=colors
            )

    def plot_layers(
        self,
        ax,
        cmap='magma',
        sigma_energy=0.05,
        sigma_layers=0.75,
        energyaxis='y',
        erange=[-6, 6],
        lrange=None,
        antialiased=False,
        fontsize=6,
        interface_layer=None,
        interface_line_color='white',
        interface_line_width=2,
        interface_line_style='--',
        log_scale=False,
        contour=False,
        levels=10,
        min_cutoff=1e-7,
        max_cutoff=None,
        atol=None,
        custom_layer_inds=None,
        custom_cbar_label=None,
        cbar_orientation='vertical',
        show_bounds=False,
        set_bounds=None,
    ):
        """
        This function plots a layer by layer heat map of the density
        of states.

        Parameters:
            ax (matplotlib.pyplot.axis): Axis to plot on
            cmap (str): Color map to use in the heat map
            sigma_energy (float): Variance for a gaussian blur with respect to the energy
                This will help smooth out spikey looking density of states
            sigma_layers (float): Variance for a gaussian blur with respect to the layers
                This will help smooth out the the pixelation that can occur between the summed
                dos with respect to the layers.
            energyaxis (str): Axis to plot the energy on. ('x' or 'y')
            erange (list): Upper and lower energy bounds for the plot.
            lrange (list): Upper and lower bounds of the layers included in the plot.
            antialiased (bool): Determines if antialiasing is used or not.
            fontsize (float): Fontsize of all the text in the group.
            interface_layer (float or None): If a value is provided, then a line will be drawn
                on the plot to identify the interface layer.
            interface_line_color (str): Color of the line drawn on the plot to mark the 
                interface.
            interface_line_width (float): Line with of the line marking the interface.
            interface_line_style (str): Style of the line marking the interface.
            log_scale (bool): Determines if the color map is applied in log scale of not.
                Recommended in order to accurately view the band gap and smaller features.
            contour (bool): Determines if the color map is plotted as a contour plot instead
                of a heatmap.
            levels (int): Number of levels used in the contour plot.
            min_cutoff (float): Minimum dos value used to determine the cut off for the plot.
                This can be adjusted to better visualize the band gap of the material.
            atol (float or None): Tolarence used in the grouping of the layers.
                This value is automatically calculated if None and is usually on the order of
                1e-3.
            custom_layer_inds (list or None): If the structure being calculated has relaxed
                atomic positions, sometimes the layer grouping algorithm can behave non-idealy.
                If this is the case, the user can input a list of list that contain the
                atomic indices in each layers of the material.
            custom_cbar_label (str or None): Custom label for the colorbar
        """
        from vaspvis.utils import group_layers
        import matplotlib.colors as colors
        energy = self.tdos_array[:,0]

        ind = np.where(
                (erange[0] - 0.1 <= energy) & (energy <= erange[-1] + 0.1)
        )
        if custom_layer_inds is None:
            groups, _ = group_layers(self.poscar.structure, atol=atol)
        else:
            groups = custom_layer_inds

        atom_index = range(len(groups))
        energies = energy[ind]
        
        atom_densities = self._sum_atoms(atoms=None)

        if self.spin == 'both' and self.combination_method == 'sub' and self.sp_method == 'percentage':
            atom_densities_up = atom_densities[0, ind].squeeze()
            atom_densities_down = atom_densities[1, ind].squeeze()
            densities_up = np.vstack([np.sum(np.vstack(atom_densities_up[:, [group]]), axis=1) for group in groups])
            densities_down = np.vstack([np.sum(np.vstack(atom_densities_down[:, [group]]), axis=1) for group in groups])
            densities = (densities_up - densities_down) / (densities_up + densities_down)
        else:
            atom_densities = atom_densities[ind]
            densities = np.vstack([np.sum(np.vstack(atom_densities[:,[group]]), axis=1) for group in groups])

        densities = np.transpose(densities)

        if lrange is not None:
            atom_index = atom_index[lrange[0]:lrange[1]+1]
            densities = densities[:, lrange[0]:lrange[1]+1]

        if sigma_energy > 0:
            for i in range(densities.shape[-1]):
                densities[:,i] = self._smear(
                    densities[:,i],
                    sigma=sigma_energy,
                )
        if sigma_layers > 0:
            densities = gaussian_filter(densities, sigma=sigma_layers)

        f = interp2d(atom_index, energies, densities, kind='cubic')
        atom_index = np.arange(np.min(atom_index), np.max(atom_index), 0.1)
        densities = f(atom_index, energies)

        if log_scale:
            if np.min(densities) <= 0:
                neg_zero_loc = np.where(densities <= 0)
                pos_loc = np.where(densities > 0)
                min_val = np.min(densities[pos_loc])
                if min_val < min_cutoff:
                    min_val = min_cutoff
                    too_small_loc = np.where(densities < min_cutoff)
                    densities[too_small_loc] = min_cutoff
                else:
                    densities[neg_zero_loc] = min_val
            else:
                min_val = np.min(densities)
                if min_val < min_cutoff:
                    min_val = min_cutoff

            if max_cutoff is not None:
                max_val = max_cutoff
            else:
                max_val = np.max(densities)
                
            norm = colors.LogNorm(vmin=min_val, vmax=max_val)
        else:
            if self.combination_method == "sub" and self.spin == "both":
                if set_bounds is None:
                    norm_val = np.max(np.abs([np.min(densities), np.max(densities)]))
                else:
                    norm_val = set_bounds

                norm = colors.Normalize(vmin=-norm_val, vmax=norm_val)
            else:
                norm = colors.Normalize(vmin=np.min(densities), vmax=np.max(densities))


        if log_scale:
            lev_exp = np.arange(
                np.floor(np.log10(densities.min())-1),
                np.ceil(np.log10(densities.max())+1),
            )
            if len(lev_exp) >= levels:
                pass
            else:
                if int(levels / len(lev_exp)) >= 3:
                    lev_exp = np.arange(
                        np.floor(np.log10(densities.min())-1),
                        np.ceil(np.log10(densities.max())+1),
                        0.25,
                    )
                else:
                    lev_exp = np.arange(
                        np.floor(np.log10(densities.min())-1),
                        np.ceil(np.log10(densities.max())+1),
                        0.5,
                    )
            levels = np.power(10, lev_exp)

        if energyaxis == 'y':

            if contour:
                im = ax.contourf(
                    atom_index, 
                    energies,
                    densities,
                    cmap=cmap,
                    levels=levels,
                    norm=norm,
                )
            else:
                im = ax.pcolormesh(
                    atom_index,
                    energies,
                    densities,
                    cmap=cmap,
                    shading='gouraud',
                    norm=norm,
                    antialiased=antialiased,
                )
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))


            if interface_layer is not None:
                ax.axvline(
                    x=interface_layer,
                    color=interface_line_color,
                    linestyle=interface_line_style,
                    linewidth=interface_line_width,
                )

        if energyaxis == 'x':
            if contour:
                im = ax.contourf(
                    energies,
                    atom_index, 
                    densities,
                    cmap=cmap,
                    levels=levels,
                    norm=norm,
                )
            else:
                im = ax.pcolormesh(
                    energies,
                    atom_index,
                    np.transpose(densities),
                    cmap=cmap,
                    shading='gouraud',
                    norm=norm,
                    antialiased=antialiased,
                )
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))

            if interface_layer is not None:
                ax.axhline(
                    y=interface_layer,
                    color=interface_line_color,
                    linestyle=interface_line_style,
                    linewidth=interface_line_width,
                )

        fig = plt.gcf()
        cbar = fig.colorbar(im, ax=ax, orientation=cbar_orientation)
        cbar.ax.tick_params(labelsize=fontsize)
        if custom_cbar_label is None:
            if self.combination_method == "sub" and self.spin == "both":
                cbar.set_label('Spin Polarization', fontsize=fontsize)
                min_val = im.norm.vmin
                max_val = im.norm.vmax
                cbar.set_ticks([min_val, max_val])
                if not show_bounds:
                    cbar.set_ticklabels(['Down', 'Up'])
            else:
                cbar.set_label('Density of States', fontsize=fontsize)
        else:
            cbar.set_label(custom_cbar_label, fontsize=fontsize)

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


if __name__ == '__main__':
    dos = Dos(folder='../../vaspvis_data/dos_InAs')
    #  dos._sum_layers(layers=[0,1,2,3])
    fig, ax = plt.subplots(figsize=(4,3), dpi=100)
    dos.plot_spd(
        ax=ax,
        erange=[-6,6],
        fill=True,
        total=False,
        energyaxis='x',
        #  color_list=['red', 'red', 'green'],
    )
    plt.show()
