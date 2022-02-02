from vaspvis.unfold import make_kpath,removeDuplicateKpoints, find_K_from_k, save2VaspKPOINTS
from vaspvis.unfold import convert
from vaspvis.passivator_utils import _append_H, _cart2sph, _get_bot_index, _get_neighbors, _get_top_index,_sort_by_z, _sph2cart, _center_slab
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import JmolNN, CrystalNN, EconNN
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.molecule_structure_comparator import CovalentRadius
from pymatgen.core.periodic_table import Element
from pymatgen.io.vasp.outputs import Eigenval, BSVasprun
from pymatgen.io.vasp.inputs import Incar, Kpoints
from pymatgen.electronic_structure.core import Spin, Orbital
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.io.ase import AseAtomsAdaptor
from ase.build import surface
from ase.build import niggli_reduce, sort
from ase.io import read, write
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.molecule_structure_comparator import CovalentRadius
from pyprocar.utilsprocar import UtilsProcar
from pyprocar.procarparser import ProcarParser
import matplotlib.pyplot as plt
from fastdtw import fastdtw
import numpy as np
import time
import copy
import os


def group_layers(structure, atol=None):
    """
    This function will find the atom indices belonging to each unique atomic layer.

    Parameters:
        structure (pymatgen.core.structure.Structure): Slab structure
        atol (float or None): Tolarence used for grouping the layers. Useful for grouping
            layers in a structure with relaxed atomic positions.

    Returns:
        A list containing the indices of each layers.
        A list of heights of each layers in fractional coordinates.
    """
    sites = structure.sites
    zvals = np.array([site.c for site in sites])
    unique_values = np.sort(np.unique(np.round(zvals, 3)))
    diff = np.mean(np.diff(unique_values)) * 0.2

    grouped = False
    groups = []
    group_heights = []
    zvals_copy = copy.deepcopy(zvals)
    while not grouped:
        if len(zvals_copy) > 0:
            if atol is None:
                group_index = np.where(
                    np.isclose(zvals, np.min(zvals_copy), atol=diff)
                )[0]
            else:
                group_index = np.where(
                    np.isclose(zvals, np.min(zvals_copy), atol=atol)
                )[0]

            group_heights.append(np.min(zvals_copy))
            zvals_copy = np.delete(zvals_copy, np.where(
                np.isin(zvals_copy, zvals[group_index]))[0])
            groups.append(group_index)
        else:
            grouped = True

    return groups, np.array(group_heights)


def convert_slab(bulk_path, slab_path, index, output='POSCAR_unfold', generate=True, print_M=True):
    """
    This function rotates a slab structure so its transformation matrix
    (M) to the primitive bulk structure becomes an integer matrix

    Parameters:
        bulk_path (str): File path to a primitive bulk POSCAR file
        slab_path (str): File patch to a slab structure
        index (list): A three element list defining the miller index of the surface
        output (str): File name of the converted slab POSCAR

    Returns:
        Converted slab POSCAR
    """
    M = convert.convert(
        bulk=bulk_path,
        slab=slab_path,
        index=index,
        output=output,
        generate=generate,
        print_M=print_M,
    )

    return M

def generate_kpoints(M, high_symmetry_points, n, output='KPOINTS'):
    """
    This function generates a KPOINTS file for a band unfolding calculation

    Parameters:
        M (list[list]): A 3x3 transformation matrix
        high_symmetry_points (list[list]): Fractional coordinated of the high symmetry points
            in the band structure path.
        n (int): Numbering of segments between each point
        output (str): File name of the KPOINTS file

    Returns:
        KPOINTS file
    """
    kpath = make_kpath(high_symmetry_points, nseg=n)
    K_in_sup = []
    for kk in kpath:
        kg, _ = find_K_from_k(kk, M)
        K_in_sup.append(kg)
    reducedK = removeDuplicateKpoints(K_in_sup)

    save2VaspKPOINTS(reducedK, output)


class BandGap():
    """
    Determines the band gap from a band structure or density of states calculation

    Parameters:
        folder (str): Folder that contains the VASP input and outputs files
        spin (str): 'both' returns the bandgap for all spins, 'up' returns the 
            bandgap for only the spin up states and 'down' returns the bandgap
            for only the spin down states.
        soc_axis (None or str): This parameter can either take the value of None or the
            it can take the value of 'x', 'y', or 'z'. If either 'x', 'y', or 'z' are given
            then spin='up' states will be defined by positive values of this spin-component
            and spin='down' states will be defined by negative values of this spin-component.
            This will only be used for showing a pseudo-spin-polarized plot for calculations
            that have SOC enabled.
        method (int): method=0 gets the band gap by finding the values closest to
            the fermi level. method=1 gets the band gap based on the average energy
            of each band.
    """
    def __init__(self, folder, spin='both', soc_axis=None, method=0) -> None:
        self.folder = folder
        self.method = method
        self.spin = spin
        self.soc_axis = soc_axis
        self.eigenval = Eigenval(os.path.join(folder, 'EIGENVAL'))
        self.efermi = float(os.popen(f'grep E-fermi {os.path.join(folder, "OUTCAR")}').read().split()[2])

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

        self.pre_loaded_bands = os.path.isfile(os.path.join(folder, 'eigenvalues.npy'))

        if soc_axis is not None and self.lsorbit:
            self.pre_loaded_spin_projections = os.path.isfile(os.path.join(folder, 'spin_projections.npy'))

        self.bg, self.vbm, self.cbm = self._get_bandgap(method=self.method)


    def _load_soc_spin_projection(self):
        """
        This function loads the project weights of the orbitals in each band
        from vasprun.xml into a dictionary of the form:
        band index --> atom index --> weights of orbitals

        Returns:
            projected_dict (dict([str][int][pd.DataFrame])): Dictionary containing the projected weights of all orbitals on each atom for each band.
        """
        if not self.lsorbit:
            raise BaseException(f"You selected soc_axis='{self.soc_axis}' for a non-soc axis calculation, please set soc_axis=None")
        if self.lsorbit and self.soc_axis == 'x':
            spin = 1
        elif self.lsorbit and self.soc_axis == 'y':
            spin = 2
        elif self.lsorbit and self.soc_axis == 'z':
            spin = 3

        if not os.path.isfile(os.path.join(self.folder, 'PROCAR_repaired')):
            UtilsProcar().ProcarRepair(
                os.path.join(self.folder, 'PROCAR'),
                os.path.join(self.folder, 'PROCAR_repaired'),
            )

        if self.pre_loaded_spin_projections:
            with open(os.path.join(self.folder, 'spin_projections.npy'), 'rb') as spin_projs:
                spin_projections = np.load(spin_projs) 
        else:
            parser = ProcarParser()
            parser.readFile(os.path.join(self.folder, 'PROCAR_repaired'))
            spin_projections = np.transpose(parser.spd[:,:,:,-1, -1], axes=(1,0,2))

            np.save(os.path.join(self.folder, 'spin_projections.npy'), spin_projections)


        spin_projections = spin_projections[:,:,spin]

        if self.hse:
            kpoint_weights = np.array(self.eigenval.kpoints_weights)
            zero_weight = np.where(kpoint_weights == 0)[0]
            spin_projections = spin_projections[:,zero_weight]

        separated_projections = np.ones((spin_projections.shape[0], spin_projections.shape[1], 2), dtype=bool)
        # separated_projections[spin_projections > 0, 0] = spin_projections[spin_projections > 0]
        # separated_projections[spin_projections < 0, 1] = -spin_projections[spin_projections < 0]
        separated_projections[spin_projections > 0, 0] = False
        separated_projections[spin_projections < 0, 1] = False

        # separated_projections = separated_projections / separated_projections.max()
        
        if self.spin == 'up':
            separated_projections = separated_projections[:,:,0]
        elif self.spin == 'down':
            separated_projections = separated_projections[:,:,1]
        else:
            raise BaseException("The soc_axis feature does not work with spin='both'")

        return separated_projections

    def _load_eigenvals(self):
        if self.pre_loaded_bands:
            with open(os.path.join(self.folder, 'eigenvalues.npy'), 'rb') as eigenvals:
                band_data = np.load(eigenvals)

            if self.ispin and not self.lsorbit:
                eigenvalues = band_data[:,:,[0,2]]
                kpoints = band_data[0,:,4:]
                eigenvalues_up = band_data[:,:,[0,1]]
                eigenvalues_down = band_data[:,:,[2,3]]
                if self.spin == 'both':
                    eigenvalues_bg = np.vstack([eigenvalues_up, eigenvalues_down])
                elif self.spin == 'up':
                    eigenvalues_bg = eigenvalues_up
                elif self.spin == 'down':
                    eigenvalues_bg = eigenvalues_down
            else:
                eigenvalues = band_data[:,:,0]
                kpoints = band_data[0,:,2:]
                eigenvalues_bg = band_data[:,:,[0,1]]
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
                if self.spin == 'both':
                    eigenvalues_bg = np.vstack([eigenvalues_up, eigenvalues_down])
                elif self.spin == 'up':
                    eigenvalues_bg = eigenvalues_up
                elif self.spin == 'down':
                    eigenvalues_bg = eigenvalues_down
            else:
                eigenvalues = np.transpose(self.eigenval.eigenvalues[Spin.up], axes=(1,0,2))
                eigenvalues[:,:,0] = eigenvalues[:,:,0] - self.efermi
                eigenvalues_bg = eigenvalues

            kpoints = np.array(self.eigenval.kpoints)

            if self.hse:
                kpoint_weights = np.array(self.eigenval.kpoints_weights)
                zero_weight = np.where(kpoint_weights == 0)[0]
                eigenvalues = eigenvalues[:,zero_weight]
                eigenvalues_bg = eigenvalues_bg[:, zero_weight]
                kpoints = kpoints[zero_weight]

            band_data = np.append(
                eigenvalues,
                np.tile(kpoints, (eigenvalues.shape[0],1,1)),
                axis=2,
            )
            np.save(os.path.join(self.folder, 'eigenvalues.npy'), band_data)

        return eigenvalues_bg

    def _method_0(self, eigenvalues):
        if len(eigenvalues.shape) == 3:
            eigenvalues = eigenvalues[:,:,0]

        occupied = eigenvalues[np.where(eigenvalues < 0)]
        unoccupied = eigenvalues[np.where(eigenvalues > 0)]

        vbm = np.nanmax(occupied)
        cbm = np.nanmin(unoccupied)

        if np.nansum(np.abs(np.diff(np.sign(eigenvalues))) > 0) == 0:
            bg = cbm - vbm
        else:
            bg = 0

        return bg, vbm, cbm

    def _method_1(self, eigenvalues):
        if len(eigenvalues.shape) == 3:
            eigenvalues = eigenvalues[:,:,0]

        band_mean = np.nanmean(eigenvalues, axis=1)

        below_index = np.where(band_mean < 0)[0]
        above_index = np.where(band_mean >= 0)[0]

        vbm = np.nanmax(eigenvalues[below_index])
        cbm = np.nanmin(eigenvalues[above_index])

        if np.nansum(np.abs(np.diff(np.sign(eigenvalues))) > 0) == 0:
            bg = cbm - vbm
        else:
            bg = 0

        return bg, vbm, cbm

    def _get_bandgap(self, method=0):
        bg, vbm, cbm = np.nan, np.nan, np.nan
        eigenvalues = self._load_eigenvals()

        if self.lsorbit:
            if self.spin == "both":
                if method == 0:
                    bg, vbm, cbm = self._method_0(eigenvalues)
                elif method == 1:
                    bg, vbm, cbm = self._method_1(eigenvalues)
            else:
                if self.soc_axis is not None:
                    mask = self._load_soc_spin_projection()
                    eigenvalues[mask] = np.nan
                    if method == 0:
                        bg, vbm, cbm = self._method_0(eigenvalues)
                    elif method == 1:
                        bg, vbm, cbm = self._method_1(eigenvalues)
                else:
                    if method == 0:
                        bg, vbm, cbm = self._method_0(eigenvalues)
                    elif method == 1:
                        bg, vbm, cbm = self._method_1(eigenvalues)
        elif self.ispin and not self.lsorbit:
            if method == 0:
                bg, vbm, cbm = self._method_0(eigenvalues)
            elif method == 1:
                bg, vbm, cbm = self._method_1(eigenvalues)
        elif not self.lsorbit and not self.ispin:
            if method == 0:
                bg, vbm, cbm = self._method_0(eigenvalues)
            elif method == 1:
                bg, vbm, cbm = self._method_1(eigenvalues)

        return bg, vbm, cbm


def get_bandgap_old(
    folder,
    printbg=True,
    return_vbm_cbm=False,
    spin='both',
    method=0,
):
    """
    Determines the band gap from a band structure calculation

    Parameters:
        folder (str): Folder that contains the VASP input and outputs files
        printbg (bool): Determines if the band gap value is printed out or not.
        return_vbm_cbm (bool): Determines if the vbm and cbm are returned.
        spin (str): 'both' returns the bandgap for all spins, 'up' returns the 
            bandgap for only the spin up states and 'down' returns the bandgap
            for only the spin down states.
        method (int): method=0 gets the band gap by finding the values closest to
            the fermi level. method=1 gets the band gap based on the average energy
            of each band.

    Returns:
        if return_vbm_cbm is False: The band gap is returned in eV
        if return_vbm_cbm is True: The band gap, vbm, and cbm are returned in eV in that order
    """
    def _get_bandgap_0(eigenvalues, printbg=printbg):

        occupied = eigenvalues[np.where(eigenvalues[:,:,0] < 0)]
        unoccupied = eigenvalues[np.where(eigenvalues[:,:,0] > 0)]

        vbm = np.max(occupied[:,0])
        cbm = np.min(unoccupied[:,0])

        if np.sum(np.diff(np.sign(eigenvalues[:,:,0])) != 0) == 0:
            bg = cbm - vbm
        else:
            bg = 0

        if printbg:
            print(f'Bandgap = {np.round(bg, 3)} eV')
            if return_vbm_cbm:
                print(f'VBM = {np.round(vbm, 3)} eV')
                print(f'CBM = {np.round(cbm, 3)} eV')

        if return_vbm_cbm:
            return bg, vbm, cbm
        else:
            return bg

    def _get_bandgap_1(eigenvalues, printbg=printbg):
        band_mean = eigenvalues[:,:,0].mean(axis=1)

        below_index = np.where(band_mean < 0)[0]
        above_index = np.where(band_mean >= 0)[0]

        vbm = np.max(eigenvalues[below_index, :, 0])
        cbm = np.min(eigenvalues[above_index, :, 0])

        if np.sum(np.diff(np.sign(eigenvalues[:,:,0])) != 0) == 0:
            bg = cbm - vbm
        else:
            bg = 0

        if printbg:
            print(f'Bandgap = {np.round(bg, 3)} eV')
            if return_vbm_cbm:
                print(f'VBM = {np.round(vbm, 3)} eV')
                print(f'CBM = {np.round(cbm, 3)} eV')

        if return_vbm_cbm:
            return bg, vbm, cbm
        else:
            return bg

    pre_loaded_bands = os.path.isfile(os.path.join(folder, 'eigenvalues.npy'))
    eigenval = Eigenval(os.path.join(folder, 'EIGENVAL'))
    efermi = float(os.popen(f'grep E-fermi {os.path.join(folder, "OUTCAR")}').read().split()[2])
    incar = Incar.from_file(
        os.path.join(folder, 'INCAR')
    )
    if 'LSORBIT' in incar:
        if incar['LSORBIT']:
            lsorbit = True
        else:
            lsorbit = False
    else:
        lsorbit = False

    if 'ISPIN' in incar:
        if incar['ISPIN'] == 2:
            ispin = True
        else:
            ispin = False
    else:
        ispin = False

    if 'LHFCALC' in incar:
        if incar['LHFCALC']:
            hse = True
        else:
            hse = False
    else:
        hse = False

    if pre_loaded_bands:
        with open(os.path.join(folder, 'eigenvalues.npy'), 'rb') as eigenvals:
            band_data = np.load(eigenvals)

        if ispin and not lsorbit:
            eigenvalues = band_data[:,:,[0,2]]
            kpoints = band_data[0,:,4:]
            eigenvalues_up = band_data[:,:,[0,1]]
            eigenvalues_down = band_data[:,:,[2,3]]
            if spin == 'both':
                eigenvalues_bg = np.vstack([eigenvalues_up, eigenvalues_down])
            elif spin == 'up':
                eigenvalues_bg = eigenvalues_up
            elif spin == 'down':
                eigenvalues_bg = eigenvalues_down
        else:
            eigenvalues = band_data[:,:,0]
            kpoints = band_data[0,:,2:]
            eigenvalues_bg = band_data[:,:,[0,1]]
        
        if method == 0:
            band_gap = _get_bandgap_0(eigenvalues=eigenvalues_bg)
        elif method == 1:
            band_gap = _get_bandgap_1(eigenvalues=eigenvalues_bg)

    else:
        if len(eigenval.eigenvalues.keys()) > 1:
            eigenvalues_up = np.transpose(eigenval.eigenvalues[Spin.up], axes=(1,0,2))
            eigenvalues_down = np.transpose(eigenval.eigenvalues[Spin.down], axes=(1,0,2))
            eigenvalues_up[:,:,0] = eigenvalues_up[:,:,0] - efermi
            eigenvalues_down[:,:,0] = eigenvalues_down[:,:,0] - efermi
            eigenvalues = np.concatenate(
                [eigenvalues_up, eigenvalues_down],
                axis=2
            )
            if spin == 'both':
                eigenvalues_bg = np.vstack([eigenvalues_up, eigenvalues_down])
            elif spin == 'up':
                eigenvalues_bg = eigenvalues_up
            elif spin == 'down':
                eigenvalues_bg = eigenvalues_down
        else:
            eigenvalues = np.transpose(eigenval.eigenvalues[Spin.up], axes=(1,0,2))
            eigenvalues[:,:,0] = eigenvalues[:,:,0] - efermi
            eigenvalues_bg = eigenvalues

        kpoints = np.array(eigenval.kpoints)

        if hse:
            kpoint_weights = np.array(eigenval.kpoints_weights)
            zero_weight = np.where(kpoint_weights == 0)[0]
            eigenvalues = eigenvalues[:,zero_weight]
            eigenvalues_bg = eigenvalues_bg[:, zero_weight]
            kpoints = kpoints[zero_weight]

        if method == 0:
            band_gap = _get_bandgap_0(eigenvalues=eigenvalues_bg)
        elif method == 1:
            band_gap = _get_bandgap_1(eigenvalues=eigenvalues_bg)

        band_data = np.append(
            eigenvalues,
            np.tile(kpoints, (eigenvalues.shape[0],1,1)),
            axis=2,
        )

        np.save(os.path.join(folder, 'eigenvalues.npy'), band_data)

    return band_gap


def passivator(
    struc,
    passivated_struc=None,
    top=True,
    bot=True,
    symmetrize=True,
    tol=0.0001,
    write_file=False,
    output='POSCAR_pas',
):
    """
    This function passivates the slabs with pseudohydrogen. The positions of the pseudo-
    hydrogen are determined by locating the bonds between the second to last and last
    layers and places hydrogen along each bond at a distance proportional to the sum of
    the covalent radii of the given species and the passivating hydrogen. If an already passivated
    structure is input, the location of the passivating layer will be used on the new structure.

    Parameters:
        struc (str or pymatgen.core.structure.Structure): Unpassivated Structure or file path to
            unpassivated structure.
        passivated_struc (str or pymatgen.core.Structure): Structure or file path to a structure whos
            passivation layer has already been relaxed
        top (bool): Determines if the top of the slab is passivated
        bot (bool): Determines if the bottom of the slab is passivated
        symmetrize (bool): Determines if the slab is symmetrized or not
        tol (float): Numerical tolerence for determining the atoms that get grouped
            into an atomic layer.

    Returns:
        struc_pas (pymatgen.core.Structure): Passivated pymatgen structure
    """

    if type(struc) == str:
        struc = Structure.from_file(struc)
    if type(passivated_struc) == str:
        passivated_struc = Structure.from_file(passivated_struc)

    struc, shift = _center_slab(struc)

    sorted_slab, z_positions = _sort_by_z(struc)

    if passivated_struc is None:
        sorted_slab2, z_positions2 = _sort_by_z(struc)
    else:
        sorted_slab2, z_positions2 = _sort_by_z(passivated_struc)

    for i in range(len(sorted_slab)):
        sorted_slab.sites[i].properties = {'to_delete': False}

    for i in range(len(sorted_slab2)):
        sorted_slab2.sites[i].properties = {'to_delete': False}

    top_index = _get_top_index(z_positions, tol=tol)

    second_top_index = _get_top_index(
        z_positions, to_delete=top_index, tol=tol
    )

    bot_index = _get_bot_index(z_positions, tol=tol)

    if symmetrize:
        z_positions = np.delete(z_positions, bot_index)

        for i in bot_index:
            del sorted_slab[-1]

        bot_index = _get_bot_index(z_positions, tol=tol)

    second_bot_index = _get_bot_index(
        z_positions, to_delete=bot_index, tol=tol
    )

    if passivated_struc is None:
        top_index2 = _get_top_index(z_positions2, tol=tol)

        second_top_index2 = _get_top_index(
            z_positions2, to_delete=top_index2, tol=tol
        )

        bot_index2 = _get_bot_index(z_positions2, tol=tol)

        if symmetrize:
            z_positions2 = np.delete(z_positions2, bot_index2)

            for i in bot_index:
                del sorted_slab2[-1]

            bot_index2 = _get_bot_index(z_positions2, tol=tol)

        second_bot_index2 = _get_bot_index(
            z_positions2, to_delete=bot_index2, tol=tol
        )

    else:
        H_index = np.where(
            np.array([str(i) for i in sorted_slab2.species]) == 'H'
        )[0]

        second_top_index2 = _get_top_index(
            z_positions2, to_delete=H_index, tol=tol
        )

        second_bot_index2 = _get_bot_index(
            z_positions2, to_delete=H_index, tol=tol
        )

        top_index2 = range(np.min(second_top_index2))

        bot_index2 = range(np.max(second_bot_index2) +
                           1, len(z_positions2))

    if bot:
        for i in bot_index:
            sorted_slab.sites[i].properties['to_delete'] = True

        for i in bot_index2:
            sorted_slab2.sites[i].properties['to_delete'] = True

    if top:
        for i in top_index:
            sorted_slab.sites[i].properties['to_delete'] = True

        for i in top_index2:
            sorted_slab2.sites[i].properties['to_delete'] = True

    elements = [str(element) for element in np.unique(sorted_slab.species)]

    max_covalent_radius = np.max(
        [CovalentRadius.radius[element] for element in elements]
    )

    if passivated_struc is None:
        new_radius = True
    else:
        new_radius = False

    if bot:
        for (i, j) in zip(second_bot_index, second_bot_index2):
            neighbor_sph_coords = _get_neighbors(
                struc=sorted_slab2,
                index=j,
                covalent_radius=max_covalent_radius
            )

            _append_H(
                struc=sorted_slab,
                index=i,
                neighbor_sph_coords=neighbor_sph_coords,
                side='bot',
                new_radius=new_radius,
            )

        sorted_slab, _ = _sort_by_z(sorted_slab)
        sorted_slab2, _ = _sort_by_z(sorted_slab2)

    if top:
        for (i, j) in zip(second_top_index, second_top_index2):
            neighbor_sph_coords = _get_neighbors(
                struc=sorted_slab2,
                index=j,
                covalent_radius=max_covalent_radius
            )

            _append_H(
                struc=sorted_slab,
                index=i,
                neighbor_sph_coords=neighbor_sph_coords,
                side='top',
                new_radius=new_radius,
            )

    sites_to_delete = [
        i for i in range(len(sorted_slab)) if sorted_slab.sites[i].properties['to_delete']
    ]
    sorted_slab.remove_sites(sites_to_delete)

    sorted_slab = sorted_slab.get_sorted_structure()
    sorted_slab.translate_sites(range(len(sorted_slab)), [0, 0, -shift])

    if write_file:
        Poscar(sorted_slab).write_file(output)

    return sorted_slab

def get_periodic_vacuum(
        slab,
        bulk,
        miller_index,
        vacuum=40,
        write_file=False,
        output='POSCAR_good_vacuum',
        periodic_vacuum=True,
):
    """
    Creates a slab with a vacuum that is an integer multiple of unit cell length in the direction of the
    miller index. This is necessary for proper band unfolding.

    Parameters:
        slab (str or pymatgen.core.structure.Structure): File path to a slab structure or a pymatgen Structure
        bulk (str or pymatgen.core.structure.Structure): File path to a bulk structure or a pymatgen Structure
        miller_index (list): 3 element list for the miller index of the surface
        vacuum (float): Size of the vacuum. This will be rounded to the closest integer multiple of the unit
            cell length in the direction of the given miller index
        write_file (bool): Determines if a POSCAR is written
        output (str): File name of output POSCAR

    Returns:
        slab structure that has a vacuum with the correct size for band unfolding
    """
    if type(slab) == str:
        slab = Structure.from_file(slab)
    if type(bulk) == str:
        bulk = Structure.from_file(bulk)

    index = np.array(miller_index).reshape(1,-1)
    metric_tensor = bulk.lattice.metric_tensor
    unit_cell_len = np.sqrt(np.squeeze(np.matmul(index,np.matmul(metric_tensor, index.T))))
    c_len = np.linalg.norm(slab.lattice.matrix[-1])

    if np.isin(np.array(slab.species, dtype=str), 'H').any():
        inds = np.isin(np.array(slab.species, dtype=str), 'H', invert=True)
        min_z = np.min(slab.frac_coords[inds,-1])
        slab.translate_sites(
            range(len(slab)),
            [0,0,((unit_cell_len / c_len) - min_z)],
        )
    else:
        min_z = np.min(slab.frac_coords[:,-1])
        slab.translate_sites(
            range(len(slab)),
            [0,0, -min_z],
        )

    max_z2 = np.max(slab.frac_coords[:,-1])
    min_z2 = np.min(slab.frac_coords[:,-1])
    slab_height_in_unit_cells = np.round(((max_z2 - min_z2) * c_len) / unit_cell_len, 0)
    vacuum_height_in_unit_cells = np.round(vacuum / unit_cell_len, 0)
    new_lattice = copy.deepcopy(slab.lattice.matrix)

    if periodic_vacuum:
        new_lattice[-1] = (new_lattice[-1] / np.linalg.norm(new_lattice[-1])) * (slab_height_in_unit_cells + vacuum_height_in_unit_cells) * unit_cell_len
    else:
        new_lattice[-1] = (new_lattice[-1] / np.linalg.norm(new_lattice[-1])) * (((max_z2 - min_z2) * c_len) + vacuum)

    new_c_len = np.linalg.norm(new_lattice[-1])
    new_frac_coords = copy.deepcopy(slab.frac_coords)
    new_frac_coords[:,-1] = new_frac_coords[:,-1] * (c_len / new_c_len)

    new_structure = Structure(
        lattice=Lattice(new_lattice),
        species=slab.species,
        coords=new_frac_coords,
        to_unit_cell=True,
    )

    if write_file:
        Poscar(new_structure).write_file(output)
    else:
        return new_structure



def make_supercell(slab, scaling_matrix):
    """
    Generates a supercell given a scaling matrix

    Parameters:
        slab (str or pymatgen.core.structure.Structure): File path to a slab structure or a pymatgen Structure
        scaling_matrix (list): 3 element scaling matrix

    Returns:
        Super cell slab
    """
    if type(slab) == str:
        slab = Structure.from_file(slab)

    supercell_slab = copy.deepcopy(slab)
    supercell_slab.make_supercell(scaling_matrix=scaling_matrix)

    return supercell_slab

def generate_slab(
        bulk,
        miller_index,
        layers,
        vacuum,
        scaling_matrix=None,
        write_file=True,
        output=None,
        passivate=False,
        passivated_file=None,
        passivate_top=True,
        passivate_bot=True,
        symmetrize=False,
        tol=0.0001,
        periodic_vacuum=True,
):
    """
    This function generates a slab structure.

    Parameters:
        bulk_structure (str): File path to the POSCAR of the conventional 
            unit cell.
        miller_index (list): Three element list to define the miller indices
        layers (int): Number of unit layers in the slab structure
        vacuum (float): Size of the vacuum in Angstoms.
        scaling_matrix (list): 3 element scaling matrix
        write_file (bool): Determines if a POSCAR is written
        output (str): File name of output POSCAR
        passivated_file (pymatgen.core.Structure): path to structure whos passivation layer has already been relaxed
        top (bool): Determines if the top of the slab is passivated
        bot (bool): Determines if the bottom of the slab is passivated
        symmetrize (bool): Determines if the slab is symmetrized (top and bottom have the same termination) or not
        tol (float): Numerical tolerence for determining the atoms that get grouped
            into an atomic layer.

    Returns: 
        Slab structure 
    """

    bulk_pmg = Structure.from_file(bulk)
    bulk_sg = SpacegroupAnalyzer(bulk_pmg)
    bulk_structure_conv = bulk_sg.get_conventional_standard_structure()
    bulk_structure_prim = bulk_sg.get_primitive_standard_structure()
    bulk_structure_ase = AseAtomsAdaptor().get_atoms(bulk_structure_conv)

    ase_slab = surface(
        bulk_structure_ase,
        miller_index,
        layers,
        vacuum=40,
        periodic=False
    )
    sorted_slab = sort(ase_slab, tags=ase_slab.positions[:,-1])
    pmg_structure = AseAtomsAdaptor().get_structure(sorted_slab)
    pmg_sorted_slab = pmg_structure.get_sorted_structure()
    slab_space = SpacegroupAnalyzer(pmg_sorted_slab)
    slab_primitive = slab_space.get_primitive_standard_structure()
    slab_primitive = get_periodic_vacuum(
        slab_primitive,
        bulk_structure_prim,
        miller_index,
        vacuum=vacuum,
        periodic_vacuum=periodic_vacuum,
    )

    if passivate:
        if passivated_file is not None:
            passivated_file = Structure.from_file(passivated_file)

        slab_primitive = passivator(
            slab_primitive,
            passivated_struc=passivated_file,
            top=passivate_top,
            bot=passivate_bot,
            symmetrize=symmetrize,
            tol=tol,
        )

    if scaling_matrix is not None:
        slab_primitive.make_supercell(scaling_matrix=scaling_matrix)

    if write_file:
        if output is None:
            Poscar(slab_primitive).write_file(f'POSCAR_{layers}', direct=True)
        else:
            Poscar(slab_primitive).write_file(output, direct=True)

    return slab_primitive


def compare_dos_to_bulk(
    bulk_folder,
    slab_folder,
    atoms,
    erange=[-6,6],
    sigma=0.05,
    spin_polarized=False,
    plot=True,
    save_plot=True,
    output='bulk_comparison.png',
    bulk_color='black',
    slab_color='green',
    fill_bulk=True,
    fill_slab=True,
    figsize=(4,3),
):
    from vaspvis.dos import Dos
    from vaspvis.standard import _figure_setup_dos
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    _figure_setup_dos(ax=ax, fontsize=12, energyaxis='x')

    if spin_polarized:
        dos_bulk_up = Dos(
            folder=bulk_folder,
            spin='up',
        )
        dos_bulk_down = Dos(
            folder=bulk_folder,
            spin='down',
        )

        dos_bulk_up.plot_plain(
            ax=ax,
            energyaxis='x',
            color=bulk_color,
            erange=erange,
            fill=fill_bulk,
        )
        dos_bulk_down.plot_plain(
            ax=ax,
            energyaxis='x',
            color=bulk_color,
            erange=erange,
            fill=fill_bulk,
        )

        dos_slab_up = Dos(
            folder=slab_folder,
            spin='up',
        )

        dos_slab_down = Dos(
            folder=slab_folder,
            spin='down',
        )

        dos_slab_up.plot_atoms(
            ax=ax,
            atoms=atoms,
            legend=False,
            energyaxis='x',
            color_list=[slab_color for _ in range(len(atoms))],
            total=False,
            fill=fill_slab,
            sum_atoms=True,
            erange=erange,
        )

        dos_slab_down.plot_atoms(
            ax=ax,
            atoms=atoms,
            legend=False,
            energyaxis='x',
            color_list=[slab_color for _ in range(len(atoms))],
            total=False,
            fill=fill_slab,
            sum_atoms=True,
            erange=erange,
        )

        x_data = [ax.lines[i].get_xdata() for i in range(4)]
        y_data = [ax.lines[i].get_ydata() for i in range(4)]

        area_diff_up = fastdtw(
            np.c_[x_data[0], y_data[0]],
            np.c_[x_data[2], y_data[2]],
        )[0]
        area_diff_down = fastdtw(
            np.c_[x_data[1], y_data[1]],
            np.c_[x_data[3], y_data[3]],
        )[0]
        area_diff = np.abs(area_diff_up) + np.abs(area_diff_down)
    else:
        ax.set_xlim(erange[0], erange[1])

        dos_bulk = Dos(
            folder=bulk_folder,
        )

        dos_bulk.plot_plain(
            ax=ax,
            energyaxis='x',
            color=bulk_color,
            erange=erange,
            fill=fill_bulk,
        )

        dos = Dos(
            folder=slab_folder,
        )

        dos.plot_atoms(
            ax=ax,
            atoms=atoms,
            legend=False,
            energyaxis='x',
            color_list=[slab_color for _ in range(len(atoms))],
            total=False,
            fill=fill_slab,
            sum_atoms=True,
            erange=erange,
        )
        x_data = [ax.lines[i].get_xdata() for i in range(2)]
        y_data = [ax.lines[i].get_ydata() for i in range(2)]

        area_diff = fastdtw(
            np.c_[x_data[0], y_data[0]],
            np.c_[x_data[1], y_data[1]],
        )[0]

        area_diff = np.abs(area_diff)

    if plot:
        fig.tight_layout()
        if save_plot:
            fig.savefig(output)

            return area_diff
        else:
            return fig, ax, area_diff
    else:
        return area_diff




if __name__ == "__main__":
    #  area_diff = compare_dos_to_bulk(
        #  bulk_folder='../../vaspvis_data/dos_InAs/',
        #  slab_folder='../../vaspvis_data/slabdos',
        #  atoms=[17,50],
    #  )
    #  slab = passivator(
        #  struc=Poscar.from_file('./POSCAR_0').structure,
        #  write_file=True,
    #  )
    #  Poscar(slab).write_file('POSCAR_pas')
    #  M = convert_slab(
        #  bulk_path='../../../../projects/unfold_test/POSCAR_bulk',
        #  slab_path=slab,
        #  index=[1,1,1],
    #  )
    #  high_symmetry_points = [
        #  [0.1, 0.1, 0.0],
        #  [0.0, 0.0, 0.0],
        #  [0.1, 0.1, 0.0],
    #  ]
    #  generate_kpoints(
        #  M=M,
        #  high_symmetry_points=high_symmetry_points,
        #  n=40,
        #  output='KPOINTS_AGA',
    #  )
    #  get_periodic_vacuum(
        #  slab='./POSCAR_30',
        #  bulk='../../../../projects/unfold_test/POSCAR_bulk',
        #  miller_index=[1,1,1],
        #  vacuum=40,
        #  write_file=True
    #  )
    # slab = passivator(
    #         struc=Poscar.from_file('./POSCAR_2').structure,
    #         # struc='POSCAR_pas',
    #         write_file=True,
    #         output='POSCAR_transfer',
    #         passivated_struc='./CONTCAR',
    #         symmetrize=False,
    #         tol=0.005,
    #         )
    gap_both = BandGap(
            folder='../../vaspvis_data/band_InAs',
            spin='both',
            soc_axis=None
            )
    print('Both')
    print('Gap =', np.round(gap_both.bg, 3))
    print('VBM =', np.round(gap_both.vbm, 3))
    print('CBM =', np.round(gap_both.cbm, 3))
    # print('')
    # gap_up = BandGap(folder='../../vaspvis_data/Ti2MnIn_band', spin='up', soc_axis='z')
    # print('Up')
    # print('Gap =', np.round(gap_up.bg, 3))
    # print('VBM =', np.round(gap_up.vbm, 3))
    # print('CBM =', np.round(gap_up.cbm, 3))
    # print('')
    # gap_down = BandGap(folder='../../vaspvis_data/Ti2MnIn_band', spin='down', soc_axis='z')
    # print('Down')
    # print('Gap =', np.round(gap_down.bg, 3))
    # print('VBM =', np.round(gap_down.vbm, 3))
    # print('CBM =', np.round(gap_down.cbm, 3))
