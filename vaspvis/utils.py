from vaspvis.unfold import make_kpath,removeDuplicateKpoints, find_K_from_k, save2VaspKPOINTS
from vaspvis.unfold import convert
from vaspvis.passivator_utils import _append_H, _cart2sph, _get_bot_index, _get_neighbors, _get_top_index,_sort_by_z, _sph2cart
from pymatgen.io.vasp.outputs import Eigenval, BSVasprun
from pymatgen.io.vasp.inputs import Incar, Kpoints
from pymatgen.electronic_structure.core import Spin, Orbital
from vaspvis.band import Band
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
import numpy as np
import copy
import os

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


def get_bandgap(folder, printbg=True):
    def _get_bandgap(eigenvalues, printbg=printbg):
        if np.sum(np.diff(np.sign(eigenvalues[:,:,0])) != 0) == 0:
            occupied = eigenvalues[np.where(eigenvalues[:,:,0] < 0)]
            unoccupied = eigenvalues[np.where(eigenvalues[:,:,0] > 0)]

            vbm = np.max(occupied[:,0])
            cbm = np.min(unoccupied[:,0])

            bg = cbm - vbm
        else:
            bg = 0

        if printbg:
            print(f'Bandgap = {np.round(bg, 3)} eV')

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
            eigenvalues_bg = np.vstack([eigenvalues_up, eigenvalues_down])
        else:
            eigenvalues = band_data[:,:,0]
            kpoints = band_data[0,:,2:]
            eigenvalues_bg = band_data[:,:,[0,1]]
        
        band_gap = _get_bandgap(eigenvalues=eigenvalues_bg)
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
            eigenvalues_bg = np.vstack([eigenvalues_up, eigenvalues_down])
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

        band_gap = _get_bandgap(eigenvalues=eigenvalues_bg)

        band_data = np.append(
            eigenvalues,
            np.tile(kpoints, (eigenvalues.shape[0],1,1)),
            axis=2,
        )

        np.save(os.path.join(folder, 'eigenvalues.npy'), band_data)

    return band_gap

def passivator(struc, passivated_struc=None, top=True, bot=True, symmetrize=True, tol=0.3):
    """
    This function passivates the slabs with pseudohydrogen. The positions of the pseudo-
    hydrogen are determined by locating the bonds between the second to last and last
    layers and places hydrogen along each bond at a distance proportional to the sum of
    the covalent radii of the given species and the passivating hydrogen. If an already passivated
    structure is input, the location of the passivating layer will be used on the new structure.

    Parameters:
        struc (pymatgen.core.Structure): Unpassivated Structure
        passivated_struc (pymatgen.core.Structure): Structure whos passivation layer has already been relaxed
        top (bool): Determines if the top of the slab is passivated
        bot (bool): Determines if the bottom of the slab is passivated
        symmetrize (bool): Determines if the slab is symmetrized or not
        tol (float): Numerical tolerence for determining the atoms that get grouped
            into an atomic layer.

    Returns:
        struc_pas (pymatgen.core.Structure): Passivated pymatgen structure
    """

    lattice = struc.lattice.matrix

    sorted_slab, z_positions = _sort_by_z(struc)

    if passivated_struc is None:
        sorted_slab2, z_positions2 = _sort_by_z(struc)
    else:
        sorted_slab2, z_positions2 = _sort_by_z(passivated_struc)

    for i in range(len(sorted_slab)):
        sorted_slab.sites[i].properties = {'to_delete': False}

    for i in range(len(sorted_slab2)):
        sorted_slab2.sites[i].properties = {'to_delete': False}

    if lattice[-1][-1] < 0:
        flipped = True
    else:
        flipped = False

    if flipped:
        top_index = _get_bot_index(z_positions, tol=tol)

        second_top_index = _get_bot_index(
            z_positions, to_delete=top_index, tol=tol
        )

        bot_index = _get_top_index(z_positions, tol=tol)

        if symmetrize:
            z_positions = np.delete(z_positions, bot_index)

            for i in bot_index:
                del sorted_slab[-1]

            bot_index = _get_top_index(z_positions, tol=tol)

        second_bot_index = _get_top_index(
            z_positions, to_delete=bot_index, tol=tol
        )

        if passivated_struc is None:
            top_index2 = _get_bot_index(z_positions2, tol=tol)

            second_top_index2 = _get_bot_index(
                z_positions2, to_delete=top_index2, tol=tol
            )

            bot_index2 = _get_top_index(z_positions2, tol=tol)

            if symmetrize:
                z_positions2 = np.delete(z_positions2, bot_index2)

                for i in bot_index:
                    del sorted_slab2[-1]

                bot_index2 = _get_top_index(z_positions2, tol=tol)

            second_bot_index2 = _get_top_index(
                z_positions2, to_delete=bot_index2, tol=tol
            )

        else:
            H_index = np.where(
                np.array([str(i) for i in sorted_slab2.species]) == 'H'
            )[0]

            second_top_index2 = _get_bot_index(
                z_positions2, to_delete=H_index, tol=tol
            )

            second_bot_index2 = _get_top_index(
                z_positions2, to_delete=H_index, tol=tol
            )

            top_index2 = range(np.max(second_top_index2) +
                               1, len(z_positions2))

            bot_index2 = range(np.min(second_bot_index2))

    else:
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

            if flipped:
                side = 'top'
            else:
                side = 'bot'


            _append_H(
                struc=sorted_slab,
                index=i,
                neighbor_sph_coords=neighbor_sph_coords,
                side=side,
                new_radius=new_radius,
            )

    if top:
        for (i, j) in zip(second_top_index, second_top_index2):
            neighbor_sph_coords = _get_neighbors(
                struc=sorted_slab2,
                index=j,
                covalent_radius=max_covalent_radius
            )

            if flipped:
                side = 'bot'
            else:
                side = 'top'


            _append_H(
                struc=sorted_slab,
                index=i,
                neighbor_sph_coords=neighbor_sph_coords,
                side=side,
                new_radius=new_radius,
            )

    sites_to_delete = [i for i in range(len(sorted_slab)) if sorted_slab.sites[i].properties['to_delete']]
    sorted_slab.remove_sites(sites_to_delete)

    sorted_slab = sorted_slab.get_sorted_structure()

    return sorted_slab


def get_periodic_vacuum(
        slab,
        bulk,
        miller_index,
        vacuum=40,
        write_file=False,
        output='POSCAR_good_vacuum',
):
    index = np.array(miller_index).reshape(1,-1)
    metric_tensor = bulk.lattice.metric_tensor
    unit_cell_len = np.sqrt(np.squeeze(np.matmul(index,np.matmul(metric_tensor, index.T))))
    min_z = np.min(slab.cart_coords[:,-1])
    slab.translate_sites(
        range(len(slab)),
        [0,0,-min_z],
        frac_coords=False
    )
    max_z2 = np.max(slab.cart_coords[:,-1])
    min_z2 = np.min(slab.cart_coords[:,-1])
    slab_height_in_unit_cells = int((max_z2 - min_z2) / unit_cell_len)
    vacuum_height_in_unit_cells = int(vacuum / unit_cell_len)
    new_lattice = copy.deepcopy(slab.lattice.matrix)
    new_lattice[-1,-1] = (slab_height_in_unit_cells + vacuum_height_in_unit_cells) * unit_cell_len

    new_structure = Structure(
        lattice=Lattice(new_lattice),
        species=slab.species,
        coords=slab.cart_coords,
        coords_are_cartesian=True,
        to_unit_cell=True,
    )

    if write_file:
        Poscar(new_structure).write_file(output)
    else:
        return new_structure

def make_supercell(slab, scaling_matrix):
    supercell_slab = copy.deepcopy(slab)
    supercell_slab.make_supercell(scaling_matrix=scaling_matrix)

    return supercell_slab

def generate_surface(
        bulk,
        miller_index,
        layers,
        vacuum,
        supercell=None,
        write_file=True,
        output=None,
        passivate=False,
        passivated_file=None,
        passivate_top=True,
        passivate_bot=True,
        symmetrize=False,
):
    """
    This function generates a slab structure.

    Parameters:
        bulk_structure (str): File path to the POSCAR of the conventional 
            unit cell.
        miller_index (list): Three element list to define the miller indices
        layers (int): Number of unit layers in the slab structure
        vacuum (float): Size of the vacuum in Angstoms.

    Returns: POSCAR_slab file
    """

    bulk_pmg = Structure.from_file(bulk)
    bulk_sg = SpacegroupAnalyzer(bulk_pmg)
    bulk_structure_conv = bulk_sg.get_conventional_standard_structure()
    bulk_structure_prim = bulk_sg.get_primitive_standard_structure()
    bulk_structure = AseAtomsAdaptor().get_atoms(bulk_structure_conv)

    struc = read(bulk_structure)
    ase_slab = surface(
        struc,
        miller_index,
        layers,
        vacuum=vacuum,
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
    )

    if supercell is not None:
        slab_primitive.make_supercell(scaling_matrix=supercell)

    if passivate:
        if passivated_file is not None:
            passivated_file = Poscar.from_file(passivated_file).structure

        slab_primitive = passivator(
            slab_primitive,
            passivated_struc=passivated_file,
            top=passivate_top,
            bot=passivate_bot,
            symmetrize=symmetrize,
            tol=0.3
        )

    if write_file:
        if output is None:
            Poscar(slab_primitive).write_file(f'POSCAR_{layers}', direct=True)
        else:
            Poscar(slab_primitive).write_file(output, direct=True)

    return slab_primitive



if __name__ == "__main__":
    get_bandgap(folder='../../vaspvis_data/band_InAs')
    run = BSVasprun('../../vaspvis_data/band_InAs/vasprun.xml')
    bs = run.get_band_structure('../../vaspvis_data/band_InAs/KPOINTS')
    print(bs.get_vbm()['energy'] - bs.efermi)
    print(bs.get_cbm()['energy'] - bs.efermi)
    print(bs.get_band_gap())
