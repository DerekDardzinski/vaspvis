from vaspvis.unfold import make_kpath,removeDuplicateKpoints, find_K_from_k, save2VaspKPOINTS
from vaspvis.unfold import convert
from vaspvis.passivator_utils import _append_H, _cart2sph, _get_bot_index, _get_neighbors, _get_top_index,_sort_by_z, _sph2cart
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import JmolNN, CrystalNN, EconNN
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.molecule_structure_comparator import CovalentRadius
from pymatgen.core.periodic_table import Element
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
    """
    Determines the band gap from a band structure calculation

    Parameters:
        folder (str): Folder that contains the VASP input and outputs files
        printbg (bool): Determines if the band gap value is printed out or not.

    Returns:
        Bandgap in eV
    """
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

    return sorted_slab

def get_periodic_vacuum(
        slab,
        bulk,
        miller_index,
        vacuum=40,
        write_file=False,
        output='POSCAR_good_vacuum',
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
    new_lattice[-1] = (new_lattice[-1] / np.linalg.norm(new_lattice[-1])) * (slab_height_in_unit_cells + vacuum_height_in_unit_cells) * unit_cell_len
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
        tol=0.3,
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

    if passivate:
        if passivated_file is not None:
            passivated_file = Poscar.from_file(passivated_file).structure

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



if __name__ == "__main__":
    slab = generate_slab(
        bulk='../../../../projects/unfold_test/POSCAR_InSb_conv',
        miller_index=[1,0,0],
        layers=8,
        vacuum=30,
        passivate=True,
    )
    #  slab = passivator(
        #  struc=Poscar.from_file('./POSCAR_100_test').structure
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
