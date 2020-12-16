import numpy as np
from vaspvis.unfold import make_kpath,removeDuplicateKpoints, find_K_from_k, save2VaspKPOINTS
from vaspvis.unfold import convert
from pymatgen.io.vasp.outputs import Eigenval, BSVasprun
from pymatgen.io.vasp.inputs import Incar, Kpoints
from pymatgen.electronic_structure.core import Spin, Orbital
from vaspvis.band import Band
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
        kg, g = find_K_from_k(kk, M)
        K_in_sup.append(kg)
    reducedK = removeDuplicateKpoints(K_in_sup)

    save2VaspKPOINTS(reducedK, output)


def get_bandgap_old(folder, printbg=True):
    """
    This function get the band gap from a bandstructure. If the structure is metallic it will
    return zero.

    Parameters:
        folder (str): This is the folder that contains the VASP files
        printbg (bool): If True, it will print the value of the bandgap. If False,
            the value of the bandgap will be returned without printing.

    Returns:
        Bandgap value in eV
    """
    band = Band(folder=folder, bandgap=True, printbg=printbg)

    return band.bg

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




if __name__ == "__main__":
    get_bandgap(folder='../../vaspvis_data/band_InAs')
    run = BSVasprun('../../vaspvis_data/band_InAs/vasprun.xml')
    bs = run.get_band_structure('../../vaspvis_data/band_InAs/KPOINTS')
    print(bs.get_vbm()['energy'] - bs.efermi)
    print(bs.get_cbm()['energy'] - bs.efermi)
    print(bs.get_band_gap())
    #  get_bandgap2(folder='../../vaspvis_data/hseInAs')
    #  high_symmetry_points = [
        #  [0.5,0,0.5],
        #  [0,0,0],
        #  [0.5,0,0.5],
    #  ]
    #  M = convert_slab(
        #  bulk_path='./unfold/POSCAR_bulk',
        #  slab_path='./unfold/POSCAR_sub_9_0_orientation_0',
        #  index=[1,1,1],
    #  )
    #  generate_kpoints(
        #  M=M,
        #  high_symmetry_points=high_symmetry_points,
        #  n=50,
    #  )
