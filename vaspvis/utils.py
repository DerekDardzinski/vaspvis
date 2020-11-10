import numpy as np
from vaspvis.unfold import make_kpath,removeDuplicateKpoints, find_K_from_k, save2VaspKPOINTS
from vaspvis.unfold import convert

def convert_slab(bulk_path, slab_path, index, output='POSCAR_unfold'):
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

if __name__ == "__main__":
    high_symmetry_points = [
        [0.5,0,0.5],
        [0,0,0],
        [0.5,0,0.5],
    ]
    M = convert_slab(
        bulk_path='./unfold/POSCAR_bulk',
        slab_path='./unfold/POSCAR_sub_9_0_orientation_0',
        index=[1,1,1],
    )
    #  generate_kpoints(
        #  M=M,
        #  high_symmetry_points=high_symmetry_points,
        #  n=50,
    #  )
