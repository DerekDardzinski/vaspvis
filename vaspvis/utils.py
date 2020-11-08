import numpy as np
from vaspvis.unfold import make_kpath,removeDuplicateKpoints, find_K_from_k, save2VaspKPOINTS
from vaspvis.unfold import convert

def convert_slab(bulk_path, slab_path, index, output='POSCAR_unfold'):
    M = convert(
        bulk=bulk_path,
        slab=slab_path,
        index=index,
        output=output,
    )

    return M

def generate_kpoints(M, high_symmetry_points, n, output='KPOINTS'):
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
    generate_kpoints(
        M=M,
        high_symmetry_points=high_symmetry_points,
        n=50,
    )
