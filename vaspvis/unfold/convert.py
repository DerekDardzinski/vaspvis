import argparse
import numpy as np
import os
import copy
import spglib
from numpy.linalg import norm, solve
from pymatgen.io.vasp.inputs import Poscar
from ase.io import write, read
from ase.utils import gcd, basestring
from ase.build import bulk
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


def surface(lattice, indices, layers, vacuum=None, tol=1e-10):
    """Create surface from a given lattice and Miller indices.

    lattice: Atoms object or str
        Bulk lattice structure of alloy or pure metal.  Note that the
        unit-cell must be the conventional cell - not the primitive cell.
        One can also give the chemical symbol as a string, in which case the
        correct bulk lattice will be generated automatically.
    indices: sequence of three int
        Surface normal in Miller indices (h,k,l).
    layers: int
        Number of equivalent layers of the slab.
    vacuum: float
        Amount of vacuum added on both sides of the slab.

    """

    indices = np.asarray(indices)

    if indices.shape != (3,) or not indices.any() or indices.dtype != int:
        raise ValueError('%s is an invalid surface type' % indices)

    if isinstance(lattice, basestring):
        lattice = bulk(lattice, cubic=True)

    h, k, l = indices
    h0, k0, l0 = (indices == 0)

    if h0 and k0 or h0 and l0 or k0 and l0:  # if two indices are zero
        if not h0:
            c1, c2, c3 = [(0, 1, 0), (0, 0, 1), (1, 0, 0)]
        if not k0:
            c1, c2, c3 = [(0, 0, 1), (1, 0, 0), (0, 1, 0)]
        if not l0:
            c1, c2, c3 = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    else:
        p, q = ext_gcd(k, l)
        a1, a2, a3 = lattice.cell

        # constants describing the dot product of basis c1 and c2:
        # dot(c1,c2) = k1+i*k2, i in Z
        k1 = np.dot(p * (k * a1 - h * a2) + q * (l * a1 - h * a3),
                    l * a2 - k * a3)
        k2 = np.dot(l * (k * a1 - h * a2) - k * (l * a1 - h * a3),
                    l * a2 - k * a3)

        if abs(k2) > tol:
            i = -int(round(k1 / k2))  # i corresponding to the optimal basis
            p, q = p + i * l, q - i * k

        a, b = ext_gcd(p * k + q * l, h)

        c1 = (p * k + q * l, -p * h, -q * h)
        c2 = np.array((0, l, -k)) // abs(gcd(l, k))
        c3 = (b, a * p, a * q)
    surf = build(lattice, np.array([c1, c2, c3]), layers, tol)
    if vacuum is not None:
        surf.center(vacuum=vacuum, axis=2)
    return surf


def build(lattice, basis, layers, tol):
    surf = lattice.copy()
    scaled = solve(basis.T, surf.get_scaled_positions().T).T
    scaled -= np.floor(scaled + tol)
    surf.set_scaled_positions(scaled)
    surf.set_cell(np.dot(basis, surf.cell), scale_atoms=True)
    surf *= (1, 1, layers)
    a1, a2, a3 = surf.cell
    surf.set_cell([a1, a2,
                   np.cross(a1, a2) * np.dot(a3, np.cross(a1, a2)) /
                   norm(np.cross(a1, a2))**2])
    surf.pbc = (True, True, False)

    # Move atoms into the unit cell:
    scaled = surf.get_scaled_positions()
    scaled[:, :2] %= 1
    surf.set_scaled_positions(scaled)

    surf.cell[2] = 0.0

    return surf


def ext_gcd(a, b):
    if b == 0:
        return 1, 0
    elif a % b == 0:
        return 0, 1
    else:
        x, y = ext_gcd(b, a % b)
        return y, x - y * (a // b)


def convert(bulk, slab, index, output, generate=True, print_M=True):
    if type(bulk) == str:
        primitiveCell = Structure.from_file(bulk)
    else:
        primitiveCell = bulk
    if type(slab) == str:
        refSlab = Structure.from_file(slab)
    else:
        refSlab = slab

    sa = SpacegroupAnalyzer(primitiveCell)
    conventionalCell = sa.get_conventional_standard_structure()
    conventionalCell.to(filename='POSCAR.conventional')
    bulk = read('POSCAR.conventional')
    os.remove('POSCAR.conventional')
    slab = surface(bulk, index, layers=2, vacuum=10)
    lattice, _, _ = spglib.standardize_cell(cell=(slab.get_cell(
    ), slab.get_scaled_positions(), slab.get_atomic_numbers()), no_idealize=True)
    lattice_params = np.sort(np.linalg.norm(lattice, axis=1))[:2]
    scales = np.round(np.array([refSlab.lattice.a, refSlab.lattice.b] / lattice_params), 2)
    newLattice = []
    oldLattice = refSlab.lattice
    for length, scale in zip([oldLattice.a, oldLattice.b], scales):
        for j in range(len(lattice)):
            if abs((np.linalg.norm(lattice[j]) * scale) - length) < 1e-1:
                newLattice.append(copy.copy(scale * lattice[j][:]))
                lattice[j] = [0, 0, 0]
                break
    for i in range(len(lattice)):
        norm = np.linalg.norm(lattice[i])
        if norm > 1e-1:
            newLattice.append(lattice[i] / norm * oldLattice.c)
            break
    newLattice = Lattice(np.array(newLattice))
    for x, y in zip(oldLattice.angles, newLattice.angles):
        assert abs(x-y) < 1e-2, "The converted lattice has incorrect angles: {} compared with reference slab: {}".format(
            " ".join(str(x) for x in newLattice.angles), " ".join(str(x) for x in oldLattice.angles))
    newSlab = Structure(newLattice, [], [])
    for atom in refSlab:
        newSlab.append(atom.specie, atom.frac_coords[:])

    if generate:
        Poscar(newSlab.get_sorted_structure()).write_file(output, direct=True)

    transformMat = newSlab.lattice.matrix.dot(
        np.linalg.inv(primitiveCell.lattice.matrix))
    transformMat = transformMat.round().astype(int)
    if print_M:
        print('-------------------------------------------')
        print('Your Transformtaion Matrix is:')
        print(' ')
        print(transformMat)
        print('-------------------------------------------')

    return transformMat

if __name__ == "__main__":
    bulk = './POSCAR_bulk'
    slab = './POSCAR_10_100_22'
    index = [1,0,0]
    #  bulk = './POSCAR_HgTe_ept'
    #  slab = './POSCAR_18'
    #  index = [1,0,0]
    output = 'POSCAR_conv2'
    M = convert(bulk, slab, index, output)
