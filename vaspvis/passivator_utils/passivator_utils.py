from pymatgen.core.surface import SlabGenerator, Slab, center_slab
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.io.ase import AseAtomsAdaptor
from ase.build import surface
from ase.build import niggli_reduce, sort
from ase.io import read, write
import numpy as np
import copy
import os
import pandas as pd
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import JmolNN, CrystalNN, EconNN
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.molecule_structure_comparator import CovalentRadius
from pymatgen.core.periodic_table import Element
import time



def _cart2sph(coords):
    """
    This function converts cartesian coordinates to spherical coordinates

    Parameters:
        coords (np.ndarray): [x, y, z] cartesian coordinates relative to the origin.

    Returns:
        sph_coords (np.ndarray): [r, theta, phi] Spherical coordinates relative to the origin
    """

    coords = np.round(coords, 6)
    x, y, z = coords[0], coords[1], coords[2]

    theta = np.arctan2(y, x)
    phi = np.arctan2(np.sqrt(np.sum(coords[:2] ** 2)), z)
    r = np.sqrt(np.sum(coords ** 2))

    sph_coords = np.array([r, theta, phi])

    return np.round(sph_coords, 6)


def _sph2cart(coords):
    """
    This function converts cartesian coordinates to spherical coordinates

    Parameters:
        coords (np.ndarray): [r, theta, phi] Spherical coordinates relative to the origin.

    Returns:
        cart_coords (np.ndarray): [x, y, z] cartesian coordinates relative to the origin.
    """

    coords = np.round(coords, 6)
    r, theta, phi = coords[0], coords[1], coords[2]

    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)

    cart_coords = np.array([x, y, z])

    return np.round(cart_coords, 6)


def _sort_by_z(struc, reverse=True):
    """
    This function sorts a structure by z-positions and returns the sorted structure

    Parameters:
        struc (pymatgen.core.Structure): Input structure to be sorted
        reverse (bool): If reverse is true it will sort in decending order

    Returns:
        sorted_slab (pymatgen.core.Structure)
    """

    def sort_func(e):
        return e._frac_coords[-1]

    sorted_slab = struc.get_sorted_structure(
        key=sort_func,
        reverse=reverse,
    )

    z_positions = np.array([site.frac_coords[-1] for site in sorted_slab])

    return sorted_slab, z_positions


def _get_bot_index(z_pos, to_delete=None, tol=0.0001):
    """
    This function returns the indices of the atoms in the bottom layer of
    a sorted structure within a certain tolerence.

    Parameters:
        z_positions (np.ndarray): Array of z-positons in the sorted structure
        tol (float): Tolerence for determining indices of atoms in the layer.

    Returns:
        bot_index (np.ndarray)
    """

    if to_delete is None:
        bot_index = np.where(
            np.isclose(z_pos, np.min(z_pos), atol=tol)
        )[0]
    else:
        bot_index = np.where(
            np.isclose(z_pos, np.min(np.delete(z_pos, to_delete)), atol=tol)
        )[0]
        bot_index = np.array([i for i in bot_index if i not in to_delete])

    return bot_index


def _get_top_index(z_pos, to_delete=None, tol=0.0001):
    """
    This function returns the indices of the atoms in the top layer of
    a sorted structure within a certain tolerence.

    Parameters:
        z_pos (np.ndarray): Array of z-positons in the sorted structure
        tol (float): Tolerence for determining indices of atoms in the layer.

    Returns:
        top_index (np.ndarray)
    """

    if to_delete is None:
        top_index = np.where(
            np.isclose(z_pos, np.max(z_pos), atol=tol)
        )[0]
    else:
        top_index = np.where(
            np.isclose(z_pos, np.max(np.delete(z_pos, to_delete)), atol=tol)
        )[0]
        top_index = np.array([i for i in top_index if i not in to_delete])

    return top_index


def _get_neighbors(struc, index, covalent_radius):
    """
    This function finds the nearest neighbors of an array of indices in a structure and
    converts the positon of each neighbor into spherical coordinates relative to the
    positon of the atom at the given index.

    neighbor_sph_coords are in the order of [[r, theta, phi], [r, theta, phi], ...]]

    Parameters:
        struc (pymatgen.core.Structure): Input structure
        index (int): Index of central atom in the nearest neighbor group.
        covalent_radius (float): Covalent radius to use when seaching for neighbors

    Returns:
        neighbor_sph_coords (list[np.ndarray])
    """

    center_coords = struc[index].coords

    for scale_factor in np.linspace(1.8, 3, 13):
        r = scale_factor * covalent_radius
        neighbors = struc.get_neighbors(struc[index], r)
        if len(neighbors) >= 4:
            break
        else:
            continue

    neighbor_cart_coords = np.array([
        neighbor[0].coords - center_coords for neighbor in neighbors
    ])

    neighbor_sph_coords = np.array([
        _cart2sph(coords) for coords in neighbor_cart_coords
    ])

    return neighbor_sph_coords

def _append_H(struc, index, neighbor_sph_coords, side, new_radius=True):
    """
    This functions takes the spherical coordinates or neighoring atoms to an
    index and adjusts them to a new radius which is proportional to the sum
    of hydrogen and whatever element it is bonded to.

    new_neighbor_cart_coords are in the order of [[x, y, z], [x, y, z], ...]]

    Parameters:
        struc (pymatgen.core.Structure): Structure calculate hydrogen coordinates for.
        index (int): Index of central atom in the nearest neighbor group.
        neighbor_sph_coords (np.array): Array of neighoring spherical coordinates

    Returns:
        struc (pymatgen.core.Structure)
    """
    center_coords = struc[index].coords
    center_frac_coords = struc[index].frac_coords
    c_norm = np.linalg.norm(struc.lattice.matrix[-1])
    tol = 0.01 / c_norm

    if new_radius:
        element = struc[index].species.elements[0]
        element_covalent_length = CovalentRadius.radius[str(element)]
        h_covalent_length = CovalentRadius.radius['H']
        new_length = 0.9 * (element_covalent_length + h_covalent_length)

        if side == 'top':
            new_neighbor_cart_coords = np.array([
                _sph2cart(np.array([new_length, c[1], c[2]])) + center_coords for c in neighbor_sph_coords
            ])
            new_neighbor_frac_coords = np.dot(new_neighbor_cart_coords, struc.lattice.inv_matrix)
            inds = np.where(new_neighbor_frac_coords[:,-1] > (center_frac_coords[-1] + tol))[0]
            new_neighbor_frac_coords = new_neighbor_frac_coords[inds]

        elif side == 'bot':
            new_neighbor_cart_coords = np.array([
                _sph2cart(np.array([new_length, c[1], c[2]])) + center_coords for c in neighbor_sph_coords
            ])
            new_neighbor_frac_coords = np.dot(new_neighbor_cart_coords, struc.lattice.inv_matrix)
            inds = np.where(new_neighbor_frac_coords[:,-1] < (center_frac_coords[-1] - tol))[0]
            new_neighbor_frac_coords = new_neighbor_frac_coords[inds]
    else:
        if side == 'top':
            new_neighbor_cart_coords = np.array([
                _sph2cart(c) + center_coords for c in neighbor_sph_coords
            ])
            new_neighbor_frac_coords = np.dot(new_neighbor_cart_coords, struc.lattice.inv_matrix)
            inds = np.where(new_neighbor_frac_coords[:,-1] > (center_frac_coords[-1] + tol))[0]
            new_neighbor_frac_coords = new_neighbor_frac_coords[inds]
        elif side == 'bot':
            new_neighbor_cart_coords = np.array([
                _sph2cart(c) + center_coords for c in neighbor_sph_coords
            ])
            new_neighbor_frac_coords = np.dot(new_neighbor_cart_coords, struc.lattice.inv_matrix)
            inds = np.where(new_neighbor_frac_coords[:,-1] < (center_frac_coords[-1] - tol))[0]
            new_neighbor_frac_coords = new_neighbor_frac_coords[inds]

    H = Element('H')

    for coords in new_neighbor_frac_coords:
        struc.append(
            species=H,
            coords=coords,
            properties={'to_delete': False},
        )

def _old_append_H(struc, index, neighbor_sph_coords, side, new_radius=True):
    """
    This functions takes the spherical coordinates or neighoring atoms to an
    index and adjusts them to a new radius which is proportional to the sum
    of hydrogen and whatever element it is bonded to.

    new_neighbor_cart_coords are in the order of [[x, y, z], [x, y, z], ...]]

    Parameters:
        struc (pymatgen.core.Structure): Structure calculate hydrogen coordinates for.
        index (int): Index of central atom in the nearest neighbor group.
        neighbor_sph_coords (np.array): Array of neighoring spherical coordinates

    Returns:
        struc (pymatgen.core.Structure)
    """
    center_coords = struc[index].coords

    if new_radius:
        element = struc[index].species.elements[0]
        element_covalent_length = CovalentRadius.radius[str(element)]
        h_covalent_length = CovalentRadius.radius['H']
        new_length = 0.9 * (element_covalent_length + h_covalent_length)

        if side == 'top':
            new_neighbor_cart_coords = np.array([
                _sph2cart(np.array([new_length, c[1], c[2]])) + center_coords for c in neighbor_sph_coords if c[-1] < np.pi / 2.1
            ])
        elif side == 'bot':
            new_neighbor_cart_coords = np.array([
                _sph2cart(np.array([new_length, c[1], c[2]])) + center_coords for c in neighbor_sph_coords if c[-1] > np.pi / 1.9
            ])
    else:
        if side == 'top':
            new_neighbor_cart_coords = np.array([
                _sph2cart(c) + center_coords for c in neighbor_sph_coords if c[-1] < np.pi / 2.1
            ])
        elif side == 'bot':
            new_neighbor_cart_coords = np.array([
                _sph2cart(c) + center_coords for c in neighbor_sph_coords if c[-1] > np.pi / 1.9
            ])

    H = Element('H')

    for coords in new_neighbor_cart_coords:
        struc.append(
            species=H,
            coords=coords,
            coords_are_cartesian=True,
            properties={'to_delete': False},
        )

def _center_slab(slab):
    bdists = sorted([nn[1] for nn in slab.get_neighbors(slab[0], 10) if nn[1] > 0])
    r = bdists[0] * 3

    all_indices = [i for i, site in enumerate(slab)]

    for site in slab:
        if any([nn[1] > slab.lattice.c for nn in slab.get_neighbors(site, r)]):
            shift = 1 - site.frac_coords[2] + 0.05
            slab.translate_sites(all_indices, [0, 0, shift])

    weights = [s.species.weight for s in slab]
    center_of_mass = np.average(slab.frac_coords, weights=weights, axis=0)
    shift = 0.5 - center_of_mass[2]
    slab.translate_sites(all_indices, [0, 0, shift])

    return slab, shift

