from pymatgen.io.vasp.outputs import Chgcar
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.core.periodic_table import Element
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.operations import SymmOp
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.patheffects as pa
import copy as copy
from vaspvis.utils import make_supercell, group_layers
from ase.data.colors import jmol_colors
from sklearn.neighbors import radius_neighbors_graph
import os
import time


class STM:
    """
    This class contains all the methods for generating STM images with VASP
    """
    
    def __init__(
        self,
        folder,
    ):
        self.folder = folder
        self.preloaded_data = os.path.isfile(os.path.join(folder, 'parchg.npy'))
        self.poscar = Poscar.from_file(
            os.path.join(folder, 'POSCAR'),
            check_for_POTCAR=False,
            read_velocities=False
        )
        self.data, self.a_vals, self.b_vals, self.c_vals = self._load_parchg()
        [self.bottom_surface,
        self.bottom_ind,
        self.top_surface,
        self.top_ind] = self._get_surface_heights()
        self.X = None
        self.Y = None
        self.Z = None
        self.x_shift = None
        self.y_shift = None

    def _load_parchg(self):
        if self.preloaded_data:
            with open(os.path.join(self.folder, 'parchg.npy'), 'rb') as p:
                data = np.load(p)
        else:
            parchg = Chgcar.from_file(os.path.join(self.folder, 'PARCHG'))
            data = parchg.data['total']
            np.save(os.path.join(self.folder, 'parchg.npy'), data)

        a_vals = np.linspace(0,1,data.shape[0])
        b_vals = np.linspace(0,1,data.shape[1])
        c_vals = np.linspace(0,1,data.shape[2])

        return data, a_vals, b_vals, c_vals

    def _get_surface_heights(self):
        bottom_surface = self.poscar.structure.frac_coords[:,-1].min()
        top_surface = self.poscar.structure.frac_coords[:,-1].max()
        bottom_ind = np.argmin((self.c_vals - bottom_surface)**2)
        top_ind = np.argmin((self.c_vals - top_surface)**2)

        return bottom_surface, bottom_ind, top_surface, top_ind

    def _interp(self, x, x1, x2, y1, y2):
        return y1 + (((y2 - y1) / (x2 - x1)) * (x - x1)) 

    def _rotate_structure(self, structure, angle):
        copy_structure = copy.copy(structure)
        angle = angle * (np.pi / 180)
        operation = SymmOp.from_rotation_and_translation(
            rotation_matrix=np.array([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0,0,1],
            ]),
            translation_vec=[0,0,0],
        )
        copy_structure.apply_operation(operation, fractional=False)

        return copy_structure

    def _get_constant_current_isosurface(self, current, sigma=6, top=True):
        slab_middle_ind = int((self.top_ind + self.bottom_ind) / 2)
        cell_middle_ind = int(self.data.shape[-1] / 2)
        shift = cell_middle_ind - slab_middle_ind
        init_shape = self.data.shape[:2]
        shifted_slab = np.roll(
            self.data,
            shift,
            axis=2,
        )
        c_vals = self.c_vals
        c_vals_extended = np.hstack([c_vals[:-1] - 1, c_vals, c_vals[1:] + 1])
        shifted_cvals = np.roll(
            c_vals_extended,
            shift,
        )
        shifted_cvals = shifted_cvals[len(c_vals)-1:(2*len(c_vals))-1]

        if top:
            shifted_slab = shifted_slab[:,:,self.top_ind+shift:]
            shifted_cvals = shifted_cvals[self.top_ind+shift:]
        else:
            shifted_slab = shifted_slab[:,:,:self.bottom_ind+shift]
            shifted_cvals = shifted_cvals[:self.bottom_ind+shift]


        if top:
            heights = np.zeros(shifted_slab.shape[:2])
            inds = np.zeros(shifted_slab.shape[:2], dtype=bool)
            for i in range(0, shifted_slab.shape[-1]-1)[::-1]:
                points = inds < (shifted_slab[:,:,i] > current)
                x1 = shifted_slab[points, i]
                x2 = shifted_slab[points, i+1]
                y1 = shifted_cvals[i]
                y2 = shifted_cvals[i+1]
                heights[points] = self._interp(
                    x=current,
                    x1=x1,
                    x2=x2,
                    y1=y1,
                    y2=y2,
                )
                inds[points] = True

            heights[heights <= self.top_surface] = heights[heights > self.top_surface].min()

        else:
            heights = np.zeros(shifted_slab.shape[:2])
            inds = np.zeros(shifted_slab.shape[:2], dtype=bool)
            for i in range(1, shifted_slab.shape[-1]):
                points = inds < (shifted_slab[:,:,i] > current)
                x1 = shifted_slab[points, i-1]
                x2 = shifted_slab[points, i]
                y1 = shifted_cvals[i-1]
                y2 = shifted_cvals[i]
                heights[points] = self._interp(
                    x=current,
                    x1=x1,
                    x2=x2,
                    y1=y1,
                    y2=y2,
                )
                inds[points] = True

            heights[heights >= self.top_surface] = heights[heights < self.top_surface].min()

        return heights

    def _generate_supercell(self, x, y, Z, scaling_matrix=[8,8]):
        x = np.concatenate([x + i for i in range(scaling_matrix[0])])
        y = np.concatenate([y + i for i in range(scaling_matrix[1])])
        Z = np.hstack([
            np.vstack([Z for _ in range(scaling_matrix[0])]) for _ in range(scaling_matrix[1])
        ])
        return x, y, Z

    def _get_intercept(self, midpoint, vector):
        if vector[0] == 0:
            intersect = [0, midpoint[1]]
        else:
            slope = vector[1] / vector[0]
            f = ((slope * midpoint[1]) + midpoint[0])/ ((slope**2) + 1)
            intersect = [f, slope * f]

        return intersect

    def _get_ratio(self, a, b):
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        ratio_array = np.ones(2)
        min_ind = np.argmin([a_norm, b_norm])
        ratio = np.max([a_norm, b_norm]) / np.min([a_norm, b_norm])
        ratio_array[min_ind] = ratio
        
        return ratio_array

    def _get_square(self, a, b):
        midpoint = 0.5 * (a + b)
        a_inter = self._get_intercept(midpoint, a)
        b_inter = self._get_intercept(midpoint, b)
        a_len = np.linalg.norm(a_inter - midpoint)
        b_len = np.linalg.norm(b_inter - midpoint)

        r = np.min([a_len, b_len])

        box_length = (2 * r) / np.sqrt(2)

        return box_length, midpoint

    def _get_scaling_matrix(self, a, b, scan_size=40):
        final_box_length = 0
        final_midpoint = np.zeros(2)
        ratio = self._get_ratio(a, b)
        scaling_matrix = np.ones(2)
        i = 1
        while final_box_length <= scan_size:
            i += 1
            a_new = a * int(i * ratio[0])
            b_new = b * int(i * ratio[1])
            box_length, midpoint = self._get_square(a_new, b_new)
            final_box_length = box_length
            final_midpoint = midpoint
            scaling_matrix[0] = int(i * ratio[0])
            scaling_matrix[1] = int(i * ratio[1])

        return scaling_matrix.astype(int), midpoint

    def _run_constant_current_scan(self, current, structure, top=True, scan_size=40):
        scaling_matrix, midpoint = self._get_scaling_matrix(
            a=structure.lattice.matrix[0,:2],
            b=structure.lattice.matrix[1,:2],
            scan_size=scan_size,
        )
        Z = self._get_constant_current_isosurface(current, top=top)
        if top:
            Z = np.abs(Z - self.top_surface)
        else:
            Z = np.abs(Z - self.bottom_surface)

        x, y, Z = self._generate_supercell(
            self.a_vals,
            self.b_vals,
            Z,
            scaling_matrix=scaling_matrix,
        )

        X, Y = np.meshgrid(x, y)

        conv_input = np.c_[np.ravel(X), np.ravel(Y), np.ravel(Z)]
        converted = structure.lattice.get_cartesian_coords(conv_input)
        X_conv, Y_conv, Z_conv = converted[:,0], converted[:,1], converted[:,2]
        X_conv = X_conv.reshape(X.shape)
        Y_conv = Y_conv.reshape(Y.shape)
        Z_conv = Z_conv.reshape(Z.shape)
        shifted_point = midpoint - (scan_size / 2)
        X_conv -= shifted_point[0]
        Y_conv -= shifted_point[1]

        return X_conv, Y_conv, Z_conv, midpoint, scaling_matrix

    def _plot_stm_general(
        self,
        ax,
        X,
        Y,
        Z,
        cmap,
    ):
        ax.pcolormesh(
            X,
            Y,
            Z.T,
            shading='gouraud',
            cmap=cmap,
            norm=Normalize(vmin=Z.min(), vmax=Z.max()),
            rasterized=True,
            antialiased=True,
        )

    def _plot_atoms_general(
        self,
        ax,
        atol,
        max_bond_length,
        atom_size,
        bond_line_width,
        scaling_matrix,
        midpoint,
        scan_size,
        legend,
        top,
        structure,
        atom_axis_bounds,
        atoms_box,
        legend_atom_size,
    ):
        supercell = make_supercell(
            structure,
            scaling_matrix=np.hstack([scaling_matrix, 1]),
        )
        inds, heights = group_layers(supercell, atol=atol)

        if top:
            surface_inds = inds[-1]
        else:
            surface_inds = inds[0]

        surface_atom_coords = supercell.cart_coords[surface_inds]
        surface_atom_symbols = np.array(supercell.species, dtype='str')[surface_inds]
        surface_atom_species = np.zeros(surface_atom_symbols.shape, dtype=int)
        surface_atom_sizes = np.zeros(surface_atom_symbols.shape, dtype=float)
        unique_species = np.unique(surface_atom_symbols)
        unique_elements = [Element(i) for i in unique_species]
        unique_zs = [Element(i).Z for i in unique_species]

        for i, z in enumerate(unique_elements):
            surface_atom_species[np.isin(surface_atom_symbols, unique_species[i])] = z.Z
            surface_atom_sizes[np.isin(surface_atom_symbols, unique_species[i])] = z.atomic_radius

        surface_atom_sizes /= surface_atom_sizes.max()

        colors = jmol_colors[surface_atom_species]

        shifted_point = midpoint - (scan_size / 2)
        surface_atom_coords[:,0] -= shifted_point[0]
        surface_atom_coords[:,1] -= shifted_point[1]

        neighbor_graph = radius_neighbors_graph(
            X=surface_atom_coords,
            radius=max_bond_length,
        ).toarray()

        bonds = []

        for i in range(neighbor_graph.shape[0]):
            for j in range(neighbor_graph.shape[0]):
                if neighbor_graph[i,j] > 0:
                    to_append = [
                        surface_atom_coords[i],
                        surface_atom_coords[j],
                        [np.nan, np.nan, np.nan],
                    ]
                    bonds.append(to_append)

        bonds = np.vstack(bonds)

        ax_atoms = ax.inset_axes(
            bounds=atom_axis_bounds,
        )
        ax_atoms.set_xlim(
            atom_axis_bounds[0] * scan_size,
            (atom_axis_bounds[0] + atom_axis_bounds[2]) * scan_size
        )
        ax_atoms.set_ylim(
            atom_axis_bounds[1] * scan_size,
            (atom_axis_bounds[1] + atom_axis_bounds[3]) * scan_size
        )
        ax_atoms.set_facecolor((0,0,0,0))

        ax_atoms.tick_params(
            bottom=False,
            left=False,
            labelbottom=False,
            labelleft=False,
        )

        if not atoms_box:
            ax_atoms.spines['left'].set_visible(False)
            ax_atoms.spines['right'].set_visible(False)
            ax_atoms.spines['top'].set_visible(False)
            ax_atoms.spines['bottom'].set_visible(False)

        ax_atoms.plot(
            bonds[:,0],
            bonds[:,1],
            color='lightgrey',
            linewidth=bond_line_width,
            zorder=5,
            path_effects=[pa.Stroke(linewidth=bond_line_width+2, foreground='black'), pa.Normal()],
        )
        ax_atoms.scatter(
            surface_atom_coords[:,0],
            surface_atom_coords[:,1],
            c=colors,
            ec='black',
            s=atom_size*surface_atom_sizes,
            zorder=10,
        )

        if legend:
            legend_lines = []
            legend_labels = []
            for name, color, element in zip(unique_species, jmol_colors[unique_zs], unique_elements):
                legend_lines.append(plt.scatter(
                    [-1],
                    [-1],
                    color=color,
                    s=legend_atom_size*element.atomic_radius,
                    ec='black',
                ))
                legend_labels.append(
                    f'{name}'
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

            l = ax.legend(
                handles,
                labels,
                ncol=1,
                loc='upper right',
                framealpha=1,
            )
            l.set_zorder(200)
            frame = l.get_frame()
            frame.set_facecolor('white')
            frame.set_edgecolor('black')

    def add_scale_bar(
        self,
        ax,
        width,
        height,
        offset_x=1,
        offset_y=1,
        units='nm',
        color='white',
        border_color='black',
        border_width=0.75,
        fontsize=16,
    ):
        rect = Rectangle(
            xy=(offset_x, offset_y),
            width=width,
            height=height,
            facecolor=color,
            edgecolor=border_color,
            linewidth=border_width,
            zorder=100,
        )
        if units == 'nm':
            annotation = f'{np.round(width / 10, 2)} nm'
        elif units == 'A':
            annotation = f'${width} \\AA$'
        else:
            raise('Please select proper units, either "nm" (nanometers) or "A" (angstroms)')

        ax.annotate(
            annotation,
            xy=(offset_x + (width / 2), offset_y + (1.05 * height)),
            xycoords='data',
            ha='center',
            va='bottom',
            color=color,
            fontsize=fontsize,
            fontweight='bold',
            path_effects=[pa.Stroke(linewidth=border_width, foreground=border_color)],
            zorder=100,
        )
        ax.add_patch(rect)


    def plot_constant_current(
        self,
        ax,
        current,
        top=True,
        scan_size=40,
        atol=0.03,
        plot_atoms=False,
        atom_size=80,
        legend_atom_size=100,
        bond_line_width=2,
        max_bond_length=3.14,
        cmap='hot',
        sigma=4,
        legend=False,
        rotation=0,
        atom_axis_bounds=[0.5,0.0,0.5,0.5],
        atoms_box=False,
    ):
        if rotation != 0:
            structure = self._rotate_structure(self.poscar.structure, angle=rotation)
        else:
            structure = self.poscar.structure

        X, Y, Z, midpoint, scaling_matrix = self._run_constant_current_scan(
            current=current,
            structure=structure,
            top=top,
            scan_size=scan_size,
        )
        Z = gaussian_filter(Z, sigma=sigma)
        shifted_midpoint = midpoint - (scan_size / 2)

        ax.set_xlim(0, scan_size)
        ax.set_ylim(0, scan_size)

        self._plot_stm_general(
            ax=ax,
            X=X,
            Y=Y,
            Z=Z,
            cmap=cmap,
        )

        if plot_atoms:
            self._plot_atoms_general(
                ax=ax,
                atol=atol,
                max_bond_length=max_bond_length,
                atom_size=atom_size,
                bond_line_width=bond_line_width,
                scaling_matrix=scaling_matrix,
                midpoint=midpoint,
                scan_size=scan_size,
                legend=legend,
                top=top,
                structure=structure,
                atom_axis_bounds=atom_axis_bounds,
                atoms_box=atoms_box,
                legend_atom_size=legend_atom_size,
            )


if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(4,4), dpi=400)
    ax.tick_params(
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )

    stm = STM(folder='../../vaspvis_data/InAs111A_stm/')
    stm.plot_constant_current(
        ax=ax,
        current=0.009,
        top=False,
        scan_size=40,
        plot_atoms=True,
        sigma=5,
        cmap='afmhot',
        atol=0.03,
        legend=True,
        rotation=-90,
        #  atom_size=20,
        #  bond_line_width=0.5,
        legend_atom_size=50,
        atom_axis_bounds=[0.5, 0.0, 0.5, 1],
        atoms_box=True,
    )
    stm.add_scale_bar(
        ax=ax,
        width=10,
        height=1.5,
        #  offset_x=3,
        #  fontsize=14,
    )
    fig.tight_layout(pad=0)
    fig.savefig('InAs111A_with_atoms.png')

