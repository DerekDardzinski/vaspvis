"""
This module contains standardized plots as well as more complex plots,
such as band structures and density of states put together, and spin
projected plots. 
"""

from vaspvis.band import Band
from vaspvis.dos import Dos
from vaspvis.utils import get_bandgap
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import Normalize, to_rgba, LinearSegmentedColormap
import numpy as np
import time


def _figure_setup(ax, fontsize=6, ylim=[-6, 6]):
    ax.set_ylabel('$E - E_{F}$ $(eV)$', fontsize=fontsize)
    ax.set_ylim(ylim[0], ylim[1])
    ax.tick_params(labelsize=fontsize, length=2.5)
    ax.tick_params(axis='x', length=0)


def _figure_setup_dos(ax, fontsize=6, energyaxis='y', log_scale=False):
    ax.tick_params(labelsize=fontsize, length=2.5)
    if energyaxis == 'y':
        ax.set_ylabel('$E - E_{F}$ $(eV)$', fontsize=fontsize)
        if log_scale:
            ax.set_xlabel('log(Density of States)', fontsize=fontsize)
        else:
            ax.set_xlabel('Density of States', fontsize=fontsize)
    if energyaxis == 'x':
        ax.set_xlabel('$E - E_{F}$ $(eV)$', fontsize=fontsize)
        if log_scale:
            ax.set_ylabel('log(Density of States)', fontsize=fontsize)
        else:
            ax.set_ylabel('Density of States', fontsize=fontsize)



def _figure_setup_band_dos(ax, fontsize, ylim):
    ax1 = ax[0]
    ax2 = ax[1]
    ax2.tick_params(axis='y', length=0)
    ax2.tick_params(axis='x', length=0, labelsize=fontsize)
    ax2.set_xlabel('Density of States', fontsize=fontsize)
    ax1.tick_params(labelsize=fontsize)
    ax1.tick_params(axis='x', length=0)
    ax1.set_ylabel('$E - E_{F}$ $(eV)$', fontsize=fontsize)
    ax1.set_xlabel('Wave Vector', fontsize=fontsize)
    ax1.set_ylim(ylim[0], ylim[1])

    return ax1, ax2


def _figure_setup_band_dos_spin_polarized(ax, fontsize, ylim):
    ax_band_up = ax[0, 0]
    ax_dos_up = ax[0, 1]
    ax_band_down = ax[1, 0]
    ax_dos_down = ax[1, 1]

    ax_dos_up.tick_params(axis='y', length=0)
    ax_dos_up.tick_params(axis='x', length=0,
                          labelsize=fontsize, labelbottom=False)
    ax_band_up.tick_params(labelsize=fontsize)
    ax_band_up.tick_params(axis='x', length=0, labelbottom=False)
    ax_band_up.set_ylabel('$E - E_{F}$ $(eV)$', fontsize=fontsize)
    ax_band_up.set_ylim(ylim[0], ylim[1])

    ax_dos_down.tick_params(axis='y', length=0)
    ax_dos_down.tick_params(axis='x', length=0, labelsize=fontsize)
    ax_dos_down.set_xlabel('Density of States', fontsize=fontsize)
    ax_band_down.tick_params(labelsize=fontsize)
    ax_band_down.tick_params(axis='x', length=0)
    ax_band_down.set_ylabel('$E - E_{F}$ $(eV)$', fontsize=fontsize)
    ax_band_down.set_xlabel('Wave Vector', fontsize=fontsize)
    ax_band_down.set_ylim(ylim[0], ylim[1])

    return ax_band_up, ax_dos_up, ax_band_down, ax_dos_down


def _figure_setup_layer_dos(ax, fontsize=6, energyaxis='y'):
    ax.tick_params(labelsize=fontsize)
    if energyaxis == 'y':
        ax.set_ylabel('$E - E_{F}$ $(eV)$', fontsize=fontsize)
        ax.set_xlabel('Layers', fontsize=fontsize)
    if energyaxis == 'x':
        ax.set_xlabel('$E - E_{F}$ $(eV)$', fontsize=fontsize)
        ax.set_ylabel('Layers', fontsize=fontsize)

def band_plain(
    folder,
    output='band_plain.png',
    spin='up',
    color='black',
    linewidth=1.25,
    linestyle='-',
    figsize=(4, 3),
    erange=[-6, 6],
    kpath=None,
    custom_kpath=None,
    n=None,
    unfold=False,
    M=None,
    high_symm_points=None,
    fontsize=12,
    scale_factor=5,
    heatmap=False,
    bins=1000,
    sigma=2,
    cmap='hot',
    vlinecolor='black',
    cbar=True,
    cbar_orientation='horizontal',
    powernorm=True,
    gamma=0.7,
    save=True,
    shift_efermi=0,
    interpolate=False,
    new_n=200,
    highlight_band=False,
    highlight_band_color='red',
    band_index=None,
):
    """
    This function generates a plain band structure

    Parameters:
        folder (str): This is the folder that contains the VASP files
        output (str): File name of the resulting plot.
        spin (str): Choose which spin direction to parse. ('up' or 'down')
        color (str): Color of the band structure lines
        linewidth (float): Line width of the band structure lines
        linestyle (str): Line style of the bands
        figsize (list / tuple): Desired size of the image in inches (width, height)
        erange (list / tuple): Range of energy to show in the plot [low, high]
        kpath (list[list]): High symmetry k-point path of band structure calculation
            Due to the nature of the KPOINTS file for unfolded calculations this
            information is a required input for proper labeling of the figure
            for unfolded calculations. This information is extracted from the KPOINTS
            files for non-unfolded calculations. (G is automatically converted to \\Gamma)
	    (e.g. For the path X-G-X, kpath=[['X', 'G'], ['G', 'X']])
	custom_kpath (list): This gives the option to only plot specific segments of a given band structure
	    calculation. For example if the kpath was G-X-W-L then there are three segements to choose from:
	    G-X, X-W, and W-L. In this case the default kpath could be plotted by defining custom_kpath=[1,2,3],
	    where 1 -> G-X, 2 -> X-W, and 3 -> W-L. If only G-X and X-W were desired then custom_kpath=[1,2].
	    If one of the segements should be flipped it can be done by making its value negative
	    (e.g. -1 -> X-G, -2 -> W-X, -3 -> L-W)
        n (int): Number of points between each high symmetry points.
            This is also only required for unfolded calculations. This number should be 
            known by the user, as it was used to generate the KPOINTS file.
        unfold (bool): Determines if the plotted band structure is from a band unfolding calculation.
        M (list[list]): Transformation matrix from the primitive bulk structure to the slab structure.
            Only required for a band unfolding calculation.
        high_symm_points (list[list]): List of fractional coordinated for each high symmetry point in 
            the band structure path. Only required for a band unfolding calculation.
        fontsize (float): Font size of the text in the figure.
        scale_factor (float): Factor to scale scatter points in unfolded plots.
        heatmap (bool): Determines if an unfolded plot is plotted as a heat map. Only valid from unfolded plots.
        bins (int): number of bins included in the histogram. More bin will result in higher resolution.
        sigma (float): Value used to smooth the heatmap.
        cmap (str): Matplotlib colormap for the unfolded heatmap.
        vlinecolor (str): Color of the vertical lines in the band structure
            useful for heatmaps when there is a dark background color.
        save (bool): Determines whether to automatically save the figure or not. If not 
            the figure and axis are return for further manipulation.

    Returns:
        If save == True, this function will return nothing and directly save the image as
        the output name. If save == False, the function will return the matplotlib figure
        and axis for further editing. 
    """

    band = Band(
        folder=folder,
        spin=spin,
        unfold=unfold,
        high_symm_points=high_symm_points, 
	interpolate=interpolate,
	new_n=new_n, 
        kpath=kpath,
	custom_kpath=custom_kpath,
        n=n,
        M=M,
	shift_efermi=shift_efermi,
    )

    if heatmap:
        if cbar:
            if cbar_orientation == 'horizontal':
                fig, (ax, cax) = plt.subplots(
                    nrows=2,
                    figsize=figsize,
                    gridspec_kw={"height_ratios": [1, 0.05]},
                    dpi=400,
                    constrained_layout=True,
                )
            elif cbar_orientation == 'vertical':
                fig, (ax, cax) = plt.subplots(
                    ncols=2,
                    figsize=figsize,
                    gridspec_kw={"width_ratios": [1, 0.05]},
                    dpi=400,
                    constrained_layout=True,
                )
            else:
                raise('This is not a valid orientation please choose either horizontal or vertical')
        else:
            fig = plt.figure(figsize=(figsize), dpi=400)
            ax = fig.add_subplot(111)
    else:
        fig = plt.figure(figsize=(figsize), dpi=400)
        ax = fig.add_subplot(111)

    _figure_setup(ax=ax, fontsize=fontsize, ylim=[erange[0], erange[1]])

    band.plot_plain(
        ax=ax,
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        erange=erange,
        scale_factor=scale_factor,
        heatmap=heatmap,
        bins=bins,
        sigma=sigma,
        cmap=cmap,
        vlinecolor=vlinecolor,
        powernorm=powernorm,
        gamma=gamma,
        highlight_band=highlight_band,
        highlight_band_color=highlight_band_color,
        band_index=band_index,
    )

    if heatmap:
        if cbar:
            im = ax.collections[0]
            min_val = im.norm.vmin
            max_val = im.norm.vmax
            cbar = fig.colorbar(im, cax=cax, orientation=cbar_orientation)
            cbar.set_ticks([min_val, max_val])
            cbar.set_ticklabels(['min', 'max'])

    if heatmap:
        if not cbar:
            fig.tight_layout(pad=0.4)
    else:
        fig.tight_layout(pad=0.4)

    if save:
        plt.savefig(output)
    else:
        return fig, ax


def band_spd(
    folder,
    output='band_spd.png',
    spin='up',
    scale_factor=5,
    orbitals='spd',
    display_order=None,
    color_list=None,
    legend=True,
    linewidth=0.75,
    band_color='black',
    figsize=(4, 3),
    erange=[-6, 6],
    kpath=None,
    custom_kpath=None,
    n=None,
    unfold=False,
    M=None,
    high_symm_points=None,
    fontsize=12,
    heatmap=False,
    bins=800,
    sigma=2,
    cmap='hot',
    vlinecolor='black',
    cbar=True,
    cbar_orientation='horizontal',
    powernorm=True,
    gamma=0.5,
    save=True,
    shift_efermi=0,
    interpolate=False,
    new_n=200,
):
    """
    This function generates a s, p, d projected band structure.
    
    Parameters:
        folder (str): This is the folder that contains the VASP files
        orbitals (str): String that contains the s, p, or d orbitals that to project onto.
            The default is 'spd', if the user only wanted to project onto the p, and d orbitals 
            than 'pd' should be passed in
        output (str): File name of the resulting plot.
        spin (str): Choose which spin direction to parse. ('up' or 'down')
        scale_factor (float): Factor to scale weights. This changes the size of the
            points in the scatter plot.
        display_order (str / None): If None, the projections will be displayed in the same order
            the user inputs them. If 'all' the projections will be plotted from largest to smallest
            so every point is visable. If 'dominant' the projections will be plotted from smallest 
            to largest so only the dominant projection is shown.
        color_list (list): List of colors that is the same length as the number of projections
            in the plot.
        legend (bool): Determines if the legend should be included or not.
        linewidth (float): Line width of the plain band structure plotted in the background.
        band_color (string): Color of the plain band structure.
        figsize (list / tuple): Desired size of the image in inches. (width, height)
        erange (list / tuple): Range of energy to show in the plot. [low, high]
        kpath (list[list]): High symmetry k-point path of band structure calculation
            Due to the nature of the KPOINTS file for unfolded calculations this
            information is a required input for proper labeling of the figure
            for unfolded calculations. This information is extracted from the KPOINTS
            files for non-unfolded calculations. (G is automatically converted to \\Gamma)
	    (e.g. For the path X-G-X, kpath=[['X', 'G'], ['G', 'X']])
	custom_kpath (list): This gives the option to only plot specific segments of a given band structure
	    calculation. For example if the kpath was G-X-W-L then there are three segements to choose from:
	    G-X, X-W, and W-L. In this case the default kpath could be plotted by defining custom_kpath=[1,2,3],
	    where 1 -> G-X, 2 -> X-W, and 3 -> W-L. If only G-X and X-W were desired then custom_kpath=[1,2].
	    If one of the segements should be flipped it can be done by making its value negative
	    (e.g. -1 -> X-G, -2 -> W-X, -3 -> L-W)
        n (int): Number of points between each high symmetry points.
            This is also only required for unfolded calculations and band unfolding. This number should be 
            known by the user, as it was used to generate the KPOINTS file.
        unfold (bool): Determines if the plotted band structure is from a band unfolding calculation.
        M (list[list]): Transformation matrix from the primitive bulk structure to the slab structure.
            Only required for a band unfolding calculation.
        high_symm_points (list[list]): List of fractional coordinated for each high symmetry point in 
            the band structure path. Only required for a band unfolding calculation.
        fontsize (float): Font size of the text in the figure.
        save (bool): Determines whether to automatically save the figure or not. If not 
            the figure and axis are return for further manipulation.

    Returns:
        If save == True, this function will return nothing and directly save the image as
        the output name. If save == False, the function will return the matplotlib figure
        and axis for further editing.
    """

    band = Band(
        folder=folder,
        spin=spin,
        projected=True,
        unfold=unfold,
        high_symm_points=high_symm_points, 
	interpolate=interpolate,
	new_n=new_n,
        kpath=kpath,
	custom_kpath=custom_kpath,
        n=n,
        M=M,
	shift_efermi=shift_efermi,
    )

    if heatmap:
        legend = False
        if cbar:
            if cbar_orientation == 'horizontal':
                fig, (ax, cax) = plt.subplots(
                    nrows=2,
                    figsize=figsize,
                    gridspec_kw={"height_ratios": [1, 0.05]},
                    dpi=400,
                    constrained_layout=True,
                )
            elif cbar_orientation == 'vertical':
                fig, (ax, cax) = plt.subplots(
                    ncols=2,
                    figsize=figsize,
                    gridspec_kw={"width_ratios": [1, 0.05]},
                    dpi=400,
                    constrained_layout=True,
                )
            else:
                raise('This is not a valid orientation please choose either horizontal or vertical')
        else:
            fig = plt.figure(figsize=(figsize), dpi=400)
            ax = fig.add_subplot(111)
    else:
        fig = plt.figure(figsize=(figsize), dpi=400)
        ax = fig.add_subplot(111)

    _figure_setup(ax=ax, fontsize=fontsize, ylim=[erange[0], erange[1]])
    band.plot_spd(
        ax=ax,
        scale_factor=scale_factor,
        orbitals=orbitals,
        erange=erange,
        display_order=display_order,
        color_list=color_list,
        legend=legend,
        linewidth=linewidth,
        band_color=band_color,
        heatmap=heatmap,
        bins=bins,
        sigma=sigma,
        cmap=cmap,
        vlinecolor=vlinecolor,
        powernorm=powernorm,
        gamma=gamma,
    )

    if heatmap:
        if cbar:
            im = ax.collections[0]
            min_val = im.norm.vmin
            max_val = im.norm.vmax
            cbar = fig.colorbar(im, cax=cax, orientation=cbar_orientation)
            cbar.set_ticks([min_val, max_val])
            cbar.set_ticklabels(['min', 'max'])

    if heatmap:
        if not cbar:
            fig.tight_layout(pad=0.4)
    else:
        fig.tight_layout(pad=0.4)

    if save:
        plt.savefig(output)
    else:
        return fig, ax


def band_atom_orbitals(
    folder,
    atom_orbital_dict,
    output='band_atom_orbitals.png',
    spin='up',
    display_order=None,
    scale_factor=5,
    color_list=None,
    legend=True,
    linewidth=0.75,
    band_color='black',
    figsize=(4, 3),
    erange=[-6, 6],
    kpath=None,
    custom_kpath=None,
    n=None,
    unfold=False,
    M=None,
    high_symm_points=None,
    fontsize=12,
    heatmap=False,
    bins=800,
    sigma=2,
    cmap='hot',
    vlinecolor='black',
    cbar=True,
    cbar_orientation='horizontal',
    powernorm=True,
    gamma=0.5,
    save=True,
    shift_efermi=0,
    interpolate=False,
    new_n=200,
):
    """
    This function generates a projected band structure on specific [atom, orbital] pairs.

    Parameters:
        folder (str): This is the folder that contains the VASP files
        atom_orbital_dict (dict[int:list]): A dictionary that contains the individual atoms and the corresponding
            orbitals to project onto. For example, if the user wants to project onto the s, py, pz, and px orbitals
            of the first atom and the s orbital of the second atom then the dictionary would be {0:[0,1,2,3], 1:[0]}
        output (str): File name of the resulting plot.
        spin (str): Choose which spin direction to parse. ('up' or 'down')
        scale_factor (float): Factor to scale weights. This changes the size of the
            points in the scatter plot.
        display_order (str / None): If None, the projections will be displayed in the same order
            the user inputs them. If 'all' the projections will be plotted from largest to smallest
            so every point is visable. If 'dominant' the projections will be plotted from smallest 
            to largest so only the dominant projection is shown.
        color_list (list): List of colors that is the same length as the number of projections
            in the plot.
        legend (bool): Determines if the legend should be included or not.
        linewidth (float): Line width of the plain band structure plotted in the background.
        band_color (string): Color of the plain band structure.
        figsize (list / tuple): Desired size of the image in inches. (width, height)
        erange (list / tuple): Range of energy to show in the plot. [low, high]
        kpath (list[list]): High symmetry k-point path of band structure calculation
            Due to the nature of the KPOINTS file for unfolded calculations this
            information is a required input for proper labeling of the figure
            for unfolded calculations. This information is extracted from the KPOINTS
            files for non-unfolded calculations. (G is automatically converted to \\Gamma)
	    (e.g. For the path X-G-X, kpath=[['X', 'G'], ['G', 'X']])
	custom_kpath (list): This gives the option to only plot specific segments of a given band structure
	    calculation. For example if the kpath was G-X-W-L then there are three segements to choose from:
	    G-X, X-W, and W-L. In this case the default kpath could be plotted by defining custom_kpath=[1,2,3],
	    where 1 -> G-X, 2 -> X-W, and 3 -> W-L. If only G-X and X-W were desired then custom_kpath=[1,2].
	    If one of the segements should be flipped it can be done by making its value negative
	    (e.g. -1 -> X-G, -2 -> W-X, -3 -> L-W)
        n (int): Number of points between each high symmetry points.
            This is also only required for unfolded calculations and band unfolding. This number should be 
            known by the user, as it was used to generate the KPOINTS file.
        unfold (bool): Determines if the plotted band structure is from a band unfolding calculation.
        M (list[list]): Transformation matrix from the primitive bulk structure to the slab structure.
            Only required for a band unfolding calculation.
        high_symm_points (list[list]): List of fractional coordinated for each high symmetry point in 
            the band structure path. Only required for a band unfolding calculation.
        fontsize (float): Font size of the text in the figure.
        save (bool): Determines whether to automatically save the figure or not. If not 
            the figure and axis are return for further manipulation.

    Returns:
        If save == True, this function will return nothing and directly save the image as
        the output name. If save == False, the function will return the matplotlib figure
        and axis for further editing.
    """

    band = Band(
        folder=folder,
        spin=spin,
        projected=True,
        unfold=unfold,
        high_symm_points=high_symm_points, 
	interpolate=interpolate,
	new_n=new_n,
        kpath=kpath,
	custom_kpath=custom_kpath,
        n=n,
        M=M,
	shift_efermi=shift_efermi,
    )

    if heatmap:
        legend = False
        if cbar:
            if cbar_orientation == 'horizontal':
                fig, (ax, cax) = plt.subplots(
                    nrows=2,
                    figsize=figsize,
                    gridspec_kw={"height_ratios": [1, 0.05]},
                    dpi=400,
                    constrained_layout=True,
                )
            elif cbar_orientation == 'vertical':
                fig, (ax, cax) = plt.subplots(
                    ncols=2,
                    figsize=figsize,
                    gridspec_kw={"width_ratios": [1, 0.05]},
                    dpi=400,
                    constrained_layout=True,
                )
            else:
                raise('This is not a valid orientation please choose either horizontal or vertical')
        else:
            fig = plt.figure(figsize=(figsize), dpi=400)
            ax = fig.add_subplot(111)
    else:
        fig = plt.figure(figsize=(figsize), dpi=400)
        ax = fig.add_subplot(111)

    _figure_setup(ax=ax, fontsize=fontsize, ylim=[erange[0], erange[1]])
    band.plot_atom_orbitals(
        ax=ax,
        atom_orbital_dict=atom_orbital_dict,
        display_order=display_order,
        scale_factor=scale_factor,
        color_list=color_list,
        legend=legend,
        linewidth=linewidth,
        band_color=band_color,
        erange=erange,
        heatmap=heatmap,
        bins=bins,
        sigma=sigma,
        cmap=cmap,
        vlinecolor=vlinecolor,
        powernorm=powernorm,
        gamma=gamma,
    )

    if heatmap:
        if cbar:
            im = ax.collections[0]
            min_val = im.norm.vmin
            max_val = im.norm.vmax
            cbar = fig.colorbar(im, cax=cax, orientation=cbar_orientation)
            cbar.set_ticks([min_val, max_val])
            cbar.set_ticklabels(['min', 'max'])

    if heatmap:
        if not cbar:
            fig.tight_layout(pad=0.4)
    else:
        fig.tight_layout(pad=0.4)

    if save:
        plt.savefig(output)
    else:
        return fig, ax


def band_orbitals(
    folder,
    orbitals,
    output='band_orbital.png',
    spin='up',
    scale_factor=5,
    display_order=None,
    color_list=None,
    legend=True,
    linewidth=0.75,
    band_color='black',
    figsize=(4, 3),
    erange=[-6, 6],
    kpath=None,
    custom_kpath=None,
    n=None,
    unfold=False,
    M=None,
    high_symm_points=None,
    fontsize=12,
    heatmap=False,
    bins=800,
    sigma=2,
    cmap='hot',
    vlinecolor='black',
    cbar=True,
    cbar_orientation='horizontal',
    powernorm=True,
    gamma=0.5,
    save=True,
    shift_efermi=0,
    interpolate=False,
    new_n=200,
):
    """
    This function generates a projected band structure on specific orbitals.

    Parameters:
        folder (str): This is the folder that contains the VASP files
        orbitals (list): List of orbitals to compare

            | 0 = s
            | 1 = py
            | 2 = pz
            | 3 = px
            | 4 = dxy
            | 5 = dyz
            | 6 = dz2
            | 7 = dxz
            | 8 = dx2-y2
            | 9 = fy3x2
            | 10 = fxyz
            | 11 = fyz2
            | 12 = fz3
            | 13 = fxz2
            | 14 = fzx3
            | 15 = fx3

        output (str): File name of the resulting plot.
        spin (str): Choose which spin direction to parse. ('up' or 'down')
        scale_factor (float): Factor to scale weights. This changes the size of the
            points in the scatter plot.
        display_order (str / None): If None, the projections will be displayed in the same order
            the user inputs them. If 'all' the projections will be plotted from largest to smallest
            so every point is visable. If 'dominant' the projections will be plotted from smallest 
            to largest so only the dominant projection is shown.
        color_list (list): List of colors that is the same length as the number of projections
            in the plot.
        legend (bool): Determines if the legend should be included or not.
        linewidth (float): Line width of the plain band structure plotted in the background.
        band_color (string): Color of the plain band structure.
        figsize (list / tuple): Desired size of the image in inches. (width, height)
        erange (list / tuple): Range of energy to show in the plot. [low, high]
        kpath (list[list]): High symmetry k-point path of band structure calculation
            Due to the nature of the KPOINTS file for unfolded calculations this
            information is a required input for proper labeling of the figure
            for unfolded calculations. This information is extracted from the KPOINTS
            files for non-unfolded calculations. (G is automatically converted to \\Gamma)
	    (e.g. For the path X-G-X, kpath=[['X', 'G'], ['G', 'X']])
	custom_kpath (list): This gives the option to only plot specific segments of a given band structure
	    calculation. For example if the kpath was G-X-W-L then there are three segements to choose from:
	    G-X, X-W, and W-L. In this case the default kpath could be plotted by defining custom_kpath=[1,2,3],
	    where 1 -> G-X, 2 -> X-W, and 3 -> W-L. If only G-X and X-W were desired then custom_kpath=[1,2].
	    If one of the segements should be flipped it can be done by making its value negative
	    (e.g. -1 -> X-G, -2 -> W-X, -3 -> L-W)
        n (int): Number of points between each high symmetry points.
            This is also only required for unfolded calculations and band unfolding. This number should be 
            known by the user, as it was used to generate the KPOINTS file.
        unfold (bool): Determines if the plotted band structure is from a band unfolding calculation.
        M (list[list]): Transformation matrix from the primitive bulk structure to the slab structure.
            Only required for a band unfolding calculation.
        high_symm_points (list[list]): List of fractional coordinated for each high symmetry point in 
            the band structure path. Only required for a band unfolding calculation.
        fontsize (float): Font size of the text in the figure.
        save (bool): Determines whether to automatically save the figure or not. If not 
            the figure and axis are return for further manipulation.

    Returns:
        If save == True, this function will return nothing and directly save the image as
        the output name. If save == False, the function will return the matplotlib figure
        and axis for further editing.
    """

    band = Band(
        folder=folder,
        spin=spin,
        projected=True,
        unfold=unfold,
        high_symm_points=high_symm_points, 
	interpolate=interpolate,
	new_n=new_n,
        kpath=kpath,
	custom_kpath=custom_kpath,
        n=n,
        M=M,
	shift_efermi=shift_efermi,
    )

    if heatmap:
        legend = False
        if cbar:
            if cbar_orientation == 'horizontal':
                fig, (ax, cax) = plt.subplots(
                    nrows=2,
                    figsize=figsize,
                    gridspec_kw={"height_ratios": [1, 0.05]},
                    dpi=400,
                    constrained_layout=True,
                )
            elif cbar_orientation == 'vertical':
                fig, (ax, cax) = plt.subplots(
                    ncols=2,
                    figsize=figsize,
                    gridspec_kw={"width_ratios": [1, 0.05]},
                    dpi=400,
                    constrained_layout=True,
                )
            else:
                raise('This is not a valid orientation please choose either horizontal or vertical')
        else:
            fig = plt.figure(figsize=(figsize), dpi=400)
            ax = fig.add_subplot(111)
    else:
        fig = plt.figure(figsize=(figsize), dpi=400)
        ax = fig.add_subplot(111)

    _figure_setup(ax=ax, fontsize=fontsize, ylim=[erange[0], erange[1]])
    band.plot_orbitals(
        ax=ax,
        orbitals=orbitals,
        scale_factor=scale_factor,
        erange=erange,
        display_order=display_order,
        color_list=color_list,
        legend=legend,
        linewidth=linewidth,
        band_color=band_color,
        heatmap=heatmap,
        bins=bins,
        sigma=sigma,
        cmap=cmap,
        vlinecolor=vlinecolor,
        powernorm=powernorm,
        gamma=gamma,
    )

    if heatmap:
        if cbar:
            im = ax.collections[0]
            min_val = im.norm.vmin
            max_val = im.norm.vmax
            cbar = fig.colorbar(im, cax=cax, orientation=cbar_orientation)
            cbar.set_ticks([min_val, max_val])
            cbar.set_ticklabels(['min', 'max'])

    if heatmap:
        if not cbar:
            fig.tight_layout(pad=0.4)
    else:
        fig.tight_layout(pad=0.4)

    if save:
        plt.savefig(output)
    else:
        return fig, ax


def band_atoms(
    folder,
    atoms,
    output='band_atoms.png',
    spin='up',
    display_order=None,
    scale_factor=5,
    color_list=None,
    legend=True,
    linewidth=0.75,
    band_color='black',
    figsize=(4, 3),
    erange=[-6, 6],
    kpath=None,
    custom_kpath=None,
    n=None,
    unfold=False,
    M=None,
    high_symm_points=None,
    fontsize=12,
    heatmap=False,
    bins=800,
    sigma=2,
    cmap='hot',
    vlinecolor='black',
    cbar=True,
    cbar_orientation='horizontal',
    powernorm=True,
    gamma=0.5,
    save=True,
    shift_efermi=0,
    interpolate=False,
    new_n=200,
):
    """
    This function generates a projected band structure on specific atoms in the POSCAR.

    Parameters:
        folder (str): This is the folder that contains the VASP files
        atoms (list): List of atoms to project onto. The indices should be zero indexed (first atom is 0)
            and the atoms are in the same order as they are in the POSCAR
        output (str): File name of the resulting plot.
        spin (str): Choose which spin direction to parse. ('up' or 'down')
        scale_factor (float): Factor to scale weights. This changes the size of the
            points in the scatter plot.
        display_order (str / None): If None, the projections will be displayed in the same order
            the user inputs them. If 'all' the projections will be plotted from largest to smallest
            so every point is visable. If 'dominant' the projections will be plotted from smallest 
            to largest so only the dominant projection is shown.
        color_list (list): List of colors that is the same length as the number of projections
            in the plot.
        legend (bool): Determines if the legend should be included or not.
        linewidth (float): Line width of the plain band structure plotted in the background.
        band_color (string): Color of the plain band structure.
        figsize (list / tuple): Desired size of the image in inches. (width, height)
        erange (list / tuple): Range of energy to show in the plot. [low, high]
        kpath (list[list]): High symmetry k-point path of band structure calculation
            Due to the nature of the KPOINTS file for unfolded calculations this
            information is a required input for proper labeling of the figure
            for unfolded calculations. This information is extracted from the KPOINTS
            files for non-unfolded calculations. (G is automatically converted to \\Gamma)
	    (e.g. For the path X-G-X, kpath=[['X', 'G'], ['G', 'X']])
	custom_kpath (list): This gives the option to only plot specific segments of a given band structure
	    calculation. For example if the kpath was G-X-W-L then there are three segements to choose from:
	    G-X, X-W, and W-L. In this case the default kpath could be plotted by defining custom_kpath=[1,2,3],
	    where 1 -> G-X, 2 -> X-W, and 3 -> W-L. If only G-X and X-W were desired then custom_kpath=[1,2].
	    If one of the segements should be flipped it can be done by making its value negative
	    (e.g. -1 -> X-G, -2 -> W-X, -3 -> L-W)
        n (int): Number of points between each high symmetry points.
            This is also only required for unfolded calculations and band unfolding. This number should be 
            known by the user, as it was used to generate the KPOINTS file.
        unfold (bool): Determines if the plotted band structure is from a band unfolding calculation.
        M (list[list]): Transformation matrix from the primitive bulk structure to the slab structure.
            Only required for a band unfolding calculation.
        high_symm_points (list[list]): List of fractional coordinated for each high symmetry point in 
            the band structure path. Only required for a band unfolding calculation.
        fontsize (float): Font size of the text in the figure.
        save (bool): Determines whether to automatically save the figure or not. If not 
            the figure and axis are return for further manipulation.

    Returns:
        If save == True, this function will return nothing and directly save the image as
        the output name. If save == False, the function will return the matplotlib figure
        and axis for further editing.
    """
    band = Band(
        folder=folder,
        spin=spin,
        projected=True,
        unfold=unfold,
        high_symm_points=high_symm_points, 
	interpolate=interpolate,
	new_n=new_n,
        kpath=kpath,
	custom_kpath=custom_kpath,
        n=n,
        M=M,
	shift_efermi=shift_efermi,
    )

    if heatmap:
        legend = False
        if cbar:
            if cbar_orientation == 'horizontal':
                fig, (ax, cax) = plt.subplots(
                    nrows=2,
                    figsize=figsize,
                    gridspec_kw={"height_ratios": [1, 0.05]},
                    dpi=400,
                    constrained_layout=True,
                )
            elif cbar_orientation == 'vertical':
                fig, (ax, cax) = plt.subplots(
                    ncols=2,
                    figsize=figsize,
                    gridspec_kw={"width_ratios": [1, 0.05]},
                    dpi=400,
                    constrained_layout=True,
                )
            else:
                raise('This is not a valid orientation please choose either horizontal or vertical')
        else:
            fig = plt.figure(figsize=(figsize), dpi=400)
            ax = fig.add_subplot(111)
    else:
        fig = plt.figure(figsize=(figsize), dpi=400)
        ax = fig.add_subplot(111)

    _figure_setup(ax=ax, fontsize=fontsize, ylim=[erange[0], erange[1]])
    band.plot_atoms(
        ax=ax,
        atoms=atoms,
        scale_factor=scale_factor,
        color_list=color_list,
        display_order=display_order,
        legend=legend,
        linewidth=linewidth,
        band_color=band_color,
        erange=erange,
        heatmap=heatmap,
        bins=bins,
        sigma=sigma,
        cmap=cmap,
        vlinecolor=vlinecolor,
        powernorm=powernorm,
        gamma=gamma,
    )

    if heatmap:
        if cbar:
            im = ax.collections[0]
            min_val = im.norm.vmin
            max_val = im.norm.vmax
            cbar = fig.colorbar(im, cax=cax, orientation=cbar_orientation)
            cbar.set_ticks([min_val, max_val])
            cbar.set_ticklabels(['min', 'max'])

    if heatmap:
        if not cbar:
            fig.tight_layout(pad=0.4)
    else:
        fig.tight_layout(pad=0.4)

    if save:
        plt.savefig(output)
    else:
        return fig, ax

def band_atom_spd(
    folder,
    atom_spd_dict,
    output='band_atom_spd.png',
    spin='up',
    display_order=None,
    scale_factor=5,
    color_list=None,
    legend=True,
    linewidth=0.75,
    band_color='black',
    figsize=(4, 3),
    erange=[-6, 6],
    kpath=None,
    custom_kpath=None,
    n=None,
    unfold=False,
    M=None,
    high_symm_points=None,
    fontsize=12,
    heatmap=False,
    bins=800,
    sigma=2,
    cmap='hot',
    vlinecolor='black',
    cbar=True,
    cbar_orientation='horizontal',
    powernorm=True,
    gamma=0.5,
    save=True,
    shift_efermi=0,
    interpolate=False,
    new_n=200,
):
    """
    This function generates a s, p, d projected band structure on specific atoms.

    Parameters:
        folder (str): This is the folder that contains the VASP files
        atom_spd_dict (dict[int:str]): A dictionary that contains the individual atoms and the
            orbitals to project onto. For example, if the user wants to project onto the s, p, d orbitals corresponding 
            of the first atom and the p orbitals of the second atom then the dictionary would be {0:'spd', 1:'p'}
        output (str): File name of the resulting plot.
        spin (str): Choose which spin direction to parse. ('up' or 'down')
        scale_factor (float): Factor to scale weights. This changes the size of the
            points in the scatter plot.
        display_order (str / None): If None, the projections will be displayed in the same order
            the user inputs them. If 'all' the projections will be plotted from largest to smallest
            so every point is visable. If 'dominant' the projections will be plotted from smallest 
            to largest so only the dominant projection is shown.
        color_list (list): List of colors that is the same length as the number of projections
            in the plot.
        legend (bool): Determines if the legend should be included or not.
        linewidth (float): Line width of the plain band structure plotted in the background.
        band_color (string): Color of the plain band structure.
        figsize (list / tuple): Desired size of the image in inches. (width, height)
        erange (list / tuple): Range of energy to show in the plot. [low, high]
        kpath (list[list]): High symmetry k-point path of band structure calculation
            Due to the nature of the KPOINTS file for unfolded calculations this
            information is a required input for proper labeling of the figure
            for unfolded calculations. This information is extracted from the KPOINTS
            files for non-unfolded calculations. (G is automatically converted to \\Gamma)
	    (e.g. For the path X-G-X, kpath=[['X', 'G'], ['G', 'X']])
	custom_kpath (list): This gives the option to only plot specific segments of a given band structure
	    calculation. For example if the kpath was G-X-W-L then there are three segements to choose from:
	    G-X, X-W, and W-L. In this case the default kpath could be plotted by defining custom_kpath=[1,2,3],
	    where 1 -> G-X, 2 -> X-W, and 3 -> W-L. If only G-X and X-W were desired then custom_kpath=[1,2].
	    If one of the segements should be flipped it can be done by making its value negative
	    (e.g. -1 -> X-G, -2 -> W-X, -3 -> L-W)
        n (int): Number of points between each high symmetry points.
            This is also only required for unfolded calculations and band unfolding. This number should be 
            known by the user, as it was used to generate the KPOINTS file.
        unfold (bool): Determines if the plotted band structure is from a band unfolding calculation.
        M (list[list]): Transformation matrix from the primitive bulk structure to the slab structure.
            Only required for a band unfolding calculation.
        high_symm_points (list[list]): List of fractional coordinated for each high symmetry point in 
            the band structure path. Only required for a band unfolding calculation.
        fontsize (float): Font size of the text in the figure.
        save (bool): Determines whether to automatically save the figure or not. If not 
            the figure and axis are return for further manipulation.

    Returns:
        If save == True, this function will return nothing and directly save the image as
        the output name. If save == False, the function will return the matplotlib figure
        and axis for further editing.
    """

    band = Band(
        folder=folder,
        spin=spin,
        projected=True,
        unfold=unfold,
        high_symm_points=high_symm_points, 
	interpolate=interpolate,
	new_n=new_n,
        kpath=kpath,
	custom_kpath=custom_kpath,
        n=n,
        M=M,
	shift_efermi=shift_efermi,
    )

    if heatmap:
        legend = False
        if cbar:
            if cbar_orientation == 'horizontal':
                fig, (ax, cax) = plt.subplots(
                    nrows=2,
                    figsize=figsize,
                    gridspec_kw={"height_ratios": [1, 0.05]},
                    dpi=400,
                    constrained_layout=True,
                )
            elif cbar_orientation == 'vertical':
                fig, (ax, cax) = plt.subplots(
                    ncols=2,
                    figsize=figsize,
                    gridspec_kw={"width_ratios": [1, 0.05]},
                    dpi=400,
                    constrained_layout=True,
                )
            else:
                raise('This is not a valid orientation please choose either horizontal or vertical')
        else:
            fig = plt.figure(figsize=(figsize), dpi=400)
            ax = fig.add_subplot(111)
    else:
        fig = plt.figure(figsize=(figsize), dpi=400)
        ax = fig.add_subplot(111)

    _figure_setup(ax=ax, fontsize=fontsize, ylim=[erange[0], erange[1]])
    band.plot_atom_spd(
        ax=ax,
        atom_spd_dict=atom_spd_dict,
        display_order=display_order,
        scale_factor=scale_factor,
        color_list=color_list,
        legend=legend,
        linewidth=linewidth,
        band_color=band_color,
        erange=erange,
        heatmap=heatmap,
        bins=bins,
        sigma=sigma,
        cmap=cmap,
        vlinecolor=vlinecolor,
        powernorm=powernorm,
        gamma=gamma,
    )

    if heatmap:
        if cbar:
            im = ax.collections[0]
            min_val = im.norm.vmin
            max_val = im.norm.vmax
            cbar = fig.colorbar(im, cax=cax, orientation=cbar_orientation)
            cbar.set_ticks([min_val, max_val])
            cbar.set_ticklabels(['min', 'max'])

    if heatmap:
        if not cbar:
            fig.tight_layout(pad=0.4)
    else:
        fig.tight_layout(pad=0.4)

    if save:
        plt.savefig(output)
    else:
        return fig, ax


def band_elements(
    folder,
    elements,
    output='band_elements.png',
    spin='up',
    scale_factor=5,
    display_order=None,
    color_list=None,
    legend=True,
    linewidth=0.75,
    band_color='black',
    figsize=(4, 3),
    erange=[-6, 6],
    kpath=None,
    custom_kpath=None,
    n=None,
    unfold=False,
    M=None,
    high_symm_points=None,
    fontsize=12,
    heatmap=False,
    bins=800,
    sigma=2,
    cmap='hot',
    vlinecolor='black',
    cbar=True,
    cbar_orientation='horizontal',
    powernorm=True,
    gamma=0.5,
    save=True,
    shift_efermi=0,
    interpolate=False,
    new_n=200,
):
    """
    This function generates a projected band structure on specific elements.

    Parameters:
        folder (str): This is the folder that contains the VASP files
        elements (list): List of elements to project onto. The list should countain the corresponding element symbols
        output (str): File name of the resulting plot.
        spin (str): Choose which spin direction to parse. ('up' or 'down')
        scale_factor (float): Factor to scale weights. This changes the size of the
            points in the scatter plot.
        display_order (str / None): If None, the projections will be displayed in the same order
            the user inputs them. If 'all' the projections will be plotted from largest to smallest
            so every point is visable. If 'dominant' the projections will be plotted from smallest 
            to largest so only the dominant projection is shown.
        color_list (list): List of colors that is the same length as the number of projections
            in the plot.
        legend (bool): Determines if the legend should be included or not.
        linewidth (float): Line width of the plain band structure plotted in the background.
        band_color (string): Color of the plain band structure.
        figsize (list / tuple): Desired size of the image in inches. (width, height)
        erange (list / tuple): Range of energy to show in the plot. [low, high]
        kpath (list[list]): High symmetry k-point path of band structure calculation
            Due to the nature of the KPOINTS file for unfolded calculations this
            information is a required input for proper labeling of the figure
            for unfolded calculations. This information is extracted from the KPOINTS
            files for non-unfolded calculations. (G is automatically converted to \\Gamma)
	    (e.g. For the path X-G-X, kpath=[['X', 'G'], ['G', 'X']])
	custom_kpath (list): This gives the option to only plot specific segments of a given band structure
	    calculation. For example if the kpath was G-X-W-L then there are three segements to choose from:
	    G-X, X-W, and W-L. In this case the default kpath could be plotted by defining custom_kpath=[1,2,3],
	    where 1 -> G-X, 2 -> X-W, and 3 -> W-L. If only G-X and X-W were desired then custom_kpath=[1,2].
	    If one of the segements should be flipped it can be done by making its value negative
	    (e.g. -1 -> X-G, -2 -> W-X, -3 -> L-W)
        n (int): Number of points between each high symmetry points.
            This is also only required for unfolded calculations and band unfolding. This number should be 
            known by the user, as it was used to generate the KPOINTS file.
        unfold (bool): Determines if the plotted band structure is from a band unfolding calculation.
        M (list[list]): Transformation matrix from the primitive bulk structure to the slab structure.
            Only required for a band unfolding calculation.
        high_symm_points (list[list]): List of fractional coordinated for each high symmetry point in 
            the band structure path. Only required for a band unfolding calculation.
        fontsize (float): Font size of the text in the figure.
        save (bool): Determines whether to automatically save the figure or not. If not 
            the figure and axis are return for further manipulation.

    Returns:
        If save == True, this function will return nothing and directly save the image as
        the output name. If save == False, the function will return the matplotlib figure
        and axis for further editing.
    """

    band = Band(
        folder=folder,
        spin=spin,
        projected=True,
        unfold=unfold,
        high_symm_points=high_symm_points, 
	interpolate=interpolate,
	new_n=new_n,
        kpath=kpath,
	custom_kpath=custom_kpath,
        n=n,
        M=M,
	shift_efermi=shift_efermi,
    )

    if heatmap:
        legend = False
        if cbar:
            if cbar_orientation == 'horizontal':
                fig, (ax, cax) = plt.subplots(
                    nrows=2,
                    figsize=figsize,
                    gridspec_kw={"height_ratios": [1, 0.05]},
                    dpi=400,
                    constrained_layout=True,
                )
            elif cbar_orientation == 'vertical':
                fig, (ax, cax) = plt.subplots(
                    ncols=2,
                    figsize=figsize,
                    gridspec_kw={"width_ratios": [1, 0.05]},
                    dpi=400,
                    constrained_layout=True,
                )
            else:
                raise('This is not a valid orientation please choose either horizontal or vertical')
        else:
            fig = plt.figure(figsize=(figsize), dpi=400)
            ax = fig.add_subplot(111)
    else:
        fig = plt.figure(figsize=(figsize), dpi=400)
        ax = fig.add_subplot(111)

    _figure_setup(ax=ax, fontsize=fontsize, ylim=[erange[0], erange[1]])
    band.plot_elements(
        ax=ax,
        elements=elements,
        scale_factor=scale_factor,
        display_order=display_order,
        color_list=color_list,
        legend=legend,
        linewidth=linewidth,
        band_color=band_color,
        erange=erange,
        heatmap=heatmap,
        bins=bins,
        sigma=sigma,
        cmap=cmap,
        vlinecolor=vlinecolor,
        powernorm=powernorm,
        gamma=gamma,
    )

    if heatmap:
        if cbar:
            im = ax.collections[0]
            min_val = im.norm.vmin
            max_val = im.norm.vmax
            cbar = fig.colorbar(im, cax=cax, orientation=cbar_orientation)
            cbar.set_ticks([min_val, max_val])
            cbar.set_ticklabels(['min', 'max'])

    if heatmap:
        if not cbar:
            fig.tight_layout(pad=0.4)
    else:
        fig.tight_layout(pad=0.4)

    if save:
        plt.savefig(output)
    else:
        return fig, ax


def band_element_orbitals(
    folder,
    element_orbital_dict,
    output='band_element_orbital.png',
    spin='up',
    display_order=None,
    scale_factor=5,
    color_list=None,
    legend=True,
    linewidth=0.75,
    band_color='black',
    figsize=(4, 3),
    erange=[-6, 6],
    kpath=None,
    custom_kpath=None,
    n=None,
    unfold=False,
    M=None,
    high_symm_points=None,
    fontsize=12,
    heatmap=False,
    bins=800,
    sigma=2,
    cmap='hot',
    vlinecolor='black',
    cbar=True,
    cbar_orientation='horizontal',
    powernorm=True,
    gamma=0.5,
    save=True,
    shift_efermi=0,
    interpolate=False,
    new_n=200,
):
    """
    This function generates a projected band structure on orbitals of specific elements.

    Parameters:
        folder (str): This is the folder that contains the VASP files
        element_orbital_dict (dict[str:list]): A dictionary that contains the individual elements and the corresponding 
            orbitals to project onto. For example, if the user wants to project onto the s, py, pz, and px orbitals
            of In and the s orbital of As for and InAs structure then the dictionary would be {'In':[0,1,2,3], 'As':[0]}
        output (str): File name of the resulting plot.
        spin (str): Choose which spin direction to parse. ('up' or 'down')
        scale_factor (float): Factor to scale weights. This changes the size of the
            points in the scatter plot.
        display_order (str / None): If None, the projections will be displayed in the same order
            the user inputs them. If 'all' the projections will be plotted from largest to smallest
            so every point is visable. If 'dominant' the projections will be plotted from smallest 
            to largest so only the dominant projection is shown.
        color_list (list): List of colors that is the same length as the number of projections
            in the plot.
        legend (bool): Determines if the legend should be included or not.
        linewidth (float): Line width of the plain band structure plotted in the background.
        band_color (string): Color of the plain band structure.
        figsize (list / tuple): Desired size of the image in inches. (width, height)
        erange (list / tuple): Range of energy to show in the plot. [low, high]
        kpath (list[list]): High symmetry k-point path of band structure calculation
            Due to the nature of the KPOINTS file for unfolded calculations this
            information is a required input for proper labeling of the figure
            for unfolded calculations. This information is extracted from the KPOINTS
            files for non-unfolded calculations. (G is automatically converted to \\Gamma)
	    (e.g. For the path X-G-X, kpath=[['X', 'G'], ['G', 'X']])
	custom_kpath (list): This gives the option to only plot specific segments of a given band structure
	    calculation. For example if the kpath was G-X-W-L then there are three segements to choose from:
	    G-X, X-W, and W-L. In this case the default kpath could be plotted by defining custom_kpath=[1,2,3],
	    where 1 -> G-X, 2 -> X-W, and 3 -> W-L. If only G-X and X-W were desired then custom_kpath=[1,2].
	    If one of the segements should be flipped it can be done by making its value negative
	    (e.g. -1 -> X-G, -2 -> W-X, -3 -> L-W)
        n (int): Number of points between each high symmetry points.
            This is also only required for unfolded calculations and band unfolding. This number should be 
            known by the user, as it was used to generate the KPOINTS file.
        unfold (bool): Determines if the plotted band structure is from a band unfolding calculation.
        M (list[list]): Transformation matrix from the primitive bulk structure to the slab structure.
            Only required for a band unfolding calculation.
        high_symm_points (list[list]): List of fractional coordinated for each high symmetry point in 
            the band structure path. Only required for a band unfolding calculation.
        fontsize (float): Font size of the text in the figure.
        save (bool): Determines whether to automatically save the figure or not. If not 
            the figure and axis are return for further manipulation.

    Returns:
        If save == True, this function will return nothing and directly save the image as
        the output name. If save == False, the function will return the matplotlib figure
        and axis for further editing.
    """

    band = Band(
        folder=folder,
        spin=spin,
        projected=True,
        unfold=unfold,
        high_symm_points=high_symm_points, 
	interpolate=interpolate,
	new_n=new_n,
        kpath=kpath,
	custom_kpath=custom_kpath,
        n=n,
        M=M,
	shift_efermi=shift_efermi,
    )

    if heatmap:
        legend = False
        if cbar:
            if cbar_orientation == 'horizontal':
                fig, (ax, cax) = plt.subplots(
                    nrows=2,
                    figsize=figsize,
                    gridspec_kw={"height_ratios": [1, 0.05]},
                    dpi=400,
                    constrained_layout=True,
                )
            elif cbar_orientation == 'vertical':
                fig, (ax, cax) = plt.subplots(
                    ncols=2,
                    figsize=figsize,
                    gridspec_kw={"width_ratios": [1, 0.05]},
                    dpi=400,
                    constrained_layout=True,
                )
            else:
                raise('This is not a valid orientation please choose either horizontal or vertical')
        else:
            fig = plt.figure(figsize=(figsize), dpi=400)
            ax = fig.add_subplot(111)
    else:
        fig = plt.figure(figsize=(figsize), dpi=400)
        ax = fig.add_subplot(111)

    _figure_setup(ax=ax, fontsize=fontsize, ylim=[erange[0], erange[1]])
    band.plot_element_orbitals(
        ax=ax,
        element_orbital_dict=element_orbital_dict,
        scale_factor=scale_factor,
        display_order=display_order,
        color_list=color_list,
        legend=legend,
        linewidth=linewidth,
        band_color=band_color,
        erange=erange,
        heatmap=heatmap,
        bins=bins,
        sigma=sigma,
        cmap=cmap,
        vlinecolor=vlinecolor,
        powernorm=powernorm,
        gamma=gamma,
    )

    if heatmap:
        if cbar:
            im = ax.collections[0]
            min_val = im.norm.vmin
            max_val = im.norm.vmax
            cbar = fig.colorbar(im, cax=cax, orientation=cbar_orientation)
            cbar.set_ticks([min_val, max_val])
            cbar.set_ticklabels(['min', 'max'])

    if heatmap:
        if not cbar:
            fig.tight_layout(pad=0.4)
    else:
        fig.tight_layout(pad=0.4)

    if save:
        plt.savefig(output)
    else:
        return fig, ax


def band_element_spd(
    folder,
    element_spd_dict,
    output='band_element_spd.png',
    spin='up',
    display_order=None,
    scale_factor=5,
    color_list=None,
    legend=True,
    linewidth=0.75,
    band_color='black',
    figsize=(4, 3),
    erange=[-6, 6],
    kpath=None,
    custom_kpath=None,
    n=None,
    unfold=False,
    M=None,
    high_symm_points=None,
    fontsize=12,
    heatmap=False,
    bins=800,
    sigma=2,
    cmap='hot',
    vlinecolor='black',
    cbar=True,
    cbar_orientation='horizontal',
    powernorm=True,
    gamma=0.5,
    save=True,
    shift_efermi=0,
    interpolate=False,
    new_n=200,
):
    """
    This function generates a s, p, d projected band structure on specific elements.

    Parameters:
        folder (str): This is the folder that contains the VASP files
        element_spd_dict (dict[str:str]): A dictionary that contains the individual atoms and the corresponding 
            orbitals to project onto. For example, if the user wants to project onto the s, p, d orbitals
            of In and the p orbitals of As for an InAs structure then the dictionary would be {'In':'spd', 'As':'p'}
        output (str): File name of the resulting plot.
        spin (str): Choose which spin direction to parse. ('up' or 'down')
        scale_factor (float): Factor to scale weights. This changes the size of the
            points in the scatter plot.
        display_order (str / None): If None, the projections will be displayed in the same order
            the user inputs them. If 'all' the projections will be plotted from largest to smallest
            so every point is visable. If 'dominant' the projections will be plotted from smallest 
            to largest so only the dominant projection is shown.
        color_list (list): List of colors that is the same length as the number of projections
            in the plot.
        legend (bool): Determines if the legend should be included or not.
        linewidth (float): Line width of the plain band structure plotted in the background.
        band_color (string): Color of the plain band structure.
        figsize (list / tuple): Desired size of the image in inches. (width, height)
        erange (list / tuple): Range of energy to show in the plot. [low, high]
        kpath (list[list]): High symmetry k-point path of band structure calculation
            Due to the nature of the KPOINTS file for unfolded calculations this
            information is a required input for proper labeling of the figure
            for unfolded calculations. This information is extracted from the KPOINTS
            files for non-unfolded calculations. (G is automatically converted to \\Gamma)
	    (e.g. For the path X-G-X, kpath=[['X', 'G'], ['G', 'X']])
	custom_kpath (list): This gives the option to only plot specific segments of a given band structure
	    calculation. For example if the kpath was G-X-W-L then there are three segements to choose from:
	    G-X, X-W, and W-L. In this case the default kpath could be plotted by defining custom_kpath=[1,2,3],
	    where 1 -> G-X, 2 -> X-W, and 3 -> W-L. If only G-X and X-W were desired then custom_kpath=[1,2].
	    If one of the segements should be flipped it can be done by making its value negative
	    (e.g. -1 -> X-G, -2 -> W-X, -3 -> L-W)
        n (int): Number of points between each high symmetry points.
            This is also only required for unfolded calculations and band unfolding. This number should be 
            known by the user, as it was used to generate the KPOINTS file.
        unfold (bool): Determines if the plotted band structure is from a band unfolding calculation.
        M (list[list]): Transformation matrix from the primitive bulk structure to the slab structure.
            Only required for a band unfolding calculation.
        high_symm_points (list[list]): List of fractional coordinated for each high symmetry point in 
            the band structure path. Only required for a band unfolding calculation.
        fontsize (float): Font size of the text in the figure.
        save (bool): Determines whether to automatically save the figure or not. If not 
            the figure and axis are return for further manipulation.

    Returns:
        If save == True, this function will return nothing and directly save the image as
        the output name. If save == False, the function will return the matplotlib figure
        and axis for further editing.
    """

    band = Band(
        folder=folder,
        spin=spin,
        projected=True,
        unfold=unfold,
        high_symm_points=high_symm_points, 
	interpolate=interpolate,
	new_n=new_n,
        kpath=kpath,
	custom_kpath=custom_kpath,
        n=n,
        M=M,
	shift_efermi=shift_efermi,
    )

    if heatmap:
        legend = False
        if cbar:
            if cbar_orientation == 'horizontal':
                fig, (ax, cax) = plt.subplots(
                    nrows=2,
                    figsize=figsize,
                    gridspec_kw={"height_ratios": [1, 0.05]},
                    dpi=400,
                    constrained_layout=True,
                )
            elif cbar_orientation == 'vertical':
                fig, (ax, cax) = plt.subplots(
                    ncols=2,
                    figsize=figsize,
                    gridspec_kw={"width_ratios": [1, 0.05]},
                    dpi=400,
                    constrained_layout=True,
                )
            else:
                raise('This is not a valid orientation please choose either horizontal or vertical')
        else:
            fig = plt.figure(figsize=(figsize), dpi=400)
            ax = fig.add_subplot(111)
    else:
        fig = plt.figure(figsize=(figsize), dpi=400)
        ax = fig.add_subplot(111)

    _figure_setup(ax=ax, fontsize=fontsize, ylim=[erange[0], erange[1]])
    band.plot_element_spd(
        ax=ax,
        element_spd_dict=element_spd_dict,
        display_order=display_order,
        scale_factor=scale_factor,
        color_list=color_list,
        legend=legend,
        linewidth=linewidth,
        band_color=band_color,
        erange=erange,
        heatmap=heatmap,
        bins=bins,
        sigma=sigma,
        cmap=cmap,
        vlinecolor=vlinecolor,
        powernorm=powernorm,
        gamma=gamma,
    )

    if heatmap:
        if cbar:
            im = ax.collections[0]
            min_val = im.norm.vmin
            max_val = im.norm.vmax
            cbar = fig.colorbar(im, cax=cax, orientation=cbar_orientation)
            cbar.set_ticks([min_val, max_val])
            cbar.set_ticklabels(['min', 'max'])

    if heatmap:
        if not cbar:
            fig.tight_layout(pad=0.4)
    else:
        fig.tight_layout(pad=0.4)

    if save:
        plt.savefig(output)
    else:
        return fig, ax


def band_plain_spin_polarized(
    folder,
    output='band_plain_sp.png',
    up_color='red',
    down_color='blue',
    background_color='black',
    linewidth=1.25,
    up_linestyle='-',
    down_linestyle=':',
    figsize=(4, 3),
    erange=[-6, 6],
    kpath=None,
    custom_kpath=None,
    n=None,
    unfold=False,
    M=None,
    high_symm_points=None,
    fontsize=12,
    save=True,
    scale_factor=5,
    shift_efermi=0,
    interpolate=False,
    new_n=200,
    heatmap=False,
    bins=1000,
    sigma=2,
    vlinecolor='black',
    cbar=True,
    cbar_orientation='horizontal',
    powernorm=True,
    gamma=0.7,
):
    """
    This function generates a plain spin polarized band structure.

    Parameters:
        folder (str): This is the folder that contains the VASP files
        output (str): File name of the resulting plot.
        color_list (list): List of colors that is the same length as the number of projections
            in the plot.
        legend (bool): Determines if the legend should be included or not.
        unprojected_band_color (str): Color of the unprojected band
        unprojected_linewidth (float): Line width of the unprojected bands
        annotations (list): Annotations to put on the top and bottom (left and right) figures.
            By default it will show the spin up and spin down arrows.
        annotation_xy (list / tuple): Fractional (x, y) coordinated of the annotation location
        stack (str): Determines how the plots are stacked (vertical or horizontal)
        linewidth (float): Line width of the plain band structure plotted in the background.
        band_color (string): Color of the plain band structure.
        figsize (list / tuple): Desired size of the image in inches. (width, height)
        erange (list / tuple): Range of energy to show in the plot. [low, high]
        kpath (list[list]): High symmetry k-point path of band structure calculation
            Due to the nature of the KPOINTS file for unfolded calculations this
            information is a required input for proper labeling of the figure
            for unfolded calculations. This information is extracted from the KPOINTS
            files for non-unfolded calculations. (G is automatically converted to \\Gamma)
	    (e.g. For the path X-G-X, kpath=[['X', 'G'], ['G', 'X']])
	custom_kpath (list): This gives the option to only plot specific segments of a given band structure
	    calculation. For example if the kpath was G-X-W-L then there are three segements to choose from:
	    G-X, X-W, and W-L. In this case the default kpath could be plotted by defining custom_kpath=[1,2,3],
	    where 1 -> G-X, 2 -> X-W, and 3 -> W-L. If only G-X and X-W were desired then custom_kpath=[1,2].
	    If one of the segements should be flipped it can be done by making its value negative
	    (e.g. -1 -> X-G, -2 -> W-X, -3 -> L-W)
        n (int): Number of points between each high symmetry points.
            This is also only required for unfolded calculations and band unfolding. This number should be 
            known by the user, as it was used to generate the KPOINTS file.
        unfold (bool): Determines if the plotted band structure is from a band unfolding calculation.
        M (list[list]): Transformation matrix from the primitive bulk structure to the slab structure.
            Only required for a band unfolding calculation.
        high_symm_points (list[list]): List of fractional coordinated for each high symmetry point in 
            the band structure path. Only required for a band unfolding calculation.
        fontsize (float): Font size of the text in the figure.
        save (bool): Determines whether to automatically save the figure or not. If not 
            the figure and axis are return for further manipulation.

    Returns:
        If save == True, this function will return nothing and directly save the image as
        the output name. If save == False, the function will return the matplotlib figure
        and axis for further editing. (fig, ax1, ax2) 
    """

    band_up = Band(
        folder=folder,
        spin='up',
        unfold=unfold,
        high_symm_points=high_symm_points, 
	interpolate=interpolate,
	new_n=new_n,
        kpath=kpath,
	custom_kpath=custom_kpath,
        n=n,
        M=M,
	shift_efermi=shift_efermi,
    )
    band_down = Band(
        folder=folder,
        spin='down',
        unfold=unfold,
        high_symm_points=high_symm_points, 
	interpolate=interpolate,
	new_n=new_n,
        kpath=kpath,
	custom_kpath=custom_kpath,
        n=n,
        M=M,
	shift_efermi=shift_efermi,
    )

    if heatmap:
        if cbar:
            if cbar_orientation == 'horizontal':
                fig, (ax, cax) = plt.subplots(
                    nrows=2,
                    figsize=figsize,
                    gridspec_kw={"height_ratios": [1, 0.05]},
                    dpi=400,
                    constrained_layout=True,
                )
            elif cbar_orientation == 'vertical':
                fig, (ax, cax) = plt.subplots(
                    ncols=2,
                    figsize=figsize,
                    gridspec_kw={"width_ratios": [1, 0.05]},
                    dpi=400,
                    constrained_layout=True,
                )
            else:
                raise('This is not a valid orientation please choose either horizontal or vertical')
        else:
            fig = plt.figure(figsize=(figsize), dpi=400)
            ax = fig.add_subplot(111)
    else:
        fig = plt.figure(figsize=(figsize), dpi=400)
        ax = fig.add_subplot(111)

    _figure_setup(ax=ax, fontsize=fontsize, ylim=[erange[0], erange[1]])

    if heatmap:
        _, axd = plt.subplots()
        spin_cmap = LinearSegmentedColormap.from_list(
            'spin',
            [
                to_rgba(down_color),
                to_rgba(down_color),
                to_rgba(down_color),
                (1,1,1,0),
                to_rgba(up_color),
                to_rgba(up_color),
                to_rgba(up_color),
            ],
            N=10000
        )

        bot_cmap = LinearSegmentedColormap.from_list(
            'background',
            [
                (1,1,1,0),
                to_rgba(background_color),
            ],
            N=10000
        )
        band_up.plot_plain(
            ax=ax,
            erange=erange,
            heatmap=heatmap,
            bins=bins,
            sigma=sigma,
            vlinecolor=vlinecolor,
            powernorm=False,
            cmap=bot_cmap,
            gamma=gamma,
        )

        band_down.plot_plain(
            ax=axd,
            erange=erange,
            heatmap=heatmap,
            bins=bins,
            sigma=sigma,
            vlinecolor=vlinecolor,
            powernorm=False,
            cmap=bot_cmap,
            gamma=gamma,
        )

        band_up.plot_plain(
            ax=ax,
            erange=erange,
            heatmap=heatmap,
            bins=bins,
            sigma=sigma,
            vlinecolor=vlinecolor,
            powernorm=False,
            cmap=spin_cmap,
            gamma=gamma,
        )

        band_down.plot_plain(
            ax=axd,
            erange=erange,
            heatmap=heatmap,
            bins=bins,
            sigma=sigma,
            vlinecolor=vlinecolor,
            cmap=spin_cmap,
            powernorm=False,
            gamma=gamma,
        )

        ax.collections[1].set_array(ax.collections[1].get_array() - axd.collections[1].get_array())
        ax.collections[0].set_array(ax.collections[0].get_array() + axd.collections[0].get_array())
        im = ax.collections[1]
        sym_val = np.max(np.abs([np.min(im.get_array()), np.max(im.get_array())]))
        norm = Normalize(
            vmin=-sym_val,
            vmax=sym_val,
        )
        ax.collections[1].set_norm(norm)
    else:
        band_up.plot_plain(
            ax=ax,
            color=up_color,
            linewidth=linewidth,
            linestyle=up_linestyle,
            erange=erange,
            scale_factor=scale_factor,
            heatmap=heatmap,
            bins=bins,
            sigma=sigma,
            vlinecolor=vlinecolor,
            powernorm=powernorm,
            gamma=gamma,
        )

        band_down.plot_plain(
            ax=ax,
            color=down_color,
            linewidth=linewidth,
            linestyle=down_linestyle,
            erange=erange,
            scale_factor=scale_factor,
            heatmap=heatmap,
            bins=bins,
            sigma=sigma,
            vlinecolor=vlinecolor,
            powernorm=powernorm,
            gamma=gamma,
        )

        if unfold:
            legend_lines = [
                plt.Line2D(
                    [0],
                    [0],
                    color=up_color,
                    linestyle='',
                    marker='o',
                    markersize=3,
                ),
                plt.Line2D(
                    [0],
                    [0],
                    color=down_color,
                    linestyle='',
                    marker='o',
                    markersize=3,
                )
            ]
        else:
            legend_lines = [
                plt.Line2D(
                    [0],
                    [0],
                    color=up_color,
                    linestyle=up_linestyle
                ),
                plt.Line2D(
                    [0],
                    [0],
                    color=down_color,
                    linestyle=down_linestyle
                )
            ]

        legend_labels = ['$\\uparrow$', '$\\downarrow$']

        ax.legend(
            legend_lines,
            legend_labels,
            ncol=1,
            loc='upper left',
            fontsize=fontsize,
            bbox_to_anchor=(1, 1),
            borderaxespad=0,
            frameon=False,
            handletextpad=0.1,
        )

    if heatmap:
        if cbar:
            im = ax.collections[1]
            min_val = im.norm.vmin
            max_val = im.norm.vmax
            cbar = fig.colorbar(im, cax=cax, orientation=cbar_orientation)
            cbar.set_ticks([min_val, max_val])
            cbar.set_ticklabels(['down', 'up'])

    if heatmap:
        if not cbar:
            fig.tight_layout(pad=0.4)
    else:
        fig.tight_layout(pad=0.4)

    if save:
        fig.savefig(output)
    else:
        return fig, ax


def band_spd_spin_polarized(
    folder,
    output='band_spd_sp.png',
    scale_factor=5,
    orbitals='spd',
    display_order=None,
    color_list=None,
    legend=True,
    linewidth=0.75,
    band_color='black',
    unprojected_band_color='gray',
    unprojected_linewidth=0.6,
    fontsize=12,
    annotations=['$\\uparrow$ ', '$\\downarrow$ '],
    annotation_xy=(0.02, 0.98),
    figsize=(4, 3),
    erange=[-6, 6],
    stack='vertical',
    kpath=None,
    custom_kpath=None,
    n=None,
    unfold=False,
    M=None,
    high_symm_points=None,
    save=True,
    shift_efermi=0,
    interpolate=False,
    new_n=200,
):
    """
    This function generates a spin polarized s, p, d projected band structure. This will plot two plots
    stacked on top or eachother or next to eachother. The top or left plot will project on the 
    spin up bands and the bottom or right plot will project onto the spin down bands.

    Parameters:
        folder (str): This is the folder that contains the VASP files
        orbitals (str): String that contains the s, p, or d orbitals that to project onto.
            The default is 'spd', if the user only wanted to project onto the p, and d orbitals 
            than 'pd' should be passed in
        output (str): File name of the resulting plot.
        scale_factor (float): Factor to scale weights. This changes the size of the
            points in the scatter plot.
        display_order (str / None): If None, the projections will be displayed in the same order
            the user inputs them. If 'all' the projections will be plotted from largest to smallest
            so every point is visable. If 'dominant' the projections will be plotted from smallest 
            to largest so only the dominant projection is shown.
        color_list (list): List of colors that is the same length as the number of projections
            in the plot.
        legend (bool): Determines if the legend should be included or not.
        unprojected_band_color (str): Color of the unprojected band
        unprojected_linewidth (float): Line width of the unprojected bands
        annotations (list): Annotations to put on the top and bottom (left and right) figures.
            By default it will show the spin up and spin down arrows.
        annotation_xy (list / tuple): Fractional (x, y) coordinated of the annotation location
        stack (str): Determines how the plots are stacked (vertical or horizontal)
        linewidth (float): Line width of the plain band structure plotted in the background.
        band_color (string): Color of the plain band structure.
        figsize (list / tuple): Desired size of the image in inches. (width, height)
        erange (list / tuple): Range of energy to show in the plot. [low, high]
        kpath (list[list]): High symmetry k-point path of band structure calculation
            Due to the nature of the KPOINTS file for unfolded calculations this
            information is a required input for proper labeling of the figure
            for unfolded calculations. This information is extracted from the KPOINTS
            files for non-unfolded calculations. (G is automatically converted to \\Gamma)
	    (e.g. For the path X-G-X, kpath=[['X', 'G'], ['G', 'X']])
	custom_kpath (list): This gives the option to only plot specific segments of a given band structure
	    calculation. For example if the kpath was G-X-W-L then there are three segements to choose from:
	    G-X, X-W, and W-L. In this case the default kpath could be plotted by defining custom_kpath=[1,2,3],
	    where 1 -> G-X, 2 -> X-W, and 3 -> W-L. If only G-X and X-W were desired then custom_kpath=[1,2].
	    If one of the segements should be flipped it can be done by making its value negative
	    (e.g. -1 -> X-G, -2 -> W-X, -3 -> L-W)
        n (int): Number of points between each high symmetry points.
            This is also only required for unfolded calculations and band unfolding. This number should be 
            known by the user, as it was used to generate the KPOINTS file.
        unfold (bool): Determines if the plotted band structure is from a band unfolding calculation.
        M (list[list]): Transformation matrix from the primitive bulk structure to the slab structure.
            Only required for a band unfolding calculation.
        high_symm_points (list[list]): List of fractional coordinated for each high symmetry point in 
            the band structure path. Only required for a band unfolding calculation.
        fontsize (float): Font size of the text in the figure.
        save (bool): Determines whether to automatically save the figure or not. If not 
            the figure and axis are return for further manipulation.

    Returns:
        If save == True, this function will return nothing and directly save the image as
        the output name. If save == False, the function will return the matplotlib figure
        and axis for further editing. (fig, ax1, ax2) 
    """

    band_up = Band(
        folder=folder,
        spin='up',
        projected=True,
        unfold=unfold,
        high_symm_points=high_symm_points, 
	interpolate=interpolate,
	new_n=new_n,
        kpath=kpath,
	custom_kpath=custom_kpath,
        n=n,
        M=M,
	shift_efermi=shift_efermi,
    )

    band_down = Band(
        folder=folder,
        spin='down',
        projected=True,
        unfold=unfold,
        high_symm_points=high_symm_points, 
	interpolate=interpolate,
	new_n=new_n,
        kpath=kpath,
	custom_kpath=custom_kpath,
        n=n,
        M=M,
	shift_efermi=shift_efermi,
    )

    if stack == 'vertical':
        fig = plt.figure(figsize=(figsize[0], 2 * figsize[1]), dpi=400)
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        _figure_setup(ax=ax1, fontsize=fontsize, ylim=[erange[0], erange[1]])
        _figure_setup(ax=ax2, fontsize=fontsize, ylim=[erange[0], erange[1]])
    elif stack == 'horizontal':
        fig = plt.figure(figsize=(2 * figsize[0], figsize[1]), dpi=400)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        _figure_setup(ax=ax1, fontsize=fontsize, ylim=[erange[0], erange[1]])
        _figure_setup(ax=ax2, fontsize=fontsize, ylim=[erange[0], erange[1]])

    bbox = dict(boxstyle='round', fc='white',
                edgecolor='gray', alpha=0.95, pad=0.3)
    ax1.annotate(
        annotations[0],
        xy=annotation_xy,
        xycoords='axes fraction',
        va='top',
        ha='left',
        bbox=bbox,
        fontsize=fontsize,
    )
    ax2.annotate(
        annotations[1],
        xy=annotation_xy,
        xycoords='axes fraction',
        va='top',
        ha='left',
        bbox=bbox,
        fontsize=fontsize,
    )

    band_up.plot_spd(
        ax=ax1,
        scale_factor=scale_factor,
        orbitals=orbitals,
        erange=erange,
        display_order=display_order,
        color_list=color_list,
        legend=legend,
        linewidth=linewidth,
        band_color=band_color,
    )
    band_down.plot_plain(
        ax=ax1,
        color=unprojected_band_color,
        linewidth=unprojected_linewidth,
        erange=erange,
    )

    band_down.plot_spd(
        ax=ax2,
        scale_factor=scale_factor,
        orbitals=orbitals,
        erange=erange,
        display_order=display_order,
        color_list=color_list,
        legend=legend,
        linewidth=linewidth,
        band_color=band_color,
    )
    band_up.plot_plain(
        ax=ax2,
        color=unprojected_band_color,
        linewidth=unprojected_linewidth,
        erange=erange,
    )

    plt.tight_layout(pad=0.4)

    if save:
        plt.savefig(output)
    else:
        return fig, ax1, ax2


def band_atom_orbitals_spin_polarized(
    folder,
    atom_orbital_dict,
    output='band_atom_orbitals_sp.png',
    display_order=None,
    scale_factor=5,
    color_list=None,
    legend=True,
    linewidth=0.75,
    band_color='black',
    unprojected_band_color='gray',
    unprojected_linewidth=0.6,
    fontsize=12,
    annotations=['$\\uparrow$ ', '$\\downarrow$ '],
    annotation_xy=(0.02, 0.98),
    figsize=(4, 3),
    erange=[-6, 6],
    stack='vertical',
    kpath=None,
    custom_kpath=None,
    n=None,
    unfold=False,
    M=None,
    high_symm_points=None,
    save=True,
    shift_efermi=0,
    interpolate=False,
    new_n=200,
):
    """
    This function generates an atom orbital spin polarized band structure. This will plot two plots
    stacked on top or eachother or next to eachother. The top or left plot will project on the 
    spin up bands and the bottom or right plot will project onto the spin down bands.

    Parameters:
        folder (str): This is the folder that contains the VASP files
        atom_orbital_dict (dict[int:list]): A dictionary that contains the individual atoms and the corresponding 
            orbitals to project onto. For example, if the user wants to project onto the s, py, pz, and px orbitals
            of the first atom and the s orbital of the second atom then the dictionary would be {0:[0,1,2,3], 1:[0]}
        output (str): File name of the resulting plot.
        scale_factor (float): Factor to scale weights. This changes the size of the
            points in the scatter plot.
        display_order (str / None): If None, the projections will be displayed in the same order
            the user inputs them. If 'all' the projections will be plotted from largest to smallest
            so every point is visable. If 'dominant' the projections will be plotted from smallest 
            to largest so only the dominant projection is shown.
        color_list (list): List of colors that is the same length as the number of projections
            in the plot.
        legend (bool): Determines if the legend should be included or not.
        unprojected_band_color (str): Color of the unprojected band
        unprojected_linewidth (float): Line width of the unprojected bands
        annotations (list): Annotations to put on the top and bottom (left and right) figures.
            By default it will show the spin up and spin down arrows.
        annotation_xy (list / tuple): Fractional (x, y) coordinated of the annotation location
        stack (str): Determines how the plots are stacked (vertical or horizontal)
        linewidth (float): Line width of the plain band structure plotted in the background.
        band_color (string): Color of the plain band structure.
        figsize (list / tuple): Desired size of the image in inches. (width, height)
        erange (list / tuple): Range of energy to show in the plot. [low, high]
        kpath (list[list]): High symmetry k-point path of band structure calculation
            Due to the nature of the KPOINTS file for unfolded calculations this
            information is a required input for proper labeling of the figure
            for unfolded calculations. This information is extracted from the KPOINTS
            files for non-unfolded calculations. (G is automatically converted to \\Gamma)
	    (e.g. For the path X-G-X, kpath=[['X', 'G'], ['G', 'X']])
	custom_kpath (list): This gives the option to only plot specific segments of a given band structure
	    calculation. For example if the kpath was G-X-W-L then there are three segements to choose from:
	    G-X, X-W, and W-L. In this case the default kpath could be plotted by defining custom_kpath=[1,2,3],
	    where 1 -> G-X, 2 -> X-W, and 3 -> W-L. If only G-X and X-W were desired then custom_kpath=[1,2].
	    If one of the segements should be flipped it can be done by making its value negative
	    (e.g. -1 -> X-G, -2 -> W-X, -3 -> L-W)
        n (int): Number of points between each high symmetry points.
            This is also only required for unfolded calculations and band unfolding. This number should be 
            known by the user, as it was used to generate the KPOINTS file.
        unfold (bool): Determines if the plotted band structure is from a band unfolding calculation.
        M (list[list]): Transformation matrix from the primitive bulk structure to the slab structure.
            Only required for a band unfolding calculation.
        high_symm_points (list[list]): List of fractional coordinated for each high symmetry point in 
            the band structure path. Only required for a band unfolding calculation.
        fontsize (float): Font size of the text in the figure.
        save (bool): Determines whether to automatically save the figure or not. If not 
            the figure and axis are return for further manipulation.

    Returns:
        If save == True, this function will return nothing and directly save the image as
        the output name. If save == False, the function will return the matplotlib figure
        and axis for further editing. (fig, ax1, ax2) 
    """

    band_up = Band(
        folder=folder,
        spin='up',
        projected=True,
        unfold=unfold,
        high_symm_points=high_symm_points, 
	interpolate=interpolate,
	new_n=new_n,
        kpath=kpath,
	custom_kpath=custom_kpath,
        n=n,
        M=M,
	shift_efermi=shift_efermi,
    )

    band_down = Band(
        folder=folder,
        spin='down',
        projected=True,
        unfold=unfold,
        high_symm_points=high_symm_points, 
	interpolate=interpolate,
	new_n=new_n,
        kpath=kpath,
	custom_kpath=custom_kpath,
        n=n,
        M=M,
	shift_efermi=shift_efermi,
    )

    if stack == 'vertical':
        fig = plt.figure(figsize=(figsize[0], 2 * figsize[1]), dpi=400)
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        _figure_setup(ax=ax1, fontsize=fontsize, ylim=[erange[0], erange[1]])
        _figure_setup(ax=ax2, fontsize=fontsize, ylim=[erange[0], erange[1]])
    elif stack == 'horizontal':
        fig = plt.figure(figsize=(2 * figsize[0], figsize[1]), dpi=400)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        _figure_setup(ax=ax1, fontsize=fontsize, ylim=[erange[0], erange[1]])
        _figure_setup(ax=ax2, fontsize=fontsize, ylim=[erange[0], erange[1]])

    bbox = dict(boxstyle='round', fc='white',
                edgecolor='gray', alpha=0.95, pad=0.3)
    ax1.annotate(
        annotations[0],
        xy=annotation_xy,
        xycoords='axes fraction',
        va='top',
        ha='left',
        bbox=bbox,
        fontsize=fontsize,
    )
    ax2.annotate(
        annotations[1],
        xy=annotation_xy,
        xycoords='axes fraction',
        va='top',
        ha='left',
        bbox=bbox,
        fontsize=fontsize,
    )

    band_up.plot_atom_orbitals(
        ax=ax1,
        atom_orbital_dict=atom_orbital_dict,
        display_order=display_order,
        scale_factor=scale_factor,
        color_list=color_list,
        legend=legend,
        linewidth=linewidth,
        band_color=band_color,
        erange=erange,
    )
    band_down.plot_plain(
        ax=ax1,
        color=unprojected_band_color,
        linewidth=unprojected_linewidth,
        erange=erange,
    )

    band_down.plot_atom_orbitals(
        ax=ax2,
        atom_orbital_dict=atom_orbital_dict,
        display_order=display_order,
        scale_factor=scale_factor,
        color_list=color_list,
        legend=legend,
        linewidth=linewidth,
        band_color=band_color,
        erange=erange,
    )
    band_up.plot_plain(
        ax=ax2,
        color=unprojected_band_color,
        linewidth=unprojected_linewidth,
        erange=erange,
    )

    plt.tight_layout(pad=0.4)

    if save:
        plt.savefig(output)
    else:
        return fig, ax1, ax2


def band_orbitals_spin_polarized(
    folder,
    orbitals,
    output='band_orbitals_sp.png',
    scale_factor=5,
    display_order=None,
    color_list=None,
    legend=True,
    linewidth=0.75,
    band_color='black',
    unprojected_band_color='gray',
    figsize=(4, 3),
    erange=[-6, 6],
    stack='vertical',
    kpath=None,
    custom_kpath=None,
    n=None,
    unfold=False,
    M=None,
    high_symm_points=None,
    fontsize=12,
    save=True,
    shift_efermi=0,
    interpolate=False,
    new_n=200,
):
    """
    This function generates an orbital spin polarized band structure. This will plot two plots
    stacked on top or eachother or next to eachother. The top or left plot will project on the 
    spin up bands and the bottom or right plot will project onto the spin down bands.

    Parameters:
        folder (str): This is the folder that contains the VASP files
        orbitals (list): List of orbitals to compare

            | 0 = s
            | 1 = py
            | 2 = pz
            | 3 = px
            | 4 = dxy
            | 5 = dyz
            | 6 = dz2
            | 7 = dxz
            | 8 = dx2-y2
            | 9 = fy3x2
            | 10 = fxyz
            | 11 = fyz2
            | 12 = fz3
            | 13 = fxz2
            | 14 = fzx3
            | 15 = fx3

        output (str): File name of the resulting plot.
        scale_factor (float): Factor to scale weights. This changes the size of the
            points in the scatter plot.
        display_order (str / None): If None, the projections will be displayed in the same order
            the user inputs them. If 'all' the projections will be plotted from largest to smallest
            so every point is visable. If 'dominant' the projections will be plotted from smallest 
            to largest so only the dominant projection is shown.
        color_list (list): List of colors that is the same length as the number of projections
            in the plot.
        legend (bool): Determines if the legend should be included or not.
        unprojected_band_color (str): Color of the unprojected band
        unprojected_linewidth (float): Line width of the unprojected bands
        annotations (list): Annotations to put on the top and bottom (left and right) figures.
            By default it will show the spin up and spin down arrows.
        annotation_xy (list / tuple): Fractional (x, y) coordinated of the annotation location
        stack (str): Determines how the plots are stacked (vertical or horizontal)
        linewidth (float): Line width of the plain band structure plotted in the background.
        band_color (string): Color of the plain band structure.
        figsize (list / tuple): Desired size of the image in inches. (width, height)
        erange (list / tuple): Range of energy to show in the plot. [low, high]
        kpath (list[list]): High symmetry k-point path of band structure calculation
            Due to the nature of the KPOINTS file for unfolded calculations this
            information is a required input for proper labeling of the figure
            for unfolded calculations. This information is extracted from the KPOINTS
            files for non-unfolded calculations. (G is automatically converted to \\Gamma)
	    (e.g. For the path X-G-X, kpath=[['X', 'G'], ['G', 'X']])
	custom_kpath (list): This gives the option to only plot specific segments of a given band structure
	    calculation. For example if the kpath was G-X-W-L then there are three segements to choose from:
	    G-X, X-W, and W-L. In this case the default kpath could be plotted by defining custom_kpath=[1,2,3],
	    where 1 -> G-X, 2 -> X-W, and 3 -> W-L. If only G-X and X-W were desired then custom_kpath=[1,2].
	    If one of the segements should be flipped it can be done by making its value negative
	    (e.g. -1 -> X-G, -2 -> W-X, -3 -> L-W)
        n (int): Number of points between each high symmetry points.
            This is also only required for unfolded calculations and band unfolding. This number should be 
            known by the user, as it was used to generate the KPOINTS file.
        unfold (bool): Determines if the plotted band structure is from a band unfolding calculation.
        M (list[list]): Transformation matrix from the primitive bulk structure to the slab structure.
            Only required for a band unfolding calculation.
        high_symm_points (list[list]): List of fractional coordinated for each high symmetry point in 
            the band structure path. Only required for a band unfolding calculation.
        fontsize (float): Font size of the text in the figure.
        save (bool): Determines whether to automatically save the figure or not. If not 
            the figure and axis are return for further manipulation.

    Returns:
        If save == True, this function will return nothing and directly save the image as
        the output name. If save == False, the function will return the matplotlib figure
        and axis for further editing. (fig, ax1, ax2) 
    """

    band_up = Band(
        folder=folder,
        spin='up',
        projected=True,
        unfold=unfold,
        high_symm_points=high_symm_points, 
	interpolate=interpolate,
	new_n=new_n,
        kpath=kpath,
	custom_kpath=custom_kpath,
        n=n,
        M=M,
	shift_efermi=shift_efermi,
    )

    band_down = Band(
        folder=folder,
        spin='down',
        projected=True,
        unfold=unfold,
        high_symm_points=high_symm_points, 
	interpolate=interpolate,
	new_n=new_n,
        kpath=kpath,
	custom_kpath=custom_kpath,
        n=n,
        M=M,
	shift_efermi=shift_efermi,
    )

    if stack == 'vertical':
        fig = plt.figure(figsize=(figsize[0], 2 * figsize[1]), dpi=400)
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        _figure_setup(ax=ax1, fontsize=fontsize, ylim=[erange[0], erange[1]])
        _figure_setup(ax=ax2, fontsize=fontsize, ylim=[erange[0], erange[1]])
    elif stack == 'horizontal':
        fig = plt.figure(figsize=(2 * figsize[0], figsize[1]), dpi=400)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        _figure_setup(ax=ax1, fontsize=fontsize, ylim=[erange[0], erange[1]])
        _figure_setup(ax=ax2, fontsize=fontsize, ylim=[erange[0], erange[1]])

    bbox = dict(boxstyle="round", fc="white")
    ax1.annotate(
        '$\\uparrow$',
        xy=(0.02, 0.98),
        xycoords='axes fraction',
        va='top',
        ha='left',
        bbox=bbox,
    )
    ax2.annotate(
        '$\\downarrow$',
        xy=(0.02, 0.98),
        xycoords='axes fraction',
        va='top',
        ha='left',
        bbox=bbox,
    )

    band_up.plot_orbitals(
        ax=ax1,
        orbitals=orbitals,
        scale_factor=scale_factor,
        erange=erange,
        display_order=display_order,
        color_list=color_list,
        legend=legend,
        linewidth=linewidth,
        band_color=band_color,
    )
    band_up.plot_plain(
        ax=ax1,
        color=unprojected_band_color,
        linewidth=0.5,
        erange=erange,
    )

    band_down.plot_orbitals(
        ax=ax2,
        orbitals=orbitals,
        scale_factor=scale_factor,
        erange=erange,
        display_order=display_order,
        color_list=color_list,
        legend=legend,
        linewidth=linewidth,
        band_color=band_color,
    )
    band_down.plot_plain(
        ax=ax2,
        color=unprojected_band_color,
        linewidth=0.5,
        erange=erange,
    )

    plt.tight_layout(pad=0.4)

    if save:
        plt.savefig(output)
    else:
        return fig, ax1, ax2


def band_atoms_spin_polarized(
    folder,
    atoms,
    output='band_atoms_sp.png',
    display_order=None,
    scale_factor=5,
    color_list=None,
    legend=True,
    linewidth=0.75,
    band_color='black',
    unprojected_band_color='gray',
    figsize=(4, 3),
    erange=[-6, 6],
    stack='vertical',
    kpath=None,
    custom_kpath=None,
    n=None,
    unfold=False,
    M=None,
    high_symm_points=None,
    fontsize=12,
    save=True,
    shift_efermi=0,
    interpolate=False,
    new_n=200,
):
    """
    This function generates an atom spin polarized band structure. This will plot two plots
    stacked on top or eachother or next to eachother. The top or left plot will project on the 
    spin up bands and the bottom or right plot will project onto the spin down bands.

    Parameters:
        folder (str): This is the folder that contains the VASP files
        atoms (list): List of atoms to project onto. The indices should be zero indexed (first atom is 0)
            and the atoms are in the same order as they are in the POSCAR
        output (str): File name of the resulting plot.
        scale_factor (float): Factor to scale weights. This changes the size of the
            points in the scatter plot.
        display_order (str / None): If None, the projections will be displayed in the same order
            the user inputs them. If 'all' the projections will be plotted from largest to smallest
            so every point is visable. If 'dominant' the projections will be plotted from smallest 
            to largest so only the dominant projection is shown.
        color_list (list): List of colors that is the same length as the number of projections
            in the plot.
        legend (bool): Determines if the legend should be included or not.
        unprojected_band_color (str): Color of the unprojected band
        unprojected_linewidth (float): Line width of the unprojected bands
        annotations (list): Annotations to put on the top and bottom (left and right) figures.
            By default it will show the spin up and spin down arrows.
        annotation_xy (list / tuple): Fractional (x, y) coordinated of the annotation location
        stack (str): Determines how the plots are stacked (vertical or horizontal)
        linewidth (float): Line width of the plain band structure plotted in the background.
        band_color (string): Color of the plain band structure.
        figsize (list / tuple): Desired size of the image in inches. (width, height)
        erange (list / tuple): Range of energy to show in the plot. [low, high]
        kpath (list[list]): High symmetry k-point path of band structure calculation
            Due to the nature of the KPOINTS file for unfolded calculations this
            information is a required input for proper labeling of the figure
            for unfolded calculations. This information is extracted from the KPOINTS
            files for non-unfolded calculations. (G is automatically converted to \\Gamma)
	    (e.g. For the path X-G-X, kpath=[['X', 'G'], ['G', 'X']])
	custom_kpath (list): This gives the option to only plot specific segments of a given band structure
	    calculation. For example if the kpath was G-X-W-L then there are three segements to choose from:
	    G-X, X-W, and W-L. In this case the default kpath could be plotted by defining custom_kpath=[1,2,3],
	    where 1 -> G-X, 2 -> X-W, and 3 -> W-L. If only G-X and X-W were desired then custom_kpath=[1,2].
	    If one of the segements should be flipped it can be done by making its value negative
	    (e.g. -1 -> X-G, -2 -> W-X, -3 -> L-W)
        n (int): Number of points between each high symmetry points.
            This is also only required for unfolded calculations and band unfolding. This number should be 
            known by the user, as it was used to generate the KPOINTS file.
        unfold (bool): Determines if the plotted band structure is from a band unfolding calculation.
        M (list[list]): Transformation matrix from the primitive bulk structure to the slab structure.
            Only required for a band unfolding calculation.
        high_symm_points (list[list]): List of fractional coordinated for each high symmetry point in 
            the band structure path. Only required for a band unfolding calculation.
        fontsize (float): Font size of the text in the figure.
        save (bool): Determines whether to automatically save the figure or not. If not 
            the figure and axis are return for further manipulation.

    Returns:
        If save == True, this function will return nothing and directly save the image as
        the output name. If save == False, the function will return the matplotlib figure
        and axis for further editing. (fig, ax1, ax2) 
    """

    band_up = Band(
        folder=folder,
        spin='up',
        projected=True,
        unfold=unfold,
        high_symm_points=high_symm_points, 
	interpolate=interpolate,
	new_n=new_n,
        kpath=kpath,
	custom_kpath=custom_kpath,
        n=n,
        M=M,
	shift_efermi=shift_efermi,
    )

    band_down = Band(
        folder=folder,
        spin='down',
        projected=True,
        unfold=unfold,
        high_symm_points=high_symm_points, 
	interpolate=interpolate,
	new_n=new_n,
        kpath=kpath,
	custom_kpath=custom_kpath,
        n=n,
        M=M,
	shift_efermi=shift_efermi,
    )

    if stack == 'vertical':
        fig = plt.figure(figsize=(figsize[0], 2 * figsize[1]), dpi=400)
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        _figure_setup(ax=ax1, fontsize=fontsize, ylim=[erange[0], erange[1]])
        _figure_setup(ax=ax2, fontsize=fontsize, ylim=[erange[0], erange[1]])
    elif stack == 'horizontal':
        fig = plt.figure(figsize=(2 * figsize[0], figsize[1]), dpi=400)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        _figure_setup(ax=ax1, fontsize=fontsize, ylim=[erange[0], erange[1]])
        _figure_setup(ax=ax2, fontsize=fontsize, ylim=[erange[0], erange[1]])

    bbox = dict(boxstyle="round", fc="white")
    ax1.annotate(
        '$\\uparrow$',
        xy=(0.02, 0.98),
        xycoords='axes fraction',
        va='top',
        ha='left',
        bbox=bbox,
    )
    ax2.annotate(
        '$\\downarrow$',
        xy=(0.02, 0.98),
        xycoords='axes fraction',
        va='top',
        ha='left',
        bbox=bbox,
    )


    band_up.plot_atoms(
        ax=ax1,
        atoms=atoms,
        scale_factor=scale_factor,
        color_list=color_list,
        display_order=display_order,
        legend=legend,
        linewidth=linewidth,
        band_color=band_color,
        erange=erange,
    )
    band_up.plot_plain(
        ax=ax1,
        color=unprojected_band_color,
        linewidth=0.5,
        erange=erange,
    )

    band_down.plot_atoms(
        ax=ax2,
        atoms=atoms,
        scale_factor=scale_factor,
        color_list=color_list,
        display_order=display_order,
        legend=legend,
        linewidth=linewidth,
        band_color=band_color,
        erange=erange,
    )
    band_down.plot_plain(
        ax=ax2,
        color=unprojected_band_color,
        linewidth=0.5,
        erange=erange,
    )

    plt.tight_layout(pad=0.4)

    if save:
        plt.savefig(output)
    else:
        return fig, ax1, ax2

def band_atom_spd_spin_polarized(
    folder,
    atom_spd_dict,
    output='band_atoms_spd_sp.png',
    display_order=None,
    scale_factor=5,
    color_list=None,
    legend=True,
    linewidth=0.75,
    band_color='black',
    unprojected_band_color='gray',
    unprojected_linewidth=0.6,
    fontsize=12,
    annotations=['$\\uparrow$ ', '$\\downarrow$ '],
    annotation_xy=(0.02, 0.98),
    figsize=(4, 3),
    erange=[-6, 6],
    stack='vertical',
    kpath=None,
    custom_kpath=None,
    n=None,
    unfold=False,
    M=None,
    high_symm_points=None,
    save=True,
    shift_efermi=0,
    interpolate=False,
    new_n=200,
):
    """
    This function generates a s, p, d spin polarized band structure on specific atoms. This will plot two plots
    stacked on top or eachother or next to eachother. The top or left plot will project on the 
    spin up bands and the bottom or right plot will project onto the spin down bands.

    Parameters:
        folder (str): This is the folder that contains the VASP files
        atom_spd_dict (dict[int:str]): A dictionary that contains the individual atoms and the corresponding 
            orbitals to project onto. For example, if the user wants to project onto the s, p, d orbitals
            of the first atom and the p orbitals of the second atom then the dictionary would be {0:'spd', 1:'p'}
        output (str): File name of the resulting plot.
        scale_factor (float): Factor to scale weights. This changes the size of the
            points in the scatter plot.
        display_order (str / None): If None, the projections will be displayed in the same order
            the user inputs them. If 'all' the projections will be plotted from largest to smallest
            so every point is visable. If 'dominant' the projections will be plotted from smallest 
            to largest so only the dominant projection is shown.
        color_list (list): List of colors that is the same length as the number of projections
            in the plot.
        legend (bool): Determines if the legend should be included or not.
        unprojected_band_color (str): Color of the unprojected band
        unprojected_linewidth (float): Line width of the unprojected bands
        annotations (list): Annotations to put on the top and bottom (left and right) figures.
            By default it will show the spin up and spin down arrows.
        annotation_xy (list / tuple): Fractional (x, y) coordinated of the annotation location
        stack (str): Determines how the plots are stacked (vertical or horizontal)
        linewidth (float): Line width of the plain band structure plotted in the background.
        band_color (string): Color of the plain band structure.
        figsize (list / tuple): Desired size of the image in inches. (width, height)
        erange (list / tuple): Range of energy to show in the plot. [low, high]
        kpath (list[list]): High symmetry k-point path of band structure calculation
            Due to the nature of the KPOINTS file for unfolded calculations this
            information is a required input for proper labeling of the figure
            for unfolded calculations. This information is extracted from the KPOINTS
            files for non-unfolded calculations. (G is automatically converted to \\Gamma)
	    (e.g. For the path X-G-X, kpath=[['X', 'G'], ['G', 'X']])
	custom_kpath (list): This gives the option to only plot specific segments of a given band structure
	    calculation. For example if the kpath was G-X-W-L then there are three segements to choose from:
	    G-X, X-W, and W-L. In this case the default kpath could be plotted by defining custom_kpath=[1,2,3],
	    where 1 -> G-X, 2 -> X-W, and 3 -> W-L. If only G-X and X-W were desired then custom_kpath=[1,2].
	    If one of the segements should be flipped it can be done by making its value negative
	    (e.g. -1 -> X-G, -2 -> W-X, -3 -> L-W)
        n (int): Number of points between each high symmetry points.
            This is also only required for unfolded calculations and band unfolding. This number should be 
            known by the user, as it was used to generate the KPOINTS file.
        unfold (bool): Determines if the plotted band structure is from a band unfolding calculation.
        M (list[list]): Transformation matrix from the primitive bulk structure to the slab structure.
            Only required for a band unfolding calculation.
        high_symm_points (list[list]): List of fractional coordinated for each high symmetry point in 
            the band structure path. Only required for a band unfolding calculation.
        fontsize (float): Font size of the text in the figure.
        save (bool): Determines whether to automatically save the figure or not. If not 
            the figure and axis are return for further manipulation.

    Returns:
        If save == True, this function will return nothing and directly save the image as
        the output name. If save == False, the function will return the matplotlib figure
        and axis for further editing. (fig, ax1, ax2) 
    """

    band_up = Band(
        folder=folder,
        spin='up',
        projected=True,
        unfold=unfold,
        high_symm_points=high_symm_points, 
	interpolate=interpolate,
	new_n=new_n,
        kpath=kpath,
	custom_kpath=custom_kpath,
        n=n,
        M=M,
	shift_efermi=shift_efermi,
    )

    band_down = Band(
        folder=folder,
        spin='down',
        projected=True,
        unfold=unfold,
        high_symm_points=high_symm_points, 
	interpolate=interpolate,
	new_n=new_n,
        kpath=kpath,
	custom_kpath=custom_kpath,
        n=n,
        M=M,
	shift_efermi=shift_efermi,
    )

    if stack == 'vertical':
        fig = plt.figure(figsize=(figsize[0], 2 * figsize[1]), dpi=400)
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        _figure_setup(ax=ax1, fontsize=fontsize, ylim=[erange[0], erange[1]])
        _figure_setup(ax=ax2, fontsize=fontsize, ylim=[erange[0], erange[1]])
    elif stack == 'horizontal':
        fig = plt.figure(figsize=(2 * figsize[0], figsize[1]), dpi=400)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        _figure_setup(ax=ax1, fontsize=fontsize, ylim=[erange[0], erange[1]])
        _figure_setup(ax=ax2, fontsize=fontsize, ylim=[erange[0], erange[1]])

    bbox = dict(boxstyle='round', fc='white',
                edgecolor='gray', alpha=0.95, pad=0.3)
    ax1.annotate(
        annotations[0],
        xy=annotation_xy,
        xycoords='axes fraction',
        va='top',
        ha='left',
        bbox=bbox,
        fontsize=fontsize,
    )
    ax2.annotate(
        annotations[1],
        xy=annotation_xy,
        xycoords='axes fraction',
        va='top',
        ha='left',
        bbox=bbox,
        fontsize=fontsize,
    )

    band_up.plot_atom_spd(
        ax=ax1,
        atom_spd_dict=atom_spd_dict,
        display_order=display_order,
        scale_factor=scale_factor,
        color_list=color_list,
        legend=legend,
        linewidth=linewidth,
        band_color=band_color,
        erange=erange,
    )
    band_down.plot_plain(
        ax=ax1,
        color=unprojected_band_color,
        linewidth=unprojected_linewidth,
        erange=erange,
    )

    band_down.plot_atom_spd(
        ax=ax2,
        atom_spd_dict=atom_spd_dict,
        display_order=display_order,
        scale_factor=scale_factor,
        color_list=color_list,
        legend=legend,
        linewidth=linewidth,
        band_color=band_color,
        erange=erange,
    )
    band_up.plot_plain(
        ax=ax2,
        color=unprojected_band_color,
        linewidth=unprojected_linewidth,
        erange=erange,
    )

    plt.tight_layout(pad=0.4)

    if save:
        plt.savefig(output)
    else:
        return fig, ax1, ax2


def band_elements_spin_polarized(
    folder,
    elements,
    output='band_elements_sp.png',
    scale_factor=5,
    display_order=None,
    color_list=None,
    legend=True,
    linewidth=0.75,
    band_color='black',
    unprojected_band_color='gray',
    figsize=(4, 3),
    erange=[-6, 6],
    stack='vertical',
    kpath=None,
    custom_kpath=None,
    n=None,
    unfold=False,
    M=None,
    high_symm_points=None,
    fontsize=12,
    save=True,
    shift_efermi=0,
    interpolate=False,
    new_n=200,
):
    """
    This function generates an element spin polarized band structure. This will plot two plots
    stacked on top or eachother or next to eachother. The top or left plot will project on the 
    spin up bands and the bottom or right plot will project onto the spin down bands.

    Parameters:
        folder (str): This is the folder that contains the VASP files
        elements (list): List of elements to project onto. The list should countain the corresponding element symbols
        output (str): File name of the resulting plot.
        scale_factor (float): Factor to scale weights. This changes the size of the
            points in the scatter plot.
        display_order (str / None): If None, the projections will be displayed in the same order
            the user inputs them. If 'all' the projections will be plotted from largest to smallest
            so every point is visable. If 'dominant' the projections will be plotted from smallest 
            to largest so only the dominant projection is shown.
        color_list (list): List of colors that is the same length as the number of projections
            in the plot.
        legend (bool): Determines if the legend should be included or not.
        unprojected_band_color (str): Color of the unprojected band
        unprojected_linewidth (float): Line width of the unprojected bands
        annotations (list): Annotations to put on the top and bottom (left and right) figures.
            By default it will show the spin up and spin down arrows.
        annotation_xy (list / tuple): Fractional (x, y) coordinated of the annotation location
        stack (str): Determines how the plots are stacked (vertical or horizontal)
        linewidth (float): Line width of the plain band structure plotted in the background.
        band_color (string): Color of the plain band structure.
        figsize (list / tuple): Desired size of the image in inches. (width, height)
        erange (list / tuple): Range of energy to show in the plot. [low, high]
        kpath (list[list]): High symmetry k-point path of band structure calculation
            Due to the nature of the KPOINTS file for unfolded calculations this
            information is a required input for proper labeling of the figure
            for unfolded calculations. This information is extracted from the KPOINTS
            files for non-unfolded calculations. (G is automatically converted to \\Gamma)
	    (e.g. For the path X-G-X, kpath=[['X', 'G'], ['G', 'X']])
	custom_kpath (list): This gives the option to only plot specific segments of a given band structure
	    calculation. For example if the kpath was G-X-W-L then there are three segements to choose from:
	    G-X, X-W, and W-L. In this case the default kpath could be plotted by defining custom_kpath=[1,2,3],
	    where 1 -> G-X, 2 -> X-W, and 3 -> W-L. If only G-X and X-W were desired then custom_kpath=[1,2].
	    If one of the segements should be flipped it can be done by making its value negative
	    (e.g. -1 -> X-G, -2 -> W-X, -3 -> L-W)
        n (int): Number of points between each high symmetry points.
            This is also only required for unfolded calculations and band unfolding. This number should be 
            known by the user, as it was used to generate the KPOINTS file.
        unfold (bool): Determines if the plotted band structure is from a band unfolding calculation.
        M (list[list]): Transformation matrix from the primitive bulk structure to the slab structure.
            Only required for a band unfolding calculation.
        high_symm_points (list[list]): List of fractional coordinated for each high symmetry point in 
            the band structure path. Only required for a band unfolding calculation.
        fontsize (float): Font size of the text in the figure.
        save (bool): Determines whether to automatically save the figure or not. If not 
            the figure and axis are return for further manipulation.

    Returns:
        If save == True, this function will return nothing and directly save the image as
        the output name. If save == False, the function will return the matplotlib figure
        and axis for further editing. (fig, ax1, ax2) 
    """

    band_up = Band(
        folder=folder,
        spin='up',
        projected=True,
        unfold=unfold,
        high_symm_points=high_symm_points, 
	interpolate=interpolate,
	new_n=new_n,
        kpath=kpath,
	custom_kpath=custom_kpath,
        n=n,
        M=M,
	shift_efermi=shift_efermi,
    )

    band_down = Band(
        folder=folder,
        spin='down',
        projected=True,
        unfold=unfold,
        high_symm_points=high_symm_points, 
	interpolate=interpolate,
	new_n=new_n,
        kpath=kpath,
	custom_kpath=custom_kpath,
        n=n,
        M=M,
	shift_efermi=shift_efermi,
    )

    if stack == 'vertical':
        fig = plt.figure(figsize=(figsize[0], 2 * figsize[1]), dpi=400)
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        _figure_setup(ax=ax1, fontsize=fontsize, ylim=[erange[0], erange[1]])
        _figure_setup(ax=ax2, fontsize=fontsize, ylim=[erange[0], erange[1]])
    elif stack == 'horizontal':
        fig = plt.figure(figsize=(2 * figsize[0], figsize[1]), dpi=400)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        _figure_setup(ax=ax1, fontsize=fontsize, ylim=[erange[0], erange[1]])
        _figure_setup(ax=ax2, fontsize=fontsize, ylim=[erange[0], erange[1]])

    bbox = dict(boxstyle="round", fc="white")
    ax1.annotate(
        '$\\uparrow$',
        xy=(0.02, 0.98),
        xycoords='axes fraction',
        va='top',
        ha='left',
        bbox=bbox,
    )
    ax2.annotate(
        '$\\downarrow$',
        xy=(0.02, 0.98),
        xycoords='axes fraction',
        va='top',
        ha='left',
        bbox=bbox,
    )

    band_up.plot_elements(
        ax=ax1,
        elements=elements,
        scale_factor=scale_factor,
        display_order=display_order,
        color_list=color_list,
        legend=legend,
        linewidth=linewidth,
        band_color=band_color,
        erange=erange,
    )
    band_up.plot_plain(
        ax=ax1,
        color=unprojected_band_color,
        linewidth=0.5,
        erange=erange,
    )

    band_down.plot_elements(
        ax=ax2,
        elements=elements,
        scale_factor=scale_factor,
        display_order=display_order,
        color_list=color_list,
        legend=legend,
        linewidth=linewidth,
        band_color=band_color,
        erange=erange,
    )
    band_down.plot_plain(
        ax=ax2,
        color=unprojected_band_color,
        linewidth=0.5,
        erange=erange,
    )

    plt.tight_layout(pad=0.4)

    if save:
        plt.savefig(output)
    else:
        return fig, ax1, ax2


def band_element_orbital_spin_polarized(
    folder,
    element_orbital_dict,
    output='band_element_orbital_sp.png',
    display_order=None,
    scale_factor=5,
    color_list=None,
    legend=True,
    linewidth=0.75,
    band_color='black',
    unprojected_band_color='gray',
    unprojected_linewidth=0.6,
    fontsize=12,
    annotations=['$\\uparrow$ ', '$\\downarrow$ '],
    annotation_xy=(0.02, 0.98),
    figsize=(4, 3),
    erange=[-6, 6],
    stack='vertical',
    kpath=None,
    custom_kpath=None,
    n=None,
    unfold=False,
    M=None,
    high_symm_points=None,
    save=True,
    shift_efermi=0,
    interpolate=False,
    new_n=200,
):
    """
    This function generates an element orbital spin polarized band structure. This will plot two plots
    stacked on top or eachother or next to eachother. The top or left plot will project on the 
    spin up bands and the bottom or right plot will project onto the spin down bands.

    Parameters:
        folder (str): This is the folder that contains the VASP files
        element_orbital_dict (dict[str:list]): A dictionary that contains the individual elements and the corresponding 
            orbitals to project onto. For example, if the user wants to project onto the s, py, pz, and px orbitals
            of In and the s orbital of As for and InAs structure then the dictionary would be {'In':[0,1,2,3], 'As':[0]}
        output (str): File name of the resulting plot.
        scale_factor (float): Factor to scale weights. This changes the size of the
            points in the scatter plot.
        display_order (str / None): If None, the projections will be displayed in the same order
            the user inputs them. If 'all' the projections will be plotted from largest to smallest
            so every point is visable. If 'dominant' the projections will be plotted from smallest 
            to largest so only the dominant projection is shown.
        color_list (list): List of colors that is the same length as the number of projections
            in the plot.
        legend (bool): Determines if the legend should be included or not.
        unprojected_band_color (str): Color of the unprojected band
        unprojected_linewidth (float): Line width of the unprojected bands
        annotations (list): Annotations to put on the top and bottom (left and right) figures.
            By default it will show the spin up and spin down arrows.
        annotation_xy (list / tuple): Fractional (x, y) coordinated of the annotation location
        stack (str): Determines how the plots are stacked (vertical or horizontal)
        linewidth (float): Line width of the plain band structure plotted in the background.
        band_color (string): Color of the plain band structure.
        figsize (list / tuple): Desired size of the image in inches. (width, height)
        erange (list / tuple): Range of energy to show in the plot. [low, high]
        kpath (list[list]): High symmetry k-point path of band structure calculation
            Due to the nature of the KPOINTS file for unfolded calculations this
            information is a required input for proper labeling of the figure
            for unfolded calculations. This information is extracted from the KPOINTS
            files for non-unfolded calculations. (G is automatically converted to \\Gamma)
	    (e.g. For the path X-G-X, kpath=[['X', 'G'], ['G', 'X']])
	custom_kpath (list): This gives the option to only plot specific segments of a given band structure
	    calculation. For example if the kpath was G-X-W-L then there are three segements to choose from:
	    G-X, X-W, and W-L. In this case the default kpath could be plotted by defining custom_kpath=[1,2,3],
	    where 1 -> G-X, 2 -> X-W, and 3 -> W-L. If only G-X and X-W were desired then custom_kpath=[1,2].
	    If one of the segements should be flipped it can be done by making its value negative
	    (e.g. -1 -> X-G, -2 -> W-X, -3 -> L-W)
        n (int): Number of points between each high symmetry points.
            This is also only required for unfolded calculations and band unfolding. This number should be 
            known by the user, as it was used to generate the KPOINTS file.
        unfold (bool): Determines if the plotted band structure is from a band unfolding calculation.
        M (list[list]): Transformation matrix from the primitive bulk structure to the slab structure.
            Only required for a band unfolding calculation.
        high_symm_points (list[list]): List of fractional coordinated for each high symmetry point in 
            the band structure path. Only required for a band unfolding calculation.
        fontsize (float): Font size of the text in the figure.
        save (bool): Determines whether to automatically save the figure or not. If not 
            the figure and axis are return for further manipulation.

    Returns:
        If save == True, this function will return nothing and directly save the image as
        the output name. If save == False, the function will return the matplotlib figure
        and axis for further editing. (fig, ax1, ax2) 
    """

    band_up = Band(
        folder=folder,
        spin='up',
        projected=True,
        unfold=unfold,
        high_symm_points=high_symm_points, 
	interpolate=interpolate,
	new_n=new_n,
        kpath=kpath,
	custom_kpath=custom_kpath,
        n=n,
        M=M,
	shift_efermi=shift_efermi,
    )

    band_down = Band(
        folder=folder,
        spin='down',
        projected=True,
        unfold=unfold,
        high_symm_points=high_symm_points, 
	interpolate=interpolate,
	new_n=new_n,
        kpath=kpath,
	custom_kpath=custom_kpath,
        n=n,
        M=M,
	shift_efermi=shift_efermi,
    )

    if stack == 'vertical':
        fig = plt.figure(figsize=(figsize[0], 2 * figsize[1]), dpi=400)
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        _figure_setup(ax=ax1, fontsize=fontsize, ylim=[erange[0], erange[1]])
        _figure_setup(ax=ax2, fontsize=fontsize, ylim=[erange[0], erange[1]])
    elif stack == 'horizontal':
        fig = plt.figure(figsize=(2 * figsize[0], figsize[1]), dpi=400)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        _figure_setup(ax=ax1, fontsize=fontsize, ylim=[erange[0], erange[1]])
        _figure_setup(ax=ax2, fontsize=fontsize, ylim=[erange[0], erange[1]])

    bbox = dict(boxstyle='round', fc='white',
                edgecolor='gray', alpha=0.95, pad=0.3)
    ax1.annotate(
        annotations[0],
        xy=annotation_xy,
        xycoords='axes fraction',
        va='top',
        ha='left',
        bbox=bbox,
        fontsize=fontsize,
    )
    ax2.annotate(
        annotations[1],
        xy=annotation_xy,
        xycoords='axes fraction',
        va='top',
        ha='left',
        bbox=bbox,
        fontsize=fontsize,
    )

    band_up.plot_element_orbitals(
        ax=ax1,
        element_orbital_dict=element_orbital_dict,
        scale_factor=scale_factor,
        display_order=display_order,
        color_list=color_list,
        legend=legend,
        linewidth=linewidth,
        band_color=band_color,
        erange=erange,
    )
    band_down.plot_plain(
        ax=ax1,
        color=unprojected_band_color,
        linewidth=unprojected_linewidth,
        erange=erange,
    )

    band_down.plot_element_orbitals(
        ax=ax2,
        element_orbital_dict=element_orbital_dict,
        scale_factor=scale_factor,
        display_order=display_order,
        color_list=color_list,
        legend=legend,
        linewidth=linewidth,
        band_color=band_color,
        erange=erange,
    )
    band_up.plot_plain(
        ax=ax2,
        color=unprojected_band_color,
        linewidth=unprojected_linewidth,
        erange=erange,
    )

    plt.tight_layout(pad=0.4)

    if save:
        plt.savefig(output)
    else:
        return fig, ax1, ax2


def band_element_spd_spin_polarized(
    folder,
    element_spd_dict,
    output='band_elements_spd_sp.png',
    display_order=None,
    scale_factor=5,
    color_list=None,
    legend=True,
    linewidth=0.75,
    band_color='black',
    unprojected_band_color='gray',
    unprojected_linewidth=0.6,
    fontsize=12,
    annotations=['$\\uparrow$ ', '$\\downarrow$ '],
    annotation_xy=(0.02, 0.98),
    figsize=(4, 3),
    erange=[-6, 6],
    stack='vertical',
    kpath=None,
    custom_kpath=None,
    n=None,
    unfold=False,
    M=None,
    high_symm_points=None,
    save=True,
    shift_efermi=0,
    interpolate=False,
    new_n=200,
):
    """
    This function generates a s, p, d spin polarized band structure on specific elements. This will plot two plots
    stacked on top or eachother or next to eachother. The top or left plot will project on the 
    spin up bands and the bottom or right plot will project onto the spin down bands.

    Parameters:
        folder (str): This is the folder that contains the VASP files
        element_spd_dict (dict[str:str]): A dictionary that contains the individual atoms and the corresponding 
            orbitals to project onto. For example, if the user wants to project onto the s, p, d orbitals
            of In and the p orbitals of As for an InAs structure then the dictionary would be {'In':'spd', 'As':'p'}
        output (str): File name of the resulting plot.
        scale_factor (float): Factor to scale weights. This changes the size of the
            points in the scatter plot.
        display_order (str / None): If None, the projections will be displayed in the same order
            the user inputs them. If 'all' the projections will be plotted from largest to smallest
            so every point is visable. If 'dominant' the projections will be plotted from smallest 
            to largest so only the dominant projection is shown.
        color_list (list): List of colors that is the same length as the number of projections
            in the plot.
        legend (bool): Determines if the legend should be included or not.
        unprojected_band_color (str): Color of the unprojected band
        unprojected_linewidth (float): Line width of the unprojected bands
        annotations (list): Annotations to put on the top and bottom (left and right) figures.
            By default it will show the spin up and spin down arrows.
        annotation_xy (list / tuple): Fractional (x, y) coordinated of the annotation location
        stack (str): Determines how the plots are stacked (vertical or horizontal)
        linewidth (float): Line width of the plain band structure plotted in the background.
        band_color (string): Color of the plain band structure.
        figsize (list / tuple): Desired size of the image in inches. (width, height)
        erange (list / tuple): Range of energy to show in the plot. [low, high]
        kpath (list[list]): High symmetry k-point path of band structure calculation
            Due to the nature of the KPOINTS file for unfolded calculations this
            information is a required input for proper labeling of the figure
            for unfolded calculations. This information is extracted from the KPOINTS
            files for non-unfolded calculations. (G is automatically converted to \\Gamma)
	    (e.g. For the path X-G-X, kpath=[['X', 'G'], ['G', 'X']])
	custom_kpath (list): This gives the option to only plot specific segments of a given band structure
	    calculation. For example if the kpath was G-X-W-L then there are three segements to choose from:
	    G-X, X-W, and W-L. In this case the default kpath could be plotted by defining custom_kpath=[1,2,3],
	    where 1 -> G-X, 2 -> X-W, and 3 -> W-L. If only G-X and X-W were desired then custom_kpath=[1,2].
	    If one of the segements should be flipped it can be done by making its value negative
	    (e.g. -1 -> X-G, -2 -> W-X, -3 -> L-W)
        n (int): Number of points between each high symmetry points.
            This is also only required for unfolded calculations and band unfolding. This number should be 
            known by the user, as it was used to generate the KPOINTS file.
        unfold (bool): Determines if the plotted band structure is from a band unfolding calculation.
        M (list[list]): Transformation matrix from the primitive bulk structure to the slab structure.
            Only required for a band unfolding calculation.
        high_symm_points (list[list]): List of fractional coordinated for each high symmetry point in 
            the band structure path. Only required for a band unfolding calculation.
        fontsize (float): Font size of the text in the figure.
        save (bool): Determines whether to automatically save the figure or not. If not 
            the figure and axis are return for further manipulation.

    Returns:
        If save == True, this function will return nothing and directly save the image as
        the output name. If save == False, the function will return the matplotlib figure
        and axis for further editing. (fig, ax1, ax2) 
    """

    band_up = Band(
        folder=folder,
        spin='up',
        projected=True,
        unfold=unfold,
        high_symm_points=high_symm_points, 
	interpolate=interpolate,
	new_n=new_n,
        kpath=kpath,
	custom_kpath=custom_kpath,
        n=n,
        M=M,
	shift_efermi=shift_efermi,
    )

    band_down = Band(
        folder=folder,
        spin='down',
        projected=True,
        unfold=unfold,
        high_symm_points=high_symm_points, 
	interpolate=interpolate,
	new_n=new_n,
        kpath=kpath,
	custom_kpath=custom_kpath,
        n=n,
        M=M,
	shift_efermi=shift_efermi,
    )

    if stack == 'vertical':
        fig = plt.figure(figsize=(figsize[0], 2 * figsize[1]), dpi=400)
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        _figure_setup(ax=ax1, fontsize=fontsize, ylim=[erange[0], erange[1]])
        _figure_setup(ax=ax2, fontsize=fontsize, ylim=[erange[0], erange[1]])
    elif stack == 'horizontal':
        fig = plt.figure(figsize=(2 * figsize[0], figsize[1]), dpi=400)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        _figure_setup(ax=ax1, fontsize=fontsize, ylim=[erange[0], erange[1]])
        _figure_setup(ax=ax2, fontsize=fontsize, ylim=[erange[0], erange[1]])

    bbox = dict(boxstyle='round', fc='white',
                edgecolor='gray', alpha=0.95, pad=0.3)
    ax1.annotate(
        annotations[0],
        xy=annotation_xy,
        xycoords='axes fraction',
        va='top',
        ha='left',
        bbox=bbox,
        fontsize=fontsize,
    )
    ax2.annotate(
        annotations[1],
        xy=annotation_xy,
        xycoords='axes fraction',
        va='top',
        ha='left',
        bbox=bbox,
        fontsize=fontsize,
    )

    band_up.plot_element_spd(
        ax=ax1,
        element_spd_dict=element_spd_dict,
        display_order=display_order,
        scale_factor=scale_factor,
        color_list=color_list,
        legend=legend,
        linewidth=linewidth,
        band_color=band_color,
        erange=erange,
    )
    band_down.plot_plain(
        ax=ax1,
        color=unprojected_band_color,
        linewidth=unprojected_linewidth,
        erange=erange,
    )

    band_down.plot_element_spd(
        ax=ax2,
        element_spd_dict=element_spd_dict,
        display_order=display_order,
        scale_factor=scale_factor,
        color_list=color_list,
        legend=legend,
        linewidth=linewidth,
        band_color=band_color,
        erange=erange,
    )
    band_up.plot_plain(
        ax=ax2,
        color=unprojected_band_color,
        linewidth=unprojected_linewidth,
        erange=erange,
    )

    plt.tight_layout(pad=0.4)

    if save:
        plt.savefig(output)
    else:
        return fig, ax1, ax2

# =============================================================
# -------------------- Density of States ----------------------
# =============================================================


def dos_plain(
    folder,
    output='dos_plain.png',
    linewidth=1.5,
    fill=True,
    alpha=0.3,
    sigma=0.05,
    energyaxis='x',
    color='black',
    figsize=(4, 3),
    erange=[-6, 6],
    spin='up',
    soc_axis=None,
    combination_method='add',
    fontsize=12,
    save=True,
    shift_efermi=0,
):
    """
    This function plots the total density of states.

    Parameters:
        folder (str): This is the folder that contains the VASP files
        output (str): File name of the resulting plot.
        fill (bool): Determines wether or not to fill underneath the plot
        alpha (float): Alpha value for the fill
        linewidth (float): Linewidth of lines
        sigma (float): Standard deviation for gaussian filter
        energyaxis (str): Determines the axis to plot the energy on ('x' or 'y')
        color (list): Color of plot and fill.
        spin (str): Which spin direction to parse ('up' or 'down')
        figsize (list / tuple): Desired size of the image in inches (width, height)
        erange (list): Energy range for the DOS plot ([lower bound, upper bound])
        combination_method (str): If spin == 'both', this determines if the spin up and spin down
            desnities are added or subtracted. ('add' or 'sub')
        fontsize (float): Font size of the text in the figure.
        save (bool): Determines whether to automatically save the figure or not. If not 
            the figure and axis are return for further manipulation.

    Returns:
        If save == True, this function will return nothing and directly save the image as
        the output name. If save == False, the function will return the matplotlib figure
        and axis for further editing. 
    """

    dos = Dos(shift_efermi=shift_efermi, folder=folder, spin=spin, combination_method=combination_method)

    fig = plt.figure(figsize=figsize, dpi=400)
    ax = fig.add_subplot(111)
    _figure_setup_dos(ax=ax, fontsize=fontsize, energyaxis=energyaxis)

    dos.plot_plain(
        ax=ax,
        linewidth=linewidth,
        fill=fill,
        alpha=alpha,
        sigma=sigma,
        energyaxis=energyaxis,
        color=color,
        erange=erange,
    )

    plt.tight_layout(pad=0.4)

    if save:
        plt.savefig(output)
    else:
        return fig, ax

def dos_ldos(
    folder,
    layers,
    output='dos_ldos.png',
    linewidth=1.5,
    fill=True,
    alpha=0.3,
    sigma=0.05,
    energyaxis='x',
    color='black',
    figsize=(4, 3),
    erange=[-6, 6],
    spin='up',
    soc_axis=None,
    combination_method='add',
    fontsize=12,
    save=True,
    shift_efermi=0,
    log_scale=False,
    atol=None,
):
    """
    This function plots the local density of states for atomic layers. Useful for comparing 
    to STS measurments.

    Parameters:
        folder (str): This is the folder that contains the VASP files
        layers (list): List of atomic layers to include in the ldos plot. 0 = first layer 
        output (str): File name of the resulting plot.
        fill (bool): Determines wether or not to fill underneath the plot
        alpha (float): Alpha value for the fill
        linewidth (float): Linewidth of lines
        sigma (float): Standard deviation for gaussian filter
        energyaxis (str): Determines the axis to plot the energy on ('x' or 'y')
        color (list): Color of plot and fill.
        spin (str): Which spin direction to parse ('up' or 'down')
        figsize (list / tuple): Desired size of the image in inches (width, height)
        erange (list): Energy range for the DOS plot ([lower bound, upper bound])
        combination_method (str): If spin == 'both', this determines if the spin up and spin down
            desnities are added or subtracted. ('add' or 'sub')
        fontsize (float): Font size of the text in the figure.
        save (bool): Determines whether to automatically save the figure or not. If not 
            the figure and axis are return for further manipulation.

    Returns:
        If save == True, this function will return nothing and directly save the image as
        the output name. If save == False, the function will return the matplotlib figure
        and axis for further editing. 
    """

    dos = Dos(shift_efermi=shift_efermi, folder=folder, spin=spin, combination_method=combination_method)

    fig = plt.figure(figsize=figsize, dpi=400)
    ax = fig.add_subplot(111)
    _figure_setup_dos(ax=ax, fontsize=fontsize, energyaxis=energyaxis, log_scale=log_scale)

    dos.plot_ldos(
        ax=ax,
        layers=layers,
        linewidth=linewidth,
        fill=fill,
        alpha=alpha,
        sigma=sigma,
        energyaxis=energyaxis,
        color=color,
        erange=erange,
        log_scale=log_scale,
        atol=atol,
    )

    plt.tight_layout(pad=0.4)

    if save:
        plt.savefig(output)
    else:
        return fig, ax


def dos_spd(
    folder,
    output='dos_spd.png',
    orbitals='spd',
    fill=True,
    alpha=0.3,
    linewidth=1.5,
    sigma=0.05,
    energyaxis='x',
    color_list=None,
    legend=True,
    total=True,
    figsize=(4, 3),
    erange=[-6, 6],
    spin='up',
    soc_axis=None,
    combination_method='add',
    fontsize=12,
    save=True,
    shift_efermi=0,
):
    """
    This function plots an s, p, d projected density of states.

    Parameters:
        folder (str): This is the folder that contains the VASP files
        orbitals (str): String that contains the s, p, or d orbitals that to project onto.
            The default is 'spd', if the user only wanted to project onto the p, and d orbitals 
            than 'pd' should be passed in
        output (str): File name of the resulting plot.
        fill (bool): Determines wether or not to fill underneath the plot
        alpha (float): Alpha value for the fill
        linewidth (float): Linewidth of lines
        sigma (float): Standard deviation for gaussian filter
        energyaxis (str): Determines the axis to plot the energy on ('x' or 'y')
        color_list (list): List of colors that is the same length as the number of projections
            in the plot.
        legend (bool): Determines whether to draw the legend or not
        total (bool): Determines wheth to draw the total density of states or not
        spin (str): Which spin direction to parse ('up' or 'down')
        figsize (list / tuple): Desired size of the image in inches (width, height)
        erange (list): Energy range for the DOS plot ([lower bound, upper bound])
        combination_method (str): If spin == 'both', this determines if the spin up and spin down
            desnities are added or subtracted. ('add' or 'sub')
        fontsize (float): Font size of the text in the figure.
        save (bool): Determines whether to automatically save the figure or not. If not 
            the figure and axis are return for further manipulation.

    Returns:
        If save == True, this function will return nothing and directly save the image as
        the output name. If save == False, the function will return the matplotlib figure
        and axis for further editing. 
    """

    dos = Dos(shift_efermi=shift_efermi, folder=folder, spin=spin, combination_method=combination_method)

    fig = plt.figure(figsize=figsize, dpi=400)
    ax = fig.add_subplot(111)
    _figure_setup_dos(ax=ax, fontsize=fontsize, energyaxis=energyaxis)

    dos.plot_spd(
        ax=ax,
        orbitals=orbitals,
        fill=fill,
        alpha=alpha,
        linewidth=linewidth,
        sigma=sigma,
        energyaxis=energyaxis,
        color_list=color_list,
        legend=legend,
        total=total,
        erange=erange,
    )

    plt.tight_layout(pad=0.4)

    if save:
        plt.savefig(output)
    else:
        return fig, ax


def dos_atom_orbitals(
    folder,
    atom_orbital_dict,
    output='dos_atom_orbitals.png',
    fill=True,
    alpha=0.3,
    linewidth=1.5,
    sigma=0.05,
    energyaxis='x',
    color_list=None,
    legend=True,
    total=True,
    figsize=(4, 3),
    erange=[-6, 6],
    spin='up',
    soc_axis=None,
    combination_method='add',
    fontsize=12,
    save=True,
    shift_efermi=0,
):
    """
    This function plots the orbital projected density of states on specific atoms.

    Parameters:
        folder (str): This is the folder that contains the VASP files
        atom_orbital_dict (dict[int:list]): A dictionary that contains the individual atoms and the corresponding 
            orbitals to project onto. For example, if the user wants to project onto the s, py, pz, and px orbitals
            of the first atom and the s orbital of the second atom then the dictionary would be {0:[0,1,2,3], 1:[0]}
        output (str): File name of the resulting plot.
        fill (bool): Determines wether or not to fill underneath the plot
        alpha (float): Alpha value for the fill
        linewidth (float): Linewidth of lines
        sigma (float): Standard deviation for gaussian filter
        energyaxis (str): Determines the axis to plot the energy on ('x' or 'y')
        color_list (list): List of colors that is the same length as the number of projections
            in the plot.
        legend (bool): Determines whether to draw the legend or not
        total (bool): Determines wheth to draw the total density of states or not
        spin (str): Which spin direction to parse ('up' or 'down')
        figsize (list / tuple): Desired size of the image in inches (width, height)
        erange (list): Energy range for the DOS plot ([lower bound, upper bound])
        combination_method (str): If spin == 'both', this determines if the spin up and spin down
            desnities are added or subtracted. ('add' or 'sub')
        fontsize (float): Font size of the text in the figure.
        save (bool): Determines whether to automatically save the figure or not. If not 
            the figure and axis are return for further manipulation.

    Returns:
        If save == True, this function will return nothing and directly save the image as
        the output name. If save == False, the function will return the matplotlib figure
        and axis for further editing. 
    """

    dos = Dos(shift_efermi=shift_efermi, folder=folder, spin=spin, combination_method=combination_method)

    fig = plt.figure(figsize=figsize, dpi=400)
    ax = fig.add_subplot(111)
    _figure_setup_dos(ax=ax, fontsize=fontsize, energyaxis=energyaxis)

    dos.plot_atom_orbitals(
        ax=ax,
        atom_orbital_dict=atom_orbital_dict,
        fill=fill,
        alpha=alpha,
        linewidth=linewidth,
        sigma=sigma,
        energyaxis=energyaxis,
        color_list=color_list,
        legend=legend,
        total=total,
        erange=erange,
    )

    plt.tight_layout(pad=0.4)

    if save:
        plt.savefig(output)
    else:
        return fig, ax


def dos_orbitals(
    folder,
    orbitals,
    output='dos_orbitals.png',
    fill=True,
    alpha=0.3,
    linewidth=1.5,
    sigma=0.05,
    energyaxis='x',
    color_list=None,
    legend=True,
    total=True,
    figsize=(4, 3),
    erange=[-6, 6],
    spin='up',
    soc_axis=None,
    combination_method='add',
    fontsize=12,
    save=True,
    shift_efermi=0,
):
    """
    This function plots the orbital projected density of states.

    Parameters:
        folder (str): This is the folder that contains the VASP files
        orbitals (list): List of orbitals to compare

            | 0 = s
            | 1 = py
            | 2 = pz
            | 3 = px
            | 4 = dxy
            | 5 = dyz
            | 6 = dz2
            | 7 = dxz
            | 8 = dx2-y2
            | 9 = fy3x2
            | 10 = fxyz
            | 11 = fyz2
            | 12 = fz3
            | 13 = fxz2
            | 14 = fzx3
            | 15 = fx3

        output (str): File name of the resulting plot.
        fill (bool): Determines wether or not to fill underneath the plot
        alpha (float): Alpha value for the fill
        linewidth (float): Linewidth of lines
        sigma (float): Standard deviation for gaussian filter
        energyaxis (str): Determines the axis to plot the energy on ('x' or 'y')
        color_list (list): List of colors that is the same length as the number of projections
            in the plot.
        legend (bool): Determines whether to draw the legend or not
        total (bool): Determines wheth to draw the total density of states or not
        spin (str): Which spin direction to parse ('up' or 'down')
        figsize (list / tuple): Desired size of the image in inches (width, height)
        erange (list): Energy range for the DOS plot ([lower bound, upper bound])
        combination_method (str): If spin == 'both', this determines if the spin up and spin down
            desnities are added or subtracted. ('add' or 'sub')
        fontsize (float): Font size of the text in the figure.
        save (bool): Determines whether to automatically save the figure or not. If not 
            the figure and axis are return for further manipulation.

    Returns:
        If save == True, this function will return nothing and directly save the image as
        the output name. If save == False, the function will return the matplotlib figure
        and axis for further editing. 
    """

    dos = Dos(shift_efermi=shift_efermi, folder=folder, spin=spin, combination_method=combination_method)

    fig = plt.figure(figsize=figsize, dpi=400)
    ax = fig.add_subplot(111)
    _figure_setup_dos(ax=ax, fontsize=fontsize, energyaxis=energyaxis)

    dos.plot_orbitals(
        ax=ax,
        orbitals=orbitals,
        fill=fill,
        alpha=alpha,
        linewidth=linewidth,
        sigma=sigma,
        energyaxis=energyaxis,
        color_list=color_list,
        legend=legend,
        total=total,
        erange=erange,
    )

    plt.tight_layout(pad=0.4)

    if save:
        plt.savefig(output)
    else:
        return fig, ax


def dos_atoms(
    folder,
    atoms,
    output='dos_atoms.png',
    fill=True,
    alpha=0.3,
    linewidth=1.5,
    sigma=0.05,
    energyaxis='x',
    color_list=None,
    legend=True,
    total=True,
    figsize=(4, 3),
    erange=[-6, 6],
    spin='up',
    soc_axis=None,
    combination_method='add',
    fontsize=12,
    save=True,
    shift_efermi=0,
):
    """
    This function plots the atom projected density of states.

    Parameters:
        folder (str): This is the folder that contains the VASP files
        atoms (list): List of atoms to project onto. The indices should be zero indexed (first atom is 0)
            and the atoms are in the same order as they are in the POSCAR
        output (str): File name of the resulting plot.
        fill (bool): Determines wether or not to fill underneath the plot
        alpha (float): Alpha value for the fill
        linewidth (float): Linewidth of lines
        sigma (float): Standard deviation for gaussian filter
        energyaxis (str): Determines the axis to plot the energy on ('x' or 'y')
        color_list (list): List of colors that is the same length as the number of projections
            in the plot.
        legend (bool): Determines whether to draw the legend or not
        total (bool): Determines wheth to draw the total density of states or not
        spin (str): Which spin direction to parse ('up' or 'down')
        figsize (list / tuple): Desired size of the image in inches (width, height)
        erange (list): Energy range for the DOS plot ([lower bound, upper bound])
        combination_method (str): If spin == 'both', this determines if the spin up and spin down
            desnities are added or subtracted. ('add' or 'sub')
        fontsize (float): Font size of the text in the figure.
        save (bool): Determines whether to automatically save the figure or not. If not 
            the figure and axis are return for further manipulation.

    Returns:
        If save == True, this function will return nothing and directly save the image as
        the output name. If save == False, the function will return the matplotlib figure
        and axis for further editing. 
    """

    dos = Dos(shift_efermi=shift_efermi, folder=folder, spin=spin, combination_method=combination_method)

    fig = plt.figure(figsize=figsize, dpi=400)
    ax = fig.add_subplot(111)
    _figure_setup_dos(ax=ax, fontsize=fontsize, energyaxis=energyaxis)

    dos.plot_atoms(
        ax=ax,
        atoms=atoms,
        fill=fill,
        alpha=alpha,
        linewidth=linewidth,
        sigma=sigma,
        energyaxis=energyaxis,
        color_list=color_list,
        legend=legend,
        total=total,
        erange=erange,
    )

    plt.tight_layout(pad=0.4)

    if save:
        plt.savefig(output)
    else:
        return fig, ax

def dos_atom_spd(
    folder,
    atom_spd_dict,
    output='dos_atom_spd.png',
    fill=True,
    alpha=0.3,
    linewidth=1.5,
    sigma=0.05,
    energyaxis='x',
    color_list=None,
    legend=True,
    total=True,
    figsize=(4, 3),
    erange=[-6, 6],
    spin='up',
    soc_axis=None,
    combination_method='add',
    fontsize=12,
    save=True,
    shift_efermi=0,
):
    """
    This function plots the atom projected density of states of the s, p, and d orbitals.

    Parameters:
        folder (str): This is the folder that contains the VASP files
        atom_spd_dict (dict[int:str]): A dictionary that contains the individual atoms and the corresponding 
            orbitals to project onto. For example, if the user wants to project onto the s, p, d orbitals
            of the first atom and the p orbitals of the second atom then the dictionary would be {0:'spd', 1:'p'}
        output (str): File name of the resulting plot.
        fill (bool): Determines wether or not to fill underneath the plot
        alpha (float): Alpha value for the fill
        linewidth (float): Linewidth of lines
        sigma (float): Standard deviation for gaussian filter
        energyaxis (str): Determines the axis to plot the energy on ('x' or 'y')
        color_list (list): List of colors that is the same length as the number of projections
            in the plot.
        legend (bool): Determines whether to draw the legend or not
        total (bool): Determines wheth to draw the total density of states or not
        spin (str): Which spin direction to parse ('up' or 'down')
        figsize (list / tuple): Desired size of the image in inches (width, height)
        erange (list): Energy range for the DOS plot ([lower bound, upper bound])
        combination_method (str): If spin == 'both', this determines if the spin up and spin down
            desnities are added or subtracted. ('add' or 'sub')
        fontsize (float): Font size of the text in the figure.
        save (bool): Determines whether to automatically save the figure or not. If not 
            the figure and axis are return for further manipulation.

    Returns:
        If save == True, this function will return nothing and directly save the image as
        the output name. If save == False, the function will return the matplotlib figure
        and axis for further editing. 
    """

    dos = Dos(shift_efermi=shift_efermi, folder=folder, spin=spin, combination_method=combination_method)

    fig = plt.figure(figsize=figsize, dpi=400)
    ax = fig.add_subplot(111)
    _figure_setup_dos(ax=ax, fontsize=fontsize, energyaxis=energyaxis)

    dos.plot_atom_spd(
        ax=ax,
        atom_spd_dict=atom_spd_dict,
        fill=fill,
        alpha=alpha,
        linewidth=linewidth,
        sigma=sigma,
        energyaxis=energyaxis,
        color_list=color_list,
        legend=legend,
        total=total,
        erange=erange,
    )

    plt.tight_layout(pad=0.4)

    if save:
        plt.savefig(output)
    else:
        return fig, ax


def dos_elements(
    folder,
    elements,
    output='dos_elements.png',
    fill=True,
    alpha=0.3,
    linewidth=1.5,
    sigma=0.05,
    energyaxis='x',
    color_list=None,
    legend=True,
    total=True,
    figsize=(4, 3),
    erange=[-6, 6],
    spin='up',
    soc_axis=None,
    combination_method='add',
    fontsize=12,
    save=True,
    shift_efermi=0,
):
    """
    This function plots the element projected density of states.

    Parameters:
        folder (str): This is the folder that contains the VASP files
        elements (list): List of elements to project onto. The list should countain the corresponding element symbols
        output (str): File name of the resulting plot.
        fill (bool): Determines wether or not to fill underneath the plot
        alpha (float): Alpha value for the fill
        linewidth (float): Linewidth of lines
        sigma (float): Standard deviation for gaussian filter
        energyaxis (str): Determines the axis to plot the energy on ('x' or 'y')
        color_list (list): List of colors that is the same length as the number of projections
            in the plot.
        legend (bool): Determines whether to draw the legend or not
        total (bool): Determines wheth to draw the total density of states or not
        spin (str): Which spin direction to parse ('up' or 'down')
        figsize (list / tuple): Desired size of the image in inches (width, height)
        erange (list): Energy range for the DOS plot ([lower bound, upper bound])
        combination_method (str): If spin == 'both', this determines if the spin up and spin down
            desnities are added or subtracted. ('add' or 'sub')
        fontsize (float): Font size of the text in the figure.
        save (bool): Determines whether to automatically save the figure or not. If not 
            the figure and axis are return for further manipulation.

    Returns:
        If save == True, this function will return nothing and directly save the image as
        the output name. If save == False, the function will return the matplotlib figure
        and axis for further editing. 
    """

    dos = Dos(shift_efermi=shift_efermi, folder=folder, spin=spin, combination_method=combination_method)

    fig = plt.figure(figsize=figsize, dpi=400)
    ax = fig.add_subplot(111)
    _figure_setup_dos(ax=ax, fontsize=fontsize, energyaxis=energyaxis)

    dos.plot_elements(
        ax=ax,
        elements=elements,
        fill=fill,
        alpha=alpha,
        linewidth=linewidth,
        sigma=sigma,
        energyaxis=energyaxis,
        color_list=color_list,
        legend=legend,
        total=total,
        erange=erange,
    )

    plt.tight_layout(pad=0.4)

    if save:
        plt.savefig(output)
    else:
        return fig, ax


def dos_element_spd(
    folder,
    element_spd_dict,
    output='dos_element_spd.png',
    fill=True,
    alpha=0.3,
    linewidth=1.5,
    sigma=0.05,
    energyaxis='x',
    color_list=None,
    legend=True,
    total=True,
    figsize=(4, 3),
    erange=[-6, 6],
    spin='up',
    soc_axis=None,
    combination_method='add',
    fontsize=12,
    save=True,
    shift_efermi=0,
):
    """
    This function plots the element projected density of states of the s, p, and d orbitals.

    Parameters:
        folder (str): This is the folder that contains the VASP files
        element_spd_dict (dict[str:str]): A dictionary that contains the individual atoms and the corresponding 
            orbitals to project onto. For example, if the user wants to project onto the s, p, d orbitals
            of In and the p orbitals of As for an InAs structure then the dictionary would be {'In':'spd', 'As':'p'}
        output (str): File name of the resulting plot.
        fill (bool): Determines wether or not to fill underneath the plot
        alpha (float): Alpha value for the fill
        linewidth (float): Linewidth of lines
        sigma (float): Standard deviation for gaussian filter
        energyaxis (str): Determines the axis to plot the energy on ('x' or 'y')
        color_list (list): List of colors that is the same length as the number of projections
            in the plot.
        legend (bool): Determines whether to draw the legend or not
        total (bool): Determines wheth to draw the total density of states or not
        spin (str): Which spin direction to parse ('up' or 'down')
        figsize (list / tuple): Desired size of the image in inches (width, height)
        erange (list): Energy range for the DOS plot ([lower bound, upper bound])
        combination_method (str): If spin == 'both', this determines if the spin up and spin down
            desnities are added or subtracted. ('add' or 'sub')
        fontsize (float): Font size of the text in the figure.
        save (bool): Determines whether to automatically save the figure or not. If not 
            the figure and axis are return for further manipulation.

    Returns:
        If save == True, this function will return nothing and directly save the image as
        the output name. If save == False, the function will return the matplotlib figure
        and axis for further editing. 
    """

    dos = Dos(shift_efermi=shift_efermi, folder=folder, spin=spin, combination_method=combination_method)

    fig = plt.figure(figsize=figsize, dpi=400)
    ax = fig.add_subplot(111)
    _figure_setup_dos(ax=ax, fontsize=fontsize, energyaxis=energyaxis)

    dos.plot_element_spd(
        ax=ax,
        element_spd_dict=element_spd_dict,
        fill=fill,
        alpha=alpha,
        linewidth=linewidth,
        sigma=sigma,
        energyaxis=energyaxis,
        color_list=color_list,
        legend=legend,
        total=total,
        erange=erange,
    )

    plt.tight_layout(pad=0.4)

    if save:
        plt.savefig(output)
    else:
        return fig, ax


def dos_element_orbitals(
    folder,
    element_orbital_dict,
    output='dos_element_orbitals.png',
    fill=True,
    alpha=0.3,
    linewidth=1.5,
    sigma=0.05,
    energyaxis='x',
    color_list=None,
    legend=True,
    total=True,
    figsize=(4, 3),
    erange=[-6, 6],
    spin='up',
    soc_axis=None,
    combination_method='add',
    fontsize=12,
    save=True,
    shift_efermi=0,
):
    """
    This function plots the element projected density of states on specific orbitals.

    Parameters:
        folder (str): This is the folder that contains the VASP files
        element_orbital_dict (dict[str:list]): A dictionary that contains the individual elements and the corresponding 
            orbitals to project onto. For example, if the user wants to project onto the s, py, pz, and px orbitals
            of In and the s orbital of As for and InAs structure then the dictionary would be {'In':[0,1,2,3], 'As':[0]}
        output (str): File name of the resulting plot.
        fill (bool): Determines wether or not to fill underneath the plot
        alpha (float): Alpha value for the fill
        linewidth (float): Linewidth of lines
        sigma (float): Standard deviation for gaussian filter
        energyaxis (str): Determines the axis to plot the energy on ('x' or 'y')
        color_list (list): List of colors that is the same length as the number of projections
            in the plot.
        legend (bool): Determines whether to draw the legend or not
        total (bool): Determines wheth to draw the total density of states or not
        spin (str): Which spin direction to parse ('up' or 'down')
        figsize (list / tuple): Desired size of the image in inches (width, height)
        erange (list): Energy range for the DOS plot ([lower bound, upper bound])
        combination_method (str): If spin == 'both', this determines if the spin up and spin down
            desnities are added or subtracted. ('add' or 'sub')
        fontsize (float): Font size of the text in the figure.
        save (bool): Determines whether to automatically save the figure or not. If not 
            the figure and axis are return for further manipulation.

    Returns:
        If save == True, this function will return nothing and directly save the image as
        the output name. If save == False, the function will return the matplotlib figure
        and axis for further editing. 
    """

    dos = Dos(shift_efermi=shift_efermi, folder=folder, spin=spin, combination_method=combination_method)

    fig = plt.figure(figsize=figsize, dpi=400)
    ax = fig.add_subplot(111)
    _figure_setup_dos(ax=ax, fontsize=fontsize, energyaxis=energyaxis)

    dos.plot_element_orbitals(
        ax=ax,
        element_orbital_dict=element_orbital_dict,
        fill=fill,
        alpha=alpha,
        linewidth=linewidth,
        sigma=sigma,
        energyaxis=energyaxis,
        color_list=color_list,
        legend=legend,
        total=total,
        erange=erange,
    )

    plt.tight_layout(pad=0.4)

    if save:
        plt.savefig(output)
    else:
        return fig, ax


def dos_plain_spin_polarized(
    folder,
    output='dos_plain_sp.png',
    linewidth=1.5,
    fill=True,
    alpha=0.3,
    sigma=0.05,
    energyaxis='x',
    soc_axis='z',
    color='black',
    figsize=(4, 3),
    erange=[-6, 6],
    fontsize=12,
    save=True,
    shift_efermi=0,
):
    """
    This function plots the spin polarized total density of states.

    Parameters:
        folder (str): This is the folder that contains the VASP files
        output (str): File name of the resulting plot.
        fill (bool): Determines wether or not to fill underneath the plot
        alpha (float): Alpha value for the fill
        linewidth (float): Linewidth of lines
        sigma (float): Standard deviation for gaussian filter
        energyaxis (str): Determines the axis to plot the energy on ('x' or 'y')
        color (str): Color of plot and fill.
        figsize (list / tuple): Desired size of the image in inches (width, height)
        erange (list): Energy range for the DOS plot ([lower bound, upper bound])
        fontsize (float): Font size of the text in the figure.
        save (bool): Determines whether to automatically save the figure or not. If not 
            the figure and axis are return for further manipulation.

    Returns:
        If save == True, this function will return nothing and directly save the image as
        the output name. If save == False, the function will return the matplotlib figure
        and axis for further editing. 
    """

    dos_up = Dos(shift_efermi=shift_efermi, folder=folder, spin='up', soc_axis=soc_axis)
    dos_down = Dos(shift_efermi=shift_efermi, folder=folder, spin='down', soc_axis=soc_axis)

    fig = plt.figure(figsize=figsize, dpi=400)
    ax = fig.add_subplot(111)
    _figure_setup_dos(ax=ax, fontsize=fontsize, energyaxis=energyaxis)

    dos_up.plot_plain(
        ax=ax,
        linewidth=linewidth,
        fill=fill,
        alpha=alpha,
        sigma=sigma,
        energyaxis=energyaxis,
        color=color,
        erange=erange,
    )

    dos_down.plot_plain(
        ax=ax,
        linewidth=linewidth,
        fill=fill,
        alpha=alpha,
        sigma=sigma,
        energyaxis=energyaxis,
        color=color,
        erange=erange,
    )

    plt.tight_layout(pad=0.4)

    if save:
        plt.savefig(output)
    else:
        return fig, ax


def dos_spd_spin_polarized(
    folder,
    output='dos_spd_sp.png',
    orbitals='spd',
    fill=True,
    alpha=0.3,
    linewidth=1.5,
    sigma=0.05,
    energyaxis='x',
    soc_axis='z',
    color_list=None,
    legend=True,
    total=True,
    figsize=(4, 3),
    erange=[-6, 6],
    fontsize=12,
    save=True,
    shift_efermi=0,
):
    """
    This function plots a spin polarized s, p, d projected density of states.

    Parameters:
        folder (str): This is the folder that contains the VASP files
        orbitals (str): String that contains the s, p, or d orbitals that to project onto.
            The default is 'spd', if the user only wanted to project onto the p, and d orbitals 
            than 'pd' should be passed in
        output (str): File name of the resulting plot.
        fill (bool): Determines wether or not to fill underneath the plot
        alpha (float): Alpha value for the fill
        linewidth (float): Linewidth of lines
        sigma (float): Standard deviation for gaussian filter
        energyaxis (str): Determines the axis to plot the energy on ('x' or 'y')
        color_list (list): List of colors that is the same length as the number of projections
            in the plot.
        legend (bool): Determines whether to draw the legend or not
        total (bool): Determines wheth to draw the total density of states or not
        figsize (list / tuple): Desired size of the image in inches (width, height)
        erange (list): Energy range for the DOS plot ([lower bound, upper bound])
        fontsize (float): Font size of the text in the figure.
        save (bool): Determines whether to automatically save the figure or not. If not 
            the figure and axis are return for further manipulation.

    Returns:
        If save == True, this function will return nothing and directly save the image as
        the output name. If save == False, the function will return the matplotlib figure
        and axis for further editing. 
    """

    dos_up = Dos(shift_efermi=shift_efermi, folder=folder, spin='up', soc_axis=soc_axis)
    dos_down = Dos(shift_efermi=shift_efermi, folder=folder, spin='down', soc_axis=soc_axis)

    fig = plt.figure(figsize=figsize, dpi=400)
    ax = fig.add_subplot(111)
    _figure_setup_dos(ax=ax, fontsize=fontsize, energyaxis=energyaxis)

    dos_up.plot_spd(
        ax=ax,
        orbitals=orbitals,
        fill=fill,
        alpha=alpha,
        linewidth=linewidth,
        sigma=sigma,
        energyaxis=energyaxis,
        color_list=color_list,
        legend=legend,
        total=total,
        erange=erange,
    )

    dos_down.plot_spd(
        ax=ax,
        orbitals=orbitals,
        fill=fill,
        alpha=alpha,
        linewidth=linewidth,
        sigma=sigma,
        energyaxis=energyaxis,
        color_list=color_list,
        legend=False,
        total=total,
        erange=erange,
    )

    plt.tight_layout(pad=0.4)

    if save:
        plt.savefig(output)
    else:
        return fig, ax


def dos_atom_orbitals_spin_polarized(
    folder,
    atom_orbital_dict,
    output='dos_atom_orbitals_sp.png',
    fill=True,
    alpha=0.3,
    linewidth=1.5,
    sigma=0.05,
    energyaxis='y',
    soc_axis='z',
    color_list=None,
    legend=True,
    total=True,
    figsize=(4, 3),
    erange=[-6, 6],
    fontsize=12,
    save=True,
    shift_efermi=0,
):
    """
    This function plots a spin polarized orbital projected density of states on specific atoms.

    Parameters:
        folder (str): This is the folder that contains the VASP files
        atom_orbital_dict (dict[int:list]): A dictionary that contains the individual atoms and the corresponding 
            orbitals to project onto. For example, if the user wants to project onto the s, py, pz, and px orbitals
            of the first atom and the s orbital of the second atom then the dictionary would be {0:[0,1,2,3], 1:[0]}
        output (str): File name of the resulting plot.
        fill (bool): Determines wether or not to fill underneath the plot
        alpha (float): Alpha value for the fill
        linewidth (float): Linewidth of lines
        sigma (float): Standard deviation for gaussian filter
        energyaxis (str): Determines the axis to plot the energy on ('x' or 'y')
        color_list (list): List of colors that is the same length as the number of projections
            in the plot.
        legend (bool): Determines whether to draw the legend or not
        total (bool): Determines wheth to draw the total density of states or not
        figsize (list / tuple): Desired size of the image in inches (width, height)
        erange (list): Energy range for the DOS plot ([lower bound, upper bound])
        fontsize (float): Font size of the text in the figure.
        save (bool): Determines whether to automatically save the figure or not. If not 
            the figure and axis are return for further manipulation.

    Returns:
        If save == True, this function will return nothing and directly save the image as
        the output name. If save == False, the function will return the matplotlib figure
        and axis for further editing. 
    """

    dos_up = Dos(shift_efermi=shift_efermi, folder=folder, spin='up', soc_axis=soc_axis)
    dos_down = Dos(shift_efermi=shift_efermi, folder=folder, spin='down', soc_axis=soc_axis)

    fig = plt.figure(figsize=figsize, dpi=400)
    ax = fig.add_subplot(111)
    _figure_setup_dos(ax=ax, fontsize=fontsize, energyaxis=energyaxis)

    dos_up.plot_atom_orbitals(
        ax=ax,
        atom_orbital_dict=atom_orbital_dict,
        fill=fill,
        alpha=alpha,
        linewidth=linewidth,
        sigma=sigma,
        energyaxis=energyaxis,
        color_list=color_list,
        legend=legend,
        total=total,
        erange=erange,
    )

    dos_down.plot_atom_orbitals(
        ax=ax,
        atom_orbital_dict=atom_orbital_dict,
        fill=fill,
        alpha=alpha,
        linewidth=linewidth,
        sigma=sigma,
        energyaxis=energyaxis,
        color_list=color_list,
        legend=False,
        total=total,
        erange=erange,
    )

    plt.tight_layout(pad=0.4)

    if save:
        plt.savefig(output)
    else:
        return fig, ax


def dos_orbitals_spin_polarized(
    folder,
    orbitals,
    output='dos_orbitals_sp.png',
    fill=True,
    alpha=0.3,
    linewidth=1.5,
    sigma=0.05,
    energyaxis='y',
    soc_axis='z',
    color_list=None,
    legend=True,
    total=True,
    figsize=(4, 3),
    erange=[-6, 6],
    fontsize=12,
    save=True,
    shift_efermi=0,
):
    """
    This function plots a spin polarized orbital projected density of states.

    Parameters:
        folder (str): This is the folder that contains the VASP files
        orbitals (list): List of orbitals to compare

            | 0 = s
            | 1 = py
            | 2 = pz
            | 3 = px
            | 4 = dxy
            | 5 = dyz
            | 6 = dz2
            | 7 = dxz
            | 8 = dx2-y2
            | 9 = fy3x2
            | 10 = fxyz
            | 11 = fyz2
            | 12 = fz3
            | 13 = fxz2
            | 14 = fzx3
            | 15 = fx3

        output (str): File name of the resulting plot.
        fill (bool): Determines wether or not to fill underneath the plot
        alpha (float): Alpha value for the fill
        linewidth (float): Linewidth of lines
        sigma (float): Standard deviation for gaussian filter
        energyaxis (str): Determines the axis to plot the energy on ('x' or 'y')
        color_list (list): List of colors that is the same length as the number of projections
            in the plot.
        legend (bool): Determines whether to draw the legend or not
        total (bool): Determines wheth to draw the total density of states or not
        figsize (list / tuple): Desired size of the image in inches (width, height)
        erange (list): Energy range for the DOS plot ([lower bound, upper bound])
        fontsize (float): Font size of the text in the figure.
        save (bool): Determines whether to automatically save the figure or not. If not 
            the figure and axis are return for further manipulation.

    Returns:
        If save == True, this function will return nothing and directly save the image as
        the output name. If save == False, the function will return the matplotlib figure
        and axis for further editing. 
    """

    dos_up = Dos(shift_efermi=shift_efermi, folder=folder, spin='up', soc_axis=soc_axis)
    dos_down = Dos(shift_efermi=shift_efermi, folder=folder, spin='down', soc_axis=soc_axis)

    fig = plt.figure(figsize=figsize, dpi=400)
    ax = fig.add_subplot(111)
    _figure_setup_dos(ax=ax, fontsize=fontsize, energyaxis=energyaxis)

    dos_up.plot_orbitals(
        ax=ax,
        orbitals=orbitals,
        fill=fill,
        alpha=alpha,
        linewidth=linewidth,
        sigma=sigma,
        energyaxis=energyaxis,
        color_list=color_list,
        legend=legend,
        total=total,
        erange=erange,
    )

    dos_down.plot_orbitals(
        ax=ax,
        orbitals=orbitals,
        fill=fill,
        alpha=alpha,
        linewidth=linewidth,
        sigma=sigma,
        energyaxis=energyaxis,
        color_list=color_list,
        legend=False,
        total=total,
        erange=erange,
    )

    plt.tight_layout(pad=0.4)

    if save:
        plt.savefig(output)
    else:
        return fig, ax


def dos_atoms_spin_polarized(
    folder,
    atoms,
    output='dos_atoms_sp.png',
    fill=True,
    alpha=0.3,
    linewidth=1.5,
    sigma=0.05,
    energyaxis='y',
    soc_axis='z',
    color_list=None,
    legend=True,
    total=True,
    figsize=(4, 3),
    erange=[-6, 6],
    fontsize=12,
    save=True,
    shift_efermi=0,
):
    """
    This function plots a spin polarized atom projected density of states.

    Parameters:
        folder (str): This is the folder that contains the VASP files
        atoms (list): List of atoms to project onto. The indices should be zero indexed (first atom is 0)
            and the atoms are in the same order as they are in the POSCAR
        output (str): File name of the resulting plot.
        fill (bool): Determines wether or not to fill underneath the plot
        alpha (float): Alpha value for the fill
        linewidth (float): Linewidth of lines
        sigma (float): Standard deviation for gaussian filter
        energyaxis (str): Determines the axis to plot the energy on ('x' or 'y')
        color_list (list): List of colors that is the same length as the number of projections
            in the plot.
        legend (bool): Determines whether to draw the legend or not
        total (bool): Determines wheth to draw the total density of states or not
        figsize (list / tuple): Desired size of the image in inches (width, height)
        erange (list): Energy range for the DOS plot ([lower bound, upper bound])
        fontsize (float): Font size of the text in the figure.
        save (bool): Determines whether to automatically save the figure or not. If not 
            the figure and axis are return for further manipulation.

    Returns:
        If save == True, this function will return nothing and directly save the image as
        the output name. If save == False, the function will return the matplotlib figure
        and axis for further editing. 
    """

    dos_up = Dos(shift_efermi=shift_efermi, folder=folder, spin='up', soc_axis=soc_axis)
    dos_down = Dos(shift_efermi=shift_efermi, folder=folder, spin='down', soc_axis=soc_axis)

    fig = plt.figure(figsize=figsize, dpi=400)
    ax = fig.add_subplot(111)
    _figure_setup_dos(ax=ax, fontsize=fontsize, energyaxis=energyaxis)

    dos_up.plot_atoms(
        ax=ax,
        atoms=atoms,
        fill=fill,
        alpha=alpha,
        linewidth=linewidth,
        sigma=sigma,
        energyaxis=energyaxis,
        color_list=color_list,
        legend=legend,
        total=total,
        erange=erange,
    )

    dos_down.plot_atoms(
        ax=ax,
        atoms=atoms,
        fill=fill,
        alpha=alpha,
        linewidth=linewidth,
        sigma=sigma,
        energyaxis=energyaxis,
        color_list=color_list,
        legend=False,
        total=total,
        erange=erange,
    )

    plt.tight_layout(pad=0.4)

    if save:
        plt.savefig(output)
    else:
        return fig, ax

def dos_atom_spd_spin_polarized(
    folder,
    atom_spd_dict,
    output='dos_atom_spd_sp.png',
    fill=True,
    alpha=0.3,
    linewidth=1.5,
    sigma=0.05,
    energyaxis='x',
    soc_axis='z',
    color_list=None,
    legend=True,
    total=True,
    figsize=(4, 3),
    erange=[-6, 6],
    fontsize=12,
    save=True,
    shift_efermi=0,
):
    """
    This function plots a spin polarized atom projected density of states of the s, p, and d orbitals.

    Parameters:
        folder (str): This is the folder that contains the VASP files
        atom_spd_dict (dict[int:str]): A dictionary that contains the individual atoms and the corresponding 
            orbitals to project onto. For example, if the user wants to project onto the s, p, d orbitals
            of the first atom and the p orbitals of the second atom then the dictionary would be {0:'spd', 1:'p'}
        output (str): File name of the resulting plot.
        fill (bool): Determines wether or not to fill underneath the plot
        alpha (float): Alpha value for the fill
        linewidth (float): Linewidth of lines
        sigma (float): Standard deviation for gaussian filter
        energyaxis (str): Determines the axis to plot the energy on ('x' or 'y')
        color_list (list): List of colors that is the same length as the number of projections
            in the plot.
        legend (bool): Determines whether to draw the legend or not
        total (bool): Determines wheth to draw the total density of states or not
        spin (str): Which spin direction to parse ('up' or 'down')
        figsize (list / tuple): Desired size of the image in inches (width, height)
        erange (list): Energy range for the DOS plot ([lower bound, upper bound])
        combination_method (str): If spin == 'both', this determines if the spin up and spin down
            desnities are added or subtracted. ('add' or 'sub')
        fontsize (float): Font size of the text in the figure.
        save (bool): Determines whether to automatically save the figure or not. If not 
            the figure and axis are return for further manipulation.

    Returns:
        If save == True, this function will return nothing and directly save the image as
        the output name. If save == False, the function will return the matplotlib figure
        and axis for further editing. 
    """

    dos_up = Dos(shift_efermi=shift_efermi, folder=folder, spin='up', soc_axis=soc_axis)
    dos_down = Dos(shift_efermi=shift_efermi, folder=folder, spin='down', soc_axis=soc_axis)

    fig = plt.figure(figsize=figsize, dpi=400)
    ax = fig.add_subplot(111)
    _figure_setup_dos(ax=ax, fontsize=fontsize, energyaxis=energyaxis)

    dos_up.plot_atom_spd(
        ax=ax,
        atom_spd_dict=atom_spd_dict,
        fill=fill,
        alpha=alpha,
        linewidth=linewidth,
        sigma=sigma,
        energyaxis=energyaxis,
        color_list=color_list,
        legend=legend,
        total=total,
        erange=erange,
    )

    dos_down.plot_atom_spd(
        ax=ax,
        atom_spd_dict=atom_spd_dict,
        fill=fill,
        alpha=alpha,
        linewidth=linewidth,
        sigma=sigma,
        energyaxis=energyaxis,
        color_list=color_list,
        legend=False,
        total=total,
        erange=erange,
    )

    plt.tight_layout(pad=0.4)

    if save:
        plt.savefig(output)
    else:
        return fig, ax


def dos_elements_spin_polarized(
    folder,
    elements,
    output='dos_elements_sp.png',
    fill=True,
    alpha=0.3,
    linewidth=1.5,
    sigma=0.05,
    energyaxis='y',
    soc_axis='z',
    color_list=None,
    legend=True,
    total=True,
    figsize=(4, 3),
    erange=[-6, 6],
    fontsize=12,
    save=True,
    shift_efermi=0,
):
    """
    This function plots a spin polarized element projected density of states.

    Parameters:
        folder (str): This is the folder that contains the VASP files
        elements (list): List of elements to project onto. The list should countain the corresponding element symbols
        output (str): File name of the resulting plot.
        fill (bool): Determines wether or not to fill underneath the plot
        alpha (float): Alpha value for the fill
        linewidth (float): Linewidth of lines
        sigma (float): Standard deviation for gaussian filter
        energyaxis (str): Determines the axis to plot the energy on ('x' or 'y')
        color_list (list): List of colors that is the same length as the number of projections
            in the plot.
        legend (bool): Determines whether to draw the legend or not
        total (bool): Determines wheth to draw the total density of states or not
        spin (str): Which spin direction to parse ('up' or 'down')
        figsize (list / tuple): Desired size of the image in inches (width, height)
        erange (list): Energy range for the DOS plot ([lower bound, upper bound])
        combination_method (str): If spin == 'both', this determines if the spin up and spin down
            desnities are added or subtracted. ('add' or 'sub')
        fontsize (float): Font size of the text in the figure.
        save (bool): Determines whether to automatically save the figure or not. If not 
            the figure and axis are return for further manipulation.

    Returns:
        If save == True, this function will return nothing and directly save the image as
        the output name. If save == False, the function will return the matplotlib figure
        and axis for further editing. 
    """

    dos_up = Dos(shift_efermi=shift_efermi, folder=folder, spin='up', soc_axis=soc_axis)
    dos_down = Dos(shift_efermi=shift_efermi, folder=folder, spin='down', soc_axis=soc_axis)

    fig = plt.figure(figsize=figsize, dpi=400)
    ax = fig.add_subplot(111)
    _figure_setup_dos(ax=ax, fontsize=fontsize, energyaxis=energyaxis)

    dos_up.plot_elements(
        ax=ax,
        elements=elements,
        fill=fill,
        alpha=alpha,
        linewidth=linewidth,
        sigma=sigma,
        energyaxis=energyaxis,
        color_list=color_list,
        legend=legend,
        total=total,
        erange=erange,
    )

    dos_down.plot_elements(
        ax=ax,
        elements=elements,
        fill=fill,
        alpha=alpha,
        linewidth=linewidth,
        sigma=sigma,
        energyaxis=energyaxis,
        color_list=color_list,
        legend=False,
        total=total,
        erange=erange,
    )

    plt.tight_layout(pad=0.4)

    if save:
        plt.savefig(output)
    else:
        return fig, ax


def dos_element_spd_spin_polarized(
    folder,
    element_spd_dict,
    output='dos_element_spd_sp.png',
    fill=True,
    alpha=0.3,
    linewidth=1.5,
    sigma=0.05,
    energyaxis='x',
    soc_axis='z',
    color_list=None,
    legend=True,
    total=True,
    figsize=(4, 3),
    erange=[-6, 6],
    fontsize=12,
    save=True,
    shift_efermi=0,
):
    """
    This function plots a spin polarized element projected density of states of the s, p, and d orbitals.

    Parameters:
        folder (str): This is the folder that contains the VASP files
        element_spd_dict (dict[str:str]): A dictionary that contains the individual atoms and the corresponding 
            orbitals to project onto. For example, if the user wants to project onto the s, p, d orbitals
            of In and the p orbitals of As for an InAs structure then the dictionary would be {'In':'spd', 'As':'p'}
        output (str): File name of the resulting plot.
        fill (bool): Determines wether or not to fill underneath the plot
        alpha (float): Alpha value for the fill
        linewidth (float): Linewidth of lines
        sigma (float): Standard deviation for gaussian filter
        energyaxis (str): Determines the axis to plot the energy on ('x' or 'y')
        color_list (list): List of colors that is the same length as the number of projections
            in the plot.
        legend (bool): Determines whether to draw the legend or not
        total (bool): Determines wheth to draw the total density of states or not
        spin (str): Which spin direction to parse ('up' or 'down')
        figsize (list / tuple): Desired size of the image in inches (width, height)
        erange (list): Energy range for the DOS plot ([lower bound, upper bound])
        combination_method (str): If spin == 'both', this determines if the spin up and spin down
            desnities are added or subtracted. ('add' or 'sub')
        fontsize (float): Font size of the text in the figure.
        save (bool): Determines whether to automatically save the figure or not. If not 
            the figure and axis are return for further manipulation.

    Returns:
        If save == True, this function will return nothing and directly save the image as
        the output name. If save == False, the function will return the matplotlib figure
        and axis for further editing. 
    """

    dos_up = Dos(shift_efermi=shift_efermi, folder=folder, spin='up', soc_axis=soc_axis)
    dos_down = Dos(shift_efermi=shift_efermi, folder=folder, spin='down', soc_axis=soc_axis)

    fig = plt.figure(figsize=figsize, dpi=400)
    ax = fig.add_subplot(111)
    _figure_setup_dos(ax=ax, fontsize=fontsize, energyaxis=energyaxis)

    dos_up.plot_element_spd(
        ax=ax,
        element_spd_dict=element_spd_dict,
        fill=fill,
        alpha=alpha,
        linewidth=linewidth,
        sigma=sigma,
        energyaxis=energyaxis,
        color_list=color_list,
        legend=legend,
        total=total,
        erange=erange,
    )

    dos_down.plot_element_spd(
        ax=ax,
        element_spd_dict=element_spd_dict,
        fill=fill,
        alpha=alpha,
        linewidth=linewidth,
        sigma=sigma,
        energyaxis=energyaxis,
        color_list=color_list,
        legend=False,
        total=total,
        erange=erange,
    )

    plt.tight_layout(pad=0.4)

    if save:
        plt.savefig(output)
    else:
        return fig, ax


def dos_element_orbitals_spin_polarized(
    folder,
    element_orbital_dict,
    output='dos_element_orbitals_sp.png',
    fill=True,
    alpha=0.3,
    linewidth=1.5,
    sigma=0.05,
    energyaxis='y',
    soc_axis='z',
    color_list=None,
    legend=True,
    total=True,
    figsize=(4, 3),
    erange=[-6, 6],
    fontsize=12,
    save=True,
    shift_efermi=0,
):
    """
    This function plots a spin polarized element projected density of states on specific orbitals.

    Parameters:
        folder (str): This is the folder that contains the VASP files
        element_orbital_dict (dict[str:list]): A dictionary that contains the individual elements and the corresponding 
            orbitals to project onto. For example, if the user wants to project onto the s, py, pz, and px orbitals
            of In and the s orbital of As for and InAs structure then the dictionary would be {'In':[0,1,2,3], 'As':[0]}
        output (str): File name of the resulting plot.
        fill (bool): Determines wether or not to fill underneath the plot
        alpha (float): Alpha value for the fill
        linewidth (float): Linewidth of lines
        sigma (float): Standard deviation for gaussian filter
        energyaxis (str): Determines the axis to plot the energy on ('x' or 'y')
        color_list (list): List of colors that is the same length as the number of projections
            in the plot.
        legend (bool): Determines whether to draw the legend or not
        total (bool): Determines wheth to draw the total density of states or not
        spin (str): Which spin direction to parse ('up' or 'down')
        figsize (list / tuple): Desired size of the image in inches (width, height)
        erange (list): Energy range for the DOS plot ([lower bound, upper bound])
        combination_method (str): If spin == 'both', this determines if the spin up and spin down
            desnities are added or subtracted. ('add' or 'sub')
        fontsize (float): Font size of the text in the figure.
        save (bool): Determines whether to automatically save the figure or not. If not 
            the figure and axis are return for further manipulation.

    Returns:
        If save == True, this function will return nothing and directly save the image as
        the output name. If save == False, the function will return the matplotlib figure
        and axis for further editing. 
    """

    dos_up = Dos(shift_efermi=shift_efermi, folder=folder, spin='up', soc_axis=soc_axis)
    dos_down = Dos(shift_efermi=shift_efermi, folder=folder, spin='down', soc_axis=soc_axis)

    fig = plt.figure(figsize=figsize, dpi=400)
    ax = fig.add_subplot(111)
    _figure_setup_dos(ax=ax, fontsize=fontsize, energyaxis=energyaxis)

    dos_up.plot_element_orbitals(
        ax=ax,
        element_orbital_dict=element_orbital_dict,
        fill=fill,
        alpha=alpha,
        linewidth=linewidth,
        sigma=sigma,
        energyaxis=energyaxis,
        color_list=color_list,
        legend=legend,
        total=total,
        erange=erange,
    )

    dos_down.plot_element_orbitals(
        ax=ax,
        element_orbital_dict=element_orbital_dict,
        fill=fill,
        alpha=alpha,
        linewidth=linewidth,
        sigma=sigma,
        energyaxis=energyaxis,
        color_list=color_list,
        legend=False,
        total=total,
        erange=erange,
    )

    plt.tight_layout(pad=0.4)

    if save:
        plt.savefig(output)
    else:
        return fig, ax


# =============================================================
# ---------------------- Band-Dos Plots -----------------------
# =============================================================

def band_dos_plain(
    band_folder,
    dos_folder,
    output='band_dos_plain.png',
    spin='up',
    color='black',
    linewidth=1.25,
    linestyle='-',
    figsize=(6, 3),
    width_ratios=[7, 3],
    erange=[-6, 6],
    kpath=None,
    custom_kpath=None,
    n=None,
    unfold=False,
    M=None,
    high_symm_points=None,
    fontsize=12,
    save=True,
    shift_efermi=0,
    interpolate=False,
    new_n=200,
    fill=True,
    alpha=0.3,
    sigma=0.05,
):
    """
    This function plots a plain band structure and density of states next to eachother.

    Parameters:
        band_folder (str): This is the folder that contains the VASP files for the band structure calculation
        dos_folder (str): This is the folder that contains the VASP files for the density of states calculation
        output (str): File name of the resulting plot.
        spin (str): Choose which spin direction to parse. ('up' or 'down')
        color (str): Color of the band structure lines
        linewidth (float): Line width of the band structure lines
        linestyle (str): Line style of the bands
        figsize (list / tuple): Desired size of the image in inches (width, height)
        width_ratios (list / tuple): Width ration of the band plot and dos plot. 
        erange (list / tuple): Range of energy to show in the plot [low, high]
        kpath (list[list]): High symmetry k-point path of band structure calculation
            Due to the nature of the KPOINTS file for unfolded calculations this
            information is a required input for proper labeling of the figure
            for unfolded calculations. This information is extracted from the KPOINTS
            files for non-unfolded calculations. (G is automatically converted to \\Gamma)
	    (e.g. For the path X-G-X, kpath=[['X', 'G'], ['G', 'X']])
	custom_kpath (list): This gives the option to only plot specific segments of a given band structure
	    calculation. For example if the kpath was G-X-W-L then there are three segements to choose from:
	    G-X, X-W, and W-L. In this case the default kpath could be plotted by defining custom_kpath=[1,2,3],
	    where 1 -> G-X, 2 -> X-W, and 3 -> W-L. If only G-X and X-W were desired then custom_kpath=[1,2].
	    If one of the segements should be flipped it can be done by making its value negative
	    (e.g. -1 -> X-G, -2 -> W-X, -3 -> L-W)
        n (int): Number of points between each high symmetry points.
            This is also only required for unfolded calculations. This number should be 
            known by the user, as it was used to generate the KPOINTS file.
        fontsize (float): Font size of the text in the figure.
        save (bool): Determines whether to automatically save the figure or not. If not 
            the figure and axis are return for further manipulation.
        fill (bool): Determines wether or not to fill underneath the plot
        alpha (float): Alpha value for the fill
        sigma (float): Standard deviation for gaussian filter

    Returns:
        If save == True, this function will return nothing and directly save the image as
        the output name. If save == False, the function will return the matplotlib figure
        and axis for further editing. 
    """
    fig, ax = plt.subplots(
        nrows=1,
        ncols=2,
        sharey=True,
        figsize=figsize,
        dpi=400,
        gridspec_kw={'width_ratios': width_ratios}
    )

    ax1, ax2 = _figure_setup_band_dos(
        ax=ax,
        fontsize=fontsize,
        ylim=[erange[0], erange[1]]
    )

    band = Band(
        folder=band_folder,
        spin=spin,
        unfold=unfold,
        high_symm_points=high_symm_points, 
	interpolate=interpolate,
	new_n=new_n,
        kpath=kpath,
	custom_kpath=custom_kpath,
        n=n,
        M=M,
	shift_efermi=shift_efermi,
    )

    dos = Dos(shift_efermi=shift_efermi, folder=dos_folder, spin=spin)

    band.plot_plain(
        ax=ax1,
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
    )

    dos.plot_plain(
        ax=ax2,
        linewidth=linewidth,
        fill=fill,
        alpha=alpha,
        sigma=sigma,
        energyaxis='y',
        color=color,
        erange=erange,
    )

    fig.canvas.draw()
    labels = ax2.get_xticklabels()
    labels[0] = ''
    ax2.set_xticklabels(labels)

    plt.tight_layout(pad=0.4)
    plt.subplots_adjust(wspace=0)

    if save:
        plt.savefig(output)
    else:
        return fig, ax1, ax2


def band_dos_spd(
    band_folder,
    dos_folder,
    output='band_dos_spd.png',
    spin='up',
    scale_factor=5,
    orbitals='spd',
    color_list=None,
    legend=True,
    linewidth=0.75,
    band_color='black',
    figsize=(6, 3),
    width_ratios=[7, 3],
    erange=[-6, 6],
    kpath=None,
    custom_kpath=None,
    n=None,
    display_order=None,
    unfold=False,
    M=None,
    high_symm_points=None,
    fontsize=12,
    save=True,
    shift_efermi=0,
    interpolate=False,
    new_n=200,
    fill=True,
    alpha=0.3,
    sigma=0.05,
):
    """
    This function plots a s, p, d projected band structure next to and s, p, d projected
    density of states. 

    Parameters:
        band_folder (str): This is the folder that contains the VASP files for the band structure
        dos_folder (str): This is the folder that contains the VASP files for the density of states
        orbitals (str): String that contains the s, p, or d orbitals that to project onto.
            The default is 'spd', if the user only wanted to project onto the p, and d orbitals 
            than 'pd' should be passed in
        output (str): File name of the resulting plot.
        spin (str): Choose which spin direction to parse. ('up' or 'down')
        scale_factor (float): Factor to scale weights. This changes the size of the
            points in the scatter plot
        display_order (str / None): If None, the projections will be displayed in the same order
            the user inputs them. If 'all' the projections will be plotted from largest to smallest
            so every point is visable. If 'dominant' the projections will be plotted from smallest 
            to largest so only the dominant projection is shown.
        color_list (list): List of colors that is the same length as the number of projections
            in the plot.
        legend (bool): Determines if the legend should be included or not.
        linewidth (float): Line width of the plain band structure plotted in the background
        band_color (string): Color of the plain band structure
        figsize (list / tuple): Desired size of the image in inches (width, height)
        width_ratios (list / tuple): Width ration of the band plot and dos plot. 
        erange (list / tuple): Range of energy to show in the plot [low, high]
        kpath (list[list]): High symmetry k-point path of band structure calculation
            Due to the nature of the KPOINTS file for unfolded calculations this
            information is a required input for proper labeling of the figure
            for unfolded calculations. This information is extracted from the KPOINTS
            files for non-unfolded calculations. (G is automatically converted to \\Gamma)
	    (e.g. For the path X-G-X, kpath=[['X', 'G'], ['G', 'X']])
	custom_kpath (list): This gives the option to only plot specific segments of a given band structure
	    calculation. For example if the kpath was G-X-W-L then there are three segements to choose from:
	    G-X, X-W, and W-L. In this case the default kpath could be plotted by defining custom_kpath=[1,2,3],
	    where 1 -> G-X, 2 -> X-W, and 3 -> W-L. If only G-X and X-W were desired then custom_kpath=[1,2].
	    If one of the segements should be flipped it can be done by making its value negative
	    (e.g. -1 -> X-G, -2 -> W-X, -3 -> L-W)
        n (int): Number of points between each high symmetry points.
            This is also only required for unfolded calculations and band unfolding. This number should be 
            known by the user, as it was used to generate the KPOINTS file.
        unfold (bool): Determines if the plotted band structure is from a band unfolding calculation.
        M (list[list]): Transformation matrix from the primitive bulk structure to the slab structure.
            Only required for a band unfolding calculation.
        high_symm_points (list[list]): List of fractional coordinated for each high symmetry point in 
            the band structure path. Only required for a band unfolding calculation.
        fontsize (float): Font size of the text in the figure.
        fill (bool): Determines wether or not to fill underneath the plot
        alpha (float): Alpha value for the fill
        sigma (float): Standard deviation for gaussian filter
        save (bool): Determines whether to automatically save the figure or not. If not 
            the figure and axis are return for further manipulation.

    Returns:
        If save == True, this function will return nothing and directly save the image as
        the output name. If save == False, the function will return the matplotlib figure
        and axis for further editing. 
    """

    fig, ax = plt.subplots(
        nrows=1,
        ncols=2,
        sharey=True,
        figsize=figsize,
        dpi=400,
        gridspec_kw={'width_ratios': width_ratios}
    )

    ax1, ax2 = _figure_setup_band_dos(
        ax=ax,
        fontsize=fontsize,
        ylim=[erange[0], erange[1]]
    )

    band = Band(
        folder=band_folder,
        spin=spin,
        projected=True,
        unfold=unfold,
        high_symm_points=high_symm_points, 
	interpolate=interpolate,
	new_n=new_n,
        kpath=kpath,
	custom_kpath=custom_kpath,
        n=n,
        M=M,
	shift_efermi=shift_efermi,
    )

    dos = Dos(shift_efermi=shift_efermi, folder=dos_folder, spin=spin)

    band.plot_spd(
        ax=ax1,
        scale_factor=scale_factor,
        orbitals=orbitals,
        erange=erange,
        display_order=display_order,
        color_list=color_list,
        legend=False,
        linewidth=linewidth,
        band_color=band_color,
    )

    dos.plot_spd(
        ax=ax2,
        orbitals=orbitals,
        fill=fill,
        alpha=alpha,
        linewidth=linewidth,
        sigma=sigma,
        energyaxis='y',
        color_list=color_list,
        legend=legend,
        total=True,
        erange=erange,
    )

    fig.canvas.draw()
    nbins = len(ax2.get_xticklabels())
    ax2.xaxis.set_major_locator(MaxNLocator(nbins=nbins - 1, prune='lower'))

    plt.tight_layout(pad=0.4)
    plt.subplots_adjust(wspace=0)

    if save:
        plt.savefig(output)
    else:
        return fig, ax1, ax2


def band_dos_atom_orbitals(
    band_folder,
    dos_folder,
    atom_orbital_dict,
    output='band_dos_atom_orbitals.png',
    spin='up',
    scale_factor=5,
    color_list=None,
    legend=True,
    linewidth=0.75,
    band_color='black',
    figsize=(6, 3),
    width_ratios=[7, 3],
    erange=[-6, 6],
    kpath=None,
    custom_kpath=None,
    n=None,
    display_order=None,
    unfold=False,
    M=None,
    high_symm_points=None,
    fontsize=12,
    save=True,
    shift_efermi=0,
    interpolate=False,
    new_n=200,
    fill=True,
    alpha=0.3,
    sigma=0.05,
):
    """
    This function generates an [atom, orbital] projected band structure next to
    a projected density of states of the same [atom, orbitals] pairs.

    Parameters:
        band_folder (str): This is the folder that contains the VASP files for the band structure
        dos_folder (str): This is the folder that contains the VASP files for the density of states
        atom_orbital_dict (dict[int:list]): A dictionary that contains the individual atoms and the
            corresponding orbitals to project onto. For example, if the user wants to project onto the s, py, pz, and px orbitals
            of the first atom and the s orbital of the second atom then the dictionary would be {0:[0,1,2,3], 1:[0]}
        output (str): File name of the resulting plot.
        spin (str): Choose which spin direction to parse. ('up' or 'down')
        scale_factor (float): Factor to scale weights. This changes the size of the
            points in the scatter plot
        display_order (str / None): If None, the projections will be displayed in the same order
            the user inputs them. If 'all' the projections will be plotted from largest to smallest
            so every point is visable. If 'dominant' the projections will be plotted from smallest 
            to largest so only the dominant projection is shown.
        color_list (list): List of colors that is the same length as the number of projections
            in the plot.
        legend (bool): Determines if the legend should be included or not.
        linewidth (float): Line width of the plain band structure plotted in the background
        band_color (string): Color of the plain band structure
        figsize (list / tuple): Desired size of the image in inches (width, height)
        width_ratios (list / tuple): Width ration of the band plot and dos plot. 
        erange (list / tuple): Range of energy to show in the plot [low, high]
        kpath (list[list]): High symmetry k-point path of band structure calculation
            Due to the nature of the KPOINTS file for unfolded calculations this
            information is a required input for proper labeling of the figure
            for unfolded calculations. This information is extracted from the KPOINTS
            files for non-unfolded calculations. (G is automatically converted to \\Gamma)
	    (e.g. For the path X-G-X, kpath=[['X', 'G'], ['G', 'X']])
	custom_kpath (list): This gives the option to only plot specific segments of a given band structure
	    calculation. For example if the kpath was G-X-W-L then there are three segements to choose from:
	    G-X, X-W, and W-L. In this case the default kpath could be plotted by defining custom_kpath=[1,2,3],
	    where 1 -> G-X, 2 -> X-W, and 3 -> W-L. If only G-X and X-W were desired then custom_kpath=[1,2].
	    If one of the segements should be flipped it can be done by making its value negative
	    (e.g. -1 -> X-G, -2 -> W-X, -3 -> L-W)
        n (int): Number of points between each high symmetry points.
            This is also only required for unfolded calculations and band unfolding. This number should be 
            known by the user, as it was used to generate the KPOINTS file.
        unfold (bool): Determines if the plotted band structure is from a band unfolding calculation.
        M (list[list]): Transformation matrix from the primitive bulk structure to the slab structure.
            Only required for a band unfolding calculation.
        high_symm_points (list[list]): List of fractional coordinated for each high symmetry point in 
            the band structure path. Only required for a band unfolding calculation.
        fontsize (float): Font size of the text in the figure.
        fill (bool): Determines wether or not to fill underneath the plot
        alpha (float): Alpha value for the fill
        sigma (float): Standard deviation for gaussian filter
        save (bool): Determines whether to automatically save the figure or not. If not 
            the figure and axis are return for further manipulation.

    Returns:
        If save == True, this function will return nothing and directly save the image as
        the output name. If save == False, the function will return the matplotlib figure
        and axis for further editing. 
    """

    fig, ax = plt.subplots(
        nrows=1,
        ncols=2,
        sharey=True,
        figsize=figsize,
        dpi=400,
        gridspec_kw={'width_ratios': width_ratios}
    )

    ax1, ax2 = _figure_setup_band_dos(
        ax=ax,
        fontsize=fontsize,
        ylim=[erange[0], erange[1]]
    )

    band = Band(
        folder=band_folder,
        spin=spin,
        projected=True,
        unfold=unfold,
        high_symm_points=high_symm_points, 
	interpolate=interpolate,
	new_n=new_n,
        kpath=kpath,
	custom_kpath=custom_kpath,
        n=n,
        M=M,
	shift_efermi=shift_efermi,
    )

    dos = Dos(shift_efermi=shift_efermi, folder=dos_folder, spin=spin)

    band.plot_atom_orbitals(
        ax=ax1,
        scale_factor=scale_factor,
        atom_orbital_dict=atom_orbital_dict,
        color_list=color_list,
        legend=False,
        erange=erange,
        display_order=display_order,
        linewidth=linewidth,
        band_color=band_color,
    )

    dos.plot_atom_orbitals(
        ax=ax2,
        atom_orbital_dict=atom_orbital_dict,
        fill=fill,
        alpha=alpha,
        linewidth=linewidth,
        sigma=sigma,
        energyaxis='y',
        color_list=color_list,
        legend=legend,
        total=True,
        erange=erange,
    )

    fig.canvas.draw()
    nbins = len(ax2.get_xticklabels())
    ax2.xaxis.set_major_locator(MaxNLocator(nbins=nbins - 1, prune='lower'))

    plt.tight_layout(pad=0.4)
    plt.subplots_adjust(wspace=0)

    if save:
        plt.savefig(output)
    else:
        return fig, ax1, ax2


def band_dos_orbitals(
    band_folder,
    dos_folder,
    orbitals,
    output='band_dos_orbitals.png',
    spin='up',
    scale_factor=5,
    color_list=None,
    legend=True,
    linewidth=0.75,
    band_color='black',
    figsize=(6, 3),
    width_ratios=[7, 3],
    erange=[-6, 6],
    kpath=None,
    custom_kpath=None,
    n=None,
    display_order=None,
    unfold=False,
    M=None,
    high_symm_points=None,
    fontsize=12,
    save=True,
    shift_efermi=0,
    interpolate=False,
    new_n=200,
    fill=True,
    alpha=0.3,
    sigma=0.05,
):
    """
    This function generates a band structure projected on specific orbitals next to a
    projected density of states on the same orbitals. 

    Parameters:
        band_folder (str): This is the folder that contains the VASP files for the band structure
        dos_folder (str): This is the folder that contains the VASP files for the density of states
        orbitals (list): List of orbitals to compare

            | 0 = s
            | 1 = py
            | 2 = pz
            | 3 = px
            | 4 = dxy
            | 5 = dyz
            | 6 = dz2
            | 7 = dxz
            | 8 = dx2-y2
            | 9 = fy3x2
            | 10 = fxyz
            | 11 = fyz2
            | 12 = fz3
            | 13 = fxz2
            | 14 = fzx3
            | 15 = fx3

        output (str): File name of the resulting plot.
        spin (str): Choose which spin direction to parse. ('up' or 'down')
        scale_factor (float): Factor to scale weights. This changes the size of the
            points in the scatter plot
        display_order (str / None): If None, the projections will be displayed in the same order
            the user inputs them. If 'all' the projections will be plotted from largest to smallest
            so every point is visable. If 'dominant' the projections will be plotted from smallest 
            to largest so only the dominant projection is shown.
        color_list (list): List of colors that is the same length as the number of projections
            in the plot.
        legend (bool): Determines if the legend should be included or not.
        linewidth (float): Line width of the plain band structure plotted in the background
        band_color (string): Color of the plain band structure
        figsize (list / tuple): Desired size of the image in inches (width, height)
        width_ratios (list / tuple): Width ration of the band plot and dos plot. 
        erange (list / tuple): Range of energy to show in the plot [low, high]
        kpath (list[list]): High symmetry k-point path of band structure calculation
            Due to the nature of the KPOINTS file for unfolded calculations this
            information is a required input for proper labeling of the figure
            for unfolded calculations. This information is extracted from the KPOINTS
            files for non-unfolded calculations. (G is automatically converted to \\Gamma)
	    (e.g. For the path X-G-X, kpath=[['X', 'G'], ['G', 'X']])
	custom_kpath (list): This gives the option to only plot specific segments of a given band structure
	    calculation. For example if the kpath was G-X-W-L then there are three segements to choose from:
	    G-X, X-W, and W-L. In this case the default kpath could be plotted by defining custom_kpath=[1,2,3],
	    where 1 -> G-X, 2 -> X-W, and 3 -> W-L. If only G-X and X-W were desired then custom_kpath=[1,2].
	    If one of the segements should be flipped it can be done by making its value negative
	    (e.g. -1 -> X-G, -2 -> W-X, -3 -> L-W)
        n (int): Number of points between each high symmetry points.
            This is also only required for unfolded calculations and band unfolding. This number should be 
            known by the user, as it was used to generate the KPOINTS file.
        unfold (bool): Determines if the plotted band structure is from a band unfolding calculation.
        M (list[list]): Transformation matrix from the primitive bulk structure to the slab structure.
            Only required for a band unfolding calculation.
        high_symm_points (list[list]): List of fractional coordinated for each high symmetry point in 
            the band structure path. Only required for a band unfolding calculation.
        fontsize (float): Font size of the text in the figure.
        fill (bool): Determines wether or not to fill underneath the plot
        alpha (float): Alpha value for the fill
        sigma (float): Standard deviation for gaussian filter
        save (bool): Determines whether to automatically save the figure or not. If not 
            the figure and axis are return for further manipulation.

    Returns:
        If save == True, this function will return nothing and directly save the image as
        the output name. If save == False, the function will return the matplotlib figure
        and axis for further editing. 
    """

    fig, ax = plt.subplots(
        nrows=1,
        ncols=2,
        sharey=True,
        figsize=figsize,
        dpi=400,
        gridspec_kw={'width_ratios': width_ratios}
    )

    ax1, ax2 = _figure_setup_band_dos(
        ax=ax,
        fontsize=fontsize,
        ylim=[erange[0], erange[1]]
    )

    band = Band(
        folder=band_folder,
        spin=spin,
        projected=True,
        unfold=unfold,
        high_symm_points=high_symm_points, 
	interpolate=interpolate,
	new_n=new_n,
        kpath=kpath,
	custom_kpath=custom_kpath,
        n=n,
        M=M,
	shift_efermi=shift_efermi,
    )

    dos = Dos(shift_efermi=shift_efermi, folder=dos_folder, spin=spin)

    band.plot_orbitals(
        ax=ax1,
        orbitals=orbitals,
        scale_factor=scale_factor,
        color_list=color_list,
        erange=erange,
        display_order=display_order,
        legend=False,
        linewidth=linewidth,
        band_color=band_color,
    )

    dos.plot_orbitals(
        ax=ax2,
        orbitals=orbitals,
        fill=fill,
        alpha=alpha,
        linewidth=linewidth,
        sigma=sigma,
        energyaxis='y',
        color_list=color_list,
        legend=legend,
        total=True,
        erange=erange,
    )

    fig.canvas.draw()
    nbins = len(ax2.get_xticklabels())
    ax2.xaxis.set_major_locator(MaxNLocator(nbins=nbins - 1, prune='lower'))

    plt.tight_layout(pad=0.4)
    plt.subplots_adjust(wspace=0)

    if save:
        plt.savefig(output)
    else:
        return fig, ax1, ax2


def band_dos_atoms(
    band_folder,
    dos_folder,
    atoms,
    output='band_dos_atoms.png',
    spin='up',
    scale_factor=5,
    color_list=None,
    legend=True,
    linewidth=0.75,
    band_color='black',
    figsize=(6, 3),
    width_ratios=[7, 3],
    erange=[-6, 6],
    kpath=None,
    custom_kpath=None,
    n=None,
    display_order=None,
    unfold=False,
    M=None,
    high_symm_points=None,
    fontsize=12,
    save=True,
    shift_efermi=0,
    interpolate=False,
    new_n=200,
    fill=True,
    alpha=0.3,
    sigma=0.05,
):
    """
    This function generates a projected band structure on specific atoms next to a 
    projected density of states on the same atoms.

    Parameters:
        band_folder (str): This is the folder that contains the VASP files for the band structure
        dos_folder (str): This is the folder that contains the VASP files for the density of states
        atoms (list): List of atoms to project onto. The indices should be zero indexed (first atom is 0)
            and the atoms are in the same order as they are in the POSCAR
        output (str): File name of the resulting plot.
        spin (str): Choose which spin direction to parse. ('up' or 'down')
        scale_factor (float): Factor to scale weights. This changes the size of the
            points in the scatter plot
        display_order (str / None): If None, the projections will be displayed in the same order
            the user inputs them. If 'all' the projections will be plotted from largest to smallest
            so every point is visable. If 'dominant' the projections will be plotted from smallest 
            to largest so only the dominant projection is shown.
        color_list (list): List of colors that is the same length as the number of projections
            in the plot.
        legend (bool): Determines if the legend should be included or not.
        linewidth (float): Line width of the plain band structure plotted in the background
        band_color (string): Color of the plain band structure
        figsize (list / tuple): Desired size of the image in inches (width, height)
        width_ratios (list / tuple): Width ration of the band plot and dos plot. 
        erange (list / tuple): Range of energy to show in the plot [low, high]
        kpath (list[list]): High symmetry k-point path of band structure calculation
            Due to the nature of the KPOINTS file for unfolded calculations this
            information is a required input for proper labeling of the figure
            for unfolded calculations. This information is extracted from the KPOINTS
            files for non-unfolded calculations. (G is automatically converted to \\Gamma)
	    (e.g. For the path X-G-X, kpath=[['X', 'G'], ['G', 'X']])
	custom_kpath (list): This gives the option to only plot specific segments of a given band structure
	    calculation. For example if the kpath was G-X-W-L then there are three segements to choose from:
	    G-X, X-W, and W-L. In this case the default kpath could be plotted by defining custom_kpath=[1,2,3],
	    where 1 -> G-X, 2 -> X-W, and 3 -> W-L. If only G-X and X-W were desired then custom_kpath=[1,2].
	    If one of the segements should be flipped it can be done by making its value negative
	    (e.g. -1 -> X-G, -2 -> W-X, -3 -> L-W)
        n (int): Number of points between each high symmetry points.
            This is also only required for unfolded calculations and band unfolding. This number should be 
            known by the user, as it was used to generate the KPOINTS file.
        unfold (bool): Determines if the plotted band structure is from a band unfolding calculation.
        M (list[list]): Transformation matrix from the primitive bulk structure to the slab structure.
            Only required for a band unfolding calculation.
        high_symm_points (list[list]): List of fractional coordinated for each high symmetry point in 
            the band structure path. Only required for a band unfolding calculation.
        fontsize (float): Font size of the text in the figure.
        fill (bool): Determines wether or not to fill underneath the plot
        alpha (float): Alpha value for the fill
        sigma (float): Standard deviation for gaussian filter
        save (bool): Determines whether to automatically save the figure or not. If not 
            the figure and axis are return for further manipulation.

    Returns:
        If save == True, this function will return nothing and directly save the image as
        the output name. If save == False, the function will return the matplotlib figure
        and axis for further editing. 
    """

    fig, ax = plt.subplots(
        nrows=1,
        ncols=2,
        sharey=True,
        figsize=figsize,
        dpi=400,
        gridspec_kw={'width_ratios': width_ratios}
    )

    ax1, ax2 = _figure_setup_band_dos(
        ax=ax,
        fontsize=fontsize,
        ylim=[erange[0], erange[1]]
    )

    band = Band(
        folder=band_folder,
        spin=spin,
        projected=True,
        unfold=unfold,
        high_symm_points=high_symm_points, 
	interpolate=interpolate,
	new_n=new_n,
        kpath=kpath,
	custom_kpath=custom_kpath,
        n=n,
        M=M,
	shift_efermi=shift_efermi,
    )

    dos = Dos(shift_efermi=shift_efermi, folder=dos_folder, spin=spin)

    band.plot_atoms(
        ax=ax1,
        atoms=atoms,
        scale_factor=scale_factor,
        color_list=color_list,
        erange=erange,
        display_order=display_order,
        legend=False,
        linewidth=linewidth,
        band_color=band_color,
    )

    dos.plot_atoms(
        ax=ax2,
        atoms=atoms,
        fill=fill,
        alpha=alpha,
        linewidth=linewidth,
        sigma=sigma,
        energyaxis='y',
        color_list=color_list,
        legend=legend,
        total=True,
        erange=erange,
    )

    fig.canvas.draw()
    nbins = len(ax2.get_xticklabels())
    ax2.xaxis.set_major_locator(MaxNLocator(nbins=nbins - 1, prune='lower'))

    plt.tight_layout(pad=0.4)
    plt.subplots_adjust(wspace=0)

    if save:
        plt.savefig(output)
    else:
        return fig, ax1, ax2

def band_dos_atom_spd(
    band_folder,
    dos_folder,
    atom_spd_dict,
    output='band_dos_atom_spd.png',
    spin='up',
    scale_factor=5,
    color_list=None,
    legend=True,
    linewidth=0.75,
    band_color='black',
    figsize=(6, 3),
    width_ratios=[7, 3],
    erange=[-6, 6],
    kpath=None,
    custom_kpath=None,
    n=None,
    display_order=None,
    unfold=False,
    M=None,
    high_symm_points=None,
    fontsize=12,
    save=True,
    shift_efermi=0,
    interpolate=False,
    new_n=200,
    fill=True,
    alpha=0.3,
    sigma=0.05,
):
    """
    This function generates a s, p, d projected band structure on specific atoms next to a 
    projected density of states on the s, p, d orbitals for the same atoms.

    Parameters:
        band_folder (str): This is the folder that contains the VASP files for the band structure
        dos_folder (str): This is the folder that contains the VASP files for the density of states
        atom_spd_dict (dict[int:str]): A dictionary that contains the individual atoms and the corresponding 
            orbitals to project onto. For example, if the user wants to project onto the s, p, d orbitals
            of the first atom and the p orbitals of the second atom then the dictionary would be {0:'spd', 1:'p'}
        output (str): File name of the resulting plot.
        spin (str): Choose which spin direction to parse. ('up' or 'down')
        scale_factor (float): Factor to scale weights. This changes the size of the
            points in the scatter plot
        display_order (str / None): If None, the projections will be displayed in the same order
            the user inputs them. If 'all' the projections will be plotted from largest to smallest
            so every point is visable. If 'dominant' the projections will be plotted from smallest 
            to largest so only the dominant projection is shown.
        color_list (list): List of colors that is the same length as the number of projections
            in the plot.
        legend (bool): Determines if the legend should be included or not.
        linewidth (float): Line width of the plain band structure plotted in the background
        band_color (string): Color of the plain band structure
        figsize (list / tuple): Desired size of the image in inches (width, height)
        width_ratios (list / tuple): Width ration of the band plot and dos plot. 
        erange (list / tuple): Range of energy to show in the plot [low, high]
        kpath (list[list]): High symmetry k-point path of band structure calculation
            Due to the nature of the KPOINTS file for unfolded calculations this
            information is a required input for proper labeling of the figure
            for unfolded calculations. This information is extracted from the KPOINTS
            files for non-unfolded calculations. (G is automatically converted to \\Gamma)
	    (e.g. For the path X-G-X, kpath=[['X', 'G'], ['G', 'X']])
	custom_kpath (list): This gives the option to only plot specific segments of a given band structure
	    calculation. For example if the kpath was G-X-W-L then there are three segements to choose from:
	    G-X, X-W, and W-L. In this case the default kpath could be plotted by defining custom_kpath=[1,2,3],
	    where 1 -> G-X, 2 -> X-W, and 3 -> W-L. If only G-X and X-W were desired then custom_kpath=[1,2].
	    If one of the segements should be flipped it can be done by making its value negative
	    (e.g. -1 -> X-G, -2 -> W-X, -3 -> L-W)
        n (int): Number of points between each high symmetry points.
            This is also only required for unfolded calculations and band unfolding. This number should be 
            known by the user, as it was used to generate the KPOINTS file.
        unfold (bool): Determines if the plotted band structure is from a band unfolding calculation.
        M (list[list]): Transformation matrix from the primitive bulk structure to the slab structure.
            Only required for a band unfolding calculation.
        high_symm_points (list[list]): List of fractional coordinated for each high symmetry point in 
            the band structure path. Only required for a band unfolding calculation.
        fontsize (float): Font size of the text in the figure.
        fill (bool): Determines wether or not to fill underneath the plot
        alpha (float): Alpha value for the fill
        sigma (float): Standard deviation for gaussian filter
        save (bool): Determines whether to automatically save the figure or not. If not 
            the figure and axis are return for further manipulation.

    Returns:
        If save == True, this function will return nothing and directly save the image as
        the output name. If save == False, the function will return the matplotlib figure
        and axis for further editing. 
    """

    fig, ax = plt.subplots(
        nrows=1,
        ncols=2,
        sharey=True,
        figsize=figsize,
        dpi=400,
        gridspec_kw={'width_ratios': width_ratios}
    )

    ax1, ax2 = _figure_setup_band_dos(
        ax=ax,
        fontsize=fontsize,
        ylim=[erange[0], erange[1]]
    )

    band = Band(
        folder=band_folder,
        spin=spin,
        projected=True,
        unfold=unfold,
        high_symm_points=high_symm_points, 
	interpolate=interpolate,
	new_n=new_n,
        kpath=kpath,
	custom_kpath=custom_kpath,
        n=n,
        M=M,
	shift_efermi=shift_efermi,
    )

    dos = Dos(shift_efermi=shift_efermi, folder=dos_folder, spin=spin)

    band.plot_atom_spd(
        ax=ax1,
        atom_spd_dict=atom_spd_dict,
        display_order=display_order,
        scale_factor=scale_factor,
        color_list=color_list,
        legend=False,
        linewidth=linewidth,
        band_color=band_color,
        erange=erange,
    )

    dos.plot_atom_spd(
        ax=ax2,
        atom_spd_dict=atom_spd_dict,
        fill=fill,
        alpha=alpha,
        linewidth=linewidth,
        sigma=sigma,
        energyaxis='y',
        color_list=color_list,
        legend=legend,
        total=True,
        erange=erange,
    )

    fig.canvas.draw()
    nbins = len(ax2.get_xticklabels())
    ax2.xaxis.set_major_locator(MaxNLocator(nbins=nbins - 1, prune='lower'))

    plt.tight_layout(pad=0.4)
    plt.subplots_adjust(wspace=0)

    if save:
        plt.savefig(output)
    else:
        return fig, ax1, ax2

def band_dos_elements(
    band_folder,
    dos_folder,
    elements,
    output='band_dos_elements.png',
    spin='up',
    scale_factor=5,
    color_list=None,
    legend=True,
    linewidth=0.75,
    band_color='black',
    figsize=(6, 3),
    width_ratios=[7, 3],
    erange=[-6, 6],
    kpath=None,
    custom_kpath=None,
    n=None,
    display_order=None,
    unfold=False,
    M=None,
    high_symm_points=None,
    fontsize=12,
    save=True,
    shift_efermi=0,
    interpolate=False,
    new_n=200,
    fill=True,
    alpha=0.3,
    sigma=0.05,
):
    """
    This function generates a projected band structure on specific elements next to a projected
    density of states on the same elements.

    Parameters:
        band_folder (str): This is the folder that contains the VASP files for the band structure
        dos_folder (str): This is the folder that contains the VASP files for the density of states
        elements (list): List of elements to project onto. The list should countain the corresponding element symbols
        output (str): File name of the resulting plot.
        spin (str): Choose which spin direction to parse. ('up' or 'down')
        scale_factor (float): Factor to scale weights. This changes the size of the
            points in the scatter plot
        display_order (str / None): If None, the projections will be displayed in the same order
            the user inputs them. If 'all' the projections will be plotted from largest to smallest
            so every point is visable. If 'dominant' the projections will be plotted from smallest 
            to largest so only the dominant projection is shown.
        color_list (list): List of colors that is the same length as the number of projections
            in the plot.
        legend (bool): Determines if the legend should be included or not.
        linewidth (float): Line width of the plain band structure plotted in the background
        band_color (string): Color of the plain band structure
        figsize (list / tuple): Desired size of the image in inches (width, height)
        width_ratios (list / tuple): Width ration of the band plot and dos plot. 
        erange (list / tuple): Range of energy to show in the plot [low, high]
        kpath (list[list]): High symmetry k-point path of band structure calculation
            Due to the nature of the KPOINTS file for unfolded calculations this
            information is a required input for proper labeling of the figure
            for unfolded calculations. This information is extracted from the KPOINTS
            files for non-unfolded calculations. (G is automatically converted to \\Gamma)
	    (e.g. For the path X-G-X, kpath=[['X', 'G'], ['G', 'X']])
	custom_kpath (list): This gives the option to only plot specific segments of a given band structure
	    calculation. For example if the kpath was G-X-W-L then there are three segements to choose from:
	    G-X, X-W, and W-L. In this case the default kpath could be plotted by defining custom_kpath=[1,2,3],
	    where 1 -> G-X, 2 -> X-W, and 3 -> W-L. If only G-X and X-W were desired then custom_kpath=[1,2].
	    If one of the segements should be flipped it can be done by making its value negative
	    (e.g. -1 -> X-G, -2 -> W-X, -3 -> L-W)
        n (int): Number of points between each high symmetry points.
            This is also only required for unfolded calculations and band unfolding. This number should be 
            known by the user, as it was used to generate the KPOINTS file.
        unfold (bool): Determines if the plotted band structure is from a band unfolding calculation.
        M (list[list]): Transformation matrix from the primitive bulk structure to the slab structure.
            Only required for a band unfolding calculation.
        high_symm_points (list[list]): List of fractional coordinated for each high symmetry point in 
            the band structure path. Only required for a band unfolding calculation.
        fontsize (float): Font size of the text in the figure.
        fill (bool): Determines wether or not to fill underneath the plot
        alpha (float): Alpha value for the fill
        sigma (float): Standard deviation for gaussian filter
        save (bool): Determines whether to automatically save the figure or not. If not 
            the figure and axis are return for further manipulation.

    Returns:
        If save == True, this function will return nothing and directly save the image as
        the output name. If save == False, the function will return the matplotlib figure
        and axis for further editing. 
    """

    fig, ax = plt.subplots(
        nrows=1,
        ncols=2,
        sharey=True,
        figsize=figsize,
        dpi=400,
        gridspec_kw={'width_ratios': width_ratios}
    )

    ax1, ax2 = _figure_setup_band_dos(
        ax=ax,
        fontsize=fontsize,
        ylim=[erange[0], erange[1]]
    )

    band = Band(
        folder=band_folder,
        spin=spin,
        projected=True,
        unfold=unfold,
        high_symm_points=high_symm_points, 
	interpolate=interpolate,
	new_n=new_n,
        kpath=kpath,
	custom_kpath=custom_kpath,
        n=n,
        M=M,
	shift_efermi=shift_efermi,
    )

    dos = Dos(shift_efermi=shift_efermi, folder=dos_folder, spin=spin)

    band.plot_elements(
        ax=ax1,
        elements=elements,
        scale_factor=scale_factor,
        color_list=color_list,
        erange=erange,
        display_order=display_order,
        legend=False,
        linewidth=linewidth,
        band_color=band_color,
    )

    dos.plot_elements(
        ax=ax2,
        elements=elements,
        fill=fill,
        alpha=alpha,
        linewidth=linewidth,
        sigma=sigma,
        energyaxis='y',
        color_list=color_list,
        legend=legend,
        total=True,
        erange=erange,
    )

    fig.canvas.draw()
    nbins = len(ax2.get_xticklabels())
    ax2.xaxis.set_major_locator(MaxNLocator(nbins=nbins - 1, prune='lower'))

    plt.tight_layout(pad=0.4)
    plt.subplots_adjust(wspace=0)

    if save:
        plt.savefig(output)
    else:
        return fig, ax1, ax2


def band_dos_element_spd(
    band_folder,
    dos_folder,
    element_spd_dict,
    output='band_dos_element_spd.png',
    spin='up',
    scale_factor=5,
    color_list=None,
    legend=True,
    linewidth=0.75,
    band_color='black',
    figsize=(6, 3),
    width_ratios=[7, 3],
    erange=[-6, 6],
    kpath=None,
    custom_kpath=None,
    n=None,
    display_order=None,
    unfold=False,
    M=None,
    high_symm_points=None,
    fontsize=12,
    save=True,
    shift_efermi=0,
    interpolate=False,
    new_n=200,
    fill=True,
    alpha=0.3,
    sigma=0.05,
):
    """
    This function generates a s, p, d projected band structure on specific elements next to a 
    projected density of states on the s, p, d orbitals for the same elements.

    Parameters:
        band_folder (str): This is the folder that contains the VASP files for the band structure
        dos_folder (str): This is the folder that contains the VASP files for the density of states
        element_spd_dict (dict[str:str]): A dictionary that contains the individual atoms and the corresponding 
            orbitals to project onto. For example, if the user wants to project onto the s, p, d orbitals
            of In and the p orbitals of As for an InAs structure then the dictionary would be {'In':'spd', 'As':'p'}
        output (str): File name of the resulting plot.
        spin (str): Choose which spin direction to parse. ('up' or 'down')
        scale_factor (float): Factor to scale weights. This changes the size of the
            points in the scatter plot
        display_order (str / None): If None, the projections will be displayed in the same order
            the user inputs them. If 'all' the projections will be plotted from largest to smallest
            so every point is visable. If 'dominant' the projections will be plotted from smallest 
            to largest so only the dominant projection is shown.
        color_list (list): List of colors that is the same length as the number of projections
            in the plot.
        legend (bool): Determines if the legend should be included or not.
        linewidth (float): Line width of the plain band structure plotted in the background
        band_color (string): Color of the plain band structure
        figsize (list / tuple): Desired size of the image in inches (width, height)
        width_ratios (list / tuple): Width ration of the band plot and dos plot. 
        erange (list / tuple): Range of energy to show in the plot [low, high]
        kpath (list[list]): High symmetry k-point path of band structure calculation
            Due to the nature of the KPOINTS file for unfolded calculations this
            information is a required input for proper labeling of the figure
            for unfolded calculations. This information is extracted from the KPOINTS
            files for non-unfolded calculations. (G is automatically converted to \\Gamma)
	    (e.g. For the path X-G-X, kpath=[['X', 'G'], ['G', 'X']])
	custom_kpath (list): This gives the option to only plot specific segments of a given band structure
	    calculation. For example if the kpath was G-X-W-L then there are three segements to choose from:
	    G-X, X-W, and W-L. In this case the default kpath could be plotted by defining custom_kpath=[1,2,3],
	    where 1 -> G-X, 2 -> X-W, and 3 -> W-L. If only G-X and X-W were desired then custom_kpath=[1,2].
	    If one of the segements should be flipped it can be done by making its value negative
	    (e.g. -1 -> X-G, -2 -> W-X, -3 -> L-W)
        n (int): Number of points between each high symmetry points.
            This is also only required for unfolded calculations and band unfolding. This number should be 
            known by the user, as it was used to generate the KPOINTS file.
        unfold (bool): Determines if the plotted band structure is from a band unfolding calculation.
        M (list[list]): Transformation matrix from the primitive bulk structure to the slab structure.
            Only required for a band unfolding calculation.
        high_symm_points (list[list]): List of fractional coordinated for each high symmetry point in 
            the band structure path. Only required for a band unfolding calculation.
        fontsize (float): Font size of the text in the figure.
        fill (bool): Determines wether or not to fill underneath the plot
        alpha (float): Alpha value for the fill
        sigma (float): Standard deviation for gaussian filter
        save (bool): Determines whether to automatically save the figure or not. If not 
            the figure and axis are return for further manipulation.

    Returns:
        If save == True, this function will return nothing and directly save the image as
        the output name. If save == False, the function will return the matplotlib figure
        and axis for further editing. 
    """

    fig, ax = plt.subplots(
        nrows=1,
        ncols=2,
        sharey=True,
        figsize=figsize,
        dpi=400,
        gridspec_kw={'width_ratios': width_ratios}
    )

    ax1, ax2 = _figure_setup_band_dos(
        ax=ax,
        fontsize=fontsize,
        ylim=[erange[0], erange[1]]
    )

    band = Band(
        folder=band_folder,
        spin=spin,
        projected=True,
        unfold=unfold,
        high_symm_points=high_symm_points, 
	interpolate=interpolate,
	new_n=new_n,
        kpath=kpath,
	custom_kpath=custom_kpath,
        n=n,
        M=M,
	shift_efermi=shift_efermi,
    )

    dos = Dos(shift_efermi=shift_efermi, folder=dos_folder, spin=spin)

    band.plot_element_spd(
        ax=ax1,
        element_spd_dict=element_spd_dict,
        display_order=display_order,
        scale_factor=scale_factor,
        color_list=color_list,
        legend=False,
        linewidth=linewidth,
        band_color=band_color,
        erange=erange,
    )

    dos.plot_element_spd(
        ax=ax2,
        element_spd_dict=element_spd_dict,
        fill=fill,
        alpha=alpha,
        linewidth=linewidth,
        sigma=sigma,
        energyaxis='y',
        color_list=color_list,
        legend=legend,
        total=True,
        erange=erange,
    )

    fig.canvas.draw()
    nbins = len(ax2.get_xticklabels())
    ax2.xaxis.set_major_locator(MaxNLocator(nbins=nbins - 1, prune='lower'))

    plt.tight_layout(pad=0.4)
    plt.subplots_adjust(wspace=0)

    if save:
        plt.savefig(output)
    else:
        return fig, ax1, ax2


def band_dos_element_orbitals(
    band_folder,
    dos_folder,
    element_orbital_dict,
    output='band_dos_element_orbitals.png',
    spin='up',
    scale_factor=5,
    color_list=None,
    legend=True,
    linewidth=0.75,
    band_color='black',
    figsize=(6, 3),
    width_ratios=[7, 3],
    erange=[-6, 6],
    kpath=None,
    custom_kpath=None,
    n=None,
    display_order=None,
    unfold=False,
    M=None,
    high_symm_points=None,
    fontsize=12,
    save=True,
    shift_efermi=0,
    interpolate=False,
    new_n=200,
    fill=True,
    alpha=0.3,
    sigma=0.05,
):
    """
    This function generates a projected band structure on orbitals of specific elements next to a
    projected density of states on the same orbitals for the same elements.

    Parameters:
        band_folder (str): This is the folder that contains the VASP files for the band structure
        dos_folder (str): This is the folder that contains the VASP files for the density of states
        element_orbital_dict (dict[str:list]): A dictionary that contains the individual elements and the corresponding 
            orbitals to project onto. For example, if the user wants to project onto the s, py, pz, and px orbitals
            of In and the s orbital of As for and InAs structure then the dictionary would be {'In':[0,1,2,3], 'As':[0]}
        output (str): File name of the resulting plot.
        spin (str): Choose which spin direction to parse. ('up' or 'down')
        scale_factor (float): Factor to scale weights. This changes the size of the
            points in the scatter plot
        display_order (str / None): If None, the projections will be displayed in the same order
            the user inputs them. If 'all' the projections will be plotted from largest to smallest
            so every point is visable. If 'dominant' the projections will be plotted from smallest 
            to largest so only the dominant projection is shown.
        color_list (list): List of colors that is the same length as the number of projections
            in the plot.
        legend (bool): Determines if the legend should be included or not.
        linewidth (float): Line width of the plain band structure plotted in the background
        band_color (string): Color of the plain band structure
        figsize (list / tuple): Desired size of the image in inches (width, height)
        width_ratios (list / tuple): Width ration of the band plot and dos plot. 
        erange (list / tuple): Range of energy to show in the plot [low, high]
        kpath (list[list]): High symmetry k-point path of band structure calculation
            Due to the nature of the KPOINTS file for unfolded calculations this
            information is a required input for proper labeling of the figure
            for unfolded calculations. This information is extracted from the KPOINTS
            files for non-unfolded calculations. (G is automatically converted to \\Gamma)
	    (e.g. For the path X-G-X, kpath=[['X', 'G'], ['G', 'X']])
	custom_kpath (list): This gives the option to only plot specific segments of a given band structure
	    calculation. For example if the kpath was G-X-W-L then there are three segements to choose from:
	    G-X, X-W, and W-L. In this case the default kpath could be plotted by defining custom_kpath=[1,2,3],
	    where 1 -> G-X, 2 -> X-W, and 3 -> W-L. If only G-X and X-W were desired then custom_kpath=[1,2].
	    If one of the segements should be flipped it can be done by making its value negative
	    (e.g. -1 -> X-G, -2 -> W-X, -3 -> L-W)
        n (int): Number of points between each high symmetry points.
            This is also only required for unfolded calculations and band unfolding. This number should be 
            known by the user, as it was used to generate the KPOINTS file.
        unfold (bool): Determines if the plotted band structure is from a band unfolding calculation.
        M (list[list]): Transformation matrix from the primitive bulk structure to the slab structure.
            Only required for a band unfolding calculation.
        high_symm_points (list[list]): List of fractional coordinated for each high symmetry point in 
            the band structure path. Only required for a band unfolding calculation.
        fontsize (float): Font size of the text in the figure.
        fill (bool): Determines wether or not to fill underneath the plot
        alpha (float): Alpha value for the fill
        sigma (float): Standard deviation for gaussian filter
        save (bool): Determines whether to automatically save the figure or not. If not 
            the figure and axis are return for further manipulation.

    Returns:
        If save == True, this function will return nothing and directly save the image as
        the output name. If save == False, the function will return the matplotlib figure
        and axis for further editing. 
    """

    fig, ax = plt.subplots(
        nrows=1,
        ncols=2,
        sharey=True,
        figsize=figsize,
        dpi=400,
        gridspec_kw={'width_ratios': width_ratios}
    )

    ax1, ax2 = _figure_setup_band_dos(
        ax=ax,
        fontsize=fontsize,
        ylim=[erange[0], erange[1]]
    )

    band = Band(
        folder=band_folder,
        spin=spin,
        projected=True,
        unfold=unfold,
        high_symm_points=high_symm_points, 
	interpolate=interpolate,
	new_n=new_n,
        kpath=kpath,
	custom_kpath=custom_kpath,
        n=n,
        M=M,
	shift_efermi=shift_efermi,
    )

    dos = Dos(shift_efermi=shift_efermi, folder=dos_folder, spin=spin)

    band.plot_element_orbitals(
        ax=ax1,
        scale_factor=scale_factor,
        element_orbital_dict=element_orbital_dict,
        display_order=display_order,
        erange=erange,
        color_list=color_list,
        legend=False,
        linewidth=linewidth,
        band_color=band_color,
    )

    dos.plot_element_orbitals(
        ax=ax2,
        element_orbital_dict=element_orbital_dict,
        fill=fill,
        alpha=alpha,
        linewidth=linewidth,
        sigma=sigma,
        energyaxis='y',
        color_list=color_list,
        legend=legend,
        total=True,
        erange=erange,
    )

    fig.canvas.draw()
    nbins = len(ax2.get_xticklabels())
    ax2.xaxis.set_major_locator(MaxNLocator(nbins=nbins - 1, prune='lower'))

    plt.tight_layout(pad=0.4)
    plt.subplots_adjust(wspace=0)

    if save:
        plt.savefig(output)
    else:
        return fig, ax1, ax2


def band_dos_plain_spin_polarized(
    band_folder,
    dos_folder,
    output='band_dos_plain_sp.png',
    up_color='black',
    down_color='red',
    linewidth=1.25,
    up_linestyle='-',
    down_linestyle=':',
    figsize=(6, 3),
    width_ratios=[7, 3],
    erange=[-6, 6],
    kpath=None,
    custom_kpath=None,
    n=None,
    unfold=False,
    M=None,
    high_symm_points=None,
    fontsize=12,
    save=True,
    shift_efermi=0,
    interpolate=False,
    new_n=200,
    fill=True,
    alpha=0.3,
    sigma=0.05,
):
    """
    This function plots a plain spin polarized band structure and density of states next to eachother.

    Parameters:
        band_folder (str): This is the folder that contains the VASP files for the band structure calculation
        dos_folder (str): This is the folder that contains the VASP files for the density of states calculation
        output (str): File name of the resulting plot.
        up_color (str): Color of the spin up band structure lines
        down_color (str): Color of the spin down band structure lines
        linewidth (float): Line width of the band structure lines
        up_linestyle (str): Line style of the spin up bands
        down_linestyle (str): Line style of the spin down bands
        figsize (list / tuple): Desired size of the image in inches (width, height)
        width_ratios (list / tuple): Width ration of the band plot and dos plot. 
        erange (list / tuple): Range of energy to show in the plot [low, high]
        kpath (list[list]): High symmetry k-point path of band structure calculation
            Due to the nature of the KPOINTS file for unfolded calculations this
            information is a required input for proper labeling of the figure
            for unfolded calculations. This information is extracted from the KPOINTS
            files for non-unfolded calculations. (G is automatically converted to \\Gamma)
	    (e.g. For the path X-G-X, kpath=[['X', 'G'], ['G', 'X']])
	custom_kpath (list): This gives the option to only plot specific segments of a given band structure
	    calculation. For example if the kpath was G-X-W-L then there are three segements to choose from:
	    G-X, X-W, and W-L. In this case the default kpath could be plotted by defining custom_kpath=[1,2,3],
	    where 1 -> G-X, 2 -> X-W, and 3 -> W-L. If only G-X and X-W were desired then custom_kpath=[1,2].
	    If one of the segements should be flipped it can be done by making its value negative
	    (e.g. -1 -> X-G, -2 -> W-X, -3 -> L-W)
        n (int): Number of points between each high symmetry points.
            This is also only required for unfolded calculations. This number should be 
            known by the user, as it was used to generate the KPOINTS file.
        fontsize (float): Font size of the text in the figure.
        save (bool): Determines whether to automatically save the figure or not. If not 
            the figure and axis are return for further manipulation.
        fill (bool): Determines wether or not to fill underneath the plot
        alpha (float): Alpha value for the fill
        sigma (float): Standard deviation for gaussian filter

    Returns:
        If save == True, this function will return nothing and directly save the image as
        the output name. If save == False, the function will return the matplotlib figure
        and axis for further editing. 
    """
    fig, ax = plt.subplots(
        nrows=1,
        ncols=2,
        sharey=True,
        figsize=figsize,
        dpi=400,
        gridspec_kw={'width_ratios': width_ratios}
    )

    ax1, ax2 = _figure_setup_band_dos(
        ax=ax,
        fontsize=fontsize,
        ylim=[erange[0], erange[1]]
    )

    band_up = Band(
        folder=band_folder,
        spin='up',
        unfold=unfold,
        high_symm_points=high_symm_points, 
	interpolate=interpolate,
	new_n=new_n,
        kpath=kpath,
	custom_kpath=custom_kpath,
        n=n,
        M=M,
	shift_efermi=shift_efermi,
    )

    band_down = Band(
        folder=band_folder,
        spin='down',
        unfold=unfold,
        high_symm_points=high_symm_points, 
	interpolate=interpolate,
	new_n=new_n,
        kpath=kpath,
	custom_kpath=custom_kpath,
        n=n,
        M=M,
	shift_efermi=shift_efermi,
    )

    dos_up = Dos(shift_efermi=shift_efermi, folder=dos_folder, spin='up')
    dos_down = Dos(shift_efermi=shift_efermi, folder=dos_folder, spin='down')

    band_up.plot_plain(
        ax=ax1,
        color=up_color,
        linewidth=linewidth,
        linestyle=up_linestyle,
        erange=erange,
    )

    band_down.plot_plain(
        ax=ax1,
        color=down_color,
        linewidth=linewidth,
        linestyle=down_linestyle,
        erange=erange,
    )

    dos_up.plot_plain(
        ax=ax2,
        linewidth=linewidth,
        fill=fill,
        alpha=alpha,
        sigma=sigma,
        energyaxis='y',
        color=up_color,
        erange=erange,
    )

    dos_down.plot_plain(
        ax=ax2,
        linewidth=linewidth,
        fill=fill,
        alpha=alpha,
        sigma=sigma,
        energyaxis='y',
        color=down_color,
        erange=erange,
    )

    fig.canvas.draw()
    nbins = len(ax2.get_xticklabels())
    ax2.xaxis.set_major_locator(MaxNLocator(nbins=nbins - 1, prune='lower'))

    plt.tight_layout(pad=0.4)
    plt.subplots_adjust(wspace=0)

    if save:
        plt.savefig(output)
    else:
        return fig, ax1, ax2


def band_dos_spd_spin_polarized(
    band_folder,
    dos_folder,
    output='band_dos_spd_sp.png',
    scale_factor=5,
    orbitals='spd',
    color_list=None,
    legend=True,
    linewidth=0.75,
    band_color='black',
    unprojected_band_color='gray',
    unprojected_linewidth=0.6,
    figsize=(8, 6),
    width_ratios=[7, 3],
    erange=[-6, 6],
    kpath=None,
    custom_kpath=None,
    n=None,
    display_order=None,
    unfold=False,
    M=None,
    high_symm_points=None,
    fontsize=8,
    annotations=['$\\uparrow$ ', '$\\downarrow$ '],
    annotation_xy=(0.0125, 0.98),
    save=True,
    shift_efermi=0,
    interpolate=False,
    new_n=200,
    fill=True,
    alpha=0.3,
    sigma=0.05,
):
    """
    This function plots a spin polarized s, p, d projected band structure next to a spin polarized s, p, d projected
    density of states. The top figure highlights the spin up bands and the bottom figure highlights the spin down bands.

    Parameters:
        band_folder (str): This is the folder that contains the VASP files for the band structure
        dos_folder (str): This is the folder that contains the VASP files for the density of states
        orbitals (str): String that contains the s, p, or d orbitals that to project onto.
            The default is 'spd', if the user only wanted to project onto the p, and d orbitals 
            than 'pd' should be passed in
        output (str): File name of the resulting plot.
        scale_factor (float): Factor to scale weights. This changes the size of the
            points in the scatter plot
        display_order (str / None): If None, the projections will be displayed in the same order
            the user inputs them. If 'all' the projections will be plotted from largest to smallest
            so every point is visable. If 'dominant' the projections will be plotted from smallest 
            to largest so only the dominant projection is shown.
        color_list (list): List of colors that is the same length as the number of projections
            in the plot.
        legend (bool): Determines if the legend should be included or not.
        linewidth (float): Line width of the plain band structure plotted in the background
        band_color (string): Color of the plain band structure
        unprojected_band_color (str): Color of the unprojected band
        unprojected_linewidth (float): Line width of the unprojected bands
        figsize (list / tuple): Desired size of the image in inches (width, height)
        width_ratios (list / tuple): Width ration of the band plot and dos plot. 
        erange (list / tuple): Range of energy to show in the plot [low, high]
        kpath (list[list]): High symmetry k-point path of band structure calculation
            Due to the nature of the KPOINTS file for unfolded calculations this
            information is a required input for proper labeling of the figure
            for unfolded calculations. This information is extracted from the KPOINTS
            files for non-unfolded calculations. (G is automatically converted to \\Gamma)
	    (e.g. For the path X-G-X, kpath=[['X', 'G'], ['G', 'X']])
	custom_kpath (list): This gives the option to only plot specific segments of a given band structure
	    calculation. For example if the kpath was G-X-W-L then there are three segements to choose from:
	    G-X, X-W, and W-L. In this case the default kpath could be plotted by defining custom_kpath=[1,2,3],
	    where 1 -> G-X, 2 -> X-W, and 3 -> W-L. If only G-X and X-W were desired then custom_kpath=[1,2].
	    If one of the segements should be flipped it can be done by making its value negative
	    (e.g. -1 -> X-G, -2 -> W-X, -3 -> L-W)
        n (int): Number of points between each high symmetry points.
            This is also only required for unfolded calculations and band unfolding. This number should be 
            known by the user, as it was used to generate the KPOINTS file.
        unfold (bool): Determines if the plotted band structure is from a band unfolding calculation.
        M (list[list]): Transformation matrix from the primitive bulk structure to the slab structure.
            Only required for a band unfolding calculation.
        high_symm_points (list[list]): List of fractional coordinated for each high symmetry point in 
            the band structure path. Only required for a band unfolding calculation.
        fontsize (float): Font size of the text in the figure.
        annotations (list): Annotations to put on the top and bottom (left and right) figures.
            By default it will show the spin up and spin down arrows.
        annotation_xy (list / tuple): Fractional (x, y) coordinated of the annotation location
        fill (bool): Determines wether or not to fill underneath the plot
        alpha (float): Alpha value for the fill
        sigma (float): Standard deviation for gaussian filter
        save (bool): Determines whether to automatically save the figure or not. If not 
            the figure and axis are return for further manipulation.

    Returns:
        If save == True, this function will return nothing and directly save the image as
        the output name. If save == False, the function will return the matplotlib figure
        and axis for further editing. 
    """

    fig, ax = plt.subplots(
        nrows=2,
        ncols=2,
        sharey=True,
        figsize=figsize,
        dpi=400,
        gridspec_kw={'width_ratios': width_ratios}
    )

    ax_band_up, ax_dos_up, ax_band_down, ax_dos_down = _figure_setup_band_dos_spin_polarized(
        ax=ax,
        fontsize=fontsize,
        ylim=[erange[0], erange[1]]
    )

    band_up = Band(
        folder=band_folder,
        spin='up',
        projected=True,
        unfold=unfold,
        high_symm_points=high_symm_points, 
	interpolate=interpolate,
	new_n=new_n,
        kpath=kpath,
	custom_kpath=custom_kpath,
        n=n,
        M=M,
	shift_efermi=shift_efermi,
    )

    band_down = Band(
        folder=band_folder,
        spin='down',
        projected=True,
        unfold=unfold,
        high_symm_points=high_symm_points, 
	interpolate=interpolate,
	new_n=new_n,
        kpath=kpath,
	custom_kpath=custom_kpath,
        n=n,
        M=M,
	shift_efermi=shift_efermi,
    )

    dos_up = Dos(shift_efermi=shift_efermi, folder=dos_folder, spin='up')
    dos_down = Dos(shift_efermi=shift_efermi, folder=dos_folder, spin='down')

    bbox = dict(boxstyle='round', fc='white',
                edgecolor='gray', alpha=0.95, pad=0.3)

    ax_band_up.annotate(
        annotations[0],
        xy=annotation_xy,
        xycoords='axes fraction',
        va='top',
        ha='left',
        bbox=bbox,
        fontsize=fontsize + 1,
    )
    ax_band_down.annotate(
        annotations[1],
        xy=annotation_xy,
        xycoords='axes fraction',
        va='top',
        ha='left',
        bbox=bbox,
        fontsize=fontsize + 1,
    )

    band_up.plot_spd(
        ax=ax_band_up,
        scale_factor=scale_factor,
        orbitals=orbitals,
        color_list=color_list,
        legend=False,
        linewidth=linewidth,
        band_color=band_color,
        display_order=display_order,
        erange=erange,
    )

    band_down.plot_plain(
        ax=ax_band_up,
        color=unprojected_band_color,
        linewidth=unprojected_linewidth,
        erange=erange,
    )

    dos_up.plot_spd(
        ax=ax_dos_up,
        orbitals=orbitals,
        fill=fill,
        alpha=alpha,
        linewidth=linewidth,
        sigma=sigma,
        energyaxis='y',
        color_list=color_list,
        legend=legend,
        total=True,
        erange=erange,
    )

    dos_down.plot_spd(
        ax=ax_dos_up,
        orbitals=orbitals,
        fill=fill,
        alpha=0.25 * alpha,
        alpha_line=0.25 * alpha,
        linewidth=linewidth,
        sigma=sigma,
        energyaxis='y',
        color_list=color_list,
        legend=False,
        total=True,
        erange=erange,
    )

    band_down.plot_spd(
        ax=ax_band_down,
        scale_factor=scale_factor,
        orbitals=orbitals,
        color_list=color_list,
        legend=False,
        linewidth=linewidth,
        band_color=band_color,
        display_order=display_order,
        erange=erange,
    )

    band_up.plot_plain(
        ax=ax_band_down,
        color=unprojected_band_color,
        linewidth=unprojected_linewidth,
        erange=erange,
    )

    dos_down.plot_spd(
        ax=ax_dos_down,
        orbitals=orbitals,
        fill=fill,
        alpha=alpha,
        linewidth=linewidth,
        sigma=sigma,
        energyaxis='y',
        color_list=color_list,
        legend=legend,
        total=True,
        erange=erange,
    )

    dos_up.plot_spd(
        ax=ax_dos_down,
        orbitals=orbitals,
        fill=fill,
        alpha=0.25 * alpha,
        alpha_line=0.25 * alpha,
        linewidth=linewidth,
        sigma=sigma,
        energyaxis='y',
        color_list=color_list,
        legend=False,
        total=True,
        erange=erange,
    )

    fig.canvas.draw()
    nbins = len(ax_dos_down.get_xticklabels())
    ax_dos_down.xaxis.set_major_locator(
        MaxNLocator(nbins=nbins - 1, prune='lower'))

    plt.tight_layout(pad=0.4)
    plt.subplots_adjust(wspace=0, hspace=0.05)

    if save:
        plt.savefig(output)
    else:
        return fig, ax_band_up, ax_dos_up, ax_band_down, ax_dos_down


def band_dos_atom_orbitals_spin_polarized(
    band_folder,
    dos_folder,
    atom_orbital_dict,
    output='band_dos_atom_orbitals_sp.png',
    scale_factor=5,
    color_list=None,
    legend=True,
    linewidth=0.75,
    band_color='black',
    unprojected_band_color='gray',
    unprojected_linewidth=0.6,
    figsize=(8, 6),
    width_ratios=[7, 3],
    erange=[-6, 6],
    kpath=None,
    custom_kpath=None,
    n=None,
    display_order=None,
    unfold=False,
    M=None,
    high_symm_points=None,
    fontsize=8,
    annotations=['$\\uparrow$ ', '$\\downarrow$ '],
    annotation_xy=(0.0125, 0.98),
    save=True,
    shift_efermi=0,
    interpolate=False,
    new_n=200,
    fill=True,
    alpha=0.3,
    sigma=0.05,
    total=True,
):
    """
    This function plots a spin polarized band structure projected onto specified [atom, orbital] pairs next to a spin
    polarized density of states projected onto the same [atom, orbital] pairs. The top figure highlights the spin up
    bands and the bottom figure highlights the spin down bands.

    Parameters:
        band_folder (str): This is the folder that contains the VASP files for the band structure
        dos_folder (str): This is the folder that contains the VASP files for the density of states
        atom_orbital_dict (dict[int:list]): A dictionary that contains the individual atoms and the
            corresponding orbitals to project onto. For example, if the user wants to project onto the s, py, pz, and px orbitals
            of the first atom and the s orbital of the second atom then the dictionary would be {0:[0,1,2,3], 1:[0]}
        output (str): File name of the resulting plot.
        scale_factor (float): Factor to scale weights. This changes the size of the
            points in the scatter plot
        display_order (str / None): If None, the projections will be displayed in the same order
            the user inputs them. If 'all' the projections will be plotted from largest to smallest
            so every point is visable. If 'dominant' the projections will be plotted from smallest 
            to largest so only the dominant projection is shown.
        color_list (list): List of colors that is the same length as the number of projections
            in the plot.
        legend (bool): Determines if the legend should be included or not.
        linewidth (float): Line width of the plain band structure plotted in the background
        band_color (string): Color of the plain band structure
        unprojected_band_color (str): Color of the unprojected band
        unprojected_linewidth (float): Line width of the unprojected bands
        figsize (list / tuple): Desired size of the image in inches (width, height)
        width_ratios (list / tuple): Width ration of the band plot and dos plot. 
        erange (list / tuple): Range of energy to show in the plot [low, high]
        kpath (list[list]): High symmetry k-point path of band structure calculation
            Due to the nature of the KPOINTS file for unfolded calculations this
            information is a required input for proper labeling of the figure
            for unfolded calculations. This information is extracted from the KPOINTS
            files for non-unfolded calculations. (G is automatically converted to \\Gamma)
	    (e.g. For the path X-G-X, kpath=[['X', 'G'], ['G', 'X']])
	custom_kpath (list): This gives the option to only plot specific segments of a given band structure
	    calculation. For example if the kpath was G-X-W-L then there are three segements to choose from:
	    G-X, X-W, and W-L. In this case the default kpath could be plotted by defining custom_kpath=[1,2,3],
	    where 1 -> G-X, 2 -> X-W, and 3 -> W-L. If only G-X and X-W were desired then custom_kpath=[1,2].
	    If one of the segements should be flipped it can be done by making its value negative
	    (e.g. -1 -> X-G, -2 -> W-X, -3 -> L-W)
        n (int): Number of points between each high symmetry points.
            This is also only required for unfolded calculations and band unfolding. This number should be 
            known by the user, as it was used to generate the KPOINTS file.
        unfold (bool): Determines if the plotted band structure is from a band unfolding calculation.
        M (list[list]): Transformation matrix from the primitive bulk structure to the slab structure.
            Only required for a band unfolding calculation.
        high_symm_points (list[list]): List of fractional coordinated for each high symmetry point in 
            the band structure path. Only required for a band unfolding calculation.
        fontsize (float): Font size of the text in the figure.
        annotations (list): Annotations to put on the top and bottom (left and right) figures.
            By default it will show the spin up and spin down arrows.
        annotation_xy (list / tuple): Fractional (x, y) coordinated of the annotation location
        fill (bool): Determines wether or not to fill underneath the plot
        alpha (float): Alpha value for the fill
        sigma (float): Standard deviation for gaussian filter
        save (bool): Determines whether to automatically save the figure or not. If not 
            the figure and axis are return for further manipulation.

    Returns:
        If save == True, this function will return nothing and directly save the image as
        the output name. If save == False, the function will return the matplotlib figure
        and axis for further editing. 
    """

    fig, ax = plt.subplots(
        nrows=2,
        ncols=2,
        sharey=True,
        figsize=figsize,
        dpi=400,
        gridspec_kw={'width_ratios': width_ratios}
    )

    ax_band_up, ax_dos_up, ax_band_down, ax_dos_down = _figure_setup_band_dos_spin_polarized(
        ax=ax,
        fontsize=fontsize,
        ylim=[erange[0], erange[1]]
    )

    band_up = Band(
        folder=band_folder,
        spin='up',
        projected=True,
        unfold=unfold,
        high_symm_points=high_symm_points, 
	interpolate=interpolate,
	new_n=new_n,
        kpath=kpath,
	custom_kpath=custom_kpath,
        n=n,
        M=M,
	shift_efermi=shift_efermi,
    )

    band_down = Band(
        folder=band_folder,
        spin='down',
        projected=True,
        unfold=unfold,
        high_symm_points=high_symm_points, 
	interpolate=interpolate,
	new_n=new_n,
        kpath=kpath,
	custom_kpath=custom_kpath,
        n=n,
        M=M,
	shift_efermi=shift_efermi,
    )

    dos_up = Dos(shift_efermi=shift_efermi, folder=dos_folder, spin='up')
    dos_down = Dos(shift_efermi=shift_efermi, folder=dos_folder, spin='down')

    bbox = dict(boxstyle='round', fc='white',
                edgecolor='gray', alpha=0.95, pad=0.3)

    ax_band_up.annotate(
        annotations[0],
        xy=annotation_xy,
        xycoords='axes fraction',
        va='top',
        ha='left',
        bbox=bbox,
        fontsize=fontsize + 1,
    )
    ax_band_down.annotate(
        annotations[1],
        xy=annotation_xy,
        xycoords='axes fraction',
        va='top',
        ha='left',
        bbox=bbox,
        fontsize=fontsize + 1,
    )

    band_up.plot_atom_orbitals(
        ax=ax_band_up,
        scale_factor=scale_factor,
        atom_orbital_dict=atom_orbital_dict,
        color_list=color_list,
        legend=False,
        linewidth=linewidth,
        band_color=band_color,
        display_order=display_order,
        erange=erange,
    )

    band_down.plot_plain(
        ax=ax_band_up,
        color=unprojected_band_color,
        linewidth=unprojected_linewidth,
        erange=erange,
    )

    dos_up.plot_atom_orbitals(
        ax=ax_dos_up,
        atom_orbital_dict=atom_orbital_dict,
        fill=fill,
        alpha=alpha,
        linewidth=linewidth,
        sigma=sigma,
        energyaxis='y',
        color_list=color_list,
        legend=legend,
        total=total,
        erange=erange,
    )

    dos_down.plot_atom_orbitals(
        ax=ax_dos_up,
        atom_orbital_dict=atom_orbital_dict,
        fill=fill,
        alpha=0.25 * alpha,
        alpha_line=0.25 * alpha,
        linewidth=linewidth,
        sigma=sigma,
        energyaxis='y',
        color_list=color_list,
        legend=False,
        total=total,
        erange=erange,
    )

    band_down.plot_atom_orbitals(
        ax=ax_band_down,
        scale_factor=scale_factor,
        atom_orbital_dict=atom_orbital_dict,
        color_list=color_list,
        legend=False,
        linewidth=linewidth,
        band_color=band_color,
        display_order=display_order,
        erange=erange,
    )

    band_up.plot_plain(
        ax=ax_band_down,
        color=unprojected_band_color,
        linewidth=unprojected_linewidth,
        erange=erange,
    )

    dos_down.plot_atom_orbitals(
        ax=ax_dos_down,
        atom_orbital_dict=atom_orbital_dict,
        fill=fill,
        alpha=alpha,
        linewidth=linewidth,
        sigma=sigma,
        energyaxis='y',
        color_list=color_list,
        legend=legend,
        total=total,
        erange=erange,
    )

    dos_up.plot_atom_orbitals(
        ax=ax_dos_down,
        atom_orbital_dict=atom_orbital_dict,
        fill=fill,
        alpha=0.25 * alpha,
        alpha_line=0.25 * alpha,
        linewidth=linewidth,
        sigma=sigma,
        energyaxis='y',
        color_list=color_list,
        legend=False,
        total=total,
        erange=erange,
    )

    fig.canvas.draw()
    nbins = len(ax_dos_down.get_xticklabels())
    ax_dos_down.xaxis.set_major_locator(
        MaxNLocator(nbins=nbins - 1, prune='lower'))

    plt.tight_layout(pad=0.4)
    plt.subplots_adjust(wspace=0, hspace=0.05)

    if save:
        plt.savefig(output)
    else:
        return fig, ax_band_up, ax_dos_up, ax_band_down, ax_dos_down


def band_dos_orbitals_spin_polarized(
    band_folder,
    dos_folder,
    orbitals,
    output='band_dos_orbitals_sp.png',
    scale_factor=5,
    color_list=None,
    legend=True,
    linewidth=0.75,
    band_color='black',
    unprojected_band_color='gray',
    unprojected_linewidth=0.6,
    figsize=(8, 6),
    width_ratios=[7, 3],
    erange=[-6, 6],
    kpath=None,
    custom_kpath=None,
    n=None,
    fontsize=8,
    annotations=['$\\uparrow$ ', '$\\downarrow$ '],
    annotation_xy=(0.0125, 0.98),
    save=True,
    shift_efermi=0,
    interpolate=False,
    new_n=200,
    fill=True,
    alpha=0.3,
    sigma=0.05,
):
    """
    This function plots a spin polarized bandstructure projected onto specified orbitals next to its
    corresponding density of states projected onto the same orbitals. The top figure highlights the
    spin up bands and the bottom figure highlight the spin down bands.

    Parameters:
        band_folder (str): This is the folder that contains the VASP files for the band structure
        dos_folder (str): This is the folder that contains the VASP files for the density of states
        orbitals (list): List of orbitals to compare

            | 0 = s
            | 1 = py
            | 2 = pz
            | 3 = px
            | 4 = dxy
            | 5 = dyz
            | 6 = dz2
            | 7 = dxz
            | 8 = dx2-y2
            | 9 = fy3x2
            | 10 = fxyz
            | 11 = fyz2
            | 12 = fz3
            | 13 = fxz2
            | 14 = fzx3
            | 15 = fx3

        output (str): File name of the resulting plot.
        scale_factor (float): Factor to scale weights. This changes the size of the
            points in the scatter plot
        display_order (str / None): If None, the projections will be displayed in the same order
            the user inputs them. If 'all' the projections will be plotted from largest to smallest
            so every point is visable. If 'dominant' the projections will be plotted from smallest 
            to largest so only the dominant projection is shown.
        color_list (list): List of colors that is the same length as the number of projections
            in the plot.
        legend (bool): Determines if the legend should be included or not.
        linewidth (float): Line width of the plain band structure plotted in the background
        band_color (string): Color of the plain band structure
        unprojected_band_color (str): Color of the unprojected band
        unprojected_linewidth (float): Line width of the unprojected bands
        figsize (list / tuple): Desired size of the image in inches (width, height)
        width_ratios (list / tuple): Width ration of the band plot and dos plot. 
        erange (list / tuple): Range of energy to show in the plot [low, high]
        kpath (list[list]): High symmetry k-point path of band structure calculation
            Due to the nature of the KPOINTS file for unfolded calculations this
            information is a required input for proper labeling of the figure
            for unfolded calculations. This information is extracted from the KPOINTS
            files for non-unfolded calculations. (G is automatically converted to \\Gamma)
	    (e.g. For the path X-G-X, kpath=[['X', 'G'], ['G', 'X']])
	custom_kpath (list): This gives the option to only plot specific segments of a given band structure
	    calculation. For example if the kpath was G-X-W-L then there are three segements to choose from:
	    G-X, X-W, and W-L. In this case the default kpath could be plotted by defining custom_kpath=[1,2,3],
	    where 1 -> G-X, 2 -> X-W, and 3 -> W-L. If only G-X and X-W were desired then custom_kpath=[1,2].
	    If one of the segements should be flipped it can be done by making its value negative
	    (e.g. -1 -> X-G, -2 -> W-X, -3 -> L-W)
        n (int): Number of points between each high symmetry points.
            This is also only required for unfolded calculations and band unfolding. This number should be 
            known by the user, as it was used to generate the KPOINTS file.
        unfold (bool): Determines if the plotted band structure is from a band unfolding calculation.
        M (list[list]): Transformation matrix from the primitive bulk structure to the slab structure.
            Only required for a band unfolding calculation.
        high_symm_points (list[list]): List of fractional coordinated for each high symmetry point in 
            the band structure path. Only required for a band unfolding calculation.
        fontsize (float): Font size of the text in the figure.
        annotations (list): Annotations to put on the top and bottom (left and right) figures.
            By default it will show the spin up and spin down arrows.
        annotation_xy (list / tuple): Fractional (x, y) coordinated of the annotation location
        fill (bool): Determines wether or not to fill underneath the plot
        alpha (float): Alpha value for the fill
        sigma (float): Standard deviation for gaussian filter
        save (bool): Determines whether to automatically save the figure or not. If not 
            the figure and axis are return for further manipulation.

    Returns:
        If save == True, this function will return nothing and directly save the image as
        the output name. If save == False, the function will return the matplotlib figure
        and axis for further editing. 
    """

    fig, ax = plt.subplots(
        nrows=2,
        ncols=2,
        sharey=True,
        figsize=figsize,
        dpi=400,
        gridspec_kw={'width_ratios': width_ratios}
    )

    ax_band_up, ax_dos_up, ax_band_down, ax_dos_down = _figure_setup_band_dos_spin_polarized(
        ax=ax,
        fontsize=fontsize,
        ylim=[erange[0], erange[1]]
    )

    band_up = Band(
        folder=band_folder,
        spin='up',
        projected=True,
        kpath=kpath,
	custom_kpath=custom_kpath,
        n=n,
    )

    band_down = Band(
        folder=band_folder,
        spin='down',
        projected=True,
        kpath=kpath,
	custom_kpath=custom_kpath,
        n=n,
    )

    dos_up = Dos(shift_efermi=shift_efermi, folder=dos_folder, spin='up')
    dos_down = Dos(shift_efermi=shift_efermi, folder=dos_folder, spin='down')

    bbox = dict(boxstyle='round', fc='white',
                edgecolor='gray', alpha=0.95, pad=0.3)

    ax_band_up.annotate(
        annotations[0],
        xy=annotation_xy,
        xycoords='axes fraction',
        va='top',
        ha='left',
        bbox=bbox,
        fontsize=fontsize + 1,
    )
    ax_band_down.annotate(
        annotations[1],
        xy=annotation_xy,
        xycoords='axes fraction',
        va='top',
        ha='left',
        bbox=bbox,
        fontsize=fontsize + 1,
    )

    band_up.plot_orbitals(
        ax=ax_band_up,
        scale_factor=scale_factor,
        orbitals=orbitals,
        color_list=color_list,
        legend=False,
        linewidth=linewidth,
        band_color=band_color,
    )

    band_down.plot_plain(
        ax=ax_band_up,
        color=unprojected_band_color,
        linewidth=unprojected_linewidth,
    )

    dos_up.plot_orbitals(
        ax=ax_dos_up,
        orbitals=orbitals,
        fill=fill,
        alpha=alpha,
        linewidth=linewidth,
        sigma=sigma,
        energyaxis='y',
        color_list=color_list,
        legend=legend,
        total=True,
        erange=erange,
    )

    dos_down.plot_orbitals(
        ax=ax_dos_up,
        orbitals=orbitals,
        fill=fill,
        alpha=0.25 * alpha,
        alpha_line=0.25 * alpha,
        linewidth=linewidth,
        sigma=sigma,
        energyaxis='y',
        color_list=color_list,
        legend=False,
        total=True,
        erange=erange,
    )

    band_down.plot_orbitals(
        ax=ax_band_down,
        scale_factor=scale_factor,
        orbitals=orbitals,
        color_list=color_list,
        legend=False,
        linewidth=linewidth,
        band_color=band_color,
    )

    band_up.plot_plain(
        ax=ax_band_down,
        color=unprojected_band_color,
        linewidth=unprojected_linewidth,
    )

    dos_down.plot_orbitals(
        ax=ax_dos_down,
        orbitals=orbitals,
        fill=fill,
        alpha=alpha,
        linewidth=linewidth,
        sigma=sigma,
        energyaxis='y',
        color_list=color_list,
        legend=legend,
        total=True,
        erange=erange,
    )

    dos_up.plot_orbitals(
        ax=ax_dos_down,
        orbitals=orbitals,
        fill=fill,
        alpha=0.25 * alpha,
        alpha_line=0.25 * alpha,
        linewidth=linewidth,
        sigma=sigma,
        energyaxis='y',
        color_list=color_list,
        legend=False,
        total=True,
        erange=erange,
    )

    fig.canvas.draw()
    nbins = len(ax_dos_down.get_xticklabels())
    ax_dos_down.xaxis.set_major_locator(
        MaxNLocator(nbins=nbins - 1, prune='lower'))

    plt.tight_layout(pad=0.4)
    plt.subplots_adjust(wspace=0, hspace=0.05)

    if save:
        plt.savefig(output)
    else:
        return fig, ax_band_up, ax_dos_up, ax_band_down, ax_dos_down


def band_dos_atoms_spin_polarized(
    band_folder,
    dos_folder,
    atoms,
    output='band_dos_atoms_sp.png',
    scale_factor=5,
    color_list=None,
    legend=True,
    linewidth=0.75,
    band_color='black',
    unprojected_band_color='gray',
    unprojected_linewidth=0.6,
    figsize=(8, 6),
    width_ratios=[7, 3],
    erange=[-6, 6],
    kpath=None,
    custom_kpath=None,
    n=None,
    fontsize=8,
    annotations=['$\\uparrow$ ', '$\\downarrow$ '],
    annotation_xy=(0.0125, 0.98),
    save=True,
    shift_efermi=0,
    interpolate=False,
    new_n=200,
    fill=True,
    alpha=0.3,
    sigma=0.05,
):
    """
    This function plots a spin polarized bandstructure projected onto specified atoms in the POSCAR next
    to its corresponding density of states projected onto the same atoms. The top figure highlights the
    spin up bands and the bottom figure highlight the spin down bands.

    Parameters:
        band_folder (str): This is the folder that contains the VASP files for the band structure
        dos_folder (str): This is the folder that contains the VASP files for the density of states
        atoms (list): List of atoms to project onto. The indices should be zero indexed (first atom is 0)
            and the atoms are in the same order as they are in the POSCAR
        output (str): File name of the resulting plot.
        scale_factor (float): Factor to scale weights. This changes the size of the
            points in the scatter plot
        display_order (str / None): If None, the projections will be displayed in the same order
            the user inputs them. If 'all' the projections will be plotted from largest to smallest
            so every point is visable. If 'dominant' the projections will be plotted from smallest 
            to largest so only the dominant projection is shown.
        color_list (list): List of colors that is the same length as the number of projections
            in the plot.
        legend (bool): Determines if the legend should be included or not.
        linewidth (float): Line width of the plain band structure plotted in the background
        band_color (string): Color of the plain band structure
        unprojected_band_color (str): Color of the unprojected band
        unprojected_linewidth (float): Line width of the unprojected bands
        figsize (list / tuple): Desired size of the image in inches (width, height)
        width_ratios (list / tuple): Width ration of the band plot and dos plot. 
        erange (list / tuple): Range of energy to show in the plot [low, high]
        kpath (list[list]): High symmetry k-point path of band structure calculation
            Due to the nature of the KPOINTS file for unfolded calculations this
            information is a required input for proper labeling of the figure
            for unfolded calculations. This information is extracted from the KPOINTS
            files for non-unfolded calculations. (G is automatically converted to \\Gamma)
	    (e.g. For the path X-G-X, kpath=[['X', 'G'], ['G', 'X']])
	custom_kpath (list): This gives the option to only plot specific segments of a given band structure
	    calculation. For example if the kpath was G-X-W-L then there are three segements to choose from:
	    G-X, X-W, and W-L. In this case the default kpath could be plotted by defining custom_kpath=[1,2,3],
	    where 1 -> G-X, 2 -> X-W, and 3 -> W-L. If only G-X and X-W were desired then custom_kpath=[1,2].
	    If one of the segements should be flipped it can be done by making its value negative
	    (e.g. -1 -> X-G, -2 -> W-X, -3 -> L-W)
        n (int): Number of points between each high symmetry points.
            This is also only required for unfolded calculations and band unfolding. This number should be 
            known by the user, as it was used to generate the KPOINTS file.
        unfold (bool): Determines if the plotted band structure is from a band unfolding calculation.
        M (list[list]): Transformation matrix from the primitive bulk structure to the slab structure.
            Only required for a band unfolding calculation.
        high_symm_points (list[list]): List of fractional coordinated for each high symmetry point in 
            the band structure path. Only required for a band unfolding calculation.
        fontsize (float): Font size of the text in the figure.
        annotations (list): Annotations to put on the top and bottom (left and right) figures.
            By default it will show the spin up and spin down arrows.
        annotation_xy (list / tuple): Fractional (x, y) coordinated of the annotation location
        fill (bool): Determines wether or not to fill underneath the plot
        alpha (float): Alpha value for the fill
        sigma (float): Standard deviation for gaussian filter
        save (bool): Determines whether to automatically save the figure or not. If not 
            the figure and axis are return for further manipulation.

    Returns:
        If save == True, this function will return nothing and directly save the image as
        the output name. If save == False, the function will return the matplotlib figure
        and axis for further editing. 
    """

    fig, ax = plt.subplots(
        nrows=2,
        ncols=2,
        sharey=True,
        figsize=figsize,
        dpi=400,
        gridspec_kw={'width_ratios': width_ratios}
    )

    ax_band_up, ax_dos_up, ax_band_down, ax_dos_down = _figure_setup_band_dos_spin_polarized(
        ax=ax,
        fontsize=fontsize,
        ylim=[erange[0], erange[1]]
    )

    band_up = Band(
        folder=band_folder,
        spin='up',
        projected=True,
        kpath=kpath,
	custom_kpath=custom_kpath,
        n=n,
    )

    band_down = Band(
        folder=band_folder,
        spin='down',
        projected=True,
        kpath=kpath,
	custom_kpath=custom_kpath,
        n=n,
    )

    dos_up = Dos(shift_efermi=shift_efermi, folder=dos_folder, spin='up')
    dos_down = Dos(shift_efermi=shift_efermi, folder=dos_folder, spin='down')

    bbox = dict(boxstyle='round', fc='white',
                edgecolor='gray', alpha=0.95, pad=0.3)

    ax_band_up.annotate(
        annotations[0],
        xy=annotation_xy,
        xycoords='axes fraction',
        va='top',
        ha='left',
        bbox=bbox,
        fontsize=fontsize + 1,
    )
    ax_band_down.annotate(
        annotations[1],
        xy=annotation_xy,
        xycoords='axes fraction',
        va='top',
        ha='left',
        bbox=bbox,
        fontsize=fontsize + 1,
    )

    band_up.plot_atoms(
        ax=ax_band_up,
        scale_factor=scale_factor,
        atoms=atoms,
        color_list=color_list,
        legend=False,
        linewidth=linewidth,
        band_color=band_color,
    )

    band_down.plot_plain(
        ax=ax_band_up,
        color=unprojected_band_color,
        linewidth=unprojected_linewidth,
    )

    dos_up.plot_atoms(
        ax=ax_dos_up,
        atoms=atoms,
        fill=fill,
        alpha=alpha,
        linewidth=linewidth,
        sigma=sigma,
        energyaxis='y',
        color_list=color_list,
        legend=legend,
        total=True,
        erange=erange,
    )

    dos_down.plot_atoms(
        ax=ax_dos_up,
        atoms=atoms,
        fill=fill,
        alpha=0.25 * alpha,
        alpha_line=0.25 * alpha,
        linewidth=linewidth,
        sigma=sigma,
        energyaxis='y',
        color_list=color_list,
        legend=False,
        total=True,
        erange=erange,
    )

    band_down.plot_atoms(
        ax=ax_band_down,
        scale_factor=scale_factor,
        atoms=atoms,
        color_list=color_list,
        legend=False,
        linewidth=linewidth,
        band_color=band_color,
    )

    band_up.plot_plain(
        ax=ax_band_down,
        color=unprojected_band_color,
        linewidth=unprojected_linewidth,
    )

    dos_down.plot_atoms(
        ax=ax_dos_down,
        atoms=atoms,
        fill=fill,
        alpha=alpha,
        linewidth=linewidth,
        sigma=sigma,
        energyaxis='y',
        color_list=color_list,
        legend=legend,
        total=True,
        erange=erange,
    )

    dos_up.plot_atoms(
        ax=ax_dos_down,
        atoms=atoms,
        fill=fill,
        alpha=0.25 * alpha,
        alpha_line=0.25 * alpha,
        linewidth=linewidth,
        sigma=sigma,
        energyaxis='y',
        color_list=color_list,
        legend=False,
        total=True,
        erange=erange,
    )

    fig.canvas.draw()
    nbins = len(ax_dos_down.get_xticklabels())
    ax_dos_down.xaxis.set_major_locator(
        MaxNLocator(nbins=nbins - 1, prune='lower'))

    plt.tight_layout(pad=0.4)
    plt.subplots_adjust(wspace=0, hspace=0.05)

    if save:
        plt.savefig(output)
    else:
        return fig, ax_band_up, ax_dos_up, ax_band_down, ax_dos_down

def band_dos_atom_spd_spin_polarized(
    band_folder,
    dos_folder,
    atom_spd_dict,
    output='band_dos_atom_spd_sp.png',
    scale_factor=5,
    color_list=None,
    legend=True,
    linewidth=0.75,
    band_color='black',
    unprojected_band_color='gray',
    unprojected_linewidth=0.6,
    figsize=(8, 6),
    width_ratios=[7, 3],
    erange=[-6, 6],
    kpath=None,
    custom_kpath=None,
    n=None,
    fontsize=8,
    annotations=['$\\uparrow$ ', '$\\downarrow$ '],
    annotation_xy=(0.0125, 0.98),
    save=True,
    shift_efermi=0,
    interpolate=False,
    new_n=200,
    fill=True,
    alpha=0.3,
    sigma=0.05,
):
    """
    This function plots a spin polarized s, p, d projected band structure on a given atom next to a spin polarized
    s, p, d projected density of states on the same atom. The top figure highlights the spin up bands and the bottom
    figure highlights the spin down bands.

    Parameters:
        band_folder (str): This is the folder that contains the VASP files for the band structure
        dos_folder (str): This is the folder that contains the VASP files for the density of states
        atom_spd_dict (dict[int:str]): A dictionary that contains the individual atoms and the corresponding 
            orbitals to project onto. For example, if the user wants to project onto the s, p, d orbitals
            of the first atom and the p orbitals of the second atom then the dictionary would be {0:'spd', 1:'p'}
        output (str): File name of the resulting plot.
        scale_factor (float): Factor to scale weights. This changes the size of the
            points in the scatter plot
        display_order (str / None): If None, the projections will be displayed in the same order
            the user inputs them. If 'all' the projections will be plotted from largest to smallest
            so every point is visable. If 'dominant' the projections will be plotted from smallest 
            to largest so only the dominant projection is shown.
        color_list (list): List of colors that is the same length as the number of projections
            in the plot.
        legend (bool): Determines if the legend should be included or not.
        linewidth (float): Line width of the plain band structure plotted in the background
        band_color (string): Color of the plain band structure
        unprojected_band_color (str): Color of the unprojected band
        unprojected_linewidth (float): Line width of the unprojected bands
        figsize (list / tuple): Desired size of the image in inches (width, height)
        width_ratios (list / tuple): Width ration of the band plot and dos plot. 
        erange (list / tuple): Range of energy to show in the plot [low, high]
        kpath (list[list]): High symmetry k-point path of band structure calculation
            Due to the nature of the KPOINTS file for unfolded calculations this
            information is a required input for proper labeling of the figure
            for unfolded calculations. This information is extracted from the KPOINTS
            files for non-unfolded calculations. (G is automatically converted to \\Gamma)
	    (e.g. For the path X-G-X, kpath=[['X', 'G'], ['G', 'X']])
	custom_kpath (list): This gives the option to only plot specific segments of a given band structure
	    calculation. For example if the kpath was G-X-W-L then there are three segements to choose from:
	    G-X, X-W, and W-L. In this case the default kpath could be plotted by defining custom_kpath=[1,2,3],
	    where 1 -> G-X, 2 -> X-W, and 3 -> W-L. If only G-X and X-W were desired then custom_kpath=[1,2].
	    If one of the segements should be flipped it can be done by making its value negative
	    (e.g. -1 -> X-G, -2 -> W-X, -3 -> L-W)
        n (int): Number of points between each high symmetry points.
            This is also only required for unfolded calculations and band unfolding. This number should be 
            known by the user, as it was used to generate the KPOINTS file.
        unfold (bool): Determines if the plotted band structure is from a band unfolding calculation.
        M (list[list]): Transformation matrix from the primitive bulk structure to the slab structure.
            Only required for a band unfolding calculation.
        high_symm_points (list[list]): List of fractional coordinated for each high symmetry point in 
            the band structure path. Only required for a band unfolding calculation.
        fontsize (float): Font size of the text in the figure.
        annotations (list): Annotations to put on the top and bottom (left and right) figures.
            By default it will show the spin up and spin down arrows.
        annotation_xy (list / tuple): Fractional (x, y) coordinated of the annotation location
        fill (bool): Determines wether or not to fill underneath the plot
        alpha (float): Alpha value for the fill
        sigma (float): Standard deviation for gaussian filter
        save (bool): Determines whether to automatically save the figure or not. If not 
            the figure and axis are return for further manipulation.

    Returns:
        If save == True, this function will return nothing and directly save the image as
        the output name. If save == False, the function will return the matplotlib figure
        and axis for further editing. 
    """

    fig, ax = plt.subplots(
        nrows=2,
        ncols=2,
        sharey=True,
        figsize=figsize,
        dpi=400,
        gridspec_kw={'width_ratios': width_ratios}
    )

    ax_band_up, ax_dos_up, ax_band_down, ax_dos_down = _figure_setup_band_dos_spin_polarized(
        ax=ax,
        fontsize=fontsize,
        ylim=[erange[0], erange[1]]
    )

    band_up = Band(
        folder=band_folder,
        spin='up',
        projected=True,
        kpath=kpath,
	custom_kpath=custom_kpath,
        n=n,
    )

    band_down = Band(
        folder=band_folder,
        spin='down',
        projected=True,
        kpath=kpath,
	custom_kpath=custom_kpath,
        n=n,
    )

    dos_up = Dos(shift_efermi=shift_efermi, folder=dos_folder, spin='up')
    dos_down = Dos(shift_efermi=shift_efermi, folder=dos_folder, spin='down')

    bbox = dict(boxstyle='round', fc='white',
                edgecolor='gray', alpha=0.95, pad=0.3)

    ax_band_up.annotate(
        annotations[0],
        xy=annotation_xy,
        xycoords='axes fraction',
        va='top',
        ha='left',
        bbox=bbox,
        fontsize=fontsize + 1,
    )
    ax_band_down.annotate(
        annotations[1],
        xy=annotation_xy,
        xycoords='axes fraction',
        va='top',
        ha='left',
        bbox=bbox,
        fontsize=fontsize + 1,
    )

    band_up.plot_atom_spd(
        ax=ax_band_up,
        atom_spd_dict=atom_spd_dict,
        scale_factor=scale_factor,
        color_list=color_list,
        legend=False,
        linewidth=linewidth,
        band_color=band_color,
    )

    band_down.plot_plain(
        ax=ax_band_up,
        color=unprojected_band_color,
        linewidth=unprojected_linewidth,
    )

    dos_up.plot_atom_spd(
        ax=ax_dos_up,
        atom_spd_dict=atom_spd_dict,
        fill=fill,
        alpha=alpha,
        linewidth=linewidth,
        sigma=sigma,
        energyaxis='y',
        color_list=color_list,
        legend=legend,
        total=True,
        erange=erange,
    )

    dos_down.plot_atom_spd(
        ax=ax_dos_up,
        atom_spd_dict=atom_spd_dict,
        fill=fill,
        alpha=0.25 * alpha,
        alpha_line=0.25 * alpha,
        linewidth=linewidth,
        sigma=sigma,
        energyaxis='y',
        color_list=color_list,
        legend=False,
        total=True,
        erange=erange,
    )

    band_down.plot_atom_spd(
        ax=ax_band_down,
        atom_spd_dict=atom_spd_dict,
        scale_factor=scale_factor,
        color_list=color_list,
        legend=False,
        linewidth=linewidth,
        band_color=band_color,
    )

    band_up.plot_plain(
        ax=ax_band_down,
        color=unprojected_band_color,
        linewidth=unprojected_linewidth,
    )

    dos_down.plot_atom_spd(
        ax=ax_dos_down,
        atom_spd_dict=atom_spd_dict,
        fill=fill,
        alpha=alpha,
        linewidth=linewidth,
        sigma=sigma,
        energyaxis='y',
        color_list=color_list,
        legend=legend,
        total=True,
        erange=erange,
    )

    dos_up.plot_atom_spd(
        ax=ax_dos_down,
        atom_spd_dict=atom_spd_dict,
        fill=fill,
        alpha=0.25 * alpha,
        alpha_line=0.25 * alpha,
        linewidth=linewidth,
        sigma=sigma,
        energyaxis='y',
        color_list=color_list,
        legend=False,
        total=True,
        erange=erange,
    )

    fig.canvas.draw()
    nbins = len(ax_dos_down.get_xticklabels())
    ax_dos_down.xaxis.set_major_locator(
        MaxNLocator(nbins=nbins - 1, prune='lower'))

    plt.tight_layout(pad=0.4)
    plt.subplots_adjust(wspace=0, hspace=0.05)

    if save:
        plt.savefig(output)
    else:
        return fig, ax_band_up, ax_dos_up, ax_band_down, ax_dos_down


def band_dos_elements_spin_polarized(
    band_folder,
    dos_folder,
    elements,
    output='band_dos_elements_sp.png',
    scale_factor=5,
    color_list=None,
    legend=True,
    linewidth=0.75,
    band_color='black',
    unprojected_band_color='gray',
    unprojected_linewidth=0.6,
    figsize=(8, 6),
    width_ratios=[7, 3],
    erange=[-6, 6],
    kpath=None,
    custom_kpath=None,
    n=None,
    fontsize=8,
    annotations=['$\\uparrow$ ', '$\\downarrow$ '],
    annotation_xy=(0.0125, 0.98),
    save=True,
    shift_efermi=0,
    interpolate=False,
    new_n=200,
    fill=True,
    alpha=0.3,
    sigma=0.05,
):
    """
    This function plots a spin polarized bandstructure projected onto specified elements in the structure next
    to its corresponding density of states projected onto the same elements. The top figure highlights the
    spin up bands and the bottom figure highlight the spin down bands.

    Parameters:
        band_folder (str): This is the folder that contains the VASP files for the band structure
        dos_folder (str): This is the folder that contains the VASP files for the density of states
        elements (list): List of elements to project onto. The list should countain the corresponding element symbols
        output (str): File name of the resulting plot.
        scale_factor (float): Factor to scale weights. This changes the size of the
            points in the scatter plot
        display_order (str / None): If None, the projections will be displayed in the same order
            the user inputs them. If 'all' the projections will be plotted from largest to smallest
            so every point is visable. If 'dominant' the projections will be plotted from smallest 
            to largest so only the dominant projection is shown.
        color_list (list): List of colors that is the same length as the number of projections
            in the plot.
        legend (bool): Determines if the legend should be included or not.
        linewidth (float): Line width of the plain band structure plotted in the background
        band_color (string): Color of the plain band structure
        unprojected_band_color (str): Color of the unprojected band
        unprojected_linewidth (float): Line width of the unprojected bands
        figsize (list / tuple): Desired size of the image in inches (width, height)
        width_ratios (list / tuple): Width ration of the band plot and dos plot. 
        erange (list / tuple): Range of energy to show in the plot [low, high]
        kpath (list[list]): High symmetry k-point path of band structure calculation
            Due to the nature of the KPOINTS file for unfolded calculations this
            information is a required input for proper labeling of the figure
            for unfolded calculations. This information is extracted from the KPOINTS
            files for non-unfolded calculations. (G is automatically converted to \\Gamma)
	    (e.g. For the path X-G-X, kpath=[['X', 'G'], ['G', 'X']])
	custom_kpath (list): This gives the option to only plot specific segments of a given band structure
	    calculation. For example if the kpath was G-X-W-L then there are three segements to choose from:
	    G-X, X-W, and W-L. In this case the default kpath could be plotted by defining custom_kpath=[1,2,3],
	    where 1 -> G-X, 2 -> X-W, and 3 -> W-L. If only G-X and X-W were desired then custom_kpath=[1,2].
	    If one of the segements should be flipped it can be done by making its value negative
	    (e.g. -1 -> X-G, -2 -> W-X, -3 -> L-W)
        n (int): Number of points between each high symmetry points.
            This is also only required for unfolded calculations and band unfolding. This number should be 
            known by the user, as it was used to generate the KPOINTS file.
        unfold (bool): Determines if the plotted band structure is from a band unfolding calculation.
        M (list[list]): Transformation matrix from the primitive bulk structure to the slab structure.
            Only required for a band unfolding calculation.
        high_symm_points (list[list]): List of fractional coordinated for each high symmetry point in 
            the band structure path. Only required for a band unfolding calculation.
        fontsize (float): Font size of the text in the figure.
        annotations (list): Annotations to put on the top and bottom (left and right) figures.
            By default it will show the spin up and spin down arrows.
        annotation_xy (list / tuple): Fractional (x, y) coordinated of the annotation location
        fill (bool): Determines wether or not to fill underneath the plot
        alpha (float): Alpha value for the fill
        sigma (float): Standard deviation for gaussian filter
        save (bool): Determines whether to automatically save the figure or not. If not 
            the figure and axis are return for further manipulation.

    Returns:
        If save == True, this function will return nothing and directly save the image as
        the output name. If save == False, the function will return the matplotlib figure
        and axis for further editing. 
    """

    fig, ax = plt.subplots(
        nrows=2,
        ncols=2,
        sharey=True,
        figsize=figsize,
        dpi=400,
        gridspec_kw={'width_ratios': width_ratios}
    )

    ax_band_up, ax_dos_up, ax_band_down, ax_dos_down = _figure_setup_band_dos_spin_polarized(
        ax=ax,
        fontsize=fontsize,
        ylim=[erange[0], erange[1]]
    )

    band_up = Band(
        folder=band_folder,
        spin='up',
        projected=True,
        kpath=kpath,
	custom_kpath=custom_kpath,
        n=n,
    )

    band_down = Band(
        folder=band_folder,
        spin='down',
        projected=True,
        kpath=kpath,
	custom_kpath=custom_kpath,
        n=n,
    )

    dos_up = Dos(shift_efermi=shift_efermi, folder=dos_folder, spin='up')
    dos_down = Dos(shift_efermi=shift_efermi, folder=dos_folder, spin='down')

    bbox = dict(boxstyle='round', fc='white',
                edgecolor='gray', alpha=0.95, pad=0.3)

    ax_band_up.annotate(
        annotations[0],
        xy=annotation_xy,
        xycoords='axes fraction',
        va='top',
        ha='left',
        bbox=bbox,
        fontsize=fontsize + 1,
    )
    ax_band_down.annotate(
        annotations[1],
        xy=annotation_xy,
        xycoords='axes fraction',
        va='top',
        ha='left',
        bbox=bbox,
        fontsize=fontsize + 1,
    )

    band_up.plot_elements(
        ax=ax_band_up,
        scale_factor=scale_factor,
        elements=elements,
        color_list=color_list,
        legend=False,
        linewidth=linewidth,
        band_color=band_color,
    )

    band_down.plot_plain(
        ax=ax_band_up,
        color=unprojected_band_color,
        linewidth=unprojected_linewidth,
    )

    dos_up.plot_elements(
        ax=ax_dos_up,
        elements=elements,
        fill=fill,
        alpha=alpha,
        linewidth=linewidth,
        sigma=sigma,
        energyaxis='y',
        color_list=color_list,
        legend=legend,
        total=True,
        erange=erange,
    )

    dos_down.plot_elements(
        ax=ax_dos_up,
        elements=elements,
        fill=fill,
        alpha=0.25 * alpha,
        alpha_line=0.25 * alpha,
        linewidth=linewidth,
        sigma=sigma,
        energyaxis='y',
        color_list=color_list,
        legend=False,
        total=True,
        erange=erange,
    )

    band_down.plot_elements(
        ax=ax_band_down,
        scale_factor=scale_factor,
        elements=elements,
        color_list=color_list,
        legend=False,
        linewidth=linewidth,
        band_color=band_color,
    )

    band_up.plot_plain(
        ax=ax_band_down,
        color=unprojected_band_color,
        linewidth=unprojected_linewidth,
    )

    dos_down.plot_elements(
        ax=ax_dos_down,
        elements=elements,
        fill=fill,
        alpha=alpha,
        linewidth=linewidth,
        sigma=sigma,
        energyaxis='y',
        color_list=color_list,
        legend=legend,
        total=True,
        erange=erange,
    )

    dos_up.plot_elements(
        ax=ax_dos_down,
        elements=elements,
        fill=fill,
        alpha=0.25 * alpha,
        alpha_line=0.25 * alpha,
        linewidth=linewidth,
        sigma=sigma,
        energyaxis='y',
        color_list=color_list,
        legend=False,
        total=True,
        erange=erange,
    )

    fig.canvas.draw()
    nbins = len(ax_dos_down.get_xticklabels())
    ax_dos_down.xaxis.set_major_locator(
        MaxNLocator(nbins=nbins - 1, prune='lower'))

    plt.tight_layout(pad=0.4)
    plt.subplots_adjust(wspace=0, hspace=0.05)

    if save:
        plt.savefig(output)
    else:
        return fig, ax_band_up, ax_dos_up, ax_band_down, ax_dos_down


def band_dos_element_orbitals_spin_polarized(
    band_folder,
    dos_folder,
    element_orbital_dict,
    output='band_dos_element_orbitals_sp.png',
    scale_factor=5,
    color_list=None,
    legend=True,
    linewidth=0.75,
    band_color='black',
    unprojected_band_color='gray',
    unprojected_linewidth=0.6,
    figsize=(8, 6),
    width_ratios=[7, 3],
    erange=[-6, 6],
    kpath=None,
    custom_kpath=None,
    n=None,
    fontsize=8,
    annotations=['$\\uparrow$ ', '$\\downarrow$ '],
    annotation_xy=(0.0125, 0.98),
    save=True,
    shift_efermi=0,
    interpolate=False,
    new_n=200,
    fill=True,
    alpha=0.3,
    sigma=0.05,
):
    """
    This function plots a spin polarized band structure projected onto specified [element, orbital] dict next to a spin
    polarized density of states projected onto the same [element, orbital] dict. The top figure highlights the spin up
    bands and the bottom figure highlights the spin down bands.

    Parameters:
        band_folder (str): This is the folder that contains the VASP files for the band structure
        dos_folder (str): This is the folder that contains the VASP files for the density of states
        element_orbital_dict (dict[str:list]): A dictionary that contains the individual elements and the corresponding 
            orbitals to project onto. For example, if the user wants to project onto the s, py, pz, and px orbitals
            of In and the s orbital of As for and InAs structure then the dictionary would be {'In':[0,1,2,3], 'As':[0]}
        output (str): File name of the resulting plot.
        scale_factor (float): Factor to scale weights. This changes the size of the
            points in the scatter plot
        display_order (str / None): If None, the projections will be displayed in the same order
            the user inputs them. If 'all' the projections will be plotted from largest to smallest
            so every point is visable. If 'dominant' the projections will be plotted from smallest 
            to largest so only the dominant projection is shown.
        color_list (list): List of colors that is the same length as the number of projections
            in the plot.
        legend (bool): Determines if the legend should be included or not.
        linewidth (float): Line width of the plain band structure plotted in the background
        band_color (string): Color of the plain band structure
        unprojected_band_color (str): Color of the unprojected band
        unprojected_linewidth (float): Line width of the unprojected bands
        figsize (list / tuple): Desired size of the image in inches (width, height)
        width_ratios (list / tuple): Width ration of the band plot and dos plot. 
        erange (list / tuple): Range of energy to show in the plot [low, high]
        kpath (list[list]): High symmetry k-point path of band structure calculation
            Due to the nature of the KPOINTS file for unfolded calculations this
            information is a required input for proper labeling of the figure
            for unfolded calculations. This information is extracted from the KPOINTS
            files for non-unfolded calculations. (G is automatically converted to \\Gamma)
	    (e.g. For the path X-G-X, kpath=[['X', 'G'], ['G', 'X']])
	custom_kpath (list): This gives the option to only plot specific segments of a given band structure
	    calculation. For example if the kpath was G-X-W-L then there are three segements to choose from:
	    G-X, X-W, and W-L. In this case the default kpath could be plotted by defining custom_kpath=[1,2,3],
	    where 1 -> G-X, 2 -> X-W, and 3 -> W-L. If only G-X and X-W were desired then custom_kpath=[1,2].
	    If one of the segements should be flipped it can be done by making its value negative
	    (e.g. -1 -> X-G, -2 -> W-X, -3 -> L-W)
        n (int): Number of points between each high symmetry points.
            This is also only required for unfolded calculations and band unfolding. This number should be 
            known by the user, as it was used to generate the KPOINTS file.
        unfold (bool): Determines if the plotted band structure is from a band unfolding calculation.
        M (list[list]): Transformation matrix from the primitive bulk structure to the slab structure.
            Only required for a band unfolding calculation.
        high_symm_points (list[list]): List of fractional coordinated for each high symmetry point in 
            the band structure path. Only required for a band unfolding calculation.
        fontsize (float): Font size of the text in the figure.
        annotations (list): Annotations to put on the top and bottom (left and right) figures.
            By default it will show the spin up and spin down arrows.
        annotation_xy (list / tuple): Fractional (x, y) coordinated of the annotation location
        fill (bool): Determines wether or not to fill underneath the plot
        alpha (float): Alpha value for the fill
        sigma (float): Standard deviation for gaussian filter
        save (bool): Determines whether to automatically save the figure or not. If not 
            the figure and axis are return for further manipulation.

    Returns:
        If save == True, this function will return nothing and directly save the image as
        the output name. If save == False, the function will return the matplotlib figure
        and axis for further editing. 
    """

    fig, ax = plt.subplots(
        nrows=2,
        ncols=2,
        sharey=True,
        figsize=figsize,
        dpi=400,
        gridspec_kw={'width_ratios': width_ratios}
    )

    ax_band_up, ax_dos_up, ax_band_down, ax_dos_down = _figure_setup_band_dos_spin_polarized(
        ax=ax,
        fontsize=fontsize,
        ylim=[erange[0], erange[1]]
    )

    band_up = Band(
        folder=band_folder,
        spin='up',
        projected=True,
        kpath=kpath,
	custom_kpath=custom_kpath,
        n=n,
    )

    band_down = Band(
        folder=band_folder,
        spin='down',
        projected=True,
        kpath=kpath,
	custom_kpath=custom_kpath,
        n=n,
    )

    dos_up = Dos(shift_efermi=shift_efermi, folder=dos_folder, spin='up')
    dos_down = Dos(shift_efermi=shift_efermi, folder=dos_folder, spin='down')

    bbox = dict(boxstyle='round', fc='white',
                edgecolor='gray', alpha=0.95, pad=0.3)

    ax_band_up.annotate(
        annotations[0],
        xy=annotation_xy,
        xycoords='axes fraction',
        va='top',
        ha='left',
        bbox=bbox,
        fontsize=fontsize + 1,
    )
    ax_band_down.annotate(
        annotations[1],
        xy=annotation_xy,
        xycoords='axes fraction',
        va='top',
        ha='left',
        bbox=bbox,
        fontsize=fontsize + 1,
    )

    band_up.plot_element_orbitals(
        ax=ax_band_up,
        scale_factor=scale_factor,
        element_orbital_dict=element_orbital_dict,
        color_list=color_list,
        legend=False,
        linewidth=linewidth,
        band_color=band_color,
    )

    band_down.plot_plain(
        ax=ax_band_up,
        color=unprojected_band_color,
        linewidth=unprojected_linewidth,
    )

    dos_up.plot_element_orbitals(
        ax=ax_dos_up,
        element_orbital_dict=element_orbital_dict,
        fill=fill,
        alpha=alpha,
        linewidth=linewidth,
        sigma=sigma,
        energyaxis='y',
        color_list=color_list,
        legend=legend,
        total=True,
        erange=erange,
    )

    dos_down.plot_element_orbitals(
        ax=ax_dos_up,
        element_orbital_dict=element_orbital_dict,
        fill=fill,
        alpha=0.25 * alpha,
        alpha_line=0.25 * alpha,
        linewidth=linewidth,
        sigma=sigma,
        energyaxis='y',
        color_list=color_list,
        legend=False,
        total=True,
        erange=erange,
    )

    band_down.plot_element_orbitals(
        ax=ax_band_down,
        scale_factor=scale_factor,
        element_orbital_dict=element_orbital_dict,
        color_list=color_list,
        legend=False,
        linewidth=linewidth,
        band_color=band_color,
    )

    band_up.plot_plain(
        ax=ax_band_down,
        color=unprojected_band_color,
        linewidth=unprojected_linewidth,
    )

    dos_down.plot_element_orbitals(
        ax=ax_dos_down,
        element_orbital_dict=element_orbital_dict,
        fill=fill,
        alpha=alpha,
        linewidth=linewidth,
        sigma=sigma,
        energyaxis='y',
        color_list=color_list,
        legend=legend,
        total=True,
        erange=erange,
    )

    dos_up.plot_element_orbitals(
        ax=ax_dos_down,
        element_orbital_dict=element_orbital_dict,
        fill=fill,
        alpha=0.25 * alpha,
        alpha_line=0.25 * alpha,
        linewidth=linewidth,
        sigma=sigma,
        energyaxis='y',
        color_list=color_list,
        legend=False,
        total=True,
        erange=erange,
    )

    fig.canvas.draw()
    nbins = len(ax_dos_down.get_xticklabels())
    ax_dos_down.xaxis.set_major_locator(
        MaxNLocator(nbins=nbins - 1, prune='lower'))

    plt.tight_layout(pad=0.4)
    plt.subplots_adjust(wspace=0, hspace=0.05)

    if save:
        plt.savefig(output)
    else:
        return fig, ax_band_up, ax_dos_up, ax_band_down, ax_dos_down


def band_dos_element_spd_spin_polarized(
    band_folder,
    dos_folder,
    element_spd_dict,
    output='band_dos_element_spd_sp.png',
    scale_factor=5,
    color_list=None,
    legend=True,
    linewidth=0.75,
    band_color='black',
    unprojected_band_color='gray',
    unprojected_linewidth=0.6,
    figsize=(8, 6),
    width_ratios=[7, 3],
    erange=[-6, 6],
    kpath=None,
    custom_kpath=None,
    n=None,
    fontsize=8,
    annotations=['$\\uparrow$ ', '$\\downarrow$ '],
    annotation_xy=(0.0125, 0.98),
    save=True,
    shift_efermi=0,
    interpolate=False,
    new_n=200,
    fill=True,
    alpha=0.3,
    sigma=0.05,
):
    """
    This function plots a spin polarized s, p, d projected band structure on a given element next to a spin polarized
    s, p, d projected density of states on the same element. The top figure highlights the spin up bands and the bottom
    figure highlights the spin down bands.

    Parameters:
        band_folder (str): This is the folder that contains the VASP files for the band structure
        dos_folder (str): This is the folder that contains the VASP files for the density of states
        element_spd_dict (dict[str:str]): A dictionary that contains the individual atoms and the corresponding 
            orbitals to project onto. For example, if the user wants to project onto the s, p, d orbitals
            of In and the p orbitals of As for an InAs structure then the dictionary would be {'In':'spd', 'As':'p'}
        output (str): File name of the resulting plot.
        scale_factor (float): Factor to scale weights. This changes the size of the
            points in the scatter plot
        display_order (str / None): If None, the projections will be displayed in the same order
            the user inputs them. If 'all' the projections will be plotted from largest to smallest
            so every point is visable. If 'dominant' the projections will be plotted from smallest 
            to largest so only the dominant projection is shown.
        color_list (list): List of colors that is the same length as the number of projections
            in the plot.
        legend (bool): Determines if the legend should be included or not.
        linewidth (float): Line width of the plain band structure plotted in the background
        band_color (string): Color of the plain band structure
        unprojected_band_color (str): Color of the unprojected band
        unprojected_linewidth (float): Line width of the unprojected bands
        figsize (list / tuple): Desired size of the image in inches (width, height)
        width_ratios (list / tuple): Width ration of the band plot and dos plot. 
        erange (list / tuple): Range of energy to show in the plot [low, high]
        kpath (list[list]): High symmetry k-point path of band structure calculation
            Due to the nature of the KPOINTS file for unfolded calculations this
            information is a required input for proper labeling of the figure
            for unfolded calculations. This information is extracted from the KPOINTS
            files for non-unfolded calculations. (G is automatically converted to \\Gamma)
	    (e.g. For the path X-G-X, kpath=[['X', 'G'], ['G', 'X']])
	custom_kpath (list): This gives the option to only plot specific segments of a given band structure
	    calculation. For example if the kpath was G-X-W-L then there are three segements to choose from:
	    G-X, X-W, and W-L. In this case the default kpath could be plotted by defining custom_kpath=[1,2,3],
	    where 1 -> G-X, 2 -> X-W, and 3 -> W-L. If only G-X and X-W were desired then custom_kpath=[1,2].
	    If one of the segements should be flipped it can be done by making its value negative
	    (e.g. -1 -> X-G, -2 -> W-X, -3 -> L-W)
        n (int): Number of points between each high symmetry points.
            This is also only required for unfolded calculations and band unfolding. This number should be 
            known by the user, as it was used to generate the KPOINTS file.
        unfold (bool): Determines if the plotted band structure is from a band unfolding calculation.
        M (list[list]): Transformation matrix from the primitive bulk structure to the slab structure.
            Only required for a band unfolding calculation.
        high_symm_points (list[list]): List of fractional coordinated for each high symmetry point in 
            the band structure path. Only required for a band unfolding calculation.
        fontsize (float): Font size of the text in the figure.
        annotations (list): Annotations to put on the top and bottom (left and right) figures.
            By default it will show the spin up and spin down arrows.
        annotation_xy (list / tuple): Fractional (x, y) coordinated of the annotation location
        fill (bool): Determines wether or not to fill underneath the plot
        alpha (float): Alpha value for the fill
        sigma (float): Standard deviation for gaussian filter
        save (bool): Determines whether to automatically save the figure or not. If not 
            the figure and axis are return for further manipulation.

    Returns:
        If save == True, this function will return nothing and directly save the image as
        the output name. If save == False, the function will return the matplotlib figure
        and axis for further editing. 
    """

    fig, ax = plt.subplots(
        nrows=2,
        ncols=2,
        sharey=True,
        figsize=figsize,
        dpi=400,
        gridspec_kw={'width_ratios': width_ratios}
    )

    ax_band_up, ax_dos_up, ax_band_down, ax_dos_down = _figure_setup_band_dos_spin_polarized(
        ax=ax,
        fontsize=fontsize,
        ylim=[erange[0], erange[1]]
    )

    band_up = Band(
        folder=band_folder,
        spin='up',
        projected=True,
        kpath=kpath,
	custom_kpath=custom_kpath,
        n=n,
    )

    band_down = Band(
        folder=band_folder,
        spin='down',
        projected=True,
        kpath=kpath,
	custom_kpath=custom_kpath,
        n=n,
    )

    dos_up = Dos(shift_efermi=shift_efermi, folder=dos_folder, spin='up')
    dos_down = Dos(shift_efermi=shift_efermi, folder=dos_folder, spin='down')

    bbox = dict(boxstyle='round', fc='white',
                edgecolor='gray', alpha=0.95, pad=0.3)

    ax_band_up.annotate(
        annotations[0],
        xy=annotation_xy,
        xycoords='axes fraction',
        va='top',
        ha='left',
        bbox=bbox,
        fontsize=fontsize + 1,
    )
    ax_band_down.annotate(
        annotations[1],
        xy=annotation_xy,
        xycoords='axes fraction',
        va='top',
        ha='left',
        bbox=bbox,
        fontsize=fontsize + 1,
    )

    band_up.plot_element_spd(
        ax=ax_band_up,
        element_spd_dict=element_spd_dict,
        scale_factor=scale_factor,
        color_list=color_list,
        legend=False,
        linewidth=linewidth,
        band_color=band_color,
    )

    band_down.plot_plain(
        ax=ax_band_up,
        color=unprojected_band_color,
        linewidth=unprojected_linewidth,
    )

    dos_up.plot_element_spd(
        ax=ax_dos_up,
        element_spd_dict=element_spd_dict,
        fill=fill,
        alpha=alpha,
        linewidth=linewidth,
        sigma=sigma,
        energyaxis='y',
        color_list=color_list,
        legend=legend,
        total=True,
        erange=erange,
    )

    dos_down.plot_element_spd(
        ax=ax_dos_up,
        element_spd_dict=element_spd_dict,
        fill=fill,
        alpha=0.25 * alpha,
        alpha_line=0.25 * alpha,
        linewidth=linewidth,
        sigma=sigma,
        energyaxis='y',
        color_list=color_list,
        legend=False,
        total=True,
        erange=erange,
    )

    band_down.plot_element_spd(
        ax=ax_band_down,
        element_spd_dict=element_spd_dict,
        scale_factor=scale_factor,
        color_list=color_list,
        legend=False,
        linewidth=linewidth,
        band_color=band_color,
    )

    band_up.plot_plain(
        ax=ax_band_down,
        color=unprojected_band_color,
        linewidth=unprojected_linewidth,
    )

    dos_down.plot_element_spd(
        ax=ax_dos_down,
        element_spd_dict=element_spd_dict,
        fill=fill,
        alpha=alpha,
        linewidth=linewidth,
        sigma=sigma,
        energyaxis='y',
        color_list=color_list,
        legend=legend,
        total=True,
        erange=erange,
    )

    dos_up.plot_element_spd(
        ax=ax_dos_down,
        element_spd_dict=element_spd_dict,
        fill=fill,
        alpha=0.25 * alpha,
        alpha_line=0.25 * alpha,
        linewidth=linewidth,
        sigma=sigma,
        energyaxis='y',
        color_list=color_list,
        legend=False,
        total=True,
        erange=erange,
    )

    fig.canvas.draw()
    nbins = len(ax_dos_down.get_xticklabels())
    ax_dos_down.xaxis.set_major_locator(
        MaxNLocator(nbins=nbins - 1, prune='lower'))

    plt.tight_layout(pad=0.4)
    plt.subplots_adjust(wspace=0, hspace=0.05)

    if save:
        plt.savefig(output)
    else:
        return fig, ax_band_up, ax_dos_up, ax_band_down, ax_dos_down


def dos_layers(
    folder,
    output='dos_layers.png',
    energyaxis='y',
    figsize=(5, 3),
    erange=[-3, 3],
    lrange=None,
    spin='up',
    soc_axis=None,
    combination_method='add',
    fontsize=12,
    save=True,
    shift_efermi=0,
    cmap='magma',
    sigma_energy=0.05,
    sigma_layers=0.75,
    antialiased=False,
    show_structure=False,
    interface_layer=None,
    interface_line_color='white',
    interface_line_style='--',
    interface_line_width=2,
    log_scale=True,
    contour=False,
    levels=10,
    min_cutoff=1e-7,
    max_cutoff=None,
    atol=None,
    custom_layer_inds=None,
    custom_cbar_label=None,
    plot_vbm_cbm=False,
    vbm_color='white',
    cbm_color='white',
    vbm_linestyle=':',
    cbm_linestyle=':',
    cbar_orientation='vertical',
    show_bounds=False,
    set_bounds=None,
    sp_method='percentage',
):
    """
    This function is used to plot a layer by layer density of states heat map for slab structures. It is useful for
    visualizing band alignment of interfaces.

    Parameters:
        folder (str): This is the folder that contains the VASP files
        output (str): File name of the resulting plot.
        figsize (list / tuple): Desired size of the image in inches (width, height)
        spin (str): Which spin direction to parse ('up' or 'down')
        combination_method (str): If spin == 'both', this determines if the spin up and spin down
            desnities are added or subtracted. ('add' or 'sub')
        show_structure (bool): If true, a projected of the structure will be plotted next to the
            layer by later dos. This feature works best with large slab structures, and has not
            been fully tested.
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
        max_cutoff (float): Maximum dos value used to determine the cut off for the plot.
            This can be adjusted to better visualize the band gap of the material.
        atol (float or None): Tolarence used in the grouping of the layers.
            This value is automatically calculated if None and is usually on the order of
            1e-3.
        custom_layer_inds (list or None): If the structure being calculated has relaxed
            atomic positions, sometimes the layer grouping algorithm can behave non-idealy.
            If this is the case, the user can input a list of list that contain the
            atomic indices in each layers of the material.
        custom_cbar_label (str or None): Custom label for the colorbar
        save (bool): Determines whether to automatically save the figure or not. If not 
            the figure and axis are return for further manipulation.
        plot_vbm_cbm (bool): Determines if the valence band maximum and conduction band minimum
            are plotted on the graph.
        vbm_color (str): Color of the vbm line.
        cbm_color (str): Color of the cbm line.
        vbm_linestyle (str): Linestyle of the vbm line.
        cbm_linestyle (str): Linestyle of the cbm line.

    Returns:
        If save == True, this function will return nothing and directly save the image as
        the output name. If save == False, the function will return the matplotlib figure
        and axis for further editing. 
    """
    if show_structure:
        if energyaxis == 'x':
            fig, axs = plt.subplots(
                nrows=1,
                ncols=2,
                gridspec_kw={'width_ratios':[1,6]},
                figsize=figsize,
                dpi=400,
            )
            structure_ax = axs[0]
            dos_ax = axs[1]
            
        elif energyaxis == 'y':
            fig, axs = plt.subplots(
                nrows=2,
                ncols=1,
                gridspec_kw={'height_ratios':[1,6], 'hspace': -0.02},
                figsize=figsize,
                dpi=400,
            )
            structure_ax = axs[0]
            sm = plt.cm.ScalarMappable()
            sm.set_array([])
            fig.colorbar(sm, ax=structure_ax).ax.set_visible(False)
            dos_ax = axs[1]

        structure_ax.spines['left'].set_visible(False)
        structure_ax.spines['right'].set_visible(False)
        structure_ax.spines['top'].set_visible(False)
        structure_ax.spines['bottom'].set_visible(False)

        structure_ax.tick_params(
            left=False,
            bottom=False,
            labelleft=False,
            labelbottom=False,
        )

    else:
        fig = plt.figure(figsize=figsize, dpi=400)
        dos_ax = fig.add_subplot(111)
        
    _figure_setup_layer_dos(ax=dos_ax, fontsize=fontsize, energyaxis=energyaxis)

    dos = Dos(
        shift_efermi=shift_efermi,
        folder=folder,
        spin=spin,
        soc_axis=soc_axis,
        combination_method=combination_method,
        sp_method=sp_method,
    )
    dos.plot_layers(
        ax=dos_ax,
        sigma_layers=sigma_layers,
        sigma_energy=sigma_energy,
        cmap=cmap,
        erange=erange,
        lrange=lrange,
        antialiased=antialiased,
        fontsize=fontsize,
        energyaxis=energyaxis,
        interface_layer=interface_layer,
        interface_line_color=interface_line_color,
        interface_line_style=interface_line_style,
        interface_line_width=interface_line_width,
        log_scale=log_scale,
        contour=contour,
        levels=levels,
        min_cutoff=min_cutoff,
        max_cutoff=max_cutoff,
        atol=atol,
        custom_cbar_label=custom_cbar_label,
        custom_layer_inds=custom_layer_inds,
        cbar_orientation=cbar_orientation,
        show_bounds=show_bounds,
        set_bounds=set_bounds,
    )

    if plot_vbm_cbm:
        _, vbm, cbm = get_bandgap(folder=folder, printbg=False, return_vbm_cbm=True)

        if energyaxis == 'y':
            dos_ax.axhline(
                y=cbm,
                color=cbm_color,
                linestyle=cbm_linestyle,
            )
            dos_ax.axhline(
                y=vbm,
                color=vbm_color,
                linestyle=vbm_linestyle,
            )
        elif energyaxis == 'x':
            dos_ax.axvline(
                x=cbm,
                color=cbm_color,
                linestyle=cbm_linestyle,
            )
            dos_ax.axvline(
                x=vbm,
                color=vbm_color,
                linestyle=vbm_linestyle,
            )

    if energyaxis == 'y':
        dos_ax.set_ylim(erange[0], erange[1])
    elif energyaxis == 'x':
        dos_ax.set_xlim(erange[0], erange[1])

    if show_structure:
        if energyaxis == 'y':
            dos.plot_structure(ax=structure_ax, rotation=[90,90,90])
        elif energyaxis == 'x':
            dos.plot_structure(ax=structure_ax, rotation=[0,90,90])
            fig.tight_layout(pad=0.4)

    if save:
        plt.savefig(output, bbox_inches='tight')
    else:
        if show_structure:
            return fig, dos_ax, structure_ax
        else:
            return fig, dos_ax


def _main():
    #  band_plain_spin_polarized(
        #  folder='../../vaspvis_data/band_Cr2O3',
        #  figsize=(7,3),
        #  #  interpolate=True
    #  )
    band_spd(
        folder='../../vaspvis_data/band_InAs',
        interpolate=False,
        custom_kpath=[1,2,3,4,5,-5,-4,-3,-2,-1]
    )


if __name__ == "__main__":
    _main()
