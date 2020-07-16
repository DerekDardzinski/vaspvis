from .band import Band
from .dos import Dos
import matplotlib.pyplot as plt


def _figure_setup(ax, fontsize=6, ylim=[-6, 6]):
    ax.set_ylabel('$E - E_{F}$ $(eV)$', fontsize=fontsize)
    ax.set_ylim(ylim[0], ylim[1])
    ax.tick_params(labelsize=fontsize, length=2.5)
    ax.tick_params(axis='x', length=0)


def _figure_setup_dos(ax, fontsize=6, energyaxis='y'):
    ax.tick_params(labelsize=fontsize, length=2.5)
    if energyaxis == 'y':
        ax.set_ylabel('$E - E_{F}$ $(eV)$', fontsize=fontsize)
        ax.set_xlabel('Density of States', fontsize=fontsize)
    if energyaxis == 'x':
        ax.set_xlabel('$E - E_{F}$ $(eV)$', fontsize=fontsize)
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


def band_plain(
    folder,
    output='band_plain.png',
    spin='up',
    color='black',
    linewidth=1.25,
    linestyle='-',
    figsize=(4, 3),
    erange=[-6, 6],
    hse=False,
    kpath=None,
    n=None,
    fontsize=7,
    save=True,
):

    band = Band(
        folder=folder,
        spin=spin,
        hse=hse,
        kpath=kpath,
        n=n,
    )
    fig = plt.figure(figsize=(figsize), dpi=300)
    ax = fig.add_subplot(111)
    _figure_setup(ax=ax, fontsize=fontsize, ylim=[erange[0], erange[1]])
    band.plot_plain(
        ax=ax,
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
    )
    plt.tight_layout(pad=0.2)

    if save:
        plt.savefig(output)
    else:
        return fig, ax


def band_spd(
    folder,
    output='band_spd.png',
    spin='up',
    scale_factor=5,
    order=['s', 'p', 'd'],
    color_dict=None,
    legend=True,
    linewidth=0.75,
    band_color='black',
    figsize=(4, 3),
    erange=[-6, 6],
    hse=False,
    kpath=None,
    n=None,
    fontsize=7,
    save=True,
):

    band = Band(
        folder=folder,
        spin=spin,
        projected=True,
        hse=hse,
        kpath=kpath,
        n=n,
    )
    fig = plt.figure(figsize=(figsize), dpi=300)
    ax = fig.add_subplot(111)
    _figure_setup(ax=ax, fontsize=fontsize, ylim=[erange[0], erange[1]])
    band.plot_spd(
        ax=ax,
        scale_factor=scale_factor,
        order=order,
        color_dict=color_dict,
        legend=legend,
        linewidth=linewidth,
        band_color=band_color,
    )
    plt.tight_layout(pad=0.2)

    if save:
        plt.savefig(output)
    else:
        return fig, ax


def band_atom_orbital(
    folder,
    atom_orbital_pairs,
    output='band_atom_orbital.png',
    spin='up',
    scale_factor=5,
    color_list=None,
    legend=True,
    linewidth=0.75,
    band_color='black',
    figsize=(4, 3),
    erange=[-6, 6],
    hse=False,
    kpath=None,
    n=None,
    fontsize=7,
    save=True,
):

    band = Band(
        folder=folder,
        spin=spin,
        projected=True,
        hse=hse,
        kpath=kpath,
        n=n,
    )
    fig = plt.figure(figsize=(figsize), dpi=300)
    ax = fig.add_subplot(111)
    _figure_setup(ax=ax, fontsize=fontsize, ylim=[erange[0], erange[1]])
    band.plot_atom_orbitals(
        ax=ax,
        atom_orbital_pairs=atom_orbital_pairs,
        scale_factor=scale_factor,
        color_list=color_list,
        legend=legend,
        linewidth=linewidth,
        band_color=band_color,
    )
    plt.tight_layout(pad=0.2)

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
    color_dict=None,
    legend=True,
    linewidth=0.75,
    band_color='black',
    figsize=(4, 3),
    erange=[-6, 6],
    hse=False,
    kpath=None,
    n=None,
    fontsize=7,
    save=True,
):

    band = Band(
        folder=folder,
        spin=spin,
        projected=True,
        hse=hse,
        kpath=kpath,
        n=n,
    )
    fig = plt.figure(figsize=(figsize), dpi=300)
    ax = fig.add_subplot(111)
    _figure_setup(ax=ax, fontsize=fontsize, ylim=[erange[0], erange[1]])
    band.plot_orbitals(
        ax=ax,
        orbitals=orbitals,
        scale_factor=scale_factor,
        color_dict=color_dict,
        legend=legend,
        linewidth=linewidth,
        band_color=band_color,
    )
    plt.tight_layout(pad=0.2)

    if save:
        plt.savefig(output)
    else:
        return fig, ax


def band_atoms(
    folder,
    atoms,
    output='band_orbital.png',
    spin='up',
    scale_factor=5,
    color_list=None,
    legend=True,
    linewidth=0.75,
    band_color='black',
    figsize=(4, 3),
    erange=[-6, 6],
    hse=False,
    kpath=None,
    n=None,
    fontsize=7,
    save=True,
):

    band = Band(
        folder=folder,
        spin=spin,
        projected=True,
        hse=hse,
        kpath=kpath,
        n=n,
    )
    fig = plt.figure(figsize=(figsize), dpi=300)
    ax = fig.add_subplot(111)
    _figure_setup(ax=ax, fontsize=fontsize, ylim=[erange[0], erange[1]])
    band.plot_atoms(
        ax=ax,
        atoms=atoms,
        scale_factor=scale_factor,
        color_list=color_list,
        legend=legend,
        linewidth=linewidth,
        band_color=band_color,
    )
    plt.tight_layout(pad=0.2)

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
    color_list=None,
    legend=True,
    linewidth=0.75,
    band_color='black',
    figsize=(4, 3),
    erange=[-6, 6],
    hse=False,
    kpath=None,
    n=None,
    fontsize=7,
    save=True,
):

    band = Band(
        folder=folder,
        spin=spin,
        projected=True,
        hse=hse,
        kpath=kpath,
        n=n,
    )
    fig = plt.figure(figsize=(figsize), dpi=300)
    ax = fig.add_subplot(111)
    _figure_setup(ax=ax, fontsize=fontsize, ylim=[erange[0], erange[1]])
    band.plot_elements(
        ax=ax,
        elements=elements,
        scale_factor=scale_factor,
        color_list=color_list,
        legend=legend,
        linewidth=linewidth,
        band_color=band_color,
    )
    plt.tight_layout(pad=0.2)

    if save:
        plt.savefig(output)
    else:
        return fig, ax


def band_element_orbitals(
    folder,
    element_orbital_pairs,
    output='band_element_orbital.png',
    spin='up',
    scale_factor=5,
    color_list=None,
    legend=True,
    linewidth=0.75,
    band_color='black',
    figsize=(4, 3),
    erange=[-6, 6],
    hse=False,
    kpath=None,
    n=None,
    fontsize=7,
    save=True,
):

    band = Band(
        folder=folder,
        spin=spin,
        projected=True,
        hse=hse,
        kpath=kpath,
        n=n,
    )
    fig = plt.figure(figsize=(figsize), dpi=300)
    ax = fig.add_subplot(111)
    _figure_setup(ax=ax, fontsize=fontsize, ylim=[erange[0], erange[1]])
    band.plot_element_orbitals(
        ax=ax,
        element_orbital_pairs=element_orbital_pairs,
        scale_factor=scale_factor,
        color_list=color_list,
        legend=legend,
        linewidth=linewidth,
        band_color=band_color,
    )
    plt.tight_layout(pad=0.2)

    if save:
        plt.savefig(output)
    else:
        return fig, ax


def band_element_spd(
    folder,
    elements,
    order=['s', 'p', 'd'],
    output='band_element_spd.png',
    spin='up',
    scale_factor=5,
    color_dict=None,
    legend=True,
    linewidth=0.75,
    band_color='black',
    figsize=(4, 3),
    erange=[-6, 6],
    hse=False,
    kpath=None,
    n=None,
    fontsize=7,
    save=True,
):

    band = Band(
        folder=folder,
        spin=spin,
        projected=True,
        hse=hse,
        kpath=kpath,
        n=n,
    )
    fig = plt.figure(figsize=(figsize), dpi=300)
    ax = fig.add_subplot(111)
    _figure_setup(ax=ax, fontsize=fontsize, ylim=[erange[0], erange[1]])
    band.plot_element_spd(
        ax=ax,
        elements=elements,
        order=order,
        scale_factor=scale_factor,
        color_dict=color_dict,
        legend=legend,
        linewidth=linewidth,
        band_color=band_color,
    )
    plt.tight_layout(pad=0.2)

    if save:
        plt.savefig(output)
    else:
        return fig, ax


def band_plain_spin_projected(
    folder,
    output='band_plain_sp.png',
    up_color='black',
    down_color='red',
    linewidth=1.25,
    up_linestyle='-',
    down_linestyle='-',
    figsize=(4, 3),
    erange=[-6, 6],
    hse=False,
    kpath=None,
    n=None,
    fontsize=7,
    save=True,
):

    band_up = Band(
        folder=folder,
        spin='up',
        hse=hse,
        kpath=kpath,
        n=n,
    )
    band_down = Band(
        folder=folder,
        spin='down',
        hse=hse,
        kpath=kpath,
        n=n,
    )
    fig = plt.figure(figsize=(figsize), dpi=300)
    ax = fig.add_subplot(111)
    _figure_setup(ax=ax, fontsize=fontsize, ylim=[erange[0], erange[1]])
    band_up.plot_plain(
        ax=ax,
        color=up_color,
        linewidth=linewidth,
        linestyle=up_linestyle,
    )

    band_down.plot_plain(
        ax=ax,
        color=down_color,
        linewidth=linewidth,
        linestyle=down_linestyle,
    )

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

    plt.tight_layout(pad=0.2)

    if save:
        plt.savefig(output)
    else:
        return fig, ax


def band_spd_spin_projected(
    folder,
    output='band_spd_sp.png',
    scale_factor=2,
    order=['s', 'p', 'd'],
    color_dict=None,
    legend=True,
    linewidth=0.75,
    band_color='black',
    unprojected_band_color='gray',
    unprojected_linewidth=0.6,
    fontsize=7,
    annotations=['$\\uparrow$ ', '$\\downarrow$ '],
    annotation_xy=(0.02, 0.98),
    figsize=(4, 3),
    erange=[-6, 6],
    stack='vertical',
    hse=False,
    kpath=None,
    n=None,
    save=True,
):

    band_up = Band(
        folder=folder,
        spin='up',
        projected=True,
        hse=hse,
        kpath=kpath,
        n=n,
    )

    band_down = Band(
        folder=folder,
        spin='down',
        projected=True,
        hse=hse,
        kpath=kpath,
        n=n,
    )

    if stack == 'vertical':
        fig = plt.figure(figsize=(figsize[0], 2 * figsize[1]), dpi=300)
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        _figure_setup(ax=ax1, fontsize=fontsize, ylim=[erange[0], erange[1]])
        _figure_setup(ax=ax2, fontsize=fontsize, ylim=[erange[0], erange[1]])
    elif stack == 'horizontal':
        fig = plt.figure(figsize=(2 * figsize[0], figsize[1]), dpi=300)
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
        order=order,
        color_dict=color_dict,
        legend=legend,
        linewidth=linewidth,
        band_color=band_color,
    )
    band_down.plot_plain(
        ax=ax1,
        color=unprojected_band_color,
        linewidth=unprojected_linewidth,
    )

    band_down.plot_spd(
        ax=ax2,
        scale_factor=scale_factor,
        order=order,
        color_dict=color_dict,
        legend=legend,
        linewidth=linewidth,
        band_color=band_color,
    )
    band_up.plot_plain(
        ax=ax2,
        color=unprojected_band_color,
        linewidth=unprojected_linewidth,
    )

    plt.tight_layout(pad=0.2)

    if save:
        plt.savefig(output)
    else:
        return fig, ax1, ax2


def band_atom_orbital_spin_projected(
    folder,
    atom_orbital_pairs,
    output='band_atom_orbital_sp.png',
    scale_factor=5,
    color_list=None,
    legend=True,
    linewidth=0.75,
    band_color='black',
    unprojected_band_color='gray',
    unprojected_linewidth=0.6,
    fontsize=7,
    annotations=['$\\uparrow$ ', '$\\downarrow$ '],
    annotation_xy=(0.02, 0.98),
    figsize=(4, 3),
    erange=[-6, 6],
    stack='vertical',
    hse=False,
    kpath=None,
    n=None,
    save=True,
):

    band_up = Band(
        folder=folder,
        spin='up',
        projected=True,
        hse=hse,
        kpath=kpath,
        n=n,
    )

    band_down = Band(
        folder=folder,
        spin='down',
        projected=True,
        hse=hse,
        kpath=kpath,
        n=n,
    )

    if stack == 'vertical':
        fig = plt.figure(figsize=(figsize[0], 2 * figsize[1]), dpi=300)
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        _figure_setup(ax=ax1, fontsize=fontsize, ylim=[erange[0], erange[1]])
        _figure_setup(ax=ax2, fontsize=fontsize, ylim=[erange[0], erange[1]])
    elif stack == 'horizontal':
        fig = plt.figure(figsize=(2 * figsize[0], figsize[1]), dpi=300)
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
        scale_factor=scale_factor,
        atom_orbital_pairs=atom_orbital_pairs,
        color_list=color_list,
        legend=legend,
        linewidth=linewidth,
        band_color=band_color,
    )
    band_down.plot_plain(
        ax=ax1,
        color=unprojected_band_color,
        linewidth=unprojected_linewidth,
    )

    band_down.plot_atom_orbitals(
        ax=ax2,
        scale_factor=scale_factor,
        atom_orbital_pairs=atom_orbital_pairs,
        color_list=color_list,
        legend=legend,
        linewidth=linewidth,
        band_color=band_color,
    )
    band_up.plot_plain(
        ax=ax2,
        color=unprojected_band_color,
        linewidth=unprojected_linewidth,
    )

    plt.tight_layout(pad=0.2)

    if save:
        plt.savefig(output)
    else:
        return fig, ax1, ax2


def band_orbitals_spin_projected(
    folder,
    orbitals,
    output='band_orbitals_sp.png',
    scale_factor=5,
    color_dict=None,
    legend=True,
    linewidth=0.75,
    band_color='black',
    unprojected_band_color='gray',
    figsize=(4, 3),
    erange=[-6, 6],
    stack='vertical',
    hse=False,
    kpath=None,
    n=None,
    fontsize=7,
    save=True,
):

    band_up = Band(
        folder=folder,
        spin='up',
        projected=True,
        hse=hse,
        kpath=kpath,
        n=n,
    )

    band_down = Band(
        folder=folder,
        spin='down',
        projected=True,
        hse=hse,
        kpath=kpath,
        n=n,
    )

    if stack == 'vertical':
        fig = plt.figure(figsize=(figsize[0], 2 * figsize[1]), dpi=300)
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        _figure_setup(ax=ax1, fontsize=fontsize, ylim=[erange[0], erange[1]])
        _figure_setup(ax=ax2, fontsize=fontsize, ylim=[erange[0], erange[1]])
    elif stack == 'horizontal':
        fig = plt.figure(figsize=(2 * figsize[0], figsize[1]), dpi=300)
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
        scale_factor=scale_factor,
        orbitals=orbitals,
        color_dict=color_dict,
        legend=legend,
        linewidth=linewidth,
        band_color=band_color,
    )
    band_up.plot_plain(
        ax=ax1,
        color=unprojected_band_color,
        linewidth=0.5,
    )

    band_down.plot_orbitals(
        ax=ax1,
        scale_factor=scale_factor,
        orbitals=orbitals,
        color_dict=color_dict,
        legend=legend,
        linewidth=linewidth,
        band_color=band_color,
    )
    band_down.plot_plain(
        ax=ax1,
        color=unprojected_band_color,
        linewidth=0.5,
    )

    plt.tight_layout(pad=0.2)

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
    energyaxis='y',
    color='black',
    figsize=(4, 3),
    erange=[-6, 6],
    spin='up',
    fontsize=7,
    save=True,
):

    dos = Dos(folder=folder, spin=spin)

    fig = plt.figure(figsize=figsize, dpi=300)
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

    plt.tight_layout(pad=0.2)

    if save:
        plt.savefig(output)
    else:
        return fig, ax


def dos_spd(
    folder,
    output='dos_spd.png',
    order=['s', 'p', 'd'],
    fill=True,
    alpha=0.3,
    linewidth=1.5,
    sigma=0.05,
    energyaxis='y',
    color_dict=None,
    legend=True,
    total=True,
    figsize=(4, 3),
    erange=[-6, 6],
    spin='up',
    fontsize=7,
    save=True,
):

    dos = Dos(folder=folder, spin=spin)

    fig = plt.figure(figsize=figsize, dpi=300)
    ax = fig.add_subplot(111)
    _figure_setup_dos(ax=ax, fontsize=fontsize, energyaxis=energyaxis)

    dos.plot_spd(
        ax=ax,
        order=order,
        fill=fill,
        alpha=alpha,
        linewidth=linewidth,
        sigma=sigma,
        energyaxis=energyaxis,
        color_dict=color_dict,
        legend=legend,
        total=total,
        erange=erange,
    )

    plt.tight_layout(pad=0.2)

    if save:
        plt.savefig(output)
    else:
        return fig, ax


def dos_atom_orbitals(
    folder,
    atom_orbital_pairs,
    output='dos_atom_orbitals.png',
    fill=True,
    alpha=0.3,
    linewidth=1.5,
    sigma=0.05,
    energyaxis='y',
    color_list=None,
    legend=True,
    total=True,
    figsize=(4, 3),
    erange=[-6, 6],
    spin='up',
    fontsize=7,
    save=True,
):

    dos = Dos(folder=folder, spin=spin)

    fig = plt.figure(figsize=figsize, dpi=300)
    ax = fig.add_subplot(111)
    _figure_setup_dos(ax=ax, fontsize=fontsize, energyaxis=energyaxis)

    dos.plot_atom_orbitals(
        ax=ax,
        atom_orbital_pairs=atom_orbital_pairs,
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

    plt.tight_layout(pad=0.2)

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
    energyaxis='y',
    color_dict=None,
    legend=True,
    total=True,
    figsize=(4, 3),
    erange=[-6, 6],
    spin='up',
    fontsize=7,
    save=True,
):

    dos = Dos(folder=folder, spin=spin)

    fig = plt.figure(figsize=figsize, dpi=300)
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
        color_dict=color_dict,
        legend=legend,
        total=total,
        erange=erange,
    )

    plt.tight_layout(pad=0.2)

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
    energyaxis='y',
    color_list=None,
    legend=True,
    total=True,
    figsize=(4, 3),
    erange=[-6, 6],
    spin='up',
    fontsize=7,
    save=True,
):

    dos = Dos(folder=folder, spin=spin)

    fig = plt.figure(figsize=figsize, dpi=300)
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

    plt.tight_layout(pad=0.2)

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
    energyaxis='y',
    color_list=None,
    legend=True,
    total=True,
    figsize=(4, 3),
    erange=[-6, 6],
    spin='up',
    fontsize=7,
    save=True,
):

    dos = Dos(folder=folder, spin=spin)

    fig = plt.figure(figsize=figsize, dpi=300)
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

    plt.tight_layout(pad=0.2)

    if save:
        plt.savefig(output)
    else:
        return fig, ax


def dos_element_spd(
    folder,
    elements,
    order=['s', 'p', 'd'],
    output='dos_element_spd.png',
    fill=True,
    alpha=0.3,
    linewidth=1.5,
    sigma=0.05,
    energyaxis='y',
    color_dict=None,
    legend=True,
    total=True,
    figsize=(4, 3),
    erange=[-6, 6],
    spin='up',
    fontsize=7,
    save=True,
):

    dos = Dos(folder=folder, spin=spin)

    fig = plt.figure(figsize=figsize, dpi=300)
    ax = fig.add_subplot(111)
    _figure_setup_dos(ax=ax, fontsize=fontsize, energyaxis=energyaxis)

    dos.plot_element_spd(
        ax=ax,
        elements=elements,
        order=order,
        fill=fill,
        alpha=alpha,
        linewidth=linewidth,
        sigma=sigma,
        energyaxis=energyaxis,
        color_dict=color_dict,
        legend=legend,
        total=total,
        erange=erange,
    )

    plt.tight_layout(pad=0.2)

    if save:
        plt.savefig(output)
    else:
        return fig, ax


def dos_element_orbitals(
    folder,
    element_orbital_pairs,
    output='dos_element_orbitals.png',
    fill=True,
    alpha=0.3,
    linewidth=1.5,
    sigma=0.05,
    energyaxis='y',
    color_list=None,
    legend=True,
    total=True,
    figsize=(4, 3),
    erange=[-6, 6],
    spin='up',
    fontsize=7,
    save=True,
):

    dos = Dos(folder=folder, spin=spin)

    fig = plt.figure(figsize=figsize, dpi=300)
    ax = fig.add_subplot(111)
    _figure_setup_dos(ax=ax, fontsize=fontsize, energyaxis=energyaxis)

    dos.plot_element_orbitals(
        ax=ax,
        element_orbital_pairs=element_orbital_pairs,
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

    plt.tight_layout(pad=0.2)

    if save:
        plt.savefig(output)
    else:
        return fig, ax


def dos_plain_spin_projected(
    folder,
    output='dos_plain_sp.png',
    linewidth=1.5,
    fill=True,
    alpha=0.3,
    sigma=0.05,
    energyaxis='y',
    color='black',
    figsize=(4, 3),
    erange=[-6, 6],
    fontsize=7,
    save=True,
):

    dos_up = Dos(folder=folder, spin='up')
    dos_down = Dos(folder=folder, spin='down')

    fig = plt.figure(figsize=figsize, dpi=300)
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

    plt.tight_layout(pad=0.2)

    if save:
        plt.savefig(output)
    else:
        return fig, ax


def dos_spd_spin_projected(
    folder,
    output='dos_spd_sp.png',
    order=['s', 'p', 'd'],
    fill=True,
    alpha=0.3,
    linewidth=1.5,
    sigma=0.05,
    energyaxis='y',
    color_dict=None,
    legend=True,
    total=True,
    figsize=(4, 3),
    erange=[-6, 6],
    fontsize=7,
    save=True,
):

    dos_up = Dos(folder=folder, spin='up')
    dos_down = Dos(folder=folder, spin='down')

    fig = plt.figure(figsize=figsize, dpi=300)
    ax = fig.add_subplot(111)
    _figure_setup_dos(ax=ax, fontsize=fontsize, energyaxis=energyaxis)

    dos_up.plot_spd(
        ax=ax,
        order=order,
        fill=fill,
        alpha=alpha,
        linewidth=linewidth,
        sigma=sigma,
        energyaxis=energyaxis,
        color_dict=color_dict,
        legend=legend,
        total=total,
        erange=erange,
    )

    dos_down.plot_spd(
        ax=ax,
        order=order,
        fill=fill,
        alpha=alpha,
        linewidth=linewidth,
        sigma=sigma,
        energyaxis=energyaxis,
        color_dict=color_dict,
        legend=False,
        total=total,
        erange=erange,
    )

    plt.tight_layout(pad=0.2)

    if save:
        plt.savefig(output)
    else:
        return fig, ax


def dos_atom_orbitals_spin_projected(
    folder,
    atom_orbital_pairs,
    output='dos_atom_orbitals_sp.png',
    fill=True,
    alpha=0.3,
    linewidth=1.5,
    sigma=0.05,
    energyaxis='y',
    color_list=None,
    legend=True,
    total=True,
    figsize=(4, 3),
    erange=[-6, 6],
    fontsize=7,
    save=True,
):

    dos_up = Dos(folder=folder, spin='up')
    dos_down = Dos(folder=folder, spin='down')

    fig = plt.figure(figsize=figsize, dpi=300)
    ax = fig.add_subplot(111)
    _figure_setup_dos(ax=ax, fontsize=fontsize, energyaxis=energyaxis)

    dos_up.plot_atom_orbitals(
        ax=ax,
        atom_orbital_pairs=atom_orbital_pairs,
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
        atom_orbital_pairs=atom_orbital_pairs,
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

    plt.tight_layout(pad=0.2)

    if save:
        plt.savefig(output)
    else:
        return fig, ax


def dos_orbitals_spin_projected(
    folder,
    orbitals,
    output='dos_orbitals_sp.png',
    fill=True,
    alpha=0.3,
    linewidth=1.5,
    sigma=0.05,
    energyaxis='y',
    color_dict=None,
    legend=True,
    total=True,
    figsize=(4, 3),
    erange=[-6, 6],
    fontsize=7,
    save=True,
):

    dos_up = Dos(folder=folder, spin='up')
    dos_down = Dos(folder=folder, spin='down')

    fig = plt.figure(figsize=figsize, dpi=300)
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
        color_dict=color_dict,
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
        color_dict=color_dict,
        legend=False,
        total=total,
        erange=erange,
    )

    plt.tight_layout(pad=0.2)

    if save:
        plt.savefig(output)
    else:
        return fig, ax


def dos_atoms_spin_projected(
    folder,
    atoms,
    output='dos_atoms_sp.png',
    fill=True,
    alpha=0.3,
    linewidth=1.5,
    sigma=0.05,
    energyaxis='y',
    color_list=None,
    legend=True,
    total=True,
    figsize=(4, 3),
    erange=[-6, 6],
    fontsize=7,
    save=True,
):

    dos_up = Dos(folder=folder, spin='up')
    dos_down = Dos(folder=folder, spin='down')

    fig = plt.figure(figsize=figsize, dpi=300)
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

    plt.tight_layout(pad=0.2)

    if save:
        plt.savefig(output)
    else:
        return fig, ax


def dos_elements_spin_projected(
    folder,
    elements,
    output='dos_elements_sp.png',
    fill=True,
    alpha=0.3,
    linewidth=1.5,
    sigma=0.05,
    energyaxis='y',
    color_list=None,
    legend=True,
    total=True,
    figsize=(4, 3),
    erange=[-6, 6],
    fontsize=7,
    save=True,
):

    dos_up = Dos(folder=folder, spin='up')
    dos_down = Dos(folder=folder, spin='down')

    fig = plt.figure(figsize=figsize, dpi=300)
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

    plt.tight_layout(pad=0.2)

    if save:
        plt.savefig(output)
    else:
        return fig, ax


def dos_element_spd_spin_projected(
    folder,
    elements,
    order=['s', 'p', 'd'],
    output='dos_element_spd_sp.png',
    fill=True,
    alpha=0.3,
    linewidth=1.5,
    sigma=0.05,
    energyaxis='y',
    color_dict=None,
    legend=True,
    total=True,
    figsize=(4, 3),
    erange=[-6, 6],
    fontsize=7,
    save=True,
):

    dos_up = Dos(folder=folder, spin='up')
    dos_down = Dos(folder=folder, spin='down')

    fig = plt.figure(figsize=figsize, dpi=300)
    ax = fig.add_subplot(111)
    _figure_setup_dos(ax=ax, fontsize=fontsize, energyaxis=energyaxis)

    dos_up.plot_element_spd(
        ax=ax,
        elements=elements,
        order=order,
        fill=fill,
        alpha=alpha,
        linewidth=linewidth,
        sigma=sigma,
        energyaxis=energyaxis,
        color_dict=color_dict,
        legend=legend,
        total=total,
        erange=erange,
    )

    dos_down.plot_element_spd(
        ax=ax,
        elements=elements,
        order=order,
        fill=fill,
        alpha=alpha,
        linewidth=linewidth,
        sigma=sigma,
        energyaxis=energyaxis,
        color_dict=color_dict,
        legend=False,
        total=total,
        erange=erange,
    )

    plt.tight_layout(pad=0.2)

    if save:
        plt.savefig(output)
    else:
        return fig, ax


def dos_element_orbitals_spin_projected(
    folder,
    element_orbital_pairs,
    output='dos_element_orbitals_sp.png',
    fill=True,
    alpha=0.3,
    linewidth=1.5,
    sigma=0.05,
    energyaxis='y',
    color_list=None,
    legend=True,
    total=True,
    figsize=(4, 3),
    erange=[-6, 6],
    fontsize=7,
    save=True,
):

    dos_up = Dos(folder=folder, spin='up')
    dos_down = Dos(folder=folder, spin='down')

    fig = plt.figure(figsize=figsize, dpi=300)
    ax = fig.add_subplot(111)
    _figure_setup_dos(ax=ax, fontsize=fontsize, energyaxis=energyaxis)

    dos_up.plot_element_orbitals(
        ax=ax,
        element_orbital_pairs=element_orbital_pairs,
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
        element_orbital_pairs=element_orbital_pairs,
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

    plt.tight_layout(pad=0.2)

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
    hse=False,
    kpath=None,
    n=None,
    fontsize=7,
    save=True,
    fill=True,
    alpha=0.3,
    sigma=0.05,
):
    fig, ax = plt.subplots(
        nrows=1,
        ncols=2,
        sharey=True,
        figsize=figsize,
        dpi=300,
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
        hse=hse,
        kpath=kpath,
        n=n,
    )

    dos = Dos(folder=dos_folder, spin=spin)

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

    plt.tight_layout(pad=0.2)
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
    order=['s', 'p', 'd'],
    color_dict=None,
    legend=True,
    linewidth=0.75,
    band_color='black',
    figsize=(6, 3),
    width_ratios=[7, 3],
    erange=[-6, 6],
    hse=False,
    kpath=None,
    n=None,
    fontsize=7,
    save=True,
    fill=True,
    alpha=0.3,
    sigma=0.05,
):
    fig, ax = plt.subplots(
        nrows=1,
        ncols=2,
        sharey=True,
        figsize=figsize,
        dpi=300,
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
        hse=hse,
        kpath=kpath,
        n=n,
    )

    dos = Dos(folder=dos_folder, spin=spin)

    band.plot_spd(
        ax=ax1,
        scale_factor=scale_factor,
        order=order,
        color_dict=color_dict,
        legend=False,
        linewidth=linewidth,
        band_color=band_color,
    )

    dos.plot_spd(
        ax=ax2,
        order=order,
        fill=fill,
        alpha=alpha,
        linewidth=linewidth,
        sigma=sigma,
        energyaxis='y',
        color_dict=color_dict,
        legend=legend,
        total=True,
        erange=erange,
    )

    fig.canvas.draw()
    labels = ax2.get_xticklabels()
    labels[0] = ''
    ax2.set_xticklabels(labels)

    plt.tight_layout(pad=0.2)
    plt.subplots_adjust(wspace=0)

    if save:
        plt.savefig(output)
    else:
        return fig, ax1, ax2


def band_dos_atom_orbitals(
    band_folder,
    dos_folder,
    atom_orbital_pairs,
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
    hse=False,
    kpath=None,
    n=None,
    fontsize=7,
    save=True,
    fill=True,
    alpha=0.3,
    sigma=0.05,
):
    fig, ax = plt.subplots(
        nrows=1,
        ncols=2,
        sharey=True,
        figsize=figsize,
        dpi=300,
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
        hse=hse,
        kpath=kpath,
        n=n,
    )

    dos = Dos(folder=dos_folder, spin=spin)

    band.plot_atom_orbitals(
        ax=ax1,
        scale_factor=scale_factor,
        atom_orbital_pairs=atom_orbital_pairs,
        color_list=color_list,
        legend=False,
        linewidth=linewidth,
        band_color=band_color,
    )

    dos.plot_atom_orbitals(
        ax=ax2,
        atom_orbital_pairs=atom_orbital_pairs,
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
    labels = ax2.get_xticklabels()
    labels[0] = ''
    ax2.set_xticklabels(labels)

    plt.tight_layout(pad=0.2)
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
    color_dict=None,
    legend=True,
    linewidth=0.75,
    band_color='black',
    figsize=(6, 3),
    width_ratios=[7, 3],
    erange=[-6, 6],
    hse=False,
    kpath=None,
    n=None,
    fontsize=7,
    save=True,
    fill=True,
    alpha=0.3,
    sigma=0.05,
):
    fig, ax = plt.subplots(
        nrows=1,
        ncols=2,
        sharey=True,
        figsize=figsize,
        dpi=300,
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
        hse=hse,
        kpath=kpath,
        n=n,
    )

    dos = Dos(folder=dos_folder, spin=spin)

    band.plot_orbitals(
        ax=ax1,
        orbitals=orbitals,
        scale_factor=scale_factor,
        color_dict=color_dict,
        legend=legend,
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
        color_dict=color_dict,
        legend=legend,
        total=True,
        erange=erange,
    )

    fig.canvas.draw()
    labels = ax2.get_xticklabels()
    labels[0] = ''
    ax2.set_xticklabels(labels)

    plt.tight_layout(pad=0.2)
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
    hse=False,
    kpath=None,
    n=None,
    fontsize=7,
    save=True,
    fill=True,
    alpha=0.3,
    sigma=0.05,
):
    fig, ax = plt.subplots(
        nrows=1,
        ncols=2,
        sharey=True,
        figsize=figsize,
        dpi=300,
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
        hse=hse,
        kpath=kpath,
        n=n,
    )

    dos = Dos(folder=dos_folder, spin=spin)

    band.plot_atoms(
        ax=ax1,
        atoms=atoms,
        scale_factor=scale_factor,
        color_list=color_list,
        legend=legend,
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
    labels = ax2.get_xticklabels()
    labels[0] = ''
    ax2.set_xticklabels(labels)

    plt.tight_layout(pad=0.2)
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
    hse=False,
    kpath=None,
    n=None,
    fontsize=7,
    save=True,
    fill=True,
    alpha=0.3,
    sigma=0.05,
):
    fig, ax = plt.subplots(
        nrows=1,
        ncols=2,
        sharey=True,
        figsize=figsize,
        dpi=300,
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
        hse=hse,
        kpath=kpath,
        n=n,
    )

    dos = Dos(folder=dos_folder, spin=spin)

    band.plot_elements(
        ax=ax1,
        elements=elements,
        scale_factor=scale_factor,
        color_list=color_list,
        legend=legend,
        linewidth=linewidth,
        band_color=band_color,
    )

    dos.plot_atoms(
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
    labels = ax2.get_xticklabels()
    labels[0] = ''
    ax2.set_xticklabels(labels)

    plt.tight_layout(pad=0.2)
    plt.subplots_adjust(wspace=0)

    if save:
        plt.savefig(output)
    else:
        return fig, ax1, ax2


def band_dos_element_spd(
    band_folder,
    dos_folder,
    elements,
    output='band_dos_element_spd.png',
    spin='up',
    scale_factor=5,
    order=['s', 'p', 'd'],
    color_dict=None,
    legend=True,
    linewidth=0.75,
    band_color='black',
    figsize=(6, 3),
    width_ratios=[7, 3],
    erange=[-6, 6],
    hse=False,
    kpath=None,
    n=None,
    fontsize=7,
    save=True,
    fill=True,
    alpha=0.3,
    sigma=0.05,
):
    fig, ax = plt.subplots(
        nrows=1,
        ncols=2,
        sharey=True,
        figsize=figsize,
        dpi=300,
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
        hse=hse,
        kpath=kpath,
        n=n,
    )

    dos = Dos(folder=dos_folder, spin=spin)

    band.plot_element_spd(
        ax=ax1,
        elements=elements,
        scale_factor=scale_factor,
        order=order,
        color_dict=color_dict,
        legend=False,
        linewidth=linewidth,
        band_color=band_color,
    )

    dos.plot_element_spd(
        ax=ax2,
        elements=elements,
        order=order,
        fill=fill,
        alpha=alpha,
        linewidth=linewidth,
        sigma=sigma,
        energyaxis='y',
        color_dict=color_dict,
        legend=legend,
        total=True,
        erange=erange,
    )

    fig.canvas.draw()
    labels = ax2.get_xticklabels()
    labels[0] = ''
    ax2.set_xticklabels(labels)

    plt.tight_layout(pad=0.2)
    plt.subplots_adjust(wspace=0)

    if save:
        plt.savefig(output)
    else:
        return fig, ax1, ax2


def band_dos_element_orbitals(
    band_folder,
    dos_folder,
    element_orbital_pairs,
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
    hse=False,
    kpath=None,
    n=None,
    fontsize=7,
    save=True,
    fill=True,
    alpha=0.3,
    sigma=0.05,
):
    fig, ax = plt.subplots(
        nrows=1,
        ncols=2,
        sharey=True,
        figsize=figsize,
        dpi=300,
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
        hse=hse,
        kpath=kpath,
        n=n,
    )

    dos = Dos(folder=dos_folder, spin=spin)

    band.plot_element_orbitals(
        ax=ax1,
        scale_factor=scale_factor,
        element_orbital_pairs=element_orbital_pairs,
        color_list=color_list,
        legend=False,
        linewidth=linewidth,
        band_color=band_color,
    )

    dos.plot_element_orbitals(
        ax=ax2,
        element_orbital_pairs=element_orbital_pairs,
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
    labels = ax2.get_xticklabels()
    labels[0] = ''
    ax2.set_xticklabels(labels)

    plt.tight_layout(pad=0.2)
    plt.subplots_adjust(wspace=0)

    if save:
        plt.savefig(output)
    else:
        return fig, ax1, ax2


def band_dos_plain_spin_projected():
    pass


def band_dos_spd_spin_projected():
    pass


def band_dos_atom_orbitals_spin_projected():
    pass


def band_dos_orbitals_spin_projected():
    pass


def band_dos_atoms_spin_projected():
    pass


def band_dos_elements_spin_projected():
    pass


def band_dos_element_spd_spin_projected():
    pass


def band_dos_element_orbitals_spin_projected():
    pass


def main():
    band_folder = '../../vaspvis_data/band'
    dos_folder = '../../vaspvis_data/dos'
    band_dos_orbitals(
        band_folder=band_folder,
        dos_folder=dos_folder,
        orbitals=[0, 1, 2, 3, 4, 5, 6, 7, 8]
    )


if __name__ == "__main__":
    main()

