from band import Band
from dos import Dos
import matplotlib.pyplot as plt


def _figure_setup(ax, fontsize=6, ylim=[-6, 6]):
    ax.set_ylabel('$E - E_{F}$ $(eV)$', fontsize=fontsize)
    ax.set_ylim(ylim[0], ylim[1])
    ax.tick_params(labelsize=fontsize, length=2.5)
    ax.tick_params(axis='x', length=0)


def band_plain(
        folder,
        output='plain_band.png',
        spin='up',
        color='black',
        linewidth=1.25,
        linestyle='-',
        figsize=(4, 3),
        erange=[-6, 6],
):

    band = Band(folder=folder, spin=spin)
    fig = plt.figure(figsize=(figsize), dpi=300)
    ax = fig.add_subplot(111)
    _figure_setup(ax=ax, fontsize=6, ylim=[erange[0], erange[1]])
    band.plot_plain(
        ax=ax,
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
    )
    plt.tight_layout(pad=0.2)
    plt.savefig(output)


def band_spd(
        folder,
        output='spd_band.png',
        spin='up',
        scale_factor=5,
        order=['s', 'p', 'd'],
        color_dict=None,
        legend=True,
        linewidth=0.75,
        band_color='black',
        figsize=(4, 3),
        erange=[-6, 6],
        right_margin=0.89,
):

    band = Band(
        folder=folder,
        spin=spin,
        projected=True,
    )
    fig = plt.figure(figsize=(figsize), dpi=300)
    ax = fig.add_subplot(111)
    _figure_setup(ax=ax, fontsize=6, ylim=[erange[0], erange[1]])
    band.plot_spd(
        ax,
        scale_factor=scale_factor,
        order=order,
        color_dict=color_dict,
        legend=legend,
        linewidth=linewidth,
        band_color=band_color,
    )
    plt.tight_layout(pad=0.2)
    plt.subplots_adjust(right=right_margin)
    plt.savefig(output)


def band_atom_orbital(
        folder,
        atom_orbital_pairs,
        output='atom_orbital_band.png',
        spin='up',
        scale_factor=5,
        color_dict=None,
        legend=True,
        linewidth=0.75,
        band_color='black',
        figsize=(4, 3),
        erange=[-6, 6],
        right_margin=0.83,
):

    band = Band(
        folder=folder,
        spin=spin,
        projected=True,
    )
    fig = plt.figure(figsize=(figsize), dpi=300)
    ax = fig.add_subplot(111)
    _figure_setup(ax=ax, fontsize=6, ylim=[erange[0], erange[1]])
    band.plot_atom_orbitals(
        ax=ax,
        atom_orbital_pairs=atom_orbital_pairs,
        scale_factor=scale_factor,
        color_dict=color_dict,
        legend=legend,
        linewidth=linewidth,
        band_color=band_color,
    )
    plt.tight_layout(pad=0.2)
    plt.subplots_adjust(right=right_margin)
    plt.savefig(output)


def band_orbitals(
        folder,
        orbitals,
        output='orbital_band.png',
        spin='up',
        scale_factor=5,
        color_dict=None,
        legend=True,
        linewidth=0.75,
        band_color='black',
        figsize=(4, 3),
        erange=[-6, 6],
        right_margin=0.85,
):

    band = Band(
        folder=folder,
        spin=spin,
        projected=True,
    )
    fig = plt.figure(figsize=(figsize), dpi=300)
    ax = fig.add_subplot(111)
    _figure_setup(ax=ax, fontsize=6, ylim=[erange[0], erange[1]])
    band.plot_orbitals(
        ax,
        orbitals=orbitals,
        scale_factor=scale_factor,
        color_dict=color_dict,
        legend=legend,
        linewidth=linewidth,
        band_color=band_color,
    )
    plt.tight_layout(pad=0.2)
    plt.subplots_adjust(right=right_margin)
    plt.savefig(output)


def band_atoms(
        folder,
        atoms,
        output='orbital_band.png',
        spin='up',
        scale_factor=5,
        color_dict=None,
        legend=True,
        linewidth=0.75,
        band_color='black',
        figsize=(4, 3),
        erange=[-6, 6],
        right_margin=0.85,
):

    band = Band(
        folder=folder,
        spin=spin,
        projected=True,
    )
    fig = plt.figure(figsize=(figsize), dpi=300)
    ax = fig.add_subplot(111)
    _figure_setup(ax=ax, fontsize=6, ylim=[erange[0], erange[1]])
    band.plot_orbitals(
        ax,
        atoms=atoms,
        scale_factor=scale_factor,
        color_dict=color_dict,
        legend=legend,
        linewidth=linewidth,
        band_color=band_color,
    )
    plt.tight_layout(pad=0.2)
    plt.subplots_adjust(right=right_margin)
    plt.savefig(output)


def band_elements(
        folder,
        elements,
        output='orbital_band.png',
        spin='up',
        scale_factor=5,
        color_dict=None,
        legend=True,
        linewidth=0.75,
        band_color='black',
        figsize=(4, 3),
        erange=[-6, 6],
        right_margin=0.85,
):

    band = Band(
        folder=folder,
        spin=spin,
        projected=True,
    )
    fig = plt.figure(figsize=(figsize), dpi=300)
    ax = fig.add_subplot(111)
    _figure_setup(ax=ax, fontsize=6, ylim=[erange[0], erange[1]])
    band.plot_elements(
        ax,
        elements=elements,
        scale_factor=scale_factor,
        color_dict=color_dict,
        legend=legend,
        linewidth=linewidth,
        band_color=band_color,
    )
    plt.tight_layout(pad=0.2)
    plt.subplots_adjust(right=right_margin)
    plt.savefig(output)


def band_element_orbitals(
        folder,
        element_orbital_pairs,
        output='atom_orbital_band.png',
        spin='up',
        scale_factor=5,
        color_dict=None,
        legend=True,
        linewidth=0.75,
        band_color='black',
        figsize=(4, 3),
        erange=[-6, 6],
        right_margin=0.83,
):

    band = Band(
        folder=folder,
        spin=spin,
        projected=True,
    )
    fig = plt.figure(figsize=(figsize), dpi=300)
    ax = fig.add_subplot(111)
    _figure_setup(ax=ax, fontsize=6, ylim=[erange[0], erange[1]])
    band.plot_element_orbitals(
        ax,
        element_orbital_pairs=element_orbital_pairs,
        scale_factor=scale_factor,
        color_dict=color_dict,
        legend=legend,
        linewidth=linewidth,
        band_color=band_color,
    )
    plt.tight_layout(pad=0.2)
    plt.subplots_adjust(right=right_margin)
    plt.savefig(output)


def band_element_spd(
        folder,
        order=['s', 'p', 'd'],
        output='atom_orbital_band.png',
        spin='up',
        scale_factor=5,
        color_dict=None,
        legend=True,
        linewidth=0.75,
        band_color='black',
        figsize=(4, 3),
        erange=[-6, 6],
        right_margin=0.83,
):

    band = Band(
        folder=folder,
        spin=spin,
        projected=True,
    )
    fig = plt.figure(figsize=(figsize), dpi=300)
    ax = fig.add_subplot(111)
    _figure_setup(ax=ax, fontsize=6, ylim=[erange[0], erange[1]])
    band.plot_element_spd(
        ax,
        order=order,
        scale_factor=scale_factor,
        color_dict=color_dict,
        legend=legend,
        linewidth=linewidth,
        band_color=band_color,
    )
    plt.tight_layout(pad=0.2)
    plt.subplots_adjust(right=right_margin)
    plt.savefig(output)


def band_plain_spin_projected(
        folder,
        output='plain_sp_band.png',
        up_color='black',
        down_color='red',
        linewidth=1.25,
        up_linestyle='-',
        down_linestyle='-',
        figsize=(4, 3),
        erange=[-6, 6],
        right_margin=0.89,
):

    band_up = Band(folder=folder, spin='up')
    band_down = Band(folder=folder, spin='down')
    fig = plt.figure(figsize=(figsize), dpi=300)
    ax = fig.add_subplot(111)
    _figure_setup(ax=ax, fontsize=6, ylim=[erange[0], erange[1]])
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
        fontsize=5,
        bbox_to_anchor=(1, 1),
        borderaxespad=0,
        frameon=False,
        handletextpad=0.1,
    )

    plt.tight_layout(pad=0.2)
    plt.subplots_adjust(right=right_margin)
    plt.savefig(output)


def band_spd_spin_projected(
        folder,
        output='spd_sp_band.png',
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
        right_margin=0.89,
        stack='vertical',
):

    band_up = Band(
        folder=folder,
        spin='up',
        projected=True,
    )

    band_down = Band(
        folder=folder,
        spin='down',
        projected=True,
    )

    if stack == 'vertical':
        fig = plt.figure(figsize=(figsize[0], 2 * figsize[1]), dpi=300)
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        _figure_setup(ax=ax1, fontsize=6, ylim=[erange[0], erange[1]])
        _figure_setup(ax=ax2, fontsize=6, ylim=[erange[0], erange[1]])
    elif stack == 'horizontal':
        fig = plt.figure(figsize=(2 * figsize[0], figsize[1]), dpi=300)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        _figure_setup(ax=ax1, fontsize=6, ylim=[erange[0], erange[1]])
        _figure_setup(ax=ax2, fontsize=6, ylim=[erange[0], erange[1]])

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
        ax1,
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
        ax2,
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

    plt.tight_layout()
    plt.subplots_adjust(right=right_margin)
    plt.savefig(output)


def band_atom_orbital_spin_projected(
        folder,
        atom_orbital_pairs,
        output='atom_orbital_sp_band.png',
        scale_factor=5,
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
        right_margin=0.86,
        stack='vertical',
):

    band_up = Band(
        folder=folder,
        spin='up',
        projected=True,
    )

    band_down = Band(
        folder=folder,
        spin='down',
        projected=True,
    )

    if stack == 'vertical':
        fig = plt.figure(figsize=(figsize[0], 2 * figsize[1]), dpi=300)
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        _figure_setup(ax=ax1, fontsize=6, ylim=[erange[0], erange[1]])
        _figure_setup(ax=ax2, fontsize=6, ylim=[erange[0], erange[1]])
    elif stack == 'horizontal':
        fig = plt.figure(figsize=(2 * figsize[0], figsize[1]), dpi=300)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        _figure_setup(ax=ax1, fontsize=6, ylim=[erange[0], erange[1]])
        _figure_setup(ax=ax2, fontsize=6, ylim=[erange[0], erange[1]])

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

    band_down.plot_atom_orbitals(
        ax=ax2,
        scale_factor=scale_factor,
        atom_orbital_pairs=atom_orbital_pairs,
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

    plt.tight_layout()
    plt.subplots_adjust(right=right_margin)
    plt.savefig(output)


def band_orbitals_spin_projected(
        folder,
        orbitals,
        output='spd_sp_band.png',
        scale_factor=5,
        color_dict=None,
        legend=True,
        linewidth=0.75,
        band_color='black',
        unprojected_band_color='gray',
        figsize=(4, 3),
        erange=[-6, 6],
        right_margin=0.85,
        stack='vertical',
):

    band_up = Band(
        folder=folder,
        spin='up',
        projected=True,
    )

    band_down = Band(
        folder=folder,
        spin='down',
        projected=True,
    )

    if stack == 'vertical':
        fig = plt.figure(figsize=(figsize[0], 2 * figsize[1]), dpi=300)
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        _figure_setup(ax=ax1, fontsize=6, ylim=[erange[0], erange[1]])
        _figure_setup(ax=ax2, fontsize=6, ylim=[erange[0], erange[1]])
    elif stack == 'horizontal':
        fig = plt.figure(figsize=(2 * figsize[0], figsize[1]), dpi=300)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        _figure_setup(ax=ax1, fontsize=6, ylim=[erange[0], erange[1]])
        _figure_setup(ax=ax2, fontsize=6, ylim=[erange[0], erange[1]])

    bbox = dict(boxstyle="round", fc="white")
    ax1.annotate(
        '$\\uparrow$',
        xy=(0.02, 0.98),
        xycoords='axes fraction',
        ha='top',
        va='left',
        bbox=bbox,
    )
    ax2.annotate(
        '$\\downarrow$',
        xy=(0.02, 0.98),
        xycoords='axes fraction',
        ha='top',
        va='left',
        bbox=bbox,
    )

    band_up.plot_spd(
        ax1,
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

    band_down.plot_spd(
        ax1,
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

    plt.tight_layout()
    plt.subplots_adjust(right=right_margin)
    plt.savefig(output)


# =============================================================
# -------------------- Density of States ----------------------
# =============================================================

def dos_plain():
    pass

def dos_spd():
    pass

def dos_atom_orbitals():
    pass

def dos_orbitals():
    pass

def dos_atoms():
    pass

def dos_elements():
    pass

def dos_element_spd():
    pass

def dos_element_orbitals():
    pass

def dos_plain_spin_projected():
    pass

def dos_spd_spin_projected():
    pass

def dos_atom_orbitals_spin_projected():
    pass

def dos_orbitals_spin_projected():
    pass

def dos_atoms_spin_projected():
    pass

def dos_elements_spin_projected():
    pass

def dos_element_spd_spin_projected():
    pass

def dos_element_orbitals_spin_projected():
    pass


# =============================================================
# ---------------------- Band-Dos Plots -----------------------
# =============================================================

def band_dos_plain():
    pass

def band_dos_spd():
    pass

def band_dos_atom_orbitals():
    pass

def band_dos_orbitals():
    pass

def band_dos_atoms():
    pass

def band_dos_elements():
    pass

def band_dos_element_spd():
    pass

def band_dos_element_orbitals():
    pass

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
    band_atom_orbital_spin_projected(
        folder='../../../../../../../for_James/band',
        atom_orbital_pairs=[[0,3], [1,7]],
    )
    band_atom_orbital(
        folder='../../vaspvis_data/band',
        atom_orbital_pairs=[[0, 3], [1, 4]]
    )


if __name__ == "__main__":
    main()


# # Load Data
# band = Band(
    # folder=band_folder,
    # projected=True,
    # spin='up',
# )


# # ==========================================================
# # ----------------- Plain Band Structure -------------------
# # ==========================================================
# fig1 = plt.figure(figsize=(4, 3), dpi=300)
# ax1 = fig1.add_subplot(111)
# plt.ylabel('$E - E_{F}$ $(eV)$', fontsize=6)
# plt.ylim(-6, 6)
# plt.tight_layout(pad=0.5)
# plt.tick_params(labelsize=6, length=1.5)
# plt.tick_params(axis='x', length=0)

# band.plot_plain(ax=ax1, linewidth=1)

# plt.savefig('./img/plain_band.png')
# plt.close()
# # ==========================================================
# # ==========================================================


# # ==========================================================
# # ------------------- SPD Band Structure -------------------
# # ==========================================================
# fig2 = plt.figure(figsize=(4, 3), dpi=300)
# ax2 = fig2.add_subplot(111)
# plt.ylabel('$E - E_{F}$ $(eV)$', fontsize=6)
# plt.ylim(-6, 6)
# plt.tight_layout(pad=0.2)
# plt.subplots_adjust(right=0.89)
# plt.tick_params(labelsize=6, length=2.5)
# plt.tick_params(axis='x', length=0)

# band.plot_spd(ax=ax2)

# plt.savefig('./img/spd_band.png')
# plt.close()
# # ==========================================================
# # ==========================================================


# # ==========================================================
# # ---------------- Orbitals Band Structure -----------------
# # ==========================================================
# fig3 = plt.figure(figsize=(4, 3), dpi=300)
# ax3 = fig3.add_subplot(111)
# plt.ylabel('$E - E_{F}$ $(eV)$', fontsize=6)
# plt.ylim(-6, 6)
# plt.tight_layout(pad=0.2)
# plt.subplots_adjust(right=0.89)
# plt.tick_params(labelsize=6, length=2.5)
# plt.tick_params(axis='x', length=0)

# band.plot_orbitals(ax=ax3, orbitals=[
    # 0, 1, 2, 3, 4, 5, 6, 7, 8], scale_factor=8)

# plt.savefig('./img/orbital_band.png')
# plt.close()
# # ==========================================================
# # ==========================================================


# # ==========================================================
# # ------------------- Atom Band Structure ------------------
# # ==========================================================
# fig4 = plt.figure(figsize=(4, 3), dpi=300)
# ax4 = fig4.add_subplot(111)
# plt.ylabel('$E - E_{F}$ $(eV)$', fontsize=6)
# plt.ylim(-6, 6)
# plt.tight_layout(pad=0.2)
# plt.subplots_adjust(right=0.89)
# plt.tick_params(labelsize=6, length=2.5)
# plt.tick_params(axis='x', length=0)

# band.plot_atoms(ax=ax4, atoms=[0, 1], scale_factor=8)

# plt.savefig('./img/atom_band.png')
# plt.close()
# # ==========================================================
# # ==========================================================


# # ==========================================================
# # ------------- Atom Orbital Band Structure ----------------
# # ==========================================================
# fig5 = plt.figure(figsize=(4, 3), dpi=300)
# ax5 = fig5.add_subplot(111)
# plt.ylabel('$E - E_{F}$ $(eV)$', fontsize=6)
# plt.ylim(-6, 6)
# plt.tight_layout(pad=0.2)
# plt.subplots_adjust(right=0.88)
# plt.tick_params(labelsize=6, length=2.5)
# plt.tick_params(axis='x', length=0)

# band.plot_atom_orbitals(ax=ax5, atom_orbital_pairs=[
    # [0, 0], [0, 1], [0, 3]], scale_factor=8)

# plt.savefig('./img/atom_orbital_band.png')
# plt.close()
# # ==========================================================
# # ==========================================================

# # Load Slab Data
# band_slab = Band(
    # folder=band_slab_folder,
    # projected=True,
    # spin='up',
# )


# # ==========================================================
# # ------------- Element Band Structure ----------------
# # ==========================================================
# fig6 = plt.figure(figsize=(3, 3), dpi=300)
# ax6 = fig6.add_subplot(111)
# plt.ylabel('$E - E_{F}$ $(eV)$', fontsize=6)
# plt.ylim(-6, 6)
# plt.tight_layout(pad=0.2)
# plt.subplots_adjust(right=0.86)
# plt.tick_params(labelsize=6, length=2.5)
# plt.tick_params(axis='x', length=0)

# band_slab.plot_elements(
    # ax=ax6, elements=['In', 'As', 'Eu', 'S'], scale_factor=4)

# plt.savefig('./img/element_band.png')
# plt.close()
# # ==========================================================
# # ==========================================================


# # ==========================================================
# # ------------- Element SPD Band Structure ----------------
# # ==========================================================
# fig7 = plt.figure(figsize=(3, 3), dpi=300)
# ax7 = fig7.add_subplot(111)
# plt.ylabel('$E - E_{F}$ $(eV)$', fontsize=6)
# plt.ylim(-6, 6)
# plt.tight_layout(pad=0.2)
# plt.subplots_adjust(right=0.84)
# plt.tick_params(labelsize=6, length=2.5)
# plt.tick_params(axis='x', length=0)

# band_slab.plot_element_spd(
    # ax=ax7, elements=['As'], scale_factor=4, order=['p', 's', 'd'])

# plt.savefig('./img/element_spd_band.png')
# plt.close()
# # ==========================================================
# # ==========================================================


# # ==========================================================
# # ------------- Element SPD Band Structure ----------------
# # ==========================================================
# fig8 = plt.figure(figsize=(3, 3), dpi=300)
# ax8 = fig8.add_subplot(111)
# plt.ylabel('$E - E_{F}$ $(eV)$', fontsize=6)
# plt.ylim(-6, 6)
# plt.tight_layout(pad=0.2)
# plt.subplots_adjust(right=0.84)
# plt.tick_params(labelsize=6, length=2.5)
# plt.tick_params(axis='x', length=0)

# band_slab.plot_element_orbitals(ax=ax8, element_orbital_pairs=[['In', 1], [
    # 'As', 3], ['S', 0], ['Eu', 6]], scale_factor=4)

# plt.savefig('./img/element_orbital_band.png')
# plt.close()
# # ==========================================================
# # ==========================================================


# """
# This section will plot the density of states
# """

# dos = Dos(
    # folder=dos_folder,
# )

# # ==========================================================
# # ----------------------- Plain DOS ------------------------
# # ==========================================================
# fig9 = plt.figure(figsize=(7, 3), dpi=300)
# ax9_1 = fig9.add_subplot(121)
# plt.ylabel('$E - E_{F}$ $(eV)$', fontsize=6)
# plt.xlabel('Density', fontsize=6)
# plt.title("energyaxis='y'", fontsize=8)
# plt.ylim(-6, 6)
# plt.xlim(0, 3)
# plt.tight_layout(pad=0.2)
# plt.tick_params(labelsize=6, length=2.5)
# plt.tick_params(axis='x', length=0)

# ax9_2 = fig9.add_subplot(122)
# plt.xlabel('$E - E_{F}$ $(eV)$', fontsize=6)
# plt.ylabel('Density', fontsize=6)
# plt.title("energyaxis='x'", fontsize=8)
# plt.xlim(-6, 6)
# plt.ylim(0, 3)
# plt.tight_layout(pad=0.2)
# plt.tick_params(labelsize=6, length=2.5)
# plt.tick_params(axis='x', length=0)

# plt.subplots_adjust(right=0.98, wspace=0.2)

# dos.plot_plain(ax=ax9_1)
# dos.plot_plain(ax=ax9_2, energyaxis='x')

# plt.savefig('./img/dos/plain_dos.png')
# plt.close()
# # ==========================================================
# # ==========================================================


# # ==========================================================
# # ----------------------- SPD DOS ------------------------
# # ==========================================================
# fig10 = plt.figure(figsize=(7, 3), dpi=300)
# ax10_1 = fig10.add_subplot(121)
# plt.ylabel('$E - E_{F}$ $(eV)$', fontsize=6)
# plt.xlabel('Density', fontsize=6)
# plt.title("energyaxis='y'", fontsize=8)
# plt.ylim(-6, 6)
# plt.xlim(0, 3)
# plt.tight_layout(pad=0.2)
# plt.tick_params(labelsize=6, length=2.5)
# plt.tick_params(axis='x', length=0)

# ax10_2 = fig10.add_subplot(122)
# plt.xlabel('$E - E_{F}$ $(eV)$', fontsize=6)
# plt.ylabel('Density', fontsize=6)
# plt.title("energyaxis='x'", fontsize=8)
# plt.xlim(-6, 6)
# plt.ylim(0, 3)
# plt.tight_layout(pad=0.2)
# plt.tick_params(labelsize=6, length=2.5)
# plt.tick_params(axis='x', length=0)

# plt.subplots_adjust(right=0.95, wspace=0.25)

# dos.plot_spd(ax=ax10_1)
# dos.plot_spd(ax=ax10_2, energyaxis='x')

# plt.savefig('./img/dos/spd_dos.png')
# plt.close()
# # ==========================================================
# # ==========================================================


# # ==========================================================
# # --------------------- Orbital DOS ------------------------
# # ==========================================================
# fig11 = plt.figure(figsize=(7, 3), dpi=300)
# ax11_1 = fig11.add_subplot(121)
# plt.ylabel('$E - E_{F}$ $(eV)$', fontsize=6)
# plt.xlabel('Density', fontsize=6)
# plt.title("energyaxis='y'", fontsize=8)
# plt.ylim(-6, 6)
# plt.xlim(0, 3)
# plt.tight_layout(pad=0.2)
# plt.tick_params(labelsize=6, length=2.5)
# plt.tick_params(axis='x', length=0)

# ax11_2 = fig11.add_subplot(122)
# plt.xlabel('$E - E_{F}$ $(eV)$', fontsize=6)
# plt.ylabel('Density', fontsize=6)
# plt.title("energyaxis='x'", fontsize=8)
# plt.xlim(-6, 6)
# plt.ylim(0, 3)
# plt.tight_layout(pad=0.2)
# plt.tick_params(labelsize=6, length=2.5)
# plt.tick_params(axis='x', length=0)

# plt.subplots_adjust(right=0.93, wspace=0.35)

# dos.plot_orbitals(ax=ax11_1, orbitals=[0, 1, 2, 3, 4, 5, 6, 7, 8])
# dos.plot_orbitals(ax=ax11_2, orbitals=[
    # 0, 1, 2, 3, 4, 5, 6, 7, 8], energyaxis='x')

# plt.savefig('./img/dos/orbital_dos.png')
# plt.close()
# # ==========================================================
# # ==========================================================


# # ==========================================================
# # ------------------ Atom Orbital DOS ----------------------
# # ==========================================================
# fig12 = plt.figure(figsize=(7, 3), dpi=300)
# ax12_1 = fig12.add_subplot(121)
# plt.ylabel('$E - E_{F}$ $(eV)$', fontsize=6)
# plt.xlabel('Density', fontsize=6)
# plt.title("energyaxis='y'", fontsize=8)
# plt.ylim(-6, 6)
# plt.xlim(0, 3)
# plt.tight_layout(pad=0.2)
# plt.tick_params(labelsize=6, length=2.5)
# plt.tick_params(axis='x', length=0)

# ax12_2 = fig12.add_subplot(122)
# plt.xlabel('$E - E_{F}$ $(eV)$', fontsize=6)
# plt.ylabel('Density', fontsize=6)
# plt.title("energyaxis='x'", fontsize=8)
# plt.xlim(-6, 6)
# plt.ylim(0, 3)
# plt.tight_layout(pad=0.2)
# plt.tick_params(labelsize=6, length=2.5)
# plt.tick_params(axis='x', length=0)

# plt.subplots_adjust(right=0.93, wspace=0.35)

# dos.plot_atom_orbitals(ax=ax12_1, atom_orbital_pairs=[[0, 0], [0, 1], [0, 3]])
# dos.plot_atom_orbitals(ax=ax12_2, atom_orbital_pairs=[
    # [0, 0], [0, 1], [0, 3]], energyaxis='x')

# plt.savefig('./img/dos/atom_orbital_dos.png')
# plt.close()
# # ==========================================================
# # ==========================================================


# # ==========================================================
# # ---------------------- Atom DOS --------------------------
# # ==========================================================
# fig13 = plt.figure(figsize=(7, 3), dpi=300)
# ax13_1 = fig13.add_subplot(121)
# plt.ylabel('$E - E_{F}$ $(eV)$', fontsize=6)
# plt.xlabel('Density', fontsize=6)
# plt.title("energyaxis='y'", fontsize=8)
# plt.ylim(-6, 6)
# plt.xlim(0, 3)
# plt.tight_layout(pad=0.2)
# plt.tick_params(labelsize=6, length=2.5)
# plt.tick_params(axis='x', length=0)

# ax13_2 = fig13.add_subplot(122)
# plt.xlabel('$E - E_{F}$ $(eV)$', fontsize=6)
# plt.ylabel('Density', fontsize=6)
# plt.title("energyaxis='x'", fontsize=8)
# plt.xlim(-6, 6)
# plt.ylim(0, 3)
# plt.tight_layout(pad=0.2)
# plt.tick_params(labelsize=6, length=2.5)
# plt.tick_params(axis='x', length=0)

# plt.subplots_adjust(right=0.93, wspace=0.35)

# dos.plot_atoms(ax=ax13_1, atoms=[0, 1])
# dos.plot_atoms(ax=ax13_2, atoms=[0, 1], energyaxis='x')

# plt.savefig('./img/dos/atom_dos.png')
# plt.close()
# # ==========================================================
# # ==========================================================


# dos_slab = Dos(
    # folder=slab_dos_folder,
# )

# # ==========================================================
# # ------------------- Element DOS --------------------------
# # ==========================================================
# fig14 = plt.figure(figsize=(7, 3), dpi=300)
# ax14_1 = fig14.add_subplot(121)
# plt.ylabel('$E - E_{F}$ $(eV)$', fontsize=6)
# plt.xlabel('Density', fontsize=6)
# plt.title("energyaxis='y'", fontsize=8)
# plt.ylim(-6, 6)
# plt.xlim(0, 260)
# plt.tight_layout(pad=0.2)
# plt.tick_params(labelsize=6, length=2.5)
# plt.tick_params(axis='x', length=0)

# ax14_2 = fig14.add_subplot(122)
# plt.xlabel('$E - E_{F}$ $(eV)$', fontsize=6)
# plt.ylabel('Density', fontsize=6)
# plt.title("energyaxis='x'", fontsize=8)
# plt.xlim(-6, 6)
# plt.ylim(0, 260)
# plt.tight_layout(pad=0.2)
# plt.tick_params(labelsize=6, length=2.5)
# plt.tick_params(axis='x', length=0)

# plt.subplots_adjust(right=0.93, wspace=0.35)

# dos_slab.plot_elements(ax=ax14_1, elements=['In', 'As', 'Eu', 'S'])
# dos_slab.plot_elements(ax=ax14_2, elements=[
    # 'In', 'As', 'Eu', 'S'], energyaxis='x')

# plt.savefig('./img/dos/elements_dos.png')
# plt.close()
# # ==========================================================
# # ==========================================================


# # ==========================================================
# # ---------------- Element Orbital DOS ---------------------
# # ==========================================================
# fig15 = plt.figure(figsize=(7, 3), dpi=300)
# ax15_1 = fig15.add_subplot(121)
# plt.ylabel('$E - E_{F}$ $(eV)$', fontsize=6)
# plt.xlabel('Density', fontsize=6)
# plt.title("energyaxis='y'", fontsize=8)
# plt.ylim(-6, 6)
# plt.xlim(0, 260)
# plt.tight_layout(pad=0.2)
# plt.tick_params(labelsize=6, length=2.5)
# plt.tick_params(axis='x', length=0)

# ax15_2 = fig15.add_subplot(122)
# plt.xlabel('$E - E_{F}$ $(eV)$', fontsize=6)
# plt.ylabel('Density', fontsize=6)
# plt.title("energyaxis='x'", fontsize=8)
# plt.xlim(-6, 6)
# plt.ylim(0, 260)
# plt.tight_layout(pad=0.2)
# plt.tick_params(labelsize=6, length=2.5)
# plt.tick_params(axis='x', length=0)

# plt.subplots_adjust(right=0.93, wspace=0.35)

# dos_slab.plot_element_orbitals(ax=ax15_1, element_orbital_pairs=[['In', 0], ['As', 1], [
    # 'Eu', 0], ['Eu', 5], ['Eu', 6], ['Eu', 7], ['Eu', 8], ['S', 3], ['Eu', 10]])
# dos_slab.plot_element_orbitals(ax=ax15_2, element_orbital_pairs=[['In', 0], ['As', 1], [
    # 'Eu', 0], ['Eu', 5], ['Eu', 6], ['Eu', 7], ['Eu', 8], ['S', 3], ['Eu', 10]], energyaxis='x')

# plt.savefig('./img/dos/elements_orbitals_dos.png')
# plt.close()
# # ==========================================================
# # ==========================================================


# # ==========================================================
# # ------------------- Element SPD DOS ----------------------
# # ==========================================================
# fig16 = plt.figure(figsize=(7, 3), dpi=300)
# ax16_1 = fig16.add_subplot(121)
# plt.ylabel('$E - E_{F}$ $(eV)$', fontsize=6)
# plt.xlabel('Density', fontsize=6)
# plt.title("energyaxis='y'", fontsize=8)
# plt.ylim(-6, 6)
# plt.xlim(0, 260)
# plt.tight_layout(pad=0.2)
# plt.tick_params(labelsize=6, length=2.5)
# plt.tick_params(axis='x', length=0)

# ax16_2 = fig16.add_subplot(122)
# plt.xlabel('$E - E_{F}$ $(eV)$', fontsize=6)
# plt.ylabel('Density', fontsize=6)
# plt.title("energyaxis='x'", fontsize=8)
# plt.xlim(-6, 6)
# plt.ylim(0, 260)
# plt.tight_layout(pad=0.2)
# plt.tick_params(labelsize=6, length=2.5)
# plt.tick_params(axis='x', length=0)

# plt.subplots_adjust(right=0.93, wspace=0.35)

# dos_slab.plot_element_spd(ax=ax16_1, elements=['Eu'])
# dos_slab.plot_element_spd(ax=ax16_2, elements=['Eu'], energyaxis='x')

# plt.savefig('./img/dos/elements_spd_dos.png')
# plt.close()
# # ==========================================================
# # ==========================================================


# # ==========================================================
# # ---------------------- Layer DOS -------------------------
# # ==========================================================
# fig17 = plt.figure(figsize=(7, 3), dpi=300)
# ax17_1 = fig17.add_subplot(121)
# plt.ylabel('$E - E_{F}$ $(eV)$', fontsize=6)
# plt.xlabel('Layers', fontsize=6)
# plt.title("energyaxis='y'", fontsize=8)
# plt.ylim(-2, 2.8)
# # plt.xlim(0,3)
# plt.tight_layout(pad=0.2)
# plt.tick_params(labelsize=6, length=2.5)
# plt.tick_params(axis='x', length=0)

# ax17_2 = fig17.add_subplot(122)
# plt.xlabel('$E - E_{F}$ $(eV)$', fontsize=6)
# plt.ylabel('Layers', fontsize=6)
# plt.title("energyaxis='x'", fontsize=8)
# plt.xlim(-2, 2.8)
# # plt.ylim(0,3)
# plt.tight_layout(pad=0.2)
# plt.tick_params(labelsize=6, length=2.5)
# plt.tick_params(axis='x', length=0)

# plt.subplots_adjust(right=0.93, wspace=0.35)

# dos_slab.plot_layers(ax=ax17_1, sigma=1.5, cmap='magma', ylim=[-2, 3])
# dos_slab.plot_layers(ax=ax17_2, energyaxis='x', sigma=1.5,
    # cmap='magma', ylim=[-2, 3])
# ax17_1.axvline(x=42, color='white', linewidth=2, linestyle='--')
# ax17_2.axhline(y=42, color='white', linewidth=2, linestyle='--')
# ax17_1.annotate('InAs', xy=(20, 0.5), xycoords='data',
    # color='white', ha='center')
# ax17_1.annotate('EuS', xy=(60, 0.5), xycoords='data',
    # color='white', ha='center')
# ax17_2.annotate('InAs', xy=(0.5, 20), xycoords='data',
    # color='white', ha='center')
# ax17_2.annotate('EuS', xy=(0.5, 60), xycoords='data',
    # color='white', ha='center')

# plt.savefig('./img/dos/layer_dos.png')
# plt.close()
# # ==========================================================
# # ==========================================================
