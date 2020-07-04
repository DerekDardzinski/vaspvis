from core import band
from core import dos
import matplotlib.pyplot as plt
from collections import OrderedDict

band_folder = '../vaspvis_data/band'
dos_folder = '../vaspvis_data/dos'
hse_folder = '../vaspvis_data/hse'
slab_folder = '../vaspvis_data/bandInterface'


"""
This section will plot band structures calculated without HSE
"""

# Load Data
# pbe = band.BandStructure(
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

# pbe.plot_plain(ax=ax1, linewidth=1)

# plt.savefig('./img/plain_pbe.png')
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

# pbe.plot_spd(ax=ax2)

# plt.savefig('./img/spd_pbe.png')
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

# pbe.plot_orbitals(ax=ax3, orbitals=[0,1,2,3,4,5,6,7,8], scale_factor=8)

# plt.savefig('./img/orbital_pbe.png')
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

# pbe.plot_atoms(ax=ax4, atoms=[0,1], scale_factor=8)

# plt.savefig('./img/atom_pbe.png')
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

# pbe.plot_atom_orbitals(ax=ax5, atom_orbital_pairs=[[0,0], [0,1], [0,3]], scale_factor=8)

# plt.savefig('./img/atom_orbital_pbe.png')
# plt.close()
# # ==========================================================
# # ==========================================================

# Load Slab Data
slab = band.BandStructure(
    folder=slab_folder,
    projected=True,
    spin='up',
)


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

# slab.plot_elements(ax=ax6, elements=['In', 'As', 'Eu', 'S'], scale_factor=4)

# plt.savefig('./img/element_pbe.png')
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

# slab.plot_element_spd(ax=ax7, elements=['As'], scale_factor=4, order=['p', 's', 'd'])

# plt.savefig('./img/element_spd_pbe.png')
# plt.close()
# # ==========================================================
# # ==========================================================



# ==========================================================
# ------------- Element SPD Band Structure ----------------
# ==========================================================
fig8 = plt.figure(figsize=(3, 3), dpi=300)
ax8 = fig8.add_subplot(111)
plt.ylabel('$E - E_{F}$ $(eV)$', fontsize=6)
plt.ylim(-6, 6)
plt.tight_layout(pad=0.2)
plt.subplots_adjust(right=0.84)
plt.tick_params(labelsize=6, length=2.5)
plt.tick_params(axis='x', length=0)

slab.plot_element_orbitals(ax=ax8, element_orbital_pairs=[['In', 1], ['As', 3], ['S', 0], ['Eu', 6]], scale_factor=4)

plt.savefig('./img/element_orbital_pbe.png')
plt.close()
# ==========================================================
# ==========================================================


