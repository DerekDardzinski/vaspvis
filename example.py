from vaspvis import Band, Dos
import matplotlib.pyplot as plt
from collections import OrderedDict

band_folder = '../vaspvis_data/band'
dos_folder = '../vaspvis_data/dos'
hse_folder = '../vaspvis_data/hse'
slab_folder = '../vaspvis_data/bandInterface'
slab_dos_folder = '../vaspvis_data/dosInterface'


"""
This section will plot band structures
"""

#Load Data
band = Band(
folder=band_folder,
projected=True,
spin='up',
)


# ==========================================================
# ----------------- Plain Band Structure -------------------
# ==========================================================
fig1 = plt.figure(figsize=(4, 3), dpi=300)
ax1 = fig1.add_subplot(111)
plt.ylabel('$E - E_{F}$ $(eV)$', fontsize=6)
plt.ylim(-6, 6)
plt.tight_layout(pad=0.5)
plt.tick_params(labelsize=6, length=1.5)
plt.tick_params(axis='x', length=0)

band.plot_plain(ax=ax1, linewidth=1)

plt.savefig('./img/plain_band.png')
plt.close()
# ==========================================================
# ==========================================================


# ==========================================================
# ------------------- SPD Band Structure -------------------
# ==========================================================
fig2 = plt.figure(figsize=(4, 3), dpi=300)
ax2 = fig2.add_subplot(111)
plt.ylabel('$E - E_{F}$ $(eV)$', fontsize=6)
plt.ylim(-6, 6)
plt.tight_layout(pad=0.2)
plt.subplots_adjust(right=0.89)
plt.tick_params(labelsize=6, length=2.5)
plt.tick_params(axis='x', length=0)

band.plot_spd(ax=ax2)

plt.savefig('./img/spd_band.png')
plt.close()
# ==========================================================
# ==========================================================


# ==========================================================
# ---------------- Orbitals Band Structure -----------------
# ==========================================================
fig3 = plt.figure(figsize=(4, 3), dpi=300)
ax3 = fig3.add_subplot(111)
plt.ylabel('$E - E_{F}$ $(eV)$', fontsize=6)
plt.ylim(-6, 6)
plt.tight_layout(pad=0.2)
plt.subplots_adjust(right=0.89)
plt.tick_params(labelsize=6, length=2.5)
plt.tick_params(axis='x', length=0)

band.plot_orbitals(ax=ax3, orbitals=[0,1,2,3,4,5,6,7,8], scale_factor=8)

plt.savefig('./img/orbital_band.png')
plt.close()
# ==========================================================
# ==========================================================


# ==========================================================
# ------------------- Atom Band Structure ------------------
# ==========================================================
fig4 = plt.figure(figsize=(4, 3), dpi=300)
ax4 = fig4.add_subplot(111)
plt.ylabel('$E - E_{F}$ $(eV)$', fontsize=6)
plt.ylim(-6, 6)
plt.tight_layout(pad=0.2)
plt.subplots_adjust(right=0.89)
plt.tick_params(labelsize=6, length=2.5)
plt.tick_params(axis='x', length=0)

band.plot_atoms(ax=ax4, atoms=[0,1], scale_factor=8)

plt.savefig('./img/atom_band.png')
plt.close()
# ==========================================================
# ==========================================================


# ==========================================================
# ------------- Atom Orbital Band Structure ----------------
# ==========================================================
fig5 = plt.figure(figsize=(4, 3), dpi=300)
ax5 = fig5.add_subplot(111)
plt.ylabel('$E - E_{F}$ $(eV)$', fontsize=6)
plt.ylim(-6, 6)
plt.tight_layout(pad=0.2)
plt.subplots_adjust(right=0.88)
plt.tick_params(labelsize=6, length=2.5)
plt.tick_params(axis='x', length=0)

band.plot_atom_orbitals(ax=ax5, atom_orbital_pairs=[[0,0], [0,1], [0,3]], scale_factor=8)

plt.savefig('./img/atom_orbital_band.png')
plt.close()
# ==========================================================
# ==========================================================

# Load Slab Data
band_slab = Band(
folder=band_slab_folder,
projected=True,
spin='up',
)


# ==========================================================
# ------------- Element Band Structure ----------------
# ==========================================================
fig6 = plt.figure(figsize=(3, 3), dpi=300)
ax6 = fig6.add_subplot(111)
plt.ylabel('$E - E_{F}$ $(eV)$', fontsize=6)
plt.ylim(-6, 6)
plt.tight_layout(pad=0.2)
plt.subplots_adjust(right=0.86)
plt.tick_params(labelsize=6, length=2.5)
plt.tick_params(axis='x', length=0)

band_slab.plot_elements(ax=ax6, elements=['In', 'As', 'Eu', 'S'], scale_factor=4)

plt.savefig('./img/element_band.png')
plt.close()
# ==========================================================
# ==========================================================


# ==========================================================
# ------------- Element SPD Band Structure ----------------
# ==========================================================
fig7 = plt.figure(figsize=(3, 3), dpi=300)
ax7 = fig7.add_subplot(111)
plt.ylabel('$E - E_{F}$ $(eV)$', fontsize=6)
plt.ylim(-6, 6)
plt.tight_layout(pad=0.2)
plt.subplots_adjust(right=0.84)
plt.tick_params(labelsize=6, length=2.5)
plt.tick_params(axis='x', length=0)

band_slab.plot_element_spd(ax=ax7, elements=['As'], scale_factor=4, order=['p', 's', 'd'])

plt.savefig('./img/element_spd_band.png')
plt.close()
# ==========================================================
# ==========================================================


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

band_slab.plot_element_orbitals(ax=ax8, element_orbital_pairs=[['In', 1], ['As', 3], ['S', 0], ['Eu', 6]], scale_factor=4)

plt.savefig('./img/element_orbital_band.png')
plt.close()
# ==========================================================
# ==========================================================


"""
This section will plot the density of states
"""

dos = Dos(
    folder=dos_folder,
)

# ==========================================================
# ----------------------- Plain DOS ------------------------
# ==========================================================
fig9 = plt.figure(figsize=(7, 3), dpi=300)
ax9_1 = fig9.add_subplot(121)
plt.ylabel('$E - E_{F}$ $(eV)$', fontsize=6)
plt.xlabel('Density', fontsize=6)
plt.title("energyaxis='y'", fontsize=8)
plt.ylim(-6, 6)
plt.xlim(0,3)
plt.tight_layout(pad=0.2)
plt.tick_params(labelsize=6, length=2.5)
plt.tick_params(axis='x', length=0)

ax9_2 = fig9.add_subplot(122)
plt.xlabel('$E - E_{F}$ $(eV)$', fontsize=6)
plt.ylabel('Density', fontsize=6)
plt.title("energyaxis='x'", fontsize=8)
plt.xlim(-6, 6)
plt.ylim(0,3)
plt.tight_layout(pad=0.2)
plt.tick_params(labelsize=6, length=2.5)
plt.tick_params(axis='x', length=0)

plt.subplots_adjust(right=0.98, wspace=0.2)

dos.plot_plain(ax=ax9_1)
dos.plot_plain(ax=ax9_2, energyaxis='x')

plt.savefig('./img/dos/plain_dos.png')
plt.close()
# ==========================================================
# ==========================================================


# ==========================================================
# ----------------------- SPD DOS ------------------------
# ==========================================================
fig10 = plt.figure(figsize=(7, 3), dpi=300)
ax10_1 = fig10.add_subplot(121)
plt.ylabel('$E - E_{F}$ $(eV)$', fontsize=6)
plt.xlabel('Density', fontsize=6)
plt.title("energyaxis='y'", fontsize=8)
plt.ylim(-6, 6)
plt.xlim(0,3)
plt.tight_layout(pad=0.2)
plt.tick_params(labelsize=6, length=2.5)
plt.tick_params(axis='x', length=0)

ax10_2 = fig10.add_subplot(122)
plt.xlabel('$E - E_{F}$ $(eV)$', fontsize=6)
plt.ylabel('Density', fontsize=6)
plt.title("energyaxis='x'", fontsize=8)
plt.xlim(-6, 6)
plt.ylim(0,3)
plt.tight_layout(pad=0.2)
plt.tick_params(labelsize=6, length=2.5)
plt.tick_params(axis='x', length=0)

plt.subplots_adjust(right=0.95, wspace=0.25)

dos.plot_spd(ax=ax10_1)
dos.plot_spd(ax=ax10_2, energyaxis='x')

plt.savefig('./img/dos/spd_dos.png')
plt.close()
# ==========================================================
# ==========================================================


# ==========================================================
# --------------------- Orbital DOS ------------------------
# ==========================================================
fig11 = plt.figure(figsize=(7, 3), dpi=300)
ax11_1 = fig11.add_subplot(121)
plt.ylabel('$E - E_{F}$ $(eV)$', fontsize=6)
plt.xlabel('Density', fontsize=6)
plt.title("energyaxis='y'", fontsize=8)
plt.ylim(-6, 6)
plt.xlim(0,3)
plt.tight_layout(pad=0.2)
plt.tick_params(labelsize=6, length=2.5)
plt.tick_params(axis='x', length=0)

ax11_2 = fig11.add_subplot(122)
plt.xlabel('$E - E_{F}$ $(eV)$', fontsize=6)
plt.ylabel('Density', fontsize=6)
plt.title("energyaxis='x'", fontsize=8)
plt.xlim(-6, 6)
plt.ylim(0,3)
plt.tight_layout(pad=0.2)
plt.tick_params(labelsize=6, length=2.5)
plt.tick_params(axis='x', length=0)

plt.subplots_adjust(right=0.93, wspace=0.35)

dos.plot_orbitals(ax=ax11_1, orbitals=[0,1,2,3,4,5,6,7,8])
dos.plot_orbitals(ax=ax11_2, orbitals=[0,1,2,3,4,5,6,7,8], energyaxis='x')

plt.savefig('./img/dos/orbital_dos.png')
plt.close()
# ==========================================================
# ==========================================================


# ==========================================================
# ------------------ Atom Orbital DOS ----------------------
# ==========================================================
fig12 = plt.figure(figsize=(7, 3), dpi=300)
ax12_1 = fig12.add_subplot(121)
plt.ylabel('$E - E_{F}$ $(eV)$', fontsize=6)
plt.xlabel('Density', fontsize=6)
plt.title("energyaxis='y'", fontsize=8)
plt.ylim(-6, 6)
plt.xlim(0,3)
plt.tight_layout(pad=0.2)
plt.tick_params(labelsize=6, length=2.5)
plt.tick_params(axis='x', length=0)

ax12_2 = fig12.add_subplot(122)
plt.xlabel('$E - E_{F}$ $(eV)$', fontsize=6)
plt.ylabel('Density', fontsize=6)
plt.title("energyaxis='x'", fontsize=8)
plt.xlim(-6, 6)
plt.ylim(0,3)
plt.tight_layout(pad=0.2)
plt.tick_params(labelsize=6, length=2.5)
plt.tick_params(axis='x', length=0)

plt.subplots_adjust(right=0.93, wspace=0.35)

dos.plot_atom_orbitals(ax=ax12_1, atom_orbital_pairs=[[0,0],[0,1],[0,3]])
dos.plot_atom_orbitals(ax=ax12_2, atom_orbital_pairs=[[0,0],[0,1],[0,3]], energyaxis='x')

plt.savefig('./img/dos/atom_orbital_dos.png')
plt.close()
# ==========================================================
# ==========================================================


# ==========================================================
# ---------------------- Atom DOS --------------------------
# ==========================================================
fig13 = plt.figure(figsize=(7, 3), dpi=300)
ax13_1 = fig13.add_subplot(121)
plt.ylabel('$E - E_{F}$ $(eV)$', fontsize=6)
plt.xlabel('Density', fontsize=6)
plt.title("energyaxis='y'", fontsize=8)
plt.ylim(-6, 6)
plt.xlim(0,3)
plt.tight_layout(pad=0.2)
plt.tick_params(labelsize=6, length=2.5)
plt.tick_params(axis='x', length=0)

ax13_2 = fig13.add_subplot(122)
plt.xlabel('$E - E_{F}$ $(eV)$', fontsize=6)
plt.ylabel('Density', fontsize=6)
plt.title("energyaxis='x'", fontsize=8)
plt.xlim(-6, 6)
plt.ylim(0,3)
plt.tight_layout(pad=0.2)
plt.tick_params(labelsize=6, length=2.5)
plt.tick_params(axis='x', length=0)

plt.subplots_adjust(right=0.93, wspace=0.35)

dos.plot_atoms(ax=ax13_1, atoms=[0,1])
dos.plot_atoms(ax=ax13_2, atoms=[0,1], energyaxis='x')

plt.savefig('./img/dos/atom_dos.png')
plt.close()
# ==========================================================
# ==========================================================



dos_slab = Dos(
    folder=slab_dos_folder,
)

# ==========================================================
# ------------------- Element DOS --------------------------
# ==========================================================
fig14 = plt.figure(figsize=(7, 3), dpi=300)
ax14_1 = fig14.add_subplot(121)
plt.ylabel('$E - E_{F}$ $(eV)$', fontsize=6)
plt.xlabel('Density', fontsize=6)
plt.title("energyaxis='y'", fontsize=8)
plt.ylim(-6, 6)
plt.xlim(0,260)
plt.tight_layout(pad=0.2)
plt.tick_params(labelsize=6, length=2.5)
plt.tick_params(axis='x', length=0)

ax14_2 = fig14.add_subplot(122)
plt.xlabel('$E - E_{F}$ $(eV)$', fontsize=6)
plt.ylabel('Density', fontsize=6)
plt.title("energyaxis='x'", fontsize=8)
plt.xlim(-6, 6)
plt.ylim(0,260)
plt.tight_layout(pad=0.2)
plt.tick_params(labelsize=6, length=2.5)
plt.tick_params(axis='x', length=0)

plt.subplots_adjust(right=0.93, wspace=0.35)

dos_slab.plot_elements(ax=ax14_1, elements=['In', 'As', 'Eu', 'S'])
dos_slab.plot_elements(ax=ax14_2, elements=['In', 'As', 'Eu', 'S'], energyaxis='x')

plt.savefig('./img/dos/elements_dos.png')
plt.close()
# ==========================================================
# ==========================================================


# ==========================================================
# ---------------- Element Orbital DOS ---------------------
# ==========================================================
fig15 = plt.figure(figsize=(7, 3), dpi=300)
ax15_1 = fig15.add_subplot(121)
plt.ylabel('$E - E_{F}$ $(eV)$', fontsize=6)
plt.xlabel('Density', fontsize=6)
plt.title("energyaxis='y'", fontsize=8)
plt.ylim(-6, 6)
plt.xlim(0,260)
plt.tight_layout(pad=0.2)
plt.tick_params(labelsize=6, length=2.5)
plt.tick_params(axis='x', length=0)

ax15_2 = fig15.add_subplot(122)
plt.xlabel('$E - E_{F}$ $(eV)$', fontsize=6)
plt.ylabel('Density', fontsize=6)
plt.title("energyaxis='x'", fontsize=8)
plt.xlim(-6, 6)
plt.ylim(0,260)
plt.tight_layout(pad=0.2)
plt.tick_params(labelsize=6, length=2.5)
plt.tick_params(axis='x', length=0)

plt.subplots_adjust(right=0.93, wspace=0.35)

dos_slab.plot_element_orbitals(ax=ax15_1, element_orbital_pairs=[['In',0],['As',1],['Eu',0],['Eu',5],['Eu',6],['Eu',7],['Eu',8],['S',3], ['Eu', 10]])
dos_slab.plot_element_orbitals(ax=ax15_2, element_orbital_pairs=[['In',0],['As',1],['Eu',0],['Eu',5],['Eu',6],['Eu',7],['Eu',8],['S',3], ['Eu', 10]], energyaxis='x')

plt.savefig('./img/dos/elements_orbitals_dos.png')
plt.close()
# ==========================================================
# ==========================================================


# ==========================================================
# ------------------- Element SPD DOS ----------------------
# ==========================================================
fig16 = plt.figure(figsize=(7, 3), dpi=300)
ax16_1 = fig16.add_subplot(121)
plt.ylabel('$E - E_{F}$ $(eV)$', fontsize=6)
plt.xlabel('Density', fontsize=6)
plt.title("energyaxis='y'", fontsize=8)
plt.ylim(-6, 6)
plt.xlim(0,260)
plt.tight_layout(pad=0.2)
plt.tick_params(labelsize=6, length=2.5)
plt.tick_params(axis='x', length=0)

ax16_2 = fig16.add_subplot(122)
plt.xlabel('$E - E_{F}$ $(eV)$', fontsize=6)
plt.ylabel('Density', fontsize=6)
plt.title("energyaxis='x'", fontsize=8)
plt.xlim(-6, 6)
plt.ylim(0,260)
plt.tight_layout(pad=0.2)
plt.tick_params(labelsize=6, length=2.5)
plt.tick_params(axis='x', length=0)

plt.subplots_adjust(right=0.93, wspace=0.35)

dos_slab.plot_element_spd(ax=ax16_1, elements=['Eu'])
dos_slab.plot_element_spd(ax=ax16_2, elements=['Eu'], energyaxis='x')

plt.savefig('./img/dos/elements_spd_dos.png')
plt.close()
# ==========================================================
# ==========================================================


# ==========================================================
# ---------------------- Layer DOS -------------------------
# ==========================================================
fig17 = plt.figure(figsize=(7, 3), dpi=300)
ax17_1 = fig17.add_subplot(121)
plt.ylabel('$E - E_{F}$ $(eV)$', fontsize=6)
plt.xlabel('Layers', fontsize=6)
plt.title("energyaxis='y'", fontsize=8)
plt.ylim(-2, 2.8)
# plt.xlim(0,3)
plt.tight_layout(pad=0.2)
plt.tick_params(labelsize=6, length=2.5)
plt.tick_params(axis='x', length=0)

ax17_2 = fig17.add_subplot(122)
plt.xlabel('$E - E_{F}$ $(eV)$', fontsize=6)
plt.ylabel('Layers', fontsize=6)
plt.title("energyaxis='x'", fontsize=8)
plt.xlim(-2, 2.8)
# plt.ylim(0,3)
plt.tight_layout(pad=0.2)
plt.tick_params(labelsize=6, length=2.5)
plt.tick_params(axis='x', length=0)

plt.subplots_adjust(right=0.93, wspace=0.35)

dos_slab.plot_layers(ax=ax17_1, sigma=1.5, cmap='magma', ylim=[-2,3])
dos_slab.plot_layers(ax=ax17_2, energyaxis='x', sigma=1.5, cmap='magma', ylim=[-2,3])
ax17_1.axvline(x=42, color='white', linewidth=2, linestyle='--')
ax17_2.axhline(y=42, color='white', linewidth=2, linestyle='--')
ax17_1.annotate('InAs', xy=(20,0.5), xycoords='data', color='white', ha='center')
ax17_1.annotate('EuS', xy=(60,0.5), xycoords='data', color='white', ha='center')
ax17_2.annotate('InAs', xy=(0.5,20), xycoords='data', color='white', ha='center')
ax17_2.annotate('EuS', xy=(0.5,60), xycoords='data', color='white', ha='center')

plt.savefig('./img/dos/layer_dos.png')
plt.close()
# ==========================================================
# ==========================================================


