from core import band
from core import dos
import matplotlib.pyplot as plt

band_folder = '../vaspvis_data/band'
dos_folder = '../vaspvis_data/dos'
hse_folder = '../vaspvis_data/hse'
slab_folder = '../vaspvis_data/slab'


"""
This section will plot band structures calculated without HSE
"""

# Load Data
pbe = band.BandStructure(
    folder=band_folder,
    projected=True,
    spin='up',
)


# Plain Band Structure
fig1 = plt.figure(figsize=(4, 3), dpi=300)
ax1 = fig1.add_subplot(111)
plt.ylabel('$E - E_{F}$ $(eV)$', fontsize=6)
plt.ylim(-6, 6)
plt.tight_layout(pad=0.5)
plt.tick_params(labelsize=6, length=1.5)
plt.tick_params(axis='x', length=0)

pbe.plot_plain(ax=ax1, linewidth=1)

plt.savefig('./img/plain_pbe.png')
plt.close()


# spd Band Structure
fig2 = plt.figure(figsize=(4, 3), dpi=300)
ax2 = fig2.add_subplot(111)
plt.ylabel('$E - E_{F}$ $(eV)$', fontsize=6)
plt.ylim(-6, 6)
plt.tight_layout(pad=0.5)
plt.tick_params(labelsize=6, length=2.5)
plt.tick_params(axis='x', length=0)

pbe.plot_spd(ax=ax2)

plt.legend(ncol=3, loc='upper left', fontsize=5)
plt.savefig('./img/spd_pbe.png')
plt.close()
