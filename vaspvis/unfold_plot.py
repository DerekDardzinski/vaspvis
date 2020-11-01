from unfold import unfold, make_kpath, removeDuplicateKpoints, EBS_scatter
import matplotlib.pyplot as plt
import numpy as np
import os

# basis vector of the primitive cell

fermi = os.popen('grep fermi ../../vaspvis_data/band_unfolding2/OUTCAR').read().split()[14]
#  fermi = 4
M = [[-1,1,0],[-1,-1,1],[0,0,1]]
#  print(np.linalg.det(M))
WaveSuper = unfold(M=M, wavecar='../../vaspvis_data/band_unfolding2/WAVECAR', lsorbit=True)
cell = [[0, 3.2397,3.2397],
      [3.2397, 0, 3.2397],
      [3.2397, 3.2397, 0]]

kpts = [[0.0, 0.5, 0],            # M
        [0.0, 0.0, 0],            # G
        [0.5, 0.5, 0]]            # M
kpath = make_kpath(kpts, nseg=30)

sw = WaveSuper.spectral_weight(kpath)
bands = []
probs = []
for i in range(len(sw[0][0])):
    bands.append(sw[0,:,i,0])
    probs.append(sw[0,:,i,1])
#  print(sw[0][0][:,0])
#  [print(i) for i in sw[0]]
# show the effective band structure with scatter

fig, ax = plt.subplots(figsize=(3,4), dpi=300)
colors=['red', 'green', 'blue', 'orange', 'purple', 'grey', 'yellow', 'black']
sizes=[100,75,50,25,5]

#  for (band, prob) in zip(bands, probs):
for i in range(len(bands)):
    ax.scatter(
        range(len(bands[i])),
        np.array(bands[i]) - float(fermi),
        c="red",
        s=probs[i] * 10,
        #  s=5,
        #  c=colors[i],
    )
    ax.plot(
        range(len(bands[i])),
        np.array(bands[i]) - float(fermi),
        color="black",
        #  s=probs[i],
        #  s=5,
        #  c=colors[i],
    )


ax.set_ylim(-6,6)
#  ax.set_xlim(15,45)
plt.show()

#  EBS_scatter(kpath, cell, sw, nseg=30, eref=float(fermi),
                #  ylim=(-5, 0),kpath_label=['X','G','X'],
                #  factor=20)
