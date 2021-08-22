from pyprocar.utilsprocar import UtilsProcar
from pyprocar.procarparser import ProcarParser
import numpy as np
import os

folder = '../../vaspvis_data/band_InAs'

parser = ProcarParser()
parser.readFile(os.path.join(folder, 'PROCAR_repaired'))

projected_eigenvals = np.transpose(parser.spd[:, :, -1, -1, -1])

print(projected_eigenvals[:, 0])
