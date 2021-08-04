from pymatgen.io.vasp.outputs import Chgcar
from pymatgen.io.vasp.inputs import Poscar
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from multiprocessing import Pool
import numpy as np
import time
import os

class Charge:
    """
    This class contains all the methods for performing charge transfer calculations
    """

    def __init__(
        self,
        interface_folder,
        substrate_folder,
        film_folder,
    ):
        self.interface_folder = interface_folder
        self.substrate_folder = substrate_folder
        self.film_folder = film_folder
        self.interface_poscar = Poscar.from_file(
            os.path.join(self.interface_folder, 'POSCAR'),
            check_for_POTCAR=False,
            read_velocities=False
        )
        self.film_poscar = Poscar.from_file(
            os.path.join(self.film_folder, 'POSCAR'),
            check_for_POTCAR=False,
            read_velocities=False
        )
        self.substrate_poscar = Poscar.from_file(
            os.path.join(self.substrate_folder, 'POSCAR'),
            check_for_POTCAR=False,
            read_velocities=False
        )
        pool = Pool(processes=3)
        [self.interface_data, self.substrate_data, self.film_data] = pool.map(
            self._load_data,
            (self.interface_folder, self.substrate_folder, self.film_folder),
        )
        pool.close()
        pool.join()

    def _load_data(self, folder):
        preloaded_data = os.path.isfile(os.path.join(folder, 'chgcar.npy'))
        if preloaded_data:
            with open(os.path.join(folder, 'chgcar.npy'), 'rb') as p:
                data = np.load(p)
        else:
            chgcar = Chgcar.from_file(os.path.join(folder, 'CHGCAR'))
            data = chgcar.data['total']
            np.save(os.path.join(folder, 'chgcar.npy'), data)

        a_vals = np.linspace(0,1,data.shape[0])
        b_vals = np.linspace(0,1,data.shape[1])
        c_vals = np.linspace(0,1,data.shape[2])

        return data, a_vals, b_vals, c_vals

    def _planar_average(self, data):
        return data.mean(axis=(0,1))

    def _locate_ind(self, data, coord):
        return np.abs(data - coord).argmin()

    #  def macroscopic_average(potential, periodicity, resolution):
        #  """Getting the macroscopic average of potential
        #  Args:
            #  potential : array containig the electrostaticpotential/charge density
            #  periodicity : real number; the period over which to average
            #  resolution : the grid resolution in the direction of averaging
        #  Returns:
            #  macro_average : array with the macroscopically averaged values"""
#
#
        #  macro_average = np.zeros(shape=(len(potential)))
        #  period_points = int((periodicity/resolution))
        #  # Period points must be even
        #  if period_points % 2 != 0:
            #  period_points = period_points + 1
#
        #  length = len(potential)
        #  for i in range(length):
            #  start = i - int(period_points / 2)
            #  end = i + int(period_points / 2)
            #  if start < 0:
                #  start = start + length
                #  macro_average[i] = macro_average[i] + sum(potential[0:end]) + sum(potential[start:length])
                #  macro_average[i] = macro_average[i] / period_points
            #  elif end >= length:
                #  end = end - length
                #  macro_average[i] = macro_average[i] + sum(potential[start:length]) + sum(potential[0:end])
                #  macro_average[i] = macro_average[i] / period_points
            #  else:
                #  macro_average[i] = macro_average[i] + sum(potential[start:end]) / period_points
#
        #  print("Average of the average = ", numpy.average(macro_average))
#
        #  return macro_average

    def plot_charge_transfer(
        self,
        ax,
        color='black',
        linestyle='-',
        linewidth=1,
        interface_line_color='red',
        interface_linestyle='-',
        interface_linewidth=0.75,
        method=0,
        plot_interface_line=True,
    ):
        int_data, _, _, int_c = self.interface_data
        sub_data, _, _, sub_c = self.substrate_data
        film_data, _, _, film_c = self.film_data

        if method == 0:
            int_data = self._planar_average(int_data)
            sub_data = self._planar_average(sub_data)
            film_data = self._planar_average(film_data)

        int_min = self.interface_poscar.structure.frac_coords[:,-1].min()
        int_max = self.interface_poscar.structure.frac_coords[:,-1].max()
        sub_max = self.substrate_poscar.structure.frac_coords[:,-1].max()
        film_min = self.film_poscar.structure.frac_coords[:,-1].min()
        mid_line = (sub_max + film_min) / 2
        shift = 0.5 - mid_line
        mid_ind = self._locate_ind(int_c, mid_line)
        roll_val = int(int_c.shape[0] / 2) - mid_ind

        matrix = self.interface_poscar.structure.lattice.matrix
        area = np.linalg.norm(np.cross(matrix[0], matrix[1]))
        volume = np.linalg.det(matrix)
        c_len = np.linalg.norm(matrix[2])

        data = np.roll(int_data - sub_data - film_data, roll_val) * (area / volume)

        ax.plot(
            np.linspace(-c_len/2, c_len/2, int_c.shape[0]),
            data,
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
        )
        ax.set_xlim(-c_len/2,c_len/2)
        #  ax.xaxis.set_major_locator(MultipleLocator(0.25))

        if plot_interface_line:
            ax.axvline(
                x=0,
                color=interface_line_color,
                linewidth=interface_linewidth,
                linestyle=interface_linestyle,
            )


if __name__ == "__main__":
    chg = Charge(
        interface_folder='../../vaspvis_data/chgxfer/int',
        substrate_folder='../../vaspvis_data/chgxfer/sub',
        film_folder='../../vaspvis_data/chgxfer/film',
    )
    fig, ax = plt.subplots(figsize=(4,3), dpi=400)
    ax.set_ylabel('Charge Transfer $(e/\\AA)$')
    ax.set_xlabel('z-distance $(\\AA)$')
    chg.plot_charge_transfer(ax=ax)
    fig.tight_layout()
    fig.savefig('test.png')

        
