from pymatgen.io.vasp.outputs import Chgcar
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.core.structure import Structure
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from multiprocessing import Pool
from scipy.interpolate import CubicSpline
from scipy.integrate import trapz
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
        self.interface_structure = Structure.from_file(
            os.path.join(self.interface_folder, "POSCAR"),
        )
        self.film_structure = Structure.from_file(
            os.path.join(self.film_folder, "POSCAR"),
        )
        self.substrate_structure = Structure.from_file(
            os.path.join(self.substrate_folder, "POSCAR"),
        )
        self.area = self._get_area()
        pool = Pool(processes=3)
        [self.interface_data, self.substrate_data, self.film_data] = pool.map(
            self._load_data,
            (self.interface_folder, self.substrate_folder, self.film_folder),
        )
        pool.close()
        pool.join()

        (
            self.x_plot,
            self.y_plot,
            self.film_min,
            self.film_y,
            self.sub_max,
            self.sub_y,
            self.sub_diff,
            self.film_diff,
        ) = self._get_planar_charge_transfer()

    @property
    def substrate_charge_transfer(self):
        return self.sub_diff

    @property
    def film_charge_transfer(self):
        return self.film_diff

    def _get_area(self):
        a = self.interface_structure.lattice.matrix[0]
        b = self.interface_structure.lattice.matrix[1]

        return np.linalg.norm(np.cross(a, b))

    def _load_data(self, folder):
        preloaded_data = os.path.isfile(os.path.join(folder, "chgcar.npy"))
        if preloaded_data:
            with open(os.path.join(folder, "chgcar.npy"), "rb") as p:
                data = np.load(p)
        else:
            chgcar = Chgcar.from_file(os.path.join(folder, "CHGCAR"))
            data = chgcar.data["total"]
            np.save(os.path.join(folder, "chgcar.npy"), data)

        a_vals = np.linspace(0, 1, data.shape[0])
        b_vals = np.linspace(0, 1, data.shape[1])
        c_vals = np.linspace(0, 1, data.shape[2])

        return data, a_vals, b_vals, c_vals

    def _planar_average(self, data):
        return data.mean(axis=(0, 1)) / self.area

    def _locate_ind(self, data, coord):
        return np.abs(data - coord).argmin()

    def _get_planar_charge_transfer(self):
        int_data, _, _, int_c = self.interface_data
        sub_data, _, _, sub_c = self.substrate_data
        film_data, _, _, film_c = self.film_data

        total_c_len = self.interface_structure.lattice.matrix[-1, -1]

        int_data = self._planar_average(int_data)
        sub_data = self._planar_average(sub_data)
        film_data = self._planar_average(film_data)

        int_cs = CubicSpline(int_c * total_c_len, int_data)
        sub_cs = CubicSpline(sub_c * total_c_len, sub_data)
        film_cs = CubicSpline(film_c * total_c_len, film_data)

        int_min = self.interface_structure.cart_coords[:, -1].min()
        int_max = self.interface_structure.cart_coords[:, -1].max()
        sub_max = self.substrate_structure.cart_coords[:, -1].max()
        film_min = self.film_structure.cart_coords[:, -1].min()
        mid_line = (sub_max + film_min) / 2

        sub_to_mid_z = np.linspace(int_min, mid_line, 1001)
        mid_to_film_z = np.linspace(mid_line, int_max, 1001)

        sub_to_mid_data = (
            int_cs(sub_to_mid_z) - film_cs(sub_to_mid_z) - sub_cs(sub_to_mid_z)
        )

        mid_to_film_data = (
            int_cs(mid_to_film_z)
            - film_cs(mid_to_film_z)
            - sub_cs(mid_to_film_z)
        )

        sub_chg = int_cs(sub_max) - film_cs(sub_max) - sub_cs(sub_max)
        film_chg = int_cs(film_min) - film_cs(film_min) - sub_cs(film_min)

        sub_charge = trapz(y=sub_to_mid_data, x=sub_to_mid_z)
        film_charge = trapz(y=mid_to_film_data, x=mid_to_film_z)

        all_z = np.linspace(int_min, int_max, 1001)
        all_data = int_cs(all_z) - film_cs(all_z) - sub_cs(all_z)
        all_z -= mid_line

        return (
            all_z,
            all_data,
            film_min - mid_line,
            film_chg,
            sub_max - mid_line,
            sub_chg,
            sub_charge,
            film_charge,
        )

    def plot_charge_transfer(
        self,
        ax,
        color="black",
        linestyle="-",
        linewidth=1,
        film_line_color="red",
        film_linestyle="-",
        film_linewidth=0.75,
        substrate_line_color="blue",
        substrate_linestyle="-",
        substrate_linewidth=0.75,
        method=0,
        plot_film_and_substrate_lines=True,
        add_labels=True,
        xlim=[-10, 10],
        label_fontsize=10,
    ):

        ax.axhline(y=0, color="grey", linewidth=0.5)

        if plot_film_and_substrate_lines:
            ax.axvline(
                x=self.film_min,
                color=film_line_color,
                linewidth=film_linewidth,
                linestyle=film_linestyle,
            )
            ax.axvspan(
                self.film_min,
                xlim[1],
                color=film_line_color,
                alpha=0.05,
            )
            ax.scatter(
                [self.film_min],
                [self.film_y],
                ec=film_line_color,
                fc="white",
                marker="o",
                s=15,
                zorder=100,
            )
            ax.axvline(
                x=self.sub_max,
                color=substrate_line_color,
                linewidth=substrate_linewidth,
                linestyle=substrate_linestyle,
            )
            ax.axvspan(
                xlim[0],
                self.sub_max,
                color=substrate_line_color,
                alpha=0.05,
            )
            ax.scatter(
                [self.sub_max],
                [self.sub_y],
                ec=substrate_line_color,
                fc="white",
                marker="o",
                s=15,
                zorder=100,
            )

        ax.plot(
            self.x_plot,
            self.y_plot,
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
        )
        ax.set_xlim(xlim[0], xlim[1])
        y_min, y_max = ax.get_ylim()
        new_y = max(abs(y_min), abs(y_max))
        ax.set_ylim(-new_y, new_y)

        if add_labels:
            sub_formula = self.substrate_structure.composition.reduced_formula
            film_formula = self.film_structure.composition.reduced_formula

            latex_sub_formula = []
            for i in sub_formula:
                if i.isdigit():
                    latex_sub_formula.append("$_{" + f"{i}" + "}$")
                else:
                    latex_sub_formula.append(i)

            latex_sub_formula = "".join(latex_sub_formula)

            latex_film_formula = []
            for i in film_formula:
                if i.isdigit():
                    latex_film_formula.append("$_{" + f"{i}" + "}$")
                else:
                    latex_film_formula.append(i)

            latex_film_formula = "".join(latex_film_formula)

            sub_annotation = "\n".join(
                [latex_sub_formula, f"{self.sub_diff:.3f} eV"]
            )

            film_annotation = "\n".join(
                [latex_film_formula, f"{self.film_diff:.3f} eV"]
            )

            bbox_dict = dict(
                boxstyle="round,pad=0.3",
                fc="white",
                ec="black",
                lw=1,
            )

            ax.annotate(
                sub_annotation,
                xy=((xlim[0] + self.sub_max) / 2, 0.90 * new_y),
                xycoords="data",
                ha="center",
                va="top",
                bbox=bbox_dict,
                fontsize=label_fontsize,
            )

            ax.annotate(
                film_annotation,
                xy=((xlim[1] + self.film_min) / 2, 0.90 * new_y),
                xycoords="data",
                ha="center",
                va="top",
                bbox=bbox_dict,
                fontsize=label_fontsize,
            )


if __name__ == "__main__":
    chg = Charge(
        interface_folder="../../vaspvis_data/chgxfer/int",
        substrate_folder="../../vaspvis_data/chgxfer/sub",
        film_folder="../../vaspvis_data/chgxfer/film",
    )
    fig, ax = plt.subplots(figsize=(4, 3), dpi=400)
    ax.set_ylabel("Charge Transfer $(e/\\AA)$")
    ax.set_xlabel("z-distance $(\\AA)$")
    chg.plot_charge_transfer(ax=ax)
    fig.tight_layout()
    fig.savefig("test.png")
