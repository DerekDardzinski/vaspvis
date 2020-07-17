# vaspvis
A highly flexible and customizable library for visualizing electronic structure data from VASP calculations.

[![Documentation Status](https://readthedocs.org/projects/vaspvis/badge/?version=latest)](https://vaspvis.readthedocs.io/en/latest/?badge=latest)
      

# Installation

```bash
pip install vaspvis
```

# Loading Data


```python
from vaspvis import Band, Dos

# Non-HSE Calculation (plain band structure)
bs = Band(folder='path to vasp output folder')


# Non-HSE Calculation (projected band structure)
bs_projected = Band(folder='path to vasp output folder', projected=True)

# HSE Calculation (plain band structure)
bs_hse = Band(
    folder='path to vasp output folder',
    hse=True,
    kpath='GXWLGK', # Path used in calculation
    n=30, # Number of points between with high symmetry points
)

# HSE Calculation (projected band structure)
bs_hse = Band(
    folder='path to vasp output folder',
    projected=True,
    hse=True,
    kpath='GXWLGK', # Path used in calculation
    n=30, # Number of points between with high symmetry points
)

# Density of states (projected or non-projected)
dos = Dos(folder='path to vasp output folder')
```

**Important Note:** For spin projected orbitals you must load the spin up and spin down chanels separately using the `spin = 'up'` or `spin = 'down'` options with loading data. Default is `spin = 'up'`. An example of a spin projected band plot is coming soon.


# Examples

## Plain Band Structure
<img src="./img/band/plain_band.png"  width="600" height="450">

## s, p, d Projected Band Structure
<img src="./img/band/spd_band.png"  width="600" height="450">

## Orbital Projected Band Structure
<img src="./img/band/orbital_band.png"  width="600" height="450">

## Atom-Orbtial Projected Band Structure
<img src="./img/band/atom_orbital_band.png"  width="600" height="450">

## Atom Projected Band Structure
<img src="./img/band/atom_band.png"  width="600" height="450">

<!--<img src="./img/band/element_band.png"  width="450" height="450">-->

<!--<img src="./img/band/element_orbital_band.png"  width="450" height="450">-->


## Plain Density of States
<img src="./img/dos/plain_dos.png"  width="1050" height="425">

## s, p, d Projected Density of States
<img src="./img/dos/spd_dos.png"  width="1050" height="425">

## Orbtial Projected Density of States
<img src="./img/dos/orbital_dos.png"  width="1050" height="425">

## Atom-Orbtial Projected Density of States
<img src="./img/dos/atom_orbital_dos.png"  width="1050" height="425">

## Atom Projected Density of States
<img src="./img/dos/atom_dos.png"  width="1050" height="425">

<!--<img src="./img/dos/elements_dos.png"  width="1050" height="425">-->

<!--<img src="./img/dos/elements_orbitals_dos.png"  width="1050" height="425">-->

<!--<img src="./img/dos/elements_spd_dos.png"  width="1050" height="425">-->

<!--<img src="./img/dos/layer_dos.png"  width="1050" height="425">-->









