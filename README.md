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

## Band Structures

### Plain Band Structure
```python
from vaspvis import standard

standard.band_plain(
    folder=band_folder
)
```
<img src="./img/band_plain.png"  width="600" height="450">


### s, p, d Projected Band Structure
```python
from vaspvis import standard

standard.band_spd(
    folder=band_folder
)
```
<img src="./img/band_spd.png"  width="600" height="450">


### Orbital Projected Band Structure
```python
from vaspvis import standard

standard.band_orbitals(
    folder=band_folder,
    orbitals=[0, 1, 2, 3, 4, 5, 6, 7, 8],
)
```
<img src="./img/band_orbital.png"  width="600" height="450">

### Atom-Orbtial Projected Band Structure
```python
from vaspvis import standard

standard.band_atom_orbital(
    folder=band_folder,
    atom_orbital_pairs=[[0,1], [0,3], [1, 1], [1,7]]
)
```
<img src="./img/band_dos_atom_orbitals.png"  width="600" height="450">


### Atom Projected Band Structure
```python
from vaspvis import standard

standard.band_atoms(
    folder=band_folder,
    atoms=[0, 1],
)
```
<img src="./img/band_atoms.png"  width="600" height="450">


### Element Projected Band Structure
```python
from vaspvis import standard

standard.band_elements(
    folder=band_folder,
    elements=['In', 'As'],
)
```
<img src="./img/band_elements.png"  width="600" height="450">


### Element s, p, d Projected Band Structure
```python
from vaspvis import standard

standard.band_element_spd(
    folder=band_folder,
    elements=['As'],
)
```
<img src="./img/band_element_spd.png"  width="600" height="450">


### Element Orbital Projected Band Structure
```python
from vaspvis import standard

standard.band_element_orbitals(
    folder=band_folder,
    element_orbital_pairs=[['As', 2], ['In', 3]],
)
```
<img src="./img/band_element_orbital.png"  width="600" height="450">



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









