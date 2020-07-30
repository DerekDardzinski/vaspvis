# vaspvis
A highly flexible and customizable library for visualizing electronic structure data from VASP calculations.

[![Documentation Status](https://readthedocs.org/projects/vaspvis/badge/?version=latest)](https://vaspvis.readthedocs.io/en/latest/?badge=latest)
      

# Usage

This package was designed to give VASP users a flexible and easy to understand method
for generating a wide variaty of band structures and density of states plots. There are
three main modules in this package:

* `Band`
* `Dos`
* `standard`

The `Band` and `Dos` modules allow for the highest level of flexibility because the user
needs to pass in their own matplotlib axis, letting the user completely design the 
external appearence of their plot. The `Band` and `Dos` modules will then parse the
VASP output data and append the it to the axis.

The `standard` module uses the `Band` and `Dos` modules internally and
was designed for those people who are not familiar with matplotlib
or don't need to completely customize their own figure. There are a total of 48 different
styles of plots to choose from in this module. It gives the user the capability to project
onto any orbital, any atom, or any element in their structure, as well as individual orbitals
on any atom or element. There are also options for spin polarized band structures and density
of states as well, letting the user make intricate plots with only a few lines of code.


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
**Important Note:** This package parses the vasprun.xml, KPOINTS, and POSCAR files, be sure that
they are in the folder you load into vaspvis.

**Important Note:** For spin projected orbitals you must load the spin up and spin down chanels separately using the `spin = 'up'` or `spin = 'down'` options with loading data. Default is `spin = 'up'`.

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


### Plain Band Structure HSE

> All other band structure can be plotted for HSE this is just an example on how to
load the structure using the standard module.

```python
from vaspvis import standard

standard.band_plain(
    folder=band_folder_hse,
    hse=True,
    kpath='GXWLGK',
    n=20,
)
```
<img src="./img/band_plain_hse.png"  width="600" height="450">


### s, p, d Projected Band Structure
```python
from vaspvis import standard

standard.band_spd(
    folder=band_folder
)
```
<img src="./img/band_spd.png"  width="650" height="450">


### Orbital Projected Band Structure
```python
from vaspvis import standard

standard.band_orbitals(
    folder=band_folder,
    orbitals=[0, 1, 2, 3, 4, 5, 6, 7, 8],
)
```
<img src="./img/band_orbital.png"  width="650" height="450">

### Atom-Orbtial Projected Band Structure
```python
from vaspvis import standard

standard.band_atom_orbitals(
    folder=band_folder,
    atom_orbital_pairs=[[0,1], [0,3], [1, 1], [1,7]]
)
```
<img src="./img/band_atom_orbitals.png"  width="650" height="450">


### Atom Projected Band Structure
```python
from vaspvis import standard

standard.band_atoms(
    folder=band_folder,
    atoms=[0, 1],
)
```
<img src="./img/band_atoms.png"  width="650" height="450">


### Element Projected Band Structure
```python
from vaspvis import standard

standard.band_elements(
    folder=band_folder,
    elements=['In', 'As'],
)
```
<img src="./img/band_elements.png"  width="650" height="450">


### Element s, p, d Projected Band Structure
```python
from vaspvis import standard

standard.band_element_spd(
    folder=band_folder,
    elements=['As'],
)
```
<img src="./img/band_element_spd.png"  width="650" height="450">


### Element Orbital Projected Band Structure
```python
from vaspvis import standard

standard.band_element_orbitals(
    folder=band_folder,
    element_orbital_pairs=[['As', 2], ['In', 3]],
)
```
<img src="./img/band_element_orbital.png"  width="650" height="450">


## Density of Statess

### Plain Density of States
```python
from vaspvis import standard

standard.dos_plain(
    folder=dos_folder,
    energyaxis='x',
)
```
<img src="./img/dos_plain.png"  width="600" height="450">


### s, p, d Projected Density of States
```python
from vaspvis import standard

standard.dos_spd(
    folder=dos_folder,
    energyaxis='x',
)
```
<img src="./img/dos_spd.png"  width="650" height="450">


### Orbital Projected Density of States
```python
from vaspvis import standard

standard.dos_orbitals(
    folder=dos_folder,
    orbitals=[0, 1, 2, 3, 4, 5, 6, 7, 8],
    energyaxis='x',
)
```
<img src="./img/dos_orbitals.png"  width="650" height="450">

### Atom-Orbtial Projected Density of States
```python
from vaspvis import standard

standard.dos_atom_orbital(
    folder=dos_folder,
    atom_orbital_pairs=[[0,1], [0,3], [1, 1], [1,7]],
    energyaxis='x',
)
```
<img src="./img/dos_atom_orbitals.png"  width="650" height="450">


### Atom Projected Density of States
```python
from vaspvis import standard

standard.dos_atoms(
    folder=dos_folder,
    atoms=[0, 1],
    energyaxis='x',
)
```
<img src="./img/dos_atoms.png"  width="650" height="450">


### Element Projected Density of States
```python
from vaspvis import standard

standard.dos_elements(
    folder=dos_folder,
    elements=['In', 'As'],
    energyaxis='x',
)
```
<img src="./img/dos_elements.png"  width="650" height="450">


### Element s, p, d Projected Density of States
```python
from vaspvis import standard

standard.dos_element_spd(
    folder=dos_folder,
    elements=['As'],
    energyaxis='x',
)
```
<img src="./img/dos_element_spd.png"  width="650" height="450">


### Element Orbital Projected Density of States
```python
from vaspvis import standard

standard.dos_element_orbitals(
    folder=dos_folder,
    element_orbital_pairs=[['As', 2], ['In', 3]],
    energyaxis='x',
)
```
<img src="./img/dos_element_orbitals.png"  width="650" height="450">


## Band Structure / Density of Statess

### Plain Band Structure / Density of States
```python
from vaspvis import standard

standard.band_dos_plain(
    band_folder=band_folder,
    dos_folder=dos_folder,
)
```
<img src="./img/band_dos_plain.png"  width="900" height="450">


### s, p, d Projected Band Structure / Density of States
```python
from vaspvis import standard

standard.band_dos_spd(
    band_folder=band_folder,
    dos_folder=dos_folder,
)
```
<img src="./img/band_dos_spd.png"  width="950" height="450">


### Orbital Projected Band Structure / Density of States
```python
from vaspvis import standard

standard.band_dos_orbitals(
    band_folder=band_folder,
    dos_folder=dos_folder,
    orbitals=[0, 1, 2, 3, 4, 5, 6, 7, 8],
)
```
<img src="./img/band_dos_orbitals.png"  width="950" height="450">

### Atom-Orbtial Projected Band Structure / Density of States
```python
from vaspvis import standard

standard.band_dos_atom_orbital(
    band_folder=band_folder,
    dos_folder=dos_folder,
    atom_orbital_pairs=[[0,1], [0,3], [1, 1], [1,7]]
)
```
<img src="./img/band_dos_atom_orbitals.png"  width="950" height="450">


### Atom Projected Band Structure / Density of States
```python
from vaspvis import standard

standard.band_dos_atoms(
    band_folder=band_folder,
    dos_folder=dos_folder,
    atoms=[0, 1],
)
```
<img src="./img/band_dos_atoms.png"  width="950" height="450">


### Element Projected Band Structure / Density of States
```python
from vaspvis import standard

standard.band_dos_elements(
    band_folder=band_folder,
    dos_folder=dos_folder,
    elements=['In', 'As'],
)
```
<img src="./img/band_dos_elements.png"  width="950" height="450">


### Element s, p, d Projected Band Structure / Density of States
```python
from vaspvis import standard

standard.band_dos_element_spd(
    band_folder=band_folder,
    dos_folder=dos_folder,
    elements=['As'],
)
```
<img src="./img/band_dos_element_spd.png"  width="950" height="450">


### Element Orbital Projected Band Structure / Density of States
```python
from vaspvis import standard

standard.band_dos_element_orbitals(
    band_folder=band_folder,
    dos_folder=dos_folder,
    element_orbital_pairs=[['As', 2], ['In', 3]],
)
```
<img src="./img/band_dos_element_orbitals.png"  width="950" height="450">





