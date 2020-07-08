# vaspvis
A highly flexible and customizable library for visualizing electronic structure data from VASP calculations.

# vaspvis.core.band

## `class Band()`

Method for constructing and plotting band structures from VASP calculations.

```python
plot_plain(ax, color='black', linewidth=1.25)
```
<img src="./img/band/plain_band.png"  width="600" height="450">


```python
plot_spd(self, ax, scale_factor=5, order=['s', 'p', 'd'], color_dict=None, legend=True, linewidth=0.75, band_color='black')
```
<img src="./img/band/spd_band.png"  width="600" height="450">

```python
plot_orbitals(self, orbitals, ax, scale_factor=5, color_dict=None, legend=True, linewidth=0.75, band_color='black')
```
<img src="./img/band/orbital_band.png"  width="600" height="450">

```python
plot_atom_orbitals(self, atom_orbital_pairs, ax, scale_factor=5, color_dict=None, legend=True, linewidth=0.75, band_color='black')
```
<img src="./img/band/atom_orbital_band.png"  width="600" height="450">

```python
plot_atoms(self, atoms, ax, scale_factor=5, color_dict=None, legend=True, linewidth=0.75, band_color='black')
```
<img src="./img/band/atom_band.png"  width="600" height="450">

```python
plot_elements(self, elements, ax, scale_factor=5, color_dict=None, legend=True, linewidth=0.75, band_color='black')
```
<!--<img src="./img/band/element_band.png"  width="450" height="450">-->

```python
plot_element_orbitals(self, element_orbital_pairs, ax, scale_factor=5, color_dict=None, legend=True, linewidth=0.75, band_color='black')
```
<!--<img src="./img/band/element_orbital_band.png"  width="450" height="450">-->

```python
plot_element_spd(self, elements, ax, order=['s', 'p', 'd'], scale_factor=5, color_dict=None, legend=True, linewidth=0.75, band_color='black')
```

# vaspvis.core.dos

## `class Dos()`

Method for constructing and plotting the density of states from VASP calculations.

```python
plot_plain(ax, linewidth=1.5, fill=True, alpha=0.3, sigma=0.05, energyaxis='y')
```
<img src="./img/dos/plain_dos.png"  width="1050" height="425">

```python
plot_spd(ax, order=['s', 'p', 'd'], fill=True, alpha=0.3, linewidth=1.5, sigma=0.05, energyaxis='y', color_dict=None)
```
<img src="./img/dos/spd_dos.png"  width="1050" height="425">

```python
plot_orbitals(ax, orbitals, fill=True, alpha=0.3, linewidth=1.5, sigma=0.05, energyaxis='y', color_dict=None)
```
<img src="./img/dos/orbital_dos.png"  width="1050" height="425">

```python
plot_atom_orbitals(ax, atom_orbital_pairs, fill=True, alpha=0.3, linewidth=1.5, sigma=0.05, energyaxis='y', color_dict=None)
```
<img src="./img/dos/atom_orbital_dos.png"  width="1050" height="425">

```python
plot_atoms(ax, atoms, fill=True, alpha=0.3, linewidth=1.5, sigma=0.05, energyaxis='y', color_dict=None)
```
<img src="./img/dos/atom_dos.png"  width="1050" height="425">

```python
plot_elements(ax, elements, fill=True, alpha=0.3, linewidth=1.5, sigma=0.05, energyaxis='y', color_dict=None)
```
<!--<img src="./img/dos/elements_dos.png"  width="1050" height="425">-->

```python
plot_element_orbitals(ax, elements, orbitals, fill=True, alpha=0.3, linewidth=1.5, sigma=0.05, energyaxis='y', color_dict=None)
```
<!--<img src="./img/dos/elements_orbitals_dos.png"  width="1050" height="425">-->

```python
plot_element_spd(ax, elements, order=['s', 'p', 'd'], fill=True, alpha=0.3, linewidth=1.5, sigma=0.05, energyaxis='y', color_dict=None)
```
<!--<img src="./img/dos/elements_spd_dos.png"  width="1050" height="425">-->

```python
plot_layers(ax, ylim=[-6, 6], cmap='magma', sigma=5)
```
<!--<img src="./img/dos/layer_dos.png"  width="1050" height="425">-->









