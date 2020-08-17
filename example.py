from vaspvis import standard


band_folder = '../vaspvis_data/band_InAs'
dos_folder = '../vaspvis_data/dos_InAs'

band_folder_hse = '../../vaspvis_data/band_InAs_hse'

# ==================================================
# -------------- Plain Band Structure --------------
# ==================================================

standard.band_plain(
    folder=band_folder
)


# ==================================================
# ----------- HSE Plain Band Structure -------------
# ==================================================

standard.band_plain(
    folder=band_folder_hse,
    output='band_plain_hse.png',
    hse=True,
    kpath='GXWLGK',
    n=20,
)


# ==================================================
# --------------- SPD Band Structure ---------------
# ==================================================

standard.band_spd(
    folder=band_folder
)


# ==================================================
# ------------ Orbital Band Structure --------------
# ==================================================

standard.band_orbitals(
    folder=band_folder,
    orbitals=[0, 1, 2, 3, 4, 5, 6, 7, 8],
)


# ==================================================
# -------------- Atom Band Structure ---------------
# ==================================================

standard.band_atoms(
    folder=band_folder,
    atoms=[0, 1],
)


# ==================================================
# ---------- Atom Orbital Band Structure -----------
# ==================================================

standard.band_atom_orbitals(
    folder=band_folder,
    atom_orbital_pairs=[[0,1], [0,3], [1, 1], [1,7]]
)


# ==================================================
# ------------ Element Band Structure --------------
# ==================================================

standard.band_elements(
    folder=band_folder,
    elements=['In', 'As'],
)


# ==================================================
# ---------- Element SPD Band Structure ------------
# ==================================================

standard.band_element_spd(
    folder=band_folder,
    elements=['As'],
)


# ==================================================
# ------- Element Orbitals Band Structure ----------
# ==================================================

standard.band_element_orbitals(
    folder=band_folder,
    element_orbital_pairs=[['As', 2], ['In', 3]],
)


# ==================================================
# -------------- Plain Dos Structure --------------
# ==================================================

standard.dos_plain(
    folder=dos_folder,
    energyaxis='x',
)


# ==================================================
# --------------- SPD Dos Structure ---------------
# ==================================================

standard.dos_spd(
    folder=dos_folder,
    energyaxis='x',
)


# ==================================================
# ------------ Orbital Dos Structure --------------
# ==================================================

standard.dos_orbitals(
    folder=dos_folder,
    orbitals=[0, 1, 2, 3, 4, 5, 6, 7, 8],
    energyaxis='x',
)


# ==================================================
# -------------- Atom Dos Structure ---------------
# ==================================================

standard.dos_atoms(
    folder=dos_folder,
    atoms=[0, 1],
    energyaxis='x',
)


# ==================================================
# ---------- Atom Orbital Dos Structure -----------
# ==================================================

standard.dos_atom_orbitals(
    folder=dos_folder,
    atom_orbital_pairs=[[0,1], [0,3], [1, 1], [1,7]],
    energyaxis='x',
)


# ==================================================
# ------------ Element Dos Structure --------------
# ==================================================

standard.dos_elements(
    folder=dos_folder,
    elements=['In', 'As'],
    energyaxis='x',
)


# ==================================================
# ---------- Element SPD Dos Structure ------------
# ==================================================

standard.dos_element_spd(
    folder=dos_folder,
    elements=['As'],
    energyaxis='x',
)


# ==================================================
# ------- Element Orbitals Dos Structure ----------
# ==================================================

standard.dos_element_orbitals(
    folder=dos_folder,
    element_orbital_pairs=[['As', 0], ['In', 3], ['As', 8]],
    energyaxis='x',
)


# ==================================================
# -------------- Plain Dos Structure --------------
# ==================================================

standard.band_dos_plain(
    band_folder=band_folder,
    dos_folder=dos_folder
)


# ==================================================
# --------------- SPD Dos Structure ---------------
# ==================================================

standard.band_dos_spd(
    band_folder=band_folder,
    dos_folder=dos_folder
)


# ==================================================
# ------------ Orbital Dos Structure --------------
# ==================================================

standard.band_dos_orbitals(
    band_folder=band_folder,
    dos_folder=dos_folder,
    orbitals=[0, 1, 2, 3, 4, 5, 6, 7, 8],
)


# ==================================================
# -------------- Atom Dos Structure ---------------
# ==================================================

standard.band_dos_atoms(
    band_folder=band_folder,
    dos_folder=dos_folder,
    atoms=[0, 1],
)


# ==================================================
# ---------- Atom Orbital Dos Structure -----------
# ==================================================

standard.band_dos_atom_orbitals(
    band_folder=band_folder,
    dos_folder=dos_folder,
    atom_orbital_pairs=[[0,1], [0,3], [1, 1], [1,7]]
)


# ==================================================
# ------------ Element Dos Structure --------------
# ==================================================

standard.band_dos_elements(
    band_folder=band_folder,
    dos_folder=dos_folder,
    elements=['In', 'As'],
)


# ==================================================
# ---------- Element SPD Dos Structure ------------
# ==================================================

standard.band_dos_element_spd(
    band_folder=band_folder,
    dos_folder=dos_folder,
    elements=['As'],
)


# ==================================================
# ------- Element Orbitals Dos Structure ----------
# ==================================================

standard.band_dos_element_orbitals(
    band_folder=band_folder,
    dos_folder=dos_folder,
    element_orbital_pairs=[['As', 2], ['In', 3]],
)















