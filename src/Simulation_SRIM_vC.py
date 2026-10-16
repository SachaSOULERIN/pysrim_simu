"""
Simulation_SRIM.py
===================

Simule, pour 3 faisceaux d'ions (H, C, O) a energie initiale fixee :

  1) la perte d'energie et le pouvoir d'arrete en fonction de la profondeur
     d'EAU traversee (courbe de Bragg), via le module SR (Stopping & Range)
     de SRIM et une integration CSDA (Continuous-Slowing-Down-Approximation).

  2) la perte d'energie et le pouvoir d'arrete en fonction de la profondeur
     de DETECTEUR traversee, pour 3 detecteurs differents (PMMA + poudre
     scintillante Rouge / Verte / Bleue), modelises comme un empilement de
     "cellules" a 2 couches (PMMA + scintillateur) dont les epaisseurs
     respectent un RATIO DE MASSE fixe entre les deux composants.
     Une boucle adaptative reduit la taille de cellule (donc l'epaisseur
     totale simulee) tant que les ions sont completement arretes avant la
     fin du detecteur, jusqu'a un plancher de 1 micron.

Auteur : Claude + Sacha SOULERIN

------------------------------------------------------------------------
CE QUI RESTE A FAIRE AVANT DE LANCER (marque "TO DO" dans le fichier) :
------------------------------------------------------------------------
  - SRIM_DIRECTORY : chemin vers votre installation SRIM-2013 (TRIM.exe).
  - Les energies de deplacement (E_d) de chaque element/materiau : les
    valeurs mises ici sont soit issues de vos notebooks d'exemple, soit
    une valeur par defaut generique de 25 eV (defaut SRIM) LARGEMENT
    utilisee faute de mieux -> A VERIFIER/CORRIGER dans la litterature
    pour chaque materiau (eau, PMMA, ZnCdS:Ag, YAG:Ce, LSO:Ce).
  - Les densites/compositions exactes de vos 3 poudres scintillantes.
  - Le nombre d'ions par simulation TRIM (NUMBER_IONS_DETECTOR) : augmentez
    pour lisser les courbes IONIZ/PHONON (mais le temps de calcul augmente
    d'autant).
  - Le nombre de cellules du detecteur (N_CELLS_DETECTOR) qui represente
    la granularite reelle de votre composite PMMA+poudre.
------------------------------------------------------------------------
"""

import os
import shutil
import subprocess
import warnings
from pathlib import Path
from dataclasses import dataclass, field
import re
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

from srim import TRIM, SR, Target, Layer, Ion, Element
from srim.output import Results


# =========================================================================
# 0. CONFIGURATION GENERALE
# =========================================================================

# Chemin vers le dossier contenant TRIM.exe / SRModule.exe (SRIM-2013)
SRIM_DIRECTORY = Path(r'C:\Program Files\SRIM-2013')

# Dossier ou seront copies tous les fichiers bruts SRIM + figures produites
DATA_SAVING_PATH = Path(r'C:\Users\Sacha\Documents\Simulations\SRIM\OUTPUT\pySRIM')
DATA_SAVING_PATH.mkdir(parents=True, exist_ok=True)

# Energie de deplacement par defaut si un element n'a pas de valeur connue
# (c'est la valeur par defaut de SRIM lui-meme). A remplacer element par
# element avec des valeurs de la litterature si vous en avez.
DEFAULT_E_D = 25.0  # eV

# Nombre d'ions simules pour les calculs TRIM detaillees (detecteurs)
NUMBER_IONS_DETECTOR = 50

# Nombre de "cellules" (PMMA + scintillateur) representant le detecteur.
# L'epaisseur totale simulee = N_CELLS_DETECTOR * taille_de_cellule.
N_CELLS_DETECTOR = 1

# Epaisseur minimale (plancher) d'une cellule PMMA+scintillateur, en microns
MIN_CELL_THICKNESS_UM = 1.0

# Ratio de masse scintillateur : PMMA a respecter dans chaque cellule
# (~2 g de scintillateur pour 1 g de PMMA d'apres vos detecteurs reels)
MASS_RATIO_SCINT_TO_PMMA = 2.0


# =========================================================================
# 1. FAISCEAUX D'IONS
# =========================================================================

# energie en eV. A ajuster vous-meme comme indique dans votre message.
IONS = [
    {"identifier": "H", "energy": 58.4e6},   # 58.4 MeV
    {"identifier": "C", "energy": 1.14e9},   # 1.14 GeV
    {"identifier": "O", "energy": 1.52e9},   # 1.52 GeV
]


# =========================================================================
# 2. DEFINITION DES MATERIAUX (eau, PMMA, scintillateurs)
# =========================================================================

def elt(stoich: float, E_d: float = DEFAULT_E_D, lattice: float = 0.0,
        surface: float = 3.0) -> dict:
    """Raccourci pour definir un element dans un Layer pysrim."""
    return {"stoich": stoich, "E_d": E_d, "lattice": lattice, "surface": surface}


# --- eau : H2O -----------------------------------------------------------
WATER_DENSITY = 1.0  # g/cm3
WATER_COMPOSITION = {
    # TODO : E_d de l'hydrogene dans l'eau mal defini dans la litterature ;
    # 10 eV est une valeur souvent utilisee a defaut de mieux (a verifier).
    "H": elt(stoich=2.0, E_d=10.0),
    "O": elt(stoich=1.0, E_d=25.0),
}

# --- PMMA : (C5O2H8)n ------------------------------------------------------
PMMA_DENSITY = 1.18  # g/cm3 (TODO a verifier / mesurer)
PMMA_COMPOSITION = {
    "H": elt(stoich=8.0, E_d=10.0),   # TODO
    "O": elt(stoich=2.0, E_d=25.0),
    "C": elt(stoich=5.0, E_d=20.0),
}

# --- scintillateur "Rouge" : ZnS:Ag (Zn,Cd)S:Ag approx ZnCdS:Ag ----------
RED_DENSITY = 4.18  # g/cm3 (TODO a verifier, m_red = 2.03 g dans vos fichiers ISD_fab)
RED_COMPOSITION = {
    "Zn": elt(stoich=8.0, E_d=25.0),   # TODO valeur par defaut, a verifier
    "Cd": elt(stoich=2.0, E_d=25.0),   # TODO
    "S":  elt(stoich=5.0, E_d=25.0),   # TODO
    "Ag": elt(stoich=0.01, E_d=25.0),   # TODO
}

# --- scintillateur "Vert" : Y3Al5O12:Ce (YAG:Ce) --------------------------
GREEN_DENSITY = 3.848  # g/cm3 (densite YAG massif ; 3.848 et 4.15 notes dans
                       # vos fichiers concernent probablement la poudre -
                       # TODO choisir/verifier la bonne valeur)
GREEN_COMPOSITION = {
    "Y":  elt(stoich=3.0, E_d=25.0),   # TODO
    "Al": elt(stoich=5.0, E_d=25.0),   # TODO
    "O":  elt(stoich=12.0, E_d=25.0),
    # Ce en dopant : stoechiometrie tres faible (dopage typique < 1 at.%)
    # remplacee ici par une valeur non-nulle car SRIM/pysrim n'accepte pas
    # un stoich de 0.0 pour un element declare. TODO ajuster au taux de
    # dopage reel (ex. 0.01-0.03 pour quelques % de Ce).
    "Ce": elt(stoich=0.01, E_d=25.0),  # TODO
}

# --- scintillateur "Bleu" : Lu2SiO5:Ce (LSO:Ce) ---------------------------
BLUE_DENSITY = 4.15  # g/cm3 (TODO a verifier, valeur LSO massif typique)
BLUE_COMPOSITION = {
    "Lu": elt(stoich=2.0, E_d=25.0),   # TODO
    "Si": elt(stoich=1.0, E_d=35.0),
    "O":  elt(stoich=5.0, E_d=25.0),
    "Ce": elt(stoich=0.01, E_d=25.0),  # TODO cf. remarque ci-dessus
}

DETECTORS = {
    "Red": {"composition": RED_COMPOSITION, "density": RED_DENSITY},
    "Green":    {"composition": GREEN_COMPOSITION, "density": GREEN_DENSITY},
    "Blue":    {"composition": BLUE_COMPOSITION, "density": BLUE_DENSITY},
}


# =========================================================================
# 3. PARTIE EAU : courbe de Bragg via module SR + integration CSDA
# =========================================================================

def _to_float_locale(token: str) -> float:
    """Convertit un nombre eventuellement au format europeen (virgule
    decimale) en float. C'est ce qui plante dans la version d'origine
    (ValueError sur '999,999')."""
    token = token.strip()
    # Cas typique d'un artefact/valeur de saturation SRIM (ex '999,999')
    # -> on le remplace juste par un point decimal, la valeur redevient
    # exploitable (999.999) mais on la signalera comme suspecte plus haut.
    return float(token.replace(",", "."))


def read_sr_output(directory) -> dict:
    """Parseur maison, robuste, du fichier SR_OUTPUT.txt produit par
    srim.SR.run(). Remplace le parseur pysrim natif qui plante des que
    SRIM ecrit ses nombres avec une virgule decimale (installation
    localisee en francais) ou une ligne de saturation type '999,999'.

    Retourne un dict avec les tableaux numpy :
        energy_keV, Se, Sn, range_um, long_straggle_um, lat_straggle_um
    """
    path = Path(directory) / "SR_OUTPUT.txt"
    with open(path, "r", encoding="latin1") as f:
        lines = f.readlines()

    energy_conv = {"eV": 1e-3, "keV": 1.0, "MeV": 1e3, "GeV": 1e6}  # -> keV
    length_conv = {"A": 1e-4, "um": 1.0, "mm": 1e3, "cm": 1e4}      # -> um

    header_idx = None
    footer_idx = None
    for i, line in enumerate(lines):
        if header_idx is None and "dE/dx" in line and "Ion" in line:
            header_idx = i
        if footer_idx is None and "Multiply" in line and "Stopping" in line:
            footer_idx = i
            break

    if header_idx is None or footer_idx is None:
        raise ValueError(f"Impossible de localiser le tableau dans {path}")

    # les donnees commencent typiquement 3 lignes apres l'en-tete
    # ("Energy" / unites / "-----") et finissent quelques lignes avant
    # "Multiply Stopping..." (une ligne de tirets separe les deux)
    data_lines = lines[header_idx + 3: footer_idx]

    out = {"energy_keV": [], "Se": [], "Sn": [], "range_um": [],
           "long_straggle_um": [], "lat_straggle_um": []}

    for raw_line in data_lines:
        line = raw_line.strip()
        if not line or set(line) <= {"-"}:
            continue
        tokens = line.split()
        try:
            energy_val = _to_float_locale(tokens[0])
            energy_unit = tokens[1]
            Se = _to_float_locale(tokens[2])
            Sn = _to_float_locale(tokens[3])
            range_val = _to_float_locale(tokens[4])
            range_unit = tokens[5]
            long_strag = _to_float_locale(tokens[6])
            long_unit = tokens[7]
            lat_strag = _to_float_locale(tokens[8])
            lat_unit = tokens[9]
        except (IndexError, ValueError, KeyError):
            warnings.warn(f"Ligne SR_OUTPUT.txt ignoree (format inattendu) : {line!r}")
            continue

        if energy_unit not in energy_conv or range_unit not in length_conv:
            warnings.warn(f"Unite inconnue, ligne ignoree : {line!r}")
            continue

        out["energy_keV"].append(energy_val * energy_conv[energy_unit])
        out["Se"].append(Se)
        out["Sn"].append(Sn)
        out["range_um"].append(range_val * length_conv[range_unit])
        out["long_straggle_um"].append(long_strag * length_conv.get(long_unit, np.nan))
        out["lat_straggle_um"].append(lat_strag * length_conv.get(lat_unit, np.nan))

    for k in out:
        out[k] = np.array(out[k], dtype=float)

    # Trie par energie croissante et retire d'eventuels doublons/points
    # de saturation aberrants (range non-monotone)
    order = np.argsort(out["energy_keV"])
    for k in out:
        out[k] = out[k][order]

    good = np.concatenate(([True], np.diff(out["range_um"]) > 0))
    n_removed = np.sum(~good)
    if n_removed:
        warnings.warn(f"{n_removed} point(s) non-monotone(s) retires de SR_OUTPUT.txt "
                       "(probable artefact de saturation SRIM)")
    for k in out:
        out[k] = out[k][good]

    return out
 
 
def run_sr_safe(sr: SR, srim_directory) -> dict:
    """Lance le module SR (Stopping & Range) SANS passer par sr.run().
 
    sr.run() de pysrim ecrit l'input, lance SRModule.exe, PUIS essaie de
    parser lui-meme SR_OUTPUT.txt avec son propre parseur (SRResults),
    qui plante des que SRIM ecrit un nombre avec une virgule decimale ou
    une valeur de saturation (ex '999,999') -> c'est exactement l'erreur
    que vous avez rencontree, et elle se produit AVANT que notre propre
    parseur robuste (read_sr_output) ait la moindre chance d'intervenir.
 
    On reproduit donc ici manuellement les deux premieres etapes de
    sr.run() (ecriture de l'input + lancement de l'exe), sans jamais
    appeler SRResults, puis on lit le fichier avec read_sr_output().
    """
    current_directory = os.getcwd()
    sr_module_dir = Path(srim_directory) / "SR Module"
    try:
        os.chdir(sr_module_dir)
        sr._write_input_file()
        exe = str(Path(".") / "SRModule.exe")
        if shutil.which("wine"):
            subprocess.check_call(["wine", exe])
        else:
            subprocess.check_call([exe])
    finally:
        os.chdir(current_directory)
 
    return read_sr_output(sr_module_dir)


def water_bragg_curve(ion_def: dict, srim_directory=SRIM_DIRECTORY,
                       output_type: int = 3, n_points: int = 400) -> dict:
    """Calcule E(profondeur) et pouvoir d'arret(profondeur) dans l'eau
    pour un ion donne, par integration CSDA des tables SR (Se, Sn, Range
    vs Energie). output_type=3 -> stopping en MeV/mm.

    Retourne un dict : depth_um, energy_MeV, Se, Sn, Se_plus_Sn
    (unites de Se/Sn = celles choisies par output_type, ici MeV/mm)
    """
    ion = Ion(ion_def["identifier"], energy=ion_def["energy"])

    water_layer = Layer(WATER_COMPOSITION, density=WATER_DENSITY, width=1.0)
    # width n'a pas d'importance pour un calcul SR (pas de TRIM ici)

    sr = SR(water_layer, ion, output_type=output_type)
    table = run_sr_safe(sr, srim_directory)

    energy_keV = table["energy_keV"]
    range_um = table["range_um"]
    Se, Sn = table["Se"], table["Sn"]

    E0_keV = ion_def["energy"] / 1.0e3  # eV -> keV
    if E0_keV > energy_keV.max():
        warnings.warn(
            f"Energie initiale ({E0_keV/1e3:.2f} MeV) au-dela de la table SR "
            f"(max {energy_keV.max()/1e3:.2f} MeV) : extrapolation, a verifier."
        )

    # CSDA range de l'ion a son energie initiale (interpolation de Range(E))
    R0_um = np.interp(E0_keV, energy_keV, range_um)

    depth_um = np.linspace(0.0, R0_um, n_points)
    # E(depth) = inverse de Range(E) : range restant = R0 - depth
    remaining_range = np.clip(R0_um - depth_um, range_um.min(), None)
    # Range(E) est croissante avec E -> on peut inverser par interpolation
    energy_at_depth_keV = np.interp(remaining_range, range_um, energy_keV)

    Se_at_depth = np.interp(energy_at_depth_keV, energy_keV, Se)
    Sn_at_depth = np.interp(energy_at_depth_keV, energy_keV, Sn)

    return {
        "ion": ion_def["identifier"],
        "depth_um": depth_um,
        "energy_MeV": energy_at_depth_keV / 1e3,
        "Se": Se_at_depth,
        "Sn": Sn_at_depth,
        "Se_plus_Sn": Se_at_depth + Sn_at_depth,
        "stopping_units": table,
    }


def plot_water_curves(curves: List[dict], save_path: Optional[Path] = None):
    fig, (ax_E, ax_S) = plt.subplots(1, 2, figsize=(14, 5))

    for c in curves:
        ax_E.plot(c["depth_um"], c["energy_MeV"], label=c["ion"])
        ax_S.plot(c["depth_um"], c["Se_plus_Sn"], label=c["ion"])

    ax_E.set_xlabel("Profondeur d'eau (µm)")
    ax_E.set_ylabel("Energie de l'ion (MeV)")
    ax_E.set_title("Courbe de Bragg (energie) dans l'eau")
    ax_E.legend()
    ax_E.grid(alpha=0.3)

    ax_S.set_xlabel("Profondeur d'eau (µm)")
    ax_S.set_ylabel("Pouvoir d'arret Se+Sn (MeV/mm)")
    ax_S.set_title("Pouvoir d'arret dans l'eau")
    ax_S.legend()
    ax_S.grid(alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


# =========================================================================
# 4. PARTIE DETECTEUR : cellule PMMA+scintillateur a ratio de masse fixe
# =========================================================================

def cell_layer_widths(rho_pmma: float, rho_scint: float,
                      mass_ratio: float,
                       cell_thickness_um: float) -> Tuple[float, float]:
    """Calcule les epaisseurs (en Angstrom) de PMMA et de scintillateur
    dans une cellule d'epaisseur totale donnee, telles que :

        (masse scint) / (masse PMMA) = mass_ratio
        (rho scint) / (rho PMMA) = rho_ratio
        w_pmma + w_scint = cell_thickness
        (w_scint) / (w_pmma) = np.cbrt(mass_ratio / rho_ratio)

    Les deux dernieres equations froment un systeme a deux inconnues,
    dont le resultat est renvoyer en Angstroms (1µm = 1000 Angestroms):
    w_pmma_um = cell_thickness_um / (1 + length_ratio)
    w_scint_um = (cell_thickness_um * length_ratio) / (1 + length_ratio)
    """
    #mass_ratio = mass_scint / mass_pmma
    rho_ratio = rho_scint / rho_pmma
    length_ratio = np.cbrt(mass_ratio / rho_ratio) # cube root pour passer du volume a l'epaisseur lineaire
    w_pmma_um = cell_thickness_um / (1.0 + length_ratio)
    w_scint_um = cell_thickness_um - w_pmma_um
    #w_scint_um = (cell_thickness_um * length_ratio) / (1 + length_ratio)

    ANGSTROM_PER_UM = 1.0e4
    return w_pmma_um * ANGSTROM_PER_UM, w_scint_um * ANGSTROM_PER_UM


def build_detector_target(scint_composition: dict,
                          rho_pmma: float, rho_scint: float,
                          mass_ratio: float,
                           cell_thickness_um: float, n_cells: int) -> Target:
    """Construit un Target pysrim = n_cells x [couche PMMA, couche scint],
    avec les epaisseurs de chaque couche calculees pour respecter le ratio
    de masse scintillateur:PMMA demande."""
    w_pmma_A, w_scint_A = cell_layer_widths(rho_pmma, rho_scint, mass_ratio,
                                             cell_thickness_um)

    pmma_layer = Layer(PMMA_COMPOSITION, density=rho_pmma, width=w_pmma_A)
    scint_layer = Layer(scint_composition, density=rho_scint, width=w_scint_A)

    layers = []
    for _ in range(n_cells):
        layers.append(pmma_layer) # Talk to Carole to know what is the order of her layers
        layers.append(scint_layer)

    return Target(layers)


def estimate_initial_cell_size(ion_def: dict, rho_pmma: float, rho_scint: float,
                                srim_directory=SRIM_DIRECTORY,
                                n_cells: int = N_CELLS_DETECTOR,
                                min_cell_um: float = MIN_CELL_THICKNESS_UM) -> float:
    """Estime une taille de cellule initiale 'raisonnable' a partir du
    parcours CSDA (module SR) de l'ion dans le scintillateur SEUL (cas le
    plus defavorable car generalement le plus dense). L'epaisseur totale
    visee (n_cells * cellule) est fixee a ~1.2x ce parcours, avec une
    marge pour que l'ion traverse tout le detecteur des la 1ere tentative."""
    ion = Ion(ion_def["identifier"], energy=ion_def["energy"])
    # composition non utilisee ici, seule la densite compte pour l'ordre
    # de grandeur du parcours -> on utilise la couche PMMA comme proxy
    # simple et rapide a calculer (evite de dupliquer un SR par materiau).
    probe_layer = Layer(PMMA_COMPOSITION, density=max(rho_pmma, rho_scint), width=1.0)
    sr = SR(probe_layer, ion, output_type=2)  # keV/micron
    table = run_sr_safe(sr, srim_directory)

    E0_keV = ion_def["energy"] / 1e3
    R0_um = float(np.interp(E0_keV, table["energy_keV"], table["range_um"]))

    total_thickness_um = 1.2 * R0_um
    cell_um = max(min_cell_um, total_thickness_um / n_cells)
    return cell_um
 
 
def find_output_file(srim_directory, filename: str) -> Path:
    """TRIM ecrit parfois ses fichiers de sortie directement dans
    srim_directory, parfois dans srim_directory/'SRIM Outputs' (selon la
    version de SRIM). On cherche dans les deux."""
    direct = Path(srim_directory) / filename
    if direct.is_file():
        return direct
    nested = Path(srim_directory) / "SRIM Outputs" / filename
    if nested.is_file():
        return nested
    raise FileNotFoundError(f"{filename} introuvable dans {srim_directory}")
 
 
def read_depth_table(path) -> np.ndarray:
    """Lit un fichier de sortie TRIM binne par profondeur (IONIZ.txt,
    E2RECOIL.txt, PHONON.txt, RANGE.txt, VACANCY.txt, ...) de facon
    robuste aux virgules decimales.
 
    Le lecteur natif de pysrim (utilise par trim.run() -> Results() ->
    Ioniz/EnergyToRecoils/... -> SRIM_Output._read_table) s'appuie sur
    np.genfromtxt, qui transforme SILENCIEUSEMENT toute valeur mal
    formee (ex. virgule decimale d'une installation SRIM localisee en
    francais) en NaN, SANS lever d'erreur. C'est exactement la cause des
    courbes de detecteur vides/NaN observees, sans aucun message dans le
    terminal : le calcul se deroule "normalement" mais les donnees sont
    silencieusement corrompues.
 
    Cette fonction reconstruit le tableau ligne par ligne, en remplacant
    les virgules par des points avant conversion, et s'arrete proprement
    des qu'une ligne n'est plus exploitable (fin du tableau) plutot que
    de produire des NaN.
    """
    with open(path, "rb") as f:
        content = f.read()
 
    match = re.search(rb'=+(.*)-+(?:\s+-+)+', content, re.DOTALL)
    if not match:
        raise ValueError(f"Impossible de localiser le tableau dans {path}")
    text = content[match.end():].decode("latin1")
 
    rows = []
    ncols = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if rows:
                break  # ligne vide apres le debut des donnees -> fin du tableau
            continue
        tokens = line.replace(",", ".").split()
        try:
            values = [float(t) for t in tokens]
        except ValueError:
            break
        if ncols is None:
            ncols = len(values)
        elif len(values) != ncols:
            break
        rows.append(values)
 
    if not rows:
        raise ValueError(f"Aucune donnee exploitable trouvee dans {path}")
 
    return np.array(rows)
 
 
def read_ioniz(srim_directory) -> Tuple[np.ndarray, np.ndarray]:
    """depth [Angstrom], ionisation par l'ion primaire [eV/(Angstrom.ion)]"""
    data = read_depth_table(find_output_file(srim_directory, "IONIZ.txt"))
    return data[:, 0], data[:, 1]
 
 
def read_e2recoil(srim_directory) -> Tuple[np.ndarray, np.ndarray]:
    """depth [Angstrom], energie nucleaire transferee par l'ion primaire
    [eV/(Angstrom.ion)]"""
    data = read_depth_table(find_output_file(srim_directory, "E2RECOIL.txt"))
    return data[:, 0], data[:, 1]
 
 
def run_trim_safe(trim: TRIM, srim_directory) -> Tuple[Tuple[np.ndarray, np.ndarray],
                                                         Tuple[np.ndarray, np.ndarray]]:
    """Lance un calcul TRIM SANS passer par trim.run().
 
    trim.run() de pysrim ecrit l'input, lance TRIM.exe, PUIS construit
    lui-meme un objet Results(srim_directory) qui relit tous les fichiers
    de sortie avec le parseur fragile decrit dans read_depth_table
    ci-dessus (NaN silencieux au lieu d'une erreur). On reproduit donc
    manuellement l'ecriture de l'input + le lancement de l'exe, sans
    jamais construire cet objet Results, puis on lit IONIZ.txt et
    E2RECOIL.txt nous-memes.
 
    Retourne ((depth_ioniz, ions_ioniz), (depth_e2recoil, ions_e2recoil)).
    """
    current_directory = os.getcwd()
    try:
        os.chdir(srim_directory)
        trim._write_input_files()
        exe = str(Path(".") / "TRIM.exe")
        if shutil.which("wine"):
            subprocess.check_call(["wine", exe])
        else:
            subprocess.check_call([exe])
    finally:
        os.chdir(current_directory)
 
    return read_ioniz(srim_directory), read_e2recoil(srim_directory)


def stopping_power_vs_depth(ioniz_data: Tuple[np.ndarray, np.ndarray],
                             e2recoil_data: Tuple[np.ndarray, np.ndarray]
                             ) -> Tuple[np.ndarray, np.ndarray]:
    """Reconstruit le pouvoir d'arret local de l'ion primaire en fonction
    de la profondeur, en sommant :
      - IONIZ.txt    -> perte electronique de l'ion primaire (eV/Angstrom-ion)
      - E2RECOIL.txt -> perte nucleaire (energie transferee aux atomes de
        recul) de l'ion primaire (eV/Angstrom-ion)
 
    ATTENTION : PHONON.txt (nombre de phonons/Angstrom, PAS une energie)
    ne doit pas etre utilise ici (erreur d'une version precedente de ce
    script).
 
    Unites retournees : depth en Angstrom, dE/dx en eV/Angstrom.
    """
    depth_ioniz, ions_ioniz = ioniz_data
    depth_e2r, ions_e2r = e2recoil_data
 
    depth = depth_ioniz
    ioniz_ion = np.interp(depth, depth_ioniz, ions_ioniz)
    nuclear_ion = np.interp(depth, depth_e2r, ions_e2r)
 
    dEdx = ioniz_ion + nuclear_ion  # eV / Angstrom
    return depth, dEdx


def energy_vs_depth(depth_A: np.ndarray, dEdx_eV_per_A: np.ndarray,
                     E0_eV: float) -> np.ndarray:
    """Integre le pouvoir d'arret pour obtenir l'energie residuelle de
    l'ion en fonction de la profondeur (energie nulle = ion arrete)."""
    cumulative_loss = np.concatenate(([0.0], np.cumsum(
        0.5 * (dEdx_eV_per_A[1:] + dEdx_eV_per_A[:-1]) * np.diff(depth_A)
    )))
    energy = E0_eV - cumulative_loss
    return np.clip(energy, 0.0, None)


def run_detector_adaptive(ion_def: dict, detector_name: str,
                           srim_directory=SRIM_DIRECTORY,
                           n_cells: int = N_CELLS_DETECTOR,
                           number_ions: int = NUMBER_IONS_DETECTOR,
                           mass_ratio: float = MASS_RATIO_SCINT_TO_PMMA,
                           min_cell_um: float = MIN_CELL_THICKNESS_UM,
                           max_iterations: int = 8) -> dict:
    """Boucle adaptative :
        1) construit le detecteur (n_cells cellules PMMA+scint) avec la
           taille de cellule courante,
        2) lance TRIM,
        3) verifie si les ions sont tous arretes avant la fin du target,
        4) si oui et si la taille de cellule est encore au-dessus du
           plancher, on divise la taille de cellule par 2 et on relance,
        5) sinon on s'arrete et on retourne les courbes obtenues.
    """
    scint = DETECTORS[detector_name]
    rho_pmma, rho_scint = PMMA_DENSITY, scint["density"]

    cell_um = estimate_initial_cell_size(ion_def, rho_pmma, rho_scint,
                                          srim_directory, n_cells, min_cell_um)

    out_dir = DATA_SAVING_PATH / f"{ion_def['identifier']}_{detector_name.split()[0]}"
    out_dir.mkdir(parents=True, exist_ok=True)

    last_result = None
    for iteration in range(max_iterations):
        target = build_detector_target(scint["composition"], rho_pmma, rho_scint,
                                        mass_ratio, cell_um, n_cells)
        ion = Ion(ion_def["identifier"], energy=ion_def["energy"])
        trim = TRIM(target, ion, number_ions=number_ions, calculation=1)
        ioniz_data, e2recoil_data = run_trim_safe(trim, srim_directory)
        iter_dir = out_dir / f"iter{iteration}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        TRIM.copy_output_files(str(srim_directory), str(out_dir / f"iter{iteration}"))

        depth_A, dEdx = stopping_power_vs_depth(ioniz_data, e2recoil_data)
        energy_eV = energy_vs_depth(depth_A, dEdx, ion_def["energy"])

        target_thickness_A = depth_A.max()
        # "arrete avant la fin" : l'energie residuelle touche 0 avant la
        # derniere tranche de profondeur simulee (avec une petite marge)
        stopped_early = bool(np.any(energy_eV <= 0.0) and
                              depth_A[np.argmax(energy_eV <= 0.0)] < 0.98 * target_thickness_A)

        last_result = {
            "ion": ion_def["identifier"],
            "detector": detector_name,
            "iteration": iteration,
            "cell_um": cell_um,
            "n_cells": n_cells,
            "total_thickness_um": target_thickness_A / 1e4,
            "depth_um": depth_A / 1e4,
            "dEdx_eV_per_A": dEdx,
            "energy_eV": energy_eV,
            "stopped_early": stopped_early,
        }

        if not stopped_early or cell_um <= min_cell_um:
            break

        cell_um = max(min_cell_um, cell_um / 2.0)

    return last_result


def plot_detector_curves(results_list: List[dict], save_path: Optional[Path] = None):
    fig, (ax_E, ax_S) = plt.subplots(1, 2, figsize=(14, 5))

    for r in results_list:
        label = f"{r['ion']} / {r['detector']}"
        ax_E.plot(r["depth_um"], r["energy_eV"] / 1e6, label=label)
        ax_S.plot(r["depth_um"], r["dEdx_eV_per_A"] * 1e4 / 1e6, label=label)  # eV/A -> MeV/um

    ax_E.set_xlabel("Profondeur de detecteur (µm)")
    ax_E.set_ylabel("Energie residuelle (MeV)")
    ax_E.set_title("Energie vs profondeur dans le detecteur")
    ax_E.legend(fontsize=8)
    ax_E.grid(alpha=0.3)

    ax_S.set_xlabel("Profondeur de detecteur (µm)")
    ax_S.set_ylabel("Pouvoir d'arret (MeV/µm)")
    ax_S.set_title("Pouvoir d'arret vs profondeur dans le detecteur")
    ax_S.legend(fontsize=8)
    ax_S.grid(alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


# =========================================================================
# 5. SCRIPT PRINCIPAL
# =========================================================================

def main():
    # ---- 1) courbes dans l'eau, pour chaque ion --------------------------
    print("=== Calcul des courbes dans l'eau (module SR + CSDA) ===")
    water_curves = []
    for ion_def in IONS:
        print(f"  -> ion {ion_def['identifier']} ({ion_def['energy']/1e6:.1f} MeV)")
        water_curves.append(water_bragg_curve(ion_def))

    print(water_curves[0])
    plot_water_curves(water_curves)#, save_path=DATA_SAVING_PATH / "water_curves.png")

    # ---- 2) courbes dans chaque detecteur, pour chaque ion ---------------
#    print("=== Calcul des courbes dans les detecteurs (TRIM + boucle adaptative) ===")
#    detector_results = []
#    for ion_def in IONS:
#        for detector_name in DETECTORS:
#            print(f"  -> ion {ion_def['identifier']} / detecteur {detector_name}")
#            res = run_detector_adaptive(ion_def, detector_name)
#            print(f"     taille de cellule finale : {res['cell_um']:.3f} µm "
#                  f"(epaisseur totale simulee : {res['total_thickness_um']:.3f} µm), "
#                  f"arret precoce : {res['stopped_early']}")
#            detector_results.append(res)
#
#    plot_detector_curves(detector_results)#, save_path=DATA_SAVING_PATH / "detector_curves.png")
#
#    plt.show()


if __name__ == "__main__":
    main()
