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


DEFAULT_E_D = 25.0 # default value for displacement energy (eV) in pysrim

SRIM_DIRECTORY = Path(r'C:\Program Files\SRIM-2013')


### Water part : elements and function definitions


def elt(stoich: float, E_d: float = DEFAULT_E_D, lattice: float = 0.0,
        surface: float = 3.0) -> dict:
    """shortcut to define an element in a pysrim layer.
    """
    element = {
        "stoich": stoich,
        "E_d": E_d,
        "lattice": lattice,
        "surface": surface
    }
    return element

WATER_DENSITY = 1.0 # g/cm3
WATER_COMPOSITION = { #energy_displacement to be checked
    'H': elt(2.0, E_d=10.0),
    'O':elt(1, E_d=25.0)
}


def _to_float_locale(token: str) -> float:
    """Convert a number possibly in European format (coma as decimal separator) to float.
    This is what fails in the version I have (ValueError on '999,999').
    """

    token = token.strip()

    return float(token.replace(",", "."))


def read_sr_output(directory) -> dict:
    """Replace the native pysrim parser for SR_OUTPUT.txt, which fails
    when SRIM writes numbers with a comma as decimal separator (French local)

    Returns
    -------
    dict with numpy arrays:
        energy_keV, Se, Sn, range_um, long_straggle_um, lat_straggle_um
    """
    # reading the file: SR_OUTPUT.txt
    path = Path(directory) / "SR_OUTPUT.txt"
    with open(path, "r", encoding="latin1") as f:
        lines = f.readlines()

    # dictionnary to convert the units into keV and um
    energy_conv = {"eV": 1e-3, "keV": 1.0, "MeV": 1e3, "GeV": 1e6}  # -> keV
    length_conv = {"A": 1e-4, "um": 1.0, "mm": 1e3, "cm": 1e4}      # -> um

    # finding the lines delimiting the data values in the file
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

    # data lines start typically 3 lines after the header ("Energy" / units / "-----")
    # and end a few lines before "Multiply Stopping..." (a line of dashes separates the two)
    data_lines = lines[header_idx + 3: footer_idx]

    # defining the output dictionnary
    out = {"energy_keV": [], "Se": [], "Sn": [], "range_um": [],
           "long_straggle_um": [], "lat_straggle_um": []}

    # getting the values line by line, converting the units and storing them in
    # the output dictionnary
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

    # convert the list into arrays
    for k in out:
        out[k] = np.array(out[k], dtype=float)

    # Energy sorted
    # Remove non-monotone points (saturation artefacts)
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
    """Run the SR module of pysrim without using its own parser sr.run()
    which fails on numbers with a comma as decimal separator
    It reproduce manually the first two steps of sr.run() (writing the input + running the exe),
    then read the output with read_sr_output() instead of the native parser.
    It never calls SRResults, that is the part that fails in the native sr.run().

    Returns
    -------
    dict with numpy arrays:
        energy_keV, Se, Sn, range_um, long_straggle_um, lat_straggle_um
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
                       output_type: int = 3, n_points: int = 1000) -> dict:
    """Compute the energy in function of the depth, the stopping power in
    function of the depth, and the CSDA range in water for a given ion.

    Returns
    -------
    dict
        depth_um, energy_MeV, Se, Sn, Se_plus_Sn
        (units of Se/Sn/Se_plus_Sn: MeV/mm in this case)
    """
    ion = Ion(ion_def["identifier"], energy=ion_def["energy"])

    water_layer = Layer(WATER_COMPOSITION, density=WATER_DENSITY, width=1.0)
    # width does not have an impact for an SR calculation (no TRIM here)

    sr = SR(water_layer, ion, output_type=output_type)
    table = run_sr_safe(sr, srim_directory)

    energy_keV = table["energy_keV"]
    range_um = table["range_um"]
    Se, Sn = table["Se"], table["Sn"]

    E0_keV = ion_def["energy"] / 1.0e3  # eV -> keV
    if E0_keV > energy_keV.max():
        warnings.warn(
            f"Initial energy ({E0_keV/1e3:.2f} MeV) beyond the SR table "
            f"(max {energy_keV.max()/1e3:.2f} MeV): extrapolation, to check!"
        )

    # CSDA range of the ion at its initial energy (interpolation of Range(E))
    R0_um = np.interp(E0_keV, energy_keV, range_um)

    depth_um = np.linspace(0.0, R0_um, n_points)
    #depth_um = np.logspace(0.0, np.log10(R0_um), n_points)
    # E(depth) = inverse of Range(E) : range restant = R0 - depth
    remaining_range = np.clip(R0_um - depth_um, range_um.min(), None)
    # Range(E) is increasing with E -> we can invert by interpolation
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
    """Plot the curves of the energy of the ions, and their stopping power, along the depth of water.

    Args:
    -----
        curves (List[dict]): list of the different curves, each one being a dict from
    the water_bragg_curve() function
        save_path (Optional[Path], optional): The path where to save the plot on the computer. If None, the plot is not saved.
    Defaults to None.

    Returns:
    --------
        _type_: _description_
    """
    fig, (ax_E, ax_S, ax_SvE) = plt.subplots(1, 3, figsize=(14, 5))

    for c in curves:
        lbl = "{}: $E_0$ = {} MeV".format(c["ion"], c["energy_MeV"][0])
        ax_E.plot(c["depth_um"], c["energy_MeV"], label=lbl)
        ax_S.plot(c["depth_um"], c["Se_plus_Sn"]/c["Se_plus_Sn"][0], label=lbl)
        ax_SvE.plot(c["energy_MeV"], c["Se_plus_Sn"]/c["Se_plus_Sn"][0], label=lbl)

    ax_E.set_xlabel("Water depth (µm)")
    ax_E.set_ylabel("Ion energy (MeV)")
    ax_E.set_title("Ion energy along its travel into water")
    #ax_E.set_yscale("log")
    ax_E.legend()
    ax_E.grid(alpha=0.3)

    ax_S.set_xlabel("Water depth (µm)")
    ax_S.set_ylabel("Normalized stopping power (Se + Sn)")
    ax_S.set_title("Normalized total stopping powers into water")
    ax_S.legend()
    ax_S.grid(alpha=0.3)

    ax_SvE.set_xlabel("Ion energy (MeV)")
    ax_SvE.set_ylabel("Normalized stopping power (Se + Sn)")
    ax_SvE.set_title("Normalized total stopping powers into water")
    ax_SvE.set_xscale("log")
    ax_SvE.legend()
    ax_SvE.grid(alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


### Scintillator part : elements and function definitions


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