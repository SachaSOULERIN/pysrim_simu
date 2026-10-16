#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Perte d'energie d'ions (C, O) dans un empilement scintillateur + PMMA.

Ce script automatise des calculs SRIM (via le paquet ``pysrim``) pour
determiner l'energie perdue par un ion (carbone ou oxygene) traversant
un empilement de deux couches : un scintillateur plastique et du PMMA,
dont le ratio d'epaisseur est fixe (pour conserver le ratio de masse
du detecteur experimental reel).

Methode physique
-----------------
Plutot que de s'appuyer sur une simulation Monte-Carlo TRIM complete et
sur le fichier ``TRANSMIT.txt`` (dont le parsing n'est PAS implemente
dans pysrim : ``srim.output.Transmit`` est un stub "TODO"), ce script
utilise le module SR de SRIM (Stopping and Range) pour obtenir les
tables de pouvoir d'arret total dE/dx(E) = S_electronique + S_nucleaire
de chaque materiau, puis integre numeriquement la perte d'energie de
l'ion le long de sa trajectoire (approximation CSDA : Continuous
Slowing Down Approximation). C'est l'approche standard pour calculer
une perte d'energie moyenne a travers un empilement de couches minces,
et elle est nettement plus robuste a automatiser que le parsing de
TRANSMIT.txt.

Limite assumee : cette approche donne la perte d'energie *moyenne*
(valeur "CSDA"), sans straggling en energie (dispersion statistique
ion a ion). Si vous avez besoin de la distribution complete (straggling,
angle de sortie, etc.), il faudra passer par un TRIM complet
(``calculation=2``, option ``transmit=True``) et ecrire vous-meme le
parseur de ``TRANSMIT.txt`` -- une base est proposee en commentaire en
fin de fichier.

Prerequis
---------
- SRIM/TRIM installe localement (http://www.srim.org), avec ``SRIM.exe``
  a la racine du dossier d'installation.
- Sous Linux/macOS : ``wine`` installe (pysrim l'utilise automatiquement).
- Le paquet pysrim : ``pip install pysrim``

Utilisation
-----------
    python srim_stack_energy_loss.py

Le script est interactif : il demande le dossier SRIM, l'ion, la ou les
energies initiales, et les epaisseurs de reference des deux couches.
"""

from __future__ import annotations

import csv
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from srim import SR, Ion, Layer
    from srim.output import SRResults
except ImportError as exc:  # pragma: no cover - dependance externe
    raise SystemExit(
        "Le paquet 'pysrim' est requis mais introuvable.\n"
        "Installez-le avec :\n"
        "    pip install pysrim\n"
        "Documentation : https://pypi.org/project/pysrim/"
    ) from exc


LOGGER = logging.getLogger("srim_stack")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# ---------------------------------------------------------------------------
# 1. Definition physique des materiaux
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MaterialDefinition:
    """Composition atomique et densite d'un materiau pour SRIM.

    ``composition`` suit le format attendu par ``srim.Layer`` :
    ``{symbole: {'stoich': ..., 'E_d': ..., 'lattice': ..., 'surface': ...}}``
    """

    name: str
    composition: Dict[str, Dict[str, float]]
    density_g_cm3: float


# Scintillateur plastique generique (type PVT, cf. EJ-200 / EJ-212).
# ADAPTEZ 'stoich' et les densites a la formulation exacte de votre
# scintillateur experimental.
SCINTILLATOR = MaterialDefinition(
    name="Scintillateur plastique (PVT)",
    composition={
        "H": {"stoich": 1.10, "E_d": 10.0, "lattice": 0.0, "surface": 2.0},
        "C": {"stoich": 1.00, "E_d": 28.0, "lattice": 0.0, "surface": 7.4},
    },
    density_g_cm3=1.023,
)

# PMMA : monomere (C5O2H8)n
PMMA = MaterialDefinition(
    name="PMMA",
    composition={
        "H": {"stoich": 8.0, "E_d": 10.0, "lattice": 0.0, "surface": 2.0},
        "C": {"stoich": 5.0, "E_d": 28.0, "lattice": 0.0, "surface": 7.4},
        "O": {"stoich": 2.0, "E_d": 28.0, "lattice": 0.0, "surface": 2.0},
    },
    density_g_cm3=1.18,
)

VALID_IONS: Dict[str, str] = {"C": "Carbone", "O": "Oxygene"}

# output_type=2 du module SR : keV / micron (pratique car les epaisseurs
# sont manipulees en micrometres dans ce script).
SR_OUTPUT_TYPE_KEV_PER_UM = 2


# ---------------------------------------------------------------------------
# 2. Tables de pouvoir d'arret via le module SR de SRIM
# ---------------------------------------------------------------------------

def _locate_sr_output(srim_directory: Path) -> Path:
    """Trouve le dossier contenant SR_OUTPUT.txt apres un calcul SR.

    Selon la version de SRIM/pysrim, le fichier peut se trouver soit a
    la racine du dossier SRIM, soit dans le sous-dossier 'SR Module'.
    """
    candidates = [
        srim_directory / "SR Module",
        srim_directory,
    ]
    for candidate in candidates:
        if (candidate / "SR_OUTPUT.txt").exists():
            return candidate
    raise FileNotFoundError(
        "SR_OUTPUT.txt introuvable apres le calcul SR. Verifiez votre "
        f"installation SRIM dans {srim_directory}."
    )


def compute_stopping_power_table(
    material: MaterialDefinition,
    ion_symbol: str,
    energy_max_ev: float,
    srim_directory: Path,
) -> np.ndarray:
    """Calcule la table de pouvoir d'arret total dE/dx via le module SR.

    Retourne un tableau numpy (N, 2) trie par energie croissante :
    colonne 0 = energie [keV], colonne 1 = dE/dx total [keV/um].
    """
    ion = Ion(ion_symbol, energy=energy_max_ev)
    # La largeur de la couche n'intervient pas dans un calcul SR (table
    # de pouvoir d'arret intrinseque au materiau) ; valeur factice.
    layer = Layer(
        material.composition,
        density=material.density_g_cm3,
        width=1.0e6,
    )
    sr_calc = SR(layer, ion, output_type=SR_OUTPUT_TYPE_KEV_PER_UM)
    sr_calc.run(str(srim_directory))

    output_dir = _locate_sr_output(srim_directory)
    results = SRResults(str(output_dir))
    data = np.asarray(results.data, dtype=float)

    energy_kev = data[:, 0]
    stopping_total_kev_per_um = data[:, 1] + data[:, 2]

    order = np.argsort(energy_kev)
    return np.column_stack(
        (energy_kev[order], stopping_total_kev_per_um[order])
    )


# ---------------------------------------------------------------------------
# 3. Integration CSDA de la perte d'energie couche par couche
# ---------------------------------------------------------------------------

def stopping_power_kev_per_um(energy_kev: float, table: np.ndarray) -> float:
    """Interpole (lineairement) le pouvoir d'arret total a une energie."""
    e_min, e_max = table[0, 0], table[-1, 0]
    e_clamped = min(max(energy_kev, e_min), e_max)
    return float(np.interp(e_clamped, table[:, 0], table[:, 1]))


def propagate_through_layer(
    energy_in_kev: float,
    thickness_um: float,
    table: np.ndarray,
    step_um: float = 0.05,
) -> Tuple[float, bool, float]:
    """Propage l'ion a travers une couche (integration d'Euler explicite).

    Returns
    -------
    energy_out_kev : energie de sortie [keV] (0.0 si l'ion s'arrete).
    transmitted : True si l'ion traverse completement la couche.
    stop_or_final_depth_um : profondeur d'arret, ou epaisseur totale si
        transmission complete.
    """
    depth_um = 0.0
    energy_kev = energy_in_kev
    while depth_um < thickness_um:
        dx = min(step_um, thickness_um - depth_um)
        dedx = stopping_power_kev_per_um(energy_kev, table)
        d_energy = dedx * dx
        if d_energy >= energy_kev:
            depth_stop = depth_um + (energy_kev / dedx if dedx > 0 else 0.0)
            return 0.0, False, depth_stop
        energy_kev -= d_energy
        depth_um += dx
    return energy_kev, True, thickness_um


@dataclass
class StackResult:
    """Resultat du calcul pour une energie initiale donnee."""

    ion_symbol: str
    initial_energy_mev: float
    scale: float
    thickness_scint_um: float
    thickness_pmma_um: float
    energy_after_scint_kev: float
    energy_after_pmma_kev: float
    transmitted: bool
    stop_depth_um: Optional[float]


def find_transmitting_thickness(
    ion_symbol: str,
    initial_energy_mev: float,
    base_thickness_scint_um: float,
    base_thickness_pmma_um: float,
    table_scint: np.ndarray,
    table_pmma: np.ndarray,
    reduction_factor: float = 0.5,
    min_scale: float = 1.0e-4,
) -> StackResult:
    """Reduit iterativement l'epaisseur (a ratio constant) jusqu'a ce
    que l'ion traverse completement l'empilement scintillateur + PMMA.

    Le ratio ``base_thickness_scint_um / base_thickness_pmma_um`` est
    conserve a chaque iteration (facteur d'echelle global ``scale``),
    afin de garder le ratio de masse du detecteur experimental.
    """
    scale = 1.0
    energy_in_kev = initial_energy_mev * 1000.0
    last_stop_depth = None

    while scale >= min_scale:
        t_scint = base_thickness_scint_um * scale
        t_pmma = base_thickness_pmma_um * scale

        e_after_scint, ok_scint, depth = propagate_through_layer(
            energy_in_kev, t_scint, table_scint
        )
        if not ok_scint:
            LOGGER.info(
                "E0=%.3f MeV, scale=%.4g : ion arrete dans le "
                "scintillateur (profondeur %.3f um). Reduction de "
                "l'epaisseur.",
                initial_energy_mev, scale, depth,
            )
            last_stop_depth = depth
            scale *= reduction_factor
            continue

        e_after_pmma, ok_pmma, depth_pmma = propagate_through_layer(
            e_after_scint, t_pmma, table_pmma
        )
        if not ok_pmma:
            LOGGER.info(
                "E0=%.3f MeV, scale=%.4g : ion arrete dans le PMMA "
                "(profondeur %.3f um). Reduction de l'epaisseur.",
                initial_energy_mev, scale, depth_pmma,
            )
            last_stop_depth = depth_pmma
            scale *= reduction_factor
            continue

        return StackResult(
            ion_symbol=ion_symbol,
            initial_energy_mev=initial_energy_mev,
            scale=scale,
            thickness_scint_um=t_scint,
            thickness_pmma_um=t_pmma,
            energy_after_scint_kev=e_after_scint,
            energy_after_pmma_kev=e_after_pmma,
            transmitted=True,
            stop_depth_um=None,
        )

    LOGGER.warning(
        "E0=%.3f MeV : aucune epaisseur testee (jusqu'a scale=%.1e) "
        "ne permet la transmission complete.",
        initial_energy_mev, min_scale,
    )
    return StackResult(
        ion_symbol=ion_symbol,
        initial_energy_mev=initial_energy_mev,
        scale=scale,
        thickness_scint_um=base_thickness_scint_um * scale,
        thickness_pmma_um=base_thickness_pmma_um * scale,
        energy_after_scint_kev=0.0,
        energy_after_pmma_kev=0.0,
        transmitted=False,
        stop_depth_um=last_stop_depth,
    )


# ---------------------------------------------------------------------------
# 4. Interaction utilisateur
# ---------------------------------------------------------------------------

def ask_float(prompt: str, default: Optional[float] = None) -> float:
    """Demande une valeur flottante, avec valeur par defaut optionnelle."""
    while True:
        raw = input(prompt).strip().replace(",", ".")
        if not raw and default is not None:
            return default
        try:
            return float(raw)
        except ValueError:
            print("Valeur numerique invalide, reessayez.")


def ask_energies(prompt: str) -> List[float]:
    """Demande une liste d'energies initiales (MeV), separees par virgules."""
    while True:
        raw = input(prompt).strip()
        if raw:
            try:
                values = [
                    float(chunk.strip().replace(",", "."))
                    for chunk in raw.split(",")
                    if chunk.strip()
                ]
                if values:
                    return values
            except ValueError:
                pass
        print("Format invalide. Exemple : 50, 80, 120")


def ask_ion() -> str:
    """Demande l'element de l'ion incident (Carbone ou Oxygene)."""
    while True:
        raw = input(
            "Ion incident - Carbone (C) ou Oxygene (O) ? [C/O] : "
        ).strip().upper()
        if raw in VALID_IONS:
            return raw
        print("Reponse invalide, entrez 'C' ou 'O'.")


# ---------------------------------------------------------------------------
# 5. Programme principal
# ---------------------------------------------------------------------------

def main() -> None:
    """Point d'entree interactif du script."""
    print(
        "=== Perte d'energie SRIM : empilement scintillateur + PMMA ===\n"
    )

    srim_directory = Path(
        input(
            "Chemin du dossier d'installation de SRIM "
            "(contenant SRIM.exe) : "
        ).strip()
    ).expanduser()
    if not srim_directory.exists():
        LOGGER.error("Le dossier %s est introuvable.", srim_directory)
        sys.exit(1)

    ion_symbol = ask_ion()

    energies_mev = ask_energies(
        "Energie(s) initiale(s) de l'ion en MeV, separees par des "
        "virgules (ex: 50, 80, 120) : "
    )

    t_scint0 = ask_float(
        "Epaisseur de reference du scintillateur en um "
        "[defaut 50] : ",
        default=50.0,
    )
    t_pmma0 = ask_float(
        "Epaisseur de reference du PMMA en um [defaut 50] : ",
        default=50.0,
    )

    mass_ratio = (
        SCINTILLATOR.density_g_cm3 * t_scint0
    ) / (PMMA.density_g_cm3 * t_pmma0)
    print(
        f"\nRatio d'epaisseur scintillateur/PMMA : "
        f"{t_scint0 / t_pmma0:.4f}"
        f"\nRatio de masse surfacique equivalent : {mass_ratio:.4f}"
        "\n(a comparer au ratio de masse reel de votre detecteur)\n"
    )

    energy_max_ev = max(energies_mev) * 1.0e6 * 1.2  # marge de securite
    LOGGER.info("Calcul des tables de pouvoir d'arret (module SR)...")
    table_scint = compute_stopping_power_table(
        SCINTILLATOR, ion_symbol, energy_max_ev, srim_directory
    )
    table_pmma = compute_stopping_power_table(
        PMMA, ion_symbol, energy_max_ev, srim_directory
    )
    LOGGER.info(
        "Tables recuperees (%d points scintillateur, %d points PMMA).",
        len(table_scint), len(table_pmma),
    )

    results: List[StackResult] = []
    for e0 in energies_mev:
        result = find_transmitting_thickness(
            ion_symbol, e0, t_scint0, t_pmma0, table_scint, table_pmma
        )
        results.append(result)

    print("\n=== Resultats ===")
    header = (
        f"{'E0 (MeV)':>10} | {'scale':>8} | {'t_scint (um)':>13} | "
        f"{'t_PMMA (um)':>12} | {'E residuelle (MeV)':>19} | "
        f"{'E perdue (MeV)':>15} | transmis"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        e_residual_mev = result.energy_after_pmma_kev / 1000.0
        e_lost_mev = result.initial_energy_mev - e_residual_mev
        print(
            f"{result.initial_energy_mev:10.3f} | {result.scale:8.4g} | "
            f"{result.thickness_scint_um:13.3f} | "
            f"{result.thickness_pmma_um:12.3f} | "
            f"{e_residual_mev:19.4f} | {e_lost_mev:15.4f} | "
            f"{'oui' if result.transmitted else 'non'}"
        )

    output_csv = Path("resultats_perte_energie.csv")
    with output_csv.open("w", newline="", encoding="utf-8") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(
            [
                "ion", "E0_MeV", "scale", "t_scint_um", "t_pmma_um",
                "E_apres_scint_MeV", "E_residuelle_MeV", "E_perdue_MeV",
                "transmis",
            ]
        )
        for result in results:
            e_residual_mev = result.energy_after_pmma_kev / 1000.0
            writer.writerow(
                [
                    VALID_IONS[result.ion_symbol],
                    f"{result.initial_energy_mev:.4f}",
                    f"{result.scale:.6g}",
                    f"{result.thickness_scint_um:.4f}",
                    f"{result.thickness_pmma_um:.4f}",
                    f"{result.energy_after_scint_kev / 1000.0:.4f}",
                    f"{e_residual_mev:.4f}",
                    f"{result.initial_energy_mev - e_residual_mev:.4f}",
                    "oui" if result.transmitted else "non",
                ]
            )
    print(f"\nResultats enregistres dans : {output_csv.resolve()}")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Extension possible : TRIM complet + parsing manuel de TRANSMIT.txt
# ---------------------------------------------------------------------------
# Si vous avez besoin du straggling en energie (dispersion ion a ion) et
# non de la seule perte moyenne CSDA, il faut lancer un TRIM.TRIM complet
# avec l'option transmit=True :
#
#   from srim import TRIM, Target
#   target = Target([layer_scint, layer_pmma])
#   trim_calc = TRIM(
#       target, ion, calculation=2, number_ions=1000, transmit=True
#   )
#   trim_calc.run(str(srim_directory))
#
# Le fichier TRANSMIT.txt sera alors present dans srim_directory, mais
# devra etre parse manuellement (colonnes : numero d'ion, atome,
# energie [eV], position Y, Z, cosinus directeurs), car
# srim.output.Transmit n'est pas implemente dans pysrim au moment de
# l'ecriture de ce script.
