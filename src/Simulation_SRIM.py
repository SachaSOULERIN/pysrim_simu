import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from srim import TRIM, Target, Layer, Ion, Element

DATA_SAVING_PATH = Path() #to be added

# defining the ion beams: H, C and O
# loop on initial energy here ?

ions = [
    {'identifier': 'H', 'energy':58.4e6} # = 58,4 MeV
    {'identifier': 'C', 'energy':11.4e8} # = 1.14 GeV
    {'identifier': 'O', 'energy':1.52e9} # = 1.52 GeV
]

# Defining the layers needed (water + PMMA + scintillator)
# loop on layer width ?

# chemical formula: H2O
h2o_layer = Layer(
    {
        'H': {
            'stoich': 2.0,
            'E_d': ??, # Displacement energy to be filled
            'lattice': 0.0,
            'surface': 3.0
        },
        'O': {
            'stoich': 1.0,
            'E_d': 25.0, # Displacement energy to be filled
            'lattice': 0.0,
            'surface': 3.0
        }
    },
    density=1.0,
    width=10000.0 ### in Angstrom, also TO BE CHANGED !
)

# chemical formula: (C5O2H8)n
pmma_layer = Layer(
    {
        'H': {
            'stoich': 8.0,
            'E_d': ??, # Displacement energy to be filled
            'lattice': 0.0,
            'surface': 3.0
        },
        'O': {
            'stoich': 2.0,
            'E_d': 25.0, # Displacement energy to be filled
            'lattice': 0.0,
            'surface': 3.0
        },
        'C': {
            'stoich': 5.0,
            'E_d': 20.0, # Displacement energy to be filled
            'lattice': 0.0,
            'surface': 3.0
        }
    },
    density=4.0, ### TO BE CHANGED
    width=10000.0 ### in Angstrom, also TO BE CHANGED !
)
rho_pmma = 1.18 # g/cm3, TO BE CHECKED !
m_pmma_r = 1.02 # g, (in my ISD_fab files)
m_pmma_g = 1.0131 # g, (in my ISD_fab files)
m_pmma_b = 1.0086 # g, (in my ISD_fab files)

# to be upgraded once the ICP has been done !

# ZnCdS:Ag
Red_layer = LAYER(
    {
            'Zn': {
                'stoich': 8.0, ##
                'E_d': ??, # Displacement energy to be filled: DEFAULT VALUE TO BE CHANGED
                'lattice': 0.0,
                'surface': 3.0
            },
            'Cd': {
                'stoich': 2.0, ##
                'E_d': ??, # Displacement energy to be filled: DEFAULT VALUE TO BE CHANGED
                'lattice': 0.0,
                'surface': 3.0
            },
            'S': {
                'stoich': 5.0, ##
                'E_d': ??, # Displacement energy to be filled: DEFAULT VALUE TO BE CHANGED
                'lattice': 0.0,
                'surface': 3.0
            },
            'Ag': {
                'stoich': 0.01, ##
                'E_d': ??, # Displacement energy to be filled: DEFAULT VALUE TO BE CHANGED
                'lattice': 0.0,
                'surface': 3.0
            }
        },
    density=4.18, ### TO BE CHANGED
    width=10000.0 ### in Angstrom, also TO BE CHANGED !
)
rho_red = 4.18 # g/cm3, TO BE CHECKED !
m_red = 2.03 # g, (in my ISD_fab files)

# Y3_Al5_O12_:_Ce
# CAS nbr : 12005-21-8 (can be found on commonchemistry.cas.org)
Green_layer = LAYER(
    {
            'Y': {
                'stoich': 3.0, ##
                'E_d': ??, # Displacement energy to be filled: DEFAULT VALUE TO BE CHANGED
                'lattice': 0.0,
                'surface': 3.0
            },
            'Al': {
                'stoich': 5.0, ##
                'E_d': ??, # Displacement energy to be filled: DEFAULT VALUE TO BE CHANGED
                'lattice': 0.0,
                'surface': 3.0
            },
            'O': {
                'stoich': 12.0, ##
                'E_d': 25.0, # Displacement energy to be filled
                'lattice': 0.0,
                'surface': 3.0
            },
            'Ce': {
                'stoich': 0.0, ##
                'E_d': ??, # Displacement energy to be filled: DEFAULT VALUE TO BE CHANGED
                'lattice': 0.0,
                'surface': 3.0
            }
        },
    density=3.848, ### g/cm3 check (https://commonchemistry.cas.org/detail?cas_rn=12005-21-9&search=12005-21-9
    #or also: Internationl Journal of Applied Glass Science, 2028, 9(3), 307-318)
    width=10000.0 ### in Angstrom, also TO BE CHANGED !
)
rho_green = 3.848 # g/cm3 (commonchemistry.cas.org)
density_green = 4.15 #g/mL (phosphor technology)
m_green = 2.0134 # g, (in my ISD_fab files)

# Lu2_Si_O5_:_Ce
# CAS nbr : 12168-86-4 (can be found on commonchemistry.cas.org)
Blue_layer = LAYER(
    {
            'Lu': {
                'stoich': 2.0, ##
                'E_d': ??, # Displacement energy to be filled: DEFAULT VALUE TO BE CHANGED
                'lattice': 0.0,
                'surface': 3.0
            },
            'Si': {
                'stoich': 1.0, ##
                'E_d': 35.0, # Displacement energy to be filled
                'lattice': 0.0,
                'surface': 3.0
            },
            'O': {
                'stoich': 5.0, ##
                'E_d': 25.0, # Displacement energy to be filled
                'lattice': 0.0,
                'surface': 3.0
            },
            'Ce': {
                'stoich': 0.0, ##
                'E_d': ??, # Displacement energy to be filled: DEFAULT VALUE TO BE CHANGED
                'lattice': 0.0,
                'surface': 3.0
            }
        },
    density=4.15, ### TO BE CHANGED !
    ### 7,4g/mL from ZBK58N .pdf file
    ### from its molecular mass of 458,016 uma (https://www.chemsrc.com/cas/12168-86-4_852916.html)
    width=10000.0 ### in Angstrom, also TO BE CHANGED !
)
rho_blue = 4.18 # g/cm3, TO BE CHECKED !
m_blue = 2.2885 # g, (in my ISD_fab files)

