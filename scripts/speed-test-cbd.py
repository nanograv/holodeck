"""
"""

import cProfile, pstats, io
from pstats import SortKey

import holodeck as holo
import numpy as np
import holodeck.accretion
import holodeck.discrete_cyutils
from holodeck.constants import YR

holo.log.setLevel(holo.log.INFO)

ALLOW_SOFTENING = False
ECCEN_INIT = 0.5
F_EDD = 1.0

TEST_NUM_BINS = 10


pr = cProfile.Profile()
pr.enable()

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

pop = holo.population.Pop_Illustris(select=TEST_NUM_BINS)
size = pop.size
eccen = np.ones(size) * ECCEN_INIT
pop = holo.population.Pop_Illustris(eccen=eccen, select=TEST_NUM_BINS)

hards = [
    holo.hardening.Hard_GW,
    holo.hardening.CBD_Torques(allow_softening=ALLOW_SOFTENING),
    holo.hardening.Sesana_Scattering(),
    holo.hardening.Dynamical_Friction_NFW(attenuate=True),
]

acc = holo.accretion.Accretion(
    accmod='Siwek22', f_edd=F_EDD, subpc=True, evol_mass=True, edd_lim=1.0,
)

evo = holo.evolution.Evolution(pop, hards, debug=True, acc=acc)
evo.evolve(break_after=TEST_NUM_BINS)


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------



pr.disable()

with open('cprofile_cumtime.txt', 'w') as fout:
    ps = pstats.Stats(pr, stream=fout)
    ps.sort_stats('cumtime')
    ps.print_stats()

with open('cprofile_tottime.txt', 'w') as fout:
    ps = pstats.Stats(pr, stream=fout)
    ps.sort_stats('tottime')
    ps.print_stats()
