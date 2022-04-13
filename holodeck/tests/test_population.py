"""Tests for `population.py` holodeck submodule.


"""

import numpy as np
import scipy as sp
import scipy.stats

import holodeck as holo
import holodeck.population
import holodeck.relations
from holodeck.constants import MSOL, PC


# ==============================================================================
# ====    Illustris Population    ====
# ==============================================================================


def test_pop_illustris_basic():
    print("test_pop_illustris_basic()")
    pop = holo.Pop_Illustris()

    keys = ['mbulge', 'vdisp', 'mass', 'scafa', 'sepa', 'redz']
    ranges = [[1e7, 1e13], [10, 600], [1e5, 3e10], [0.0, 1.0], [1e3, 3e5], [0.0, 10.0]]
    units = [MSOL, 1e5, MSOL, 1.0, PC, 1.0]

    for kk, rr, uu in zip(keys, ranges, units):
        vals = getattr(pop, kk) / uu
        extr = np.array(rr)
        print(f"key: {kk:10s}, {holo.utils.stats(vals)}, {extr}")
        assert np.all((extr[0] < vals) & (vals < extr[1])), f"Values for {kk} are not within bounds!"
        assert pop.size == np.shape(vals)[0], f"{kk} size ({np.shape(vals)}) differs from pop.size ({pop.size})!"

    pop._check()
    return


# ==============================================================================
# ====    Population Modifiers    ====
# ==============================================================================

# ---- PM_Mass_Reset

def test_mass_reset():
    print("test_mass_reset()")
    pop = holo.Pop_Illustris()
    mmbulge_relation = holo.relations.MMBulge_MM13()
    mod_mm13 = holo.population.PM_Mass_Reset(mmbulge_relation, scatter=False)

    mass_bef = pop.mass
    host = {'mbulge':pop.mbulge}

    pop.modify(mod_mm13)
    mass_aft = pop.mass

    assert not np.all(mass_bef == mass_aft), "Masses are unchanged after modification!"
    assert np.all((1e4 < mass_aft/MSOL) & (mass_aft/MSOL < 1e11)), "Modified masses outside of expectations!"
    check = mmbulge_relation.mbh_from_host(host, False)
    assert np.all(check == mass_aft), "Modified masses do not match mmbulge_relation values!"

    SCATTER = 0.1
    TOL_STD = 1.5 * sp.stats.norm.ppf(1.0 - 1.0 / pop.mass.size)
    print(f"TOL={TOL_STD}")
    mmbulge_relation = holo.relations.MMBulge_MM13(scatter_dex=SCATTER)
    mod_mm13 = holo.population.PM_Mass_Reset(mmbulge_relation, scatter=True)
    pop.modify(mod_mm13)
    mass_scatter = pop.mass
    aa = np.log10(mass_aft/MSOL)
    bb = np.log10(mass_scatter/MSOL)
    assert not np.all(mass_scatter == mass_aft), "Masses with scatter match without!"
    diff = (bb - aa)
    print(f"SCATTER={SCATTER:.4f}, TOL={TOL_STD:.3f}, diff={holo.utils.stats(diff)}")
    mean_diff = np.mean(diff)
    assert mean_diff < SCATTER, f"Mean difference ({np.mean(diff)}) exceeds SCATTER ({SCATTER})!"
    print(f"diff/SCATTER={holo.utils.stats(diff/SCATTER)}")
    assert np.all(np.fabs(diff/SCATTER) < TOL_STD), f"Difference exceeds tolerance ({TOL_STD})!"

    return
