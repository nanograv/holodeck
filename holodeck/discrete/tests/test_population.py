"""Tests for `population.py` holodeck submodule.
"""

import numpy as np
import scipy as sp
import scipy.stats

import pytest

import holodeck as holo
from holodeck.discrete import population
import holodeck.host_relations
from holodeck.constants import MSOL, PC


# ==============================================================================
# ====    Illustris Population    ====
# ==============================================================================


def test_pop_illustris_basic():
    """Basic Pop_Illustris tests.

    Make sure the expected attributes exist, have the expected shapes, and have the appropriate
    range of values.

    """
    print("test_pop_illustris_basic()")
    pop = population.Pop_Illustris()

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

    """`Pop_Illustris` should raise an error if passed a non-existent file
    """
    with pytest.raises(FileNotFoundError):
        pop = population.Pop_Illustris(fname='does_not_exist')

    return


def test_valid_population_subclass():
    """Ensure that subclasses of `_Population_Discrete` succeed and fail in the correct places.

    """

    # ---- Initializing a subclass that does *not* override `_init()` should fail
    class Bad_1(population._Population_Discrete):
        pass

    with pytest.raises(TypeError):
        Bad_1()

    # ---- Initializing a subclass that *does* override `_init()` and sets attributes should succeed
    class Good_1(population._Population_Discrete):
        def _init(self):
            self.mass = np.zeros((10, 2))
            self.sepa = np.zeros(10)
            self.scafa = np.zeros(10)
            return

    Good_1()

    # ---- Initializing a subclass that overrides `_init()`, but does not set attributers should fail
    # because `sepa` is not set, `_size` will be `None` which will raise a `ValueError` in `_check()`
    class Bad_2(population._Population_Discrete):
        def _init(self):
            return

    with pytest.raises(ValueError):
        Bad_2()

    class Bad_3(population._Population_Discrete):
        def _init(self):
            self.mass = np.zeros((10, 2))
            self.scafa = np.zeros(10)
            return

    with pytest.raises(ValueError):
        Bad_3()

    return


# ==============================================================================
# ====    Population Modifiers    ====
# ==============================================================================


def test_valid_pop_mod_init():
    """Make sure the `_Population_Modifier` subclasses succeed and fail appropriately in subclasses.
    """

    """PM subclasses should raise errors when not overriding `modify`
    """
    class PM_Bad_1(population._Population_Modifier):
        pass

    with pytest.raises(TypeError):
        PM_Bad_1()

    """PM subclasses should raise errors when not overriding `modify`
    """
    class PM_Good_1(population._Population_Modifier):
        def modify(self):   # nocov
            pass

    PM_Good_1()

    return


# ====    PM_Eccentricity    ====


def test_eccentricity_illustris_basics():

    # ---- No Arguments

    # Apply in constructor
    pm_ecc = population.PM_Eccentricity()
    pop = population.Pop_Illustris(mods=pm_ecc)
    assert pop.eccen.size == pop.size
    assert np.all((pop.eccen >= 0.0) & (pop.eccen <= 1.0))

    # Apply after initialization
    pop = population.Pop_Illustris()
    pop.modify(pm_ecc)
    assert pop.eccen.size == pop.size
    assert np.all((pop.eccen >= 0.0) & (pop.eccen <= 1.0))

    # ---- Valid arguments iterable of shape (2,)
    pm_ecc = population.PM_Eccentricity(np.random.uniform(0.0, 100, 2))
    pm_ecc = population.PM_Eccentricity([1.0, 2.0])
    pm_ecc = population.PM_Eccentricity((1.0, 2.0))
    pm_ecc = population.PM_Eccentricity(np.array([1.0, 2.0]))

    # ---- Invalid arguments NOT iterable of shape (2,)
    with pytest.raises(ValueError):
        pm_ecc = population.PM_Eccentricity(None)

    with pytest.raises(ValueError):
        pm_ecc = population.PM_Eccentricity(2.0)

    with pytest.raises(ValueError):
        pm_ecc = population.PM_Eccentricity([2.0])

    with pytest.raises(ValueError):
        pm_ecc = population.PM_Eccentricity([2.0, 3.0, 4.0])

    with pytest.raises(ValueError):
        pm_ecc = population.PM_Eccentricity(np.array([[1.0, 2.0]]))

    return


def test_eccentricity_illustris_trends():
    """Make sure eccentricity values make sense for different parameters.
    """

    cent_list = [0.01, 0.1, 0.5, 1.0, 10.0]
    wids_list = [0.01, 0.05, 0.1, 0.3]

    """Make sure that as the center value increases, the average eccentricity increases
    """
    # Test a range of width values
    for wid in wids_list:
        # store the average eccentricity for each cent parameter
        aves = []
        for cent in cent_list:
            pm_ecc = population.PM_Eccentricity((cent, wid))
            pop = population.Pop_Illustris(mods=pm_ecc)
            xx = pop.eccen

            aves.append(np.mean(xx))

        # make sure averages are all increasing
        err = (
            f"wid={wid} | cents={cent_list} gives non-monotonically increasing averages!  "
            f"  ::  aves={aves}"
        )
        assert np.all(np.diff(aves) > 0.0), err

    """Make sure that as the width value increases, the eccentricity stdev increases
    """
    # Test a range of center values
    for cent in cent_list:
        # store the stdev for each width parameter
        stdevs = []
        for wid in wids_list:
            pm_ecc = population.PM_Eccentricity((cent, wid))
            pop = population.Pop_Illustris(mods=pm_ecc)
            xx = pop.eccen
            stdevs.append(np.std(xx))

        # make sure averages are all increasing
        err = (
            f"cent={cent} | widths={wids_list} gives non-monotonically increasing stdevs!  "
            f"  ::  stdevs={stdevs}"
        )
        assert np.all(np.diff(stdevs) > 0.0), err

    return


# ====    PM_Resample    ====

def test_resample_basic():
    TRIES = 4
    fmt = ".4e"
    old_size = None
    for ii, resamp in enumerate(np.random.randint(6, 10, TRIES)):
        print(ii, f"resamp = {resamp}")
        pop = population.Pop_Illustris()

        if old_size is None:
            old_size = pop.size
            mt, mr = pop.mtmr
            sepa = pop.sepa
            scafa = pop.scafa
            old_aves = [np.mean(xx) for xx in [np.log10(mt), np.log10(mr), np.log10(sepa), scafa]]
            old_stdevs = [np.std(xx) for xx in [np.log10(mt), np.log10(mr), np.log10(sepa), scafa]]

            msg = [f"{xx:{fmt}}" for xx in old_aves]
            msg = ", ".join(msg)
            old_aves_str = "old_aves = " + msg

            msg = [f"{xx:{fmt}}" for xx in old_stdevs]
            msg = ", ".join(msg)
            old_stdevs_str = "old_stdevs = " + msg
        else:
            assert pop.size == old_size

        mod_resamp = population.PM_Resample(resample=resamp)
        pop.modify(mod_resamp)

        # Make sure the new size of the population is correct
        assert pop.size == old_size * resamp

        mt, mr = pop.mtmr
        sepa = pop.sepa
        scafa = pop.scafa
        new_aves = [np.mean(xx) for xx in [np.log10(mt), np.log10(mr), np.log10(sepa), scafa]]
        new_stdevs = [np.std(xx) for xx in [np.log10(mt), np.log10(mr), np.log10(sepa), scafa]]
        print(old_aves_str)
        msg = [f"{xx:{fmt}}" for xx in new_aves]
        msg = ", ".join(msg)
        print("new_aves = " + msg)

        print(old_stdevs_str)
        msg = [f"{xx:{fmt}}" for xx in new_stdevs]
        msg = ", ".join(msg)
        print("new_stdevs = " + msg)

        # Make sure the averages and stdevs match
        assert np.allclose(new_aves, old_aves, rtol=1e-1)
        assert np.allclose(new_stdevs, old_stdevs, rtol=1e-1)

    return


# ====    PM_Mass_Reset    ====

def test_mass_reset():
    print("test_mass_reset()")
    pop = population.Pop_Illustris()
    mmbulge_relation = holo.host_relations.MMBulge_MM2013()
    mod_MM2013 = population.PM_Mass_Reset(mmbulge_relation, scatter=False)

    mass_bef = pop.mass
    host = pop

    pop.modify(mod_MM2013)
    mass_aft = pop.mass

    assert not np.all(mass_bef == mass_aft), "Masses are unchanged after modification!"
    assert np.all((1e4 < mass_aft/MSOL) & (mass_aft/MSOL < 1e11)), "Modified masses outside of expectations!"
    check = mmbulge_relation.mbh_from_host(host, False)
    assert np.all(check == mass_aft), "Modified masses do not match mmbulge_relation values!"

    SCATTER = 0.1
    TOL_STD = 1.5 * sp.stats.norm.ppf(1.0 - 1.0 / pop.mass.size)
    print(f"TOL={TOL_STD}")
    mmbulge_relation = holo.host_relations.MMBulge_MM2013(scatter_dex=SCATTER)
    mod_MM2013 = population.PM_Mass_Reset(mmbulge_relation, scatter=True)
    pop.modify(mod_MM2013)
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
