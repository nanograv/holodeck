"""
"""

import numpy as np
import pytest

import holodeck as holo
from holodeck.constants import MSOL

from holodeck.sams import comps


def _test_abc_parent(parent_class):

    # Parent = comps._Galaxy_Pair_Fraction

    # Make sure the abc cannot be instantiated directly
    with pytest.raises(TypeError):
        parent_class()

    # make sure a subclass cannot be instatiated directly
    class Child_Fail(parent_class):
        pass

    with pytest.raises(TypeError):
        Child_Fail()

    # make sure that a subclass which overrides `__init__` and `__call__` CAN be instantiated
    class Child_Good(parent_class):

        def __init__(self):
            pass

        def __call__(self):
            pass

    Child_Good()

    return


# ==============================================================================
# ====    Galaxy Stellar-Mass Function    ====
# ==============================================================================


def test_gsmf_base():
    _test_abc_parent(comps._Galaxy_Stellar_Mass_Function)
    return


def test_GSMF_Schechter():
    gsmf = comps.GSMF_Schechter()
    SIZE = 100
    mstar = MSOL * (10.0**np.random.uniform(10, 15, SIZE))
    redz = np.random.uniform(0.0, 10.0, SIZE)

    vals = gsmf(mstar, redz)   # noqa
    return


# ==============================================================================
# ====    Galaxy Pair Fraction    ====
# ==============================================================================


def test_gpf_base():
    _test_abc_parent(comps._Galaxy_Pair_Fraction)
    return


def test_GPF_Power_Law():
    gpf = comps.GPF_Power_Law()
    SIZE = 100
    mstar = MSOL * (10.0**np.random.uniform(10, 15, SIZE))
    mrat = 10.0 ** np.random.uniform(-4, 0.0, SIZE)
    redz = np.random.uniform(0.0, 10.0, SIZE)

    vals = gpf(mstar, mrat, redz)   # noqa
    return


# ==============================================================================
# ====    Galaxy Merger Time    ====
# ==============================================================================


def test_gmt_base():
    _test_abc_parent(comps._Galaxy_Merger_Time)
    return


def test_GMT_Power_Law():
    gmt = comps.GMT_Power_Law()
    SIZE = 100
    mstar = MSOL * (10.0**np.random.uniform(10, 15, SIZE))
    mrat = 10.0 ** np.random.uniform(-4, 0.0, SIZE)
    redz = np.random.uniform(0.0, 10.0, SIZE)

    vals = gmt(mstar, mrat, redz)   # noqa
    vals = gmt.zprime(mstar, mrat, redz)   # noqa
    return


# ==============================================================================
# ====    Galaxy Merger Rate    ====
# ==============================================================================


def test_gmr_base():
    _test_abc_parent(comps._Galaxy_Merger_Rate)
    return


def test_GMR_Power_Law():
    gmr = comps.GMR_Power_Law()
    SIZE = 100
    mstar = MSOL * (10.0**np.random.uniform(10, 15, SIZE))
    mrat = 10.0 ** np.random.uniform(-4, 0.0, SIZE)
    redz = np.random.uniform(0.0, 10.0, SIZE)

    vals = gmr(mstar, mrat, redz)   # noqa
    return


def test_GMR_Illustris():
    gmr = comps.GMR_Illustris()
    SIZE = 100
    mstar = MSOL * (10.0**np.random.uniform(10, 15, SIZE))
    mrat = 10.0 ** np.random.uniform(-4, 0.0, SIZE)
    redz = np.random.uniform(0.0, 10.0, SIZE)

    vals = gmr(mstar, mrat, redz)   # noqa
    return

