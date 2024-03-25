"""Tests for bulge-fraction behavior in ``holodeck.host_relations``.
"""

import numpy as np
import pytest

from holodeck import host_relations
from holodeck.constants import MSOL


@pytest.fixture(scope="session")
def mstar_redz(size=9):
    mstar = (10.0 ** np.random.uniform(8, 12, size)) * MSOL
    redz = np.random.uniform(0.0, 10.0, size)
    return mstar, redz


def _check_any_bulge_frac(bf, mstar, redz):
    """Perform basic checks/diagnostics on the ``_Bulge_Frac`` subclass instance.
    """
    print(f"{__file__}::_check_any_bulge_frac() - {str(bf)}")

    assert issubclass(bf.__class__, host_relations._Bulge_Frac)
    assert isinstance(bf, host_relations._Bulge_Frac)

    # Get mbulge values to test
    mbulge = bf.mbulge_from_mstar(mstar, redz=redz)

    # make sure mbulge values are less than mstar values
    assert not np.any(mbulge > mstar)

    # make sure bulge-fractions are consistent
    bfracs = bf.bulge_frac(mstar, redz=redz)
    assert np.allclose(bfracs, mbulge/mstar)

    # make sure we recover the input stellar-masses
    mstar_test = bf.mstar_from_mbulge(mbulge, redz=redz)
    assert np.allclose(mstar_test, mstar)

    # check derivatives
    dmstar_dmbulge_test = bf.dmstar_dmbulge(mstar, redz=redz)
    # numerically calculate the derivates with finite differences
    delta = 1.0e-4
    mstar_lo = mstar * (1.0 - delta)
    mstar_hi = mstar * (1.0 + delta)
    mbulge_lo = bf.mbulge_from_mstar(mstar_lo, redz=redz)
    mbulge_hi = bf.mbulge_from_mstar(mstar_hi, redz=redz)
    dmstar_dmbulge_true = (mstar_hi - mstar_lo) / (mbulge_hi - mbulge_lo)
    # make sure values are consistent
    assert np.allclose(dmstar_dmbulge_true, dmstar_dmbulge_test)
    return mbulge


def test_bf_constant(mstar_redz):
    bulge_frac_value = np.random.uniform(0.0, 1.0)
    bf = host_relations.BF_Constant(bulge_frac_value)
    # unpack mstar and redz values from pytest.fixture
    mstar, redz = mstar_redz

    # perform basic checks, and obtain bulge-masses
    mbulge_test = _check_any_bulge_frac(bf, mstar, redz)

    # Calculate true values of mbulge
    mbulge_true = mstar * bulge_frac_value
    # make sure values are consistent
    assert np.allclose(mbulge_test, mbulge_true)

    return


def test_all_bulge_frac_classes(mstar_redz):
    print(f"{__file__}::test_all_bulge_frac_classes()")
    for name, bfrac_class in host_relations._bulge_frac_class_dict.items():
        print(name, bfrac_class)
        bf = bfrac_class()
        # unpack mstar and redz values from pytest.fixture
        mstar, redz = mstar_redz

        # perform basic checks
        _check_any_bulge_frac(bf, mstar, redz)

    return
