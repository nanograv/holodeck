"""Tests for bulge-fraction behavior in ``holodeck.host_relations``.
"""

import numpy as np
import pytest

from holodeck import host_relations
from holodeck.constants import MSOL


@pytest.fixture(scope="session")
def mstar_redz(size=9):
    """Create a set of random stellar-masses and redshifts, use pytest.fixtures to do this once.
    """
    mstar = (10.0 ** np.random.uniform(8, 12, size)) * MSOL
    redz = np.random.uniform(0.0, 10.0, size)
    mstar = np.sort(mstar)
    return mstar, redz


def _check_any_bulge_frac(bf, mstar, redz):
    """Perform basic checks/diagnostics on the ``_Bulge_Frac`` subclass instance.

    * Ensure types/inheritance seems correct.
    * Ensure bulge-fractions are less than unity.
    * Ensure returned bulge-fractions are consistent with ``mbulge_from_mstar``.
    * If ``mstar_from_mbulge`` is implemented, ensure that inverse relation of forward relation
      returns the initial inputs, i.e. $x = f^-1[ f(x) ]$.
    * Ensure that returned derivatives ``dmstar_dmbulge`` are consistent with finite differences.

    """
    print(f"{__file__}::_check_any_bulge_frac() - {bf}")

    # Make sure class types look right
    assert issubclass(bf.__class__, host_relations._Bulge_Frac)
    assert isinstance(bf, host_relations._Bulge_Frac)

    # Get mbulge values to test
    mbulge = bf.mbulge_from_mstar(mstar, redz=redz)

    # make sure mbulge values are less than mstar values
    assert not np.any(mbulge > mstar)

    # make sure bulge-fractions are consistent
    bfracs = bf.bulge_frac(mstar, redz=redz)
    assert np.allclose(bfracs, mbulge/mstar)

    # make sure we recover the input stellar-masses, if the reverse relationship is implemented
    try:
        mstar_test = bf.mstar_from_mbulge(mbulge, redz=redz)
        error_mstar = (mstar_test - mstar) / mstar
        print(f"{mstar_test=}")
        print(f"{mstar=}")
        print(f"{error_mstar=}")
        # The reverse relationship may be inverted numerically or with a fit, so allow a larger error
        assert np.allclose(mstar_test, mstar, rtol=1e-3)
    # if it's not implemented, that's okay
    except NotImplementedError:
        pass

    # ---- check derivatives

    dmstar_dmbulge_test = bf.dmstar_dmbulge(mbulge, redz=redz)
    # numerically calculate the derivates with finite differences
    delta = 1.0e-4
    mstar_lo = mstar * (1.0 - delta)
    mstar_hi = mstar * (1.0 + delta)
    mbulge_lo = bf.mbulge_from_mstar(mstar_lo, redz=redz)
    mbulge_hi = bf.mbulge_from_mstar(mstar_hi, redz=redz)
    dmstar_dmbulge_true = (mstar_hi - mstar_lo) / (mbulge_hi - mbulge_lo)
    error_deriv = (dmstar_dmbulge_test - dmstar_dmbulge_true) / dmstar_dmbulge_true
    print(f"{mstar=}")
    print(f"{dmstar_dmbulge_test=}")
    print(f"{dmstar_dmbulge_true=}")
    print(f"{error_deriv=}")
    # make sure values are consistent; calculation is numerical/fits to tolerate larger errors
    assert np.allclose(dmstar_dmbulge_true, dmstar_dmbulge_test, rtol=1e-3)

    return mbulge


def test_bf_constant(mstar_redz):
    """Test basic functionality of the ``BF_Constant`` class.
    """
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
    """Run every ``_Bulge_fraction`` subclass through the basic tests.

    Obtain subclasses using the ``holodeck.host_relations._bulge_frac_class_dict`` entries, and
    perform basic tests using ``_check_any_bulge_frac``.

    """

    print(f"{__file__}::test_all_bulge_frac_classes()")
    for name, bfrac_class in host_relations._bulge_frac_class_dict.items():
        print(name, bfrac_class)
        bf = bfrac_class()
        # unpack mstar and redz values from pytest.fixture
        mstar, redz = mstar_redz

        # perform basic checks
        _check_any_bulge_frac(bf, mstar, redz)

    return
