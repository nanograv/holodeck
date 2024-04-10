"""Tests for `host_relations.py` holodeck submodule.


"""

import numpy as np
import scipy as sp
import scipy.stats

import pytest

import holodeck as holo
from holodeck import host_relations
from holodeck.discrete import population
from holodeck.constants import MSOL


# ==============================================================================
# ====    M-MBulge Relations    ====
# ==============================================================================


@pytest.fixture(scope="session")
def mbulge_redz(size=9):
    mbulge = (10.0 ** np.random.uniform(7, 12, size)) * MSOL
    redz = np.random.uniform(0.0, 10.0, size)
    return mbulge, redz


@pytest.fixture(scope="session")
def mstar_redz(size=9):
    mbulge = (10.0 ** np.random.uniform(7, 12, size)) * MSOL
    redz = np.random.uniform(0.0, 10.0, size)
    return mbulge, redz


def _check_any_mmbulge_relation_with_mbulge(mmbulge, mass_bulge, redz):
    print(f"{__file__}:_check_any_mmbulge_relation_with_mbulge() - {mmbulge}")

    # # unpack pytest.fixture values
    # mass_bulge, redz = mbulge_redz

    # get black-hole masses
    mbh = mmbulge.mbh_from_mbulge(mass_bulge, redz=redz, scatter=False)

    # make sure MBH masses are less than bulge masses
    assert not np.any(mbh > mass_bulge)

    # make sure we recover the input bulge-masses, if the reverse relationship is implemented
    try:
        mbulge_test = mmbulge.mbulge_from_mbh(mbh, redz=redz, scatter=False)
        assert np.allclose(mbulge_test, mass_bulge)
    # if it's not implemented, that's okay
    except NotImplementedError:
        pass

    # check derivatives
    dmbulge_dmbh_test = mmbulge.dmbulge_dmbh(mass_bulge, redz=redz)
    # numerically calculate the derivates with finite differences
    delta = 1.0e-4
    mass_bulge_lo = mass_bulge * (1.0 - delta)
    mass_bulge_hi = mass_bulge * (1.0 + delta)
    mbh_lo = mmbulge.mbh_from_mbulge(mass_bulge_lo, redz=redz, scatter=False)
    mbh_hi = mmbulge.mbh_from_mbulge(mass_bulge_hi, redz=redz, scatter=False)
    dmbulge_dmbh_true = (mass_bulge_hi - mass_bulge_lo) / (mbh_hi - mbh_lo)
    # make sure values are consistent
    assert np.allclose(dmbulge_dmbh_true, dmbulge_dmbh_test)

    return mbh


def test_all_mmbulge_relation_classes_with_mbulge(mbulge_redz):
    print(f"{__file__}::test_all_mmbulge_relation_classes_with_mbulge()")

    # unpack mstar and redz values from pytest.fixture
    mass_bulge, redz = mbulge_redz

    for name, mmbulge_class in host_relations._mmbulge_relation_class_dict.items():
        print(name, mmbulge_class)

        # Use default instantiation
        mmbulge = mmbulge_class()
        # perform basic checks
        _check_any_mmbulge_relation_with_mbulge(mmbulge, mass_bulge, redz)

        # Use instantiation with custom bulge-fractions
        for bf_name, bfrac_class in host_relations._bulge_frac_class_dict.items():
            print(bf_name, bfrac_class)
            bf = bfrac_class()
            mmbulge = mmbulge_class(bulge_frac=bf)
            # unpack mstar and redz values from pytest.fixture
            mass_bulge, redz = mbulge_redz
            # perform basic checks
            _check_any_mmbulge_relation_with_mbulge(mmbulge, mass_bulge, redz)

    return


def _check_any_mmbulge_relation_with_mstar(mmbulge, mass_star, redz):
    print(f"{__file__}:_check_any_mmbulge_relation_with_mstar() - {mmbulge}")

    # get black-hole masses
    mbh = mmbulge.mbh_from_mstar(mass_star, redz=redz, scatter=False)

    # make sure MBH masses are less than bulge masses
    assert not np.any(mbh > mass_star)

    # make sure we recover the input bulge-masses, if the reverse relationship is implemented
    try:
        mstar_test = mmbulge.mstar_from_mbh(mbh, redz=redz, scatter=False)
        print(f"{mstar_test=}")
        print(f"{mass_star=}")
        # The inverse relationship can use numerical fits / interpolation, so allow larger errors
        assert np.allclose(mstar_test, mass_star, rtol=1e-2)
    # if it's not implemented, that's okay
    except NotImplementedError:
        pass

    # check derivatives
    dmstar_dmbh_test = mmbulge.dmstar_dmbh(mass_star, redz=redz)
    # numerically calculate the derivates with finite differences
    delta = 1.0e-6
    mass_star_hi = mass_star * (1.0 + delta)
    mass_star_lo = mass_star * (1.0 - delta)
    mbh_hi = mmbulge.mbh_from_mstar(mass_star_hi, redz=redz, scatter=False)
    mbh_lo = mmbulge.mbh_from_mstar(mass_star_lo, redz=redz, scatter=False)
    dmstar_dmbh_true = (mass_star_hi - mass_star_lo) / (mbh_hi - mbh_lo)
    # make sure values are consistent
    deriv_error = (dmstar_dmbh_test - dmstar_dmbh_true) / dmstar_dmbh_true
    print(f"{dmstar_dmbh_true=}")
    print(f"{dmstar_dmbh_test=}")
    print(f"{deriv_error=}")
    # The derivatives can use numerical fits / interpolation, so allow larger errors
    assert np.allclose(dmstar_dmbh_true, dmstar_dmbh_test, rtol=1e-2)

    return mbh


def test_all_mmbulge_relation_classes_with_mstar(mstar_redz):
    print(f"{__file__}::test_all_mmbulge_relation_classes_with_mstar()")

    # unpack mstar and redz values from pytest.fixture
    mass_star, redz = mstar_redz

    for name, mmbulge_class in host_relations._mmbulge_relation_class_dict.items():
        print(name, mmbulge_class)

        # Use default instantiation
        mmbulge = mmbulge_class()
        # perform basic checks
        _check_any_mmbulge_relation_with_mstar(mmbulge, mass_star, redz)

        # Use instantiation with custom bulge-fractions
        for bf_name, bfrac_class in host_relations._bulge_frac_class_dict.items():
            print(bf_name, bfrac_class)
            bf = bfrac_class()
            mmbulge = mmbulge_class(bulge_frac=bf)

            # perform basic checks
            _check_any_mmbulge_relation_with_mstar(mmbulge, mass_star, redz)

    return


def mbh_from_mbulge_MM2013(mbulge):
    """This is a manually-coded version of the TRUE [MM2013]_ relation to use for comparisons."""

    ALPHA = 8.46
    BETA = 1.05
    # EPS = 0.34
    X0 = 1e11 * MSOL
    Y0 = MSOL

    def func(xx):
        xx = xx / X0   # x units
        yy = np.power(10.0, ALPHA + BETA * np.log10(xx))
        yy = yy * Y0   # add units
        return yy

    mbh = func(mbulge)
    return mbh


def mbh_from_mbulge_KH2013(mbulge):
    """This is a manually-coded version of the TRUE [KH2013]_ relation to use for comparisons."""

    # [KH2013] Eq. 10
    # AMP = 0.49 * (1e9 * MSOL)
    AMP_LOG10 = 8.69
    PLAW = 1.17
    # EPS = 0.28
    X0 = 1e11 * MSOL

    AMP = (10.0 ** AMP_LOG10) * MSOL

    def func_KH2013(xx):
        yy = AMP * np.power(xx/X0, PLAW)
        return yy

    mbh = func_KH2013(mbulge)
    return mbh


def check_relation(mmbulge_relation, truth_func):
    print(f"check_relation() : testing '{mmbulge_relation.__class__}' against '{truth_func}'")

    # mbulge ==> mbh
    # mbulge = {'mbulge': np.logspace(8, 13, 11) * MSOL}
    class host:
        mbulge = np.logspace(8, 13, 11) * MSOL

    vals = mmbulge_relation.mbh_from_host(host, scatter=False)
    truth = truth_func(host.mbulge)

    err = (vals - truth) / truth
    print(f"mbulge [grams] = {host.mbulge}")
    print(f"vals           = {vals}")
    print(f"truth          = {truth}")
    print(f"errors         = {err}")
    assert np.allclose(vals, truth)

    # mbh ==> mbulge
    check_mbulge = mmbulge_relation.mbulge_from_mbh(vals, scatter=False)
    print(f"mbulge    = {host.mbulge}")
    print(f"check rev = {check_mbulge}")
    assert np.allclose(host.mbulge, check_mbulge)

    return


def check_scatter_per_dex(mmbulge_relation, scatter_dex):
    EXTR = [9.0, 12.0]   # values are log10(X/Msol)
    NUM = 1e4
    TOL = 4.0
    SIGMAS = [-2, -1, 0, 1, 2]
    # draw a single, random (log-uniform) bulge-mass within given bounds
    mbulge = np.random.uniform(*EXTR)
    # convert to grams
    mbulge = MSOL * np.power(10.0, mbulge)
    # create an array of `NUM` identical values
    xx = mbulge * np.ones(int(NUM))
    mbulge_log10 = np.log10(mbulge)

    # convert from mbulge to MBH including scatter, using uniform input values
    class host: mbulge = xx    # noqa
    vals = mmbulge_relation.mbh_from_host(host, scatter=True)
    # without scatter, get the expected (central) value of MBH mass
    class host: mbulge = 10.0**mbulge_log10    # noqa
    cent = mmbulge_relation.mbh_from_host(host, scatter=False)
    cent = np.log10(cent)
    vals = np.log10(vals)

    # stdev of sampling distribution, e.g. expected stdev of sample average from true average
    stdev = scatter_dex / np.sqrt(NUM)

    # --- check average and median
    ave = np.mean(vals)
    med = np.median(vals)
    # "errors" in units of sample stdev
    err_ave = (ave - cent) / stdev
    err_med = (med - cent) / stdev
    print(f"true center: {cent:.4f}, mean: {ave:.4f}, med: {med:.4f}, sample stdev: {stdev:.6f}")
    print(f"\terror mean: {err_ave:.4f}, med: {err_med:.4f} (tolerance: TOL = {TOL:.4f})")
    assert np.fabs(err_ave) < TOL
    assert np.fabs(err_med) < TOL

    # ---- check quantiles
    # difference between sample quantiles and median value
    quants = holo.utils.quantiles(vals, sigmas=SIGMAS) - med
    # expected difference between sample quantiles and median value
    expect_quants = scatter_dex * np.array(SIGMAS)
    for ii, ss in enumerate(SIGMAS):
        # I think the effective population size is smaller for quantiles ??, compensate for that
        pop_size = sp.stats.norm.cdf(-np.fabs(ss)) * NUM
        eff_stdev = stdev * np.sqrt(NUM/pop_size)
        qq = quants[ii]
        ee = expect_quants[ii]
        # difference between measured and expected quantiles, in units of effective-population-stdev
        err = (qq - ee) / eff_stdev
        print(f"{ii}, {ss:+d} : q={qq:+.4f} ({ee:+.4f}) => err={err:+.4f}")
        # make sure difference is less than tolerance
        assert np.fabs(err) < TOL

    return


def test_MM2013_scatter():
    MM2013 = host_relations.MMBulge_MM2013()
    check_scatter_per_dex(MM2013, 0.34)
    return


def test_KH2013_scatter():
    KH2013 = host_relations.MMBulge_KH2013()
    check_scatter_per_dex(KH2013, 0.28)
    return


def test_MM2013_basic():
    MM2013 = host_relations.MMBulge_MM2013()
    check_relation(MM2013, mbh_from_mbulge_MM2013)
    return


def test_KH2013_basic():
    KH2013 = host_relations.MMBulge_KH2013()
    check_relation(KH2013, mbh_from_mbulge_KH2013)
    return


def check_mass_reset(mmbulge_relation, truth_func):
    pop = population.Pop_Illustris()
    mod_MM2013 = population.PM_Mass_Reset(mmbulge_relation, scatter=False)
    pop.modify(mod_MM2013)
    mass = pop.mass

    truth = truth_func(pop.mbulge)
    print(f"Modified masses: {holo.utils.stats(mass/MSOL)}")
    print(f"Expected masses: {holo.utils.stats(truth/MSOL)}")
    assert np.allclose(mass, truth), "Modified masses do not match true values from M-Mbulge relation!"

    return


def test_mass_reset_MM2013():
    print("test_mass_reset_MM2013")
    mmbulge_relation = host_relations.MMBulge_MM2013()
    truth_func = mbh_from_mbulge_MM2013
    check_mass_reset(mmbulge_relation, truth_func)
    return


def test_mass_reset_KH2013():
    print("test_mass_reset_KH2013")
    mmbulge_relation = host_relations.MMBulge_KH2013()
    truth_func = mbh_from_mbulge_KH2013
    check_mass_reset(mmbulge_relation, truth_func)
    return


class Test_MMBulge_Standard:

    def test_dmstar_dmbh(self):

        bfrac = host_relations.BF_Constant()

        kwargs = [
            dict(),
            dict(mplaw=2.3),
            dict(mamp=1e4*MSOL, mplaw=2.3),
            dict(mamp=1e4*MSOL, mplaw=2.3, mref=1.0e9*MSOL),
        ]

        for kw in kwargs:
            # Construct a relation using these arguments
            relation = host_relations.MMBulge_Standard(bulge_frac=bfrac, **kw)
            # make sure arguments match stored values
            for kk, vv in kw.items():
                tt = getattr(relation, f"_{kk}")
                assert vv == tt, "Stored value ({tt:.8e}) does not match argument ({vv:.8e})!"

            # Choose a random stellar-mass
            MSTAR = MSOL * (10.0 ** np.random.uniform(9, 13))
            dm = 1.0e-2

            # numerically calculate derivative   dmstar / dmbh
            mstar = MSTAR * (1.0 + np.array([-dm, +dm]))
            # mbulge = bfrac.bulge_frac() * mstar
            mbulge = bfrac.mbulge_from_mstar(mstar)
            dmstar_dmbulge = (np.diff(mstar) / np.diff(mbulge))[0]
            print(f"{dmstar_dmbulge=}  {bfrac.bulge_frac()=}")

            mbh = relation.mbh_from_mbulge(mbulge, scatter=False)
            dmbh_dmbulge = np.diff(mbh) / np.diff(mbulge)
            print(f"{dmbh_dmbulge=}  {bfrac.bulge_frac()=}")
            deriv = np.diff(mstar) / np.diff(mbh)
            deriv = deriv[0]
            # use analytic function in MMBulge_Standard relation
            test = relation.dmstar_dmbh(MSTAR)

            # make sure they're equal
            err = f"MMBulge_Standard({kw}).dmstar_dmbh value ({test:.8e}) does not match truth ({deriv:.8e})!"
            assert np.isclose(test, deriv), err

        return
