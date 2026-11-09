"""Tests for `host_relations.py` holodeck submodule.


"""

import numpy as np
import scipy as sp
import scipy.stats

import pytest

import holodeck as holo
from holodeck import host_relations
from holodeck.discrete import population
from holodeck.constants import MSOL, KMPERSEC


# ==============================================================================
# ====    M-Sigma Relations    ====
# ==============================================================================


@pytest.fixture(scope="session")
def vdisp_redz(size=9):
    vdisp = (10.0 ** np.random.uniform(1.7, 2.6, size)) * KMPERSEC
    redz = np.random.uniform(0.0, 10.0, size)
    return vdisp, redz

@pytest.fixture(scope="session")
def mstar_redz(size=9):
    mstar = (10.0 ** np.random.uniform(7, 12, size)) * MSOL
    redz = np.random.uniform(0.0, 10.0, size)
    return mstar, redz

def _check_any_msigma_relation_with_sigma(msigma, vdisp, redz):
    print(f"{__file__}:_check_any_msigma_relation_with_sigma() - {msigma}")

    # get black-hole masses
    mbh = msigma.mbh_from_vdisp(vdisp, redz=redz, scatter=False)

    # make sure we recover the input velocity dispersions, if the reverse relationship is implemented
    try:
        vdisp_test = msigma.vdisp_from_mbh(mbh, redz=redz, scatter=False)
        assert np.allclose(vdisp_test, vdisp)
    # if it's not implemented, that's okay
    except NotImplementedError:
        pass

    # check derivatives
    dsigma_dmbh_test = msigma.dsigma_dmbh(vdisp, redz=redz)
    # numerically calculate the derivates with finite differences
    delta = 1.0e-4
    vdisp_lo = vdisp * (1.0 - delta)
    vdisp_hi = vdisp * (1.0 + delta)
    mbh_lo = msigma.mbh_from_vdisp(vdisp_lo, redz=redz, scatter=False)
    mbh_hi = msigma.mbh_from_vdisp(vdisp_hi, redz=redz, scatter=False)
    dsigma_dmbh_true = (vdisp_hi - vdisp_lo) / (mbh_hi - mbh_lo)
    # make sure values are consistent
    assert np.allclose(dsigma_dmbh_true, dsigma_dmbh_test)

    return mbh


def test_all_msigma_relation_classes_with_vdisp(vdisp_redz):
    print(f"{__file__}::test_all_msigma_relation_classes_with_vdisp()")

    # unpack vdisp and redz values from pytest.fixture
    vdisp, redz = vdisp_redz

    for name, msigma_class in host_relations._msigma_relation_class_dict.items():
        print(name, msigma_class)

        # Use default instantiation
        msigma = msigma_class()
        # perform basic checks
        _check_any_msigma_relation_with_vdisp(msigma, vdisp, redz)


    return


def _check_any_msigma_relation_with_mstar(msigma, mass_star, redz):
    print(f"{__file__}:_check_any_msigma_relation_with_mstar() - {msigma}")

    # get black-hole masses
    mbh = msigma.mbh_from_mstar(mass_star, redz=redz, scatter=False)

    # make sure MBH masses are less than total stellar masses
    assert not np.any(mbh > mass_star)

    # make sure we recover the input total stellar masses, if the reverse relationship is implemented
    try:
        mstar_test = msigma.mstar_from_mbh(mbh, redz=redz, scatter=False)
        print(f"{mstar_test=}")
        print(f"{mass_star=}")
        # The inverse relationship can use numerical fits / interpolation, so allow larger errors
        assert np.allclose(mstar_test, mass_star, rtol=1e-2)
    # if it's not implemented, that's okay
    except NotImplementedError:
        pass

    # check derivatives
    dmstar_dmbh_test = msigma.dmstar_dmbh(mass_star, redz=redz)
    # numerically calculate the derivates with finite differences
    delta = 1.0e-6
    mass_star_hi = mass_star * (1.0 + delta)
    mass_star_lo = mass_star * (1.0 - delta)
    mbh_hi = msigma.mbh_from_mstar(mass_star_hi, redz=redz, scatter=False)
    mbh_lo = msigma.mbh_from_mstar(mass_star_lo, redz=redz, scatter=False)
    dmstar_dmbh_true = (mass_star_hi - mass_star_lo) / (mbh_hi - mbh_lo)
    # make sure values are consistent
    deriv_error = (dmstar_dmbh_test - dmstar_dmbh_true) / dmstar_dmbh_true
    print(f"{dmstar_dmbh_true=}")
    print(f"{dmstar_dmbh_test=}")
    print(f"{deriv_error=}")
    # The derivatives can use numerical fits / interpolation, so allow larger errors
    assert np.allclose(dmstar_dmbh_true, dmstar_dmbh_test, rtol=1e-2)

    return mbh


def test_all_msigma_relation_classes_with_mstar(mstar_redz):
    print(f"{__file__}::test_all_mmbulge_relation_classes_with_mstar()")

    # unpack mstar and redz values from pytest.fixture
    mass_star, redz = mstar_redz

    for name, msigma_class in host_relations._msigma_relation_class_dict.items():
        print(name, msigma_class)

        # Use default instantiation
        msigma = msigma_class()
        # perform basic checks
        _check_any_msigma_relation_with_mstar(msigma, mass_star, redz)

    return


def mbh_from_vdisp_MM2013(vdisp):
    """This is a manually-coded version of the TRUE [MM2013]_ relation to use for comparisons."""

    ALPHA = 8.32
    BETA = 5.64
    X0 = 200 * KMPERSEC
    Y0 = MSOL

    def func(xx):
        xx = xx / X0   # x units
        yy = np.power(10.0, ALPHA + BETA * np.log10(xx))
        yy = yy * Y0   # add units
        return yy

    mbh = func(vdisp)
    return mbh


def mbh_from_vdisp_KH2013(vdisp):
    """This is a manually-coded version of the TRUE [KH2013]_ relation to use for comparisons."""


    AMP_LOG10 = 8.46
    PLAW = 4.26

    X0 = 200 * KMPERSEC

    AMP = (10.0 ** AMP_LOG10) * MSOL

    def func_KH2013(xx):
        yy = AMP * np.power(xx/X0, PLAW)
        return yy

    mbh = func_KH2013(vdisp)
    return mbh


def check_relation(msigma_relation, truth_func):
    print(f"check_relation() : testing '{msigma_relation.__class__}' against '{truth_func}'")

    # vdisp ==> mbh
    # vdisp = {'vdisp': np.logspace(1.7, 2.6, 11) * KMPERSEC}
    class host:
        vdisp = np.logspace(1.7, 2.6, 11) * KMPERSEC

    vals = msigma_relation.mbh_from_host(host, scatter=False)
    truth = truth_func(host.vdisp)

    err = (vals - truth) / truth
    print(f"vdisp [cm/s] = {host.vdisp}")
    print(f"vals           = {vals}")
    print(f"truth          = {truth}")
    print(f"errors         = {err}")
    assert np.allclose(vals, truth)

    # mbh ==> vdisp
    check_vdisp = msigma_relation.vdisp_from_mbh(vals, redz=None, scatter=False)
    print(f"vdisp    = {host.vdisp}")
    print(f"check rev = {check_vdisp}")
    assert np.allclose(host.vdisp, check_vdisp)

    return


def check_scatter_per_dex(msigma_relation, scatter_dex):
    EXTR = [1.7, 2.6]   # values are log10(X / km/s)
    NUM = 1e4
    TOL = 4.0
    SIGMAS = [-2, -1, 0, 1, 2]
    # draw a single, random (log-uniform) velocity dispersion within given bounds
    vdisp = np.random.uniform(*EXTR)
    # convert to centimeters per second
    vdisp = KMPERSEC * np.power(10.0, vdisp)
    # create an array of `NUM` identical values
    xx = vdisp * np.ones(int(NUM))
    vdisp_log10 = np.log10(vdisp)

    # convert from vdisp to MBH including scatter, using uniform input values
    class host: vdisp = xx    # noqa
    vals = msigma_relation.mbh_from_host(host, scatter=True)
    # without scatter, get the expected (central) value of MBH mass
    class host: vdisp = 10.0**vdisp_log10    # noqa
    cent = msigma_relation.mbh_from_host(host, scatter=False)
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
    MM2013 = host_relations.MSigma_MM2013()
    check_scatter_per_dex(MM2013, 0.38)
    return


def test_KH2013_scatter():
    KH2013 = host_relations.MSigma_KH2013()
    check_scatter_per_dex(KH2013, 0.30)
    return


def test_MM2013_basic():
    MM2013 = host_relations.MSigma_MM2013()
    check_relation(MM2013, mbh_from_vdisp_MM2013)
    return


def test_KH2013_basic():
    KH2013 = host_relations.MSigma_KH2013()
    check_relation(KH2013, mbh_from_vdisp_KH2013)
    return


def check_mass_reset(msigma_relation, truth_func):
    pop = population.Pop_Illustris()
    mod_MM2013 = population.PM_Mass_Reset(msigma_relation, scatter=False)
    pop.modify(mod_MM2013)
    mass = pop.mass

    truth = truth_func(pop.mbulge)
    print(f"Modified masses: {holo.utils.stats(mass/MSOL)}")
    print(f"Expected masses: {holo.utils.stats(truth/MSOL)}")
    assert np.allclose(mass, truth), "Modified masses do not match true values from M-Sigma relation!"

    return


def test_mass_reset_MM2013():
    print("test_mass_reset_MM2013")
    msigma_relation = host_relations.MSigma_MM2013()
    truth_func = mbh_from_vdisp_MM2013
    check_mass_reset(msigma_relation, truth_func)
    return


def test_mass_reset_KH2013():
    print("test_mass_reset_KH2013")
    msigma_relation = host_relations.MSigma_KH2013()
    truth_func = mbh_from_vdisp_KH2013
    check_mass_reset(msigma_relation, truth_func)
    return