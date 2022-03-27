"""Tests for `relations.py` holodeck submodule.


"""

import numpy as np
import scipy as sp
import scipy.stats

import holodeck as holo
import holodeck.relations
from holodeck.constants import MSOL


# ==============================================================================
# ====    M-MBulge Relations    ====
# ==============================================================================


def mbh_from_mbulge_mm13(mbulge):

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


def mbh_from_mbulge_kh13(mbulge):

    # [KH13] Eq. 10
    AMP = 0.49 * (1e9 * MSOL)
    PLAW = 1.17
    # EPS = 0.28
    X0 = 1e11 * MSOL

    def func_kh13(xx):
        yy = AMP * np.power(xx/X0, PLAW)
        return yy

    mbh = func_kh13(mbulge)
    return mbh


def check_relation(mmbulge_relation, truth_func):
    print(f"check_relation() : testing '{mmbulge_relation.__class__}' against '{truth_func}'")

    # mbulge ==> mbh
    mbulge = np.logspace(8, 13, 11) * MSOL
    vals = mmbulge_relation.mbh_from_mbulge(mbulge, scatter=False)
    truth = truth_func(mbulge)

    print(f"mbulge [grams] = {mbulge}")
    print(f"vals           = {vals}")
    print(f"truth          = {truth}")
    assert np.allclose(vals, truth)

    # mbh ==> mbulge
    check_mbulge = mmbulge_relation.mbulge_from_mbh(vals, scatter=False)
    print(f"mbulge    = {mbulge}")
    print(f"check rev = {check_mbulge}")
    assert np.allclose(mbulge, check_mbulge)

    return


def check_scatter_per_dex(mmbulge_relation, scatter_dex):
    EXTR = [9.0, 12.0]   # values are log10(X/Msol)
    NUM = 1e4
    TOL = 3.0
    SIGMAS = [-2, -1, 0, 1, 2]
    # draw a single, random (log-uniform) bulge-mass within given bounds
    mbulge = np.random.uniform(*EXTR)
    # convert to grams
    mbulge = MSOL * np.power(10.0, mbulge)
    # create an array of `NUM` identical values
    xx = mbulge * np.ones(int(NUM))
    mbulge_log10 = np.log10(mbulge)

    # convert from mbulge to MBH including scatter, using uniform input values
    vals = mmbulge_relation.mbh_from_mbulge(xx, scatter=True)
    # without scatter, get the expected (central) value of MBH mass
    cent = mmbulge_relation.mbh_from_mbulge(10.0**mbulge_log10, scatter=False)
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


def test_mm13_scatter():
    mm13 = holo.relations.MMBulge_MM13()
    check_scatter_per_dex(mm13, 0.34)
    return


def test_kh13_scatter():
    kh13 = holo.relations.MMBulge_KH13()
    check_scatter_per_dex(kh13, 0.28)
    return


def test_mm13_basic():
    mm13 = holo.relations.MMBulge_MM13()
    check_relation(mm13, mbh_from_mbulge_mm13)
    return


def test_kh13_basic():
    kh13 = holo.relations.MMBulge_KH13()
    check_relation(kh13, mbh_from_mbulge_kh13)
    return


def check_mass_reset(mmbulge_relation, truth_func):
    pop = holo.Pop_Illustris()
    mod_mm13 = holo.population.PM_Mass_Reset(mmbulge_relation, scatter=False)
    mbulge = pop.mbulge
    pop.modify(mod_mm13)
    mass = pop.mass

    truth = truth_func(mbulge)
    print(f"Modified masses: {holo.utils.stats(mass/MSOL)}")
    print(f"Expected masses: {holo.utils.stats(truth/MSOL)}")
    assert np.allclose(mass, truth), "Modified masses do not match true values from M-Mbulge relation!"

    return


def test_mass_reset_mm13():
    print("test_mass_reset_mm13")
    mmbulge_relation = holo.relations.MMBulge_MM13()
    truth_func = mbh_from_mbulge_mm13
    check_mass_reset(mmbulge_relation, truth_func)
    return


def test_mass_reset_kh13():
    print("test_mass_reset_kh13")
    mmbulge_relation = holo.relations.MMBulge_KH13()
    truth_func = mbh_from_mbulge_kh13
    check_mass_reset(mmbulge_relation, truth_func)
    return