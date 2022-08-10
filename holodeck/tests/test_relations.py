"""Tests for `relations.py` holodeck submodule.


"""

import numpy as np
import scipy as sp
import scipy.stats

import holodeck as holo
import holodeck.relations
import holodeck.population
from holodeck.constants import MSOL


# ==============================================================================
# ====    M-MBulge Relations    ====
# ==============================================================================


def mbh_from_mbulge_MM2013(mbulge):

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

    # [KH2013] Eq. 10
    AMP = 0.49 * (1e9 * MSOL)
    PLAW = 1.17
    # EPS = 0.28
    X0 = 1e11 * MSOL

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

    print(f"mbulge [grams] = {host.mbulge}")
    print(f"vals           = {vals}")
    print(f"truth          = {truth}")
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
    MM2013 = holo.relations.MMBulge_MM2013()
    check_scatter_per_dex(MM2013, 0.34)
    return


def test_KH2013_scatter():
    KH2013 = holo.relations.MMBulge_KH2013()
    check_scatter_per_dex(KH2013, 0.28)
    return


def test_MM2013_basic():
    MM2013 = holo.relations.MMBulge_MM2013()
    check_relation(MM2013, mbh_from_mbulge_MM2013)
    return


def test_KH2013_basic():
    KH2013 = holo.relations.MMBulge_KH2013()
    check_relation(KH2013, mbh_from_mbulge_KH2013)
    return


def check_mass_reset(mmbulge_relation, truth_func):
    pop = holo.population.Pop_Illustris()
    mod_MM2013 = holo.population.PM_Mass_Reset(mmbulge_relation, scatter=False)
    pop.modify(mod_MM2013)
    mass = pop.mass

    truth = truth_func(pop.mbulge)
    print(f"Modified masses: {holo.utils.stats(mass/MSOL)}")
    print(f"Expected masses: {holo.utils.stats(truth/MSOL)}")
    assert np.allclose(mass, truth), "Modified masses do not match true values from M-Mbulge relation!"

    return


def test_mass_reset_MM2013():
    print("test_mass_reset_MM2013")
    mmbulge_relation = holo.relations.MMBulge_MM2013()
    truth_func = mbh_from_mbulge_MM2013
    check_mass_reset(mmbulge_relation, truth_func)
    return


def test_mass_reset_KH2013():
    print("test_mass_reset_KH2013")
    mmbulge_relation = holo.relations.MMBulge_KH2013()
    truth_func = mbh_from_mbulge_KH2013
    check_mass_reset(mmbulge_relation, truth_func)
    return


class Test_MMBulge_Standard:

    def test_dmstar_dmbh(self):

        args = [
            dict(),
            dict(mplaw=2.3),
            dict(mamp=1e4*MSOL, mplaw=2.3),
            dict(mamp=1e4*MSOL, mplaw=2.3, mref=1.0e9*MSOL),
        ]

        for arg in args:
            # Construct a relation using these arguments
            relation = holo.relations.MMBulge_Standard(**arg)
            # make sure arguments match stored values
            for kk, vv in arg.items():
                tt = getattr(relation, f"_{kk}")
                assert vv == tt, "Stored value ({tt:.8e}) does not match argument ({vv:.8e})!"

            # Choose a random stellar-mass
            MSTAR = MSOL * (10.0 ** np.random.uniform(9, 13))
            dm = 1.0e-2

            # numerically calculate derivative   dmstar / dmbh
            mstar = MSTAR * (1.0 + np.array([-dm, +dm]))
            mbulge = relation._bulge_mfrac * mstar
            mbh = relation.mbh_from_mbulge(mbulge, scatter=False)
            deriv = np.diff(mstar) / np.diff(mbh)
            deriv = deriv[0]
            # use analytic function in MMBulge_Standard relation
            test = relation.dmstar_dmbh(MSTAR)

            # make sure they're equal
            err = f"MMBulge_Standard({arg}).dmstar_dmbh value ({test:.8e}) does not match truth ({deriv:.8e})!"
            assert np.isclose(test, deriv), err

        return


class Test_Behroozi_2013:

    def test_init(self):
        holo.relations.Behroozi_2013()
        return

    def test_basics(self):
        NUM = 1000
        behr = holo.relations.Behroozi_2013()

        mstar = np.random.uniform(5, 12, NUM)
        mstar = MSOL * (10.0 ** mstar)
        redz = np.random.uniform(0.0, 6.0, NUM)
        mstar = np.sort(mstar)
        redz = np.sort(redz)[::-1]

        mhalo = behr.halo_mass(mstar, redz)
        assert np.all(mhalo > 0.0)

        mstar_check = behr.stellar_mass(mhalo, redz)
        mhalo_check = behr.halo_mass(mstar, redz)
        assert np.all(mstar_check > 0.0)
        print(f"mstar  input: {holo.utils.stats(mstar)}")
        print(f"mstar output: {holo.utils.stats(mstar_check)}")
        bads = ~np.isclose(mstar, mstar_check, rtol=0.1)
        if np.any(bads):
            print(f"bad mstar input  : {mstar[bads]/MSOL}")
            print(f"bad mstar output : {mstar_check[bads]/MSOL}")
        assert not np.any(bads)

        print(f"mhalo  input: {holo.utils.stats(mhalo/MSOL)}")
        print(f"mhalo output: {holo.utils.stats(mhalo_check/MSOL)}")
        bads = ~np.isclose(mhalo, mhalo_check, rtol=0.1)
        if np.any(bads):
            print(f"bad mhalo input  : {mhalo[bads]/MSOL}")
            print(f"bad mhalo output : {mhalo_check[bads]/MSOL}")
        assert not np.any(bads)

        return
