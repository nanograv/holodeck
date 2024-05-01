"""
"""

import numpy as np

import holodeck as holo
from holodeck import utils
from holodeck.constants import MSOL, PC, YR


def _test_sam_basics(sam):
    NFREQS = 3
    NLOUDS = 4
    NREALS = 6

    # ---- check grid shape

    shape = sam.shape
    edges_shape = [len(ee) for ee in sam.edges]
    if shape != tuple(edges_shape):
        err = f"sam.shape={shape} != shape of edges={edges_shape}!"
        raise ValueError(err)

    # ---- static density

    dens = sam.static_binary_density
    if np.shape(dens) != shape:
        err = f"sam.shape={shape} != shape of density={np.shape(dens)}!"
        raise ValueError(err)

    # ---- try GWB

    fobs_cents, fobs_edges = holo.utils.pta_freqs(num=NFREQS)
    hc_ss, hc_bg = sam.gwb(fobs_edges, realize=NREALS, loudest=NLOUDS)
    if np.shape(hc_bg) != (NFREQS, NREALS):
        err = f"{np.shape(hc_bg)=} but {NFREQS=} {NREALS=}"
        raise ValueError(err)

    if np.shape(hc_ss) != (NFREQS, NREALS, NLOUDS):
        err = f"{np.shape(hc_ss)=} but {NFREQS=} {NREALS=} {NLOUDS=}"
        raise ValueError(err)

    return


def test_sam_basics():
    SHAPE = 12

    sam = holo.sams.Semi_Analytic_Model(shape=SHAPE)
    _test_sam_basics(sam)

    sam = holo.sams.Semi_Analytic_Model(shape=(11, 12, 13))
    _test_sam_basics(sam)
    return


def test_sam_basics_gpf_gmt():
    """explicitly construct SAM using GPF and GMT, and MMBULGE
    """
    SHAPE = 12

    gsmf = holo.sams.GSMF_Schechter()
    gpf = holo.sams.GPF_Power_Law()
    gmt = holo.sams.GMT_Power_Law()
    sam = holo.sams.Semi_Analytic_Model(gsmf=gsmf, gpf=gpf, gmt=gmt, shape=SHAPE)
    _test_sam_basics(sam)

    # With MMBulge also

    mmbulge = holo.host_relations.MMBulge_KH2013()
    sam = holo.sams.Semi_Analytic_Model(mmbulge=mmbulge, gpf=gpf, gmt=gmt, shape=SHAPE)
    _test_sam_basics(sam)
    return


def test_sam_basics_gmr():
    """explicitly construct SAM using GMR, and GMT, and MMBULGE
    """
    SHAPE = 12

    gsmf = holo.sams.GSMF_Schechter()
    gmr = holo.sams.GMR_Illustris()
    sam = holo.sams.Semi_Analytic_Model(gsmf=gsmf, gmr=gmr, shape=SHAPE)
    _test_sam_basics(sam)

    # With MMBulge and GMT also

    gmt = holo.sams.GMT_Power_Law()
    mmbulge = holo.host_relations.MMBulge_KH2013()
    sam = holo.sams.Semi_Analytic_Model(mmbulge=mmbulge, gmr=gmr, gmt=gmt, shape=SHAPE)
    _test_sam_basics(sam)

    return


def test_sam_basics__gsmf_double_chechter():
    """explicitly construct SAM GSMF_Double_Schechter
    """
    SHAPE = 12

    gsmf = holo.sams.GSMF_Double_Schechter()
    sam = holo.sams.Semi_Analytic_Model(gsmf=gsmf, shape=SHAPE)
    _test_sam_basics(sam)

    return


# ===========================================
# ====    Test: dynamic_binary_number    ====
# ===========================================


def test_dbn_gw_only():
    """Test the dynamic_binary_number method using GW-only evolution.

    (1) runs without error
    (2) no redz_final values should be <= 0.0
    (3) dnum values are consistent between cython and python
    (4) redz_final values are consistent between cython and python

    """

    shape = (10, 11, 12)
    sam = holo.sams.Semi_Analytic_Model(shape=shape)
    hard_gw = holo.hardening.Hard_GW()

    PTA_DUR = 20.0 * YR
    NUM_FREQS = 9
    fobs_gw_cents, fobs_gw_edges = holo.utils.pta_freqs(PTA_DUR, NUM_FREQS)
    fobs_orb_cents = fobs_gw_cents / 2.0

    # (1) make sure it runs

    grid_py, dnum_py, redz_final_py = sam.dynamic_binary_number_at_fobs(hard_gw, fobs_orb_cents, use_cython=False)
    grid_cy, dnum_cy, redz_final_cy = sam.dynamic_binary_number_at_fobs(hard_gw, fobs_orb_cents, use_cython=True)

    # (2) no redz_final values should be after redshift zero (i.e. negative, '-1.0')

    assert np.all(redz_final_py > 0.0), f"Found negative redshifts in python-version: {utils.stats(redz_final_py)=}"
    assert np.all(redz_final_cy > 0.0), f"Found negative redshifts in cython-version: {utils.stats(redz_final_cy)=}"

    # (3,) dnum consistent between cython- and python- versions of calculation

    bads = ~np.isclose(dnum_py, dnum_cy)
    if np.any(bads):
        print(f"{utils.frac_str(bads)=}")
        print(f"{utils.stats(dnum_py[bads])=}")
        print(f"{utils.stats(dnum_cy[bads])=}")
        assert not np.any(bads), f"Found {utils.frac_str(bads)} inconsistent `dnum` b/t python and cython calcs!"

    # (4,) redz_final consistent between cython- and python- versions of calculation
    bads = (~np.isclose(redz_final_py, redz_final_cy)) & (dnum_py > 0.0)
    if np.any(bads):
        print(f"{utils.frac_str(bads)=}")
        print(f"{utils.stats(redz_final_py[bads])=}")
        print(f"{utils.stats(redz_final_cy[bads])=}")
        assert not np.any(bads), f"Found {utils.frac_str(bads)} inconsistent `redz_final` b/t python and cython calcs!"

    return


def test_dbn_phenom():
    """Test the dynamic_binary_number method using Phenomenological evolution.

    (1) runs without error
    (2) dnum values are consistent between cython and python
    (3) redz_final values are consistent between cython and python

    """

    shape = (10, 11, 12)
    sam = holo.sams.Semi_Analytic_Model(shape=shape)
    TIME = 1.0e9 * YR
    hard_phenom = holo.hardening.Fixed_Time_2PL_SAM(sam, TIME, num_steps=300)

    PTA_DUR = 20.0 * YR
    NUM_FREQS = 9
    fobs_gw_cents, fobs_gw_edges = holo.utils.pta_freqs(PTA_DUR, NUM_FREQS)
    fobs_orb_cents = fobs_gw_cents / 2.0
    fobs_orb_edges = fobs_gw_edges / 2.0

    # we'll allow differences at very low redshifts, where numerical differences become significant
    ALLOW_BADS_BELOW_REDZ = 1.0e-2

    # (1) make sure it runs

    grid_py, dnum_py, redz_final_py = sam.dynamic_binary_number_at_fobs(hard_phenom, fobs_orb_cents, use_cython=False)
    grid_cy, dnum_cy, redz_final_cy = sam.dynamic_binary_number_at_fobs(hard_phenom, fobs_orb_cents, use_cython=True)
    edges_py = grid_py[:-1] + [fobs_orb_edges,]
    edges_cy = grid_cy[:-1] + [fobs_orb_edges,]

    redz_not_ignore = (redz_final_py > ALLOW_BADS_BELOW_REDZ) | (redz_final_cy > ALLOW_BADS_BELOW_REDZ)

    # (2) the same dnum values are zero

    zeros_py = (dnum_py == 0.0)
    zeros_cy = (dnum_cy == 0.0)

    # ignore mismastch at low-redshifts
    bads = (zeros_py != zeros_cy) & redz_not_ignore
    if np.any(bads):
        print(f"{utils.frac_str(bads)=}")
        print(f"{utils.stats(dnum_py[bads])=}")
        print(f"{utils.stats(dnum_cy[bads])=}")
        assert not np.any(bads), "Zero points in `dnum` do not match between python and cython!"

    # (3) dnum consistent between cython- and python- versions of calculation

    # ignore mismastch at low-redshifts
    bads = ~np.isclose(dnum_py, dnum_cy, rtol=1e-1) & redz_not_ignore
    if np.any(bads):
        errs = (dnum_py - dnum_cy) / dnum_cy
        print(f"{utils.frac_str(bads)=}")
        print(f"{utils.stats(errs)=}")
        print(f"{utils.stats(errs[bads])=}")
        print(f"{dnum_py[bads][:10]=}")
        print(f"{dnum_cy[bads][:10]=}")
        print(f"{errs[bads][:10]=}")
        print(f"{utils.stats(dnum_py[bads])=}")
        print(f"{utils.stats(dnum_cy[bads])=}")
        assert not np.any(bads), f"Found {utils.frac_str(bads)} inconsistent `dnum` b/t python and cython calcs!"

    # (3,) redz_final consistent between cython- and python- versions of calculation

    # ignore mismastch at low-redshifts
    bads = (~np.isclose(redz_final_py, redz_final_cy, rtol=1e-2)) & redz_not_ignore
    if np.any(bads):
        print(f"{utils.frac_str(bads)=}")
        print(f"{redz_final_py[bads][:10]=}")
        print(f"{redz_final_cy[bads][:10]=}")
        print(f"{utils.stats(redz_final_py[bads])=}")
        print(f"{utils.stats(redz_final_cy[bads])=}")
        assert not np.any(bads), f"Found {utils.frac_str(bads)} inconsistent `redz_final` b/t python and cython calcs!"

    # (4,) make sure that ALL numbers of binaries are consistent

    num_py = holo.sams.sam_cyutils.integrate_differential_number_3dx1d(edges_py, dnum_py)
    num_cy = holo.sams.sam_cyutils.integrate_differential_number_3dx1d(edges_cy, dnum_cy)

    # Make sure that `atol` is also set to a reasonable value
    bads = ~np.isclose(num_py, num_cy, rtol=1e-2, atol=1.0e-1)
    if np.any(bads):
        print(f"{utils.frac_str(bads)=}")
        print(f"{num_py[bads][:10]=}")
        print(f"{num_cy[bads][:10]=}")
        print(f"{utils.stats(num_py[bads])=}")
        print(f"{utils.stats(num_cy[bads])=}")
        assert not np.any(bads), f"Found {utils.frac_str(bads)} inconsistent `num` b/t python and cython calcs!"

    return

