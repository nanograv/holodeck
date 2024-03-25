"""
"""

import numpy as np

import holodeck as holo
# from holodeck.constants import MSOL, PC, YR


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