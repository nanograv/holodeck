"""
"""

from pathlib import Path

import numpy as np
import scipy as sp

import holodeck as holo


def sens_curve(expect, wlog_test, amplog_test):

    avrgfactor = 1/1.143    #factor to correct for sky average from MCMC results
    onesigmafactor = 0.63   #factor such that lack of signal is only excluded at 1 sigma, rather than at very high sigma.

    f_arr = 10**wlog_test/np.pi                    #from binary orbit to GW frequency
    # sens_240hr10s = 10**amplog_test
    sens_29200hr10s = 10**amplog_test/np.sqrt(365) #convert from NR = 10 to NR = 10 x 365
    #Now we heuristically remove the fact that the coherent signal has been observed for many cycles.
    sens_1cycle = sens_29200hr10s*np.sqrt((29200*60*60*f_arr))
    sensfinal = sens_1cycle*avrgfactor*onesigmafactor

    interp_xx = np.log10(f_arr)

    if expect == 'base':
        interp_yy = np.log10(sensfinal*3)
    elif expect == 'priority':
        interp_yy = np.log10(sensfinal)
    elif expect == 'optimistic':
        interp_yy = np.log10(sensfinal/3.3333/np.sqrt(3))
    # elif expect == 'oldcurve':
    #     hc_SNR1_curv_data = SensCurvFunction(np.log10(f_cent), FileName = "sens_curvs/oldsens_fit.csv")
    #     hc_SNR1_curv      = sc.interpolate.interp1d(frns, hc_SNR1_curv_data)
    else:
        raise

    loghc_SNR1_curv = sp.interpolate.interp1d(interp_xx, interp_yy, kind='linear', fill_value="extrapolate")

    def hc_SNR1_curv(f):
        return 10**(loghc_SNR1_curv(np.log10(f)))

    return hc_SNR1_curv


def detectable(edges, redz_final, snr, tau_obs, sens_curve_interp):
    """
    Arguments
    ---------
    edges : {mtot_edges, mrat_edges, redz_edges, fobs_orb_edges}

    """
    # edges = [sam.mtot, sam.mrat, sam.redz, fobs_orb_edges]
    mtot_edges = edges[0]
    mrat_edges = edges[1]
    fobs_orb_edges = edges[3]

    fobs_gw_edges = fobs_orb_edges * 2.0
    fobs_gw_cents = 0.5 * (fobs_gw_edges[1:] + fobs_gw_edges[:-1])
    fobs_orb_cents = fobs_gw_cents / 2.0

    # print("detectable()")
    # for ed in edges:
    #     print(f"{ed.shape=}")

    # print(f"{fobs_orb_cents.shape=}")
    # print(f"{redz_final.shape=}")

    frst_orb = holo.utils.frst_from_fobs(
        fobs_orb_cents[np.newaxis, np.newaxis, np.newaxis, :],
        redz_final
    )
    mchirp = holo.utils.chirp_mass_mtmr(
        mtot_edges[:, np.newaxis, np.newaxis, np.newaxis],
        mrat_edges[np.newaxis, :, np.newaxis, np.newaxis]
    )
    dcom = np.inf * np.ones_like(redz_final)
    idx = (redz_final > 0.0)
    dcom[idx] = holo.cosmo.z_to_dcom(redz_final[idx])

    hs_obs = holo.utils.gw_strain_source(mchirp, dcom, frst_orb)
    hc_obs = np.sqrt(fobs_gw_cents * tau_obs) * hs_obs

    hnoise = snr * sens_curve_interp(fobs_gw_cents)

    detectable = (hc_obs > hnoise)

    return detectable
