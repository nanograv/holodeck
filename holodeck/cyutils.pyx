"""
"""

import numpy as np
cimport numpy as np
np.import_array()

cimport scipy.special.cython_special as sp_special

from libc.stdio cimport printf

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cdef double ECCEN_ZERO_LIMIT = 1.0e-12

import holodeck as holo
from holodeck.constants import NWTG, SPLC, MPC


cdef double bessel_recursive(int nn, double ne, double jn_m1, double jn_m2):
    cdef double jn = (2*(nn-1) / ne) * jn_m1 - jn_m2
    return jn


# cdef api double[:] trapz_grid_weight(int index, int size, double[:] vals):
cdef api np.ndarray[np.double_t, ndim=1] trapz_grid_weight(int index, int size, np.ndarray[np.double_t, ndim=1] vals):
    cdef np.ndarray[np.double_t, ndim=1] rv = np.zeros((2,), dtype=np.double)
    if index == 0:
        rv[0] = 2.0
        rv[1] = vals[1] - vals[0]   # i.e. vals[index+1] - vals[index]
        return rv
        # return

    if index == size - 1:
        rv[0] = 2.0
        rv[1] = vals[index] - vals[index-1]
        return rv
        # return

    rv[0] = 1.0
    # same as average of both dx values:     0.5 * ((vals[index] - vals[index-1]) + (vals[index+1] - vals[index]))
    rv[1] = 0.5 * (vals[index+1] - vals[index-1])
    return rv
    # return

    # cdef int rv = <int>((index == 0) | (index == size-1))
    # rv += 1
    # return rv


cdef api double gw_freq_dist_func__scalar_scalar(int nn, double ee):

    if abs(ee) < ECCEN_ZERO_LIMIT:
        if nn == 2:
            return 1.0

        return 0.0

    cdef double jn_m2, jn_m1, jn, jn_p1, jn_p2
    cdef double aa, bb, cc, gg

    cdef double ne = nn*ee
    # cdef double n2 = np.square(nn)   # THIS CAUSES A SEG-FAULT
    cdef double n2 = nn * nn

    jn_m2 = sp_special.jv(nn-2, ne)
    jn_m1 = sp_special.jv(nn-1, ne)

    jn = bessel_recursive(nn, ne, jn_m1, jn_m2)
    jn_p1 = bessel_recursive(nn+1, ne, jn, jn_m1)
    jn_p2 = bessel_recursive(nn+2, ne, jn_p1, jn)

    aa = jn_m2 - 2.0*ee*jn_m1 + (2/nn)*jn + 2*ee*jn_p1 - jn_p2
    aa = aa * aa
    bb = jn_m2 - 2*ee*jn + jn_p2
    bb = (1 - ee*ee)*bb*bb
    cc = (4.0/(3.0*n2)) * jn * jn
    gg = (n2*n2/32) * (aa + bb + cc)

    return gg


def sam_calc_gwb_1(ndens, mtot, mrat, redz, dcom, gwfobs, sepa_evo, eccen_evo, nharms=100):
    return _sam_calc_gwb_1(ndens, mtot, mrat, redz, dcom, gwfobs, sepa_evo, eccen_evo, nharms)


cdef np.ndarray[np.double_t] _sam_calc_gwb_1(
    np.ndarray[np.double_t, ndim=3] ndens,
    np.ndarray[np.double_t, ndim=1] mtot,
    np.ndarray[np.double_t, ndim=1] mrat,
    np.ndarray[np.double_t, ndim=1] redz,
    np.ndarray[np.double_t, ndim=1] dcom,
    np.ndarray[np.double_t, ndim=1] gwfobs,
    np.ndarray[np.double_t, ndim=1] sepa_evo,
    np.ndarray[np.double_t, ndim=1] eccen_evo,
    int nharms
):

    cdef int nfreqs, n_mtot, n_mrat, n_redz
    cdef np.ndarray[np.double_t, ndim=1] test = np.zeros(5, dtype=np.double)

    nfreqs = len(gwfobs)
    gwfobs_harms = np.zeros((nfreqs, nharms))
    cdef int ii
    for ii in range(nfreqs):
        gwfo = gwfobs[ii]
        for nh in range(1, nharms+1):
            gwfobs_harms[ii, nh-1] = gwfo / nh

    n_mtot = len(mtot)
    n_mrat = len(mrat)
    n_redz = len(redz)

    assert len(dcom) == n_redz
    assert ndens.ndim == 3
    cdef double sam_shape[3]
    sam_shape[:] = [n_mtot, n_mrat, n_redz]
    for ii in range(3):
        assert ndens.shape[ii] == sam_shape[ii]

    assert len(sepa_evo) == len(eccen_evo)

    shape = (n_mtot, n_mrat, n_redz, nfreqs, nharms)
    # setup output arrays with shape (M, Q, Z, F, H)
    hc2 = np.zeros(shape)
    hs2 = np.zeros(shape)
    hsn2 = np.zeros(shape)
    tau_out = np.zeros(shape)
    ecc_out = np.zeros(shape)

    gwfr_check = np.zeros(shape[2:])

    # NOTE: should sort `gwfobs_harms` into an ascending 1D array to speed up processes

    cdef int jj, kk, aa, bb
    for (aa, bb), gwfo in np.ndenumerate(gwfobs_harms):
        nh = bb + 1
        # iterate over mtot M
        for ii, mt in enumerate(mtot):
            mt = 10.0 ** mt
            # (Q,) masses of each component for this total-mass, and all mass-ratios
            m1 = mt / (1.0 + mrat)
            m2 = mt - m1
            mchirp = mt * np.power(mrat, 3.0/5.0) / np.power(1 + mrat, 6.0/5.0)

            # (E,) rest-frame orbital frequencies for this total-mass bin
            frst_evo = (1.0/(2.0*np.pi))*np.sqrt(NWTG*mt)/np.power(sepa_evo, 1.5)

            # iterate over redshifts Z
            for kk, zz in enumerate(redz):
                # () scalar
                zterm = (1.0 + zz)
                dc = dcom[kk]   # this is still in units of [Mpc]
                dc_term = 4*np.pi*(SPLC/MPC) * (dc**2)
                # rest-frame frequency corresponding to target observer-frame frequency of GW observations
                gwfr = gwfo * zterm
                gwfr_check[kk, aa, bb] = gwfr

                sa = np.power(NWTG*mt/np.square(2.0*np.pi*gwfr), 1.0/3.0)

                # interpolate to target (rest-frame) frequency
                # this is the same for all mass-ratios
                # () scalar
                ecc = np.interp(gwfr, frst_evo, eccen_evo)

                # da/dt values are negative, get a positive rate
                const = holo.utils._GW_DADT_SEP_CONST
                tau = const * m1 * m2 * (m1 + m2) / np.power(sa, 3)
                tau = tau * holo.utils._gw_ecc_func(ecc)

                # convert to timescale
                tau = sa / - tau

                tau_out[ii, :, kk, aa, bb] = tau
                ecc_out[ii, :, kk, aa, bb] = ecc

                # Calculate the GW spectral strain at each harmonic
                #    see: [Amaro-seoane+2010 Eq.9]
                # ()
                temp = 4.0 * gw_freq_dist_func__scalar_scalar(nh, ecc) / (nh ** 2)
                # (Q,)
                hs2[ii, :, kk, aa, bb] = np.square(holo.utils._GW_SRC_CONST * mchirp * np.power(2*mchirp*gwfr, 2/3) / (dc*MPC))
                hsn2[ii, :, kk, aa, bb] = temp * hs2[ii, :, kk, aa, bb]
                hc2[ii, :, kk, aa, bb] = ndens[ii, :, kk] * dc_term * zterm * tau * hsn2[ii, :, kk, aa, bb]

    gwb_shape = (nfreqs, nharms)
    cdef np.ndarray[np.double_t, ndim=2] gwb
    gwb = np.zeros(gwb_shape)

    cdef np.ndarray[np.double_t, ndim=1] ival, jval, kval
    cdef double weight, volume

    for aa, bb in np.ndindex(gwfobs_harms.shape):
        for ii, jj, kk in np.ndindex((n_mtot, n_mrat, n_redz)):
            ival = trapz_grid_weight(ii, n_mtot, mtot)
            jval = trapz_grid_weight(jj, n_mrat, mrat)
            kval = trapz_grid_weight(kk, n_redz, redz)
            weight = 1.0 / (ival[0] * jval[0] * kval[0])
            volume = ival[1] * jval[1] * kval[1]
            gwb[aa, bb] += hc2[ii, jj, kk, aa, bb] * weight * volume

    return gwb


def sam_calc_gwb_2(ndens, mtot, mrat, redz, dcom, gwfobs, sepa_evo, eccen_evo, nharms=100):
    return _sam_calc_gwb_1(ndens, mtot, mrat, redz, dcom, gwfobs, sepa_evo, eccen_evo, nharms)


cdef np.ndarray[np.double_t] _sam_calc_gwb_2(
    np.ndarray[np.double_t, ndim=3] ndens,
    np.ndarray[np.double_t, ndim=1] mtot,
    np.ndarray[np.double_t, ndim=1] mrat,
    np.ndarray[np.double_t, ndim=1] redz,
    np.ndarray[np.double_t, ndim=1] dcom,
    np.ndarray[np.double_t, ndim=1] gwfobs,
    np.ndarray[np.double_t, ndim=1] sepa_evo,
    np.ndarray[np.double_t, ndim=1] eccen_evo,
    int nharms
):

    cdef int nfreqs, n_mtot, n_mrat, n_redz

    nfreqs = len(gwfobs)
    n_mtot = len(mtot)
    n_mrat = len(mrat)
    n_redz = len(redz)
    n_eccs = len(sepa_evo)

    cdef np.ndarray[DTYPE_t, ndim=2] gwfobs_harms = np.zeros((nfreqs, nharms))
    cdef int ii
    for ii in range(nfreqs):
        gwfo = gwfobs[ii]
        for nh in range(1, nharms+1):
            gwfobs_harms[ii, nh-1] = gwfo / nh

    shape = (n_mtot, n_mrat, n_redz, nfreqs, nharms)
    # setup output arrays with shape (M, Q, Z, F, H)
    hc2 = np.zeros(shape)
    hs2 = np.zeros(shape)
    hsn2 = np.zeros(shape)
    tau_out = np.zeros(shape)
    ecc_out = np.zeros(shape)

    # NOTE: should sort `gwfobs_harms` into an ascending 1D array to speed up processes

    cdef int jj, kk, aa, bb, nh
    cdef double m1, m2, mchirp, mt, mr, zz
    cdef double gwfo, gwfr, zterm, dc_term
    cdef np.ndarray[DTYPE_t, ndim=1] frst_evo = np.empty(n_eccs, dtype=DTYPE)

    for (aa, bb), gwfo in np.ndenumerate(gwfobs_harms):
        nh = bb + 1
        # iterate over mtot M
        for ii, mt in enumerate(mtot):
            mt = 10.0 ** mt

            # (E,) rest-frame orbital frequencies for this total-mass bin
            # frst_evo = (1.0/(2.0*np.pi))*np.sqrt(NWTG*mt)/np.power(sepa_evo, 1.5)

            # iterate over redshifts Z
            for kk, zz in enumerate(redz):
                # () scalar
                zterm = (1.0 + zz)
                dc_term = dcom[kk]   # this is still in units of [Mpc]
                dc_term = 4*np.pi*(SPLC/MPC) * (dc_term**2)
                # rest-frame frequency corresponding to target observer-frame frequency of GW observations
                gwfr = gwfo * zterm
                sa = np.power(NWTG*mt/np.square(2.0*np.pi*gwfr), 1.0/3.0)

                for jj, qq in enumerate(mrat):
                    m1 = mt / (1.0 + qq)
                    m2 = mt - m1
                    mchirp = mt * np.power(qq, 3.0/5.0) / np.power(1 + qq, 6.0/5.0)



                # interpolate to target (rest-frame) frequency
                # this is the same for all mass-ratios
                # () scalar
                ecc = np.interp(gwfr, frst_evo, eccen_evo)

                # da/dt values are negative, get a positive rate
                const = holo.utils._GW_DADT_SEP_CONST
                tau = const * m1 * m2 * (m1 + m2) / np.power(sa, 3)
                tau = tau * holo.utils._gw_ecc_func(ecc)

                # convert to timescale
                tau = sa / - tau

                tau_out[ii, :, kk, aa, bb] = tau
                ecc_out[ii, :, kk, aa, bb] = ecc

                # Calculate the GW spectral strain at each harmonic
                #    see: [Amaro-seoane+2010 Eq.9]
                # ()
                temp = 4.0 * gw_freq_dist_func__scalar_scalar(nh, ecc) / (nh ** 2)
                # (Q,)
                hs2[ii, :, kk, aa, bb] = np.square(holo.utils._GW_SRC_CONST * mchirp * np.power(2*mchirp*gwfr, 2/3) / (dc*MPC))
                hsn2[ii, :, kk, aa, bb] = temp * hs2[ii, :, kk, aa, bb]
                hc2[ii, :, kk, aa, bb] = ndens[ii, :, kk] * dc_term * zterm * tau * hsn2[ii, :, kk, aa, bb]

    gwb_shape = (nfreqs, nharms)
    cdef np.ndarray[np.double_t, ndim=2] gwb
    gwb = np.zeros(gwb_shape)

    cdef np.ndarray[np.double_t, ndim=1] ival, jval, kval
    cdef double weight, volume

    for aa, bb in np.ndindex(gwfobs_harms.shape):
        for ii, jj, kk in np.ndindex((n_mtot, n_mrat, n_redz)):
            ival = trapz_grid_weight(ii, n_mtot, mtot)
            jval = trapz_grid_weight(jj, n_mrat, mrat)
            kval = trapz_grid_weight(kk, n_redz, redz)
            weight = 1.0 / (ival[0] * jval[0] * kval[0])
            volume = ival[1] * jval[1] * kval[1]
            gwb[aa, bb] += hc2[ii, jj, kk, aa, bb] * weight * volume

    return gwb








# cdef api double[:] gw_freq_dist_func__scalar_array(int nn, double[:] eccen):
#     cdef Py_ssize_t esize = eccen.shape[0]

#     # Calculate with non-zero eccentrictiy
#     cdef double ee, jn, jn_p1, jn_p2, jn_m1, jn_m2, aa, bb, cc
#     # cdef double ne = nn*ee`
#     cdef double ne, n2 = np.square(nn)

#     gne = np.zeros(esize)

#     cdef Py_ssize_t ii
#     for ii in range(esize):
#         ee = eccen[ii]
#         if abs(ee) < ECCEN_ZERO_LIMIT:
#             if nn == 2:
#                 gne[ii] = 1.0
#             else:
#                 gne[ii] = 0.0

#             continue

#         ne = nn * ee
#         jn_m2 = sp_special.jv(nn-2, ne)
#         jn_m1 = sp_special.jv(nn-1, ne)

#         jn = bessel_recursive(nn, ne, jn_m1, jn_m2)
#         jn_p1 = bessel_recursive(nn+1, ne, jn, jn_m1)
#         jn_p2 = bessel_recursive(nn+2, ne, jn_p1, jn)

#         aa = np.square(jn_m2 - 2.0*ee*jn_m1 + (2/nn)*jn + 2*ee*jn_p1 - jn_p2)
#         bb = (1 - ee*ee)*np.square(jn_m2 - 2*ee*jn + jn_p2)
#         cc = (4.0/(3.0*n2)) * np.square(jn)
#         gne[ii] = (n2*n2/32) * (aa + bb + cc)

#     return gne

