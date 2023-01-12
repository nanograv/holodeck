"""
"""
cimport cython

import numpy as np
cimport numpy as np
np.import_array()

cimport scipy.special.cython_special as sp_special

from libc.stdio cimport printf
from libc.stdlib cimport malloc, free
from libc.math cimport pow, sqrt, abs, M_PI

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


cdef double ECCEN_ZERO_LIMIT = 1.0e-12

import holodeck as holo
from holodeck.constants import NWTG, SPLC, MPC

cdef double MY_NWTG = 6.6742999e-08
cdef double MY_SPLC = 29979245800.0
cdef double MY_MPC = 3.08567758e+24
cdef double GW_DADT_SEP_CONST = - 64.0 * pow(MY_NWTG, 3) / 5.0 / pow(MY_SPLC, 5)
cdef double GW_SRC_CONST = 8.0 * pow(MY_NWTG, 5.0/3.0) * pow(M_PI, 2.0/3.0) / sqrt(10.0) / pow(MY_SPLC, 4.0)


cdef double bessel_recursive(int nn, double ne, double jn_m1, double jn_m2):
    cdef double jn = (2*(nn-1) / ne) * jn_m1 - jn_m2
    return jn


def py_trapz_grid_weight(index, size, vals):
    return trapz_grid_weight(index, size, vals)


@cython.boundscheck(False)
@cython.wraparound(False)
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


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void my_trapz_grid_weight(int index, int size, double[:] vals, double *rv):
    if index == 0:
        rv[0] = 2.0
        rv[1] = <double>(vals[1] - vals[0])   # i.e. vals[index+1] - vals[index]
        return

    if index == size - 1:
        rv[0] = 2.0
        rv[1] = <double>(vals[index] - vals[index-1])
        return

    rv[0] = 1.0
    # same as average of both dx values:     0.5 * ((vals[index] - vals[index-1]) + (vals[index+1] - vals[index]))
    rv[1] = 0.5 * <double>(vals[index+1] - vals[index-1])
    return


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
    return _sam_calc_gwb_2(ndens, mtot, mrat, redz, dcom, gwfobs, sepa_evo, eccen_evo, nharms)


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

    cdef (int, int) gwb_shape = (nfreqs, nharms)
    # cdef np.ndarray[DTYPE_t, ndim=2] gwfobs_harms = np.zeros(gwb_shape)
    cdef int ii, nh
    cdef double gwfo
    # for ii in range(nfreqs):
    #     gwfo = gwfobs[ii]
    #     for nh in range(1, nharms+1):
    #         gwfobs_harms[ii, nh-1] = gwfo / nh

    # NOTE: should sort `gwfobs_harms` into an ascending 1D array to speed up processes

    cdef int jj, kk, aa, bb
    cdef double m1, m2, mchirp, mt, mr, zz
    cdef double gwfr, zterm, dc, dc_term, gne, hterm
    cdef np.ndarray[DTYPE_t, ndim=1] frst_evo = np.empty(n_eccs, dtype=DTYPE)

    cdef np.ndarray[np.double_t, ndim=2] gwb
    gwb = np.zeros(gwb_shape)

    cdef np.ndarray[np.double_t, ndim=1] ival, jval, kval
    cdef double weight, volume

    for aa, bb in np.ndindex(gwb_shape):
        nh = bb + 1
        gwfo = gwfobs[aa] / nh
        # iterate over mtot M
        for ii, mt in enumerate(mtot):
            mt = 10.0 ** mt

            # (E,) rest-frame orbital frequencies for this total-mass bin
            frst_evo[:] = (1.0/(2.0*np.pi))*np.sqrt(NWTG*mt)/np.power(sepa_evo, 1.5)

            # iterate over redshifts Z
            for kk, zz in enumerate(redz):
                # () scalar
                zterm = (1.0 + zz)
                dc = dcom[kk]   # this is still in units of [Mpc]
                dc_term = 4*np.pi*(SPLC/MPC) * (dc**2)
                # rest-frame frequency corresponding to target observer-frame frequency of GW observations
                gwfr = gwfo * zterm
                sa = np.power(NWTG*mt/np.square(2.0*np.pi*gwfr), 1.0/3.0)

                # () scalar
                ecc = np.interp(gwfr, frst_evo, eccen_evo)

                for jj, qq in enumerate(mrat):
                    m1 = mt / (1.0 + qq)
                    m2 = mt - m1
                    mchirp = mt * np.power(qq, 3.0/5.0) / np.power(1 + qq, 6.0/5.0)

                    # da/dt values are negative, get a positive rate
                    const = holo.utils._GW_DADT_SEP_CONST
                    tau = const * m1 * m2 * (m1 + m2) / np.power(sa, 3)
                    tau = tau * holo.utils._gw_ecc_func(ecc)

                    # convert to timescale
                    tau = sa / - tau

                    # Calculate the GW spectral strain at each harmonic
                    #    see: [Amaro-seoane+2010 Eq.9]
                    # ()
                    gne = 4.0 * gw_freq_dist_func__scalar_scalar(nh, ecc) / (nh ** 2)
                    # (Q,)
                    hterm = np.square(holo.utils._GW_SRC_CONST * mchirp * np.power(2*mchirp*gwfr, 2/3) / (dc*MPC))
                    hterm = ndens[ii, jj, kk] * dc_term * zterm * tau * gne * hterm

                    ival = trapz_grid_weight(ii, n_mtot, mtot)
                    jval = trapz_grid_weight(jj, n_mrat, mrat)
                    kval = trapz_grid_weight(kk, n_redz, redz)
                    weight = 1.0 / (ival[0] * jval[0] * kval[0])
                    volume = ival[1] * jval[1] * kval[1]
                    gwb[aa, bb] += hterm * weight * volume

    return gwb


def sam_calc_gwb_3(ndens, mtot, mrat, redz, dcom, gwfobs, sepa_evo, eccen_evo, nharms=100):
    return _sam_calc_gwb_3(ndens, mtot, mrat, redz, dcom, gwfobs, sepa_evo, eccen_evo, nharms)


cdef np.ndarray[np.double_t] _sam_calc_gwb_3(
    np.ndarray[np.double_t, ndim=3] ndens,
    np.ndarray[np.double_t, ndim=1] mtot_log10,
    np.ndarray[np.double_t, ndim=1] mrat,
    np.ndarray[np.double_t, ndim=1] redz,
    np.ndarray[np.double_t, ndim=1] dcom,
    np.ndarray[np.double_t, ndim=1] gwfobs,
    np.ndarray[np.double_t, ndim=1] sepa_evo,
    np.ndarray[np.double_t, ndim=1] eccen_evo,
    int nharms
):
    printf("_sam_calc_gwb_3()")

    cdef int nfreqs, n_mtot, n_mrat, n_redz
    nfreqs = len(gwfobs)
    n_mtot = len(mtot_log10)
    n_mrat = len(mrat)
    n_redz = len(redz)
    n_eccs = len(sepa_evo)

    cdef np.ndarray[np.double_t, ndim=1] mtot = 10.0 ** mtot_log10
    cdef (int, int) gwb_shape = (nfreqs, nharms)
    cdef int ii, nh
    cdef double gwfo

    cdef int jj, kk, aa, bb
    cdef double m1, m2, mchirp, mt, mr, zz
    cdef double gwfr, zterm, dc, dc_term, gne, hterm

    cdef np.ndarray[np.double_t, ndim=2] gwb = np.zeros(gwb_shape)
    cdef np.ndarray[np.double_t, ndim=1] ival, jval, kval
    cdef double weight, volume

    cdef double tau_const = holo.utils._GW_DADT_SEP_CONST
    cdef double four_pi_c_mpc = 4*np.pi * (SPLC/MPC)

    cdef np.ndarray[DTYPE_t, ndim=1] frst_evo_pref = np.empty(n_eccs, dtype=DTYPE)
    frst_evo_pref[:] = (1.0/(2.0*np.pi))*np.sqrt(NWTG)/np.power(sepa_evo, 1.5)
    cdef double kep_sa_term = NWTG / np.square(2.0*np.pi)
    cdef double one_third = 1.0 / 3.0
    cdef double two_third = 2.0 / 3.0
    cdef double three_fifths = 3.0 / 5.0
    cdef double six_fifths = 6.0 / 5.0
    cdef double four_over_nh_squared, sa_inverse_cubed
    cdef double fe_ecc

    for aa, bb in np.ndindex(gwb_shape):
        nh = bb + 1
        four_over_nh_squared = 4.0 / (nh * nh)
        gwfo = gwfobs[aa] / nh
        # iterate over mtot M
        for ii, mt in enumerate(mtot):
            ival = trapz_grid_weight(ii, n_mtot, mtot_log10)

            # iterate over redshifts Z
            for kk, zz in enumerate(redz):
                kval = trapz_grid_weight(kk, n_redz, redz)
                # () scalar
                zterm = (1.0 + zz)
                dc = dcom[kk]   # this is still in units of [Mpc]
                dc_term = four_pi_c_mpc * (dc*dc)
                # rest-frame frequency corresponding to target observer-frame frequency of GW observations
                gwfr = gwfo * zterm
                sa = np.power(kep_sa_term*mt/np.square(gwfr), one_third)
                sa_inverse_cubed = np.power(sa, -3.0)

                # () scalar
                ecc = np.interp(gwfr, frst_evo_pref*np.sqrt(mt), eccen_evo)
                gne = gw_freq_dist_func__scalar_scalar(nh, ecc) * four_over_nh_squared
                fe_ecc = holo.utils._gw_ecc_func(ecc)

                weight = ival[1] * kval[1] / (ival[0] * kval[0])
                for jj, qq in enumerate(mrat):
                    jval = trapz_grid_weight(jj, n_mrat, mrat)
                    m1 = mt / (1.0 + qq)
                    m2 = mt - m1
                    mchirp = mt * np.power(qq, three_fifths) / np.power(1 + qq, six_fifths)

                    # da/dt values are negative, get a positive rate
                    tau = tau_const * fe_ecc * m1 * m2 * (m1 + m2) * sa_inverse_cubed
                    # convert to timescale
                    tau = sa / - tau

                    # Calculate the GW spectral strain at each harmonic
                    #    see: [Amaro-seoane+2010 Eq.9]
                    hterm = np.square(holo.utils._GW_SRC_CONST * mchirp * np.power(2*mchirp*gwfr, two_third) / (dc*MPC))
                    hterm = ndens[ii, jj, kk] * dc_term * zterm * tau * gne * hterm

                    # weight = 1.0 / (ival[0] * jval[0] * kval[0])
                    # volume = ival[1] * jval[1] * kval[1]
                    gwb[aa, bb] += hterm * weight * (jval[1] / jval[0])

    return gwb


def sam_calc_gwb_4(ndens, mtot, mrat, redz, dcom, gwfobs, sepa_evo, eccen_evo, nharms=100):
    return _sam_calc_gwb_4(ndens, mtot, mrat, redz, dcom, gwfobs, sepa_evo, eccen_evo, nharms)


cdef np.ndarray[np.double_t] _sam_calc_gwb_4(
    np.ndarray[np.double_t, ndim=3] ndens,
    np.ndarray[np.double_t, ndim=1] mtot_log10,
    np.ndarray[np.double_t, ndim=1] mrat,
    np.ndarray[np.double_t, ndim=1] redz,
    np.ndarray[np.double_t, ndim=1] dcom,
    np.ndarray[np.double_t, ndim=1] gwfobs,
    np.ndarray[np.double_t, ndim=1] sepa_evo,
    np.ndarray[np.double_t, ndim=1] eccen_evo,
    int nharms
):
    printf("\n_sam_calc_gwb_4()\n")

    cdef int nfreqs, n_mtot, n_mrat, n_redz
    nfreqs = len(gwfobs)
    n_mtot = len(mtot_log10)
    n_mrat = len(mrat)
    n_redz = len(redz)
    n_eccs = len(sepa_evo)

    cdef np.ndarray[np.double_t, ndim=1] mtot = 10.0 ** mtot_log10
    cdef (int, int) gwb_shape = (nfreqs, nharms)
    cdef int ii, nh
    cdef double gwfo

    cdef int jj, kk, aa, bb
    cdef double m1, m2, mchirp, mt, mr, zz, sa, qq, tau
    cdef double gwfr, zterm, dc, dc_term, gne, hterm

    cdef np.ndarray[np.double_t, ndim=2] gwb = np.zeros(gwb_shape)
    cdef np.ndarray[np.double_t, ndim=1] ival, jval, kval
    cdef double weight_ik, weight

    cdef double tau_const = holo.utils._GW_DADT_SEP_CONST
    cdef double four_pi_c_mpc = 4*np.pi * (SPLC/MPC)

    cdef np.ndarray[DTYPE_t, ndim=1] frst_evo_pref = np.empty(n_eccs, dtype=DTYPE)
    frst_evo_pref[:] = (1.0/(2.0*np.pi))*np.sqrt(NWTG)/np.power(sepa_evo, 1.5)
    cdef double kep_sa_term = NWTG / np.square(2.0*np.pi)
    cdef double one_third = 1.0 / 3.0
    cdef double two_third = 2.0 / 3.0
    cdef double three_fifths = 3.0 / 5.0
    cdef double six_fifths = 6.0 / 5.0
    cdef double four_over_nh_squared, sa_inverse_cubed
    cdef double fe_ecc

    # iterate over redshifts Z
    for kk, zz in enumerate(redz):
        kval = trapz_grid_weight(kk, n_redz, redz)
        zterm = (1.0 + zz)
        dc = dcom[kk]   # this is still in units of [Mpc]
        dc_term = four_pi_c_mpc * (dc*dc)


        # iterate over mtot M
        for ii, mt in enumerate(mtot):
            ival = trapz_grid_weight(ii, n_mtot, mtot_log10)

            weight_ik = ival[1] * kval[1] / (ival[0] * kval[0])

            for jj, qq in enumerate(mrat):
                jval = trapz_grid_weight(jj, n_mrat, mrat)
                weight = weight_ik * (jval[1] / jval[0])
                m1 = mt / (1.0 + qq)
                m2 = mt - m1
                mchirp = mt * np.power(qq, three_fifths) / np.power(1 + qq, six_fifths)

                for aa, gwfo in enumerate(gwfobs):
                    for bb in range(nharms):
                        nh = bb + 1
                        four_over_nh_squared = 4.0 / (nh * nh)
                        gwfr = gwfo * zterm / nh

                        # rest-frame frequency corresponding to target observer-frame frequency of GW observations
                        sa = np.power(kep_sa_term*mt/np.square(gwfr), one_third)
                        sa_inverse_cubed = np.power(sa, -3.0)

                        # () scalar
                        ecc = np.interp(gwfr, frst_evo_pref*np.sqrt(mt), eccen_evo)
                        gne = gw_freq_dist_func__scalar_scalar(nh, ecc) * four_over_nh_squared
                        # ecc = 0.5
                        # gne = 0.25

                        fe_ecc = holo.utils._gw_ecc_func(ecc)

                        # da/dt values are negative, get a positive rate
                        tau = tau_const * fe_ecc * m1 * m2 * (m1 + m2) * sa_inverse_cubed
                        # convert to timescale
                        tau = sa / - tau

                        # Calculate the GW spectral strain at each harmonic
                        #    see: [Amaro-seoane+2010 Eq.9]
                        hterm = np.square(holo.utils._GW_SRC_CONST * mchirp * np.power(2*mchirp*gwfr, two_third) / (dc*MPC))
                        hterm = ndens[ii, jj, kk] * dc_term * zterm * tau * gne * hterm

                        gwb[aa, bb] += hterm * weight

    return gwb


cdef double _gw_ecc_func(double eccen):
    cdef double e2 = eccen*eccen
    cdef double fe = (1.0 + (73.0/24.0)*e2 + (37.0/96.0)*e2*e2) / pow(1.0 - e2, 7.0/2.0)
    return fe


def sam_calc_gwb_5(ndens, mtot, mrat, redz, dcom, gwfobs, sepa_evo, eccen_evo, nharms=100):
    return _sam_calc_gwb_5(ndens, mtot, mrat, redz, dcom, gwfobs, sepa_evo, eccen_evo, nharms)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double[:, :] _sam_calc_gwb_5(
    double[:, :, :] ndens,
    double[:] mtot_log10,
    double[:] mrat,
    double[:] redz,
    double[:] dcom,
    double[:] gwfobs,
    double[:] sepa_evo,
    double[:] eccen_evo,
    int nharms
):
    printf("\n_sam_calc_gwb_5()\n")

    cdef int nfreqs = len(gwfobs)
    cdef int n_mtot = len(mtot_log10)
    cdef int n_mrat = len(mrat)
    cdef int n_redz = len(redz)
    cdef int n_eccs = len(sepa_evo)

    # cdef np.ndarray[np.double_t, ndim=1] mtot = 10.0 ** mtot_log10
    cdef double *mtot = <double *>malloc(n_mtot * sizeof(double))
    cdef int ii, nh

    cdef int jj, kk, aa, bb
    cdef double m1, m2, mchirp, sa, qq, tau
    cdef double gwfr, zterm, dc_cm, dc_mpc, dc_term, gne, hterm

    cdef np.ndarray[np.double_t, ndim=2] gwb = np.zeros((nfreqs, nharms))
    # cdef np.ndarray[np.double_t, ndim=1] ival, jval, kval
    cdef double *ival = <double *>malloc(2 * sizeof(double))
    cdef double *jval = <double *>malloc(2 * sizeof(double))
    cdef double *kval = <double *>malloc(2 * sizeof(double))
    cdef double weight_ik, weight

    cdef double four_pi_c_mpc = 4 * M_PI * (MY_SPLC / MY_MPC)

    cdef double kep_sa_term = MY_NWTG / pow(2.0*M_PI, 2)
    cdef double one_third = 1.0 / 3.0
    cdef double two_third = 2.0 / 3.0
    cdef double three_fifths = 3.0 / 5.0
    cdef double six_fifths = 6.0 / 5.0
    cdef double four_over_nh_squared, sa_inverse_cubed
    cdef double fe_ecc, mt

    cdef np.ndarray[DTYPE_t, ndim=1] frst_evo_pref = np.empty(n_eccs, dtype=DTYPE)
    # cdef double *frst_evo_pref = <double *>malloc(n_eccs * sizeof(double))

    cdef double _freq_pref = (1.0/(2.0*M_PI)) * sqrt(MY_NWTG)
    for ii in range(n_eccs):
        frst_evo_pref[ii] = _freq_pref / pow(sepa_evo[ii], 1.5)
        # frst_evo_pref[ii] = (1.0/(2.0*M_PI))*sqrt(NWTG)/pow(sepa_evo, 1.5)

    # iterate over redshifts Z
    for kk in range(n_redz):
        my_trapz_grid_weight(kk, n_redz, redz, kval)
        zterm = (1.0 + redz[kk])
        dc_mpc = dcom[kk]   # this is still in units of [Mpc]
        dc_cm = dc_mpc * MY_MPC
        dc_term = four_pi_c_mpc * pow(dc_mpc, 2)

        # iterate over mtot M
        for ii in range(n_mtot):
            if kk == 0:
                mt = pow(10.0, mtot_log10[ii])
                mtot[ii] = mt
            else:
                mt = mtot[ii]

            my_trapz_grid_weight(ii, n_mtot, mtot_log10, ival)

            weight_ik = ival[1] * kval[1] / (ival[0] * kval[0])

            for jj in range(n_mrat):
                my_trapz_grid_weight(jj, n_mrat, mrat, jval)
                weight = weight_ik * (jval[1] / jval[0])
                m1 = mt / (1.0 + mrat[jj])
                m2 = mt - m1
                mchirp = mt * pow(mrat[jj], three_fifths) / pow(1 + mrat[jj], six_fifths)

                for aa in range(nfreqs):
                    for bb in range(nharms):
                        nh = bb + 1
                        four_over_nh_squared = 4.0 / (nh * nh)
                        gwfr = gwfobs[aa] * zterm / nh

                        # rest-frame frequency corresponding to target observer-frame frequency of GW observations
                        sa = pow(kep_sa_term*mt / pow(gwfr, 2), one_third)
                        sa_inverse_cubed = pow(sa, -3)

                        # () scalar
                        ecc = np.interp(gwfr, frst_evo_pref*sqrt(mt), eccen_evo)
                        gne = gw_freq_dist_func__scalar_scalar(nh, ecc) * four_over_nh_squared
                        # ecc = 0.5
                        # gne = 0.25

                        fe_ecc = _gw_ecc_func(ecc)

                        # da/dt values are negative, get a positive rate
                        tau = GW_DADT_SEP_CONST * fe_ecc * m1 * m2 * (m1 + m2) * sa_inverse_cubed
                        # convert to timescale
                        tau = sa / - tau

                        # Calculate the GW spectral strain at each harmonic
                        #    see: [Amaro-seoane+2010 Eq.9]
                        hterm = pow(GW_SRC_CONST * mchirp * pow(2*mchirp*gwfr, two_third) / dc_cm, 2)
                        hterm = ndens[ii, jj, kk] * dc_term * zterm * tau * gne * hterm

                        gwb[aa, bb] += hterm * weight

    free(mtot)
    free(ival)
    free(jval)
    free(kval)
    # free(frst_evo_pref)
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

