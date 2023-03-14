"""Module for methods implemented in cython.

This module can be used by including the following code:
    ```
    import pyximport
    pyximport.install(language_level=3, setup_args={"include_dirs": np.get_include()})
    import holodeck.cyutils
    ```

python setup.py build_ext -i

"""

cimport cython
import numpy as np
cimport numpy as np
np.import_array()

# There is a special implementation of `scipy.special` for use with cython
cimport scipy.special.cython_special as sp_special

# from libc.stdio cimport printf
from libc.stdlib cimport malloc, free, qsort
# make sure to use c-native math functions instead of python/numpy
from libc.math cimport pow, sqrt, abs, M_PI

from cpython.pycapsule cimport PyCapsule_IsValid, PyCapsule_GetPointer
from numpy.random cimport bitgen_t
from numpy.random import PCG64
from numpy.random.c_distributions cimport random_poisson, random_normal


# DTYPE = np.float64
# ctypedef np.float64_t DTYPE_t

# ---- Define Parameters

cdef double ECCEN_ZERO_LIMIT = 1.0e-12   #: below this value, strains are calculated as circular

# ---- Define Constants

cdef double MY_NWTG = 6.6742999e-08
cdef double MY_SPLC = 29979245800.0
cdef double MY_MPC = 3.08567758e+24
cdef double MY_YR = 31557600.0
cdef double GW_DADT_SEP_CONST = - 64.0 * pow(MY_NWTG, 3) / 5.0 / pow(MY_SPLC, 5)
cdef double GW_SRC_CONST = 8.0 * pow(MY_NWTG, 5.0/3.0) * pow(M_PI, 2.0/3.0) / sqrt(10.0) / pow(MY_SPLC, 4.0)


# ====    Utility Functions    ====


cdef double bessel_recursive(int nn, double ne, double jn_m1, double jn_m2):
    """Recursive relation for calculating bessel functions

    J_n(x) = 2 * [(n-1) / x] * J_n-1(x) - J_n-2(x)

    NOTE: the recursive function fails when the argument `ne` is zero (divide by zero).  These
          cases should be caught manually and set to the known values.
          This happens at the level of calculating the frequency distribution function g(n,e).

    Parameters
    ----------
    int nn : order of the bessel function
    double ne : argument of the bessel function (in this case, product of order and eccentricity)
    double jn_m1 : the value of the Bessel function of order n-1 of the same argument
    double jn_m2 : the value of the Bessel function of order n-2 of the same argument

    Returns
    -------
    double jn : the value of the Bessel function of the desired order.

    """
    cdef double jn = (2*(nn-1) / ne) * jn_m1 - jn_m2
    return jn


cdef double _gw_ecc_func(double eccen):
    """Calculate the GW Hardening rate eccentricitiy dependence F(e).

    See [Peters1964]_ Eq. 5.6, or [EN2007]_ Eq. 2.3

    Parameters
    ----------
    eccen : double
        Eccentricity

    Returns
    -------
    fe : double
        The value of F(e).

    """
    cdef double e2 = eccen*eccen
    cdef double fe = (1.0 + (73.0/24.0)*e2 + (37.0/96.0)*e2*e2) / pow(1.0 - e2, 7.0/2.0)
    return fe


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void my_trapz_grid_weight(int index, int size, double[:] grid, double *rv):
    """Determine the trapezoid-rule weight and bin-width for the given grid point.

    Parameters
    ----------
    index : int,
        Index into the array of `grid` positions along the desired dimension.
    size : int,
        Size of the array of grid positions along the desired dimension.
    grid : double *
        Pointer to an array that specifies the location of grid points along the desired dimension.
    rv : double *
        Pointer to a (2,) array to store the output values.  The two values are:
        0: the (inverse) weight of this grid point,
        1: the grid-width for this grid point (i.e. 'dx').

    Returns
    -------
    None
        Note that output values are stored in the `rv` parameter.

    """
    # Left edge
    if index == 0:
        rv[0] = 2.0
        rv[1] = <double>(grid[1] - grid[0])   # i.e. grid[index+1] - grid[index]
        return

    # Right edge
    if index == size - 1:
        rv[0] = 2.0
        rv[1] = <double>(grid[index] - grid[index-1])
        return

    # center points
    rv[0] = 1.0
    # this is the same as the average of dx values on each side of the grid-point, i.e.:
    #     0.5 * ((grid[index] - grid[index-1]) + (grid[index+1] - grid[index]))
    rv[1] = 0.5 * <double>(grid[index+1] - grid[index-1])

    return


cdef double gw_freq_dist_func__scalar_scalar(int nn, double ee):
    """Calculate the GW frequency distribution function at the given harmonic and eccentricity, g(n,e).

    See [EN2007]_ Eq. 2.4

    NOTE: the recursive Bessel-function calculation is much faster, but fails when the argument (n*e) is zero.
        For this reason, zero arguments are manually detected and the known values of the function are returned.
        g(n, e=0.0) is 1.0 if n=2, and otherwise 0

    Parameters
    ----------
    nn : int,
        The harmonic to consider.
    ee : double,
        The eccentricity value.

    Returns
    -------
    gg : double,
        The value of g(n,e).

    """

    if ee < ECCEN_ZERO_LIMIT:
        if nn == 2:
            return 1.0

        return 0.0

    cdef double jn_m2, jn_m1, jn, jn_p1, jn_p2
    cdef double aa, bb, cc, gg

    cdef double ne = nn*ee
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


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void unravel(int idx, int[] shape, int *ii_out, int *jj_out):
    """Convert from a 1D/flattened index into a 2D pair of (unraveled) indices.

    NOTE: row-major / c-style ordering is assumed.  This is the numpy default.

    Parameters
    ----------
    idx : int,
        Index into a 1D array to be unraveled.
    shape : int *,
        Array specifying the 2D shape of the array being indexed.
    ii_out : int *,
        Pointer to the memory location to store the output value for the 0th dimension index (x/i).
    jj_out : int *,
        Pointer to the memory location to store the output value for the 1th dimension index (y/j).

    Returns
    -------
    None
        NOTE: return values are set to the `ii_out` and `jj_out` parameters.

    """
    # find the 0th index from the size of each 0th element, `shape[1]`
    ii_out[0] = idx // shape[1]
    # find the 1th index based on the remainder
    jj_out[0] = idx - ii_out[0] * shape[1]
    return


# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.nonecheck(False)
# @cython.cdivision(True)
# cdef void ravel(int ii, int jj, int[] shape, int *idx_out):
#     """Convert from a 2D pair of indices into a 1D (raveled) index.

#     NOTE: row-major / c-style ordering is assumed.  This is the numpy default.

#     Parameters
#     ----------
#     ii : int,
#         0th dimension index (i.e. x).
#     jj : int,
#         1th dimension index (i.e. y).
#     shape : int *,
#         Array specifying the 2D shape of the array being indexed.
#     idx_out : int *,
#         Pointer to the memory location to store the 1D/flattened index.

#     Returns
#     -------
#     None
#         NOTE: return values are set to the `idx_out` parameter.

#     """
#     idx_out[0] = ii * shape[1] + jj
#     return


"""Structure that stores both the index and value of a given array element to use in sorting.
"""
cdef struct sorter:
    int index
    double value


cdef int sort_compare(const void *a, const void *b) nogil:
    """Comparison function used by the `qsort` builtin method to perform an array-sort by index.

    Parameters
    ----------
    a : void *
        Pointer to first  element of comparison, should point to a `sorter` struct instance.
    b : void *
        Pointer to second element of comparison, should point to a `sorter` struct instance.

    Returns
    -------
    int
        -1 if a < b, +1 if b > a, and 0 otherwise

    """
    cdef sorter *a1 = <sorter *>a;
    cdef sorter *a2 = <sorter *>b;
    if ((a1[0]).value < (a2[0]).value):
        return -1
    elif ((a1[0]).value > (a2[0]).value):
        return 1
    else:
        return 0


cdef void argsort(double *values, int size, int **indices):
    """Find the indices that sort the given 1D array of double values.

    This is done using an array of `sorter` struct instances which store both index and value.

    Usage example:

        ```
        cdef double test[4]
        test[:] = [1.0, -2.3, 7.8, 0.0]

        cdef (int *)indices = <int *>malloc(4 * sizeof(int))
        argsort(test, 4, &indices)
        ```

    """
    cdef (sorter *)testers = <sorter *>malloc(size * sizeof(sorter))
    cdef int ii
    for ii in range(size):
        testers[ii].index = ii
        testers[ii].value = values[ii]

    qsort(testers, size, sizeof(testers[0]), sort_compare)
    for ii in range(size):
        indices[0][ii] = testers[ii].index

    return


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double interp_at_index(int idx, double xnew, double[:] xold, double[:] yold):
    """Perform linear interpolation at the given index in a pair of arrays.

    Parameters
    ----------
    idx : int
        Index in the arrays specifying the left reference value to interpolate between, with `idx+1`
        giving the right value.
    xnew : double
        The independent (x) value to interpolate to.
    xold : double *
        Array of x-values giving the independent variables where the functions are evaluated.
    yold : double *
        Array of y-values giving the dependent variable (function values) of array.

    Returns
    -------
    ynew : double
        Interpolated function value.

    """
    cdef double ynew = yold[idx] + (yold[idx+1] - yold[idx])/(xold[idx+1] - xold[idx]) * (xnew - xold[idx])
    return ynew


def sam_calc_gwb_single_eccen(ndens, mtot_log10, mrat, redz, dcom, gwfobs, sepa_evo, eccen_evo, nharms=100):
    """Pure-python wrapper for the SAM eccentric GWB calculation method.  See: `_sam_calc_gwb_single_eccen()`.
    """
    return _sam_calc_gwb_single_eccen(ndens, mtot_log10, mrat, redz, dcom, gwfobs, sepa_evo, eccen_evo, nharms)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double[:, :] _sam_calc_gwb_single_eccen(
    double[:, :, :] ndens,
    double[:] mtot_log10,
    double[:] mrat,
    double[:] redz,
    double[:] dcom,
    double[:] gwfobs,
    double[:] sepa_evo_in,
    double[:] eccen_evo_in,
    int nharms
):
    """Calculate the GWB from an eccentric SAM evolution model.

    This function uses the precomputed binary-evolution of eccentricities over some range of separations.
    This assumes a single eccentricity evolution e(a) for all binaries.
    Redshifts are assumed to stay constant during this evolution.

    h_c^2 = (dn/dz) * h_{s,n}^2 * 4*pi*c * d_c^2 * (1+z) * tau

    h_{s,n}^2 = (32 / 5) * c^-8 * (2/n)^2 * (G Mchirp)^(10/3) * d_c^-2 * (2*pi*forb_r)^(4/3) * g(n,e)
              = GW_SRC_CONST^2 * (2/n)^2 * Mchirp^(10/3) * d_c^-2 * forb_r^(4/3) * g(n,e)


    Parameters
    ----------
    ndens : (double ***)
        3D array specifying the SAM number-density of binaries in each bin, in total-mass, mass-ratio, and redshift.
        This is typically `sam.static_binary_density`, corresponding to 'd^3 n / [dlog10M dq dz]' in units of [Mpc^-3].
    mtot_log10 : (double *)
        Array of log10 values of the total-mass (in grams) grid edges in the SAM model.  i.e. `log10(sam.mtot)`.
    mrat : (double *)
        Array of mass-ratio grid edges in the SAM model.  i.e. `sam.mrat`.
    redz : (double *)
        Array of values for the redshift grid edges in the SAM model.  i.e. `sam.redz`.
    dcom : (double *)
        Comoving distances in units of [Mpc] corresponding to each `redz` value.
    gwfobs : (double *)
        Array of GW observer-frame frequencies at which to evaluate the GWB.
    sepa_evo_in : (double *)
        The values of binary separation in units of [cm] at which the binaries have been evolved.
    eccen_evo_in : (double *)
        The values of binary eccentricity at each separation at which the binaries have been evolved.
    nharms : int,
        Number of harmonics at which to calculate GW strains.

    Returns
    -------
    gwb : (double **)
        2D array giving the characteristic-strain _squared_, at each observer-frame frequency and each harmonic.
        Shape is (nfreqs, nharms), where `nfreqs` is the length of the `gwfobs` parameters, and `nharms` is given
        as an input parameter directly.
        The characteristic strain spectrum, h_c(f) is then `np.sqrt(gwb.sum(axis=1))`; i.e. the sum in quadrature of
        strain at each harmonic.

    """

    # Initialize sizes and shapes
    cdef int nfreqs = len(gwfobs)
    cdef int n_mtot = len(mtot_log10)
    cdef int n_mrat = len(mrat)
    cdef int n_redz = len(redz)
    cdef int n_eccs = len(sepa_evo_in)
    cdef int num_freq_harm = nfreqs * nharms
    cdef (int *)shape = <int *>malloc(2 * sizeof(int))
    shape[0] = nfreqs
    shape[1] = nharms

    # Declare variables used later
    cdef int ii, nh, jj, kk, aa, bb, ff, ecc_idx, ecc_idx_beg, ii_mm, kk_zz
    cdef double m1, m2, mchirp, sa, qq, tau, num_pois
    cdef double gwfr, zterm, dc_cm, dc_mpc, dc_term, gne, hterm, number_term, number_term_pref
    cdef double weight_ik, weight, volume_ik, volume, four_over_nh_squared, sa_fourth
    cdef double fe_ecc, mt, mt_sqrt, nd, hterm_pref, frst_evo_lo, frst_evo_hi

    # Initialize constants
    cdef double four_pi_c_mpc = 4 * M_PI * (MY_SPLC / MY_MPC)
    cdef double kep_sa_term = MY_NWTG / pow(2.0*M_PI, 2)
    cdef double one_third = 1.0 / 3.0
    cdef double two_third = 2.0 / 3.0
    cdef double four_third = 4.0 / 3.0
    cdef double three_fifths = 3.0 / 5.0
    cdef double six_fifths = 6.0 / 5.0

    # Initialize arrays
    cdef np.ndarray[np.double_t, ndim=2] gwb = np.zeros((nfreqs, nharms))
    cdef double *mtot = <double *>malloc(n_mtot * sizeof(double))
    cdef double *ival = <double *>malloc(2 * sizeof(double))
    cdef double *jval = <double *>malloc(2 * sizeof(double))
    cdef double *kval = <double *>malloc(2 * sizeof(double))
    cdef double *sepa_evo = <double *>malloc(n_eccs * sizeof(double))
    cdef double *eccen_evo = <double *>malloc(n_eccs * sizeof(double))
    cdef double _freq_pref = (1.0/(2.0*M_PI)) * sqrt(MY_NWTG)
    cdef double *frst_evo_pref = <double *>malloc(n_eccs * sizeof(double))
    # Convert from numpy arrays to c-arrays (for possible speed improvements)
    for ii in range(n_eccs):
        sepa_evo[ii] = <double>sepa_evo_in[ii]
        eccen_evo[ii] = <double>eccen_evo_in[ii]
        # calculate the prefactor (i.e. everything except the mass) for kepler's law
        frst_evo_pref[ii] = _freq_pref / pow(sepa_evo[ii], 1.5)

    # Calculate all of the frequency harmonics (flattened) that are needed
    cdef (double *)freq_harms = <double *>malloc(num_freq_harm * sizeof(double))
    for ii in range(num_freq_harm):
        # convert from 1D index to 2D grid of (F, H) frequencies and harmonics
        unravel(ii, shape, &aa, &bb)
        # calculate the n = bb+1 harmonic
        freq_harms[ii] = gwfobs[aa] / (bb + 1)

    # Find the indices by which to sort the frequency harmonics (flattened)
    cdef (int *)sorted_index = <int *>malloc(num_freq_harm * sizeof(int))
    argsort(freq_harms, num_freq_harm, &sorted_index)

    # iterate over redshifts Z
    for kk in range(n_redz):
        # fill `kval` with the weight of this grid point (kval[0]), and the grid-width (kval[1])
        my_trapz_grid_weight(kk, n_redz, redz, kval)
        zterm = (1.0 + redz[kk])
        dc_mpc = dcom[kk]   # this is still in units of [Mpc]
        dc_cm = dc_mpc * MY_MPC  # convert to [cm]
        dc_term = four_pi_c_mpc * pow(dc_mpc, 2)

        # iterate over mtot M
        ecc_idx_beg = 0   # we will keep track of evolution/eccentricity indices for our target
                          # frequencies to make suring the arrays faster
        # iterate over total masses in reverse, so that's we're always going to increasing frequencies
        for ii_mm in range(n_mtot):
            # convert from forward to backward indices
            ii = n_mtot - 1 - ii_mm

            # convert from log10 to regular total masses, only the first time through the loop
            if kk == 0:
                mt = pow(10.0, mtot_log10[ii])
                mtot[ii] = mt
            else:
                mt = mtot[ii]

            # calculate some needed quantities
            mt_sqrt = sqrt(mt)
            kep_sa_mass_term = kep_sa_term * mt
            # fill `ival` with weight and grid-width in the mtot dimension
            my_trapz_grid_weight(ii, n_mtot, mtot_log10, ival)

            # precalculate some of the weighting factors over the 2D we have so far
            weight_ik = ival[1] * kval[1] / (ival[0] * kval[0])

            # iterate over mass ratios
            for jj in range(n_mrat):
                # fill `jval` with weight and grid-width in the mrat dimension
                my_trapz_grid_weight(jj, n_mrat, mrat, jval)
                # calculate the weight factor for this grid-cell

                weight = weight_ik * (jval[1] / jval[0])

                # convert for mtot, mrat to m1, m2 s.t. m2 <= m1    [grams]
                m1 = mt / (1.0 + mrat[jj])
                m2 = mt - m1
                # calculate chirp-mass [grams]
                mchirp = mt * pow(mrat[jj], three_fifths) / pow(1 + mrat[jj], six_fifths)

                # n_c * (4*pi*c*d_c^2) * (1 + z)
                hterm_pref = ndens[ii, jj, kk] * dc_term * zterm
                # GW_SRC_CONST^2 * 2^(4/3) * Mc^(10/3) * dc^-2 * n_c * (4*pi*d_c^2) * (1 + z)
                hterm_pref *= pow(GW_SRC_CONST * mchirp * pow(2.0*mchirp, two_third) / dc_cm, 2)

                ecc_idx = ecc_idx_beg
                for ff in range(num_freq_harm):
                    unravel(sorted_index[ff], shape, &aa, &bb)

                    nh = bb + 1
                    four_over_nh_squared = 4.0 / (nh * nh)
                    gwfr = gwfobs[aa] * zterm / nh

                    # rest-frame frequency corresponding to target observer-frame frequency of GW observations
                    sa = pow(kep_sa_mass_term / pow(gwfr, 2), one_third)
                    sa_fourth = pow(sa, 4)

                    # ---- Interpolate eccentricity to this frequency

                    # try to get `ecc_idx` such that   frst[idx] < gwfr < frst[idx+1]
                    frst_evo_lo = frst_evo_pref[ecc_idx] * mt_sqrt
                    frst_evo_hi = frst_evo_pref[ecc_idx+1] * mt_sqrt
                    while (gwfr > frst_evo_hi) & (ecc_idx < n_eccs - 2):
                        frst_evo_lo = frst_evo_hi
                        ecc_idx += 1
                        frst_evo_hi = frst_evo_pref[ecc_idx+1] * mt_sqrt

                    if jj == 0 and ff == 0:
                        ecc_idx_beg = ecc_idx

                    # if `gwfr` is lower than lowest evolution point, continue to next frequency
                    if (gwfr < frst_evo_lo):
                        continue

                    # if `gwfr` is > highest evolution point, also be true for all following frequencies, so break
                    if (gwfr > frst_evo_hi):
                        break

                    # calculate slope M
                    ecc = ((eccen_evo[ecc_idx+1] - eccen_evo[ecc_idx])/(frst_evo_hi - frst_evo_lo))
                    # y = y_0 + M * dx
                    ecc = eccen_evo[ecc_idx] + (gwfr - frst_evo_lo) * ecc

                    # ---- Calculate GWB contribution with this eccentricity

                    gne = gw_freq_dist_func__scalar_scalar(nh, ecc)

                    fe_ecc = _gw_ecc_func(ecc)
                    # da/dt values are negative, convert to a positive timescale
                    tau = - sa_fourth / (GW_DADT_SEP_CONST * fe_ecc * m1 * m2 * mt)

                    # Calculate the GW spectral strain at each harmonic
                    #    see: [Amaro-seoane+2010 Eq.9]
                    # GW_SRC_CONST^2 * 2^(4/3) * Mc^(10/3) * dc^-2 * n_c * (4*pi*c*d_c^2) * (1 + z)
                    #     tau * gne * (2/n)^2 * forb_r^(4/3)
                    hterm = hterm_pref * tau * gne * four_over_nh_squared * pow(gwfr, four_third)
                    gwb[aa, bb] += hterm * weight

    free(shape)
    free(sepa_evo)
    free(eccen_evo)
    free(frst_evo_pref)
    free(freq_harms)
    free(sorted_index)
    free(mtot)
    free(ival)
    free(jval)
    free(kval)
    return gwb


def sam_calc_gwb_single_eccen_discrete(ndens, mtot_log10, mrat, redz, dcom, gwfobs, sepa_evo, eccen_evo, nharms, nreals):
    """Pure-python wrapper for the SAM eccentric GWB calculation method.  See: `_sam_calc_gwb_single_eccen()`.
    """
    return _sam_calc_gwb_single_eccen_discrete(ndens, mtot_log10, mrat, redz, dcom, gwfobs, sepa_evo, eccen_evo, nharms, nreals)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double[:, :, :] _sam_calc_gwb_single_eccen_discrete(
    double[:, :, :] ndens,
    double[:] mtot_log10,
    double[:] mrat,
    double[:] redz,
    double[:] dcom,
    double[:] gwfobs,
    double[:] sepa_evo_in,
    double[:] eccen_evo_in,
    int nharms,
    int nreals
):
    """Calculate the GWB from an eccentric SAM evolution model.

    This function uses the precomputed binary-evolution of eccentricities over some range of separations.
    This assumes a single eccentricity evolution e(a) for all binaries.
    Redshifts are assumed to stay constant during this evolution.

    Parameters
    ----------
    ndens : (double ***)
        3D array specifying the SAM number-density of binaries in each bin, in total-mass, mass-ratio, and redshift.
        This is typically `sam.static_binary_density`, corresponding to 'd^3 n / [dlog10M dq dz]' in units of [Mpc^-3].
    mtot_log10 : (double *)
        Array of log10 values of the total-mass (in grams) grid edges in the SAM model.  i.e. `log10(sam.mtot)`.
    mrat : (double *)
        Array of mass-ratio grid edges in the SAM model.  i.e. `sam.mrat`.
    redz : (double *)
        Array of values for the redshift grid edges in the SAM model.  i.e. `sam.redz`.
    dcom : (double *)
        Comoving distances in units of [Mpc] corresponding to each `redz` value.
    gwfobs : (double *)
        Array of GW observer-frame frequencies at which to evaluate the GWB.
    sepa_evo_in : (double *)
        The values of binary separation in units of [cm] at which the binaries have been evolved.
    eccen_evo_in : (double *)
        The values of binary eccentricity at each separation at which the binaries have been evolved.
    nharms : int,
        Number of harmonics at which to calculate GW strains.

    Returns
    -------
    gwb : (double **)
        2D array giving the characteristic-strain _squared_, at each observer-frame frequency and each harmonic.
        Shape is (nfreqs, nharms), where `nfreqs` is the length of the `gwfobs` parameters, and `nharms` is given
        as an input parameter directly.
        The characteristic strain spectrum, h_c(f) is then `np.sqrt(gwb.sum(axis=1))`; i.e. the sum in quadrature of
        strain at each harmonic.

    """

    # Initialize sizes and shapes
    cdef int nfreqs = len(gwfobs)
    cdef int n_mtot = len(mtot_log10)
    cdef int n_mrat = len(mrat)
    cdef int n_redz = len(redz)
    cdef int n_eccs = len(sepa_evo_in)
    cdef int num_freq_harm = nfreqs * nharms
    cdef (int *)shape = <int *>malloc(2 * sizeof(int))
    shape[0] = nfreqs
    shape[1] = nharms

    # Setup random number generator from numpy library
    cdef bitgen_t *rng
    cdef const char *capsule_name = "BitGenerator"
    capsule = PCG64().capsule
    # Cast the pointer
    rng = <bitgen_t *> PyCapsule_GetPointer(capsule, capsule_name)

    # Declare variables used later
    cdef int ii, nh, jj, kk, aa, bb, ff, rr, ecc_idx, ecc_idx_beg, ii_mm, kk_zz
    cdef double m1, m2, mchirp, sa, qq, tau
    cdef double gwfr, zterm, dc_cm, dc_mpc, dc_term, gne, hterm, number_term
    cdef double weight_ik, weight, four_over_nh_squared, sa_fourth
    cdef double fe_ecc, mt, mt_sqrt, nd, hterm_pref, frst_evo_lo, frst_evo_hi

    # Initialize constants
    cdef double four_pi_c_mpc = 4 * M_PI * (MY_SPLC / MY_MPC)
    cdef double kep_sa_term = MY_NWTG / pow(2.0*M_PI, 2)
    cdef double one_third = 1.0 / 3.0
    cdef double two_third = 2.0 / 3.0
    cdef double four_third = 4.0 / 3.0
    cdef double three_fifths = 3.0 / 5.0
    cdef double six_fifths = 6.0 / 5.0

    # Initialize arrays
    cdef np.ndarray[np.double_t, ndim=3] gwb = np.zeros((nfreqs, nharms, nreals))
    cdef double *mtot = <double *>malloc(n_mtot * sizeof(double))
    cdef double *ival = <double *>malloc(2 * sizeof(double))
    cdef double *jval = <double *>malloc(2 * sizeof(double))
    cdef double *kval = <double *>malloc(2 * sizeof(double))
    cdef double *sepa_evo = <double *>malloc(n_eccs * sizeof(double))
    cdef double *eccen_evo = <double *>malloc(n_eccs * sizeof(double))
    cdef double _freq_pref = (1.0/(2.0*M_PI)) * sqrt(MY_NWTG)
    cdef double *frst_evo_pref = <double *>malloc(n_eccs * sizeof(double))
    # Convert from numpy arrays to c-arrays (for possible speed improvements)
    for ii in range(n_eccs):
        sepa_evo[ii] = <double>sepa_evo_in[ii]
        eccen_evo[ii] = <double>eccen_evo_in[ii]
        # calculate the prefactor (i.e. everything except the mass) for kepler's law
        frst_evo_pref[ii] = _freq_pref / pow(sepa_evo[ii], 1.5)

    # Calculate all of the frequency harmonics (flattened) that are needed
    cdef (double *)freq_harms = <double *>malloc(num_freq_harm * sizeof(double))
    for ii in range(num_freq_harm):
        # convert from 1D index to 2D grid of (F, H) frequencies and harmonics
        unravel(ii, shape, &aa, &bb)
        # calculate the n = bb+1 harmonic
        freq_harms[ii] = gwfobs[aa] / (bb + 1)

    # Find the indices by which to sort the frequency harmonics (flattened)
    cdef (int *)sorted_index = <int *>malloc(num_freq_harm * sizeof(int))
    argsort(freq_harms, num_freq_harm, &sorted_index)

    # iterate over redshifts Z
    for kk in range(n_redz):
        # fill `kval` with the weight of this grid point (kval[0]), and the grid-width (kval[1])
        my_trapz_grid_weight(kk, n_redz, redz, kval)
        zterm = (1.0 + redz[kk])
        dc_mpc = dcom[kk]   # this is still in units of [Mpc]
        dc_cm = dc_mpc * MY_MPC  # convert to [cm]
        dc_term = four_pi_c_mpc * pow(dc_mpc, 2)

        # iterate over mtot M
        ecc_idx_beg = 0   # we will keep track of evolution/eccentricity indices for our target
                          # frequencies to make suring the arrays faster
        # iterate over total masses in reverse, so that's we're always going to increasing frequencies
        for ii_mm in range(n_mtot):
            # convert from forward to backward indices
            ii = n_mtot - 1 - ii_mm

            # convert from log10 to regular total masses, only the first time through the loop
            if kk == 0:
                mt = pow(10.0, mtot_log10[ii])
                mtot[ii] = mt
            else:
                mt = mtot[ii]

            # calculate some needed quantities
            mt_sqrt = sqrt(mt)
            kep_sa_mass_term = kep_sa_term * mt
            # fill `ival` with weight and grid-width in the mtot dimension
            my_trapz_grid_weight(ii, n_mtot, mtot_log10, ival)

            # precalculate some of the weighting factors over the 2D we have so far
            volume_ik = ival[1] * kval[1]
            weight_ik = ival[0] * kval[0]

            # iterate over mass ratios
            for jj in range(n_mrat):
                # fill `jval` with weight and grid-width in the mrat dimension
                my_trapz_grid_weight(jj, n_mrat, mrat, jval)
                # calculate the weight factor for this grid-cell
                # weight = weight_ik * (jval[1] / jval[0])

                volume = volume_ik * jval[1]
                weight = weight_ik * jval[0]

                # convert for mtot, mrat to m1, m2 s.t. m2 <= m1    [grams]
                m1 = mt / (1.0 + mrat[jj])
                m2 = mt - m1
                # calculate chirp-mass [grams]
                mchirp = mt * pow(mrat[jj], three_fifths) / pow(1 + mrat[jj], six_fifths)

                # n_c * (4*pi*c*d_c^2) * (1 + z)
                number_term_pref = ndens[ii, jj, kk] * dc_term * zterm
                # GW_SRC_CONST^2 * 2^(4/3) * Mc^(10/3) * dc^-2
                hterm_pref = pow(GW_SRC_CONST * mchirp * pow(2.0*mchirp, two_third) / dc_cm, 2)

                ecc_idx = ecc_idx_beg
                for ff in range(num_freq_harm):
                    unravel(sorted_index[ff], shape, &aa, &bb)

                    nh = bb + 1
                    four_over_nh_squared = 4.0 / (nh * nh)
                    gwfr = gwfobs[aa] * zterm / nh

                    # rest-frame frequency corresponding to target observer-frame frequency of GW observations
                    sa = pow(kep_sa_mass_term / pow(gwfr, 2), one_third)
                    sa_fourth = pow(sa, 4)

                    # ---- Interpolate eccentricity to this frequency

                    # try to get `ecc_idx` such that   frst[idx] < gwfr < frst[idx+1]
                    frst_evo_lo = frst_evo_pref[ecc_idx] * mt_sqrt
                    frst_evo_hi = frst_evo_pref[ecc_idx+1] * mt_sqrt
                    while (gwfr > frst_evo_hi) & (ecc_idx < n_eccs - 2):
                        frst_evo_lo = frst_evo_hi
                        ecc_idx += 1
                        frst_evo_hi = frst_evo_pref[ecc_idx+1] * mt_sqrt

                    if jj == 0 and ff == 0:
                        ecc_idx_beg = ecc_idx

                    # if `gwfr` is lower than lowest evolution point, continue to next frequency
                    if (gwfr < frst_evo_lo):
                        continue

                    # if `gwfr` is > highest evolution point, also be true for all following frequencies, so break
                    if (gwfr > frst_evo_hi):
                        break

                    # calculate slope M
                    ecc = (eccen_evo[ecc_idx+1] - eccen_evo[ecc_idx]) / (frst_evo_hi - frst_evo_lo)
                    # y = y_0 + M * dx
                    ecc = eccen_evo[ecc_idx] + (gwfr - frst_evo_lo) * ecc

                    # ---- Calculate GWB contribution with this eccentricity

                    gne = gw_freq_dist_func__scalar_scalar(nh, ecc)

                    fe_ecc = _gw_ecc_func(ecc)
                    # da/dt values are negative, convert to a positive timescale
                    tau = - sa_fourth / (GW_DADT_SEP_CONST * fe_ecc * m1 * m2 * mt)

                    # Calculate the GW spectral strain at each harmonic
                    #    see: [Amaro-seoane+2010 Eq.9]

                    # n_c * (4*pi*c*d_c^2) * (1 + z) * tau * dM*dq*dz
                    number_term = number_term_pref * tau * volume

                    # GW_SRC_CONST^2 * 2^(4/3) * Mc^(10/3) * gne * (2/n)^2 * forb_r^(4/3) * dc^-2
                    hterm = hterm_pref * gne * four_over_nh_squared * pow(gwfr, four_third)

                    for rr in range(nreals):
                        # npy_int64 random_poisson(bitgen_t *bitgen_state, double lam)
                        num_pois = <double>random_poisson(rng, number_term)

                        # GW_SRC_CONST^2 * 2^(4/3) * Mc^(10/3) * gne * (2/n)^2 * forb_r^(4/3) * dc^-2 *
                        #     n_c * (4*pi*c*d_c^2) * (1 + z) * tau * dM*dq*dz * trapz-weight
                        gwb[aa, bb, rr] += hterm * num_pois / weight

    free(shape)
    free(sepa_evo)
    free(eccen_evo)
    free(frst_evo_pref)
    free(freq_harms)
    free(sorted_index)
    free(mtot)
    free(ival)
    free(jval)
    free(kval)
    return gwb


def sam_poisson_gwb(dist, hc2, nreals, normal_threshold=1e10):
    return _sam_poisson_gwb(np.array(dist.shape), dist, hc2, nreals, long(normal_threshold))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double[:, :] _sam_poisson_gwb(
    long[:] shape, double[:, :, :, :] dist, double[:, :, :, :] hc2, int nreals, long thresh
):
    cdef int nm = shape[0]
    cdef int nq = shape[1]
    cdef int nz = shape[2]
    cdef int nf = shape[3]
    cdef int ii, jj, kk, ff, rr
    cdef double num, bin_hc2, bin_num

    # Setup random number generator from numpy library
    cdef bitgen_t *rng
    cdef const char *capsule_name = "BitGenerator"
    capsule = PCG64().capsule
    # Cast the pointer
    rng = <bitgen_t *> PyCapsule_GetPointer(capsule, capsule_name)

    cdef np.ndarray[np.double_t, ndim=2] gwb = np.zeros((nf, nreals))
    for ii in range(nm):
        for jj in range(nq):
            for kk in range(nz):
                for ff in range(nf):
                    bin_num = dist[ii, jj, kk, ff]
                    bin_hc2 = hc2[ii, jj, kk, ff]
                    if bin_num > thresh:
                        bin_std = sqrt(bin_num)
                        for rr in range(nreals):
                            num = <double>random_normal(rng, bin_num, bin_std)
                            gwb[ff, rr] += num * bin_hc2
                    else:
                        for rr in range(nreals):
                            num = <double>random_poisson(rng, bin_num)
                            gwb[ff, rr] += num * bin_hc2

    return gwb

def ss_poisson_hc(number, h2fdf, nreals, normal_threshold=1e10):
    shape = np.array(number.shape)
    hc_ss = np.zeros((shape[3], nreals)) # shape (F,R)
    hc_bg = np.zeros((shape[3], nreals)) # shape (F,R)
    ssidx = np.zeros((3, shape[3], nreals), dtype=int) # shape (F,R)
    _ss_poisson_hc(shape, number, h2fdf, nreals, long(normal_threshold),
        &hc_ss, &hc_bg, &ssidx)
    return hc_ss, hc_bg, ssidx
    
  

# why is the shape a long? that doesn't seem like it'd need to be
@cython.boundscheck(True)
@cython.wraparound(True)
@cython.nonecheck(True)
@cython.cdivision(True)
# _ss_poisson_hc(long[:] shape, double[:,:,:,:] number, double[:,:,:,:] h2fdf,
#     int nreals, long thresh, *np.ndarray[np.double_t, ndim=2] hc_ss, 
#     *np.ndarray[np.double_t, ndim=2], *np.ndarray[np.int, ndim=3]): # using pointers
cdef void _ss_poisson_hc(long[:] shape, double[:,:,:,:] number, double[:,:,:,:] h2fdf,
    int nreals, long thresh, np.ndarray[np.double_t, ndim=2] *hc_ss, 
    np.ndarray[np.double_t, ndim=2] *hc_bg, np.ndarray[np.int, ndim=3] *ssidx):

    cdef int M = shape[0]
    cdef int Q = shape[1]
    cdef int Z = shape[2]
    cdef int F = shape[3]
    cdef int R = nreals

    # Setup random number generator from numpy library
    cdef bitgen_t *rng
    cdef const char *capsule_name = "BitGenerator"
    capsule = PCG64().capsule
    # Cast the pointer
    rng = <bitgen_t *> PyCapsule_GetPointer(capsule, capsule_name)

    cdef int mm, qq, zz, ff, rr, m_max, q_max, z_max
    cdef double max, num

    # passed as pointers:
    #   np.ndarray[np.double_t, ndim=2] hc_ss = np.zeros((F, R))
    #   np.ndarray[np.double_t, ndim=2] hc_bg = np.zeros((F, R))
    #   np.ndarray[np.double_t, ndim=3] ssidx = np.zeros((3, F, R))
    # defining first time here:



    # h2fdf = hs^2 * f/df
    # hc_ss = sqrt(max hsfdf)
    # hc_bg = sqrt(sum hsfdf*number in bin - max)
    for rr in range(R):
        for ff in range(F):
            # find the max
            max = 0 # max h2fdf
            sum = 0 # sum of all h2fdf*number
            for mm in range(M):
                for qq in range(Q):
                    for zz in range(Z):
                        num = number[mm,qq,zz,ff] # this bin's number
                        cur = h2fdf[mm,qq,zz,ff] # this bin's h2fdf
                        if (num>thresh):
                            std = sqrt(num)
                            num = <double>random_normal(rng, num, std)  
                        else:
                            num = <double>random_poisson(rng, num)
                        # check if max
                        if (num>0 and cur>max):
                            max = cur # update max if cur is larger
                            m_max = mm
                            q_max = qq
                            z_max = zz
                        sum += cur*num
            if (max==0): 
                print('No sources found at %dth frequency' % ff)
            sum -= max # subtract single source from the gwb
            hc_bg[ff][rr] = sqrt(sum)
            hc_ss[ff][ff] = sqrt(max)
            ssidx[:, ff, rr] = m_max, q_max, z_max

    return 
    


# idea: could sort hsamp and then random sample until a bin has val >=1, 
# which would become hsamp. Then add the rest up for the background.
                    
                        

