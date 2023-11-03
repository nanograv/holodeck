"""Module for methods implemented in cython.

To use and load this module, you will need to build the cython extension using:

$ python setup.py build_ext -i

from the holodeck root directory (containing the `setup.py` file).

And you still need to install holodeck in develop mode, using:

$ python setup.py develop

"""

cimport cython
import numpy as np
cimport numpy as np
np.import_array()

# There is a special implementation of `scipy.special` for use with cython
cimport scipy.special.cython_special as sp_special

from libc.stdio cimport printf
from libc.stdlib cimport malloc, free, qsort
# make sure to use c-native math functions instead of python/numpy
from libc.math cimport pow, sqrt, abs, M_PI, NAN, cos, sin

from cpython.pycapsule cimport PyCapsule_IsValid, PyCapsule_GetPointer
from numpy.random cimport bitgen_t
from numpy.random import PCG64
from numpy.random.c_distributions cimport random_poisson, random_normal


# DTYPE = np.float64 # define as this type
# ctypedef np.float64_t DTYPE_t # accept in cython function as this type

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
    rv[1] = 0.5 * (<double>(grid[index+1] - grid[index-1]))

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
cdef double _interp_between_vals(double xnew, double xl, double xr, double yl, double yr):
    cdef double ynew = yl + (yr - yl) * (xnew - xl) / (xr - xl)
    return ynew


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
    # cdef double ynew = yold[idx] + (yold[idx+1] - yold[idx])/(xold[idx+1] - xold[idx]) * (xnew - xold[idx])
    # return ynew
    return _interp_between_vals(xnew, xold[idx], xold[idx+1], yold[idx], yold[idx+1])


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


def ss_bg_hc(number, h2fdf, nreals, normal_threshold=1e10):
    """ Calculates the characteristic strain from loud single sources and a background of all other sources.

    Parameters
    ----------
    number : [M, Q, Z, F] ndarray
        number in each bin
    h2fdf : [M, Q, Z, F] ndarray
        strain squared x frequency / frequency bin width for each bin
    nreals
        number of realizations

    Returns
    -------
    hc2ss : (F, R) Ndarray of scalars
    hc2bg : (F, R) Ndarray of scalars
    ssidx : (3, F, R) Ndarray of ints
        Index of the loudest single source, -1 if there are none at the frequency/realization.

    """

    cdef long[:] shape = np.array(number.shape)
    F = shape[3]
    R = nreals
    cdef np.ndarray[np.double_t, ndim=2] hc2ss = np.zeros((F,R))
    cdef np.ndarray[np.double_t, ndim=2] hc2bg = np.zeros((F,R))
    cdef np.ndarray[np.longlong_t, ndim=3] ssidx = np.zeros((3,F,R), dtype=int)
    _ss_bg_hc(shape, h2fdf, number, nreals, normal_threshold,
                hc2ss, hc2bg, ssidx)
    return hc2ss, hc2bg, ssidx

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void _ss_bg_hc(long[:] shape, double[:,:,:,:] h2fdf, double[:,:,:,:] number,
            long nreals, long thresh,
            double[:,:] hc2ss, double[:,:] hc2bg, long[:,:,:] ssidx):
    """
    Calculates the characteristic strain from loud single sources and a background of all other sources.

    Parameters
    ----------
    shape : long[:] array
        shape of number, [M, Q, Z, F]
    number : double[:,:,:,:] array
        number per bin
    h2fdf : double[:,:,:,:] array
        strain amplitude squared * f/Delta f for a single source in each bin.
    nreals : int
        number of realizations.
    hc2ss : double[:,:] array
        (memory address of) single source characteristic strain squared array.
    hc2bg : double[:,:] array
        (memory address of) background characteristic strain squared array.
    ssidx : [:,:,:] long array
        (memory address of) array for indices of max strain bins.
    bgpar :
        (memory address of) array of effective average parameters

    Returns
    -------
    void
    updated via memory address: hc2ss, hc2bg, ssidx
    """

    cdef int M = shape[0]
    cdef int Q = shape[1]
    cdef int Z = shape[2]
    cdef int F = shape[3]
    cdef int R = nreals

    cdef int mm, qq, zz, ff, rr, m_max, q_max, z_max
    cdef double max, num, sum


    # Setup random number generator from numpy library
    cdef bitgen_t *rng
    cdef const char *capsule_name = "BitGenerator"
    capsule = PCG64().capsule
    # Cast the pointer
    rng = <bitgen_t *> PyCapsule_GetPointer(capsule, capsule_name)

    for rr in range(R):
        for ff in range(F):
            max=0
            sum=0
            m_max=-1
            q_max=-1
            z_max=-1
            for mm in range(M):
                for qq in range(Q):
                    for zz in range(Z):
                        num = number[mm,qq,zz,ff]
                        cur = h2fdf[mm,qq,zz,ff]
                        if (num>thresh):
                            std = sqrt(num)
                            num = <double>random_normal(rng, num, std)
                        else:
                            num = <double>random_poisson(rng, num)
                        if(cur > max and num > 0):
                            max = h2fdf[mm,qq,zz,ff]
                            m_max = mm # -1 if no single sources
                            q_max = qq # -1 if no single sources
                            z_max = zz # -1 if no single sources
                        sum += num * cur
            hc2ss[ff,rr] = max
            hc2bg[ff,rr] = sum - max
            # ssidx[:,ff,rr] = m_max, q_max, z_max
            ssidx[0,ff,rr] = m_max
            ssidx[1,ff,rr] = q_max
            ssidx[2,ff,rr] = z_max
    # still need to sqrt and sum! (or do this back in python)

    return

# I also need to pass the edges to calculate the avged ones
def ss_bg_hc_and_par(number, h2fdf, nreals, mt, mr, rz, normal_threshold=1e10):
    """
    Calculates the characteristic strain from loud single sources and a background of all other sources.

    Parameters
    ------------------------
    number : [M, Q, Z, F] NDarray
        number in each bin
    h2fdf : [M, Q, Z, F] NDarray
        Strain amplitude squared x frequency / frequency bin width for each bin.
    nreals
        Number of realizations.
    mt : (M,) 1Darray of scalars
        Total masses, M, of each bin center.
    mr : (Q,) 1Darray of scalars
        Mass ratios, q, of each bin center.
    rz : (Z,) 1Darray of scalars
        Redshifts, z, of each bin center.

    Returns
    --------------------------
    hc2ss : (F, R) Ndarray of scalars
        Char strain squared of the loudest single sources.
    hc2bg : (F, R) Ndarray of scalars
        Char strain squared of the background.
    ssidx : (3, F, R) NDarray of ints
        Indices of the loudest single sources. -1 if there are
        no single sources at that frequency/realization.
    bgpar : (3, F, R) NDarray of scalars
        Average effective M, q, z parameters of the background.
    sspar : (3, F, R) NDarray of scalars
        M, q, z parameters of the loudest single sources.
    """

    cdef long[:] shape = np.array(number.shape)
    F = shape[3]
    R = nreals
    cdef np.ndarray[np.double_t, ndim=2] hc2ss = np.zeros((F,R))
    cdef np.ndarray[np.double_t, ndim=2] hc2bg = np.zeros((F,R))
    cdef np.ndarray[np.longlong_t, ndim=3] ssidx = np.zeros((3,F,R), dtype=int)
    cdef np.ndarray[np.double_t, ndim=3] bgpar = np.zeros((3,F,R))
    cdef np.ndarray[np.double_t, ndim=3] sspar = np.zeros((3,F,R))
    _ss_bg_hc_and_par(shape, h2fdf, number, nreals, normal_threshold,
                 mt, mr, rz,
                hc2ss, hc2bg, ssidx, bgpar, sspar)
    return hc2ss, hc2bg, ssidx, bgpar, sspar

@cython.boundscheck(True)
@cython.wraparound(True)
@cython.nonecheck(True)
@cython.cdivision(True)
cdef void _ss_bg_hc_and_par(long[:] shape, double[:,:,:,:] h2fdf, double[:,:,:,:] number,
            long nreals, long thresh,
            double[:] mt, double[:] mr, double[:] rz,
            double[:,:] hc2ss, double[:,:] hc2bg, long[:,:,:] ssidx,
            double[:,:,:] bgpar, double[:,:,:] sspar):
    """
    Calculates the characteristic strain from loud single sources and a background of all other sources.

    Parameters
    __________
    shape : long[:] array
        shape of number, [M, Q, Z, F]
    number : double[:,:,:,:] array
        number per bin
    h2fdf : double[:,:,:,:] array
        strain amplitude squared * f/Delta f for a single source in each bin.
    nreals : int
        number of realizations.
    mt : (M,) 1Darray of scalars
        total masses of each bin center
    mr : (Q,) 1Darray of scalars
        mass ratios of each bin center
    rz : (Z,) 1Darray of scalars
        redshifts of each bin center

    hc2ss : double[:,:] array
        (memory address of) single source characteristic strain squared array.
    hc2bg : double[:,:] array
        (memory address of) background characteristic strain squared array.
    ssidx : [:,:,:] long array
        (memory address of) array for indices of max strain bins.
    bgpar :
        (memory address of) array of effective average parameters
    Returns
    _________
    void
    updated via memory address: hc2ss, hc2bg, ssidx, bg_par
    """

    cdef int M = shape[0]
    cdef int Q = shape[1]
    cdef int Z = shape[2]
    cdef int F = shape[3]
    cdef int R = nreals

    cdef int mm, qq, zz, ff, rr, m_max, q_max, z_max
    cdef double max, num, sum, m_avg, q_avg, z_avg


    # Setup random number generator from numpy library
    cdef bitgen_t *rng
    cdef const char *capsule_name = "BitGenerator"
    capsule = PCG64().capsule
    # Cast the pointer
    rng = <bitgen_t *> PyCapsule_GetPointer(capsule, capsule_name)
    for rr in range(R):
        for ff in range(F):
            max=0
            sum=0
            m_avg=0
            q_avg=0
            z_avg=0
            m_max=-1
            q_max=-1
            z_max=-1
            for mm in range(M):
                for qq in range(Q):
                    for zz in range(Z):
                        num = number[mm,qq,zz,ff]
                        cur = h2fdf[mm,qq,zz,ff]
                        if (num>thresh):
                            std = sqrt(num)
                            num = <double>random_normal(rng, num, std)
                        else:
                            num = <double>random_poisson(rng, num)
                        if(cur > max and num > 0):
                            max = cur
                            m_max = mm
                            q_max = qq
                            z_max = zz
                        sum += num * cur
                        m_avg += num * cur * mt[mm]
                        q_avg += num * cur * mr[qq]
                        z_avg += num * cur * rz[zz]
            # characteristic frequencies squared
            hc2ss[ff,rr] = max
            hc2bg[ff,rr] = sum - max

            # single source indices
            if (m_max<0):
                raise
            ssidx[0,ff,rr] = m_max # -1 if no single sources
            ssidx[1,ff,rr] = q_max # -1 if no single sources
            ssidx[2,ff,rr] = z_max # -1 if no single sources

            # background average parameters
            bgpar[0,ff,rr] = ((m_avg - h2fdf[m_max, q_max, z_max, ff] * mt[m_max])
                                /(sum-max))
            bgpar[1,ff,rr] = ((q_avg - h2fdf[m_max, q_max, z_max, ff] * mr[q_max])
                                /(sum-max))
            bgpar[2,ff,rr] = ((z_avg - h2fdf[m_max, q_max, z_max, ff] * rz[z_max])
                                /(sum-max))
            # single source parameters
            sspar[0,ff,rr] = mt[m_max]
            sspar[1,ff,rr] = mr[q_max]
            sspar[2,ff,rr] = rz[z_max]
            if (max==0): # sanity check
                print('No sources found at %dth frequency' % ff) # could warn
    # still need to sqrt and sum! (back in python)

    return

def sort_h2fdf(h2fdf):
    """ Get indices of sorted h2fdf.
    Parameters
    ----------
    h2fdf : (M,Q,Z) NDarray
        h_s^2 * f / df of a source in each bin.
    Returns
    -------
    indices : ?
    """
    cdef long[:] shape = np.array(h2fdf.shape)
    cdef long size = shape[0] * shape[1] * shape[2]

    cdef double[:]flat_h2fdf = h2fdf.flatten()

    print('flattened array elements')
    for ii in range(size):
        print(flat_h2fdf[ii])

    indices = _sort_h2fdf(flat_h2fdf, size)

    print('\nsorted array elements')
    for ii in range(size):
        print(flat_h2fdf[indices[ii]])

    return

cdef (int *) _sort_h2fdf(double[:] flat_h2fdf, long size):

    cdef (double *)array = <double *>malloc(size * sizeof(double))
    cdef (int *)indices = <int *>malloc(size * sizeof(int))

    for ii in range(size):
        array[ii] = flat_h2fdf[ii]

    argsort(array, size, &indices)

    return indices


def loudest_hc_from_sorted(number, h2fdf, nreals, nloudest, msort, qsort, zsort, normal_threshold=1e10):
    """
    Calculates the characteristic strain from loud single sources and a background of all other sources.

    Parameters
    ------------------------
    number : [M, Q, Z, F] NDarray
        number in each bin
    h2fdf : [M, Q, Z, F] NDarray
        Strain amplitude squared x frequency / frequency bin width for each bin.
    nreals
        Number of realizations.
    nloudest
        Number of loudest sources to separate in each frequency bin.
    msort : (M*Q*Z,) 1Darray
        M indices of each bin, sorted from largest to smallest h2fdf.
    qsort : (M*Q*Z,) 1Darray
        q indices of each bin, sorted from largest to smallest h2fdf.
    zsort : (M*Q*Z,) 1Darray
        z indices of each bin, sorted from largest to smallest h2fdf.
    normal_threshold : float
        Threshold for approximating poisson sampling as normal.

    Returns
    --------------------------
    hc2ss : (F, R, L) Ndarray of scalars
        Char strain squared of the loudest single sources.
    hc2bg : (F, R) Ndarray of scalars
        Char strain squared of the background.
    """

    cdef long[:] shape = np.array(number.shape)
    F = shape[3]
    R = nreals
    L = nloudest
    cdef np.ndarray[np.double_t, ndim=3] hc2ss = np.zeros((F,R,L))
    cdef np.ndarray[np.double_t, ndim=2] hc2bg = np.zeros((F,R))
    _loudest_hc_from_sorted(shape, h2fdf, number, nreals, nloudest, normal_threshold,
                            msort, qsort, zsort,
                            hc2ss, hc2bg)
    return hc2ss, hc2bg

@cython.boundscheck(True)
@cython.wraparound(True)
@cython.nonecheck(True)
@cython.cdivision(True)
cdef void _loudest_hc_from_sorted(long[:] shape, double[:,:,:,:] h2fdf, double[:,:,:,:] number,
            long nreals, long nloudest, long thresh,
            long[:] msort, long[:] qsort, long[:] zsort,
            double[:,:,:] hc2ss, double[:,:] hc2bg):
    """
    Calculates the characteristic strain from loud single sources and a background of all other sources.

    Parameters
    ----------
    shape : long[:] array
        Shape of number, [M, Q, Z, F].
    number : double[:,:,:,:] array
        Number per bin.
    h2fdf : double[:,:,:,:] array
        Strain amplitude squared * f/Delta f for a single source in each bin.
    nreals : int
        Number of realizations.
    nloudest : int
        Number of loudest sources at each source.
    msort : (M*Q*Z,) 1Darray
        M indices of each bin, sorted from largest to smallest h2fdf.
    qsort : (M*Q*Z,) 1Darray
        q indices of each bin, sorted from largest to smallest h2fdf.
    zsort : (M*Q*Z,) 1Darray
        z indices of each bin, sorted from largest to smallest h2fdf.
    hc2ss : double[:,:,:] array
        (Memory address of) single source characteristic strain squared array.
    hc2bg : double[:,:] array
        (Memory address of) background characteristic strain squared array.

    Returns
    -------
    void
    updated via memory address: hc2ss, hc2bg, ssidx, bg_par
    """

    cdef int M = shape[0]
    cdef int Q = shape[1]
    cdef int Z = shape[2]
    cdef int F = shape[3]
    cdef int L = nloudest
    cdef int R = nreals

    cdef int mm, qq, zz, ff, rr, ll
    cdef double num, sum

    # Setup random number generator from numpy library
    cdef bitgen_t *rng
    cdef const char *capsule_name = "BitGenerator"
    capsule = PCG64().capsule
    # Cast the pointer
    rng = <bitgen_t *> PyCapsule_GetPointer(capsule, capsule_name)
    for rr in range(R):
        for ff in range(F):
            ll = 0 # track which index in the loudest list you're currently storing
                     # start at 0 for the loudest of all.
            sum = 0
            for bb in range(M*Q*Z): #iterate through bins, loudest to quietest
                mm = msort[bb]
                qq = qsort[bb]
                zz = zsort[bb]
                num = number[mm,qq,zz,ff]
                if (num>thresh): # Gaussian sample
                    std = sqrt(num)
                    num = <double>random_normal(rng, num, std)
                else:            # Poisson sample
                    num = <double>random_poisson(rng, num)
                if(num < 1):
                    continue
                cur = h2fdf[mm,qq,zz,ff]
                if (num<1):
                    continue # to next loudest bin
                while (ll < L) and (num > 0):
                    hc2ss[ff,rr,ll] = cur
                    num -= 1
                    ll += 1
                sum += num * cur

            hc2bg[ff,rr] = sum


def loudest_hc_and_par_from_sorted(number, h2fdf, nreals, nloudest, mt, mr, rz, msort, qsort, zsort, normal_threshold=1e10):
    """
    Calculates the characteristic strain from loud single sources and a background of all other sources.

    Parameters
    ------------------------
    number : [M, Q, Z, F] NDarray
        number in each bin
    h2fdf : [M, Q, Z, F] NDarray
        Strain amplitude squared x frequency / frequency bin width for each bin.
    nreals
        Number of realizations.
    nloudest
        Number of loudest sources to separate in each frequency bin.
    mt : (M,) 1Darray of scalars
        Total masses, M, of each bin center.
    mr : (Q,) 1Darray of scalars
        Mass ratios, q, of each bin center.
    rz : (Z,) 1Darray of scalars
        Redshifts, z, of each bin center.
    msort : (M*Q*Z,) 1Darray
        M indices of each bin, sorted from largest to smallest h2fdf.
    qsort : (M*Q*Z,) 1Darray
        q indices of each bin, sorted from largest to smallest h2fdf.
    zsort : (M*Q*Z,) 1Darray
        z indices of each bin, sorted from largest to smallest h2fdf.
    normal_threshold : float
        Threshold for approximating poisson sampling as normal.

    Returns
    --------------------------
    hc2ss : (F, R, L) Ndarray of scalars
        Char strain squared of the loudest single sources.
    hc2bg : (F, R) Ndarray of scalars
        Char strain squared of the background.
    lspar : (3, F, R) NDarray of scalars
        Average effective M, q, z parameters of the loudest L sources.
    bgpar : (3, F, R) NDarray of scalars
        Average effective M, q, z parameters of the background.
    ssidx : (3, F, R, L) NDarray of ints
        Indices of the loudest single sources.
    """

    cdef long[:] shape = np.array(number.shape)
    F = shape[3]
    R = nreals
    L = nloudest
    cdef np.ndarray[np.double_t, ndim=3] hc2ss = np.zeros((F,R,L))
    cdef np.ndarray[np.double_t, ndim=2] hc2bg = np.zeros((F,R))
    cdef np.ndarray[np.double_t, ndim=3] lspar = np.zeros((3,F,R))
    cdef np.ndarray[np.double_t, ndim=3] bgpar = np.zeros((3,F,R))
    cdef np.ndarray[np.longlong_t, ndim=4] ssidx = np.zeros((3,F,R,L), dtype=int)
    _loudest_hc_and_par_from_sorted(shape, h2fdf, number, nreals, nloudest, normal_threshold,
                            mt, mr, rz, msort, qsort, zsort,
                            hc2ss, hc2bg, lspar, bgpar, ssidx)
    return hc2ss, hc2bg, lspar, bgpar, ssidx


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void _loudest_hc_and_par_from_sorted(long[:] shape, double[:,:,:,:] h2fdf, double[:,:,:,:] number,
            long nreals, long nloudest, long thresh,
            double[:] mt, double[:] mr, double[:] rz,
            long[:] msort, long[:] qsort, long[:] zsort,
            double[:,:,:] hc2ss, double[:,:] hc2bg, double[:,:,:] lspar, double[:,:,:] bgpar, long[:,:,:,:] ssidx):
    """
    Calculates the characteristic strain from loud single sources and a background of all other sources.

    Parameters
    ----------
    shape : long[:] array
        Shape of number, [M, Q, Z, F].
    number : double[:,:,:,:] array
        Number per bin.
    h2fdf : double[:,:,:,:] array
        Strain amplitude squared * f/Delta f for a single source in each bin.
    nreals : int
        Number of realizations.
    nloudest : int
        Number of loudest sources at each source.
    mt : (M,) 1Darray of scalars
        Total masses of each bin center.
    mr : (Q,) 1Darray of scalars
        Mass ratios of each bin center.
    rz : (Z,) 1Darray of scalars
        Redshifts of each bin center.
    msort : (M*Q*Z,) 1Darray
        M indices of each bin, sorted from largest to smallest h2fdf.
    qsort : (M*Q*Z,) 1Darray
        q indices of each bin, sorted from largest to smallest h2fdf.
    zsort : (M*Q*Z,) 1Darray
        z indices of each bin, sorted from largest to smallest h2fdf.
    hc2ss : double[:,:,:] array
        (Memory address of) single source characteristic strain squared array.
    hc2bg : double[:,:] array
        (Memory address of) background characteristic strain squared array.
    lspar : (3, F, R) NDarray of scalars
        Average effective M, q, z parameters of the loudest L sources.
    bgpar : (3, F, R) NDarray of scalars
        Average effective M, q, z parameters of the background.
    ssidx : (3, F, R, L) NDarray of ints
        Indices of the loudest sources.

    Returns
    -------
    void
    updated via memory address: hc2ss, hc2bg, lspar, bgpar, ssidx
    """

    cdef int M = shape[0]
    cdef int Q = shape[1]
    cdef int Z = shape[2]
    cdef int F = shape[3]
    cdef int L = nloudest
    cdef int R = nreals

    cdef int mm, qq, zz, ff, rr, ll
    cdef double num, cur, sum_bg, sum_ls, m_bg, q_bg, z_bg, m_ls, q_ls, z_ls
    cdef np.ndarray[np.double_t, ndim=3] maxes = np.zeros((F,R,L))

    # Setup random number generator from numpy library
    cdef bitgen_t *rng
    cdef const char *capsule_name = "BitGenerator"
    capsule = PCG64().capsule
    # Cast the pointer
    rng = <bitgen_t *> PyCapsule_GetPointer(capsule, capsule_name)
    for rr in range(R):
        for ff in range(F):
            ll = 0 # track which index in the loudest list you're currently storing
                     # start at 0 for the loudest of all.
            # reset strain sums
            sum_bg = 0 # sum of bg h2fdf, for parameter averaging and gwb
            sum_ls = 0 # sum of ls h2fdf, for parameter averaging
            # reset parameter averaging sums
            m_ls = 0
            q_ls = 0
            z_ls = 0
            m_bg = 0
            q_bg = 0
            z_bg = 0
            for bb in range(M*Q*Z): #iterate through bins, loudest to quietest
                mm = msort[bb]
                qq = qsort[bb]
                zz = zsort[bb]
                num = number[mm,qq,zz,ff]
                if (num>thresh): # Gaussian sample
                    std = sqrt(num)
                    num = <double>random_normal(rng, num, std)
                else:            # Poisson sample
                    num = <double>random_poisson(rng, num)
                if(num < 1):
                    continue
                cur = h2fdf[mm,qq,zz,ff] # h^2 * f/df of current bin
                if (num<1):
                    continue # to next loudest bin
                while (ll < L) and (num > 0):
                    # store ll loudest source strain
                    hc2ss[ff,rr,ll] = cur

                    # store indices of ll loudest source
                    ssidx[0,ff,rr,ll] = mm
                    ssidx[1,ff,rr,ll] = qq
                    ssidx[2,ff,rr,ll] = zz


                    sum_ls += cur # tot ls h2fdf
                    # add to average parameters of loudest sources
                    m_ls += cur * mt[mm] # tot weighted ls mass
                    q_ls += cur * mr[qq] # tot weighted ls ratio
                    z_ls += cur * rz[zz] # tot weighted ls redshift

                    # update number and ll index
                    num -= 1
                    ll += 1

                sum_bg += num * cur # tot bg h2fdf
                # add to average parameters of background sources
                m_bg += num * cur * mt[mm] # tot weight bg mass
                q_bg += num * cur * mr[qq] # tot weighted bg ratio
                z_bg += num * cur * rz[zz] # tot weighted bg redshift

            hc2bg[ff,rr] = sum_bg # background strain
            # background average parameters
            bgpar[0,ff,rr] = m_bg/sum_bg # bg avg mass
            bgpar[1,ff,rr] = q_bg/sum_bg # bg avg ratio
            bgpar[2,ff,rr] = z_bg/sum_bg # bg avg redshift
            # loudest source average parameters
            lspar[0,ff,rr] = m_ls/sum_ls # ls avg mass
            lspar[1,ff,rr] = q_ls/sum_ls # ls avg ratio
            lspar[2,ff,rr] = z_ls/sum_ls # ls avg redshift


def loudest_hc_and_par_from_sorted_redz(
    number, h2fdf, nreals, nloudest,
    mt, mr, rz, redz_final, dcom_final, sepa, angs,
    msort, qsort, zsort, normal_threshold=1e10):
    """
    Calculates the characteristic strain and binary parameters from loud single sources and a
    background of all other sources.

    Parameters
    ------------------------
    number : [M, Q, Z, F] NDarray
        number in each bin
    h2fdf : [M, Q, Z, F] NDarray
        Strain amplitude squared x frequency / frequency bin width for each bin.
    nreals
        Number of realizations.
    nloudest
        Number of loudest sources to separate in each frequency bin.
    mt : (M,) 1Darray of scalars
        Total masses, M, of each bin center.
    mr : (Q,) 1Darray of scalars
        Mass ratios, q, of each bin center.
    rz : (Z,) 1Darray of scalars
        Redshifts, z, of each bin center.
    redz_final : (M,Q,Z,F) NDarray of scalars
        Final redshifts of each bin.
    dcom_final : (M,Q,Z,F) NDarray of scalars
        Final comoving distances of each bin.
    sepa : (M,Q,Z,F) NDarray of scalars
        Final separations of each mass and frequency combination.
    angs : (M,Q,Z,F)
        Final angular separations of each bin.
    msort : (M*Q*Z,) 1Darray
        M indices of each bin, sorted from largest to smallest h2fdf.
    qsort : (M*Q*Z,) 1Darray
        q indices of each bin, sorted from largest to smallest h2fdf.
    zsort : (M*Q*Z,) 1Darray
        z indices of each bin, sorted from largest to smallest h2fdf.
    normal_threshold : float
        Threshold for approximating poisson sampling as normal.

    Returns
    --------------------------
    hc2ss : (F, R, L) Ndarray of scalars
        Char strain squared of the loudest single sources.
    hc2bg : (F, R) Ndarray of scalars
        Char strain squared of the background.
    sspar : (4, F, R) NDarray of scalars
        Effective M, q, z parameters of the loudest L sources.
        mass, ratio, redshift, redshift_final
    bgpar : (4, F, R) NDarray of scalars
        Average effective M, q, z parameters of the background.
        mass, ratio, redshift, redshift_final
    """

    cdef long[:] shape = np.array(number.shape)
    F = shape[3]
    R = nreals
    L = nloudest
    cdef np.ndarray[np.double_t, ndim=3] hc2ss = np.zeros((F,R,L))
    cdef np.ndarray[np.double_t, ndim=2] hc2bg = np.zeros((F,R))
    cdef np.ndarray[np.double_t, ndim=4] sspar = np.zeros((4,F,R,L))
    cdef np.ndarray[np.double_t, ndim=3] bgpar = np.zeros((7,F,R))
    _loudest_hc_and_par_from_sorted_redz(shape, h2fdf, number, nreals, nloudest, normal_threshold,
                            mt, mr, rz, redz_final, dcom_final, sepa, angs,
                            msort, qsort, zsort,
                            hc2ss, hc2bg, sspar, bgpar)
    return hc2ss, hc2bg, sspar, bgpar


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void _loudest_hc_and_par_from_sorted_redz(long[:] shape, double[:,:,:,:] h2fdf, double[:,:,:,:] number,
            long nreals, long nloudest, long thresh,
            double[:] mt, double[:] mr, double[:] rz,
            double[:,:,:,:] redz_final, double[:,:,:,:] dcom_final, double[:,:,:,:] sepa, double[:,:,:,:] angs,
            long[:] msort, long[:] qsort, long[:] zsort,
            double[:,:,:] hc2ss, double[:,:] hc2bg, double[:,:,:,:] sspar, double[:,:,:] bgpar):
    """
    Calculates the characteristic strain from loud single sources and a background of all other sources.

    Parameters
    ----------
    shape : long[:] array
        Shape of number, [M, Q, Z, F].
    number : double[:,:,:,:] array
        Number per bin.
    h2fdf : double[:,:,:,:] array
        Strain amplitude squared * f/Delta f for a single source in each bin.
    nreals : int
        Number of realizations.
    nloudest : int
        Number of loudest sources at each source.
    mt : (M,) 1Darray of scalars
        Total masses of each bin center.
    mr : (Q,) 1Darray of scalars
        Mass ratios of each bin center.
    rz : (Z,) 1Darray of scalars
        Redshifts, z, of each bin center.
    redz_final : (M,Q,Z,F) NDarray of scalars
        Final redshifts of each bin.
    dcom_final : (M,Q,Z,F) NDarray of scalars
        Final comoving distances of each bin.
    sepa : (M,Q,Z,F) NDarray of scalars
        Final separations of each bin.
    angs : (M,Q,Z,F)
        Final angular separations of each bin.
    msort : (M*Q*Z,) 1Darray
        M indices of each bin, sorted from largest to smallest h2fdf.
    qsort : (M*Q*Z,) 1Darray
        q indices of each bin, sorted from largest to smallest h2fdf.
    zsort : (M*Q*Z,) 1Darray
        z indices of each bin, sorted from largest to smallest h2fdf.
    hc2ss : double[:,:,:] array
        (Memory address of) single source characteristic strain squared array.
    hc2bg : double[:,:] array
        (Memory address of) background characteristic strain squared array.
    sspar : (4, F, R) NDarray of scalars
        Effective M, q, z parameters of the loudest L sources.
        mass, ratio, redshift, redshift_final
    bgpar : (4, F, R) NDarray of scalars
        Average effective M, q, z parameters of the background.
        mass, ratio, redshift, redshift_final

    Returns
    -------
    void
    updated via memory address: hc2ss, hc2bg, sspar, bgpar
    """

    cdef int M = shape[0]
    cdef int Q = shape[1]
    cdef int Z = shape[2]
    cdef int F = shape[3]
    cdef int L = nloudest
    cdef int R = nreals

    cdef int mm, qq, zz, ff, rr, ll
    cdef double num, cur, sum_bg, m_bg, q_bg, z_bg, zfinal_bg, dcom_bg, sepa_bg, angs_bg

    # # check all redz_final are positive
    # for mm in range(len(redz_final)):
    #     for qq in range(len(redz_final[0])):
    #         for zz in range(len(redz_final[0,0])):
    #             for ff in range(len(redz_final[0,0,0])):
    #                 if (redz_final[mm,qq,zz,ff]<0 and redz_final[mm,qq,zz,ff] !=-1):
    #                     err = f"redz_final[{mm},{qq},{zz},{ff},] = {redz_final[mm,qq,zz,ff]} < 0"
    #                     raise ValueError(err)
    # print("passed redz_final check in _loudest_hc_and_par_from_sorted_redz")



    # Setup random number generator from numpy library
    cdef bitgen_t *rng
    cdef const char *capsule_name = "BitGenerator"
    capsule = PCG64().capsule
    # Cast the pointer
    rng = <bitgen_t *> PyCapsule_GetPointer(capsule, capsule_name)
    for rr in range(R):
        for ff in range(F):
            ll = 0 # track which index in the loudest list you're currently storing
                     # start at 0 for the loudest of all.
            # reset strain sums
            sum_bg = 0 # sum of bg h2fdf, for parameter averaging and gwb
            # reset parameter averaging sums
            m_bg = 0
            q_bg = 0
            z_bg = 0
            zfinal_bg = 0
            dcom_bg = 0
            sepa_bg = 0
            angs_bg = 0
            for bb in range(M*Q*Z): #iterate through bins, loudest to quietest
                mm = msort[bb]
                qq = qsort[bb]
                zz = zsort[bb]
                num = number[mm,qq,zz,ff]
                if (num>thresh): # Gaussian sample
                    std = sqrt(num)
                    num = <double>random_normal(rng, num, std)
                else:            # Poisson sample
                    num = <double>random_poisson(rng, num)
                cur = h2fdf[mm,qq,zz,ff] # h^2 * f/df of current bin
                
                if (num < 1) or (cur == 0):
                    continue # to next loudest bin
                while (ll < L) and (num > 0):
                    # store ll loudest source strain
                    hc2ss[ff,rr,ll] = cur

                    # store indices of ll loudest source
                    sspar[0,ff,rr,ll] = mt[mm]
                    sspar[1,ff,rr,ll] = mr[qq]
                    sspar[2,ff,rr,ll] = rz[zz]
                    sspar[3,ff,rr,ll] = redz_final[mm,qq,zz,ff]

                    # check for negative redz_final
                    if redz_final[mm,qq,zz,ff]<0 and redz_final[mm,qq,zz,ff]!=-1:
                        # badz = badz+1
                        err = f"redz_final[{mm},{qq},{zz},{ff}] = {redz_final[mm,qq,zz,ff]} < 0"
                        print("ERROR IN CYUTILS:", err)

                    # update number and ll index
                    num -= 1
                    ll += 1

                sum_bg += num * cur # tot bg h2fdf
                # add to average parameters of background sources
                m_bg += num * cur * mt[mm] # tot weight bg mass
                q_bg += num * cur * mr[qq] # tot weighted bg ratio
                z_bg += num * cur * rz[zz] # tot weighted bg redshift
                zfinal_bg += num * cur * redz_final[mm,qq,zz,ff] # tot weighted bg redshift after hardening
                dcom_bg += num * cur * dcom_final[mm,qq,zz,ff] # tot weighted bg com. dist. after hardening
                sepa_bg += num * cur * sepa[mm,qq,zz,ff] # tot weighted bg separation after hardening
                angs_bg += num * cur * angs[mm,qq,zz,ff] # tot weighted bg angular separation after hardening

            hc2bg[ff,rr] = sum_bg # background strain
            # background average parameters
            bgpar[0,ff,rr] = m_bg/sum_bg # bg avg mass
            bgpar[1,ff,rr] = q_bg/sum_bg # bg avg ratio
            bgpar[2,ff,rr] = z_bg/sum_bg # bg avg redshift
            bgpar[3,ff,rr] = zfinal_bg/sum_bg # bg avg redshift after hardening
            bgpar[4,ff,rr] = dcom_bg/sum_bg # bg avg comoving distance after hardening
            bgpar[5,ff,rr] = sepa_bg/sum_bg # bg avg binary separation after hardening
            bgpar[6,ff,rr] = angs_bg/sum_bg # bg avg binary angular separation after hardening




def interp_2d(xnew, xold, yold, xlog=False, ylog=False, extrap=False):
    if xlog:
        xnew = np.log10(xnew)
        xold = np.log10(xold)
    if ylog:
        yold = np.log10(yold)

    assert xnew.shape[0] == xold.shape[0]
    assert xold.shape == yold.shape

    extrap_flag = 1 if extrap else 0
    ynew = _interp_2d(xnew, xold, yold, extrap_flag)

    if ylog:
        ynew = np.power(10.0, ynew)
    return ynew


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double[:, :] _interp_2d(
    double[:, :] xnew, double[:, :] xold, double[:, :] yold, int extrap,
):
    cdef int num = xnew.shape[0]
    cdef int new_size = xnew.shape[1]
    cdef int old_size = xold.shape[1]
    cdef int last = old_size - 1

    cdef int ii, nn, oo
    cdef double newval

    cdef np.ndarray[np.double_t, ndim=2] ynew = np.empty((num, new_size))
    for ii in range(num):
        oo = 0
        for nn in range(new_size):
            newval = xnew[ii, nn]
            while (xold[ii, oo+1] < newval) and (oo < last-1):
                oo += 1

            if extrap == 1 or ((xold[ii, oo] < newval) and (newval < xold[ii, oo+1])):
                ynew[ii, nn] = _interp_between_vals(newval, xold[ii, oo], xold[ii, oo+1], yold[ii, oo], yold[ii, oo+1])
            else:
                ynew[ii, nn] = NAN

    return ynew








# ==================================================================================================
# ====    DetStats Functions    ====
# ==================================================================================================


def gamma_of_rho_interp(rho, rsort, rho_interp_grid, gamma_interp_grid):
    """
    rho : 1Darray of scalars
        SNR of single sources, in flat array
    rsort : 1Darray
        order of flat rho values smallest to largest
    rho_interp_grid : 1Darray
        rho values corresponding to each gamma
    gamma_interp_grid : 1Darray
        gamma values corresponding to each rho

    """
    # pass in the interp grid
    cdef np.ndarray[np.double_t, ndim=1] gamma = np.zeros(rho.shape)

    _gamma_of_rho_interp(rho, rsort, rho_interp_grid, gamma_interp_grid, gamma)

    return gamma

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef int _gamma_of_rho_interp(
    double[:] rho, long[:] rsort,
    double[:] rho_interp_grid, double[:] gamma_interp_grid,
    # output
    double[:] gamma
    ):
    """ Find gamma of rho by interpolation over rho and gamma grids.
    """

    cdef int n_rho = rho.size
    cdef int n_interp = rho_interp_grid.size
    cdef int ii, kk, rr
    ii = 0 # get rho in order using rho[rsort[ii]]

    for kk in range(n_rho):
        rr = rsort[kk] # index of next largest rho, equiv to rev in redz calculation
        # print('kk =',kk,' rr =', rr, 'rho[rr] =', rho[rr])
        # get to the right index of the interpolation-grid
        while (rho_interp_grid[ii+1] < rho[rr]) and (ii < n_interp -2):
            ii += 1
        # print('ii =',ii, ' rho_interp[ii] =', rho_interp_grid[ii], ' rho_interp[ii+1] =', rho_interp_grid[ii+1])
        # interpolate
        gamma[rr] = interp_at_index(ii, rho[rr], rho_interp_grid, gamma_interp_grid)
        # print('rho =', rho[rr], ' gamma =', gamma[rr], '\n')\

    return 0


def snr_ss(amp, F_iplus, F_icross, iotas, dur, Phi_0, S_i, freqs):
    """ Calculate single source SNR


    Parameters
    ----------
    amp : (F,R,L) NDarray
        Dimensionless strain amplitude of loudest single sources
    F_iplus : (P,F,S,L) NDarray
        Antenna pattern function for each pulsar.
    F_icross : (P,F,S,L) NDarray
        Antenna pattern function for each pulsar.
    iotas : (F,S,L) NDarray
        Inclination, used to calculate:
        a_pol = 1 + np.cos(iotas) **2
        b_pol = -2 * np.cos(iotas)
    dur : scalar
        Duration of observations.
    Phi_0 : (F,S,L) NDarray
        Initial GW phase
    S_i : (P,F,R,L) NDarray
        Total noise of each pulsar wrt detection of each single source, in s^3
    freqs : (F,) 1Darray
        Observed frequency bin centers.

    Returns
    -------
    snr_ss : (F,R,S,L) NDarray
        SNR from the whole PTA for each single source with
        each realized sky position (S) and realized strain (R)

    """
    nfreqs, nreals, nloudest = amp.shape[0], amp.shape[1], amp.shape[2]
    npsrs, nskies = F_iplus.shape[0], F_iplus.shape[2]
    cdef np.ndarray[np.double_t, ndim=4] snr_ss = np.zeros((nfreqs, nreals, nskies, nloudest))
    _snr_ss(
        amp, F_iplus, F_icross, iotas, dur, Phi_0, S_i, freqs,
        npsrs, nfreqs, nreals, nskies, nloudest,
        snr_ss)
    return snr_ss

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef int _snr_ss(
    double[:,:,:] amp,
    double[:,:,:,:] F_iplus,
    double[:,:,:,:] F_icross,
    double[:,:,:] iotas,
    double dur,
    double[:,:,:] Phi_0,
    double[:,:,:,:] S_i,
    double[:] freqs,
    long npsrs, long nfreqs, long nreals, long nskies, long nloudest,
    # output
    double[:,:,:,:] snr_ss
    ):
    """

    Parameters
    ----------
    amp : (F,R,L) NDarray
        Dimensionless strain amplitude of loudest single sources
    F_iplus : (P,F,S,L) NDarray
        Antenna pattern function for each pulsar.
    F_icross : (P,F,S,L) NDarray
        Antenna pattern function for each pulsar.
    iotas : (F,S,L) NDarray
        Inclination, used to calculate:
        a_pol = 1 + np.cos(iotas) **2
        b_pol = -2 * np.cos(iotas)
    dur : scalar
        Duration of observations.
    Phi_0 : (F,S,L) NDarray
        Initial GW phase
    S_i : (P,F,R,L) NDarray
        Total noise of each pulsar wrt detection of each single source, in s^3
    freqs : (F,) 1Darray
        Observed frequency bin centers.
    snr_ss : (F,R,S,L) NDarray
        Pointer to single source SNR array, to be calculated.

    NOTE: This may be improved by moving some of the math outside the function.
    I.e., passing in sin/cos of NDarrays to be used.
    """

    cdef int pp, ff, rr, ss, ll
    cdef float a_pol, b_pol, Phi_T, pta_snr_sq, coef, term1, term2, term3
    # print('npsrs %d, nfreqs %d, nreals %d, nskies %d, nloudest %d' % (npsrs, nfreqs, nreals, nskies, nloudest))

    for ff in range(nfreqs):
        for ss in range(nskies):
            for ll in range(nloudest):
                a_pol = 1 + pow(cos(iotas[ff,ss,ll]), 2.0)
                b_pol = -2 * cos(iotas[ff,ss,ll])
                Phi_T = 2 * M_PI * freqs[ff] * dur + Phi_0[ff,ss,ll]
                for rr in range(nreals):
                    pta_snr_sq = 0
                    for pp in range(npsrs):
                        # calculate coefficient depending on
                        # function of amp, S_i, and freqs
                        coef = pow(amp[ff,rr,ll], 2.0) / (S_i[pp,ff,rr,ll] * 8 * pow(M_PI * freqs[ff], 3.0))

                        # calculate terms that depend on p, f, s, and l
                        # functions of F_iplus, F_icross, a_pol, b_pol, Phi_0, and Phi_T
                        term1 = (
                            pow(a_pol * F_iplus[pp,ff,ss,ll], 2.0)
                            * (Phi_T * (1.0 + 2.0 * pow(sin(Phi_0[ff,ss,ll]), 2.0))
                                + cos(Phi_T) * (-1.0 * sin(Phi_T) + 4.0 * cos(Phi_0[ff,ss,ll]))
                                - 4.0 * sin(Phi_0[ff,ss,ll])
                                )
                        )
                        term2 = (
                            pow(b_pol * F_icross[pp,ff,ss,ll], 2.0)
                            * (Phi_T * (1.0 + 2.0 * pow(cos(Phi_0[ff,ss,ll]), 2.0))
                                + sin(Phi_T) * cos(Phi_T) - 4.0 * cos(Phi_0[ff,ss,ll])
                                )
                        )
                        term3 = (
                            -2.0 * a_pol * b_pol * F_iplus[pp,ff,ss,ll] * F_icross[pp,ff,ss,ll]
                            * (2.0 * Phi_T * sin(Phi_T) *cos(Phi_0[ff,ss,ll])
                                + sin(Phi_T) * (sin(Phi_T) - 2.0 * sin(Phi_0[ff,ss,ll])
                                                + 2.0 * cos(Phi_T) * cos(Phi_0[ff,ss,ll])
                                                - 2.0 * cos(Phi_0[ff,ss,ll])
                                                )
                            )
                        )
                        pta_snr_sq += coef*(term1 + term2 + term3) # sum snr^2 of all pulsars for a single source

                    # set snr for a single source, using sum from all pulsars
                    snr_ss[ff,rr,ss,ll] = sqrt(pta_snr_sq)


def Sh_rest(hc_ss, hc_bg, freqs, nexcl):
    """
    Calculate the noise from all the single sources except the source in question
    and the next N_excl loudest sources.

    Parameters
    ----------
    hc_ss : (F,R,L) NDarray
        Characteristic strain from all loud single sources.
    hc_bg : (F,R) NDarray
        Characteristic strain from all but loudest source at each frequency.
    freqs : (F,) 1Darray
        Frequency bin centers.
    nexcl : int
        Number of loudest single sources to exclude from hc_rest noise, in addition 
        to the source in question.

    Returns
    -------
    Sh_rest : (F,R,L) NDarray of scalars
        The noise in a single pulsar from other GW sources for detecting each single source.

    """

    nfreqs, nreals, nloudest = hc_ss.shape[0], hc_ss.shape[1], hc_ss.shape[2]
    cdef np.ndarray[np.double_t, ndim=3] Sh_rest = np.zeros((nfreqs, nreals, nloudest))
    _Sh_rest(hc_ss, hc_bg, freqs, nexcl, nreals, nfreqs, nloudest, Sh_rest)
    return Sh_rest


cdef void _Sh_rest(
    double[:,:,:] hc_ss, double[:,:,] hc_bg, double[:] freqs, long nexcl,
    long nreals, long nfreqs, long nloudest,
    double[:,:,:] Sh_rest):
    """
    Calculate the noise from all the single sources except the source in question
    and the next N_excl loudest sources.

    Parameters
    ----------
    hc_ss : (F,R,L) NDarray
        Characteristic strain from all loud single sources.
    hc_bg : (F,R) NDarray
        Characteristic strain from all but loudest source at each frequency.
    freqs : (F,) 1Darray
        Frequency bin centers.
    nexcl : int
        Number of loudest single sources to exclude from hc_rest noise, in addition 
        to the source in question.

    Returns
    -------
    void
    Sh_rest : (F,R,L) NDarray of scalars, updated via memory address


    Sh = hc^2 / (f^3 12 pi^2)

    """
    cdef int ff, rr, ll, count
    cdef double Sh_ss, Sh_bg

    for ff in range(nfreqs):
        freq = freqs[ff]
        for rr in range(nreals):
            for ll in range(nloudest): # calculating for the llth loduest
                Sh_ss = 0
                count = 0
                for ii in range(nloudest): # adding other loudest with index ii
                    if (ii != ll): # check it's not our current source
                    # if current is in top N_excl, must be (N_excl+1)th or above
                    # if current is >= top N_excl, must be (N_excl)th or above
                        if ((ll < nexcl) and (ii > nexcl)) or ((ll >= nexcl) and (ii > nexcl-1)): 
                            Sh_ss += pow(hc_ss[ff,rr,ii], 2.0) / pow(freq, 3.0) / (12*pow(M_PI, 2.0))
                            count += 1
                Sh_bg = pow(hc_bg[ff,rr], 2.0) / pow(freq, 3.0) / (12*pow(M_PI, 2.0))
                Sh_rest[ff,rr,ll] = Sh_rest[ff,rr,ll] + Sh_ss + Sh_bg
                if count != (nloudest - nexcl - 1):
                    err = (f"ERROR in calculate Sh_rest! count of sources={count}")
                    print(err)
                


