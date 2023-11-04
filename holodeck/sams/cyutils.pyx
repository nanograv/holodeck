"""

$ python setup.py build_ext -i

$ python setup.py develop

"""

import cython
import numpy as np
cimport numpy as np
np.import_array()

from scipy.optimize.cython_optimize cimport brentq

from libc.stdio cimport printf, fflush, stdout
from libc.stdlib cimport malloc, free
# make sure to use c-native math functions instead of python/numpy
from libc.math cimport pow, sqrt, M_PI, NAN, log10, sin, cos

import holodeck as holo
from holodeck.cyutils cimport interp_at_index, _interp_between_vals


# ---- Define Parameters


# ---- Define Constants

cdef double MY_NWTG = 6.6742999e-08
cdef double MY_SPLC = 29979245800.0
cdef double MY_MPC = 3.08567758e+24
cdef double MY_MSOL = 1.988409870698051e+33
cdef double MY_YR = 31557600.0
cdef double MY_SCHW = 1.4852320538237328e-28     #: Schwarzschild Constant  2*G/c^2  [cm]
cdef double GW_DADT_SEP_CONST = - 64.0 * pow(MY_NWTG, 3) / 5.0 / pow(MY_SPLC, 5)

cdef double MY_PC = MY_MPC / 1.0e6
cdef double MY_GYR = MY_YR * 1.0e9
cdef double KEPLER_CONST_FREQ = (1.0 / (2.0*M_PI)) * sqrt(MY_NWTG)
cdef double KEPLER_CONST_SEPA = pow(MY_NWTG, 1.0/3.0) / pow(2.0*M_PI, 2.0/3.0)
cdef double FOUR_PI_SPLC_OVER_MPC = 4 * M_PI * MY_SPLC / MY_MPC


@cython.cdivision(True)
cpdef double hard_gw(double mtot, double mrat, double sepa):
# cdef double hard_gw(double mtot, double mrat, double sepa):
    cdef double dadt = GW_DADT_SEP_CONST * pow(mtot, 3) * mrat / pow(sepa, 3) / pow(1 + mrat, 2)
    return dadt


@cython.cdivision(True)
cdef double kepler_freq_from_sepa(double mtot, double sepa):
    cdef double freq = KEPLER_CONST_FREQ * sqrt(mtot) / pow(sepa, 1.5)
    return freq


@cython.cdivision(True)
cdef double kepler_sepa_from_freq(double mtot, double freq):
    cdef double sepa = KEPLER_CONST_SEPA * pow(mtot, 1.0/3.0) / pow(freq, 2.0/3.0)
    return sepa


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int while_while_increasing(int start, int size, double val, double[:] edges):
    """Step through an INCREASING array of `edges`, first forward, then backward, to find edges bounding `val`.

    Use this function when `start` is already a close guess, and we just need to update a bit.

    """

    cdef int index = start    #: index corresponding to the LEFT side of the edges bounding `val`

    # `index < size-2` so that the result is always within the edges array
    # `edges[index+1] < val` to get the RIGHT-edge to be MORE than `val`
    while (index < size - 2) and (edges[index+1] < val):
        index += 1

    # `edges[index] > val` to get the LEFT-edge to be LESS than `val`
    while (index > 0) and (edges[index] > val):
        index -= 1

    return index


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int while_while_decreasing(int start, int size, double val, double[:] edges):
    """Step through a DECREASING array of `edges`, first forward, then backward, to find edges bounding `val`.

    Use this function when `start` is already a close guess, and we just need to update a bit.

    """

    cdef int index = start    #: index corresponding to the LEFT side of the edges bounding `val`

    # `index < size-1` so that the result is always within the edges array
    # `edges[index+1] > val` to get the RIGHT-edge to be LESS than `val`
    while (index < size - 1) and (edges[index+1] > val):
        index += 1

    # `edges[index-1] < val` to get the LEFT-edge to be MORE than `val`
    while (index > 0) and (edges[index-1] < val):
        index -= 1

    return index


# ==================================================================================================
# ====    Integrate Bins from differential-parameter-volume to total numbers   ====
# ==================================================================================================


def integrate_differential_number_3dx1d(edges, dnum):
    """Integrate the differential number of binaries over each grid bin into total numbers of binaries.

    Trapezoid used over first 3 dims (mtot, mrat, redz), and Riemann over 4th (freq).
    (Riemann seemed empirically to be more accurate for freq, but this should be revisited.)
    mtot is integrated over `log10(mtot)` and frequency is integrated over `ln(f)`.

    Note on array shapes:
    input  `dnum` is shaped (M, Q, Z, F)
    input  `edges` must be (4,) of array_like of lengths:  M, Q, Z, F+1
    output `numb` is shaped (M-1, Q-1, Z-1, F)

    Arguments
    ---------
    edges : (4,) array_like  w/ lengths M, Q, Z, F+1
        Grid edges of `mtot`, `mrat`, `redz`, and `freq`
        NOTE: `mtot` should be passed as regular `mtot`, NOT log10(mtot)
              `freq` should be passed as regular `freq`, NOT    ln(freq)
    dnum : (M, Q, Z, F)
        Differential number of binaries, dN/[dlog10M dq qz dlnf] where 'N' is in units of dimensionless number.

    Returns
    -------
    numb : (M-1, Q-1, Z-1, F)

    """

    # each edge should have the same length as the corresponding dimension of `dnum`
    shape = [len(ee) for ee in edges]
    # except the last edge (freq), where `dnum` should be 1-shorter
    shape[-1] -= 1
    assert np.shape(dnum) == tuple(shape)
    # the number will be shaped as one-less the size of each dimension of `dnum`
    new_shape = [sh-1 for sh in dnum.shape]
    # except for the last dimension (freq) which is the same shape
    new_shape[-1] = dnum.shape[-1]

    # prepare output array
    cdef np.ndarray[np.double_t, ndim=4] numb = np.zeros(new_shape)
    # Convert from  mtot => log10(mtot)  and  freq ==> ln(freq)
    ee = [np.log10(edges[0]), edges[1], edges[2], np.diff(np.log(edges[3]))]
    # integrate
    _integrate_differential_number_3dx1d(ee[0], ee[1], ee[2], ee[3], dnum, numb)

    return numb


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void _integrate_differential_number_3dx1d(
    double[:] log10_mtot,
    double[:] mrat,
    double[:] redz,
    double[:] dln_freq,    # actually ln(freq)
    double[:, :, :, :] dnum,
    # output
    double[:, :, :, :] numb
):
    """Integrate the differential number of binaries over each grid bin into total numbers of binaries.

    Trapezoid used over first 3 dims (mtot, mrat, redz), and Riemann over 4th (freq).
    See docstrings in `integrate_differential_number_3dx1d`

    """

    cdef int n_mtot = log10_mtot.size
    cdef int n_mrat = mrat.size
    cdef int n_redz = redz.size
    cdef int n_freq = dln_freq.size

    cdef int mm, qq, zz, ff, ii, jj, kk
    cdef double dm, dmdq, dmdqdz

    for mm in range(n_mtot-1):                              # iterate over output-shape of mass-grid
        dm = log10_mtot[mm+1] - log10_mtot[mm]              # get the bin-length

        for qq in range(n_mrat-1):                          # iterate over output-shape of mass-ratio-grid
            dmdq = dm * (mrat[qq+1] - mrat[qq])             # get the bin-area

            for zz in range(n_redz-1):                      # iterate over output-shape of redsz-grid
                dmdqdz = dmdq * (redz[zz+1] - redz[zz])     # get the bin-volume

                # iterate over output-shape of frequency
                # note that this is a Riemann sum, so input and output dimensions are the same size
                for ff in range(n_freq):
                    temp = 0.0

                    # iterate over each vertex of this cube, averaging the contributions
                    for ii in range(2):                     # mass vertices
                        for jj in range(2):                 # mass-ratio vertices
                            for kk in range(2):             # redshift vertices
                                temp += dnum[mm+ii, qq+jj, zz+kk, ff]

                    numb[mm, qq, zz, ff] = temp * dmdqdz * dln_freq[ff] / 8.0

    return


# ==================================================================================================
# ====    Fixed_Time_2pwl_SAM - Hardening Model    ====
# ==================================================================================================


# @cython.cdivision(True)
# cpdef double _hard_func_2pwl(double norm, double xx, double gamma_inner, double gamma_outer):
#     cdef double dadt = - norm * pow(1.0 + xx, -gamma_outer+gamma_inner) / pow(xx, gamma_inner-1)
#     return dadt


# @cython.cdivision(True)
# cpdef double hard_func_2pwl_gw(
#     double mtot, double mrat, double sepa,
#     double norm, double rchar, double gamma_inner, double gamma_outer
# ):
#     cdef double dadt = _hard_func_2pwl(norm, sepa/rchar, gamma_inner, gamma_outer)
#     dadt += hard_gw(mtot, mrat, sepa)
#     return dadt


@cython.cdivision(True)
cdef double _hard_func_2pwl(double norm, double xx, double gamma_inner, double gamma_outer):
    cdef double dadt = - norm * pow(1.0 + xx, -gamma_outer+gamma_inner) / pow(xx, gamma_inner-1)
    return dadt


@cython.cdivision(True)
cdef double _hard_func_2pwl_gw(
    double mtot, double mrat, double sepa,
    double norm, double rchar, double gamma_inner, double gamma_outer
):
    cdef double dadt = _hard_func_2pwl(norm, sepa/rchar, gamma_inner, gamma_outer)
    dadt += hard_gw(mtot, mrat, sepa)
    return dadt


@cython.cdivision(True)
cdef double[:] _hard_func_2pwl_gw_1darray(
    double[:] mtot, double[:] mrat, double[:] sepa,
    double[:] norm, double[:] rchar, double[:] gamma_inner, double[:] gamma_outer
):
    cdef int ii
    cdef int size = mtot.size
    cdef np.ndarray[np.double_t, ndim=1] dadt = np.zeros(size)
    for ii in range(size):
        dadt[ii] = _hard_func_2pwl(norm[ii], sepa[ii]/rchar[ii], gamma_inner[ii], gamma_outer[ii])
        dadt[ii] += hard_gw(mtot[ii], mrat[ii], sepa[ii])

    return dadt


def hard_func_2pwl_gw(
    mtot, mrat, sepa,
    norm, rchar, gamma_inner, gamma_outer
):
    """

    NOTE: this function will be somewhat slow, because of the explicit broadcasting!

    """
    args = mtot, mrat, sepa, norm, rchar, gamma_inner, gamma_outer
    args = np.broadcast_arrays(*args)
    shape = args[0].shape
    mtot, mrat, sepa, norm, rchar, gamma_inner, gamma_outer = [aa.flatten() for aa in args]
    dadt = _hard_func_2pwl_gw_1darray(mtot, mrat, sepa, norm, rchar, gamma_inner, gamma_outer)
    dadt = np.array(dadt).reshape(shape)
    return dadt


def find_2pwl_hardening_norm(time, mtot, mrat, sepa_init, rchar, gamma_inner, gamma_outer, nsteps):
    assert np.ndim(time) == 0
    assert np.ndim(mtot) == 1
    assert np.shape(mtot) == np.shape(mrat)

    cdef np.ndarray[np.double_t, ndim=1] norm_log10 = np.zeros(mtot.size)

    cdef lifetime_2pwl_params args
    args.target_time = time
    args.sepa_init = sepa_init
    args.rchar = rchar
    args.gamma_inner = gamma_inner
    args.gamma_outer = gamma_outer
    args.nsteps = nsteps

    _get_hardening_norm_2pwl(mtot, mrat, args, norm_log10)

    return norm_log10


ctypedef struct lifetime_2pwl_params:
    double target_time
    double mt
    double mr
    double sepa_init
    double rchar
    double gamma_inner
    double gamma_outer
    int nsteps


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void _get_hardening_norm_2pwl(
    double[:] mtot,
    double[:] mrat,
    lifetime_2pwl_params args,
    # output
    double[:] norm_log10,
):

    cdef double XTOL = 1e-3
    cdef double RTOL = 1e-5
    cdef int MITR = 100    # note: the function doesn't return an error on failure, it still returns last try
    cdef double NORM_LOG10_LO = -20.0
    cdef double NORM_LOG10_HI = +20.0

    cdef int num = mtot.size
    assert mtot.size == mrat.size
    cdef double time

    cdef int ii
    for ii in range(num):
        args.mt = mtot[ii]
        args.mr = mrat[ii]
        norm_log10[ii] = brentq(
            get_binary_lifetime_2pwl, NORM_LOG10_LO, NORM_LOG10_HI,
            <lifetime_2pwl_params *> &args, XTOL, RTOL, MITR, NULL
        )
        # time = get_binary_lifetime_2pwl(norm_log10[ii], <lifetime_2pwl_params *> &args)
        # total_time[ii] = time + args.target_time

    return


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double get_binary_lifetime_2pwl(double norm_log10, void *args):
    cdef lifetime_2pwl_params *pars = <lifetime_2pwl_params *> args

    cdef double risco_log10 = log10(3.0 * MY_SCHW * pars.mt)
    cdef double sepa_log10 = log10(pars.sepa_init)
    cdef double norm = pow(10.0, norm_log10)

    # step-size, in log10-space, to go from sepa_init to ISCO
    cdef double dx = (sepa_log10 - risco_log10) / pars.nsteps
    cdef double time = 0.0

    cdef int ii
    cdef double sepa_right, dadt_right, dt

    cdef double sepa_left = pow(10.0, sepa_log10)
    cdef double dadt_left = _hard_func_2pwl_gw(
        pars.mt, pars.mr, sepa_left,
        norm, pars.rchar, pars.gamma_inner, pars.gamma_outer
    )

    for ii in range(pars.nsteps):
        sepa_log10 -= dx
        sepa_right = pow(10.0, sepa_log10)

        # Get total hardening rate at k+1 edge
        dadt_right = _hard_func_2pwl_gw(
            pars.mt, pars.mr, sepa_right,
            norm, pars.rchar, pars.gamma_inner, pars.gamma_outer
        )

        # Find time to move from left to right
        dt = 2.0 * (sepa_right - sepa_left) / (dadt_left + dadt_right)
        time += dt

        sepa_left = sepa_right
        dadt_left = dadt_right

    time = time - pars.target_time
    return time


def integrate_binary_evolution_2pwl(norm_log10, mtot, mrat, sepa_init, rchar, gamma_inner, gamma_outer, nsteps):
    cdef lifetime_2pwl_params args
    args.mt = mtot
    args.mr = mrat
    args.target_time = 0.0
    args.sepa_init = sepa_init
    args.rchar = rchar
    args.gamma_inner = gamma_inner
    args.gamma_outer = gamma_outer
    args.nsteps = nsteps

    time = get_binary_lifetime_2pwl(norm_log10, <lifetime_2pwl_params *> &args)
    return time


# ==================================================================================================
# ====    Dynamic Binary Number - calculate number of binaries at each frequency    ====
# ==================================================================================================


def dynamic_binary_number_at_fobs(fobs_orb, sam, hard, cosmo):

    dens = sam.static_binary_density

    shape = sam.shape + (fobs_orb.size,)
    cdef np.ndarray[np.double_t, ndim=4] diff_num = np.zeros(shape)
    cdef np.ndarray[np.double_t, ndim=4] redz_final = -1.0 * np.ones(shape)

    # ---- Fixed_Time_2pwl_SAM

    if isinstance(hard, holo.hardening.Fixed_Time_2PL_SAM):
        gmt_time = sam._gmt_time
        # if `sam` is using galaxy merger rate (GMR), then `gmt_time` will be `None`
        if gmt_time is None:
            sam._log.info("`gmt_time` not calculated in SAM.  Setting to zeros.")
            gmt_time = np.zeros(sam.shape)

        _dynamic_binary_number_at_fobs_2pwl(
            fobs_orb, hard._sepa_init, hard._num_steps,
            hard._norm, hard._rchar, hard._gamma_inner, hard._gamma_outer,
            dens, sam.mtot, sam.mrat, sam.redz, gmt_time,
            cosmo._grid_z, cosmo._grid_dcom, cosmo._grid_age,
            # output:
            redz_final, diff_num
        )

    # ---- Hard_GW

    elif isinstance(hard, holo.hardening.Hard_GW) or issubclass(hard, holo.hardening.Hard_GW):
        redz_prime = sam._redz_prime
        # if `sam` is using galaxy merger rate (GMR), then `redz_prime` will be `None`
        if redz_prime is None:
            sam._log.info("`redz_prime` not calculated in SAM.  Setting to `redz` (initial) values.")
            redz_prime = sam.redz[np.newaxis, np.newaxis, :] * np.ones(sam.shape)

        _dynamic_binary_number_at_fobs_gw(
            fobs_orb,
            dens, sam.mtot, sam.mrat, sam.redz, redz_prime,
            cosmo._grid_z, cosmo._grid_dcom,
            # output:
            redz_final, diff_num
        )

    # ---- OTHER

    else:
        raise ValueError(f"Unexpected `hard` value {hard}!")

    return redz_final, diff_num

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef int _dynamic_binary_number_at_fobs_2pwl(
    double[:] target_fobs_orb,
    double sepa_init,
    int num_steps,

    double[:, :] hard_norm,
    double hard_rchar,
    double hard_gamma_inner,
    double hard_gamma_outer,

    double[:, :, :] dens,
    double[:] mtot,
    double[:] mrat,
    double[:] redz,
    double[:, :, :] gmt_time,

    double[:] redz_interp_grid,
    double[:] dcom_interp_grid,
    double[:] tage_interp_grid,

    # output
    double[:, :, :, :] redz_final,
    double[:, :, :, :] diff_num,
) except -1:
    """Convert from binary volume-density (all separations) to binary number at particular frequencies.
    """

    cdef int n_mtot = mtot.size
    cdef int n_mrat = mrat.size
    cdef int n_redz = redz.size
    cdef int n_freq = target_fobs_orb.size
    cdef int n_interp = redz_interp_grid.size
    cdef double age_universe = tage_interp_grid[n_interp - 1]
    cdef double sepa_init_log10 = log10(sepa_init)

    cdef int ii, jj, kk, ff, step, interp_left_idx, interp_right_idx, new_interp_idx
    cdef double mt, mr, norm, risco, dx, new_redz, gmt, ftarget, target_frst_orb
    cdef double sepa_log10, sepa, sepa_left, sepa_right, dadt_left, dadt_right
    cdef double time_evo, redz_left, redz_right, time_left, time_right, new_time
    cdef double frst_orb_left, fobs_orb_left, frst_orb_right, fobs_orb_right

    # ---- Calculate ages corresponding to SAM `redz` grid

    cdef double *redz_age = <double *>malloc(n_redz * sizeof(double))     # (Z,) age of the universe in [sec]
    ii = 0
    cdef int rev
    for kk in range(n_redz):
        # iterate in reverse order to match with `redz_interp_grid` which is decreasing
        rev = n_redz - 1 - kk
        # get to the right index of the interpolation-grid
        while (redz_interp_grid[ii+1] > redz[rev]) and (ii < n_interp - 1):
            ii += 1

        # interpolate
        redz_age[rev] = interp_at_index(ii, redz[rev], redz_interp_grid, tage_interp_grid)

    # ---- calculate dynamic binary numbers for all SAM grid bins

    for ii in range(n_mtot):
        mt = mtot[ii]

        # Determine separation step-size, in log10-space, to integrate from sepa_init to ISCO
        risco = 3.0 * MY_SCHW * mt     # ISCO is 3x combined schwarzschild radius
        dx = (sepa_init_log10 - log10(risco)) / num_steps

        for jj in range(n_mrat):
            mr = mrat[jj]

            # Binary evolution is determined by M and q only
            # so integration is started for each of these bins
            sepa_log10 = sepa_init_log10                # set initial separation to initial value
            norm = hard_norm[ii, jj]                    # get hardening-rate normalization for this bin

            # Get total hardening rate at left-most edge
            sepa_left = pow(10.0, sepa_log10)
            dadt_left = _hard_func_2pwl_gw(
                mt, mr, sepa_left,
                norm, hard_rchar, hard_gamma_inner, hard_gamma_outer
            )

            # get rest-frame orbital frequency of binary at left edge
            frst_orb_left = kepler_freq_from_sepa(mt, sepa_left)

            # ---- Integrate of `num_steps` discrete intervals in binary separation from large to small

            time_evo = 0.0                  # track total binary evolution time
            interp_left_idx = 0                 # interpolation index, will be updated in each step
            for step in range(num_steps):
                # Increment the current separation
                sepa_log10 -= dx
                sepa_right = pow(10.0, sepa_log10)
                frst_orb_right = kepler_freq_from_sepa(mt, sepa_right)

                # Get total hardening rate at the right-edge of this step (left-edge already obtained)
                dadt_right = _hard_func_2pwl_gw(
                    mt, mr, sepa_right,
                    norm, hard_rchar, hard_gamma_inner, hard_gamma_outer
                )

                # Find time to move from left- to right- edges:  dt = da / (da/dt)
                dt = 2.0 * (sepa_right - sepa_left) / (dadt_left + dadt_right)
                time_evo += dt

                # ---- Iterate over starting redshift bins

                for kk in range(n_redz-1, -1, -1):
                    # get the total time from each starting redshift, plus GMT time, plus evolution time to this step
                    gmt = gmt_time[ii, jj, kk]
                    time_right = time_evo + gmt + redz_age[kk]
                    # also get the evolution-time to the left edge
                    time_left = time_right - dt

                    # if we pass the age of the universe, this binary has stalled, no further redshifts will work
                    # NOTE: if `gmt_time` decreases faster than redshift bins increase the universe age,
                    #       then systems in later `redz` bins may no longer stall, so we still need to calculate them
                    #       i.e. we can NOT use a `break` statement here
                    if time_left > age_universe:
                        continue

                    # find the redshift bins corresponding to left- and right- side of step
                    # left edge
                    interp_left_idx = while_while_increasing(interp_left_idx, n_interp, time_left, tage_interp_grid)

                    redz_left = interp_at_index(interp_left_idx, time_left, tage_interp_grid, redz_interp_grid)

                    # double check that left-edge is within age of Universe (should rarely if ever be a problem
                    # but possible due to rounding/interpolation errors
                    if redz_left < 0.0:
                        continue

                    # find right-edge starting from left edge, i.e. `interp_left_idx` (`interp_left_idx` is not a typo!)
                    interp_right_idx = while_while_increasing(interp_left_idx, n_interp, time_right, tage_interp_grid)
                    # NOTE: because `time_right` can be larger than age of universe, it can exceed `tage_interp_grid`
                    #       in this case `interp_right_idx=n_interp-2`, and the `interp_at_index` function can still
                    #       be used to extrapolate to further out values, which will likely be negative

                    redz_right = interp_at_index(interp_right_idx, time_right, tage_interp_grid, redz_interp_grid)
                    # NOTE: at this point `redz_right` could be negative, even though `redz_left` is definitely not
                    if redz_right < 0.0:
                        redz_right = 0.0

                    # convert to frequencies
                    fobs_orb_left = frst_orb_left / (1.0 + redz_left)
                    fobs_orb_right = frst_orb_right / (1.0 + redz_right)

                    # ---- Iterate over all target frequencies

                    # NOTE: there should be a more efficient way to do this.
                    #       Tried a different implementation in `_dynamic_binary_number_at_fobs_1`, but not working
                    #       some of the frequency bins seem to be getting skipped in that version.

                    for ff in range(n_freq):
                        ftarget = target_fobs_orb[ff]

                        # If the integration-step does NOT bracket the target frequency, continue to next frequency
                        if (ftarget < fobs_orb_left) or (fobs_orb_right < ftarget):
                            continue

                        # ------------------------------------------------------
                        # ---- TARGET FOUND ----

                        # At this point in the code, this target frequency is inbetween the left- and right- edges
                        # of the integration step, so we can interpolate the evolution to exactly this frequency,
                        # and perform the actual dynamic_binary_number calculation

                        new_time = _interp_between_vals(ftarget, fobs_orb_left, fobs_orb_right, time_left, time_right)

                        # `time_right` can be after age of Universe, make sure interpolated value is not
                        #    if it is, then all higher-frequencies will also, so break out of target-frequency loop
                        if new_time > tage_interp_grid[n_interp - 1]:
                            break

                        # find index in interpolation grid for this exact time
                        new_interp_idx = interp_left_idx      # start from left-step edge
                        new_interp_idx = while_while_increasing(new_interp_idx, n_interp, new_time, tage_interp_grid)

                        # get redshift
                        new_redz = interp_at_index(new_interp_idx, new_time, tage_interp_grid, redz_interp_grid)
                        # get comoving distance
                        dcom = interp_at_index(new_interp_idx, new_time, tage_interp_grid, dcom_interp_grid)

                        # Store redshift
                        redz_final[ii, jj, kk, ff] = new_redz

                        # find rest-frame orbital frequency and binary separation
                        target_frst_orb = ftarget * (1.0 + new_redz)
                        sepa = kepler_sepa_from_freq(mt, target_frst_orb)

                        # calculate total hardening rate at this exact separation
                        dadt = _hard_func_2pwl_gw(
                            mt, mr, sepa,
                            norm, hard_rchar, hard_gamma_inner, hard_gamma_outer
                        )

                        # calculate residence/hardening time = f/[df/dt] = -(2/3) a/[da/dt]
                        tres = - (2.0/3.0) * sepa / dadt

                        # calculate number of binaries
                        cosmo_fact = FOUR_PI_SPLC_OVER_MPC * (1.0 + new_redz) * pow(dcom / MY_MPC, 2)
                        diff_num[ii, jj, kk, ff] = dens[ii, jj, kk] * tres * cosmo_fact

                        # ----------------------
                        # ------------------------------------------------------

                # update new left edge
                dadt_left = dadt_right
                sepa_left = sepa_right
                frst_orb_left = frst_orb_right
                # note that we _cannot_ do this for redz or freqs because the redshift _bin_ is changing

    free(redz_age)

    return 0


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef int _dynamic_binary_number_at_fobs_gw(
    double[:] target_fobs_orb,

    double[:, :, :] dens,
    double[:] mtot,
    double[:] mrat,
    double[:] redz,
    double[:, :, :] redz_prime,

    double[:] redz_interp_grid,
    double[:] dcom_interp_grid,

    # output
    double[:, :, :, :] redz_final,
    double[:, :, :, :] diff_num,
) except -1:
    """Convert from binary volume-density (all separations) to binary number at particular frequencies.
    """

    cdef int n_mtot = mtot.size
    cdef int n_mrat = mrat.size
    cdef int n_redz = redz.size
    cdef int n_freq = target_fobs_orb.size
    cdef int n_interp = redz_interp_grid.size

    cdef int ii, jj, kk, ff, interp_idx, _kk
    cdef double mt, mr, ftarget, target_frst_orb, sepa, rad_isco, frst_orb_isco, rzp


    # ---- calculate dynamic binary numbers for all SAM grid bins

    for ii in range(n_mtot):
        mt = mtot[ii]
        rad_isco = 3.0 * MY_SCHW * mt
        frst_orb_isco = kepler_freq_from_sepa(mt, rad_isco)

        for jj in range(n_mrat):
            mr = mrat[jj]

            interp_idx = 0
            for _kk in range(n_redz):
                kk = n_redz - 1 - _kk

                # redz_prime is -1 for systems past age of Universe
                rzp = <double>redz_prime[ii, jj, kk]
                if rzp <= 0.0:
                    continue

                for ff in range(n_freq):
                    redz_final[ii, jj, kk, ff] = rzp

                    ftarget = target_fobs_orb[ff]
                    # find rest-frame orbital frequency and binary separation
                    target_frst_orb = ftarget * (1.0 + rzp)
                    # if target frequency is above ISCO freq, then all future ones will be also, so: break
                    if target_frst_orb > frst_orb_isco:
                        break

                    # get comoving distance
                    interp_idx = while_while_decreasing(interp_idx, n_interp, rzp, redz_interp_grid)
                    dcom = interp_at_index(interp_idx, rzp, redz_interp_grid, dcom_interp_grid)

                    # calculate total hardening rate at this exact separation
                    sepa = kepler_sepa_from_freq(mt, target_frst_orb)
                    dadt = hard_gw(mt, mr, sepa)

                    # calculate residence/hardening time = f/[df/dt] = -(2/3) a/[da/dt]
                    tres = - (2.0/3.0) * sepa / dadt

                    # calculate number of binaries
                    cosmo_fact = FOUR_PI_SPLC_OVER_MPC * (1.0 + rzp) * pow(dcom / MY_MPC, 2)
                    diff_num[ii, jj, kk, ff] = dens[ii, jj, kk] * tres * cosmo_fact

    return 0

