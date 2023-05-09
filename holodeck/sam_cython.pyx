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
from libc.math cimport pow, sqrt, M_PI, NAN, log10

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
cdef double hard_func(double norm, double xx, double gamma_inner, double gamma_outer):
    cdef double dadt = - norm * pow(1.0 + xx, -gamma_outer+gamma_inner) / pow(xx, gamma_inner-1)
    return dadt


@cython.cdivision(True)
cdef double hard_gw(double mtot, double mrat, double sepa):
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
cdef int while_while(int start, int size, double val, double[:] edges):
    cdef int index = start
    while (index < size - 1) and (edges[index+1] < val):
        index += 1
    while (index > 0) and (edges[index-1] > val):
        index -= 1
    return index


cdef double integrate_binary_evolution(norm, mt, mr, sepa_init, rchar, gamma_inner, gamma_outer, int nsteps):
    cdef double risco_log10 = log10(3.0 * MY_SCHW * mt)
    cdef double sepa_log10 = log10(sepa_init)

    # step-size, in log10-space, to go from sepa_init to ISCO
    cdef double dx = (sepa_log10 - risco_log10) / nsteps
    cdef double time = 0.0

    cdef int ii
    cdef double sepa_right, dadt_right, dt

    cdef double sepa_left = pow(10.0, sepa_log10)
    cdef double dadt_left = hard_func(norm, sepa_left/rchar, gamma_inner, gamma_outer)
    dadt_left += hard_gw(mt, mr, sepa_left)

    for ii in range(nsteps):
        sepa_log10 -= dx
        sepa_right = pow(10.0, sepa_log10)

        # Get total hardening rate at k+1 edge
        dadt_right = hard_func(norm, sepa_right/rchar, gamma_inner, gamma_outer)
        dadt_right += hard_gw(mt, mr, sepa_right)

        # Find time to move from left to right
        dt = 2.0 * (sepa_right - sepa_left) / (dadt_left + dadt_right)
        time += dt

    return time



def find_hardening_norm(time, sam, hard, nsteps=100):

    cdef np.ndarray[np.double_t, ndim=2] norm = np.zeros((sam.mtot.size, sam.mrat.size))

    _find_hardening_norm(
        time, sam.mtot, sam.mrat,
        hard._rchar, hard._sepa_init, hard._gamma_inner, hard._gamma_outer,
        nsteps,
        norm,
    )

    return


cdef void _find_hardening_norm(
    double time,
    double[:] mtot,
    double[:] mrat,
    double rchar,
    double sepa_init,
    double gamma_inner,
    double gamma_outer,
    int nsteps,
    # output
    double[:, :] norm,
):

    # cdef double inner_func(double nn, double mt, double mr):
    #     return integrate_binary_evolution(nn, mt, mr, sepa_init, rchar, gamma_inner, gamma_outer, nsteps)

    cdef double XTOL = 1e-3
    cdef double RTOL = 1e-3
    cdef double MITR = 10

    cdef int nm = mtot.size
    cdef int nq = mrat.size
    cdef int ii, jj
    for ii in range(nm):
        for jj in range(nq):
            # inner_func(1.0, mtot[ii], mrat[jj])
            pass
            # brentq(
            #     f, xa, xb, <test_params *> &myargs, XTOL, RTOL, MITR, NULL)

    return



def dynamic_binary_number(fobs_orb, sam, hard, cosmo, nsteps):
    dens = sam.static_binary_density
    # convert from flattened array back to grid shape  (M*Q*Z ==> (M, Q, Z,))
    norm = hard._norm.reshape(sam.shape)

    shape = sam.shape + (fobs_orb.size,)
    num_shape = tuple([ss-1 for ss in sam.shape]) + (fobs_orb.size,)
    cdef np.ndarray[np.double_t, ndim=4] redz_final = NAN * np.ones(shape)
    cdef np.ndarray[np.double_t, ndim=4] diff_num = np.zeros(shape)
    # cdef np.ndarray[np.double_t, ndim=4] number = np.zeros(num_shape)

    _dynamic_binary_number(
        fobs_orb, norm, hard._rchar, hard._gamma_inner, hard._gamma_outer,
        dens, sam.mtot, sam.mrat, sam.redz, sam._gmt_time, 1.0e4 * MY_PC, nsteps,
        cosmo._grid_z, cosmo._grid_dcom, cosmo._grid_age,

        redz_final, diff_num,
    )

    return redz_final, diff_num


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void _dynamic_binary_number(
    # input
    double[:] target_fobs_orb,
    double[:, :, :] hard_norm,
    double hard_rchar,
    double hard_gamma_inner,
    double hard_gamma_outer,
    double[:, :, :] dens,
    double[:] mtot,
    double[:] mrat,
    double[:] redz,
    double[:, :, :] gmt_time,
    double sepa_init,
    int num_steps,
    double[:] redz_interp_grid,
    double[:] dcom_interp_grid,
    double[:] tage_interp_grid,

    # output
    double[:, :, :, :] redz_final,
    double[:, :, :, :] diff_num,
):

    cdef int nf = target_fobs_orb.size
    cdef int nm = mtot.size
    cdef int nq = mtot.size
    cdef int nz = mtot.size
    cdef int ninterp = redz_interp_grid.size
    cdef double age_universe = tage_interp_grid[ninterp - 1]
    cdef double sepa_init_log10 = log10(sepa_init)

    cdef int ii, jj, kk, ss, fidx, interp_left, interp_right, ff
    cdef double mt, mr, rz, norm
    cdef double risco, risco_log10, dx, newz
    cdef double sepa_log10, sepa, sepa_left, sepa_right, dadt_left, dadt_right
    cdef double time_evo, redz_left, redz_right, time_left, time_right
    cdef double frst_orb_left, fobs_orb_left, frst_orb_right, fobs_orb_right, gmt, ftemp
    cdef double target_frst_orb

    # ---- Calculate ages corresponding to `redz`

    cdef double *redz_age = <double *>malloc(nz * sizeof(double))     # (Z,) age of the universe in [sec]
    ii = 0
    cdef int rev
    for kk in range(nz):
        # iterate in reverse order to match with `redz_interp_grid` which is decreasing
        rev = nz - 1 - kk
        while (redz_interp_grid[ii] > redz[rev]) and (ii < ninterp - 1):
            ii += 1
        redz_age[rev] = interp_at_index(ii, redz[rev], redz_interp_grid, tage_interp_grid)

    # ---- Integrate

    for ii in range(nm):
        mt = mtot[ii]
        risco = 3.0 * MY_SCHW * mt     # ISCO is 3x combined schwarzschild radius
        risco_log10 = log10(risco)
        # step-size, in log10-space, to go from sepa_init to ISCO
        dx = (sepa_init_log10 - risco_log10) / num_steps
        for jj in range(nq):
            mr = mrat[jj]
            sepa_log10 = sepa_init_log10
            norm = hard_norm[ii, jj, 0]   # redshift doesn't change normalization, so grad 0th redshift

            # Get total hardening rate at 0th edge
            sepa_left = pow(10.0, sepa_log10)
            dadt_left = hard_func(norm, sepa_left/hard_rchar, hard_gamma_inner, hard_gamma_outer)
            dadt_left += hard_gw(mt, mr, sepa_left)

            # get rest-frame orbital frequency of binary at left edge
            frst_orb_left = kepler_freq_from_sepa(mt, sepa_left)

            time_evo = 0.0
            interp_left = 0
            fidx = 0
            for ss in range(num_steps):
                # Increment the current separation
                sepa_log10 -= dx
                sepa_right = pow(10.0, sepa_log10)
                frst_orb_right = kepler_freq_from_sepa(mt, sepa_right)

                # Get total hardening rate at k+1 edge
                dadt_right = hard_func(norm, sepa_right/hard_rchar, hard_gamma_inner, hard_gamma_outer)
                dadt_right += hard_gw(mt, mr, sepa_right)

                # Find time to move from left to right
                dt = 2.0 * (sepa_right - sepa_left) / (dadt_left + dadt_right)
                time_evo += dt

                # ---- Iterate over starting redshift bins
                for kk in range(nz-1, -1, -1):

                    # get the total time from each starting redshift, plus GMT time, plus evolution time to this step
                    gmt = gmt_time[ii, jj, kk]
                    time_left = (time_evo - dt) + gmt + redz_age[kk]
                    time_right = time_evo + gmt + redz_age[kk]

                    # if we pass the age of the universe, this binary has stalled, no further redshifts will work
                    # NOTE: if `gmt_time` decreases too fast, then system won't necessary always stall,
                    #       so we cant use a `break` statement here
                    if time_left > age_universe:
                        continue

                    # ---- find the redshift bins corresponding to left- and right- side of step
                    # left edge
                    interp_left = while_while(interp_left, ninterp, time_left, tage_interp_grid)
                    # find right-edge starting from left edge, i.e. `interp_left` (i.e. `interp_left` is not a typo!)
                    interp_right = while_while(interp_left, ninterp, time_left, tage_interp_grid)
                    # get redshifts
                    redz_left = interp_at_index(interp_left, time_left, tage_interp_grid, redz_interp_grid)
                    redz_right = interp_at_index(interp_right, time_right, tage_interp_grid, redz_interp_grid)

                    # redz_left = redz[kk]
                    # redz_right = redz[kk]

                    # convert to frequencies
                    fobs_orb_left = frst_orb_left / (1.0 + redz_left)
                    fobs_orb_right = frst_orb_right / (1.0 + redz_right)

                    # find the index for target frequencies bounding the left edge
                    #    note: `fidx=0` may mean all targets are still above `fobs_orb_left`; check for this below
                    fidx = while_while(fidx, nf, fobs_orb_left, target_fobs_orb)

                    # iterate over all target frequencies between left and right step-edges; interpolate to redshift
                    ff = fidx
                    ftemp = target_fobs_orb[ff]
                    while (fobs_orb_left < ftemp) and (ftemp < fobs_orb_right) and (ff < nf):

                        # ----------------------
                        # ---- TARGET FOUND ----

                        newz = _interp_between_vals(ftemp, fobs_orb_left, fobs_orb_right, redz_left, redz_right)
                        if newz > 0.0:

                            redz_final[ii, jj, kk, ff] = newz

                            target_frst_orb = ftemp * (1.0 + newz)
                            sepa = kepler_sepa_from_freq(mt, target_frst_orb)

                            # dadt : interpolate left-right
                            # dadt = _interp_between_vals(ftemp, fobs_orb_left, fobs_orb_right, dadt_left, dadt_right)
                            dadt = hard_func(norm, sepa/hard_rchar, hard_gamma_inner, hard_gamma_outer)
                            dadt += hard_gw(mt, mr, sepa)

                            # calculate residence/hardening time = f/[df/dt] = -(2/3) a/[da/dt]
                            tres = - (2.0/3.0) * sepa / dadt
                            # get comoving distance values at left and right step-edges
                            dcom_left = interp_at_index(interp_left, time_left, tage_interp_grid, dcom_interp_grid)
                            dcom_right = interp_at_index(interp_right, time_right, tage_interp_grid, dcom_interp_grid)
                            # dcom : interpolate left-right
                            dcom = _interp_between_vals(ftemp, fobs_orb_left, fobs_orb_right, dcom_left, dcom_right)

                            cosmo_fact = FOUR_PI_SPLC_OVER_MPC * (1.0 + newz) * pow(dcom / MY_MPC, 2)
                            diff_num[ii, jj, kk, ff] = dens[ii, jj, kk] * tres * cosmo_fact

                        # ----------------------

                        ff += 1
                        if ff < nf:
                            ftemp = target_fobs_orb[ff]

                # update new left edge
                dadt_left = dadt_right
                sepa_left = sepa_right
                frst_orb_left = frst_orb_right
                # note that we _cannot_ do this for redz or freqs because the redshift bin is changing

    free(redz_age)

    return
