"""
"""

cimport cython
import numpy as np
cimport numpy as cnp
cnp.import_array()

from libc.stdio cimport printf
from libc.stdlib cimport malloc, free, realloc
# make sure to use c-native math functions instead of python/numpy
from libc.math cimport pow, sqrt, abs, M_PI, NAN

from holodeck import utils
from holodeck.cyutils cimport interp_at_index, _interp_between_vals

# cdef extern from "numpy/npy_common.h":
#     ctypedef npy_intp

# cdef extern from "numpy/arrayobject.h":
# #     PyObject* PyArray_SimpleNewFromData(int nd, int* dims, int typenum, void* data)
#     void PyArray_INCREF(cnp.ndarray arr)

# int64 type
DTYPE_LONG = np.int64
NPY_DTYPE_LONG = cnp.NPY_INT64
ctypedef cnp.int64_t DTYPE_LONG_t

# float64 type
DTYPE_DOUBLE = np.float64
NPY_DTYPE_DOUBLE = cnp.NPY_FLOAT64
ctypedef cnp.float64_t DTYPE_DOUBLE_t

ctypedef cnp.npy_intp SIZE_t


# ---- Define Constants

cdef double MY_NWTG = 6.6742999e-08
cdef double MY_MSOL = 1.988409870698e+33
cdef double MY_SPLC = 29979245800.0
cdef double MY_MPC = 3.08567758e+24
cdef double MY_YR = 31557600.0

# freq = KEPLER_CONST * sqrt(mass) / pow(sepa, 1.5)
cdef double KEPLER_CONST = sqrt(MY_NWTG) / 2.0 / M_PI


cdef struct ArrayData:
    SIZE_t final_size
    DTYPE_LONG_t* bin
    DTYPE_LONG_t* target
    double* m1
    double* m2
    double* redz
    double* eccen
    double* dadt
    double* dedt


def interp_at_fobs(evo, fobs):
    fobs = np.atleast_1d(fobs)
    assert fobs.ndim == 1

    dadt = np.sum(evo.dadt, axis=-1)
    dedt = np.sum(evo.dedt, axis=-1)
    assert evo._first_index.dtype == DTYPE_LONG
    # _interp_at_fobs(evo._first_index, evo._last_index, fobs, evo.sepa, evo.eccen, evo.redz, dadt, dedt, evo.mass)

    return


def test__interp_at_fobs_0():
    target_fobs = [
        1.0602e-6/MY_YR,
        0.167/MY_YR, 0.168/MY_YR, 0.3501/MY_YR,
        1.1802e2/MY_YR
    ]
    target_fobs = np.atleast_1d(target_fobs)

    mass_0 = np.array([3.24e9*MY_MSOL, 8.987e8*MY_MSOL])
    fobs_0 = [
        1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 7e-3, 8e-3, 9e-3,
        1e-2, 2e-2, 3e-2, 4e-2, 5e-2, 6e-2, 7e-2, 8e-2, 9e-2,
        1e-1, 2e-1, 3e-1, 4e-1, 5e-1, 6e-1, 7e-1, 8e-1, 9e-1,
        1e+0, 2e+0, 3e+0, 4e+0, 5e+0, 6e+0, 7e+0, 8e+0, 9e+0,
        1e+1,
    ]
    fobs_0 = np.asarray(fobs_0) / MY_YR
    redz_0 = np.sort(np.random.uniform(0.0, 2.0, fobs_0.size))
    frst_0 = fobs_0 * (1.0 + redz_0)
    sepa_0 = utils.kepler_sepa_from_freq(np.sum(mass_0), frst_0)
    mass_0 = np.ones((redz_0.size, 2)) * mass_0[np.newaxis, :]

    mass_1 = np.array([1.0e9*MY_MSOL, 1.0e8*MY_MSOL])
    fobs_1 = [
        1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 7e-3, 8e-3, 9e-3,
        1e-2, 2e-2, 3e-2, 4e-2, 5e-2, 6e-2, 7e-2, 8e-2, 9e-2,
        1e-1, 9e-2, 8e-2, 7e-2, 6e-2, 5e-2, 4e-2, 3e-2, 2e-2,
        1e-2, 2e-2, 3e-2, 4e-2, 5e-2, 6e-2, 7e-2, 8e-2, 9e-2,
        1e-1, 2e-1, 3e-1, 4e-1, 5e-1, 6e-1, 7e-1, 8e-1, 9e-1,
        1e+0, 9e-1, 8e-1, 7e-1, 6e-1, 5e-1, 4e-1, 3e-1, 2e-1,
        1e-1, 2e-1, 3e-1, 4e-1, 5e-1, 6e-1, 7e-1, 8e-1, 9e-1,
        1e+0, 2e+0, 3e+0, 4e+0, 5e+0, 6e+0, 7e+0, 8e+0, 9e+0,
        1e+1,
    ]
    fobs_1 = np.asarray(fobs_1) / MY_YR
    redz_1 = np.sort(np.random.uniform(0.0, 2.0, fobs_1.size))
    frst_1 = fobs_1 * (1.0 + redz_1)
    sepa_1 = utils.kepler_sepa_from_freq(np.sum(mass_1), frst_1)
    mass_1 = np.ones((redz_1.size, 2)) * mass_1[np.newaxis, :]

    sepa = np.concatenate([sepa_0, sepa_1])
    redz = np.concatenate([redz_0, redz_1])
    mass = np.concatenate([mass_0, mass_1])
    first_index = np.array([0, redz_0.size+1])
    last_index = np.array([redz_0.size-1, sepa.size-1])
    eccen = np.zeros_like(sepa)
    dadt = np.zeros_like(sepa)
    dedt = np.zeros_like(sepa)

    # _interp_at_fobs(first_index, last_index, target_fobs, sepa, eccen, redz, dadt, dedt, mass)
    return


def test__interp_at_fobs_1():
    target_fobs = [
        1.0602e-6/MY_YR,
        0.167/MY_YR, 0.168/MY_YR, 0.3501/MY_YR,
        1.1802e2/MY_YR
    ]
    target_fobs = np.atleast_1d(target_fobs)

    mass = np.array([1.0e9*MY_MSOL, 1.0e8*MY_MSOL])
    fobs = [
        1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 7e-3, 8e-3, 9e-3,
        1e-2, 2e-2, 3e-2, 4e-2, 5e-2, 6e-2, 7e-2, 8e-2, 9e-2,
        1e-1, 9e-2, 8e-2, 7e-2, 6e-2, 5e-2, 4e-2, 3e-2, 2e-2,
        1e-2, 2e-2, 3e-2, 4e-2, 5e-2, 6e-2, 7e-2, 8e-2, 9e-2,
        1e-1, 2e-1, 3e-1, 4e-1, 5e-1, 6e-1, 7e-1, 8e-1, 9e-1,
        1e+0, 9e-1, 8e-1, 7e-1, 6e-1, 5e-1, 4e-1, 3e-1, 2e-1,
        1e-1, 2e-1, 3e-1, 4e-1, 5e-1, 6e-1, 7e-1, 8e-1, 9e-1,
        1e+0, 2e+0, 3e+0, 4e+0, 5e+0, 6e+0, 7e+0, 8e+0, 9e+0,
        1e+1,
    ]
    fobs = np.asarray(fobs) / MY_YR
    redz = np.sort(np.random.uniform(0.0, 2.0, fobs.size))
    frst = fobs * (1.0 + redz)
    sepa = utils.kepler_sepa_from_freq(np.sum(mass), frst)
    mass = np.ones((redz.size, 2)) * mass[np.newaxis, :] * np.linspace(1.0, 2.0, redz.size)[:, np.newaxis]

    eccen = np.zeros_like(sepa)
    dadt = np.zeros_like(sepa)
    dedt = np.zeros_like(sepa)

    first_index = np.asarray([0])
    last_index = np.asarray([sepa.size-1])

    cdef ArrayData data

    _interp_at_fobs(first_index, last_index, target_fobs, sepa, eccen, redz, dadt, dedt, mass, &data)

    cdef cnp.ndarray[DTYPE_LONG_t, ndim=1, mode="c"] bin = cnp.PyArray_SimpleNewFromData(
        1, &data.final_size, NPY_DTYPE_LONG, <void*>data.bin
    )
    cdef cnp.ndarray[DTYPE_LONG_t, ndim=1, mode="c"] target = cnp.PyArray_SimpleNewFromData(
        1, &data.final_size, NPY_DTYPE_LONG, <void*>data.target
    )
    cdef cnp.ndarray[DTYPE_DOUBLE_t, ndim=1, mode="c"] m1 = cnp.PyArray_SimpleNewFromData(
        1, &data.final_size, NPY_DTYPE_DOUBLE, <void*>data.m1
    )

    return bin, target, m1


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void _interp_at_fobs(
    DTYPE_LONG_t[:] beg_index,
    DTYPE_LONG_t[:] end_index,
    double[:] target_fobs,
    double[:] sepa,
    double[:] eccen,
    double[:] redz,
    double[:] dadt,
    double[:] dedt,
    double[:, :] mass,
    ArrayData* data,
):

    # Initialize sizes and shapes
    cdef DTYPE_LONG_t nbins = len(beg_index)
    # cdef DTYPE_LONG_t nentries = len(sepa)
    cdef DTYPE_LONG_t nfreqs = len(target_fobs)

    cdef DTYPE_LONG_t arr_size = nfreqs * nbins
    cdef DTYPE_LONG_t arr_size_tot = arr_size

    data.bin = <DTYPE_LONG_t *>malloc(arr_size * sizeof(DTYPE_LONG_t))         # binary index number
    data.target = <DTYPE_LONG_t *>malloc(arr_size * sizeof(DTYPE_LONG_t))         # target fobs index number
    data.m1 = <double *>malloc(arr_size * sizeof(double))         # target fobs index number
    data.m2 = <double *>malloc(arr_size * sizeof(double))         # target fobs index number
    data.redz = <double *>malloc(arr_size * sizeof(double))         # target fobs index number
    data.eccen = <double *>malloc(arr_size * sizeof(double))         # target fobs index number
    data.dadt = <double *>malloc(arr_size * sizeof(double))         # target fobs index number
    data.dedt = <double *>malloc(arr_size * sizeof(double))         # target fobs index number

    cdef DTYPE_LONG_t bin, left, right, fi, idx, direction, last_direction
    cdef DTYPE_LONG_t out = 0
    cdef DTYPE_LONG_t beg, end
    cdef double fobs_l, fobs_r

    for bin in range(nbins):
        # get the first and last index (both inclusive) for this binary
        beg = beg_index[bin]
        end = end_index[bin]
        # printf("\n---- bin %04ld [%04ld, %04ld]\n\n", bin, beg, end)

        # assume that we'll start out hardening (frequency increasing)
        direction = +1
        last_direction = +1

        # initialize to the first target frequency
        fi = 0
        # initialize left edge of step to the first entry for this binary
        left = beg
        # get the observer-frame frequency at the left edge
        fobs_l = get_fobs(sepa[left], mass[left, 0]+mass[left, 1], redz[left])
        # increment over steps, make sure final `right` value *includes* `end` (hence `end+1`)
        for right in range(left+1, end+1):

            # make sure arrays have enough space
            # for a given integration-step, we can match at most `nfreqs`, so thats as much extra space as we need
            if out + nfreqs >= arr_size_tot:
                printf("out+nfreqs=%ld+%ld=%ld >= %ld, resizing\n", out, nfreqs, out+nfreqs, arr_size_tot)
                arr_size_tot += arr_size
                data.bin = <DTYPE_LONG_t *>realloc(data.bin, arr_size_tot * sizeof(DTYPE_LONG_t))
                data.target = <DTYPE_LONG_t *>realloc(data.target, arr_size_tot * sizeof(DTYPE_LONG_t))
                data.m1 = <double *>realloc(data.m1, arr_size_tot * sizeof(double))
                data.m2 = <double *>realloc(data.m2, arr_size_tot * sizeof(double))
                data.redz = <double *>realloc(data.redz, arr_size_tot * sizeof(double))
                data.eccen = <double *>realloc(data.eccen, arr_size_tot * sizeof(double))
                data.dadt = <double *>realloc(data.dadt, arr_size_tot * sizeof(double))
                data.dedt = <double *>realloc(data.dedt, arr_size_tot * sizeof(double))

            # get observer-frame freq at right edge
            fobs_r = get_fobs(sepa[right], mass[right, 0]+mass[right, 1], redz[right])
            # printf("%03ld :: %03ld ==> %03ld :: %.4e, %.4e\n", left-beg, left, right, fobs_l, fobs_r)

            # we are moving to higher frequencies = hardening
            if fobs_l < fobs_r:
                direction = +1

                # If we change directions, to decreasing in frequency, reset to the largest target frequency
                #! I don't think this is necessary (?), but it definitely works keeping it...  [LZK:2023-09-26]
                if direction != last_direction:
                    fi = 0
                    # printf("| changed direction - ==> +  :: {%ld}\n", fi)

                # increment target frequency until it's above the left-edge (it may still be above right-edge also)
                while (fobs_l > target_fobs[fi]) and (fi+1 < nfreqs):
                    fi += 1

                # printf("  next target +++ : %03ld @ %.4e\n", fi, target_fobs[fi])

                # if target is above left-edge
                if fobs_l < target_fobs[fi]:
                    # if target is below right-edge, we have a match - time to interpolate
                    while (target_fobs[fi] < fobs_r) and (fi+1 < nfreqs):
                        # printf("* %.4e %.4e %.4e ==> target=%04ld - %04ld\n", fobs_l, target_fobs[fi], fobs_r, fi, out)
                        # set output to show which binary and which target frequency this match corresponds to

                        data.bin[out] = bin
                        data.target[out] = fi
                        # interpolate desired values
                        data.m1[out] = _interp_between_vals(
                            target_fobs[fi], fobs_l, fobs_r, mass[left, 0], mass[right, 0]
                        )
                        # bin_at[out] = bin
                        # target_at[out] = fi
                        # # interpolate desired values
                        # m1_at[out] = _interp_between_vals(
                        #     target_fobs[fi], fobs_l, fobs_r, mass[left, 0], mass[right, 0]
                        # )
                        # m2_at[out] = _interp_between_vals(
                        #     target_fobs[fi], fobs_l, fobs_r, mass[left, 1], mass[right, 1]
                        # )
                        # redz_at[out] = _interp_between_vals(
                        #     target_fobs[fi], fobs_l, fobs_r, redz[left], redz[right]
                        # )
                        # dadt_at[out] = _interp_between_vals(
                        #     target_fobs[fi], fobs_l, fobs_r, dadt[left], dadt[right]
                        # )
                        # dedt_at[out] = _interp_between_vals(
                        #     target_fobs[fi], fobs_l, fobs_r, dedt[left], dedt[right]
                        # )
                        # eccen_at[out] = _interp_between_vals(
                        #     target_fobs[fi], fobs_l, fobs_r, eccen[left], eccen[right]
                        # )

                        # increment the output array index pointer
                        out += 1
                        # go up to the next target frequency
                        fi += 1

            # we are moving to lower frequencies = softening
            elif fobs_l > fobs_r:
                direction = -1

                # If we change directions, to decrease in frequency, reset to the largest target frequency
                #! I don't think this is necessary (?), but it definitely works keeping it...  [LZK:2023-09-26]
                if direction != last_direction:
                    fi = nfreqs - 1
                    # printf("| changed direction + ==> -  :: {%ld}\n", fi)

                # decrement target frequency until it's below
                while (fobs_l < target_fobs[fi]) and (fi > 0):
                    fi -= 1

                # printf("  next target --- : %03ld @ %.4e\n", fi, target_fobs[fi])

                # if target is above left-edge
                if fobs_l > target_fobs[fi]:
                    # if target is below right-edge, we have a match - time to interpolate
                    while (target_fobs[fi] > fobs_r) and (fi > 0):
                        # printf("* %.4e %.4e %.4e ==> target=%04ld - %04ld\n", fobs_l, target_fobs[fi], fobs_r, fi, out)

                        data.bin[out] = bin
                        data.target[out] = fi
                        # interpolate desired values
                        data.m1[out] = _interp_between_vals(
                            target_fobs[fi], fobs_l, fobs_r, mass[left, 0], mass[right, 0]
                        )
                        # set output to show which binary and which target frequency this match corresponds to
                        # bin_at[out] = bin
                        # target_at[out] = fi

                        # # interpolate desired values
                        # m1_at[out] = _interp_between_vals(
                        #     target_fobs[fi], fobs_l, fobs_r, mass[left, 0], mass[right, 0]
                        # )
                        # m2_at[out] = _interp_between_vals(
                        #     target_fobs[fi], fobs_l, fobs_r, mass[left, 1], mass[right, 1]
                        # )
                        # redz_at[out] = _interp_between_vals(
                        #     target_fobs[fi], fobs_l, fobs_r, redz[left], redz[right]
                        # )
                        # dadt_at[out] = _interp_between_vals(
                        #     target_fobs[fi], fobs_l, fobs_r, dadt[left], dadt[right]
                        # )
                        # dedt_at[out] = _interp_between_vals(
                        #     target_fobs[fi], fobs_l, fobs_r, dedt[left], dedt[right]
                        # )
                        # eccen_at[out] = _interp_between_vals(
                        #     target_fobs[fi], fobs_l, fobs_r, eccen[left], eccen[right]
                        # )

                        # increment the output array index pointer
                        out += 1

                        # go down to the next target frequency
                        fi -= 1

                else:
                    printf("\n\nERROR: `discrete_cyutils._interp_at_fobs()` :: ")
                    printf(
                        "fobs_l==fobs_r (%e, %e)!  bin=%ld, left=%ld, right=%ld\n\n",
                        fobs_l, fobs_r, bin, left, right
                    )

            fobs_l = fobs_r
            left = right
            last_direction = direction


    data.final_size = out
    # downsize arrays to amount of space used
    data.bin = <DTYPE_LONG_t *>realloc(data.bin, out * sizeof(DTYPE_LONG_t))
    data.target = <DTYPE_LONG_t *>realloc(data.target, out * sizeof(DTYPE_LONG_t))
    data.m1 = <double *>realloc(data.m1, out * sizeof(double))
    data.m2 = <double *>realloc(data.m2, out * sizeof(double))
    data.redz = <double *>realloc(data.redz, out * sizeof(double))
    data.eccen = <double *>realloc(data.eccen, out * sizeof(double))
    data.dadt = <double *>realloc(data.dadt, out * sizeof(double))
    data.dedt = <double *>realloc(data.dedt, out * sizeof(double))
    # printf("arr_size = %ld (tot = %ld), out =  %ld\n", arr_size, arr_size_tot, out)

    return


cdef double get_fobs(double sepa, double mass, double redz):
    cdef double fobs = KEPLER_CONST * sqrt(mass)/pow(sepa, 1.5)
    fobs /= (1.0 + redz)
    return fobs








