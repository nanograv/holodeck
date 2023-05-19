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

# ---- My Functions

def gamma_of_rho_interp(rho, rsort, rho_interp_grid, gamma_interp_grid):
    """
    rho : 1Darray of scalars
    rr_sort : 1Darray
        order of rho values smallest to largest
    rho_interp_grid : 1Darray
        rho values corresponding to each gamma
    gamma_interp_grid : 1Darray
        gamma values corresponding to each rho

    """
    # pass in the interp grid
    cdef np.ndarray[np.double_t, ndim=1] gamma = np.zeros(rho.shape)

    _gamma_of_rho_interp(rho, rsort, rho_interp_grid, gamma_interp_grid, gamma)

    return gamma


cdef int _gamma_of_rho_interp(
    double[:] rho, long[:] rsort, 
    double[:] rho_interp_grid, double[:] gamma_interp_grid,
    # output
    double[:] gamma
    ):
    """ Interpolate over gamma grids in sorted rho order to get gamma of each rho.
    """

    cdef int n_rho = rho.size
    cdef int ii, kk, rr
    ii = 0 # get rho in order using rho[rsort[ii]]

    for kk in range(n_rho): 
        rr = rsort[kk] # index of next largest rho

        # get to the right index of the interpolation-grid
        while (rho_interp_grid[ii+1] > rho[rr]) and (ii < n_rho -1):
            ii += 1
        
        # interpolate
        gamma[rsort[ii]] = interp_at_index(ii, rho[rsort[ii], rho_interp_grid, gamma_interp_grid])

    return 0