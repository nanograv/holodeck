"""Header file for cython `cyutils.py`
"""

cdef double _interp_between_vals(double xnew, double xl, double xr, double yl, double yr)
cdef double interp_at_index(int idx, double xnew, double[:] xold, double[:] yold)
cdef double gw_freq_dist_func__scalar_scalar(int nn, double ee)


