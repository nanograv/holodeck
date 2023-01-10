"""
"""

import numpy as np
cimport scipy.special.cython_special as sp_special


cdef double ECCEN_ZERO_LIMIT = 1.0e-12


cdef double bessel_recursive(int nn, double ne, double jn_m1, double jn_m2):

    # NOTE: do _NOT_ check if ne is zero here, do it before calculating ANY bessel functions
    # if abs(ne) < 1e-16:
    #     if nn == 0:
    #         return 1.0
    #     return 0.0

    cdef double jn = (2*(nn-1) / ne) * jn_m1 - jn_m2
    return jn


cdef api double gw_freq_dist_func__scalar_scalar(int nn, double ee):
    if abs(ee) < ECCEN_ZERO_LIMIT:
        if nn == 2:
            return 1.0

        return 0.0

    cdef double jn, jn_p1, jn_p2, jn_m1, jn_m2, aa, bb, cc, gg
    cdef double ne = nn*ee
    cdef double n2 = np.square(nn)

    jn_m2 = sp_special.jv(nn-2, ne)
    jn_m1 = sp_special.jv(nn-1, ne)

    jn = bessel_recursive(nn, ne, jn_m1, jn_m2)
    jn_p1 = bessel_recursive(nn+1, ne, jn, jn_m1)
    jn_p2 = bessel_recursive(nn+2, ne, jn_p1, jn)

    aa = np.square(jn_m2 - 2.0*ee*jn_m1 + (2/nn)*jn + 2*ee*jn_p1 - jn_p2)
    bb = (1 - ee*ee)*np.square(jn_m2 - 2*ee*jn + jn_p2)
    cc = (4.0/(3.0*n2)) * np.square(jn)
    gg = (n2*n2/32) * (aa + bb + cc)
    return gg


cdef api double[:] gw_freq_dist_func__scalar_array(int nn, double[:] eccen):
    cdef Py_ssize_t esize = eccen.shape[0]

    # Calculate with non-zero eccentrictiy
    cdef double ee, jn, jn_p1, jn_p2, jn_m1, jn_m2, aa, bb, cc
    # cdef double ne = nn*ee`
    cdef double ne, n2 = np.square(nn)

    gne = np.zeros(esize)

    cdef Py_ssize_t ii
    for ii in range(esize):
        ee = eccen[ii]
        if abs(ee) < ECCEN_ZERO_LIMIT:
            if nn == 2:
                gne[ii] = 1.0
            else:
                gne[ii] = 0.0

            continue

        ne = nn * ee
        jn_m2 = sp_special.jv(nn-2, ne)
        jn_m1 = sp_special.jv(nn-1, ne)

        jn = bessel_recursive(nn, ne, jn_m1, jn_m2)
        jn_p1 = bessel_recursive(nn+1, ne, jn, jn_m1)
        jn_p2 = bessel_recursive(nn+2, ne, jn_p1, jn)

        aa = np.square(jn_m2 - 2.0*ee*jn_m1 + (2/nn)*jn + 2*ee*jn_p1 - jn_p2)
        bb = (1 - ee*ee)*np.square(jn_m2 - 2*ee*jn + jn_p2)
        cc = (4.0/(3.0*n2)) * np.square(jn)
        gne[ii] = (n2*n2/32) * (aa + bb + cc)

    return gne
