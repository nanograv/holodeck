# import numpy as np
# import pyximport   # noqa
# pyximport.install(language_level=3, setup_args={"include_dirs": np.get_include()}, reload_support=True)

# import cytest







from datetime import datetime
import numpy as np
import holodeck as holo
from holodeck import cosmo
from holodeck.constants import PC, YR

import zcode.math as zmath

# NHARMS = 23
# SAM_SHAPE = (10, 11, 12)
NHARMS = 7
SAM_SHAPE = (4, 5, 6)

INIT_ECCEN = 0.999
INIT_SEPA = 10.0 * PC


def sam_evolve_eccen_uniform_single(sam, eccen_init, sepa_init, nsteps=300):
    assert (0.0 <= eccen_init) and (eccen_init <= 1.0)

    eccen = np.zeros(nsteps)
    eccen[0] = eccen_init

    sepa_max = sepa_init
    sepa_coal = holo.utils.schwarzschild_radius(sam.mtot) * 3
    # frst_coal = utils.kepler_freq_from_sepa(sam.mtot, sepa_coal)
    sepa_min = sepa_coal.min()
    sepa = np.logspace(*np.log10([sepa_max, sepa_min]), nsteps)

    for step in range(1, nsteps):
        a0 = sepa[step-1]
        a1 = sepa[step]
        da = (a1 - a0)
        e0 = eccen[step-1]

        _, e1 = zmath.numeric.rk4_step(holo.hardening.Hard_GW.deda, x0=a0, y0=e0, dx=da)
        e1 = np.clip(e1, 0.0, None)
        eccen[step] = e1

    return sepa, eccen


sam = holo.sam.Semi_Analytic_Model(shape=SAM_SHAPE)
dcom = cosmo.comoving_distance(sam.redz).to('Mpc').value
print("evolve")
sepa_evo, eccen_evo = sam_evolve_eccen_uniform_single(sam, INIT_ECCEN, INIT_SEPA)

print("interp and gwb")
gwfobs = np.logspace(-2, 1, 10) / YR

# gwfobs_harms, hc2, ecc_out, tau_out
edges = [np.log10(sam.mtot), sam.mrat, sam.redz]

dur = datetime.now()
rv_1 = holo.gravwaves.sam_calc_gwb_1(
    sam.static_binary_density, *edges, dcom,
    gwfobs, sepa_evo, eccen_evo, nharms=NHARMS
)
dur = datetime.now() - dur
print("1: ", dur.total_seconds())

gwb_1 = rv_1

dur = datetime.now()
rv_0 = holo.gravwaves.sam_calc_gwb_0(gwfobs, sam, sepa_evo, eccen_evo, nharms=NHARMS)
dur = datetime.now() - dur
print("0: ", dur.total_seconds())

gwb_0 = rv_0[1]
# gwb_0 = rv_0[3]

gwb_0 = np.sqrt(gwb_0)
gwb_1 = np.sqrt(gwb_1)
print(gwb_0.shape, gwb_1.shape)

retval = np.allclose(gwb_0, gwb_1, atol=0.0)
print(retval, np.median(gwb_0), np.median(gwb_1))

if not retval:
    for idx, ref in np.ndenumerate(gwb_0):
        check = gwb_1[idx]
        print(idx, ref, check, np.isclose(ref, check))



# import numpy as np

# import pyximport   # This is part of Cython
# pyximport.install(language_level=3)

# import holodeck.gravwaves

# holodeck.gravwaves.tester_1()


# import holodeck
# import holodeck.cyutils
# # print(f"{holodeck.cyutils.gw_freq_dist_func__scalar_scalar(5, 0.565)=}")

# aa = np.random.uniform(0.0, 1.0, 2)
# # print(f"{holodeck.cyutils.gw_freq_dist_func__scalar_array(5, aa)=}")


# from numba import njit
# import ctypes
# from numba.extending import get_cython_function_address

# addr = get_cython_function_address("holodeck.cyutils", "gw_freq_dist_func__scalar_scalar")
# functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_int, ctypes.c_double)
# func = functype(addr)


# @njit
# def tester():
#     print("numba")

#     # print(holodeck.cyutils.gw_freq_dist_func__scalar_scalar(4, 0.565))
#     print(func(4, 0.565))
#     # print(holodeck.cyutils.gw_freq_dist_func__scalar_array(5, aa))

# tester()