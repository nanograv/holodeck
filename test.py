import numpy as np

import pyximport   # This is part of Cython
pyximport.install(language_level=3)

import holodeck
import holodeck.cyutils


# print(f"{holodeck.cyutils.gw_freq_dist_func__scalar_scalar(5, 0.565)=}")

aa = np.random.uniform(0.0, 1.0, 2)
# print(f"{holodeck.cyutils.gw_freq_dist_func__scalar_array(5, aa)=}")


from numba import njit
import ctypes
from numba.extending import get_cython_function_address

addr = get_cython_function_address("holodeck.cyutils", "gw_freq_dist_func__scalar_scalar")
functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_int, ctypes.c_double)
func = functype(addr)


@njit
def tester():
    print("numba")

    # print(holodeck.cyutils.gw_freq_dist_func__scalar_scalar(4, 0.565))
    print(func(4, 0.565))
    # print(holodeck.cyutils.gw_freq_dist_func__scalar_array(5, aa))

tester()