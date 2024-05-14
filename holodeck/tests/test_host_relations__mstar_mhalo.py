"""Tests for `host_relations.py` holodeck submodule.


"""

import numpy as np
# import scipy as sp
# import scipy.stats

import holodeck as holo
from holodeck import host_relations
# from holodeck.discrete import population
from holodeck.constants import MSOL


class Test_Behroozi_2013:

    def test_init(self):
        host_relations.Behroozi_2013()
        return

    def test_basics(self):
        NUM = 1000
        behr = host_relations.Behroozi_2013()

        mstar = np.random.uniform(5, 12, NUM)
        mstar = MSOL * (10.0 ** mstar)
        redz = np.random.uniform(0.0, 6.0, NUM)
        mstar = np.sort(mstar)
        redz = np.sort(redz)[::-1]

        mhalo = behr.halo_mass(mstar, redz)
        assert np.all(mhalo > 0.0)

        mstar_check = behr.stellar_mass(mhalo, redz)
        mhalo_check = behr.halo_mass(mstar, redz)
        assert np.all(mstar_check > 0.0)
        print(f"mstar  input: {holo.utils.stats(mstar)}")
        print(f"mstar output: {holo.utils.stats(mstar_check)}")
        bads = ~np.isclose(mstar, mstar_check, rtol=0.1)
        if np.any(bads):
            print(f"bad mstar input  : {mstar[bads]/MSOL}")
            print(f"bad mstar output : {mstar_check[bads]/MSOL}")
        assert not np.any(bads)

        print(f"mhalo  input: {holo.utils.stats(mhalo/MSOL)}")
        print(f"mhalo output: {holo.utils.stats(mhalo_check/MSOL)}")
        bads = ~np.isclose(mhalo, mhalo_check, rtol=0.1)
        if np.any(bads):
            print(f"bad mhalo input  : {mhalo[bads]/MSOL}")
            print(f"bad mhalo output : {mhalo_check[bads]/MSOL}")
        assert not np.any(bads)

        return
