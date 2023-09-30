"""
"""

import numpy as np
import holodeck as holo
import holodeck.discrete_cyutils
from holodeck import utils
from holodeck.constants import YR, MSOL

# function to check eccentricity evolution and make sure smaller
# timesteps are taken when binary is close to equilibrium 
# (see sign change in abdot around q=1, eb=0.4)
def test_ecc_evol():
    pass


def test_interp_at_fobs_1():
    target_fobs = [
        1.0602e-6/YR,
        0.150/YR, 0.151/YR, 0.3501/YR,
        1.1802e2/YR
    ]
    target_fobs = np.atleast_1d(target_fobs)

    m0 = np.array([1.0e9*MSOL, 1.0e8*MSOL])
    fobs = [
        1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 7e-3, 8e-3, 9e-3,
        1e-2, 2e-2, 3e-2, 4e-2, 5e-2, 6e-2, 7e-2, 8e-2, 9e-2,

        1e-1, 9e-2, 8e-2, 7e-2, 6e-2, 5e-2, 4e-2, 3e-2, 2e-2,
        1e-2, 2e-2, 3e-2, 4e-2, 5e-2, 6e-2, 7e-2, 8e-2, 9e-2,
        1e-1, 2e-1, 3e-1, 4e-1, 5e-1, 6e-1, 7e-1, 8e-1, 9e-1,
        #   1,2          3

        1e+0, 9e-1, 8e-1, 7e-1, 6e-1, 5e-1, 4e-1, 3e-1, 2e-1,
        #                                        3,          2,1
        1e-1, 2e-1, 3e-1, 4e-1, 5e-1, 6e-1, 7e-1, 8e-1, 9e-1,
        #   1,2,          3
        1e+0, 2e+0, 3e+0, 4e+0, 5e+0, 6e+0, 7e+0, 8e+0, 9e+0,
    ]
    mm = [
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,

        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.1, 1.2, 1.3, 1.4, 1.5, 1.5, 1.5, 1.5, 1.5,

        1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.7, 1.8, 1.9,
        2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.5, 2.5, 2.5,
        2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5,
    ]
    fobs = np.asarray(fobs) / YR
    redz = np.sort(np.random.uniform(0.0, 0.0, fobs.size))
    frst = fobs * (1.0 + redz)
    mass = np.ones((redz.size, 2)) * m0[np.newaxis, :] * np.array(mm)[:, np.newaxis]
    sepa = utils.kepler_sepa_from_freq(np.sum(mass, axis=1), frst)

    answer_targets = [
        1, 2, 3,
        3, 2, 1,
        1, 2, 3,
    ]
    answer_mass_lo = [
        1.1, 1.1, 1.3,
        1.7, 1.9, 1.9,
        2.0, 2.0, 2.2,
    ]

    eccen = np.zeros_like(sepa)
    dadt = np.zeros_like(sepa)
    dedt = np.zeros_like(sepa)

    first_index = np.asarray([0])
    last_index = np.asarray([sepa.size-1])

    class Dummy:
        pass

    evo = Dummy()
    evo._first_index = first_index
    evo._last_index = last_index
    evo.sepa = sepa
    evo.eccen = eccen
    evo.redz = redz
    evo.dadt = [sepa]
    evo.dedt = [dedt]
    evo.mass = mass

    bin, target, m1, m2, redz, eccen, dadt, dedt = holo.discrete_cyutils.interp_at_fobs(evo, target_fobs)

    assert len(target) == len(answer_targets)
    for ii in range(len(target)):
        lo_0 = answer_mass_lo[ii] * m0[0]
        hi_0 = float((answer_mass_lo[ii] + 0.1) * m0[0])
        lo_1 = answer_mass_lo[ii] * m0[1]
        hi_1 = float((answer_mass_lo[ii] + 0.1) * m0[1])
        print(f"{target[ii]} ({answer_targets[ii]})")
        print(f"{lo_0/MSOL:.3e}, {m1[ii]/MSOL:.3e}, {hi_0/MSOL:.3e}")
        print(f"{lo_1/MSOL:.3e}, {m2[ii]/MSOL:.3e}, {hi_1/MSOL:.3e}")

        assert target[ii] == answer_targets[ii]
        assert (lo_0 < m1[ii]) and (m1[ii] < hi_0)
        assert (lo_1 < m2[ii]) and (m2[ii] < hi_1)
        if np.allclose(redz, 0.0, atol=0.1, rtol=0.0):
            med = 0.5 * (lo_0 + hi_0)
            assert np.isclose(m1[ii], med, rtol=0.05)
            med = 0.5 * (lo_1 + hi_1)
            assert np.isclose(m2[ii], med, rtol=0.05)

    return


def test_interp_at_fobs_2():
    target_fobs = [
        0.15/YR, 0.35/YR, 0.55/YR, 1.5/YR,
    ]
    target_fobs = np.atleast_1d(target_fobs)

    m0 = np.array([1.0e6*MSOL, 1.0e7*MSOL])
    fobs = [
        8.0e-2, 9.0e-2, 1.0e-1, 9.0e-2, 8.0e-2, 7.0e-2, 8.0e-2, 9.0e-2,
        1.0e-1, 2.0e-1, 3.0e-1, 4.0e-1, 5.0e-1, 6.0e-1, 7.0e-1, 8.0e-1, 9.0e-1,
        #     0               1                2
        1.0e+0, 9.0e-1, 8.0e-1, 7.0e-1, 6.0e-1, 5.0e-1, 4.0e-1, 3.0e-1, 4.0e-1,
        #                                      2              1        1
        5.0e-1, 6.0e-1, 7.0e-1, 8.0e-1, 9.0e-1, 1.0e+0, 2.0e+0, 3.0e+0
        #     2                                       3
    ]
    mm = [
        1.0e+0, 1.0e+0, 1.0e+0, 1.0e+0, 1.0e+0, 1.0e+0, 1.0e+0, 1.0e+0,
        1.1e+0, 1.2e+0, 1.3e+0, 1.4e+0, 1.5e+0, 1.6e+0, 1.7e+0, 1.8e+0, 1.9e+0,
        #     0               1                2
        2.0e+0, 2.1e+0, 2.2e+0, 2.3e+0, 2.4e+0, 2.5e+0, 2.6e+0, 2.7e+0, 2.8e+0,
        #                                      2              1        1
        2.9e+0, 3.0e+0, 3.1e+0, 3.2e+0, 3.3e+0, 3.4e+0, 3.5e+0, 3.5e+0,
        #     2                                       3
    ]
    fobs = np.asarray(fobs) / YR
    redz = np.sort(np.random.uniform(0.0, 0.0, fobs.size))
    frst = fobs * (1.0 + redz)
    mass = np.ones((redz.size, 2)) * m0[np.newaxis, :] * np.array(mm)[:, np.newaxis]
    sepa = utils.kepler_sepa_from_freq(np.sum(mass, axis=1), frst)

    answer_targets = [
        0, 1, 2,
        2, 1, 1,
        2, 3,
    ]
    answer_mass_lo = [
        1.1, 1.3, 1.5,
        2.4, 2.6, 2.7,
        2.9, 3.4,
    ]

    eccen = np.zeros_like(sepa)
    dadt = np.zeros_like(sepa)
    dedt = np.zeros_like(sepa)

    first_index = np.asarray([0])
    last_index = np.asarray([sepa.size-1])

    class Dummy:
        pass

    evo = Dummy()
    evo._first_index = first_index
    evo._last_index = last_index
    evo.sepa = sepa
    evo.eccen = eccen
    evo.redz = redz
    evo.dadt = [sepa]
    evo.dedt = [dedt]
    evo.mass = mass

    bin, target, m1, m2, redz, eccen, dadt, dedt = holo.discrete_cyutils.interp_at_fobs(evo, target_fobs)

    print(f"truth  ={answer_targets}")
    print(f"test   ={target}")
    assert len(target) == len(answer_targets)
    for ii in range(len(target)):
        lo_0 = answer_mass_lo[ii] * m0[0]
        hi_0 = float((answer_mass_lo[ii] + 0.1) * m0[0])
        lo_1 = answer_mass_lo[ii] * m0[1]
        hi_1 = float((answer_mass_lo[ii] + 0.1) * m0[1])
        print(f"{target[ii]} ({answer_targets[ii]})")
        print(f"{lo_0/MSOL:.3e}, {m1[ii]/MSOL:.3e}, {hi_0/MSOL:.3e}")
        print(f"{lo_1/MSOL:.3e}, {m2[ii]/MSOL:.3e}, {hi_1/MSOL:.3e}")

        assert target[ii] == answer_targets[ii]
        assert (lo_0 < m1[ii]) and (m1[ii] < hi_0)
        assert (lo_1 < m2[ii]) and (m2[ii] < hi_1)
        if np.allclose(redz, 0.0, atol=0.1, rtol=0.0):
            med = 0.5 * (lo_0 + hi_0)
            assert np.isclose(m1[ii], med, rtol=0.05)
            med = 0.5 * (lo_1 + hi_1)
            assert np.isclose(m2[ii], med, rtol=0.05)

    return


