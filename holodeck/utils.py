"""
"""

import copy

import numpy as np
import h5py

from .constants import NWTG, SCHW


def a_to_z(scfa):
    redz = (1.0 / scfa) - 1.0
    return redz


def z_to_a(redz):
    scfa = 1.0 / (redz + 1.0)
    return scfa


def load_hdf5(fname, keys=None):
    squeeze = False
    if (keys is not None) and np.isscalar(keys):
        keys = [keys]
        squeeze = True

    header = dict()
    data = dict()
    with h5py.File(fname, 'r') as h5:
        head_keys = h5.attrs.keys()
        for kk in head_keys:
            header[kk] = copy.copy(h5.attrs[kk])

        if keys is None:
            keys = h5.keys()

        for kk in keys:
            data[kk] = h5[kk][:]

    if squeeze:
        data = data[kk]

    return header, data


def log_normal_base_10(mu, sigma, size=None, shift=0.0):
    _sigma = np.log(10**sigma)
    dist = np.random.lognormal(np.log(mu) + shift*np.log(10.0), _sigma, size)
    return dist


def mtmr_from_m1m2(m1, m2=None):
    if m2 is not None:
        masses = np.stack([m1, m2], axis=-1)
    else:
        assert np.shape(m1)[-1] == 2, "If only `m1` is given, last dimension must be 2!"
        masses = np.asarray(m1)

    mtot = masses.sum(axis=-1)
    mrat = masses.min(axis=-1) / masses.max(axis=-1)
    return np.array([mtot, mrat])


def m1m2_from_mtmr(mt, mr):
    """Convert from total-mass and mass-ratio to individual masses.
    """
    mt = np.asarray(mt)
    mr = np.asarray(mr)
    m1 = mt/(1.0 + mr)
    m2 = mt - m1
    return np.array([m1, m2])


def kepler_freq_from_sep(mass, sep):
    freq = (1.0/(2.0*np.pi))*np.sqrt(NWTG*mass)/np.power(sep, 1.5)
    return freq


def kepler_sep_from_freq(mass, freq):
    sep = np.power(NWTG*mass/np.square(2.0*np.pi*freq), 1.0/3.0)
    return sep


def rad_isco(m1, m2, factor=3.0):
    """Inner-most Stable Circular Orbit, radius at which binaries 'merge'.
    """
    return factor * schwarzschild_radius(m1+m2)


def schwarzschild_radius(mass):
    rs = SCHW * mass
    return rs
