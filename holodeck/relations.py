"""Holodeck - Scaling Relations

References
----------
-

To-Do
-----
*[ ]

"""

import abc
# import inspect

# import numba
import numpy as np

# import kalepy as kale

import holodeck as holo
from holodeck import cosmo, utils, log
from holodeck.constants import GYR, SPLC, MSOL, MPC


class _MMBulge_Relation(abc.ABC):

    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        return

    @abc.abstractmethod
    def bulge_mass_frac(self, mstar):
        return

    @abc.abstractmethod
    def mbh_from_mbulge(self, mbulge):
        """Convert from stellar bulge-mass to blackhole mass.

        Units of [grams].
        """
        return

    @abc.abstractmethod
    def dmbh_dmbulge(self, mbulge):
        return

    @abc.abstractmethod
    def dmbulge_dmstar(self, mstar):
        return

    def dmbh_dmstar(self, mstar):
        mbulge = self.mbulge_from_mstar(mstar)
        # (dmbh/dmstar) = (dmbh/dmbulge) * (dmbulge/dmstar)
        dmdm = self.dmbh_dmbulge(mbulge) * self.dmbulge_dmstar(mstar)
        return dmdm

    def mbulge_from_mstar(self, mstar):
        return self.bulge_mass_frac(mstar) * mstar

    def mstar_from_mbulge(self, mbulge):
        return mbulge / self.bulge_mass_frac(mbulge)

    def mbh_from_mstar(self, mstar):
        mbulge = self.mbulge_from_mstar(mstar)
        return self.mbh_from_mbulge(mbulge)

    def mstar_from_mbh(self, mbh):
        mbulge = self.mbulge_from_mbh(mbh)
        return self.mstar_from_mbulge(mbulge)


class MMBulge_Simple(_MMBulge_Relation):
    """Mbh--MBulge relation as a simple power-law.

    M_bh = M_0 * (M_bulge / M_ref)^alpha
    M_bulge = f_bulge * M_star

    """

    def __init__(self, mass_norm=1.48e8*MSOL, mref=1.0e11*MSOL, malpha=1.01, bulge_mfrac=0.615):
        self._mass_norm = mass_norm   # log10(M/Msol) = 8.17  +/-  [-0.32, +0.35]
        self._malpha = malpha         # alpha         = 1.01  +/-  [-0.10, +0.08]

        self._mref = mref             # [Msol]
        self._bulge_mfrac = bulge_mfrac
        return

    def bulge_mass_frac(self, mstar):
        return self._bulge_mfrac

    def mbh_from_mbulge(self, mbulge):
        """Convert from stellar bulge-mass to blackhole mass.

        Units of [grams].

        """
        mbh = self._mass_norm * np.power(mbulge / self._mref, self._malpha)
        return mbh

    def mbulge_from_mbh(self, mbh):
        """Convert from blackhole mass to stellar bulge-mass.

        Units of [grams].
        """
        mbulge = self._mref * np.power(mbh / self._mass_norm, 1/self._malpha)
        return mbulge

    def dmbh_dmbulge(self, mbulge):
        dmdm = self.mbh_from_mbulge(mbulge)
        dmdm = dmdm * self._malpha / mbulge
        return dmdm

    def dmbulge_dmstar(self, mstar):
        # NOTE: this only works for a constant value, do *not* return `self.bulge_mass_frac()`
        return self._bulge_mfrac


