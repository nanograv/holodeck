"""Galaxy/Halo structure profiles (e.g. density and velocity).

References
----------
* [Behroozi2013]_ : Behroozi, Wechsler & Conroy 2013.
* [Guo2010]_ Guo, White, Li & Boylan-Kolchin 2010.
* [Klypin2016]_ Klypin et al. 2016.
* [KH2013]_ Kormendy & Ho 2013.
* [MM2013]_ McConnell & Ma 2013.
* [NFW1997]_ Navarro, Frenk & White 1997.

"""

import abc
# from typing import Type, Union

import numpy as np
from numpy.typing import ArrayLike
import scipy as sp

from holodeck import cosmo
from holodeck.constants import MSOL, NWTG

__all__ = [
    "Klypin_2016", "NFW",
]

class Klypin_2016:
    """Class to calculate dark matter halo 'concentration' parameters based on [Klypin2016]_.

    This class does not need to be instantiated, all methods are class methods, simply call
    ``Klypin_2016.concentration()``.

    Interpolate between redshifts and masses to find DM halo concentrations.
    [Klypin2016]_ Eq. 24 & Table 2.

    """
    _redz = [0.00e+00, 3.50e-01, 5.00e-01, 1.00e+00, 1.44e+00,
             2.15e+00, 2.50e+00, 2.90e+00, 4.10e+00, 5.40e+00]
    _c0 = [7.40e+00, 6.25e+00, 5.65e+00, 4.30e+00, 3.53e+00,
           2.70e+00, 2.42e+00, 2.20e+00, 1.92e+00, 1.65e+00]
    _gamma = [1.20e-01, 1.17e-01, 1.15e-01, 1.10e-01, 9.50e-02,
              8.50e-02, 8.00e-02, 8.00e-02, 8.00e-02, 8.00e-02]
    _mass0 = [5.50e+05, 1.00e+05, 2.00e+04, 9.00e+02, 3.00e+02,
              4.20e+01, 1.70e+01, 8.50e+00, 2.00e+00, 3.00e-01]

    _interp = lambda xx, yy: sp.interpolate.interp1d(xx, yy, kind='linear', fill_value='extrapolate')
    _zz = np.log10(1 + np.array(_redz))
    _lin_interp_c0 = _interp(_zz, np.log10(_c0))
    _lin_interp_gamma = _interp(_zz, np.log10(_gamma))
    _lin_interp_mass0 = _interp(_zz, np.log10(_mass0)+np.log10(1e12 * MSOL / cosmo.h))

    @classmethod
    def _c0(cls, redz):
        xx = np.log10(1 + redz)
        yy = np.power(10.0, cls._lin_interp_c0(xx))
        return yy

    @classmethod
    def _gamma(cls, redz):
        xx = np.log10(1 + redz)
        yy = np.power(10.0, cls._lin_interp_gamma(xx))
        return yy

    @classmethod
    def _mass0(cls, redz):
        xx = np.log10(1 + redz)
        yy = np.power(10.0, cls._lin_interp_mass0(xx))
        return yy

    @classmethod
    def concentration(cls, mhalo: ArrayLike, redz: ArrayLike) -> ArrayLike:
        """Return the halo concentration for the given halo mass and redshift.

        Parameters
        ----------
        mhalo : ArrayLike
            Halo mass.  [grams]
        redz : ArrayLike
            Redshift.

        Returns
        -------
        conc : ArrayLike
            Halo concentration parameters.  []

        """
        c0 = cls._c0(redz)
        gamma = cls._gamma(redz)
        mass0 = cls._mass0(redz)
        f1 = np.power(mhalo/(1e12*MSOL/cosmo.h), -gamma)
        f2 = 1 + np.power(mhalo/mass0, 0.4)
        conc = c0 * f1 * f2
        return conc


class _Density_Profile(abc.ABC):
    """Base class for implementing an arbitrary radial density profile (typically of galaxies).
    """

    @abc.abstractmethod
    def density(self, rads: ArrayLike, *args, **kwargs) -> ArrayLike:
        """Return the density at the given radii.

        Parameters
        ----------
        rads : ArrayLike
            Desired radial distances.  [cm]

        Returns
        -------
        density : ArrayLike
            Densities at the given radii.  [g/cm^3]

        """

    @classmethod
    def time_dynamical(cls, rads, *args, **kwargs):
        """Return the dynamical time, defined as :math:`(G M_enc / r^3) ^ -1/2 = r / v_circ`.

        Parameters
        ----------
        rads : ArrayLike
            Desired radial distances.  [cm]

        Returns
        -------
        tden : ArrayLike
            Dynamical times at the given radii.  [sec]

        """
        tdyn = rads / cls.velocity_circular(rads, *args, **kwargs)
        return tdyn

    @abc.abstractmethod
    def mass(cls, rads, *args, **kwargs):
        """Calculate the mass enclosed out to the given radii.

        Parameters
        ----------
        rads : ArrayLike
            Desired radial distances.  [cm]

        Returns
        -------
        mass : ArrayLike
            Enclosed masses at the given radii.  [gram]

        """
        pass

    '''
    @classmethod
    def mass(cls, rads, *args, **kwargs):
        dens = cls.density(rads, *args, **kwargs)
        yy = 4*np.pi*rads**2 * dens
        mass = utils.trapz_loglog(yy, rads)
        m0 = dens[0] * (4.0/3.0) * np.pi * rads[0] ** 3
        mass = np.concatenate([[m0], mass + m0])
        return mass
    '''

    @classmethod
    def velocity_circular(cls, rads, *args, **kwargs):
        """Circular velocity, defined as :math:`(G M_enc / r) ^ 1/2`.

        Parameters
        ----------
        rads : ArrayLike
            Desired radial distances.  [cm]

        Returns
        -------
        velo : ArrayLike
            Velocities at the given radii.  [cm/s]

        """
        mass = cls.mass(rads, *args, **kwargs)
        velo = NWTG * mass / rads
        velo = velo ** 0.5
        return velo


class NFW(_Density_Profile):
    """Navarro, Frank & White dark-matter density profile from [NFW1997]_.
    """

    @staticmethod
    def density(rads: ArrayLike, mhalo: ArrayLike, redz: ArrayLike) -> ArrayLike:
        """NFW DM Density profile.

        Parameters
        ----------
        rads : ArrayLike
            Target radial distances.  [cm]
        mhalo : ArrayLike
            Halo mass.  [grams]
        redz : ArrayLike
            Redshift.    []

        Returns
        -------
        dens : ArrayLike
            Densities at the given radii.  [g/cm^3]

        """
        rho_s, rs = NFW._nfw_rho_rad(mhalo, redz)
        dens = rads / rs
        dens = dens * np.square(1 + dens)
        dens = rho_s / dens
        return dens

    @staticmethod
    def mass(rads: ArrayLike, mhalo: ArrayLike, redz: ArrayLike) -> ArrayLike:
        """DM mass enclosed at the given radii from an NFW profile.

        Parameters
        ----------
        rads : ArrayLike
            Target radial distances.  [cm]
        mhalo : ArrayLike
            Halo mass.  [gram]
        redz : ArrayLike
            Redshift.    []

        Returns
        -------
        mass : ArrayLike
            Mass enclosed within the given radii.  [gram]

        """
        rads, mhalo, redz = np.broadcast_arrays(rads, mhalo, redz)
        # Get Halo concentration
        rho_s, rs = NFW._nfw_rho_rad(mhalo, redz)
        # NOTE: Expression causes numerical problems for rads/rs <~ 1e-8
        # only use proper analytic expression in safe regime ("hi")
        # use small radius approximation for unsafe regime ("lo")
        lo = (rads/rs < 1e-6)
        hi = ~lo
        xx = (rs[hi] + rads[hi]) / rs[hi]
        xx = np.log(xx) + 1.0/xx - 1.0
        mass = np.zeros_like(rads)
        mass[hi] = 4.0 * np.pi * rho_s[hi] * rs[hi]**3 * xx
        mass[lo] = 2.0 * np.pi * rho_s[lo] * rads[lo]**2 * rs[lo]
        return mass

    @staticmethod
    def _concentration(mhalo, redz):
        return Klypin_2016.concentration(mhalo, redz)

    @staticmethod
    def _nfw_rho_rad(mhalo, redz):
        """Return the DM halo parameters for characteristic density and halo scale radius.

        Parameters
        ----------
        mhalo : ArrayLike
            Halo mass.  [grams]
        redz : ArrayLike
            Redshift.

        Returns
        -------
        rho_s : ArrayLike
            DM halo characteristic density.   [g/cm^3]
        rs : ArrayLike
            Scale radius of the DM halo.  [cm]

        """
        conc = NFW._concentration(mhalo, redz)
        log_c_term = np.log(1 + conc) - conc/(1+conc)

        # Critical over-density
        delta_c = (200/3) * (conc**3) / log_c_term
        # NFW density (*not* the density at the characteristic-radius)
        rho_s = cosmo.critical_density(redz).cgs.value * delta_c
        # scale-radius
        rs = mhalo / (4*np.pi*rho_s*log_c_term)
        rs = np.power(rs, 1.0/3.0)
        return rho_s, rs

    @staticmethod
    def radius_scale(mhalo: ArrayLike, redz: ArrayLike) -> ArrayLike:
        """Return the DM-halo scale radius.

        Parameters
        ----------
        mhalo : ArrayLike
            Halo mass.  [grams]
        redz : ArrayLike
            Redshift.

        Returns
        -------
        rs : ArrayLike
            Scale radius of the DM halo.  [cm]

        """
        rs = NFW._nfw_rho_rad(mhalo, redz)[1]
        return rs

    @staticmethod
    def density_characteristic(mhalo: ArrayLike, redz: ArrayLike) -> ArrayLike:
        """Return the DM halo parameters for characteristic density.

        Parameters
        ----------
        mhalo : ArrayLike
            Halo mass.  [grams]
        redz : ArrayLike
            Redshift.

        Returns
        -------
        rho_s : ArrayLike
            DM halo characteristic density.   [g/cm^3]

        """
        rs = NFW._nfw_rho_rad(mhalo, redz)[0]
        return rs


