"""Holodeck - Scaling Relations

References
----------
-   Kormendy+Ho-2013 (1304.7762) = [KH13]
    Coevolution (Or Not) of Supermassive Black Holes and Host Galaxies
    https://ui.adsabs.harvard.edu/abs/2013ARA%26A..51..511K/abstract

-   McConnell+Ma-2013 (1211.2816) = [MM13]
    Revisiting the Scaling Relations of Black Hole Masses and Host Galaxy Properties
    https://ui.adsabs.harvard.edu/abs/2013ApJ...764..184M/abstract

* [NFW-97] : Navarro, Frenk & White 1997
    A Universal Density Profile from Hierarchical Clustering
    https://ui.adsabs.harvard.edu/abs/1997ApJ...490..493N/abstract

* [Guo-2010] : Guo, White, Li & Boylan-Kolchin 2010
    How do galaxies populate dark matter haloes?
    https://ui.adsabs.harvard.edu/abs/2010MNRAS.404.1111G/abstract

* [Behroozi+2013] : Behroozi, Wechsler & Conroy 2013
    The Average Star Formation Histories of Galaxies in Dark Matter Halos from z = 0-8
    https://ui.adsabs.harvard.edu/abs/2013ApJ...770...57B/abstract

* [Klypin+2016] : Klypin et al. 2016
    MultiDark simulations: the story of dark matter halo concentrations and density profiles
    https://ui.adsabs.harvard.edu/abs/2016MNRAS.457.4340K/abstract


To-Do
-----
*[ ]Pass concentration-relation (or other method to calculate) to NFW classes on instantiation
*[ ]

"""

import abc

import numpy as np
import scipy as sp

from holodeck import cosmo, utils, log
from holodeck.constants import MSOL, NWTG, KMPERSEC


class _MHost_Relation(abc.ABC):

    _PROPERTIES = []

    def get_host_properties(self, pop, copy=True):
        """Get the host properties specified in the `_PROPERTIES` list of variable names.

        NOTE: if the `copy` flag is False, then values are returned by reference and the original
        values may be modified accidentally.  Use `copy=True` to avoid modifying values in place.
        Use `copy=False` carefully, when trying to avoid unnecessary memory duplication.

        Parameters
        ----------
        pop : `holodeck.population.Population` instance
            Binary population including necessary host properties.
        copy : bool, optional
            Copy the `pop` data into new arrays instead of returning references to original values.

        Returns
        -------
        vals : dict
            Values loaded from `pop`.  Names/keys correspond to `_PROPERTIES` strings.

        """
        func = np.copy if copy else (lambda xx: xx)
        try:
            vals = {var: func(getattr(pop, var)) for var in self._PROPERTIES}
        except Exception as err:
            msg = f"{self.__class__} failed to load properties from {pop}: {self._PROPERTIES}"
            log.error(msg)
            log.error(str(err))
            raise err

        return vals

    # ---- Abstract Methods : must be overridden in subclasses  ----

    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        return

    @abc.abstractmethod
    def mbh_from_host(self, host):
        """Convert from host galaxy properties to blackhole mass.

        Units of [grams].
        """
        return


class _MMBulge_Relation(_MHost_Relation):

    _PROPERTIES = ['mbulge']

    # ---- Abstract Methods : must be overridden in subclasses  ----
    @abc.abstractmethod
    def bulge_mass_frac(self, mstar):
        return

    @abc.abstractmethod
    def dmbh_dmbulge(self, mbulge):
        return

    @abc.abstractmethod
    def dmbulge_dmstar(self, mstar):
        return

    # ---- Internal Methods ----

    def dmbh_dmstar(self, mstar):
        mbulge = self.mbulge_from_mstar(mstar)
        # (dmbh/dmstar) = (dmbh/dmbulge) * (dmbulge/dmstar)
        dmdm = self.dmbh_dmbulge(mbulge) * self.dmbulge_dmstar(mstar)
        return dmdm

    def mbulge_from_mstar(self, mstar):
        return self.bulge_mass_frac(mstar) * mstar

    def mstar_from_mbulge(self, mbulge):
        return mbulge / self.bulge_mass_frac(mbulge)

    def mbh_from_mstar(self, mstar, scatter):
        mbulge = self.mbulge_from_mstar(mstar)
        return self.mbh_from_host({'mbulge': mbulge}, scatter)

    def mstar_from_mbh(self, mbh, **kwargs):
        mbulge = self.mbulge_from_mbh(mbh, **kwargs)
        return self.mstar_from_mbulge(mbulge)


class MMBulge_Standard(_MMBulge_Relation):
    """
    """

    MASS_AMP = 3.0e8 * MSOL
    MASS_PLAW = 1.0
    MASS_REF = 1.0e11 * MSOL
    SCATTER_DEX = 0.0

    def __init__(self, mamp=None, mplaw=None, mref=None, bulge_mfrac=0.615, scatter_dex=None):
        if mamp is None:
            mamp = self.MASS_AMP
        if mplaw is None:
            mplaw = self.MASS_PLAW
        if mref is None:
            mref = self.MASS_REF
        if scatter_dex is None:
            scatter_dex = self.SCATTER_DEX

        self._mamp = mamp   # Mass-Amplitude [grams]
        self._mplaw = mplaw   # Mass Power-law index
        self._mref = mref   # Reference Mass (argument normalization)
        self._bulge_mfrac = bulge_mfrac
        self._scatter_dex = scatter_dex
        return

    def bulge_mass_frac(self, mstar):
        return self._bulge_mfrac

    def mbh_from_host(self, pop, scatter):
        """Convert from stellar bulge-mass to blackhole mass.

        Units of [grams].
        """
        if scatter:
            scatter_dex = self._scatter_dex
        else:
            scatter_dex = None

        host = self.get_host_properties(pop)
        mbulge = host['mbulge']

        mbh = _log10_relation(mbulge, self._mamp, self._mplaw, scatter_dex, x0=self._mref)
        return mbh

    def mbulge_from_mbh(self, mbh, scatter):
        """Convert from blackhole mass to stellar bulge-mass.

        Units of [grams].
        """
        if scatter:
            scatter_dex = self._scatter_dex
        else:
            scatter_dex = None

        mbulge = _log10_relation_reverse(mbh, self._mamp, self._mplaw, scatter_dex, x0=self._mref)
        return mbulge

    def dmbh_dmbulge(self, mbulge):
        # NOTE: scatter should never be used in the differential relation
        dmdm = self.mbh_from_host({'mbulge': mbulge}, scatter=False)
        dmdm = dmdm * self._mplaw / mbulge
        return dmdm

    def dmbulge_dmstar(self, mstar):
        # NOTE: this only works for a constant value, do *not* return `self.bulge_mass_frac()`
        return self._bulge_mfrac


class MMBulge_Redshift(_MMBulge_Relation):
    """
    Provides black hole mass as a function of galaxy bulge mass and redshift with a normalization
    that depends on redshift. zplaw=0 (default) is identical to MMBulge_Standard.
    mamp = mamp0 * (1 + z)**zplaw

    """

    MASS_AMP = 3.0e8 * MSOL
    MASS_PLAW = 1.0
    MASS_REF = 1.0e11 * MSOL
    SCATTER_DEX = 0.0
    Z_PLAW = 0.0

    _PROPERTIES = ['mbulge', 'redz']

    def __init__(self, mamp=None, mplaw=None, mref=None, bulge_mfrac=0.615, scatter_dex=None, zplaw=None):
        if mamp is None:
            mamp = self.MASS_AMP
        if mplaw is None:
            mplaw = self.MASS_PLAW
        if mref is None:
            mref = self.MASS_REF
        if scatter_dex is None:
            scatter_dex = self.SCATTER_DEX
        if zplaw is None:
            zplaw = self.Z_PLAW

        self._mamp = mamp   # Mass-Amplitude [grams]
        self._mplaw = mplaw   # Mass Power-law index
        self._mref = mref   # Reference Mass (argument normalization)
        self._bulge_mfrac = bulge_mfrac
        self._scatter_dex = scatter_dex
        self._zplaw = zplaw
        return

    def bulge_mass_frac(self, mstar):
        return self._bulge_mfrac

    def mbh_from_host(self, pop, scatter):
        """Convert from stellar bulge-mass to blackhole mass.

        Units of [grams].
        """
        if scatter:
            scatter_dex = self._scatter_dex
        else:
            scatter_dex = None

        host = self.get_host_properties(pop, copy=False)
        mbulge = host['mbulge']    # shape (N, 2)
        redz = host['redz']        # shape (N,)
        # Broadcast `redz` to match shape of `mbulge`, if needed
        # NOTE: this will work for (N,) ==> (N,)    or   (N,) ==> (N,X)
        redz = np.broadcast_to(redz, mbulge.T.shape).T

        zmamp = self._mamp * (1.0 + redz)**self._zplaw
        mbh = _log10_relation(mbulge, zmamp, self._mplaw, scatter_dex, x0=self._mref)
        return mbh

    def mbulge_from_mbh(self, mbh, redz, scatter):
        """Convert from blackhole mass to stellar bulge-mass.

        Units of [grams].
        """
        if scatter:
            scatter_dex = self._scatter_dex
        else:
            scatter_dex = None

        zmamp = self._mamp * (1.0 + redz)**self._zplaw
        mbulge = _log10_relation_reverse(mbh, zmamp, self._mplaw, scatter_dex, x0=self._mref)
        return mbulge

    def dmbh_dmbulge(self, mbulge):
        # NOTE: scatter should never be used in the differential relation
        dmdm = self.mbh_from_mbulge(mbulge, scatter=False)
        dmdm = dmdm * self._mplaw / mbulge
        return dmdm

    def dmbulge_dmstar(self, mstar):
        # NOTE: this only works for a constant value, do *not* return `self.bulge_mass_frac()`
        return self._bulge_mfrac


class MMBulge_Strawman(_MMBulge_Relation):
    """
    Provides a broken M_BH--M_bulge relation to test error checking.
    Do not use this for any purpose other than error testing.

    """

    MASS_AMP = 3.0e8 * MSOL
    MASS_PLAW = 1.0
    MASS_REF = 1.0e11 * MSOL
    SCATTER_DEX = 0.0
    Z_PLAW = 0.0

    def __init__(self, mamp=None, mplaw=None, mref=None, bulge_mfrac=0.615, scatter_dex=None, zplaw=None):
        if mamp is None:
            mamp = self.MASS_AMP
        if mplaw is None:
            mplaw = self.MASS_PLAW
        if mref is None:
            mref = self.MASS_REF
        if scatter_dex is None:
            scatter_dex = self.SCATTER_DEX
        if zplaw is None:
            zplaw = self.Z_PLAW

        self._mamp = mamp   # Mass-Amplitude [grams]
        self._mplaw = mplaw   # Mass Power-law index
        self._mref = mref   # Reference Mass (argument normalization)
        self._bulge_mfrac = bulge_mfrac
        self._scatter_dex = scatter_dex
        self._zplaw = zplaw
        print(f"WARNING: Using MMBulge_Strawman(). This should only be used for testing purposes.")
        return

    def bulge_mass_frac(self, mstar):
        return self._bulge_mfrac

    def mbh_from_host(self, host, scatter):
        """Convert from stellar bulge-mass to blackhole mass.

        Units of [grams].
        """
        if scatter:
            scatter_dex = self._scatter_dex
        else:
            scatter_dex = None

        zmamp = self._mamp * (1.0 + host['redz'])**self._zplaw
        mbh = _log10_relation(host['mbulge'], zmamp, self._mplaw, scatter_dex, x0=self._mref)
        return mbh

    def mbulge_from_mbh(self, mbh, redz, scatter):
        """Convert from blackhole mass to stellar bulge-mass.

        Units of [grams].
        """
        if scatter:
            scatter_dex = self._scatter_dex
        else:
            scatter_dex = None

        zmamp = self._mamp * (1.0 + redz)**self._zplaw
        mbulge = _log10_relation_reverse(mbh, zmamp, self._mplaw, scatter_dex, x0=self._mref)
        return mbulge

    def dmbh_dmbulge(self, mbulge):
        # NOTE: scatter should never be used in the differential relation
        dmdm = self.mbh_from_mbulge(mbulge, scatter=False)
        dmdm = dmdm * self._mplaw / mbulge
        return dmdm

    def dmbulge_dmstar(self, mstar):
        # NOTE: this only works for a constant value, do *not* return `self.bulge_mass_frac()`
        return self._bulge_mfrac

    def requirements(self):
        return ['mbulge', 'redz', 'fairydust', 'Santa Claus']


class _MSigma_Relation(_MHost_Relation):

    _PROPERTIES = ['vdisp']

    # ---- Abstract Methods : must be overridden in subclasses  ----
    @abc.abstractmethod
    def dmbh_dsigma(self, sigma):
        return

    # ---- Internal Methods ----


class MSigma_Standard(_MSigma_Relation):
    """
    """

    MASS_AMP = 1.0e8 * MSOL
    MASS_PLAW = 4.24
    SIGMA_REF = 200.0 * KMPERSEC
    SCATTER_DEX = 0.0

    def __init__(self, mamp=None, mplaw=None, sigmaref=None, scatter_dex=None):
        if mamp is None:
            mamp = self.MASS_AMP
        if mplaw is None:
            mplaw = self.MASS_PLAW
        if sigmaref is None:
            sigmaref = self.SIGMA_REF
        if scatter_dex is None:
            scatter_dex = self.SCATTER_DEX

        self._mamp = mamp   # Mass-Amplitude [grams]
        self._mplaw = mplaw   # Mass Power-law index
        self._sigmaref = sigmaref   # Reference Sigma (argument normalization)
        self._scatter_dex = scatter_dex
        return

    def mbh_from_host(self, host, scatter):
        """Convert from stellar velocity dispersion to blackhole mass.

        Units of [grams].
        """
        if scatter:
            scatter_dex = self._scatter_dex
        else:
            scatter_dex = None

        mbh = _log10_relation(host['vdisp'], self._mamp, self._mplaw, scatter_dex, x0=self._sigmaref)
        return mbh

    def dmbh_dsigma(self, sigma):
        # Is this needed? I don't know
        return None


class MMBulge_MM13(MMBulge_Standard):
    """Mbh-MBulge Relation from McConnell & Ma 2013

    [MM13] Eq. 2, with values taken from Table 2 ("Dynamical masses", first row, "MPFITEXY")
    """

    MASS_AMP = MSOL * 10.0 ** 8.46    # 8.46 ± 0.08   in units of [Msol]
    MASS_REF = MSOL * 1e11            # 1e11 Msol
    MASS_PLAW = 1.05                  # 1.05 ± 0.11
    SCATTER_DEX = 0.34


class MSigma_MM13(MSigma_Standard):
    """Mbh-Sigma Relation from McConnell & Ma 2013

    [MM13] Eq. 2, with values taken from Table 2 ("M-sigma all galaxies", first row, "MPFITEXY")
    """

    MASS_AMP = MSOL * 10.0 ** 8.32    # 8.32 ± 0.05   in units of [Msol]
    SIGMA_REF = KMPERSEC * 200.0      # 200 km/s
    MASS_PLAW = 5.64                  # 5.64 ± 0.32
    SCATTER_DEX = 0.38


class MMBulge_MM13_ZEvolution(MMBulge_Redshift):
    """Mbh-MBulge Relation from McConnell & Ma 2013 for z=0 plus redshift evolution of the normalization

    [MM13] Eq. 2, with values taken from Table 2 ("Dynamical masses", first row, "MPFITEXY")
    """

    MASS_AMP = MSOL * 10.0 ** 8.46    # 8.46 ± 0.08   in units of [Msol]
    MASS_REF = MSOL * 1e11            # 1e11 Msol
    MASS_PLAW = 1.05                  # 1.05 ± 0.11
    SCATTER_DEX = 0.34
    Z_PLAW = 0.0


class MMBulge_KH13(MMBulge_Standard):
    """Mbh-MBulge Relation from Kormendy & Ho 2013.

    Values taken from [KH13] Eq.10 (pg. 61 of PDF, "571" of ARAA)
    """

    MASS_AMP = 0.49 * 1e9 * MSOL   # 0.49 + 0.06 - 0.05   in units of [Msol]
    MASS_REF = MSOL * 1e11            # 1e11 Msol
    MASS_PLAW = 1.17                  # 1.17 ± 0.08
    SCATTER_DEX = 0.28


class MSigma_KH13(MSigma_Standard):
    """Mbh-Sigma Relation from Kormendy & Ho 2013

    [KH13] Eq. 10, (pg. 65 of PDF, "575" of ARAA)
    """

    MASS_AMP = MSOL * 10.0 ** 8.46    # 8.46 ± 0.07   in units of [Msol]
    SIGMA_REF = KMPERSEC * 200.0      # 200 km/s
    MASS_PLAW = 4.26                  # 4.26 ± 0.44
    SCATTER_DEX = 0.30


def _add_scatter(vals, eps_dex):
    """Add scatter to array values.
    """
    eps_dex = None if (eps_dex is False) else eps_dex
    if (eps_dex is not None):
        shp = np.shape(vals)
        # for a scalar value of `eps_dex` draw from a zero-averaged normal distribution with that stdev
        if np.isscalar(eps_dex):
            eps_dex = np.random.normal(0.0, eps_dex, size=shp)

        if not np.all(shp == np.shape(eps_dex)):
            err = f"Shape of `eps_dex` ({np.shape(eps_dex)}) does not match input values ({shp})!"
            utils.error(err, TypeError)

        vals = vals + eps_dex

    return vals


def _log10_relation(xx, amp, plaw, eps_dex, x0=1.0):
    """

    y = amp * (xx/x0)^plaw * 10^Normal(0, e)

    Arguments
    ---------
    xx : scalar or array_like,
        Input arguments for scaling relationship.
    amp : scalar or array_like,
        Amplitude (in linear space) of scaling relationship, in desired units (e.g. grams).
    plaw : scalar or array_like,
        Power-law index of scaling relationship.
    eps_dex : `None` or `False` or scalar or array_like,
        Scatter (in dex, i.e. log10 space) for the relationship.
        If `False` or `None`, no scatter is added (the same as a value of 0.0).
    x0 : scalar or array_like,
        Units/normalization of input values.

    Returns
    -------
    yy : array_like,
        Output values with the same shape as the input `xx`.

    """
    yy = np.log10(xx/x0) * plaw

    # Add scatter to scaling relationship based on given specification in `eps_dex`
    yy = _add_scatter(yy, eps_dex)

    # Convert from dex to actual values
    yy = amp * np.power(10.0, yy)
    return yy


def _log10_relation_reverse(yy, amp, plaw, eps_dex, x0=1.0):
    """
    """
    xx = np.log10(yy/amp)

    # Add scatter to scaling relationship based on given specification in `eps_dex`
    xx = _add_scatter(xx, eps_dex)

    xx = (1.0 / plaw) * xx
    # Convert from dex to actual values
    xx = x0 * np.power(10.0, xx)
    return xx


# =================================================================================================
# ====                              Density Profiles & Relations                               ====
# =================================================================================================


class Klypin_2016:
    """Interpolate between redshifts and masses to find DM halo concentrations.

    Eq. 24 & Table 2
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
    def concentration(cls, mass, redz):
        c0 = cls._c0(redz)
        gamma = cls._gamma(redz)
        mass0 = cls._mass0(redz)
        f1 = np.power(mass/(1e12*MSOL/cosmo.h), -gamma)
        f2 = 1 + np.power(mass/mass0, 0.4)
        conc = c0 * f1 * f2
        return conc


class _Density_Profile(abc.ABC):

    @abc.abstractmethod
    def density(self, rads, *args, **kwargs):
        pass

    @classmethod
    def time_dynamical(cls, rads, *args, **kwargs):
        """Dynamical time, defined as (G M_enc / r^3) ^ -1/2 = r / v_circ
        """
        tdyn = rads / cls.velocity_circular(rads, *args, **kwargs)
        return tdyn

    @abc.abstractmethod
    def mass(cls, rads, *args, **kwargs):
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
        """Circular velocity, defined as (G M_enc / r) ^ 1/2
        """
        mass = cls.mass(rads, *args, **kwargs)
        velo = NWTG * mass / rads
        velo = velo ** 0.5
        return velo


class NFW(_Density_Profile):

    @classmethod
    def density(cls, rads, mhalo, redz):
        """NFW DM Density profile.
        [NFW-97]
        """
        rho_s, rs = cls._nfw_rho_rad(mhalo, redz)
        dens = rads / rs
        dens = dens * np.square(1 + dens)
        dens = rho_s / dens
        return dens

    @classmethod
    def mass(cls, rads, mhalo, redz):
        rads, mhalo, redz = np.broadcast_arrays(rads, mhalo, redz)
        # Get Halo concentration
        rho_s, rs = cls._nfw_rho_rad(mhalo, redz)
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

    @classmethod
    def _nfw_rho_rad(cls, mhalo, redz):
        conc = cls._concentration(mhalo, redz)
        log_c_term = np.log(1 + conc) - conc/(1+conc)

        # Critical over-density
        delta_c = (200/3) * (conc**3) / log_c_term
        # NFW density (*not* the density at the characteristic-radius)
        rho_s = cosmo.critical_density(redz).cgs.value * delta_c
        # scale-radius
        rs = mhalo / (4*np.pi*rho_s*log_c_term)
        rs = np.power(rs, 1.0/3.0)
        return rho_s, rs

    @classmethod
    def radius_scale(cls, mhalo, redz):
        rs = cls._nfw_rho_rad(mhalo, redz)[1]
        return rs

    @classmethod
    def density_critical(cls, mhalo, redz):
        rs = cls._nfw_rho_rad(mhalo, redz)[0]
        return rs


# =================================================================================================
# ====                             Stellar-Mass Halo-Mass Relation                             ====
# =================================================================================================


class _StellarMass_HaloMass(abc.ABC):

    _NUM_GRID = 200
    _MHALO_GRID_EXTR = [9, 16]

    def __init__(self):
        self._mhalo_grid = np.logspace(*self._MHALO_GRID_EXTR, self._NUM_GRID) * MSOL
        self._mstar = self.stellar_mass(self._mhalo_grid)

        xx = np.log10(self._mstar / MSOL)
        yy = np.log10(self._mhalo_grid / MSOL)
        self._mhalo_from_mstar = sp.interpolate.interp1d(xx, yy, kind='linear', bounds_error=False, fill_value=np.nan)
        return

    @abc.abstractmethod
    def stellar_mass(self, mhalo):
        pass

    def halo_mass(self, mstar):
        ynew = MSOL * 10.0 ** self._mhalo_from_mstar(np.log10(mstar/MSOL))
        return ynew


class _StellarMass_HaloMass_Redshift(_StellarMass_HaloMass):

    _REDZ_GRID_EXTR = [0.0, 9.0]
    _MSTAR_GRID_EXTR = [5.0, 12.0]

    def __init__(self, extend_nearest=True):
        self._mhalo_grid = np.logspace(*self._MHALO_GRID_EXTR, self._NUM_GRID) * MSOL
        self._redz_grid = np.linspace(*self._REDZ_GRID_EXTR, self._NUM_GRID+2)
        mhalo = self._mhalo_grid[:, np.newaxis]
        redz = self._redz_grid[np.newaxis, :]
        self._mstar = self.stellar_mass(mhalo, redz)  # should be units of [Msol]

        # ---- Construct interpolator to go from (mstar, redz) ==> (mhalo)
        # first: convert data to grid of (mstar, redz) ==> (mhalo)
        mstar = np.log10(self._mstar / MSOL)
        redz = self._redz_grid
        mhalo = np.log10(self._mhalo_grid / MSOL)
        aa = mstar.ravel()
        cc, bb = np.meshgrid(mhalo, redz, indexing='ij')
        bb, cc = [bc.ravel() for bc in [bb, cc]]

        shape = self._mstar.shape
        aextr = [aa.min(), aa.max()]
        if self._MSTAR_GRID_EXTR is not None:
            extr = self._MSTAR_GRID_EXTR
            if extr[0] < aextr[0] or extr[1] > aextr[1]:
                log.info("using wider range of stellar-mass than calculated from halo-mass grid!")
                log.debug(f"\tmstar(mhalo) = [{aextr[0]:.2e}, {aextr[1]:.2e}]")
                log.debug(f"\tmstar grid   = [{extr[0]:.2e}, {extr[1]:.2e}]")

            aextr = extr

        xx = np.linspace(*aextr, shape[0])

        self._mstar_grid = MSOL * 10.0 ** xx
        self._mhalo = MSOL * 10.0 ** cc.reshape(self._mstar.shape)
        yy = redz
        xg, yg = np.meshgrid(xx, yy, indexing='ij')
        self._aa = aa    # NOTE: these are being stored for debugging/diagnostics
        self._bb = bb    # NOTE: these are being stored for debugging/diagnostics
        self._cc = cc    # NOTE: these are being stored for debugging/diagnostics
        # grid = sp.interpolate.LinearNDInterpolator((aa, bb), cc)((xg, yg))
        # grid = sp.interpolate.NearestNDInterpolator((aa, bb), cc)((xg, yg))
        grid = sp.interpolate.griddata((aa, bb), cc, (xg, yg))
        bads = ~np.isfinite(grid)
        if np.any(bads):
            if extend_nearest:
                backup = sp.interpolate.NearestNDInterpolator((aa, bb), cc)((xg, yg))
                grid[bads] = backup[bads]
            else:
                log.warning(f"Non-finite values ({utils.frac_str(bads)}) in mhalo interpolation grid!")
                log.warning("Use `extend_nearest=True` to fill with nearest values.")

        self._grid = grid  # NOTE: these are being stored for debugging/diagnostics
        # second: construct interpolator from grid to arbitrary scatter points
        interp = sp.interpolate.RegularGridInterpolator((xx, yy), grid)
        self._mhalo_from_mstar_redz = interp
        return

    @abc.abstractmethod
    def stellar_mass(self, mhalo, redz):
        pass

    def halo_mass(self, mstar, redz, clip=False):
        if (np.ndim(mstar) not in [0, 1]) or np.any(np.shape(mstar) != np.shape(redz)):
            err = f"both `mstar` ({np.shape(mstar)}) and `redz` ({np.shape(redz)}) must be 1D and same length!"
            log.error(err)
            raise ValueError(err)

        squeeze = np.isscalar(mstar)
        mstar = np.log10(mstar/MSOL)
        if clip:
            bounds = np.array([self._mstar_grid[0], self._mstar_grid[-1]])
            bounds = np.log10(bounds / MSOL)
            idx = (mstar < bounds[0]) | (bounds[1] < mstar)
            if np.any(idx):
                log.info(f"clipping {utils.frac_str(idx)} `mstar` values outside bounds ({bounds})!")
                mstar[idx] = np.clip(mstar[idx], *bounds)

        vals = np.array([mstar, redz])
        try:
            ynew = MSOL * 10.0 ** self._mhalo_from_mstar_redz(vals.T)
        except ValueError as err:
            log.error("Interplation (mstar, redz) ==> mhalo failed!")
            log.error(err)
            for vv, nn in zip(vals, ['log10(mstar/Msol)', 'redz']):
                log.error(f"{nn} : {utils.stats(vv)}")

            raise

        if squeeze:
            ynew = ynew.squeeze()

        return ynew


class Guo_2010(_StellarMass_HaloMass):
    """
    Guo+2010 : Eq.3
    https://ui.adsabs.harvard.edu/abs/2010MNRAS.404.1111G/abstract
    """

    _NORM = 0.129
    _M0 = (10**11.4) * MSOL
    _ALPHA = 0.926
    _BETA = 0.261
    _GAMMA = 2.440

    @classmethod
    def stellar_mass(cls, mhalo):
        M0 = cls._M0
        t1 = np.power(mhalo/M0, -cls._ALPHA)
        t2 = np.power(mhalo/M0, +cls._BETA)
        mstar = mhalo * cls._NORM * np.power(t1 + t2, -cls._GAMMA)
        return mstar


class Behroozi_2013(_StellarMass_HaloMass_Redshift):
    """
    [Behroozi+2013] best fit values are at the beginning of Section 5 (pg.9), uncertainties are 1-sigma
    """

    def __init__(self, *args, **kwargs):
        self._f0 = self._f_func(0.0)
        super().__init__(*args, **kwargs)
        return

    def _nu_func(sca):
        """[Behroozi+2013] Eq. 4"""
        return np.exp(-4.0 * sca*sca)

    @classmethod
    def _param_func(cls, redz, v0, va, vz, va2=None):
        """[Behroozi+2013] Eq. 4"""
        rv = v0
        sca = cosmo.z_to_a(redz)
        rv = rv + cls._nu_func(sca) * (va * (sca - 1.0) + vz * redz)
        if va2 is not None:
            rv += va2 * (sca - 1.0)
        return rv

    @classmethod
    def _eps(cls, redz=0.0):
        e0 = -1.777   # +0.133 -0.146
        ea = -0.006   # +0.113 -0.361
        ez = 0.0      # +0.003 -0.104
        ea2 = -0.119  # +0.061 -0.012
        eps = 10.0 ** cls._param_func(redz, e0, ea, ez, va2=ea2)
        return eps

    @classmethod
    def _m1(cls, redz=0.0):
        m0 = 11.514   # +0.053 -0.009
        ma = -1.793   # +0.315 -0.330
        mz = -0.251   # +0.012 -0.125
        m1 = MSOL * 10.0 ** cls._param_func(redz, m0, ma, mz)
        return m1

    @classmethod
    def _alpha(cls, redz=0.0):
        a0 = -1.412   # +0.020 -0.105
        aa = 0.731    # +0.344 -0.296
        az = 0.0
        alpha = cls._param_func(redz, a0, aa, az)
        return alpha

    @classmethod
    def _delta(cls, redz=0.0):
        d0 = 3.508   # +0.087 -0.369
        da = 2.608   # +2.446 -1.261
        dz = -0.043  # +0.958 -0.071
        delta = cls._param_func(redz, d0, da, dz)
        return delta

    @classmethod
    def _gamma(cls, redz=0.0):
        g0 = 0.316   # +0.076 -0.012
        ga = 1.319   # +0.584 -0.505
        gz = 0.279   # +0.256 -0.081
        gamma = cls._param_func(redz, g0, ga, gz)
        return gamma

    @classmethod
    def _xsi(cls, redz=0.0):
        """[Behroozi+2013] Eq.5"""
        x0 = 0.218   # +0.011 -0.033
        xa = -0.023  # +0.052 -0.068

        xsi = x0
        if redz > 0.0:
            sca = cosmo._z_to_z(redz)
            xsi += xa * (sca - 1.0)

        return xsi

    @classmethod
    def _f_func(cls, xx, redz=0.0):
        """[Behroozi+2013] Eq.3 (lower)"""
        alpha = cls._alpha(redz)
        delta = cls._delta(redz)
        gamma = cls._gamma(redz)

        t1 = -np.log10(10**(alpha*xx) + 1)
        t2 = np.log10(1 + np.exp(xx)) ** gamma
        t3 = 1 + np.exp(10.0 ** -xx)
        ff = t1 + delta * t2 / t3
        return ff

    def stellar_mass(self, mhalo, redz):
        """[Behroozi+2013] Eq.3 (upper)"""
        eps = self._eps(redz)
        m1 = self._m1(redz)
        mstar = np.log10(eps*m1/MSOL) + self._f_func(np.log10(mhalo/m1), redz) - self._f0
        mstar = np.power(10.0, mstar) * MSOL
        return mstar
