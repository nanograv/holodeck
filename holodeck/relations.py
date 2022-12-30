"""Empirical and phenomenological scaling relationships.

This module defines numerous classes and accessor methods to implement scaling relationships between
different empirical quantities, for example BH-Galaxy relations, or Stellar-Mass vs Halo-Mass
relations.  `abc` base classes are used to implement generic functionality, and define APIs while
subclasses are left to perform specific implementations.  In general most classes implement both
the forward and reverse versions of relationships (e.g. stellar-mass to halo-mass, and also
halo-mass to stellar-mass).  Reverse relationships are often interpolated over a grid.

Most of the relationships currently implemented are among three groups (and corresponding base
classes):

* **BH-Host Relations** (`_Host_Relation`): These produce mappings between host galaxy properties
  (e.g. bulge mass) and the properties of their black holes (i.e. BH mass).

  * **Mbh-Mbulge relations** ("M-Mbulge"; `_MMBulge_Relation`): mapping from host galaxy stellar
    bulge mass to black-hole mass.
  * **Mbh-Sigma relations** ("M-Sigma"; `_MSigma_Relation`): mapping from host galaxy velocity
    dispersion (sigma) to black-hole mass.

* **Density Profiles** (`_Density_Profile`): matter density as a function of spherical radius.
* **Stellar-Mass vs. Halo-Mass Relations** (`_StellarMass_HaloMass`): mapping from halo-mass to
  stellar-mass.

To-Do
-----
* Pass concentration-relation (or other method to calculate) to NFW classes on instantiation
* For redshift-dependent extensions to relations, use multiple-inheritance instead of repeating
  attributes.

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
from typing import Type, Union

import numpy as np
from numpy.typing import ArrayLike
import scipy as sp

from holodeck import cosmo, utils, log
from holodeck.constants import MSOL, NWTG, KMPERSEC


class _Host_Relation(abc.ABC):
    """Base class for general relationships between MBHs and their host galaxies.
    """

    _PROPERTIES = []    #: list of property names to retrieve from population instances.

    def get_host_properties(self, pop, copy=True) -> dict:
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
    def mbh_from_host(self, pop, *args, **kwargs) -> np.ndarray:
        """Convert from abstract host galaxy properties to blackhole mass.

        The `pop` instance must contain the attributes required for this class's scaling relations.
        The required properties are stored in this class's `_PROPERTIES` attribute.

        Parameters
        ----------
        pop : `_Discrete_Population`,
            Population instance having the attributes required by this particular scaling relation.

        Returns
        -------
        mbh : ArrayLike
            Black hole mass.  [grams]

        """
        pass


# -----------------------------------------
# ----     M – Mbulge Relationships    ----
# -----------------------------------------


class _MMBulge_Relation(_Host_Relation):
    """Base class for implementing Mbh--Mbulge relationships, between MBH and their host galaxies.
    """

    _PROPERTIES = ['mbulge']

    def mbulge_from_mstar(self, mstar):
        """Calculate stellar bulge-mass given the total galaxy stellar mass.

        Multiplies `mstar` by the result of `self.bulge_mass_frac(mstar)`.

        Parameters
        ----------
        mstar : array_like,
            Galaxy total stellar mass.

        Returns
        -------
        array_like,
            Stellar bulge mass.

        """
        return self.bulge_mass_frac(mstar) * mstar

    # ---- Abstract Methods : must be overridden in subclasses  ----

    @abc.abstractmethod
    def dmstar_dmbh(self, mstar):
        """The partial derivative of stellar mass versus black-hole mass.

        Parameters
        ----------
        mstar : array_like,
            Host stellar-mass.

        Returns
        -------
        array_like,
            Jacobian term.

        """
        return

    @abc.abstractmethod
    def bulge_mass_frac(self, mstar):
        """Return the stellar-bulge mass fraction (M_bulge / M_star).

        Parameters
        ----------
        mstar : array_like,
            Host stellar-mass.

        Returns
        -------
        array_like,
            Bulge mass fraction.

        """
        return

    @abc.abstractmethod
    def mstar_from_mbulge(self, mbulge):
        """Convert from stellar-bulge mass to total stellar-mass.

        Parameters
        ----------
        mbulge : array_like
            Mass of the stellar bulge.  [grams]

        Returns
        -------
        mstar : array_like,
            Galaxy stellar mass.  [grams]

        """
        return

    @abc.abstractmethod
    def mbh_from_mbulge(self, *args, **kwargs) -> ArrayLike:
        """Convert from stellar-bulge mass to black-hole mass.

        Returns
        -------
        mbh : array_like,
            Mass of black hole.  [grams]

        """
        return

    def mbulge_from_mbh(self, *args, **kwargs) -> ArrayLike:
        """Convert from black-hole mass to stellar-bulge mass.

        Returns
        -------
        mbulge : array_like,
            Mass of stellar bulge.  [grams]

        """
        return


class MMBulge_Standard(_MMBulge_Relation):
    """Simple Mbh-Mbulge relation as a single power-law.

    Notes
    -----
    * Single power-law relationship between BH mass and Stellar-bulge mass.
      :math:`Mbh = M0 * (M_bulge/Mref)^plaw * 10^Normal(0, eps)`
    * Constant bulge mass-fraction relative to total stellar mass.
      :math:`M_bulge = f_bulge * M_star`

    """

    MASS_AMP = None
    MASS_AMP_LOG10 = 8.17   # log10(M/Msol)
    MASS_PLAW = 1.01
    MASS_REF = 1.0e11 * MSOL
    SCATTER_DEX = 0.3

    def __init__(self, mamp=None, mamp_log10=None, mplaw=None, mref=None, bulge_mfrac=0.615, scatter_dex=None):
        if (self.MASS_AMP_LOG10 is not None) == (self.MASS_AMP is not None):
            err = "One of `MASS_AMP_LOG10` _or_ `MASS_AMP` must be set!"
            log.exception(err)
            raise ValueError(err)

        if (mamp is None) and (mamp_log10 is None):
            mamp_log10 = self.MASS_AMP_LOG10
            mamp = self.MASS_AMP
        if mplaw is None:
            mplaw = self.MASS_PLAW
        if mref is None:
            mref = self.MASS_REF
        if scatter_dex is None:
            scatter_dex = self.SCATTER_DEX

        mamp, _ = utils._parse_val_log10_val_pars(mamp, mamp_log10, MSOL, 'mamp', only_one=True)

        self._mamp = mamp     #: Mass-Amplitude [grams]
        self._mplaw = mplaw   #: Mass Power-law index
        self._mref = mref     #: Reference Mass (argument normalization)
        self._bulge_mfrac = bulge_mfrac
        self._scatter_dex = scatter_dex
        return

    def bulge_mass_frac(self, mstar):
        return self._bulge_mfrac

    def mbh_from_host(self, pop, scatter) -> ArrayLike:
        host = self.get_host_properties(pop)
        mbulge = host['mbulge']
        return self.mbh_from_mbulge(mbulge, scatter=scatter)

    def mbh_from_mbulge(self, mbulge, scatter):
        """Convert from stellar-bulge mass to black-hole mass.

        Parameters
        ----------
        mbulge : array_like,
            Stellar bulge-mass of host galaxy.  [grams]
        scatter : bool,
            Whether or not to include scatter in scaling relationship.
            Uses `self._scatter_dex` attribute.

        Returns
        -------
        mbh : array_like,
            Mass of black hole.  [grams]

        """
        scatter_dex = self._scatter_dex if scatter else None
        mbh = _log10_relation(mbulge, self._mamp, self._mplaw, scatter_dex, x0=self._mref)
        return mbh

    def mbulge_from_mbh(self, mbh, scatter):
        """Convert from black-hole mass to stellar-bulge mass.

        Parameters
        ----------
        mbh : array_like,
            Mass of black hole.  [grams]
        scatter : bool,
            Whether or not to include scatter in scaling relationship.
            Uses `self._scatter_dex` attribute.

        Returns
        -------
        mbulge : array_like,
            Mass of stellar bulge.  [grams]
        """
        scatter_dex = self._scatter_dex if scatter else None
        mbulge = _log10_relation_reverse(mbh, self._mamp, self._mplaw, scatter_dex, x0=self._mref)
        return mbulge

    def mstar_from_mbulge(self, mbulge):
        """Convert from stellar bulge-mass to black-hole mass.

        Parameters
        ----------
        mbulge : array_like,
            Stellar bulge-mass of host galaxy.  [grams]
        scatter : bool,
            Whether or not to include scatter in scaling relationship.
            Uses `self._scatter_dex` attribute.

        Returns
        -------
        mbh : array_like,
            Mass of black hole.  [grams]

        """
        return mbulge / self._bulge_mfrac

    def mbh_from_mstar(self, mstar, scatter):
        """Convert from total stellar mass to black-hole mass.

        Parameters
        ----------
        mstar : array_like,
            Total stellar mass of host galaxy.  [grams]
        scatter : bool,
            Whether or not to include scatter in scaling relationship.
            Uses `self._scatter_dex` attribute.

        Returns
        -------
        mbh : array_like,
            Mass of black hole.  [grams]

        """
        mbulge = self.mbulge_from_mstar(mstar)
        return self.mbh_from_mbulge(mbulge, scatter)

    def mstar_from_mbh(self, mbh, scatter):
        """Convert from black-hole mass to total stellar mass.

        Parameters
        ----------
        mbh : array_like,
            Mass of black hole.  [grams]
        scatter : bool,
            Whether or not to include scatter in scaling relationship.
            Uses `self._scatter_dex` attribute.

        Returns
        -------
        array_like,
            Total stellar mass of host galaxy.  [grams]

        """
        mbulge = self.mbulge_from_mbh(mbh, scatter)
        return self.mstar_from_mbulge(mbulge)

    def dmstar_dmbh(self, mstar):
        """Calculate the partial derivative of stellar mass versus BH mass :math:`d M_star / d M_bh`.

        .. math::
            d M_star / d M_bh  =  [d M_star / d M_bulge] * [d M_bulge / d M_bh] \\
                               =  [1/f_bulge] * [M_bulge / (plaw * M_bh)]

        Parameters
        ----------
        mstar : array_like,
            Total stellar mass of galaxy.  [grams]

        Returns
        -------
        deriv : array_like,
            Jacobian term.

        """
        plaw = self._mplaw
        fbulge = self._bulge_mfrac
        mbulge = mstar * fbulge
        mbh = self.mbh_from_mbulge(mbulge, scatter=False)
        deriv = mstar / (plaw * mbh)
        return deriv


class MMBulge_KH2013(MMBulge_Standard):
    """Mbh-MBulge Relation, single power-law, from Kormendy & Ho 2013.

    Values taken from [KH2013]_ Eq.10.

    """
    MASS_AMP = 0.49 * 1e9 * MSOL      # 0.49 + 0.06 - 0.05   in units of [Msol]
    MASS_AMP_LOG10 = None
    MASS_REF = MSOL * 1e11            # 1e11 Msol
    MASS_PLAW = 1.17                  # 1.17 ± 0.08
    SCATTER_DEX = 0.28                # scatter stdev in dex


class MMBulge_MM2013(MMBulge_Standard):
    """Mbh-MBulge Relation from McConnell & Ma 2013

    [MM2013]_ Eq. 2, with values taken from Table 2 ("Dynamical masses", first row, "MPFITEXY")

    """
    MASS_AMP = None
    MASS_AMP_LOG10 = 8.46    # 8.46 ± 0.08   in units of [Msol]
    MASS_REF = MSOL * 1e11            # 1e11 Msol
    MASS_PLAW = 1.05                  # 1.05 ± 0.11
    SCATTER_DEX = 0.34


# ----     M – Mbulge & Redshift Relationships    ----


class MMBulge_Redshift(MMBulge_Standard):
    """Mbh-Mbulge relation with an additional redshift power-law dependence.

    Provides black hole mass as a function of galaxy bulge mass and redshift with a normalization
    that depends on redshift. zplaw=0 (default) is identical to MMBulge_Standard.
    mamp = mamp0 * (1 + z)**zplaw

    TODO: make sure all of the inherited methods from `MMBulge_Standard` are appropriate for
          redshift dependencies!!  In particular, check `dmstar_dmbh`

    """

    MASS_AMP = 3.0e8 * MSOL
    MASS_AMP_LOG10 = None
    MASS_PLAW = 1.0
    MASS_REF = 1.0e11 * MSOL
    SCATTER_DEX = 0.0
    Z_PLAW = 0.0

    _PROPERTIES = ['mbulge', 'redz']

    def __init__(self, *args, zplaw=None, **kwargs):
        super().__init__(*args, **kwargs)

        if zplaw is None:
            zplaw = self.Z_PLAW

        self._zplaw = zplaw
        return

    def mbh_from_host(self, pop, scatter):
        host = self.get_host_properties(pop, copy=False)
        mbulge = host['mbulge']    # shape (N, 2)
        redz = host['redz']        # shape (N,)
        return self.mbh_from_mbulge(mbulge, redz, scatter=scatter)

    def mbh_from_mbulge(self, mbulge, redz, scatter):
        scatter_dex = self._scatter_dex if scatter else None
        # Broadcast `redz` to match shape of `mbulge`, if needed
        # NOTE: this will work for (N,) ==> (N,)    or   (N,) ==> (N,X)
        redz = np.broadcast_to(redz, mbulge.T.shape).T
        zmamp = self._mamp * (1.0 + redz)**self._zplaw
        mbh = _log10_relation(mbulge, zmamp, self._mplaw, scatter_dex, x0=self._mref)
        return mbh

    def mbulge_from_mbh(self, mbh, redz, scatter):
        scatter_dex = self._scatter_dex if scatter else None
        zmamp = self._mamp * (1.0 + redz)**self._zplaw
        mbulge = _log10_relation_reverse(mbh, zmamp, self._mplaw, scatter_dex, x0=self._mref)
        return mbulge

    def mbh_from_mstar(self, mstar, redz, scatter):
        mbulge = self.mbulge_from_mstar(mstar)
        return self.mbh_from_mbulge(mbulge, redz, scatter)

    def mstar_from_mbh(self, mbh, redz, scatter):
        mbulge = self.mbulge_from_mbh(mbh, redz, scatter)
        return self.mstar_from_mbulge(mbulge)

    def dmstar_dmbh(self, mstar, redz):
        plaw = self._mplaw
        fbulge = self._bulge_mfrac
        mbulge = mstar * fbulge
        mbh = self.mbh_from_mbulge(mbulge, redz, scatter=False)
        deriv = mbulge / (fbulge * plaw * mbh)
        return deriv


class MMBulge_Redshift_MM2013(MMBulge_Redshift):
    """Mbh-MBulge Relation from McConnell & Ma 2013 for z=0 plus redshift evolution of the normalization

    BUG/FIX: use multiple-inheritance for this

    [MM2013]_ Eq. 2, with values taken from Table 2 ("Dynamical masses", first row, "MPFITEXY")

    """
    MASS_AMP_LOG10 = 8.46    # 8.46 ± 0.08   in units of [Msol]
    MASS_AMP = None
    MASS_REF = MSOL * 1e11            # 1e11 Msol
    MASS_PLAW = 1.05                  # 1.05 ± 0.11
    SCATTER_DEX = 0.34
    Z_PLAW = 0.0


class MMBulge_Redshift_KH2013(MMBulge_Redshift):
    """Mbh-MBulge Relation from Kormendy & Ho 2013, w/ optional redshift evolution of normalization.

    BUG/FIX: use multiple-inheritance for this

    Values taken from [KH2013] Eq.10 (pg. 61 of PDF, "571" of ARAA)
    """
    MASS_AMP = 0.49 * 1e9 * MSOL   # 0.49 + 0.06 - 0.05   in units of [Msol]
    MASS_AMP_LOG10 = None
    MASS_REF = MSOL * 1e11            # 1e11 Msol
    MASS_PLAW = 1.17                  # 1.17 ± 0.08
    SCATTER_DEX = 0.28
    Z_PLAW = 0.0


def get_mmbulge_relation(mmbulge: Union[_MMBulge_Relation, Type[_MMBulge_Relation]] = None) -> _MMBulge_Relation:
    """Return a valid Mbh-Mbulge instance.

    Parameters
    ----------
    mmbulge : None or (type or instance of `_MMBulge_Relation`),
        If `None`, then a default M-Mbulge relation is returned.  Otherwise, the type is checked
        to make sure it is a valid instance of an `_MMBulge_Relation`.

    Returns
    -------
    _MMBulge_Relation
        Instance of an Mbh-Mbulge relationship.

    """
    return utils._get_subclass_instance(mmbulge, MMBulge_KH2013, _MMBulge_Relation)


# ----------------------------------------
# ----     M – Sigma Relationships    ----
# ----------------------------------------


class _MSigma_Relation(_Host_Relation):
    """Base class for 'M-Sigma relations' between BH mass and host velocity dispersion.
    """

    _PROPERTIES = ['vdisp']

    # @abc.abstractmethod
    # def dmbh_dsigma(self, sigma):
    #     pass

    @abc.abstractmethod
    def mbh_from_vdisp(self, vdisp, scatter):
        pass

    @abc.abstractmethod
    def vdisp_from_mbh(self, mbh, scatter):
        pass


class MSigma_Standard(_MSigma_Relation):
    """Simple M-sigma relation (BH mass vs. host velocity dispersion) as a single power-law.

    Notes
    -----
    * Single power-law relationship between BH mass and Stellar-bulge mass.
      :math:`Mbh = M0 * (sigma/sigma_ref)^plaw * 10^Normal(0, eps)`

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

    def mbh_from_host(self, pop, scatter):
        host = self.get_host_properties(pop, copy=False)
        vdisp = host['vdisp']    # shape (N, 2)
        return self.mbh_from_vdisp(vdisp, scatter=scatter)

    def mbh_from_vdisp(self, vdisp, scatter):
        """Convert from host galaxy stellar velocity dispersion to black-hole mass.

        Parameters
        ----------
        vdisp : array_like,
            Host-galaxy velocity dispersion.  [cm/s].
        scatter : bool,
            Whether or not to include scatter in scaling relationship.
            Uses `self._scatter_dex` attribute.

        Returns
        -------
        mbh : array_like,
            Mass of black hole.  [grams]

        """
        scatter_dex = self._scatter_dex if scatter else None
        mbh = _log10_relation(vdisp, self._mamp, self._mplaw, scatter_dex, x0=self._sigmaref)
        return mbh

    def vdisp_from_mbh(self, mbh, scatter):
        """Convert from black-hole mass to host galaxy stellar velocity dispersion.

        Parameters
        ----------
        mbh : array_like,
            Mass of black hole.  [grams]
        scatter : bool,
            Whether or not to include scatter in scaling relationship.
            Uses `self._scatter_dex` attribute.

        Returns
        -------
        vdisp : array_like,
            Host-galaxy velocity dispersion.  [cm/s].

        """
        scatter_dex = self._scatter_dex if scatter else None
        vdisp = _log10_relation_reverse(mbh, self._mamp, self._mplaw, scatter_dex, x0=self._sigmaref)
        return vdisp

    # def dmbh_dsigma(self, sigma):
    #     # Is this needed? I don't know
    #     return None


class MSigma_MM2013(MSigma_Standard):
    """Mbh-Sigma Relation from McConnell & Ma 2013.

    [MM2013]_ Eq. 2, with values taken from Table 2 ("M-sigma all galaxies", first row, "MPFITEXY")
    """

    MASS_AMP = MSOL * 10.0 ** 8.32    # 8.32 ± 0.05   in units of [Msol]
    SIGMA_REF = KMPERSEC * 200.0      # 200 km/s
    MASS_PLAW = 5.64                  # 5.64 ± 0.32
    SCATTER_DEX = 0.38


class MSigma_KH2013(MSigma_Standard):
    """Mbh-Sigma Relation from Kormendy & Ho 2013.

    [KH2013]_ Eq. 10, (pg. 65 of PDF, "575" of ARAA)
    """

    MASS_AMP = MSOL * 10.0 ** 8.46    # 8.46 ± 0.07   in units of [Msol]
    SIGMA_REF = KMPERSEC * 200.0      # 200 km/s
    MASS_PLAW = 4.26                  # 4.26 ± 0.44
    SCATTER_DEX = 0.30


def get_msigma_relation(msigma: Union[_MSigma_Relation, Type[_MSigma_Relation]] = None) -> _MSigma_Relation:
    """Return a valid M-sigma (BH Mass vs. host galaxy velocity dispersion) instance.

    Parameters
    ----------
    msigma : None or (class or instance of `_MSigma_Relation`),
        If `None`, then a default M-sigma relation is returned.  Otherwise, the type is checked
        to make sure it is a valid instance of an `_MSigma_Relation`.

    Returns
    -------
    _MSigma_Relation
        Instance of an Mbh-sigma relationship.

    """
    return utils._get_subclass_instance(msigma, MSigma_KH2013, _MSigma_Relation)


def _add_scatter(vals: ArrayLike, eps: ArrayLike) -> ArrayLike:
    """Add scatter to the input values with a given standard deviation.

    Parameters
    ----------
    vals : ArrayLike
        Values that scatter should be added to.
    eps : ArrayLike
        Standard deviation of the scatter that should be added.
        This must either be a single `float` value, or an ArrayLike broadcastable against `vals`.

    Returns
    -------
    vals : ArrayLike
        Values with added scatter.

    """
    eps = None if (eps is False) else eps
    if (eps is not None):
        shp = np.shape(vals)
        # for a scalar value of `eps` draw from a zero-averaged normal distribution with that stdev
        if np.isscalar(eps):
            eps = np.random.normal(0.0, eps, size=shp)
        else:
            eps = np.random.normal(0.0, eps)

        # else:
        #     err = f"Shape of `eps` ({np.shape(eps)}) does not match input values ({shp})!"
        #     log.exception(err)
        #     raise TypeError(err)

        vals = vals + eps

    return vals


def _log10_relation(xx, amp, plaw, eps_dex, x0=1.0):
    """Calculate the forward output of a standard base-10 power-law scaling relationship.

    :math:`y = amp * (xx/x0)^plaw * 10^Normal(0, e)`

    Parameters
    ----------
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
    """Calculate the reverse of a standard base-10 power-law scaling relationship.

    From the standard expression, take as input `y` and return `x`:
    :math:`y = amp * (xx/x0)^plaw * 10^Normal(0, e)`

    NOTE: the scatter (`eps_dex`) adds *additional* variance, instead of removing it.

    Parameters
    ----------
    yy : scalar or array_like,
        Arguments to be reversed.  This would be the 'output' of the standard forward relation.
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
    xx : array_like,
        Values that would be the 'inputs' for the standard forward relationship.

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


# =================================================================================================
# ====                             Stellar-Mass Halo-Mass Relation                             ====
# =================================================================================================


class _StellarMass_HaloMass(abc.ABC):
    """Base class for a general one-parameter Stellar-Mass vs Halo-Mass relation.

    Uses log-linear interpolation of a pre-calculated grid to calculate halo mass from stellar mass.

    """

    _NUM_GRID = 200              #: grid size
    _MHALO_GRID_EXTR = [7, 20]   #: extrema for the halo mass grid [log10(M/Msol)]

    def __init__(self):
        self._mhalo_grid = np.logspace(*self._MHALO_GRID_EXTR, self._NUM_GRID) * MSOL
        self._mstar = self.stellar_mass(self._mhalo_grid)

        xx = np.log10(self._mstar / MSOL)
        yy = np.log10(self._mhalo_grid / MSOL)
        self._mhalo_from_mstar = sp.interpolate.interp1d(xx, yy, kind='linear', bounds_error=False, fill_value=np.nan)
        return

    @abc.abstractmethod
    def stellar_mass(self, *args, **kwargs) -> np.ndarray:
        """Calculate the stellar-mass for the given halo mass.

        Parameters
        ----------
        mhalo : ArrayLike
            Halo mass.  [gram]

        Returns
        -------
        mstar : ArrayLike
            Stellar mass.  [gram]

        """
        pass

    def halo_mass(self, mstar: ArrayLike) -> np.ndarray:
        """Calculate the stellar-mass for the given halo mass.

        Parameters
        ----------
        mstar : ArrayLike
            Stellar mass.  [gram]

        Returns
        -------
        mhalo : ArrayLike
            Halo mass.  [gram]

        """
        ynew = MSOL * 10.0 ** self._mhalo_from_mstar(np.log10(mstar/MSOL))
        return ynew


class _StellarMass_HaloMass_Redshift(_StellarMass_HaloMass):
    """Base class for a Stellar-Mass vs Halo-Mass relation including redshift dependence.

    Uses interpolation of a pre-calculated grid to calculate halo mass from stellar mass and redz.

    """

    _REDZ_GRID_EXTR = [0.0, 10.0]     #: edges of the parameter space in redshift
    _MSTAR_GRID_EXTR = [5.0, 14.0]    #: edges of the parameter space in stellar-mass [log10(M/Msol)]

    def __init__(self, extend_nearest=True):
        self._mhalo_grid = np.logspace(*self._MHALO_GRID_EXTR, self._NUM_GRID) * MSOL   # shape (H,)
        self._redz_grid = np.linspace(*self._REDZ_GRID_EXTR, self._NUM_GRID+2)          # shape (Z,)
        mhalo = self._mhalo_grid[:, np.newaxis]
        redz = self._redz_grid[np.newaxis, :]
        # Calculate stellar-mass given the grid of halo-mass and redshift
        # Shape (H, Z)
        self._mstar = self.stellar_mass(mhalo, redz)  # units of [gram]

        # ---- Construct interpolator to go from (mstar, redz) ==> (mhalo)

        # first: convert data to grid of (mstar, redz) ==> (mhalo)
        mstar = np.log10(self._mstar / MSOL)
        # store the normal shape (H, Z)
        shape = mstar.shape
        # convert from (H, Z) ==> (H*Z,)
        mstar_rav = mstar.ravel()
        # Get grids for halo-mass and redshift, (H, Z) ==> (H*Z,)
        redz = self._redz_grid
        mhalo = np.log10(self._mhalo_grid / MSOL)
        mhalo_ravel, redz_ravel = np.meshgrid(mhalo, redz, indexing='ij')
        redz_ravel, mhalo_ravel = [bc.ravel() for bc in [redz_ravel, mhalo_ravel]]

        mstar_out_extr = [mstar_rav.min(), mstar_rav.max()]
        if self._MSTAR_GRID_EXTR is not None:
            extr = self._MSTAR_GRID_EXTR
            if extr[0] < mstar_out_extr[0] or extr[1] > mstar_out_extr[1]:
                log.debug("using wider range of stellar-mass than calculated from halo-mass grid!")
                log.debug(f"\tmstar(mhalo) = [{mstar_out_extr[0]:.2e}, {mstar_out_extr[1]:.2e}]")
                log.debug(f"\tmstar grid   = [{extr[0]:.2e}, {extr[1]:.2e}]")

            mstar_out_extr = extr

        xx = np.linspace(*mstar_out_extr, shape[0])

        self._mstar_grid = MSOL * (10.0 ** xx)
        self._mhalo = MSOL * (10.0 ** mhalo_ravel.reshape(shape))
        yy = redz
        xg, yg = np.meshgrid(xx, yy, indexing='ij')
        self._xx = xx    # NOTE: these are being stored for debugging/diagnostics
        self._yy = yy    # NOTE: these are being stored for debugging/diagnostics
        self._aa = mstar_rav    # NOTE: these are being stored for debugging/diagnostics
        self._bb = redz_ravel    # NOTE: these are being stored for debugging/diagnostics
        self._cc = mhalo_ravel    # NOTE: these are being stored for debugging/diagnostics

        grid = sp.interpolate.griddata((mstar_rav, redz_ravel), mhalo_ravel, (xg, yg))
        bads = ~np.isfinite(grid)
        if np.any(bads):
            if extend_nearest:
                backup = sp.interpolate.NearestNDInterpolator((mstar_rav, redz_ravel), mhalo_ravel)((xg, yg))
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
    def stellar_mass(self, mhalo: ArrayLike, redz: ArrayLike) -> np.ndarray:
        """Calculate the stellar-mass for the given halo mass and redshift.

        Parameters
        ----------
        mhalo : ArrayLike
            Halo mass.  [gram]
        redz : ArrayLike
            Redshift.

        Returns
        -------
        mstar : ArrayLike
            Stellar mass.  [gram]

        """
        pass

    def halo_mass(self, mstar: ArrayLike, redz: ArrayLike, clip: bool = False) -> np.ndarray:
        """Calculate the halo-mass for the given stellar mass and redshift.

        Parameters
        ----------
        mstar : ArrayLike
            Stellar mass.  [gram]
        redz : ArrayLike
            Redshift.
        clip : bool
            Whether or not to clip the input `mstar` values to the extrema of the predefined grid
            of stellar-masses (`_mstar_grid`).

        Returns
        -------
        mhalo : ArrayLike
            Halo mass.  [gram]

        """
        if (np.ndim(mstar) not in [0, 1]) or np.any(np.shape(mstar) != np.shape(redz)):
            err = f"both `mstar` ({np.shape(mstar)}) and `redz` ({np.shape(redz)}) must be 1D and same length!"
            log.error(err)
            raise ValueError(err)

        squeeze = np.isscalar(mstar)
        mstar_log10 = np.log10(mstar/MSOL)
        if clip:
            bounds = np.array([self._mstar_grid[0], self._mstar_grid[-1]])
            bounds = np.log10(bounds / MSOL)
            idx = (mstar_log10 < bounds[0]) | (bounds[1] < mstar_log10)
            if np.any(idx):
                log.debug(f"clipping {utils.frac_str(idx)} `mstar` values outside bounds ({bounds})!")
                mstar_log10[idx] = np.clip(mstar_log10[idx], *bounds)

        vals = np.array([mstar_log10, redz])
        try:
            ynew = MSOL * 10.0 ** self._mhalo_from_mstar_redz(vals.T)
        except ValueError as err:
            log.exception("Interplation (mstar, redz) ==> mhalo failed!")
            log.error(err)
            extr = [utils.minmax(xx) for xx in [self._xx, self._yy]]
            for vv, nn, ee in zip(vals, ['log10(mstar/Msol)', 'redz'], extr):
                log.error(f"{nn} (extrema = {ee}): {utils.stats(vv)}")

            raise

        if squeeze:
            ynew = ynew.squeeze()

        return ynew


class Guo_2010(_StellarMass_HaloMass):
    """Stellar-Mass - Halo-Mass relation from Guo et al. 2010.

    [Guo2010]_ Eq.3

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
    """Redshift-dependent Stellar-Mass - Halo-Mass relation based on Behroozi et al. 2013.

    [Behroozi2013]_ best fit values are at the beginning of Section 5 (pg.9), uncertainties are 1-sigma.

    """

    def __init__(self, *args, **kwargs):
        self._f0 = self._f_func(0.0)
        super().__init__(*args, **kwargs)
        return

    def stellar_mass(self, mhalo, redz):
        """This is [Behroozi2013]_ Eq.3 (upper)"""
        eps = self._eps(redz)
        m1 = self._m1(redz)
        mstar = np.log10(eps*m1/MSOL) + self._f_func(np.log10(mhalo/m1), redz) - self._f0
        mstar = np.power(10.0, mstar) * MSOL
        return mstar

    def _nu_func(sca):
        """[Behroozi2013]_ Eq. 4"""
        return np.exp(-4.0 * sca*sca)

    @classmethod
    def _param_func(cls, redz, v0, va, vz, va2=None):
        """[Behroozi2013]_ Eq. 4"""
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


def get_stellar_mass_halo_mass_relation(
    smhm: Union[_StellarMass_HaloMass, Type[_StellarMass_HaloMass]] = None
) -> _StellarMass_HaloMass:
    """Return a valid Stellar-Mass vs. Halo-Mass relation instance.

    Parameters
    ----------
    smhm : None or (type or instance of `_StellarMass_HaloMass`),
        If `None`, then a default relation is returned.  Otherwise, the type is checked
        to make sure it is a valid instance of an `_StellarMass_HaloMass`.

    Returns
    -------
    _StellarMass_HaloMass
        Instance of an Mbh-Mbulge relationship.

    """
    return utils._get_subclass_instance(smhm, Behroozi_2013, _StellarMass_HaloMass)
