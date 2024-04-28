"""Binary evolution hardening submodules.

To-Do (Hardening)
-----------------
* Dynamical_Friction_NFW
    * Allow stellar-density profiles to also be specified (instead of using a hard-coded
      Dehnen profile)
    * Generalize calculation of stellar characteristic radius.  Make self-consistent with
      stellar-profile, and user-specifiable.
* Evolution
    * `_sample_universe()` : sample in comoving-volume instead of redshift
* Sesana_Scattering
    * Allow stellar-density profile (or otherwise the binding-radius) to be user-specified
      and flexible.  Currently hard-coded to Dehnen profile estimate.
* _SHM06
    * Interpolants of hardening parameters return 1D arrays.
* Fixed_Time_2PL
    * Handle `rchar` better with respect to interpolation.  Currently not an interpolation
      variable, which restricts it's usage.
    * This class should be separated into a generic `_Fixed_Time` class that can use any
      functional form, and then a 2-power-law functional form that requires a specified
      normalization.  When they're combined, it will produce the same effect.  Another good
      functional form to implement would be GW + log-uniform hardening time, the same as the
      current phenomenological model but with both power-laws set to 0.

References
----------
* [BBR1980]_ Begelman, Blandford & Rees 1980.
* [Chen2017]_ Chen, Sesana, & Del Pozzo 2017.
* [Kelley2017a]_ Kelley, Blecha & Hernquist 2017.
* [Quinlan1996]_ Quinlan 1996.
* [Sesana2006]_ Sesana, Haardt & Madau et al. 2006.
* [Sesana2010]_ Sesana 2010.
* [Siwek2023]_ Siwek+2023

"""

from __future__ import annotations

import abc
import json
import os
import warnings

import numpy as np
import scipy as sp
import scipy.interpolate   # noqa
import pickle as pkl
from scipy.interpolate import RectBivariateSpline

import holodeck as holo
from holodeck import utils, cosmo, log, _PATH_DATA, galaxy_profiles
from holodeck.host_relations import (
    get_stellar_mass_halo_mass_relation, get_mmbulge_relation, get_msigma_relation
)
from holodeck.constants import GYR, NWTG, PC, MSOL

#: number of influence radii to set minimum radius for dens calculation
_MIN_DENS_RAD__INFL_RAD_MULT = 10.0
_SCATTERING_DATA_FILENAME = "SHM06_scattering_experiments.json"


# =================================================================================================
# ====    Hardening Classes    ====
# =================================================================================================


class _Hardening(abc.ABC):
    """Base class for binary-hardening models, providing the `dadt_dedt(evo, step, ...)` method.
    """

    CONSISTENT = None

    @abc.abstractmethod
    def dadt_dedt(self, evo, step, *args, **kwargs):
        pass

    def dadt(self, *args, **kwargs):
        rv_dadt, _dedt = self.dadt_dedt(*args, **kwargs)
        return rv_dadt

    def dedt(self, *args, **kwargs):
        _dadt, rv_dedt = self.dadt_dedt(*args, **kwargs)
        return rv_dedt


class Hard_GW(_Hardening):
    """Gravitational-wave driven binary hardening.
    """

    CONSISTENT = False

    @staticmethod
    def dadt_dedt(evo, step):
        """Calculate GW binary evolution (hardening rate) in semi-major-axis and eccentricity.

        Parameters
        ----------
        evo : `Evolution`
            Evolution instance providing the binary parameters for calculating hardening rates.
        step : int
            Evolution integration step index from which to load binary parameters.
            e.g. separations are loaded as ``evo.sepa[:, step]``.

        Returns
        -------
        dadt : np.ndarray
            Hardening rate in semi-major-axis, returns negative value, units [cm/s].
        dedt : np.ndarray
            Hardening rate in eccentricity, returns negative value, units [1/s].

        """
        m1, m2 = evo.mass[:, step, :].T    # (Binaries, Steps, 2) ==> (2, Binaries)
        sepa = evo.sepa[:, step]
        eccen = evo.eccen[:, step] if (evo.eccen is not None) else None
        dadt = utils.gw_hardening_rate_dadt(m1, m2, sepa, eccen=eccen)

        if eccen is None:
            dedt = None
        else:
            dedt = utils.gw_dedt(m1, m2, sepa, eccen)

        return dadt, dedt

    @staticmethod
    def dadt(mtot, mrat, sepa, eccen=None):
        """Calculate GW Hardening rate of semi-major-axis vs. time.

        See [Peters1964]_, Eq. 5.6

        Parameters
        ----------
        mtot : array_like
            Total mass of each binary system.  Units of [gram].
        mrat : array_like
            Mass ratio of each binary, defined as $q \\equiv m_1/m_2 \\leq 1.0$.
        sepa : array_like
            Binary semi-major axis (separation), in units of [cm].
        eccen : array_like or None
            Binary eccentricity, `None` is the same as zero eccentricity (circular orbit).

        Returns
        -------
        dadt : np.ndarray
            Hardening rate in semi-major-axis, result is negative, units [cm/s].

        """
        m1, m2 = utils.m1m2_from_mtmr(mtot, mrat)
        dadt = utils.gw_hardening_rate_dadt(m1, m2, sepa, eccen=eccen)
        return dadt

    @staticmethod
    def dedt(mtot, mrat, sepa, eccen=None):
        """Calculate GW Hardening rate of eccentricity vs. time.

        See [Peters1964]_, Eq. 5.7

        If `eccen` is `None`, zeros are returned.

        Parameters
        ----------
        mtot : array_like
            Total mass of each binary system.  Units of [gram].
        mrat : array_like
            Mass ratio of each binary, defined as $q \\equiv m_1/m_2 \\leq 1.0$.
        sepa : array_like
            Binary semi-major axis (separation), in units of [cm].
        eccen : array_like or None
            Binary eccentricity, `None` is the same as zero eccentricity (circular orbit).

        Returns
        -------
        dedt : np.ndarray
            Hardening rate in eccentricity, result is <= 0.0, units [1/s].
            Zero values if `eccen` is `None`.

        """
        if eccen is None:
            return np.zeros_like(mtot)
        m1, m2 = utils.m1m2_from_mtmr(mtot, mrat)
        dedt = utils.gw_dedt(m1, m2, sepa, eccen=eccen)
        return dedt

    @staticmethod
    def deda(sepa, eccen):
        """Rate of eccentricity change versus separation change.

        See [Peters1964]_, Eq. 5.8

        Parameters
        ----------
        sepa : array_like,
            Binary semi-major axis (i.e. separation) [cm].
        eccen : array_like,
            Binary orbital eccentricity.

        Returns
        -------
        rv : array_like,
            Binary deda rate [1/cm] due to GW emission.
            Values are always positive.

        """
        # fe = utils._gw_ecc_func(eccen)
        # rv = 19 * eccen * (1.0 + (121/304)*eccen*eccen)   # numerator
        # rv = rv / (12 * sepa * fe)
        rv = 1.0 / utils.gw_dade(sepa, eccen)
        return rv

    @property
    def consistent(self):
        return False


class CBD_Torques(_Hardening):
    """Binary Orbital Evolution based on Hydrodynamic Simulations by Siwek+23.

    This module uses data from Siwek+23, which supplies rates of change of
    binary semi-major axis a_b
    and
    binary eccentricity e_b.
    The calculation of a_b and e_b versus time requires accretion rates (for scale).
    """

    CONSISTENT = None

    def __init__(self, f_edd = 0.10, subpc = True):
        """Construct a CBD-Torque instance.

        Parameters
        ----------

        """

        self.f_edd = f_edd
        self.subpc = subpc

        return

    def dadt_dedt(self, evo, step):
        """Circumbinary Disk Torque hardening rate.

        Parameters
        ----------
        evo : `Evolution`
            Evolution instance providing binary parameters at the given intergration step.
        step : int
            Integration step at which to calculate hardening rates.

        Returns
        -------
        dadt : array_like
            Binary hardening rates in units of [cm/s], defined to be negative.
        dedt : array_like
            Binary rate-of-change of eccentricity in units of [1/sec].

        """
        mass = evo.mass[:, step, :]
        sepa = evo.sepa[:, step]
        eccen = evo.eccen[:, step] if evo.eccen is not None else None

        if evo._acc is None:
            """ If no accretion modules is supplied, use an Eddington fraction for now """
            # total_mass = mass[:,0] + mass[:,1]
            accretion_instance = holo.accretion.Accretion(f_edd = self.f_edd, subpc=self.subpc)
            mdot = accretion_instance.mdot_total(evo, step)
        if evo._acc is not None:
            """ An instance of the accretion class has been supplied,
                and binary masses are evolved through accretion
                Get total accretion rates """
            mdot = evo._acc.mdot_total(evo, step)

        dadt, dedt = self._dadt_dedt(mass, sepa, eccen, mdot)

        """ CURRENTLY WE CANNOT USE +ve dadt VALUES, SO WE SET THEM TO 0 """
        inds_dadt_pos = dadt > 0
        dadt[inds_dadt_pos] = 0.0
        inds_dadt_nan = np.isnan(dadt)
        dadt[inds_dadt_nan] = 0.0

        return dadt, dedt

    def _dadt_dedt(self, mass, sepa, eccen, mdot):
        """Circumbinary Disk Torque hardening rate from Siwek+23.

        Parameters
        ----------
        mass : (N,2) array_like
            Masses of each MBH component (0-primary, 1-secondary) in units of [gram].
        sepa : (N,) array_like
            Binary separation in units of [cm].
        eccen : (N,) array_like or `None`
            Binary eccentricity.  `None` if eccentricity is not being evolved.

        Returns
        -------
        dadt : (N,) array-like of scalar
            Binary hardening rates in units of [cm/s], defined to be negative.
        dedt : (N,) array-like of scalar  or  `None`
            Binary rate-of-change of eccentricity in units of [1/sec].
            If eccentricity is not being evolved (i.e. `eccen==None`) then `None` is returned.

        """
        mass = np.atleast_2d(mass)
        mtot = mass[:,0] + mass[:,1]
        """ MASS RATIO """
        m1 = mass[:, 0]
        m2 = mass[:, 1]
        mrat = m2/m1
        """ secondary and primary can swap indices. need to account for that and reverse the mass ratio """
        inds_rev = mrat > 1
        mrat[inds_rev] = 1./mrat[inds_rev]
        """ SEPARATION """
        sepa = np.atleast_1d(sepa)
        """ ECCENTRICITY """
        eccen = np.atleast_1d(eccen) if eccen is not None else None

        semimajor_axis = sepa #for now? we don't resolve the orbit in time (ever?) so this approximation should do?

        """ dadt and dedt from Siwek+23 are parameterized
            by the semimajor axis, mass and accretion rate
            of the accreting binary systems. Below dadt and dedt
            are converted into physical quantities:
            [dadt] = cm/s
            [dedt] = 1/s
            which depend on the physical scale
            and accretion rate of the system """
        dadt = _Siwek2023.dadt(mrat, eccen) * semimajor_axis * (mdot/mtot)
        if eccen is not None:
            dedt =  _Siwek2023.dedt(mrat, eccen) * (mdot/mtot)

        else:
            dedt = np.zeros_like(sepa)

        return dadt, dedt


class Sesana_Scattering(_Hardening):
    """Binary-Hardening Rates calculated based on the Sesana stellar-scattering model.

    This module uses the stellar-scattering rate constants from the fits in [Sesana2006]_ using the
    `_SHM06` class.  Scattering is assumed to only be effective once the binary is bound.  An
    exponential cutoff is imposed at larger radii.

    """

    def __init__(self, gamma_dehnen=1.0, mmbulge=None, msigma=None):
        """Construct an `Stellar_Scattering` instance with the given MBH-Host relations.

        Parameters
        ----------
        gamma_dehnen : array_like
            Dehnen stellar-density profile inner power-law slope.
            Fiducial Dehnen inner density profile slope ``gamma=1.0`` is used in [Chen2017]_.
        mmbulge : None or `holodeck.host_relations._MMBulge_Relation`
            Mbh-Mbulge relation to calculate stellar mass for a given BH mass.
            If `None` a default relationship is used.
        msigma : None or `holodeck.host_relations._MSigma_Relation`
            Mbh-Sigma relation to calculate stellar velocity dispersion for a given BH mass.
            If `None` a default relationship is used.

        """
        self._mmbulge = get_mmbulge_relation(mmbulge)
        self._msigma = get_msigma_relation(msigma)
        self._gamma_dehnen = gamma_dehnen
        self._shm06 = _SHM06()
        return

    def dadt_dedt(self, evo, step):
        """Stellar scattering hardening rate.

        Parameters
        ----------
        evo : `Evolution`
            Evolution instance providing binary parameters at the given intergration step.
        step : int
            Integration step at which to calculate hardening rates.

        Returns
        -------
        dadt : array_like
            Binary hardening rates in units of [cm/s], defined to be negative.
        dedt : array_like
            Binary rate-of-change of eccentricity in units of [1/sec].

        """
        mass = evo.mass[:, step, :]
        sepa = evo.sepa[:, step]
        eccen = evo.eccen[:, step] if evo.eccen is not None else None
        dadt, dedt = self._dadt_dedt(mass, sepa, eccen)
        return dadt, dedt

    def _dadt_dedt(self, mass, sepa, eccen):
        """Stellar scattering hardening rate calculated from physical quantities.

        Parameters
        ----------
        mass : (N,2) array_like
            Masses of each MBH component (0-primary, 1-secondary) in units of [gram].
        sepa : (N,) array_like
            Binary separation in units of [cm].
        eccen : (N,) array_like or `None`
            Binary eccentricity.  `None` if eccentricity is not being evolved.

        Returns
        -------
        dadt : (N,) array-like of scalar
            Binary hardening rates in units of [cm/s], defined to be negative.
        dedt : (N,) array-like of scalar  or  `None`
            Binary rate-of-change of eccentricity in units of [1/sec].
            If eccentricity is not being evolved (i.e. `eccen==None`) then `None` is returned.

        """
        mass = np.atleast_2d(mass)
        sepa = np.atleast_1d(sepa)
        eccen = np.atleast_1d(eccen) if eccen is not None else None
        mtot, mrat = utils.mtmr_from_m1m2(mass)


        mbulge = self._mmbulge.mbulge_from_mbh(mtot, scatter=False)
        vdisp = self._msigma.vdisp_from_mbh(mtot, scatter=False)
        dens = _density_at_influence_radius_dehnen(mtot, mbulge, self._gamma_dehnen)

        """ Make sure that mass ratio is always < 1, and find primary/secondary masses """
        mass_ratio_test = mass[:, 1]/mass[:, 0]
        inds_mrat_1 = mass_ratio_test>1
        secondary_mass = np.zeros(np.shape(mass[:, 1]))
        secondary_mass[inds_mrat_1] = mass[:, 0][inds_mrat_1]
        secondary_mass[~inds_mrat_1]  = mass[:, 1][~inds_mrat_1]
        #bug fix below: previously used mass[:,1] as secondary mass, this is not always true
        rhard = _Quinlan1996.radius_hardening(secondary_mass, vdisp)
        hh = self._shm06.H(mrat, sepa/rhard)
        dadt = _Quinlan1996.dadt(sepa, dens, vdisp, hh)

        rbnd = _radius_influence_dehnen(mtot, mbulge)
        atten = np.exp(-sepa / rbnd)
        dadt = dadt * atten

        if eccen is not None:
            kk = self._shm06.K(mrat, sepa/rhard, eccen)
            dedt = _Quinlan1996.dedt(sepa, dens, vdisp, hh, kk)
        else:
            dedt = None

        return dadt, dedt


class Dynamical_Friction_NFW(_Hardening):
    """Dynamical Friction (DF) hardening module assuming an NFW dark-matter density profile.

    This class calculates densities and orbital velocities based on a NFW profile with parameters based on those of
    each MBH binary.  The `holodeck.observations.NFW` class is used for profile calculations, and the halo parameters
    are calculated from Stellar-mass--Halo-mass relations (see 'arguments' below).  The 'effective-mass' of the
    inspiralling secondary is modeled as a power-law decreasing from the sum of secondary MBH and its stellar-bulge
    (calculated using the `mmbulge` - Mbh-Mbulge relation), down to just the bare secondary MBH after 10 dynamical
    times.  This is to model tidal-stripping of the secondary host galaxy.

    Attenuation of the DF hardening rate is typically also included, to account for the inefficiency of DF once the
    binary enters the hardened regime.  This is calculated using the prescription from [BBR1980]_.  The different
    characteristic radii, needed for the attenuation calculation, currently use a fixed Dehnen stellar-density profile
    as in [Chen2017]_, and a fixed scaling relationship to find the characteristic stellar-radius.

    Notes
    -----
    *   This module does not evolve eccentricity.
    *   The hardening rate (da/dt) is not allowed to be larger than the orbital/virial velocity of the halo
        (as a function of radius).

    """

    _TIDAL_STRIPPING_DYNAMICAL_TIMES = 10.0

    def __init__(self, mmbulge=None, msigma=None, smhm=None, coulomb=10.0, attenuate=True, rbound_from_density=True):
        """Create a hardening rate instance with the given parameters.

        Parameters
        ----------
        mmbulge : None or `holodeck.host_relations._MMBulge_Relation`
            Mbh-Mbulge relation to calculate stellar mass for a given BH mass.
            If `None` a default relationship is used.
        msigma : None or `holodeck.host_relations._MSigma_Relation`
            Mbh-Sigma relation to calculate stellar velocity dispersion for a given BH mass.
            If `None` a default relationship is used.
        smhm : class, instance or None
            Stellar-mass--halo-mass relation (_StellarMass_HaloMass subclass)
            If `None` the default is loaded.
        coulomb : scalar,
            coulomb-logarithm ("log(Lambda)"), typically in the range of 10-20.
            This parameter is formally the log of the ratio of maximum to minimum impact parameters.
        attenuate : bool,
            Whether the DF hardening rate should be 'attenuated' due to stellar-scattering effects at
            small radii.  If `True`, DF becomes significantly less effective for radii < R-hard and R-LC
        rbound_from_density : bool,
            Determines how the binding radius (of MBH pair) is calculated, which is used for attenuation.
            NOTE: this is only used if `attenuate==True`
            If True:  calculate R-bound using an assumed stellar density profile.
            If False: calculate R-bound using a velocity dispersion (constant in radius, from `gbh` instance).

        """
        self._mmbulge = get_mmbulge_relation(mmbulge)
        self._msigma = get_msigma_relation(msigma)
        self._smhm = get_stellar_mass_halo_mass_relation(smhm)
        self._coulomb = coulomb
        self._attenuate = attenuate
        self._rbound_from_density = rbound_from_density

        self._NFW = galaxy_profiles.NFW
        self._time_dynamical = None
        return

    def dadt_dedt(self, evo, step, attenuate=None):
        """Calculate DF hardening rate given `Evolution` instance, and an integration `step`.

        Parameters
        ----------
        evo : `Evolution` instance
            The evolutionary tracks of the binary population, providing binary parameters.
        step : int,
            Integration step at which to calculate hardening rates.

        Returns
        -------
        dadt : (N,) np.ndarray of scalar,
            Binary hardening rates in units of [cm/s].
        dedt : (N,) np.ndarray or None
            Rate-of-change of eccentricity, which is not included in this calculation, it is zero.
            `None` is returned if the input `eccen` is None.

        """
        if attenuate is None:
            attenuate = self._attenuate

        mass = evo.mass[:, step, :]
        sepa = evo.sepa[:, step]
        eccen = evo.eccen[:, step] if evo.eccen is not None else None
        dt = evo.tlook[:, 0] - evo.tlook[:, step]   # positive time-duration since 'formation'
        # NOTE `scafa` is nan for systems "after" redshift zero (i.e. do not merge before redz=0)
        redz = np.zeros_like(sepa)
        val = (evo.scafa[:, step] > 0.0)
        redz[val] = cosmo.a_to_z(evo.scafa[val, step])

        dadt, dedt = self._dadt_dedt(mass, sepa, redz, dt, eccen, attenuate)

        return dadt, dedt

    def _dadt_dedt(self, mass, sepa, redz, dt, eccen, attenuate):
        """Calculate DF hardening rate given physical quantities.

        Parameters
        ----------
        mass : (N, 2) array_like
            Masses of both MBHs (0-primary, 1-secondary) in units of [grams].
        sepa : (N,) array_like
            Binary separation in [cm].
        redz : (N,) array_like
            Binary redshifts.
        dt : (N,) array_like
            Time step [sec], required for modeling tidal stripping of secondary galaxy.
        eccen : (N,) array_like or None
            Binary eccentricity.
        attenuate : bool
            Whether to include 'attenuation' as the radius approach the stellar-scattering regime.

        Returns
        -------
        dadt : (N,) np.ndarray
            Binary hardening rates in units of [cm/s].
        dedt : (N,) np.ndarray or None
            Rate-of-change of eccentricity, which is not included in this calculation, it is zero.
            `None` is returned if the input `eccen` is None.

        """
        assert np.shape(mass)[-1] == 2 and np.ndim(mass) <= 2
        mass = np.atleast_2d(mass)
        redz = np.atleast_1d(redz)

        # Get Host DM-Halo mass
        # assume galaxies are merged, and total stellar mass is given from Mstar-Mbh of total MBH mass
        mstar = self._mmbulge.mstar_from_mbh(mass.sum(axis=-1), scatter=False)
        mhalo = self._smhm.halo_mass(mstar, redz, clip=True)

        # ---- Get effective mass of inspiraling secondary
        m2 = mass[:, 1]
        mstar_sec = self._mmbulge.mstar_from_mbh(m2, scatter=False)
        # model tidal-stripping of secondary's bulge (see: [Kelley2017a]_ Eq.6)
        time_dyn = self._NFW.time_dynamical(sepa, mhalo, redz)
        tfrac = dt / (time_dyn * self._TIDAL_STRIPPING_DYNAMICAL_TIMES)
        power_index = np.clip(1.0 - tfrac, 0.0, 1.0)
        meff = m2 * np.power((m2 + mstar_sec)/m2, power_index)
        log.debug(f"DF tfrac = {utils.stats(tfrac)}")
        log.debug(f"DF meff/m2 = {utils.stats(meff/m2)} [Msol]")

        # ---- Get local density
        # set minimum radius to be a factor times influence-radius
        rinfl = _MIN_DENS_RAD__INFL_RAD_MULT * _radius_influence_dehnen(m2, mstar_sec)
        dens_rads = np.maximum(sepa, rinfl)
        dens = self._NFW.density(dens_rads, mhalo, redz)

        # ---- Get velocity of secondary MBH
        mt, mr = utils.mtmr_from_m1m2(mass)
        vhalo = self._NFW.velocity_circular(sepa, mhalo, redz)
        vorb = utils.velocity_orbital(mt, mr, sepa=sepa)[:, 1]  # secondary velocity
        velo = np.sqrt(vhalo**2 + vorb**2)

        # ---- Calculate hardening rate
        # dvdt is negative [cm/s]
        dvdt = self._dvdt(meff, dens, velo)
        # convert from deceleration to hardening-rate assuming virialized orbit (i.e. ``GM/a = v^2``)
        dadt = 2 * time_dyn * dvdt
        dedt = None if (eccen is None) else np.zeros_like(dadt)

        # ---- Apply 'attenuation' following [BBR1980]_ to account for stellar-scattering / loss-cone effects
        if attenuate:
            atten = self._attenuation_BBR1980(sepa, mass, mstar)
            dadt = dadt / atten

        # Hardening rate cannot be larger than orbital/virial velocity
        clip = (np.fabs(dadt) > velo)
        if np.any(clip):
            log.info(f"clipping {utils.frac_str(clip)} `dadt` values to vcirc")
            dadt[clip] = - velo[clip]

        return dadt, dedt

    def _dvdt(self, mass_sec_eff, dens, velo):
        """Chandrasekhar dynamical friction formalism providing a deceleration (dv/dt).

        Parameters
        ----------
        mass_sec_eff : (N,) array-like of scalar
            Effective mass (i.e. the mass that should be used in this equation) of the inspiraling
            secondary component in units of [gram].
        dens : (N,) array-like of scalar
            Effective density at the location of the inspiraling secondary in units of [g/cm^3].
        velo : (N,) array-like of scalar
            Effective velocity of the inspiraling secondary in units of [cm/s].

        Returns
        -------
        dvdt (N,) np.ndarray of scalar
            Deceleration rate of the secondary object in units of [cm/s^2].

        """
        dvdt = - 2*np.pi * mass_sec_eff * dens * self._coulomb * np.square(NWTG / velo)
        return dvdt

    def _attenuation_BBR1980(self, sepa, m1m2, mstar):
        """Calculate attentuation factor following [BBR1980]_ prescription.

        Characteristic radii are currently calculated using hard-coded Dehnen stellar-density profiles, and a fixed
        scaling-relationship between stellar-mass and stellar characteristic radius.

        The binding radius can be calculated either using the stellar density profile, or from a velocity dispersion,
        based on the `self._rbound_from_density` flag.  See the 'arguments' section of `docs::Dynamical_Friction_NFW`.

        The attenuation factor is defined as >= 1.0, with 1.0 meaning no attenuation.

        Parameters
        ----------
        sepa : (N,) array-like of scalar,
            Binary separations in units of [cm].
        m1m2 : (N, 2) array-like of scalar,
            Masses of each binary component (0-primary, 1-secondary).
        mstar : (N,) array-like of scalar,
            Mass of the stellar-bulge / stellar-core (ambiguous).

        Returns
        -------
        atten : (N,) np.ndarray of scalar
            Attenuation factor (defined as >= 1.0).

        """

        m1, m2 = m1m2.T
        mbh = m1 + m2

        # characteristic stellar radius in [cm]
        rstar = _radius_stellar_characteristic_dabringhausen_2008(mstar)
        # characteristic hardening-radius in [cm]
        rhard = _radius_hard_BBR1980_dehnen(mbh, mstar)
        # characteristic loss-cone-radius in [cm]
        rlc = _radius_loss_cone_BBR1980_dehnen(mbh, mstar)

        # Calculate R-bound based on stellar density profile (mass enclosed)
        if self._rbound_from_density:
            rbnd = _radius_influence_dehnen(mbh, mstar)
        # Calculate R-bound based on uniform velocity dispersion (MBH scaling relation)
        else:
            vdisp = self._msigma.vdisp_from_mbh(m1)   # use primary-bh's mass (index 0)
            rbnd = NWTG * mbh / vdisp**2

        # Number of stars in the stellar bulge/core
        nstar = mstar / (0.6 * MSOL)
        # --- Attenuation for separations less than the hardening radius
        # [BBR1980] Eq.3
        atten_hard = np.maximum((rhard/sepa) * np.log(nstar), np.square(mbh/mstar) * nstar)
        # use an exponential turn-on at larger radii
        cut = np.exp(-sepa/rhard)
        atten_hard *= cut

        # --- Attenuation for separations less than the loss-cone Radius
        # [BBR1980] Eq.2
        atten_lc = np.power(m2/m1, 1.75) * nstar * np.power(rbnd/rstar, 6.75) * (rlc / sepa)
        atten_lc = np.maximum(atten_lc, 1.0)
        # use an exponential turn-on at larger radii
        cut = np.exp(-sepa/rlc)
        atten_hard *= cut

        atten = np.maximum(atten_hard, atten_lc)
        # Make sure that attenuation is always >= 1.0 (i.e. this never _increases_ the hardening rate)
        atten = np.maximum(atten, 1.0)
        return atten


class Fixed_Time_2PL(_Hardening):
    r"""Provide a binary hardening rate such that the total lifetime matches a given value.

    This class uses a phenomenological functional form (defined in :meth:`Fixed_Time_2PL.function`) to
    model the hardening rate ($da/dt$) of binaries.  The functional form is,

    .. math::
        \dot{a} = - A * (1.0 + x)^{-g_2 - 1} / x^{g_1 - 1},

    where :math:`x \equiv a / r_\mathrm{char}` is the binary separation scaled to a characteristic
    transition radius (:math:`r_\mathrm{char}`) between two power-law indices $g_1$ and $g_2$.  There is
    also an overall normalization $A$ that is calculated to yield the desired binary lifetimes.

    The normalization for each binary, to produce the desired lifetime, is calculated as follows:

    (1) A set of random test binary parameters are chosen.

    (2) The normalization constants are determined, using least-squares optimization, to yield the
        desired lifetime.

    (3) Interpolants are constructed to interpolate between the test binary parameters.

    (4) The interpolants are called on the provided binary parameters, to calculate the
        interpolated normalization constants to reach the desired lifetimes.

    Construction/Initialization: note that in addition to the standard :meth:`Fixed_Time_2PL.__init__`
    constructor, there are two additional constructors are provided:

    *   :meth:`Fixed_Time_2PL.from_pop` - accept a :class:`holodeck.population._Discrete_Population`,
    *   :meth:`Fixed_Time_2PL.from_sam` - accept a :class:`holodeck.sam.Semi_Analytic_Model`.

    #! Using a callable for `rchar` probably doesnt work - `_calculate_norm_interpolant` looks like
    #! it only accepts a scalar value.

    """

    # _INTERP_NUM_POINTS = 1e4             #: number of random data points used to construct interpolant
    _INTERP_NUM_POINTS = 3e4
    _INTERP_THRESH_PAD_FACTOR = 5.0      #: allowance for when to use chunking and when to process full array
    _NORM_CHUNK_SIZE = 1e3
    CONSISTENT = True

    def __init__(self, time, mtot, mrat, redz, sepa_init,
                 rchar=100.0*PC, gamma_inner=-1.0, gamma_outer=+1.5,
                 progress=False, interpolate_norm=False):
        """Initialize `Fixed_Time_2PL` instance for the given binary properties and function parameters.

        Parameters
        ----------
        time : float,  callable  or  array_like
            Total merger time of binaries, units of [sec], specifiable in the following ways:

            *   float : uniform merger time for all binaries
            *   callable : function `time(mtot, mrat, redz)` which returns the total merger time
            *   array_like : (N,) matching the shape of `mtot` (etc) giving the merger time for
                each binary

        mtot : array_like
            Binary total-mass [gram].
        mrat : array_like
            Binary mass-ratio $q \\equiv m_2 / m_1 \\leq 1$.
        redz : array_like
            Binary Redshift.
            NOTE: this is only used as an argument to callable `rchar` and `time` values.
        sepa_init : array_like
            Binary semi-major axis (separation) [cm].
        rchar : scalar  or  callable
            Characteristic radius dividing two power-law regimes, in units of [cm]:
            *   scalar : uniform radius for all binaries
            *   callable : function `rchar(mtot, mrat, redz)` which returns the radius
        gamma_inner : scalar
            Power-law of hardening timescale in the stellar-scattering regime,
            (small separations: $r < rchar$), at times referred to internally as `g1`.
        gamma_outer : scalar
            Power-law of hardening timescale in the dynamical-friction regime
            (large separations: $r > rchar$), at times referred to internally as `g2`.

        """
        self._progress = progress

        # ---- Initialize / Sanitize arguments

        # Ensure `time` is ndarray matching binary variables
        if np.isscalar(time):
            time = time * np.ones_like(mtot)
        elif callable(time):
            time = time(mtot, mrat, redz)
        elif np.shape(time) != np.shape(mtot):
            err = f"Shape of `time` ({np.shape(time)}) does not match `mtot` ({np.shape(mtot)})!"
            log.exception(err)
            raise ValueError(err)

        # `rchar` must be a function of only mtot, mrat; or otherwise a fixed value
        # This is because it is not being used as an interpolation variable, only an external parameter
        # FIX/BUG: either an ndarray could be allowed when interpolation is not needed (i.e. small numbers of systems)
        #      or `rchar` could be added as an explicit interpolation variable
        if callable(rchar):
            log.warning("!!It looks like you're using a callable `rchar`, this probably doesn't work!!")
            rchar = rchar(mtot, mrat, redz)
        elif not np.isscalar(rchar):
            err = "`rchar` must be a scalar or callable: (`rchar(mtot, mrat)`)!"
            log.exception(err)
            raise ValueError(err)

        # ---- Calculate normalization parameter
        self._sepa_init = sepa_init
        mtot, mrat, time, sepa_init = np.broadcast_arrays(mtot, mrat, time, sepa_init)
        if mtot.ndim != 1:
            err = f"Error in input shapes (`mtot.shape={np.shape(mtot)})"
            log.exception(err)
            raise ValueError(err)

        # if `interpolate_norm` is True:  use an interpolant.
        # if `interpolate_norm` is False: use exact calculation.
        # if `interpolate_norm` is None:  use an interpolant if there are lots of points to calculate
        interp_num_thresh = self._INTERP_THRESH_PAD_FACTOR * self._INTERP_NUM_POINTS
        log.debug(f"size={len(mtot)} vs. limit={interp_num_thresh}; `interpolate_norm`={interpolate_norm}")
        if (interpolate_norm is True) or ((interpolate_norm is None) and (len(mtot) > interp_num_thresh)):
            log.debug("constructing hardening normalization interpolant")
            log.warning("INTERPOLATED NORMALIZATION DOES NOT PERFORM AS WELL")
            # both are callable as `interp(args)`, with `args` shaped (N, 4),
            # the 4 parameters are:      [log10(M/MSOL), log10(q), time/Gyr, log10(Rmax/PC)]
            # the interpolants return the log10 of the norm values
            interp, backup = self._calculate_norm_interpolant(rchar, gamma_inner, gamma_outer)

            log.debug("calculating normalization from interpolants")
            points = [np.log10(mtot/MSOL), np.log10(mrat), time/GYR, np.log10(sepa_init/PC)]
            points = np.array(points)
            norm = interp(points.T)
            bads = ~np.isfinite(norm)
            if np.any(bads):
                msg = f"Normal interpolant failed on {utils.frac_str(bads, 4)} points.  Using backup interpolant"
                log.info(msg)
                bp = points.T[bads]
                # If scipy throws an error on the shape here, see: https://github.com/scipy/scipy/issues/4123
                # or https://stackoverflow.com/a/26806707/230468
                norm[bads] = backup(bp)
                bads = ~np.isfinite(norm)
                if np.any(bads):
                    err = f"Backup interpolant failed on {utils.frac_str(bads, 4)} points!"
                    log.exception(err)
                    raise ValueError(err)

            norm = 10.0 ** norm

        # For small numbers of points, calculate the normalization directly
        else:
            log.info("calculating normalization exactly")
            norm = self._get_norm_chunk(time, mtot, mrat, rchar, gamma_inner, gamma_outer, sepa_init, progress=progress)

        self._gamma_inner = gamma_inner
        self._gamma_outer = gamma_outer
        self._time = time
        self._norm = norm
        self._rchar = rchar
        return

    # ====     Constructors    ====

    @classmethod
    def from_pop(cls, pop, time, **kwargs):
        """Initialize a `Fixed_Time_2PL` instance using a provided `_Discrete_Population` instance.

        Parameters
        ----------
        pop : `_Discrete_Population`
            Input population, from which to use masses, redshifts and separations.
        time : float,  callable  or  array_like
            Total merger time of binaries, units of [sec], specifiable in the following ways:

            *   float : uniform merger time for all binaries
            *   callable : function `time(mtot, mrat, redz)` which returns the total merger time
            *   array_like : (N,) matching the shape of `mtot` (etc) giving the merger time for
                each binary

        **kwargs : dict
            Additional keyword-argument pairs passed to the `Fixed_Time_2PL` initialization method.

        Returns
        -------
        `Fixed_Time_2PL`
            Instance configured for the given binary population.

        """
        return cls(time, *pop.mtmr, pop.redz, pop.sepa, **kwargs)

    @classmethod
    def from_sam(cls, sam, time, sepa_init=1e4*PC, **kwargs):
        """Initialize a `Fixed_Time_2PL` instance using a provided `Semi_Analytic_Model` instance.

        Parameters
        ----------
        sam : `holodeck.sam.Semi_Analytic_Model`
            Input population, from which to use masses, redshifts and separations.
        time : float,  callable  or  array_like
            Total merger time of binaries, units of [sec], specifiable in the following ways:

            *   float : uniform merger time for all binaries
            *   callable : function `time(mtot, mrat, redz)` which returns the total merger time
            *   array_like : (N,) matching the shape of `mtot` (etc) giving the merger time for
                each binary

        sepa_init : float  or  array_like
            Initial binary separation.  Units of [cm].

            *   float : initial separation applied to all binaries,
            *   array_like : initial separations for all binaries, shaped (N,) matching the number
                binaries.

        **kwargs : dict
            Additional keyword-argument pairs passed to the `Fixed_Time_2PL` initialization method.

        Returns
        -------
        `Fixed_Time_2PL`
            Instance configured for the given binary population.

        """
        grid = np.meshgrid(*sam.edges, indexing='ij')
        mtot, mrat, redz = [gg.ravel() for gg in grid]
        return cls(time, mtot, mrat, redz, sepa_init, **kwargs)

    # ====     Hardening Rate Methods    ====

    def dadt_dedt(self, evo, step):
        """Calculate hardening rate at the given integration `step`, for the given population.

        Parameters
        ----------
        evo : `Evolution` instance
            The evolutionary tracks of the binary population, providing binary parameters.
        step : int,
            Integration step at which to calculate hardening rates.

        Returns
        -------
        dadt : (N,) np.ndarray
            Binary hardening rates in units of [cm/s].
        dedt : (N,) np.ndarray or None
            Rate-of-change of eccentricity, which is not included in this calculation, it is zero.
            `None` is returned if the input `eccen` is None.

        """
        mass = evo.mass[:, step, :]
        sepa = evo.sepa[:, step]
        mt, mr = utils.mtmr_from_m1m2(mass)
        dadt, _dedt = self._dadt_dedt(mt, mr, sepa, self._norm, self._rchar, self._gamma_inner, self._gamma_outer)
        dedt = None if evo.eccen is None else np.zeros_like(dadt)
        return dadt, dedt

    def dadt(self, mt, mr, sepa):
        dadt, _dedt = self._dadt_dedt(mt, mr, sepa, self._norm, self._rchar, self._gamma_inner, self._gamma_outer)
        return dadt

    def dedt(self, mt, mr, sepa):
        _dadt, dedt = self._dadt_dedt(mt, mr, sepa, self._norm, self._rchar, self._gamma_inner, self._gamma_outer)
        return dedt

    @classmethod
    def _dadt_dedt(cls, mtot, mrat, sepa, norm, rchar, gamma_inner, gamma_outer):
        r"""Calculate hardening rate for the given raw parameters.

        Parameters
        ----------
        mtot : array_like
            Binary total-mass [gram].
        mrat : array_like
            Binary mass-ratio :math:`q \equiv m_2 / m_1 \leq 1`.
        redz : array_like
            Redshift.
        sepa : array_like
            Binary semi-major axis (separation) [cm].
        norm : array_like
            Hardening rate normalization, units of [cm/s].
        rchar : array_like
            Characteristic transition radius between the two power-law indices of the hardening
            rate model, units of [cm].
        gamma_inner : scalar
            Power-law of hardening timescale in the inner regime (small separations: ``r < rchar``).
        gamma_outer : scalar
            Power-law of hardening timescale in the outer regime (large separations: ``r > rchar``).

        Returns
        -------
        dadt : (N,) np.ndarray
            Binary hardening rates in units of [cm/s].
        dedt : (N,) np.ndarray or None
            Rate-of-change of eccentricity, which is not included in this calculation, it is zero.
            `None` is returned if the input `eccen` is None.

        """
        m1, m2 = utils.m1m2_from_mtmr(mtot, mrat)
        dadt_gw = utils.gw_hardening_rate_dadt(m1, m2, sepa)

        xx = sepa / rchar
        dadt = cls.function(norm, xx, gamma_inner, gamma_outer)
        dadt += dadt_gw

        dedt = None
        return dadt, dedt

    @classmethod
    def function(cls, norm, xx, gamma_inner, gamma_outer):
        r"""Hardening rate given the parameters for this hardening model.

        The functional form is,

        .. math::

            \dot{a} = - A * (1.0 + x)^{-g_out + g_in} / x^{g_in - 1},

        Where $A$ is an overall normalization, and :math:`x = a / r_\textrm{char}` is the binary
        separation scaled to a characteristic transition radius (:math:`r_\textrm{char}`) between two
        power-law indices $g_inner$ and $g_outer$.

        Parameters
        ----------
        norm : array_like
            Hardening rate normalization, units of [cm/s].
        xx : array_like
            Dimensionless binary separation, the semi-major axis in units of the characteristic
            (i.e. transition) radius of the model `rchar`.
        gamma_inner : scalar
            Power-law of hardening timescale in the inner regime (small separations: r < rchar).
        gamma_outer : scalar
            Power-law of hardening timescale in the outer regime (large separations: r > rchar).

        """
        dadt = - norm * np.power(1.0 + xx, -gamma_outer+gamma_inner) / np.power(xx, gamma_inner-1)
        return dadt

    # ====     Internal Functions    ====

    @classmethod
    def _calculate_norm_interpolant(cls, rchar, gamma_inner, gamma_outer):
        """Generate interpolants to map from binary parameters to hardening rate normalization.

        Interpolants are calculated as follows:
        (1) A set of random test binary parameters and lifetimes are chosen.
        (2) The normalizations to yield those binary lifetimes are calculated with least-squares
            optimization.
        (3) Interpolants are constructed to yield the normalization paramters for the given
            binary parameters and binary lifetime.

        Two interpolators are returned, a linear-interpolator that is the preferable one, and a
        backup nearest-interplator that is more robust and works at times when the linear
        interpolator fails.

        Parameters
        ----------
        rchar : scalar  or  array_like  #! Possible that only a scalar value is currently working!
            Characteristic radius separating the two power-law regimes, in units of [cm]:
            *   scalar : uniform radius for all binaries
            *   array_like : characteristic radius for each binary.
        gamma_inner : scalar
            Power-law of hardening timescale in the stellar-scattering regime,
            (small separations: r < rchar), at times referred to internally as `g1`.
        gamma_outer : scalar
            Power-law of hardening timescale in the dynamical-friction regime
            (large separations: r > rchar), at times referred to internally as `g1`.

        Returns
        -------
        interp : callable
            Linear interpolator from (M, q, t, r) => A
            (total-mass, mass-ratio, lifetime, initial-radius) => hardening normalization
        backup : callable
            Nearest interpolator from (M, q, t, r) => A, to use as a backup when `interp` fails.
            (total-mass, mass-ratio, lifetime, initial-radius) => hardening normalization

        """

        def get_norm_for_random_points(num_points):
            num = int(num_points)
            max_fix_tries = 10

            # ---- Initialization
            # Define the range of parameters to be explored
            mt = [1e5, 1e11]   #: total mass [Msol]
            mr = [1e-5, 1.0]   #: mass ratio
            # td = [0.0, 20.0]   #: lifetime [Gyr]    LINEAR
            td = [1.0e-3, 3.0e1]   #: lifetime [Gyr]        LOG
            rm = [1e3, 1e5]    #: radius maximum (initial separation) [pc]

            # Choose random test binary parameters
            mt = 10.0 ** np.random.uniform(*np.log10(mt), num) * MSOL
            mr = 10.0 ** np.random.uniform(*np.log10(mr), num)
            td = np.random.uniform(*td, num+1)[1:] * GYR
            # td = 10.0 ** np.random.uniform(*np.log10(td), num) * GYR
            rm = 10.0 ** np.random.uniform(*np.log10(rm), num) * PC

            # ---- Get normalization for these parameters
            norm = cls._get_norm_chunk(td, mt, mr, rchar, gamma_inner, gamma_outer, rm)
            points = [mt, mr, td, rm]
            bads = ~(np.isfinite(norm) & (norm > 0.0))
            fix_tries = 0
            while np.any(bads) and (fix_tries < max_fix_tries):
                err = f"bad random point norms {utils.frac_str(bads)}, norms: {utils.stats(norm)}"
                if fix_tries >= max_fix_tries:
                    log.exception(err)
                    raise RuntimeError(err)

                # bads = np.where(bads)[0]
                # units = [MSOL, 1.0, GYR, PC]
                # for pp, un in zip(points, units):
                #     # print(f"{utils.stats(pp[bads]/un)}")
                #     print(f"{pp[bads]/un}")

                args = [aa[bads] for aa in [td, mt, mr, rm]]
                num = args[0].size
                args = [aa*np.random.normal(1.0, 0.05, num) for aa in args]
                # args = [td, mt, mr, rchar, gamma_inner, gamma_outer, rm]
                args = args[:-1] + [rchar, gamma_inner, gamma_outer] + [args[-1],]
                norm[bads] = cls._get_norm_chunk(*args)
                fix_tries += 1
                log.debug(f"fixing bad norm values: {fix_tries} :: {norm[bads]=}")
                bads = ~(np.isfinite(norm) & (norm > 0.0))

            return norm, points

        def convert_points_to_interp_vals(points):
            units = [MSOL, 1.0, GYR, PC]
            logs = [True, True, False, True]   #: which parameters to interpolate in log-space
            # logs = [True, True, True, True]   #: which parameters to interpolate in log-space
            vals = [pp/uu for pp, uu in zip(points, units)]
            vals = [np.log10(pp) if ll else pp for pp, ll in zip(vals, logs)]
            vals = np.array(vals).T
            return vals

        num_points = int(cls._INTERP_NUM_POINTS)
        log.debug("calculating exact normalization for {num_points:.2e} points")
        norm, points = get_norm_for_random_points(num_points)

        # Make sure results are valid
        valid = np.isfinite(norm) & (norm > 0.0)
        frac_val = np.count_nonzero(valid) / valid.size
        if frac_val > 0.9:
            norm = norm[valid]
            points = [pp[valid] for pp in points]
        else:
            log.error(f"norms from random points: {utils.stats(norm)}")
            err = f"Invalid normalizations!  {utils.frac_str(valid)}"
            log.exception(err)
            raise ValueError(err)

        vals = convert_points_to_interp_vals(points)

        # ---- Construct interpolants

        # construct both a linear (1th order) and nearest (0th order) interpolant
        log.debug("constructing interpolants")
        interp = sp.interpolate.LinearNDInterpolator(vals, np.log10(norm), rescale=True)
        backup = sp.interpolate.NearestNDInterpolator(vals, np.log10(norm), rescale=True)

        log.debug("testing interpolants")
        check_norm, check_points = get_norm_for_random_points(1000)
        check_vals = convert_points_to_interp_vals(check_points)
        interp_norm = 10.0 ** interp(check_vals)
        backup_norm = 10.0 ** backup(check_vals)
        error_interp = (interp_norm - check_norm) / check_norm
        error_backup = (backup_norm - check_norm) / check_norm

        log.debug(f"{utils.stats(check_norm)=}")
        log.debug(f"{utils.stats(interp_norm)=}")
        log.debug(f"{utils.stats(backup_norm)=}")
        log.debug(f"{utils.stats(error_interp)=}")
        log.debug(f"{utils.stats(error_backup)=}")

        return interp, backup

    @classmethod
    def _get_norm_chunk(cls, target_time, *args, progress=True, **kwargs):
        """Calculate normalizations in 'chunks' of the input arrays, to obtain the target lifetime.

        Calculates normalizations for groups of parameters of size `chunk` at a time.  Loops over
        these chunks until all inputs have been processed.  Calls :meth:`Fixed_Time_2PL._get_norm` to
        calculate the normalization for each chunk.

        Parameters
        ----------
        target_time : (N,) np.ndarray
            Target binary lifetimes, units of [sec].
        *args : list[np.ndarray]
            The parameters eventually passed to :meth:`Fixed_Time_2PL._time_total`, to get the total
            lifetime.  The normalization parameter is varied until the `_time_total` return value
            matches the target input lifetime.
        guess : float
            Initial value of the normalization parameter for the optimization routine to start on.
            Units of [cm/s].
        chunk : float
            Size of each 'chunk' of parameters to process at a time, cast to `int`.
        progress : bool
            Whether or not to show a `tqdm` progress bar while iterating over chunks.

        Returns
        -------
        norm : (N,) np.ndarray
            The normalizations required to produce the target lifetimes given by `target_time`.

        """
        if np.ndim(target_time) not in [0, 1]:
            raise

        chunk_size = int(cls._NORM_CHUNK_SIZE)
        size = np.size(target_time)
        # if number of array elements is less than (or comparable to) chunk size, to it all in one pass
        if size <= chunk_size * cls._INTERP_THRESH_PAD_FACTOR:
            return cls._get_norm(target_time, *args, **kwargs)

        # broadcast arrays to all be the same shape (some `floats` are passed in)
        args = [target_time, *args]
        target_time, *args = np.broadcast_arrays(*args)

        # iterate over each chunk, storing the normalization values
        num = int(np.ceil(size / chunk_size))
        norm = np.zeros_like(target_time)
        step_iter = range(num)
        step_iter = utils.tqdm(step_iter, desc='calculating hardening normalization') if progress else step_iter
        for ii in step_iter:
            lo = ii * chunk_size
            hi = np.minimum((ii + 1) * chunk_size, size)
            cut = slice(lo, hi)
            # calculate normalizations for this chunk
            norm[cut] = cls._get_norm(target_time[cut], *[aa[cut] for aa in args], **kwargs)

        return norm

    @classmethod
    def _get_norm(cls, target_time, *args, guess=1e7, max_err=1e-6):
        """Calculate normalizations of the input arrays, to obtain the target binary lifetime.

        Uses deterministic least-squares optimization to find the best normalization values, using
        `scipy.optimize.newton`.

        Parameters
        ----------
        target_time : (N,) np.ndarray
            Target binary lifetimes, units of [sec].
        *args : list[np.ndarray]
            The parameters eventually passed to :meth:`Fixed_Time_2PL._time_total`, to get the total
            lifetime.  The normalization parameter is varied until the `_time_total` return value
            matches the target input lifetime.
        guess : float
            Initial value of the normalization parameter for the optimization routine to start on.
            Units of [cm/s].

        Returns
        -------
        norm : (N,) np.ndarray
            The normalizations required to produce the target lifetimes given by `target_time`.

        """

        # convenience wrapper function
        def integ(norm):
            return cls._time_total(norm, *args)

        # Assume linear scaling to refine the first guess
        g0 = guess * np.ones_like(target_time)
        test = integ(g0)
        g1 = g0 * (test / target_time)
        log.debug(f"Guess {guess:.4e} ==> {utils.stats(g1)}")

        # perform optimization
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            norm = sp.optimize.newton(lambda xx: integ(xx) - target_time, g1, maxiter=200, tol=1e-6)
            err = (integ(norm) - target_time) / target_time
            log.debug(f"Fixed_Time_2PL._get_norm() : errors = {utils.stats(err)}")
            if np.any(err > max_err):
                fail = f"Errors in `Fixed_Time_2PL` norm exceed allowed: {utils.stats(err)} vs. {max_err})"
                log.exception(fail)
                raise ValueError(fail)

        return norm

    def time_total(self, mt, mr):
        args = [self._norm, mt, mr, self._rchar, self._gamma_inner, self._gamma_outer, self._sepa_init]
        args = np.broadcast_arrays(*args)
        return self._time_total(*args)

    @classmethod
    def _time_total(cls, norm, mt, mr, rchar, gamma_inner, gamma_outer, sepa_init, num=123):
        r"""For the given parameters, integrate the binary evolution to find total lifetime.

        Parameters
        ----------
        norm : float  or  array_like
            Hardening rate normalization, units of [cm/s].
        mtot : float  or  array_like
            Binary total-mass [gram].
        mrat : float  or  array_like
            Binary mass-ratio $q \equiv m_2 / m_1 \leq 1$.
        rchar : float  or  array_like
            Characteristic transition radius between the two power-law indices of the hardening
            rate model, units of [cm].
        gamma_inner : float  or  array_like
            Power-law of hardening timescale in the stellar-scattering regime,
            (small separations: r < rchar).
        gamma_outer : float  or  array_like
            Power-law of hardening timescale in the dynamical-friction regime,
            (large separations: r > rchar).
        sepa_init : float  or  array_like
            Initial binary separation.  Units of [cm].
        num : int
            Number of steps in separation overwhich to integrate the binary evolution

        Returns
        -------
        tt : np.ndarray
            Total binary lifetime [sec].

        """

        # Convert input values to broadcastable np.ndarray's
        norm = np.atleast_1d(norm)
        rmin = utils.rad_isco(mt)

        args = [norm, mt, mr, rchar, gamma_inner, gamma_outer, sepa_init, rmin]
        args = np.broadcast_arrays(*args)
        norm, mt, mr, rchar, gamma_inner, gamma_outer, sepa_init, rmin = args

        # define separations (radii) for each binary's evolution
        rextr = np.log10([rmin, sepa_init]).T
        rads = np.linspace(0.0, 1.0, num)[np.newaxis, :]
        rads = rextr[:, 0, np.newaxis] + rads * np.diff(rextr, axis=1)
        # (N, R) for N-binaries and R-radii (`num`)
        rads = 10.0 ** rads

        # Make hardening parameters broadcastable
        args = [norm, mt, mr, rchar, gamma_inner, gamma_outer]
        args = [aa[:, np.newaxis] for aa in args]
        norm, mt, mr, rchar, gamma_inner, gamma_outer = args

        # Calculate hardening rates along full evolutionary history
        dadt, _ = cls._dadt_dedt(mt, mr, rads, norm, rchar, gamma_inner, gamma_outer)

        # Integrate (inverse) hardening rates to calculate total lifetime
        tt = utils.trapz_loglog(- 1.0 / dadt, rads, axis=-1)
        tt = tt[:, -1]

        # dadt_left = dadt[..., :-1]
        # dadt_right = dadt[..., 1:]
        # sepa_left = rads[..., :-1]
        # sepa_right = rads[..., 1:]
        # tt = 2.0 * (sepa_left - sepa_right) / (dadt_left + dadt_right)
        # tt = np.sum(tt, axis=-1)

        return tt


class Fixed_Time_2PL_SAM(_Hardening):
    """SAM-Optimized version of `Fixed_Time_2PL`: binary evolution for a fixed total lifetime.
    """

    CONSISTENT = True

    def __init__(self, sam, time, sepa_init=1.0e3*PC, rchar=10.0*PC, gamma_inner=-1.0, gamma_outer=+1.5, num_steps=300):
        """Initialize a `Fixed_Time_2PL_SAM` instance using a provided `Semi_Analytic_Model` instance.

        Parameters
        ----------
        sam : `holodeck.sam.Semi_Analytic_Model`
            Input population, from which to use masses, redshifts and separations.
        time : float,
            Total merger time of binaries, units of [sec].
        sepa_init : float,
            Initial binary separation.  Units of [cm].
        **kwargs : dict
            Additional keyword-argument pairs passed to the `Fixed_Time_2PL_SAM` initialization method.

        Returns
        -------
        `Fixed_Time_2PL_SAM`
            Instance configured for the given binary population.

        """
        import holodeck.sams  # noqa
        import holodeck.sams.sam_cyutils  # noqa

        assert np.ndim(time) == 0
        assert np.ndim(rchar) == 0
        assert np.ndim(gamma_inner) == 0
        assert np.ndim(gamma_outer) == 0
        mtot, mrat = np.meshgrid(sam.mtot, sam.mrat, indexing='ij')
        shape = mtot.shape
        mt, mr = [mm.flatten() for mm in [mtot, mrat]]

        norm_log10 = holo.sams.sam_cyutils.find_2pwl_hardening_norm(
            time, mt, mr,
            sepa_init, rchar, gamma_inner, gamma_outer, num_steps,
        )
        # (M*Q,) ==> (M, Q)
        norm_log10 = np.reshape(norm_log10, shape)

        self._target_time = time
        self._norm = 10.0 ** norm_log10
        self._num_steps = num_steps
        self._sepa_init = sepa_init
        self._rchar = rchar
        self._gamma_inner = gamma_inner
        self._gamma_outer = gamma_outer

        return

    def __str__(self):
        msg = (
            f"{super().__str__()} :: "
            f"target_time/Gyr={self._target_time/GYR:.2e} num_steps={self._num_steps} "
            f"sepa_init/pc={self._sepa_init/PC:.2e} rchar/pc={self._rchar/PC:.2e} "
            f"gamma_inner={self._gamma_inner:.2e} gamma_outer={self._gamma_outer:.2e} "
        )
        return msg

    def dadt_dedt(self, evo, step, *args, **kwargs):
        raise NotImplementedError()

    def dadt(self, mtot, mrat, sepa, norm=None):
        import holodeck.sams.sam_cyutils   # noqa

        if norm is None:
            norm = self._norm

        args = np.broadcast_arrays(mtot, mrat, sepa, norm)
        shape = args[0].shape
        args = [aa.flatten() for aa in args]
        mtot, mrat, sepa, norm = args
        dadt_vals = holo.sams.sam_cyutils.hard_func_2pwl_gw(
            mtot, mrat, sepa, norm,                             # must all be 1darrays of matching size (X,)
            self._rchar, self._gamma_inner, self._gamma_outer   # must all be scalars
        )
        dadt_vals = dadt_vals.reshape(shape)
        return dadt_vals


# =================================================================================================
# ====    Utility Classes and Functions    ====
# =================================================================================================


class _Quinlan1996:
    """Hardening rates from stellar scattering parametrized as in [Quinlan1996]_.

    Fits from scattering experiments must be provided as `hparam` and `kparam`.

    """

    @staticmethod
    def dadt(sepa, rho, sigma, hparam):
        """Binary hardening rate from stellar scattering.

        [Sesana2010]_ Eq.8

        Parameters
        ----------
        sepa : (N,) array-like of scalar
            Binary separation in units of [cm].
        rho : (N,) array-like of scalar
            Effective stellar-density at binary separation in units of [g/cm^3].
        sigma : (N,) array-like of scalar
            Stellar velocity-dispersion at binary separation in units of [cm/s].
        hparam : (N,) array-like of scalar
            Binary hardening efficiency parameter "H" (unitless).

        Returns
        -------
        rv : (N,) np.ndarray of scalar
            Binary hardening rate in units of [cm/s].

        """
        rv = - (sepa ** 2) * NWTG * rho * hparam / sigma
        return rv

    @staticmethod
    def dedt(sepa, rho, sigma, hparam, kparam):
        """Binary rate-of-change of eccentricity from stellar scattering.

        [Sesana2010]_ Eq.9

        Parameters
        ----------
        sepa : (N,) array-like of scalar
            Binary separation in units of [cm].
        rho : (N,) array-like of scalar
            Effective stellar-density at binary separation in units of [g/cm^3].
        sigma : (N,) array-like of scalar
            Stellar velocity-dispersion at binary separation in units of [cm/s].
        hparam : (N,) array-like of scalar
            Binary hardening efficiency parameter "H" (unitless).
        kparam : (N,) array-like of scalar
            Binary eccentricity-change efficiency parameter "K" (unitless).

        Returns
        -------
        rv : (N,) np.ndarray of scalar
            Change of eccentricity rate in units of [1/s].

        """
        rv = sepa * NWTG * rho * hparam * kparam / sigma
        return rv

    @staticmethod
    def radius_hardening(msec, sigma):
        """
        [Sesana2010]_ Eq. 10
        """
        rv = NWTG * msec / (4 * sigma**2)
        return rv


class _SHM06:
    """Fits to stellar-scattering hardening rates from [Sesana2006]_, based on the [Quinlan1996]_ formalism.

    Parameters describe the efficiency of hardening as a function of mass-ratio (`mrat`) and separation (`sepa`).

    """

    def __init__(self):
        self._bound_H = [0.0, 40.0]    # See [Sesana2006]_ Fig.3
        self._bound_K = [0.0, 0.4]     # See [Sesana2006]_ Fig.4

        # Get the data filename
        fname = os.path.join(_PATH_DATA, _SCATTERING_DATA_FILENAME)
        if not os.path.isfile(fname):
            err = f"file ({fname}) not does exist!"
            log.error(err)
            raise FileNotFoundError(err)

        # Load Data
        data = json.load(open(fname, 'r'))
        self._data = data['SHM06']
        # 'H' : Hardening Rate
        self._init_h()
        # 'K' : Eccentricity growth
        self._init_k()
        return

    def H(self, mrat, sepa_rhard):
        """Hardening rate efficiency parameter.

        Parameters
        ----------
        mrat : (N,) array-like of scalar
            Binary mass-ratio (q = M2/M1 <= 1.0).
        sepa_rhard : (N,) array-like of scalar
            Binary separation in *units of hardening radius (r_h)*.

        Returns
        -------
        hh : (N,) np.ndarray of scalar
            Hardening parameter.

        """
        xx = sepa_rhard / self._H_a0(mrat)
        hh = self._H_A(mrat) * np.power(1.0 + xx, self._H_g(mrat))
        hh = np.clip(hh, *self._bound_H)
        return hh

    def K(self, mrat, sepa_rhard, ecc):
        """Eccentricity hardening rate efficiency parameter.

        Parameters
        ----------
        mrat : (N,) array-like of scalar
            Binary mass-ratio (q = M2/M1 <= 1.0).
        sepa_rhard : (N,) array-like of scalar
            Binary separation in *units of hardening radius (r_h)*.
        ecc : (N,) array-like of scalar
            Binary eccentricity.

        Returns
        -------
        kk : (N,) np.ndarray of scalar
            Eccentricity change parameter.

        """
        use_a = (sepa_rhard / self._K_a0(mrat, ecc))
        A = self._K_A(mrat, ecc)
        g = self._K_g(mrat, ecc)
        B = self._K_B(mrat, ecc)

        kk = A * np.power((1 + use_a), g) + B
        kk = np.clip(kk, *self._bound_K)
        return kk

    def _init_k(self):
        """Initialize and store the interpolants for calculating the K parameter.
        """
        data = self._data['K']
        # Get all of the mass ratios (ignore other keys)
        _kq_keys = list(data.keys())
        """ Need to reverse _kq_keys into descending order for later interpolation """
        _kq_keys.reverse()
        kq_keys = []
        for kq in _kq_keys:
            try:
                int(kq)
                kq_keys.append(kq)
            except (TypeError, ValueError):
                pass

        nq = len(kq_keys)
        if nq < 2:
            raise ValueError("Something is wrong... `kq_keys` = '{}'\ndata:\n{}".format(kq_keys, data))
        k_mass_ratios = 1.0/np.array([int(kq) for kq in kq_keys])
        k_eccen = np.array(data[kq_keys[0]]['e'])
        ne = len(k_eccen)
        k_A = np.zeros((ne, nq))
        k_a0 = np.zeros((ne, nq))
        k_g = np.zeros((ne, nq))
        k_B = np.zeros((ne, nq))

        for ii, kq in enumerate(kq_keys):
            _dat = data[kq]
            k_A[:, ii] = _dat['A']
            k_a0[:, ii] = _dat['a0']
            k_g[:, ii] = _dat['g']
            k_B[:, ii] = _dat['B']

        """
            Interpolate using RectBivariateSpline
            the interpolation functions below are assigned
            RectBivariateSpline().ev
            to evaluate the interpolation at individual points,
            allowing q_b and e_b in future calls of the function
            to be in non-ascending order
        """
        self._K_A = RectBivariateSpline(k_mass_ratios, k_eccen, np.array(k_A).T, kx=1, ky=1).ev
        self._K_a0 = RectBivariateSpline(k_mass_ratios, k_eccen, np.array(k_a0).T, kx=1, ky=1).ev
        self._K_g = RectBivariateSpline(k_mass_ratios, k_eccen, np.array(k_g).T, kx=1, ky=1).ev
        self._K_B = RectBivariateSpline(k_mass_ratios, k_eccen, np.array(k_B).T, kx=1, ky=1).ev

        return

    def _init_h(self):
        """Initialize and store the interpolants for calculating the H parameter.
        """
        _dat = self._data['H']
        h_mass_ratios = 1.0/np.array(_dat['q'])
        h_A = np.array(_dat['A'])
        h_a0 = np.array(_dat['a0'])
        h_g = np.array(_dat['g'])

        self._H_A = sp.interpolate.interp1d(h_mass_ratios, h_A, kind='linear', fill_value='extrapolate')
        self._H_a0 = sp.interpolate.interp1d(h_mass_ratios, h_a0, kind='linear', fill_value='extrapolate')
        self._H_g = sp.interpolate.interp1d(h_mass_ratios, h_g, kind='linear', fill_value='extrapolate')
        return


class _Siwek2023:
    r"""Hardening rates from circumbinary disk simulations as in [Siwek2023]_.

    Mass ratios and eccentricities must be provided.

    The lookup tables (in form of a dictionary) are located here:
    data/cbd_torques/siwek+23/ebdot_abdot_tmin3000Pb_tmax10000Pb.pkl
    and contain $\dot{e}$ and $\dot{a}$ as a function of q,e

    """

    @staticmethod
    def dadt(mrat, eccen):
        """Binary hardening rate from circumbinary disk torques.

        [Siwek2023]_ Table 2

        Parameters
        ----------
        sepa : (N,) array-like of scalar
            Binary separation in units of [cm].

        Returns
        -------
        dadt : (N,) np.ndarray of scalar
            Binary hardening rate in units of [cm/s].

        """
        fp_dadt_dedt_pkl = 'cbd_torques/siwek+23/ebdot_abdot_tmin3000Pb_tmax10000Pb.pkl'
        fp_dadt_dedt_pkl = os.path.join(_PATH_DATA, fp_dadt_dedt_pkl)
        fp_mean_ebdot_abdot = open(fp_dadt_dedt_pkl, 'rb')
        mean_ebdot_abdot = pkl.load(fp_mean_ebdot_abdot)
        all_es = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]
        all_qs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        torque_contribution = 'sum_grav_acc'

        mean_abdot_arr = np.zeros((len(all_qs),len(all_es)))
        # mean_ebdot_arr = np.zeros((len(all_qs),len(all_es)))
        for i,q in enumerate(all_qs):
            for j,e in enumerate(all_es):
                this_key_ab = 'e=%.2f_q=%.2f_ab_dot_ab_%s' %(e,q,torque_contribution)
                mean_abdot_arr[i][j] = mean_ebdot_abdot[this_key_ab]

        dadt_qe_interp = RectBivariateSpline(np.array(all_qs), np.array(all_es), np.array(mean_abdot_arr), kx=1, ky=1)
        dadt = dadt_qe_interp.ev(mrat, eccen)
        return dadt

    @staticmethod
    def dedt(mrat, eccen):
        """Binary eccentricity rate of change from circumbinary disk torques.

        [Siwek2023]_ Table 3

        Parameters
        ----------
        sepa : (N,) array-like of scalar
            Binary separation in units of [cm].

        Returns
        -------
        dedt : (N,) np.ndarray of scalar
            Binary eccentricity rate of change in units of [cm/s].

        """

        fp_dadt_dedt_pkl = 'cbd_torques/siwek+23/ebdot_abdot_tmin3000Pb_tmax10000Pb.pkl'
        fp_dadt_dedt_pkl = os.path.join(_PATH_DATA, fp_dadt_dedt_pkl)
        fp_mean_ebdot_abdot = open(fp_dadt_dedt_pkl, 'rb')
        mean_ebdot_abdot = pkl.load(fp_mean_ebdot_abdot)
        all_es = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]
        all_qs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        torque_contribution = 'sum_grav_acc'

        mean_ebdot_arr = np.zeros((len(all_qs),len(all_es)))
        for i, q in enumerate(all_qs):
            for j, e in enumerate(all_es):
                this_key_eb = 'e=%.2f_q=%.2f_eb_dot_%s' %(e, q, torque_contribution)
                mean_ebdot_arr[i][j] = mean_ebdot_abdot[this_key_eb]

        dedt_qe_interp = RectBivariateSpline(np.array(all_qs), np.array(all_es), np.array(mean_ebdot_arr), kx=1, ky=1)
        dedt = dedt_qe_interp.ev(mrat, eccen)
        return dedt


def _radius_stellar_characteristic_dabringhausen_2008(mstar, gamma=1.0):
    """Characteristic stellar radius based on total stellar mass.

    [Chen2017]_ Eq.27 - from [Dabringhausen+2008]
    """
    rchar = 239 * PC * (np.power(2.0, 1.0/(3.0 - gamma)) - 1.0)
    rchar *= np.power(mstar / (1e9*MSOL), 0.596)
    return rchar


def _radius_influence_dehnen(mbh, mstar, gamma=1.0):
    """Characteristic radius of influence based on a Dehnen density profile.

    [Chen2017]_ Eq.25
    """
    rchar = _radius_stellar_characteristic_dabringhausen_2008(mstar, gamma)
    rinfl = np.power(2*mbh/mstar, 1.0/(gamma - 3.0))
    rinfl = rchar / (rinfl - 1.0)
    return rinfl


def _density_at_influence_radius_dehnen(mbh, mstar, gamma=1.0):
    """Density at the characteristic influence radius, based on a Dehnen density profile.
    [Chen2017]_ Eq.26
    """
    # [Chen2017] Eq.27 - from [Dabringhausen+2008]
    rchar = _radius_stellar_characteristic_dabringhausen_2008(mstar, gamma)
    dens = mstar * (3.0 - gamma) / np.power(rchar, 3.0) / (4.0 * np.pi)
    dens *= np.power(2*mbh / mstar, gamma / (gamma - 3.0))
    return dens


def _radius_hard_BBR1980_dehnen(mbh, mstar, gamma=1.0):
    """Characteristic 'hardened' radius from [BBR1980]_, assuming a Dehnen stellar density profile.

    [Kelley2017a]_ paragraph below Eq.8 - from [BBR1980]_
    """
    rbnd = _radius_influence_dehnen(mbh, mstar, gamma=gamma)
    rstar = _radius_stellar_characteristic_dabringhausen_2008(mstar, gamma)
    rhard = rstar * (rbnd/rstar) ** 3
    return rhard


def _radius_loss_cone_BBR1980_dehnen(mbh, mstar, gamma=1.0):
    """Characteristic 'loss-cone' radius from [BBR1980]_, assuming a Dehnen stellar density profile.

    [Kelley2017a]_ Eq.9 - from [BBR1980]_
    """
    mass_of_a_star = 0.6 * MSOL
    rbnd = _radius_influence_dehnen(mbh, mstar, gamma=gamma)
    rstar = _radius_stellar_characteristic_dabringhausen_2008(mstar, gamma)
    rlc = np.power(mass_of_a_star / mbh, 0.25) * np.power(rbnd/rstar, 2.25) * rstar
    return rlc
