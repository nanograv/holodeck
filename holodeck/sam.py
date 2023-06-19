r"""Semi Analytic Modeling (SAM) submodule.

The core element of the SAM module is the :class:`Semi_Analytic_Model` class.  This class requires four
components as arguments:

(1) Galaxy Stellar Mass Function (GSMF): gives the comoving number-density of galaxies as a function
    of stellar mass.  This is implemented as subclasses of the :class:`_Galaxy_Stellar_Mass_Function`
    base class.
(2) Galaxy Pair Fraction (GPF): gives the fraction of galaxies that are in a 'pair' with a given
    mass ratio (and typically a function of redshift and primary-galaxy mass).  Implemented as
    subclasses of the :class:`_Galaxy_Pair_Fraction` subclass.
(3) Galaxy Merger Time (GMT): gives the characteristic time duration for galaxy 'mergers' to occur.
    Implemented as subclasses of the :class:`_Galaxy_Merger_Time` subclass.
(4) M_bh - M_bulge Relation (mmbulge): gives MBH properties for a given galaxy stellar-bulge mass.
    Implemented as subcalsses of the :class:`holodeck.relations._MMBulge_Relation` subclass.

The :class:`Semi_Analytic_Model` class defines a grid in parameter space of total MBH mass ($M=M_1 + M_2$),
MBH mass ratio ($q \\equiv M_1/M_2$), redshift ($z$), and at times binary separation
(semi-major axis $a$) or binary rest-frame orbital-frequency ($f_r$).  Over this grid, the distribution of
comoving number-density of MBH binaries in the Universe is calculated.  Methods are also provided
that interface with the `kalepy` package to draw 'samples' (discretized binaries) from the
distribution, and to calculate GW signatures.

The step of going from a number-density of binaries in $(M, q, z)$ space, to also the distribution
in $a$ or $f$ is subtle, as it requires modeling the binary evolution (i.e. hardening rate).


To-Do
-----
* Allow SAM class to take M-sigma in addition to M-Mbulge.

References
----------
* [Sesana2008]_ Sesana, Vecchio, Colacino 2008.
* [Chen2019]_ Chen, Sesana, Conselice 2019.

"""

import abc
from datetime import datetime

import numpy as np
import scipy as sp
import scipy.interpolate  # noqa

import kalepy as kale

import holodeck as holo
from holodeck import cosmo, utils
from holodeck.constants import GYR, SPLC, MSOL, MPC
from holodeck import relations, gravwaves, single_sources
import holodeck.sam_cython

REDZ_SAMPLE_VOLUME = True    #: get redshifts by sampling uniformly in 3D spatial volume, and converting

GSMF_USES_MTOT = False       #: the mass used in the GSMF is interpretted as M=m1+m2, otherwise use primary m1
GPF_USES_MTOT = False        #: the mass used in the GPF  is interpretted as M=m1+m2, otherwise use primary m1
GMT_USES_MTOT = False        #: the mass used in the GMT  is interpretted as M=m1+m2, otherwise use primary m1



# ==============================
# ====    SAM Components    ====
# ==============================


# ----    Galaxy Stellar-Mass Function    ----


class _Galaxy_Stellar_Mass_Function(abc.ABC):
    """Galaxy Stellar-Mass Function base-class.  Used to calculate number-density of galaxies.


    """

    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        return

    @abc.abstractmethod
    def __call__(self, mstar, redz):
        """Return the number-density of galaxies at a given stellar mass, per log10 interval of stellar-mass.

        i.e. Phi = dn / dlog10(M)

        Parameters
        ----------
        mstar : scalar or ndarray
            Galaxy stellar-mass in units of [grams]
        redz : scalar or ndarray
            Redshift.

        Returns
        -------
        rv : scalar or ndarray
            Number-density of galaxies in units of [Mpc^-3]

        """
        return

    def mbh_mass_func(self, mbh, redz, mmbulge, scatter=None):
        """Convert from the GSMF to a MBH mass function (number density), using a given Mbh-Mbulge relation.

        Parameters
        ----------
        mbh : array_like
            Blackhole masses at which to evaluate the mass function.
        redz : array_like
            Redshift(s) at which to evaluate the mass function.
        mmbulge : `relations._MMBulge_Relation` subclass instance
            Scaling relation between galaxy and MBH masses.
        scatter : None, bool, or float
            Introduce scatter in masses.
            * `None` or `True` : use the value from `mmbulge._scatter_dex`
            * `False` : do not introduce scatter
            * float : introduce scatter with this amplitude (in dex)

        Returns
        -------
        ndens : array_like
            Number density of MBHs, in units of [Mpc^-3]

        """
        if scatter in [None, True]:
            scatter = mmbulge._scatter_dex

        mstar = mmbulge.mstar_from_mbh(mbh, scatter=False)
        # This is `dn_star / dlog10(M_star)`
        ndens = self(mstar, redz)    # units of  [1/Mpc^3]

        # dM_star / dM_bh
        dmstar_dmbh = mmbulge.dmstar_dmbh(mstar)   # [unitless]
        # convert to dlog10(M_star) / dlog10(M_bh) = (M_bh / M_star) * (dM_star / dM_bh)
        jac = (mbh/mstar) * dmstar_dmbh
        # convert galaxy number density to  to dn_bh / dlog10(M_bh)
        ndens *= jac

        if scatter is not False:
            ndens = holo.utils.scatter_redistribute_densities(mbh, ndens, scatter=scatter)

        return ndens



class GSMF_Schechter(_Galaxy_Stellar_Mass_Function):
    r"""Single Schechter Function - Galaxy Stellar Mass Function.

    This is density per unit log10-interval of stellar mass, i.e. $Phi = dn / d\\log_{10}(M)$

    See: [Chen2019]_ Eq.9 and enclosing section.

    """

    def __init__(self, phi0=-2.77, phiz=-0.27, mchar0_log10=11.24, mchar0=None, mcharz=0.0, alpha0=-1.24, alphaz=-0.03):
        mchar0, _ = utils._parse_val_log10_val_pars(
            mchar0, mchar0_log10, val_units=MSOL, name='mchar0', only_one=True
        )

        self._phi0 = phi0         # - 2.77  +/- [-0.29, +0.27]  [log10(1/Mpc^3)]
        self._phiz = phiz         # - 0.27  +/- [-0.21, +0.23]  [log10(1/Mpc^3)]
        self._mchar0 = mchar0       # 10^ (+11.24  +/- [-0.17, +0.20]  [log10(Msol)])
        self._mcharz = mcharz       #  0.0                        [log10(Msol)]    # noqa
        self._alpha0 = alpha0     # -1.24   +/- [-0.16, +0.16]
        self._alphaz = alphaz     # -0.03   +/- [-0.14, +0.16]
        return

    def __call__(self, mstar, redz):
        r"""Return the number-density of galaxies at a given stellar mass.

        See: [Chen2019] Eq.8

        Parameters
        ----------
        mstar : scalar or ndarray
            Galaxy stellar-mass in units of [grams]
        redz : scalar or ndarray
            Redshift.

        Returns
        -------
        rv : scalar or ndarray
            Number-density of galaxies per log-interval of mass in units of [Mpc^-3]
            i.e.  ``Phi = dn / d\\log_{10}(M)``

        """
        phi = self._phi_func(redz)
        mchar = self._mchar_func(redz)
        alpha = self._alpha_func(redz)
        xx = mstar / mchar
        # [Chen2019]_ Eq.8
        rv = np.log(10.0) * phi * np.power(xx, 1.0 + alpha) * np.exp(-xx)
        return rv

    def _phi_func(self, redz):
        """See: [Chen2019]_ Eq.9
        """
        return np.power(10.0, self._phi0 + self._phiz * redz)

    def _mchar_func(self, redz):
        """See: [Chen2019]_ Eq.10 - NOTE: added `redz` term
        """
        return self._mchar0 + self._mcharz * redz

    def _alpha_func(self, redz):
        """See: [Chen2019]_ Eq.11
        """
        return self._alpha0 + self._alphaz * redz


# ----    Galaxy Pair Fraction    ----


class _Galaxy_Pair_Fraction(abc.ABC):
    """Galaxy Pair Fraction base class, used to describe the fraction of galaxies in mergers/pairs.
    """

    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        return

    @abc.abstractmethod
    def __call__(self, mass, mrat, redz):
        """Return the fraction of galaxies in pairs of the given parameters.

        Parameters
        ----------
        mass : array_like,
            Mass of the system, units of [grams].
            NOTE: the definition of mass is ambiguous, i.e. whether it is the primary mass, or the
            combined system mass.
        mrat : array_like,
            Mass-ratio of the system (m2/m1 <= 1.0), dimensionless.
        redz : array_like,
            Redshift.

        Returns
        -------
        rv : scalar or ndarray,
            Galaxy pair fraction, dimensionless.

        """
        return


class GPF_Power_Law(_Galaxy_Pair_Fraction):
    """Galaxy Pair Fraction - Single Power-Law
    """

    def __init__(self, frac_norm_allq=0.025, frac_norm=None, mref=None, mref_log10=11.0,
                 malpha=0.0, zbeta=0.8, qgamma=0.0, obs_conv_qlo=0.25, max_frac=1.0):

        mref, _ = utils._parse_val_log10_val_pars(
            mref, mref_log10, val_units=MSOL, name='mref', only_one=True
        )

        # If the pair-fraction integrated over all mass-ratios is given (f0), convert to regular (f0-prime)
        if frac_norm is None:
            if frac_norm_allq is None:
                raise ValueError("If `frac_norm` is not given, `frac_norm_allq` is requried!")
            pow = qgamma + 1.0
            qlo = obs_conv_qlo
            qhi = 1.00
            pair_norm = (qhi**pow - qlo**pow) / pow
            frac_norm = frac_norm_allq / pair_norm

        # normalization corresponds to f0-prime in [Chen2019]_
        self._frac_norm = frac_norm   # f0 = 0.025 b/t [+0.02, +0.03]  [+0.01, +0.05]
        self._malpha = malpha         #      0.0   b/t [-0.2 , +0.2 ]  [-0.5 , +0.5 ]  # noqa
        self._zbeta = zbeta           #      0.8   b/t [+0.6 , +0.1 ]  [+0.0 , +2.0 ]  # noqa
        self._qgamma = qgamma         #      0.0   b/t [-0.2 , +0.2 ]  [-0.2 , +0.2 ]  # noqa

        if (max_frac < 0.0) or (1.0 < max_frac):
            err = f"Given `max_frac`={max_frac:.4f} must be between [0.0, 1.0]!"
            holo.log.exception(err)
            raise ValueError(err)
        self._max_frac = max_frac

        self._mref = mref   # NOTE: this is `a * M_0 = 1e11 Msol` in papers
        return

    def __call__(self, mass, mrat, redz):
        """Return the fraction of galaxies in pairs of the given parameters.

        Parameters
        ----------
        mass : array_like,
            Mass of the system, units of [grams].
            NOTE: the definition of mass is ambiguous, i.e. whether it is the primary mass, or the
            combined system mass.
        mrat : array_like,
            Mass-ratio of the system (m2/m1 <= 1.0), dimensionless.
        redz : array_like,
            Redshift.

        Returns
        -------
        rv : scalar or ndarray,
            Galaxy pair fraction, dimensionless.

        """
        f0p = self._frac_norm
        am0 = self._mref
        aa = self._malpha
        bb = self._zbeta
        gg = self._qgamma
        rv = f0p * np.power(mass/am0, aa) * np.power(1.0 + redz, bb) * np.power(mrat, gg)
        rv = np.clip(rv, None, self._max_frac)
        return rv


# ----    Galaxy Merger Time    ----


class _Galaxy_Merger_Time(abc.ABC):
    """Galaxy Merger Time base class, used to model merger timescale of galaxy pairs.
    """

    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        return

    @abc.abstractmethod
    def __call__(self, mass, mrat, redz):
        """Return the galaxy merger time for the given parameters.

        Parameters
        ----------
        mass : (N,) array_like[scalar]
            Mass of the system, units of [grams].
            NOTE: the definition of mass is ambiguous, i.e. whether it is the primary mass, or the
            combined system mass.
        mrat : scalar or ndarray,
            Mass-ratio of the system (m2/m1 <= 1.0), dimensionless.
        redz : scalar or ndarray,
            Redshift.

        Returns
        -------
        rv : scalar or ndarray,
            Galaxy merger time, in units of [sec].

        """
        return

    def zprime(self, mass, mrat, redz, **kwargs):
        """Return the redshift after merger (i.e. input `redz` delayed by merger time).
        """
        tau0 = self(mass, mrat, redz, **kwargs)  # sec
        # Find the redshift of  t(z) + tau
        redz_prime = utils.redz_after(tau0, redz=redz)
        return redz_prime, tau0


class GMT_Power_Law(_Galaxy_Merger_Time):
    """Galaxy Merger Time - simple power law prescription
    """

    def __init__(self, time_norm=0.55*GYR, mref0=1.0e11*MSOL, malpha=0.0, zbeta=-0.5, qgamma=0.0):
        # tau0  [sec]
        self._time_norm = time_norm   # +0.55  b/t [+0.1, +2.0]  [+0.1, +10.0]  values for [Gyr]
        self._malpha = malpha         # +0.0   b/t [-0.2, +0.2]  [-0.2, +0.2 ]
        self._zbeta = zbeta           # -0.5   b/t [-2.0, +1.0]  [-3.0, +1.0 ]
        self._qgamma = qgamma         # +0.0   b/t [-0.2, +0.2]  [-0.2, +0.2 ]

        # [Msol]  NOTE: this is `b * M_0 = 0.4e11 Msol / h0` in [Chen2019]_
        # 7.2e10*MSOL
        mref = mref0 * (0.4 / cosmo.h)
        self._mref = mref
        return

    def __call__(self, mass, mrat, redz):
        """Return the galaxy merger time for the given parameters.

        Parameters
        ----------
        mass : (N,) array_like[scalar]
            Mass of the system, units of [grams].
            NOTE: the definition of mass is ambiguous, i.e. whether it is the primary mass, or the
            combined system mass.
        mrat : (N,) array_like[scalar]
            Mass ratio of each binary.
        redz : (N,) array_like[scalar]
            Redshifts of each binary.

        Returns
        -------
        mtime : (N,) ndarray[float]
            Merger time for each binary in [sec].

        """
        # convert to primary mass
        # mpri = utils.m1m2_from_mtmr(mtot, mrat)[0]   # [grams]
        tau0 = self._time_norm                       # [sec]
        bm0 = self._mref                             # [grams]
        aa = self._malpha
        bb = self._zbeta
        gg = self._qgamma
        mtime = tau0 * np.power(mass/bm0, aa) * np.power(1.0 + redz, bb) * np.power(mrat, gg)
        mtime = mtime
        return mtime


# ===================================
# ====    Semi-Analytic Model    ====
# ===================================


class Semi_Analytic_Model:
    """Semi-Analytic Model of MBH Binary populations.

    Based on four components:
    * Galaxy Stellar-Mass Function (GSMF): the distribution of galaxy masses
    * Galaxy Pair Fraction (GPF): the probability of galaxies having a companion
    * Galaxy Merger Time (GMT): the expected galaxy-merger timescale for a pair of galaxies
    * M-MBulge relation: relation between host-galaxy (bulge-mass) and MBH (mass) properties

    """

    def __init__(
        self, mtot=(1.0e4*MSOL, 1.0e12*MSOL, 91), mrat=(1e-3, 1.0, 81), redz=(1e-3, 10.0, 101),
        shape=None, log=None,
        gsmf=GSMF_Schechter, gpf=GPF_Power_Law, gmt=GMT_Power_Law, mmbulge=relations.MMBulge_MM2013,
        **kwargs
    ):
        """Construct a new Semi_Analytic_Model instance.

        Parameters
        ----------
        mtot : list, optional
        mrat : list, optional
        redz : list, optional
        shape : _type_, optional
        gsmf : _type_, optional
        gpf : _type_, optional
        gmt : _type_, optional
        mmbulge : _type_, optional

        """
        if log is None:
            log = holo.log
        self._log = log

        deprecated_keys = ['ZERO_DYNAMIC_STALLED_SYSTEMS', 'ZERO_GMT_STALLED_SYSTEMS']
        for key, val in kwargs.items():
            if key in deprecated_keys:
                log.error(f"Using deprecated kwarg: {key}: {val}!  In the future this will raise an error.")
            else:
                err = f"Unexpected kwarg {key=}: {val=}!"
                log.exception(err)
                raise ValueError(err)

        # ---- Process SAM components

        gsmf = utils._get_subclass_instance(gsmf, None, _Galaxy_Stellar_Mass_Function)
        gpf = utils._get_subclass_instance(gpf, None, _Galaxy_Pair_Fraction)
        gmt = utils._get_subclass_instance(gmt, None, _Galaxy_Merger_Time)
        mmbulge = utils._get_subclass_instance(mmbulge, None, relations._MMBulge_Relation)
        self._gsmf = gsmf             #: Galaxy Stellar-Mass Function (`_Galaxy_Stellar_Mass_Function` instance)
        self._gpf = gpf               #: Galaxy Pair Fraction (`_Galaxy_Pair_Fraction` instance)
        self._gmt = gmt               #: Galaxy Merger Time (`_Galaxy_Merger_Time` instance)
        self._mmbulge = mmbulge       #: Mbh-Mbulge relation (`relations._MMBulge_Relation` instance)

        # ---- Create SAM grid edges

        if shape is not None:
            if np.isscalar(shape):
                shape = [shape for ii in range(3)]

        params = [mtot, mrat, redz]
        param_names = ['mtot', 'mrat', 'redz']
        for ii, (par, name) in enumerate(zip(params, param_names)):
            if not isinstance(par, tuple) and (len(par) == 3):
                err = (
                    f"{name} (type={type(par)}, len={len(par)}) must be a (3,) tuple specifying a log-spacing, "
                    "or ndarray of grid edges!"
                )
                log.exception(err)
                raise ValueError(err)

            par = [pp for pp in par]
            if shape is not None:
                if shape[ii] is not None:
                    par[2] = shape[ii]
            params[ii] = np.logspace(*np.log10(par[:2]), par[2])
            log.debug(f"{name}: [{params[ii][0]}, {params[ii][-1]}] {params[ii].size}")

        mtot, mrat, redz = params
        self.mtot = mtot
        self.mrat = mrat
        self.redz = redz

        # ---- Set other parameters

        # These values are calculated as needed by the class when the corresponding methods are called
        self._density = None          #: Binary comoving number-density
        self._shape = None            #: Shape of the parameter-space domain (mtot, mrat, redz)
        self._gmt_time = None         #: GMT timescale of galaxy mergers [sec]
        self._redz_prime = None       #: redshift following galaxy merger process

        return

    @property
    def edges(self):
        """The grid edges defining the domain (list of: [`mtot`, `mrat`, `redz`])
        """
        return [self.mtot, self.mrat, self.redz]

    @property
    def shape(self):
        """Shape of the parameter space domain (number of edges in each dimension), (3,) tuple
        """
        if self._shape is None:
            self._shape = tuple([len(ee) for ee in self.edges])
        return self._shape

    def mass_stellar(self):
        """Calculate stellar masses for each MBH based on the M-MBulge relation.

        Returns
        -------
        masses : (2, N) ndarray of scalar,
            Galaxy total stellar masses for all MBH. [0, :] is primary, [1, :] is secondary [grams].

        """
        # total-mass, mass-ratio ==> (M1, M2)
        masses = utils.m1m2_from_mtmr(self.mtot[:, np.newaxis], self.mrat[np.newaxis, :])
        # BH-masses to stellar-masses
        masses = self._mmbulge.mstar_from_mbh(masses, scatter=False)
        return masses

    @property
    def static_binary_density(self):
        """The number-density of binaries in each bin, 'd^3 n / [dlog10M dq dz]' in units of [Mpc^-3].

        This is calculated once and cached.

        Returns
        -------
        density : (M, Q, Z) ndarray
            Number density of binaries, per unit redshift, mass-ratio, and log10 of mass.  Units of [Mpc^-3].

        Notes
        -----
        * This function effectively calculates Eq.21 & 5 of [Chen2019]_; or equivalently, Eq. 6 of [Sesana2008]_.
        * Bins which 'merge' after redshift zero are set to zero density (using the `self._gmt` instance).

        """
        if self._density is None:
            log = self._log

            # ---- convert from MBH ===> mstar

            # `mstar_tot` starts as the secondary mass, sorry
            mstar_pri, mstar_tot = self.mass_stellar()
            # q = m2 / m1
            mstar_rat = mstar_tot / mstar_pri
            # M = m1 + m2
            mstar_tot = mstar_pri + mstar_tot
            redz = self.redz[np.newaxis, np.newaxis, :]
            args = [mstar_pri[..., np.newaxis], mstar_rat[..., np.newaxis], mstar_tot[..., np.newaxis], redz]
            # Convert to shape (M, Q, Z)
            mstar_pri, mstar_rat, mstar_tot, redz = np.broadcast_arrays(*args)

            # choose whether the primary mass, or total mass, is used in different calculations
            mass_gsmf = mstar_tot if GSMF_USES_MTOT else mstar_pri
            mass_gpf = mstar_tot if GPF_USES_MTOT else mstar_pri
            mass_gmt = mstar_tot if GMT_USES_MTOT else mstar_pri

            # GMT returns `-1.0` for values beyond age of universe
            zprime, gmt_time = self._gmt.zprime(mass_gmt, mstar_rat, redz)
            self._gmt_time = gmt_time
            self._redz_prime = zprime

            # find valid entries (M, Q, Z)
            idx_stalled = (zprime < 0.0)
            log.info(f"Stalled SAM bins based on GMT: {utils.frac_str(idx_stalled)}")

            # ---- Get Galaxy Merger Rate  [Chen2019] Eq.5
            log.debug(f"GSMF_USES_MTOT={GSMF_USES_MTOT}")
            log.debug(f"GPF_USES_MTOT ={GPF_USES_MTOT}")
            log.debug(f"GMT_USES_MTOT ={GMT_USES_MTOT}")

            # `gsmf` returns [1/Mpc^3]   `dtdz` returns [sec]
            dens = self._gsmf(mass_gsmf, redz) * self._gpf(mass_gpf, mstar_rat, redz) * cosmo.dtdz(redz)
            # `gmt` returns [sec]
            dens /= gmt_time
            # now `dens` is  ``dn_gal / [dlog10(Mstar) dq_gal dz]``  with units of [Mpc^-3]

            # ---- Convert to MBH Binary density

            # we want ``dn_mbhb / [dlog10(M_bh) dq_bh qz]``
            # so far we have ``dn_gal / [dlog10(M_gal) dq_gal dz]``

            # dn / [dM dq dz] = (dn_gal / [dM_gal dq_gal dz]) * (dM_gal/dM_bh) * (dq_gal / dq_bh)
            mplaw = self._mmbulge._mplaw
            dqbh_dqgal = mplaw * np.power(mstar_rat, mplaw - 1.0)
            # (dMstar-pri / dMbh-pri) * (dMbh-pri/dMbh-tot) = (dMstar-pri / dMstar-tot) * (dMstar-tot/dMbh-tot)
            # ==> (dMstar-tot/dMbh-tot) = (dMstar-pri / dMbh-pri) * (dMbh-pri/dMbh-tot) / (dMstar-pri / dMstar-tot)
            #                           = (dMstar-pri / dMbh-pri) * (1 / (1+q_bh)) / (1 / (1+q_star))
            #                           = (dMstar-pri / dMbh-pri) * ((1+q_star) / (1+q_bh))
            dmstar_dmbh_pri = self._mmbulge.dmstar_dmbh(mstar_pri)   # [unitless]
            qterm = (1.0 + mstar_rat) / (1.0 + self.mrat[np.newaxis, :, np.newaxis])
            dmstar_dmbh = dmstar_dmbh_pri * qterm

            dens *= (self.mtot[:, np.newaxis, np.newaxis] / mstar_tot) * (dmstar_dmbh / dqbh_dqgal)

            # Add scatter from the M-Mbulge relation
            scatter = self._mmbulge._scatter_dex
            log.debug(f"mmbulge scatter = {scatter}")
            if scatter > 0.0:
                log.info(f"Adding MMbulge scatter ({scatter:.4e})")
                log.info(f"\tdens bef: ({utils.stats(dens)})")
                dur = datetime.now()
                mass_bef = self._integrated_binary_density(dens, sum=True)
                self._dens_bef = np.copy(dens)
                dens = add_scatter_to_masses(self.mtot, self.mrat, dens, scatter, log=log)
                self._dens_aft = np.copy(dens)

                mass_aft = self._integrated_binary_density(dens, sum=True)
                dur = datetime.now() - dur
                dm = (mass_aft - mass_bef) / mass_bef
                log.info(f"Scatter added after {dur.total_seconds()} sec")
                log.info(f"\tdens aft: ({utils.stats(dens)})")
                msg = f"mass: {mass_bef:.2e} ==> {mass_aft:.2e} || change = {dm:.4e}"
                log.info(f"\t{msg}")
                if np.fabs(dm) > 0.2:
                    err = f"Warning, significant change in number-mass!  {msg}"
                    log.error(err)

            # set values after redshift zero to have zero density
            log.info(f"zeroing out {utils.frac_str(idx_stalled)} systems stalled from GMT")
            dens[idx_stalled] = 0.0
            # else:
            #     log.info("NOT zeroing out systems with GMTs extending past redshift zero!")

            self._density = dens

        return self._density

    @utils.deprecated_fail("`dynamic_binary_number_at_fobs` or `sam_cython.dynamic_binary_number_at_fobs`")
    def dynamic_binary_number(self, *args, **kwargs):
        pass

    def dynamic_binary_number_at_fobs(self, hard, fobs_orb, **kwargs):

        if hard.CONSISTENT:
            edges, dnum, redz_final = self._dynamic_binary_number_at_fobs_consistent(hard, fobs_orb, **kwargs)
        else:
            edges, dnum, redz_final = self._dynamic_binary_number_at_fobs_inconsistent(hard, fobs_orb, **kwargs)

        return edges, dnum, redz_final

    def _dynamic_binary_number_at_fobs_consistent(self, hard, fobs_orb, steps=200, details=False):
        """Get correct redshifts for full binary-number calculation.

        Slower but more correct than old `dynamic_binary_number`.
        Same as new cython implementation `holo.sam_cython.dynamic_binary_number_at_fobs`, which is
        more than 10x faster.
        LZK 2023-05-11

        # BUG doesn't work for Fixed_Time_2PL

        """
        fobs_orb = np.asarray(fobs_orb)
        edges = self.edges + [fobs_orb, ]

        # shape: (M, Q, Z)
        dens = self.static_binary_density   # d3n/[dlog10(M) dq dz]  units: [Mpc^-3]


        # start from the hardening model's initial separation
        rmax = hard._sepa_init
        # (M,) end at the ISCO
        rmin = utils.rad_isco(self.mtot)
        # Choose steps for each binary, log-spaced between rmin and rmax
        extr = np.log10([rmax * np.ones_like(rmin), rmin])     # (2,M,)
        rads = np.linspace(0.0, 1.0, steps)[np.newaxis, :]     # (1,X)
        # (M, S)  =  (M,1) * (1,S)
        rads = extr[0][:, np.newaxis] + (extr[1] - extr[0])[:, np.newaxis] * rads
        rads = 10.0 ** rads

        # (M, Q, S)
        mt, mr, rads, norm = np.broadcast_arrays(
            self.mtot[:, np.newaxis, np.newaxis],
            self.mrat[np.newaxis, :, np.newaxis],
            rads[:, np.newaxis, :],
            hard._norm[:, :, np.newaxis],
        )
        dadt_evo = hard.dadt(mt, mr, rads, norm=norm)

        # (M, Q, S-1)
        # Integrate (inverse) hardening rates to calculate total lifetime to each separation
        times_evo = -utils.trapz_loglog(-1.0 / dadt_evo, rads, axis=-1, cumsum=True)
        # Combine the binary-evolution time, with the galaxy-merger time
        # (M, Q, Z, S-1)
        rz = self.redz[np.newaxis, np.newaxis, :, np.newaxis]
        times_tot = times_evo[:, :, np.newaxis, :] + self._gmt_time[:, :, :, np.newaxis]
        redz_evo = utils.redz_after(times_tot, redz=rz)

        # convert from separations to rest-frame orbital frequencies
        # (M, Q, S)
        frst_orb_evo = utils.kepler_freq_from_sepa(mt, rads)
        # (M, Q, Z, S)
        fobs_orb_evo = frst_orb_evo[:, :, np.newaxis, :] / (1.0 + rz)

        # ---- interpolate to target frequencies
        # `ndinterp` interpolates over 1th dimension

        # (M, Q, Z, S-1)  ==>  (M*Q*Z, S-1)
        fobs_orb_evo, redz_evo = [mm.reshape(-1, steps-1) for mm in [fobs_orb_evo[:, :, :, 1:], redz_evo]]
        # (M*Q*Z, X)
        redz_final = utils.ndinterp(fobs_orb, fobs_orb_evo, redz_evo, xlog=True, ylog=False)

        # (M*Q*Z, X) ===> (M, Q, Z, X)
        redz_final = redz_final.reshape(self.shape + (fobs_orb.size,))
        coal = (redz_final > 0.0)
        frst_orb = fobs_orb * (1.0 + redz_final)
        frst_orb[frst_orb < 0.0] = 0.0
        redz_final[~coal] = -1.0

        # (M, Q, Z, X) comoving-distance in [Mpc]
        dc = np.zeros_like(redz_final)
        dc[coal] = cosmo.comoving_distance(redz_final[coal]).to('Mpc').value

        # (M, Q, Z, X) this is `(dVc/dz) * (dz/dt)` in units of [Mpc^3/s]
        cosmo_fact = np.zeros_like(redz_final)
        cosmo_fact[coal] = 4 * np.pi * (SPLC/MPC) * np.square(dc[coal]) * (1.0 + redz_final[coal])

        # (M, Q) calculate chirp-mass
        mt = self.mtot[:, np.newaxis, np.newaxis, np.newaxis]
        mr = self.mrat[np.newaxis, :, np.newaxis, np.newaxis]

        # Convert from observer-frame orbital freq, to rest-frame orbital freq
        sa = utils.kepler_sepa_from_freq(mt, frst_orb)
        mt, mr, sa, norm = np.broadcast_arrays(mt, mr, sa, hard._norm[:, :, np.newaxis, np.newaxis])
        # hardening rate, negative values, units of [cm/sec]
        dadt = hard.dadt(mt, mr, sa, norm=norm)
        # Calculate `tau = dt/dlnf_r = f_r / (df_r/dt)`
        # dfdt is positive (increasing frequency)
        dfdt, frst_orb = utils.dfdt_from_dadt(dadt, sa, frst_orb=frst_orb)
        tau = frst_orb / dfdt

        # (M, Q, Z, X) units: [1/s] i.e. number per second
        dnum = dens[..., np.newaxis] * cosmo_fact * tau
        dnum[~coal] = 0.0

        if details:
            tau[~coal] = 0.0
            dadt[~coal] = 0.0
            sa[~coal] = 0.0
            cosmo_fact[~coal] = 0.0
            # (M, Q, X)  ==>  (M, Q, Z, X)
            dets = dict(tau=tau, cosmo_fact=cosmo_fact, dadt=dadt, fobs=fobs_orb, sepa=sa)
            return edges, dnum, redz_final, dets

        return edges, dnum, redz_final

    def _dynamic_binary_number_at_fobs_inconsistent(self, hard, fobs_orb):
        fobs_orb = np.asarray(fobs_orb)
        edges = self.edges + [fobs_orb, ]

        # shape: (M, Q, Z)
        dens = self.static_binary_density   # d3n/[dlog10(M) dq dz]  units: [Mpc^-3]
        shape = dens.shape
        new_shape = shape + (fobs_orb.size, )

        rz = self._redz_prime[..., np.newaxis] * np.ones(new_shape)
        coal = (rz > 0.0)

        dc = cosmo.comoving_distance(rz[coal]).to('Mpc').value
        frst_orb = utils.frst_from_fobs(
            fobs_orb[np.newaxis, np.newaxis, np.newaxis, :], rz
        )

        # (Z,) this is `(dVc/dz) * (dz/dt)` in units of [Mpc^3/s]
        cosmo_fact = 4 * np.pi * (SPLC/MPC) * np.square(dc) * (1.0 + rz[coal])

        # # (M, Q) calculate chirp-mass
        mt = self.mtot[:, np.newaxis, np.newaxis, np.newaxis]
        mr = self.mrat[np.newaxis, :, np.newaxis, np.newaxis]
        mt, mr = [(mm * np.ones(new_shape))[coal] for mm in [mt, mr]]

        # Convert from observer-frame orbital freq, to rest-frame orbital freq
        sa = utils.kepler_sepa_from_freq(mt, frst_orb[coal])
        # (X, M*Q*Z), hardening rate, negative values, units of [cm/sec]
        args = [mt, mr, sa]
        dadt = hard.dadt(*args)
        # Calculate `tau = dt/dlnf_r = f_r / (df_r/dt)`
        # dfdt is positive (increasing frequency)
        dfdt, _ = utils.dfdt_from_dadt(dadt, sa, frst_orb=frst_orb[coal])
        tau = frst_orb[coal] / dfdt

        # (M, Q, Z) units: [1/s] i.e. number per second
        dnum = np.zeros(new_shape)
        dnum[coal] = (dens[..., np.newaxis] * np.ones(new_shape))[coal] * cosmo_fact * tau

        return edges, dnum, rz

    def _dynamic_binary_number_at_sepa_consistent(self, hard, target_sepa, steps=200, details=False):
        """Get correct redshifts for full binary-number calculation.

        Slower but more correct than old `dynamic_binary_number`.
        Same as new cython implementation `holo.sam_cython.dynamic_binary_number_at_fobs`, which is
        more than 10x faster.
        LZK 2023-05-11

        """
        target_sepa = np.asarray(target_sepa)
        ntarget = target_sepa.size    # this will be refered to as 'X' in shapes
        edges = self.edges + [target_sepa, ]
        nmtot, nmrat, nredz = self.shape

        # start from the hardening model's initial separation
        rmax = hard._sepa_init
        # (M,) end at the ISCO
        rmin = utils.rad_isco(self.mtot)
        # Choose steps for each binary, log-spaced between rmin and rmax
        extr = np.log10([rmax * np.ones_like(rmin), rmin])     # (2,M,)
        rads = np.linspace(0.0, 1.0, steps)[np.newaxis, :]     # (1,X)
        # (M, S)  =  (M,1) * (1,S)
        rads = extr[0][:, np.newaxis] + (extr[1] - extr[0])[:, np.newaxis] * rads
        rads = 10.0 ** rads

        # (M, Q, Z, S)
        norm = hard._norm
        mt, mr, rz, rads, norm = np.broadcast_arrays(
            self.mtot[:, np.newaxis, np.newaxis, np.newaxis],
            self.mrat[np.newaxis, :, np.newaxis, np.newaxis],
            self.redz[np.newaxis, np.newaxis, :, np.newaxis],
            rads[:, np.newaxis, np.newaxis, :],
            norm[:, :, np.newaxis, np.newaxis]
        )

        # (M, Q, Z, S)  ==>  (M, Q, Z*S)
        mt, mr, rz, rads, norm = [mm.reshape(nmtot, nmrat, -1) for mm in [mt, mr, rz, rads, norm]]
        dadt_evo = hard.dadt(mt, mr, rads, norm=norm)

        # (M, Q, Z*S)  ==>  (M, Q, Z, S)
        dadt_evo, rz, rads = [mm.reshape(nmtot, nmrat, nredz, steps) for mm in [dadt_evo, rz, rads]]
        # dadt_evo = dadt_evo.reshape(nmtot, nmrat, nredz, steps)

        # Integrate (inverse) hardening rates to calculate total lifetime to each separation
        times_evo = -utils.trapz_loglog(-1.0 / dadt_evo, rads, axis=-1, cumsum=True)
        # Combine the binary-evolution time, with the galaxy-merger time
        times_tot = times_evo + self._gmt_time[:, :, :, np.newaxis]
        redz_evo = utils.redz_after(times_tot, redz=rz[:, :, :, 1:])

        # ---- interpolate to target frequencies

        # `ndinterp` interpolates over axis=1,  so get steps (S,) and target radii (X,) to axis=1

        # get our target separations in the appropriate shape to match evolution arrays
        # (X,)  ==>  (M, Q, Z, X)
        sepa = target_sepa[np.newaxis, np.newaxis, np.newaxis, :] * np.ones(self.shape)[..., np.newaxis]
        # (M, Q, Z, X)  ==>  (M*Q*Z, X)
        sepa = sepa.reshape(-1, target_sepa.size)
        # (M, Q, Z, S-1) ==>  (M*Q*Z, S-1)
        rads, redz_evo = [mm.reshape(-1, steps-1) for mm in [rads[:, :, :, 1:], redz_evo]]

        # `rads` MUST BE INCREASING for interpolation, so reverse the steps
        rads = rads[:, ::-1]
        redz_evo = rads[:, ::-1]
        redz_final = utils.ndinterp(sepa, rads, redz_evo, xlog=True, ylog=False)

        # (M*Q*Z, X) ===> (M, Q, Z, X)
        redz_final = redz_final.reshape(self.shape + (ntarget,))
        coal = (redz_final > 0.0)
        redz_final[~coal] = -1.0

        # shape: (M, Q, Z)
        dens = self.static_binary_density   # d3n/[dlog10(M) dq dz]  units: [Mpc^-3]

        # (M, Q, Z, X) comoving-distance in [Mpc]
        dc = np.zeros_like(redz_final)
        dc[coal] = cosmo.comoving_distance(redz_final[coal]).to('Mpc').value

        # (M, Q, Z, X) this is `(dVc/dz) * (dz/dt)` in units of [Mpc^3/s]
        cosmo_fact = np.zeros_like(redz_final)
        cosmo_fact[coal] = 4 * np.pi * (SPLC/MPC) * np.square(dc[coal]) * (1.0 + redz_final[coal])

        # ---- Calculate timescale `tau = dt/dlnf_r = f_r / (df_r/dt)`

        mt, mr, sepa, norm = np.broadcast_arrays(
            self.mtot[:, np.newaxis, np.newaxis],
            self.mrat[np.newaxis, :, np.newaxis],
            target_sepa[np.newaxis, np.newaxis, :],
            hard._norm[:, :, np.newaxis],
        )
        # hardening rate, negative values, units of [cm/sec]
        dadt = hard.dadt(mt, mr, sepa, norm=norm)
        # dfdt is positive (increasing frequency)
        dfdt, frst_orb = utils.dfdt_from_dadt(dadt, sepa, mtot=mt)
        tau = frst_orb / dfdt

        # (M, Q, Z) units: [1/s] i.e. number per second
        dnum = dens[..., np.newaxis] * cosmo_fact * tau[:, :, np.newaxis, :]
        dnum[~coal] = 0.0

        if details:
            # (M, Q, X)  ==>  (M, Q, Z, X)
            tau = tau[:, :, np.newaxis, :] * np.ones_like(redz_final)
            dadt = dadt[:, :, np.newaxis, :] * np.ones_like(redz_final)
            dets = dict(tau=tau, cosmo_fact=cosmo_fact, dadt=dadt, sepa=target_sepa)
            return edges, dnum, redz_final, dets

        return edges, dnum, redz_final

    @utils.deprecated_fail("`gwb_new`")
    def new_gwb(self, *args, **kwargs):
        pass

    def gwb_new(self, fobs_gw_edges, hard=holo.hardening.Hard_GW(), realize=100):
        """Calculate GWB using new cython implementation, 10x faster!
        """

        assert isinstance(hard, (holo.hardening.Fixed_Time_2PL_SAM, holo.hardening.Hard_GW))

        fobs_gw_cents = kale.utils.midpoints(fobs_gw_edges)

        # convert to orbital-frequency (from GW-frequency)
        fobs_orb_edges = fobs_gw_edges / 2.0
        fobs_orb_cents = fobs_gw_cents / 2.0

        # ---- Calculate number of binaries in each bin

        redz_final, diff_num = holo.sam_cython.dynamic_binary_number_at_fobs(
            fobs_orb_cents, self, hard, cosmo
        )

        edges = [self.mtot, self.mrat, self.redz, fobs_orb_edges]
        number = holo.sam_cython.integrate_differential_number_3dx1d(edges, diff_num)

        # ---- Get the GWB spectrum from number of binaries over grid

        gwb = gravwaves._gws_from_number_grid_integrated_redz(edges, redz_final, number, realize)

        return gwb

    def gwb_old(self, fobs_gw_edges, hard=holo.hardening.Hard_GW, realize=100):
        """Calculate GWB using new `dynamic_binary_number_at_fobs` method, better, but slower.
        """

        fobs_gw_edges = np.atleast_1d(fobs_gw_edges)
        fobs_gw_cents = kale.utils.midpoints(fobs_gw_edges)
        # convert to orbital-frequency (from GW-frequency)
        fobs_orb_edges = fobs_gw_edges / 2.0
        fobs_orb_cents = fobs_gw_cents / 2.0

        edges, dnum, redz_final = self.dynamic_binary_number_at_fobs(hard, fobs_orb_cents)
        edges[-1] = fobs_orb_edges

        number = utils._integrate_grid_differential_number(edges, dnum, freq=False)
        number = number * np.diff(np.log(fobs_gw_edges))

        # ---- Get the GWB spectrum from number of binaries over grid

        gwb = gravwaves._gws_from_number_grid_integrated_redz(edges, redz_final, number, realize)

        return gwb

    def gwb_ideal(self, fobs_gw, sum=True, redz_prime=True):
        """Calculate the idealized, continuous GWB amplitude.

        Calculation follows [Phinney2001]_ (Eq.5) or equivalently [Enoki+Nagashima-2007] (Eq.3.6).
        This calculation assumes a smooth, continuous population of binaries that are purely GW driven.
        * There are no finite-number effects.
        * There are no environmental or non-GW driven evolution effects.
        * There is no coalescence of binaries cutting them off at high-frequencies.

        """
        mstar_pri, mstar_tot = self.mass_stellar()
        # q = m2 / m1
        mstar_rat = mstar_tot / mstar_pri
        # M = m1 + m2
        mstar_tot = mstar_pri + mstar_tot

        rz = self.redz
        rz = rz[np.newaxis, np.newaxis, :]
        if redz_prime:
            args = [mstar_pri[..., np.newaxis], mstar_rat[..., np.newaxis], mstar_tot[..., np.newaxis], rz]
            # Convert to shape (M, Q, Z)
            mstar_pri, mstar_rat, mstar_tot, rz = np.broadcast_arrays(*args)

            gmt_mass = mstar_tot if GMT_USES_MTOT else mstar_pri
            rz, _ = self._gmt.zprime(gmt_mass, mstar_rat, rz)
            print(f"{self} :: {utils.stats(rz)=}")

        # d^3 n / [dlog10(M) dq dz] in units of [Mpc^-3], convert to [cm^-3]
        ndens = self.static_binary_density / (MPC**3)

        mt = self.mtot[:, np.newaxis, np.newaxis]
        mr = self.mrat[np.newaxis, :, np.newaxis]
        gwb = gravwaves.gwb_ideal(fobs_gw, ndens, mt, mr, rz, dlog10=True, sum=sum)
        return gwb

    def gwb(self, fobs_gw_edges, hard=holo.hardening.Hard_GW(), realize=100, loudest=1, params=False):
        """Calculate the (smooth/semi-analytic) GWB and CWs at the given observed GW-frequencies.

        Parameters
        ----------
        fobs_gw_edges : (F,) array_like of scalar,
            Observer-frame GW-frequencies. [1/sec]
            These are the frequency bin edges, which are integrated across to get the number of binaries in each
            frequency bin.
        hard : holodeck.evolution._Hardening class or instance
            Hardening mechanism to apply over the range of `fobs_gw`.
        realize : int
            Specification of how many discrete realizations to construct.
            Realizations approximate the finite-source effects of a realistic population.
        loudest : int
            Number of loudest single sources to distinguish from the background.
        params : Boolean
            Whether or not to return astrophysical parameters of the binaries.


        Returns
        -------
        hc_ss : (F, R, L) NDarray of scalars
            The characteristic strain of the L loudest single sources at each frequency.
        hc_bg : (F, R) NDarray of scalars
            Characteristic strain of the GWB.
        sspar : (4, F, R, L) NDarray of scalars
            Astrophysical parametes (total mass, mass ratio, initial redshift, final redshift) of each
            loud single sources, for each frequency and realization.
            Returned only if params = True.
        bgpar : (7, F, R) NDarray of scalars
            Average effective binary astrophysical parameters (total mass, mass ratio, initial redshift,
            final redshift, final comoving distance, final separation, final angular separation)
            for background sources at each frequency and realization,
            Returned only if params = True.
        """

        if not isinstance(hard, (holo.hardening.Fixed_Time_2PL_SAM, holo.hardening.Hard_GW)):
            err = (
                "`sam_cython` methods only work with `Fixed_Time_2PL_SAM` or `Hard_GW` hardening models!  "
                "Use `gwb_only` for alternative classes!"
            )
            self._log.exception(err)
            raise ValueError(err)

        fobs_gw_cents = kale.utils.midpoints(fobs_gw_edges)

        # convert to orbital-frequency (from GW-frequency)
        fobs_orb_edges = fobs_gw_edges / 2.0
        fobs_orb_cents = fobs_gw_cents / 2.0

        # ---- Calculate number of binaries in each bin

        redz_final, diff_num = holo.sam_cython.dynamic_binary_number_at_fobs(
            fobs_orb_cents, self, hard, cosmo
        )

        edges = [self.mtot, self.mrat, self.redz, fobs_orb_edges]
        number = holo.sam_cython.integrate_differential_number_3dx1d(edges, diff_num)

        # ---- Get the Single Source and GWB spectrum from number of binaries over grid

        ret_vals = single_sources.ss_gws_redz(edges, redz_final, number,
                                              realize=realize, loudest=loudest, params=params)

        hc_ss = ret_vals[0]
        hc_bg = ret_vals[1]
        if params:
            sspar = ret_vals[2]
            bgpar = ret_vals[3]

        if params:
            return hc_ss, hc_bg, sspar, bgpar

        return hc_ss, hc_bg

    def _ndens_gal(self, mass_gal, mrat_gal, redz):
        if GSMF_USES_MTOT or GPF_USES_MTOT or GMT_USES_MTOT:
            self._log.warning("{self.__class__}._ndens_gal assumes that primary mass is used for GSMF, GPF and GMT!")

        # NOTE: dlog10(M_1) / dlog10(M) = (M/M_1) * (dM_1/dM) = 1
        nd = self._gsmf(mass_gal, redz) * self._gpf(mass_gal, mrat_gal, redz)
        nd = nd * cosmo.dtdz(redz) / self._gmt(mass_gal, mrat_gal, redz)
        return nd

    def _ndens_mbh(self, mass_gal, mrat_gal, redz):
        if GSMF_USES_MTOT or GPF_USES_MTOT or GMT_USES_MTOT:
            self._log.warning("{self.__class__}._ndens_mbh assumes that primary mass is used for GSMF, GPF and GMT!")

        # this is  d^3 n / [dlog10(M_gal-pri) dq_gal dz]
        nd_gal = self._ndens_gal(mass_gal, mrat_gal, redz)

        mplaw = self._mmbulge._mplaw
        dqbh_dqgal = mplaw * np.power(mrat_gal, mplaw - 1.0)

        dmstar_dmbh__pri = self._mmbulge.dmstar_dmbh(mass_gal)   # [unitless]
        mbh_pri = self._mmbulge.mbh_from_mstar(mass_gal, scatter=False)
        mbh_sec = self._mmbulge.mbh_from_mstar(mass_gal * mrat_gal, scatter=False)
        mbh = mbh_pri + mbh_sec
        mrat_mbh = mbh_sec / mbh_pri

        dlm_dlm = (mbh / mass_gal) * dmstar_dmbh__pri / (1.0 + mrat_mbh)
        dens = nd_gal * dlm_dlm / dqbh_dqgal
        return dens

    def _integrated_binary_density(self, ndens=None, sum=True):
        # d^3 n / [dlog10M dq dz]
        if ndens is None:
            ndens = self.static_binary_density
        integ = utils.trapz(ndens, np.log10(self.mtot), axis=0, cumsum=False)
        integ = utils.trapz(integ, self.mrat, axis=1, cumsum=False)
        integ = utils.trapz(integ, self.redz, axis=2, cumsum=False)
        if sum:
            integ = integ.sum()
        return integ


# ===========================================
# ====    Evolution & Utility Methods    ====
# ===========================================


def sample_sam_with_hardening(
        sam, hard,
        fobs_orb=None, sepa=None, sample_threshold=10.0, cut_below_mass=None, limit_merger_time=None,
        **sample_kwargs
):
    """Discretize Semi-Analytic Model into sampled binaries assuming the given binary hardening rate.

    Parameters
    ----------
    sam : `Semi_Analytic_Model`
        Instance of an initialized semi-analytic model.
    hard : `holodeck.evolution._Hardening`
        Binary hardening model for calculating binary hardening rates (dadt or dfdt).
    fobs_orb : ArrayLike
        Observer-frame orbital-frequencies.  Units of [1/sec].
        NOTE: Either `fobs_orb` or `sepa` must be provided, and not both.
    sepa : ArrayLike
        Binary orbital separation.  Units of [cm].
        NOTE: Either `fobs_orb` or `sepa` must be provided, and not both.

    Returns
    -------
    vals : (4, S) ndarray of scalar
        Parameters of sampled binaries.  Four parameters are:
        * mtot : total mass of binary (m1+m2) in [grams]
        * mrat : mass ratio of binary (m2/m1 <= 1)
        * redz : redshift of binary
        * fobs_orb / sepa : observer-frame orbital-frequency [1/s]  or  binary separation [cm]
    weights : (S,) ndarray of scalar
        Weights of each sample point.
    edges : (4,) of list of scalars
        Edges of parameter-space grid for each of above parameters (mtot, mrat, redz, fobs_orb)
        The lengths of each list will be [(M,), (Q,), (Z,), (F,)]
    dnum : (M, Q, Z, F) ndarray of scalar
        Number-density of binaries over grid specified by `edges`.

    """

    if (sample_threshold < 1.0) and (sample_threshold > 0.0):
        msg = (
            f"`sample_threshold={sample_threshold}` values less than unity can lead to surprising behavior!"
        )
        log.warning(msg)

    # returns  dN/[dlog10(M) dq dz dln(f_r)]
    # edges: Mtot [grams], mrat (q), redz (z), {fobs_orb (f) [1/s]   OR   sepa (a) [cm]}
    # `fobs_orb` is observer-frame orbital-frequency
    edges, dnum = sam.dynamic_binary_number(hard, fobs_orb=fobs_orb, sepa=sepa, limit_merger_time=limit_merger_time)

    edges_integrate = [np.copy(ee) for ee in edges]
    edges_sample = [np.log10(edges[0]), edges[1], edges[2], np.log(edges[3])]

    if cut_below_mass is not None:
        m2 = edges[0][:, np.newaxis] * edges[1][np.newaxis, :]
        bads = (m2 < cut_below_mass)
        dnum[bads] = 0.0
        num_bad = np.count_nonzero(bads)
        msg = (
            f"Cutting out systems with secondary below {cut_below_mass/MSOL:.2e} Msol;"
            f" {num_bad:.2e}/{bads.size:.2e} = {num_bad/bads.size:.4f}"
        )
        log.warning(msg)

    # Sample redshift by first converting to comoving volume, sampling, then convert back
    if REDZ_SAMPLE_VOLUME:
        redz = edges[2]
        volume = cosmo.comoving_volume(redz).to('Mpc3').value

        # convert from dN/dz to dN/dVc, dN/dVc = (dN/dz) * (dz/dVc) = (dN/dz) / (dVc/dz)
        dvcdz = cosmo.dVcdz(redz, cgs=False).value
        dnum = dnum / dvcdz[np.newaxis, np.newaxis, :, np.newaxis]

        # change variable from redshift to comoving-volume, both sampling and integration
        edges_sample[2] = volume
        edges_integrate[2] = volume
    else:
        msg = (
            "Sampling redshifts directly, instead of via comoving-volume.  This is less accurate!"
        )
        log.warning(msg)

    # Find the 'mass' (total number of binaries in each bin) by multiplying each bin by its volume
    # NOTE: this needs to be done manually, instead of within kalepy, because of log-spacings
    mass = utils._integrate_grid_differential_number(edges_integrate, dnum, freq=True)

    # ---- sample binaries from distribution
    if (sample_threshold is None) or (sample_threshold == 0.0):
        msg = (
            f"Sampling *all* binaries (~{mass.sum():.2e}).  "
            "Set `sample_threshold` to only sample outliers."
        )
        log.warning(msg)
        vals = kale.sample_grid(edges_sample, dnum, mass=mass, **sample_kwargs)
        weights = np.ones(vals.shape[1], dtype=int)
    else:
        vals, weights = kale.sample_outliers(
            edges_sample, dnum, sample_threshold, mass=mass, **sample_kwargs
        )

    vals[0] = 10.0 ** vals[0]
    vals[3] = np.e ** vals[3]

    # If we sampled in comoving-volume, instead of redshift, convert back to redshift
    if REDZ_SAMPLE_VOLUME:
        vals[2] = np.power(vals[2] / (4.0*np.pi/3.0), 1.0/3.0)
        vals[2] = cosmo.dcom_to_z(vals[2] * MPC)

    # Remove low-mass systems after sampling also
    if cut_below_mass is not None:
        bads = (vals[0] * vals[1] < cut_below_mass)
        vals = vals.T[~bads].T
        weights = weights[~bads]

    return vals, weights, edges, dnum, mass


def evolve_eccen_uniform_single(sam, eccen_init, sepa_init, nsteps):
    """Evolve binary eccentricity from an initial value along a range of separations.

    Parameters
    ----------
    sam : `holodeck.sam.Semi_Analytic_Model` instance
        The input semi-analytic model.  All this does is provide the range of total-masses to
        determine the minimum ISCO radius, which then determines the smallest separations to
        evolve until.
    eccen_init : float,
        Initial eccentricity of binaries at the given initial separation `sepa_init`.
        Must be between [0.0, 1.0).
    sepa_init : float,
        Initial binary separation at which evolution begins.  Units of [cm].
    nsteps : int,
        Number of (log-spaced) steps in separation between the initial separation `sepa_init`,
        and the final separation which is determined as the minimum ISCO radius based on the
        smallest total-mass of binaries in the `sam` instance.

    Returns
    -------
    sepa : (E,) ndarray of float
        The separations at which the eccentricity evolution is defined over.  This is the
        independent variable of the evolution.
        The shape `E` is the value of the `nsteps` parameter.
    eccen : (E,)
        The eccentricity of the binaries at each location in separation given by `sepa`.
        The shape `E` is the value of the `nsteps` parameter.

    """
    assert (0.0 <= eccen_init) and (eccen_init < 1.0)

    #! CHECK FOR COALESCENCE !#

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

        _, e1 = holo.utils.rk4_step(holo.hardening.Hard_GW.deda, x0=a0, y0=e0, dx=da)
        e1 = np.clip(e1, 0.0, None)
        eccen[step] = e1

    return sepa, eccen


def add_scatter_to_masses(mtot, mrat, dens, scatter, refine=4, log=None):
    """Add the given scatter to masses m1 and m2, for the given distribution of binaries.

    The procedure is as follows (see `dev-notebooks/sam-ndens-scatter.ipynb`):
    * (1) The density is first interpolated to a uniform, regular grid in (m1, m2) space.
          A 2nd-order interpolant is used first.  A 0th-order interpolant is used to fill-in bad values.
          In-between, a 1st-order interpolant is used if `linear_interp_backup` is True.
    * (2) The density distribution is convolved with a smoothing function along each axis (m1, m2) to
          account for scatter.
    * (3) The new density distribution is interpolated back to the original (mtot, mrat) grid.

    Parameters
    ----------
    mtot : (M,) ndarray
        Total masses in grams.
    mrat : (Q,) ndarray
        Mass ratios.
    dens : (M, Q) ndarray
        Density of binaries over the given mtot and mrat domain.
    scatter : float
        Amount of scatter in the M-MBulge relationship, in dex (i.e. over log10 of masses).
    refine : int,
        The increased density of grid-points used in the intermediate (m1, m2) domain, in step (1).
    linear_interp_backup : bool,
        Whether a linear interpolant is used to fill-in bad values after the 2nd order interpolant.
        This generally doesn't seem to fix any values.
    logspace_interp : bool,
        Whether interpolation should be performed in the log-space of masses.
        NOTE: strongly recommended.

    Returns
    -------
    m1m2_dens : (M, Q) ndarray,
        Binary density with scatter introduced.

    """
    if log is None:
        log = holo.log

    assert np.ndim(dens) == 3
    assert np.shape(dens)[:2] == (mtot.size, mrat.size)
    dist = sp.stats.norm(loc=0.0, scale=scatter)
    output = np.zeros_like(dens)

    # Get the primary and secondary masses corresponding to these total-mass and mass-ratios
    m1, m2 = utils.m1m2_from_mtmr(mtot[:, np.newaxis], mrat[np.newaxis, :])
    m1m2_on_mtmr_grid = (m1.flatten(), m2.flatten())

    # Construct a symmetric rectilinear grid in (m1, m2) space
    grid_size = m1.shape[0] * refine
    # make sure the extrema will fully span the required domain
    mextr = utils.minmax([0.9*mtot[0]*mrat[0]/(1.0 + mrat[0]), mtot[-1]*(1.0 + mrat[0])/mrat[0]])
    _mgrid = np.logspace(*np.log10(mextr), grid_size)

    # Interpolate in log-space [recommended]
    mgrid_log10 = np.log10(_mgrid)
    m1m2_on_mtmr_grid = tuple([np.log10(mm) for mm in m1m2_on_mtmr_grid])

    m1m2_grid = np.meshgrid(mgrid_log10, mgrid_log10, indexing='ij')

    # Interpolate from irregular m1m2 space (based on mtmr space), into regular m1m2 grid
    numz = np.shape(dens)[2]
    dlay = None
    weights = utils._get_rolled_weights(mgrid_log10, dist)

    for ii in range(numz):
        dens_redz = dens[:, :, ii]
        if dlay is None:
            points = m1m2_on_mtmr_grid
        else:
            points = dlay
        interp = sp.interpolate.CloughTocher2DInterpolator(points, dens_redz.flatten())
        m1m2_dens = interp(tuple(m1m2_grid))
        if dlay is None:
            dlay = interp.tri

        # Fill in problematic values with zeroth-order interpolant
        bads = np.isnan(m1m2_dens) | (m1m2_dens < 0.0)
        log.debug(f"After interpolation, {utils.frac_str(bads)} bad values exist")
        if np.any(bads):
            # temp = sp.interpolate.griddata(m1m2_on_mtmr_grid, dens.flatten(), tuple(m1m2_grid), method='nearest')
            interp = sp.interpolate.NearestNDInterpolator(points, dens_redz.flatten())
            temp = interp(tuple(m1m2_grid))
            m1m2_dens[bads] = temp[bads]
            bads = np.isnan(m1m2_dens) | (m1m2_dens < 0.0)
            log.debug(f"After 0th order interpolation, {utils.frac_str(bads)} bad values exist")
            if np.any(bads):
                err = f"After 0th order interpolation, {utils.frac_str(bads)} remain!"
                log.exception(err)
                raise ValueError(err)

        # Introduce scatter along both the 0th (primary) and 1th (secondary) axes
        m1m2_dens = utils._scatter_with_weights(m1m2_dens, weights, axis=0)
        m1m2_dens = utils._scatter_with_weights(m1m2_dens, weights, axis=1)

        # Interpolate result back to mtmr grid
        interp = sp.interpolate.RegularGridInterpolator((mgrid_log10, mgrid_log10), m1m2_dens)
        m1m2_dens = interp(m1m2_on_mtmr_grid, method='linear').reshape(m1.shape)
        output[:, :, ii] = m1m2_dens[...]

    return output



'''
def _add_scatter_to_masses(mtot, mrat, dens, scatter, refine=4, linear_interp_backup=True, logspace_interp=True):
    """Add the given scatter to masses m1 and m2, for the given distribution of binaries.

    The procedure is as follows (see `dev-notebooks/sam-ndens-scatter.ipynb`):
    * (1) The density is first interpolated to a uniform, regular grid in (m1, m2) space.
          A 2nd-order interpolant is used first.  A 0th-order interpolant is used to fill-in bad values.
          In-between, a 1st-order interpolant is used if `linear_interp_backup` is True.
    * (2) The density distribution is convolved with a smoothing function along each axis (m1, m2) to
          account for scatter.
    * (3) The new density distribution is interpolated back to the original (mtot, mrat) grid.

    Parameters
    ----------
    mtot : (M,) ndarray
        Total masses in grams.
    mrat : (Q,) ndarray
        Mass ratios.
    dens : (M, Q) ndarray
        Density of binaries over the given mtot and mrat domain.
    scatter : float
        Amount of scatter in the M-MBulge relationship, in dex (i.e. over log10 of masses).
    refine : int,
        The increased density of grid-points used in the intermediate (m1, m2) domain, in step (1).
    linear_interp_backup : bool,
        Whether a linear interpolant is used to fill-in bad values after the 2nd order interpolant.
        This generally doesn't seem to fix any values.
    logspace_interp : bool,
        Whether interpolation should be performed in the log-space of masses.
        NOTE: strongly recommended.

    Returns
    -------
    m1m2_dens : (M, Q) ndarray,
        Binary density with scatter introduced.

    """
    assert np.shape(dens) == (mtot.size, mrat.size)

    dist = sp.stats.norm(loc=0.0, scale=scatter)

    # Get the primary and secondary masses corresponding to these total-mass and mass-ratios
    m1, m2 = utils.m1m2_from_mtmr(mtot[:, np.newaxis], mrat[np.newaxis, :])
    m1m2_on_mtmr_grid = (m1.flatten(), m2.flatten())

    # Construct a symmetric rectilinear grid in (m1, m2) space
    grid_size = m1.shape[0] * refine
    # make sure the extrema will fully span the required domain
    mextr = utils.minmax([0.9*mtot[0]*mrat[0]/(1.0 + mrat[0]), mtot[-1]*(1.0 + mrat[0])/mrat[0]])
    mgrid = np.logspace(*np.log10(mextr), grid_size)

    # Interpolate in log-space [recommended]
    scatter_mgrid = mgrid.copy()
    if logspace_interp:
        mgrid = np.log10(mgrid)
        m1m2_on_mtmr_grid = tuple([np.log10(mm) for mm in m1m2_on_mtmr_grid])

    m1m2_grid = np.meshgrid(mgrid, mgrid, indexing='ij')
    # Interpolate from irregular m1m2 space (based on mtmr space), into regular m1m2 grid
    m1m2_dens = sp.interpolate.griddata(m1m2_on_mtmr_grid, dens.flatten(), tuple(m1m2_grid), method='cubic')
    # Fill in problematic values with first-order interpolant
    if linear_interp_backup:
        bads = np.isnan(m1m2_dens) | (m1m2_dens <= 0.0)
        temp = sp.interpolate.griddata(m1m2_on_mtmr_grid, dens.flatten(), tuple(m1m2_grid), method='linear')
        log.debug(f"After 2nd order interpolation, {utils.frac_str(bads)} bad values")
        m1m2_dens[bads] = temp[bads]

    # Fill in problematic values with zeroth-order interpolant
    bads = np.isnan(m1m2_dens) | (m1m2_dens <= 0.0)
    log.debug(f"After interpolation, {utils.frac_str(bads)} bad values exist")
    if np.any(bads):
        temp = sp.interpolate.griddata(m1m2_on_mtmr_grid, dens.flatten(), tuple(m1m2_grid), method='nearest')
        m1m2_dens[bads] = temp[bads]
        bads = np.isnan(m1m2_dens) | (m1m2_dens <= 0.0)
        log.debug(f"After 0th order interpolation, {utils.frac_str(bads)} bad values exist")
        if np.any(bads):
            err = f"After 0th order interpolation, {utils.frac_str(bads)} remain!"
            log.exception(err)
            log.error(f"{utils.stats(dens.flatten())}")
            raise ValueError(err)

    # Introduce scatter along both the 0th (primary) and 1th (secondary) axes
    m1m2_dens = utils.scatter_redistribute(scatter_mgrid, dist, m1m2_dens, axis=0)
    m1m2_dens = utils.scatter_redistribute(scatter_mgrid, dist, m1m2_dens, axis=1)

    # Interpolate result back to mtmr grid
    interp = sp.interpolate.RegularGridInterpolator((mgrid, mgrid), m1m2_dens)
    m1m2_dens = interp(m1m2_on_mtmr_grid, method='linear').reshape(m1.shape)
    return m1m2_dens
'''
