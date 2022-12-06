"""Semi Analytic Modeling (SAM) submodule.

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
* Change mass-ratios and redshifts (1+z) to log-space; expand q parameter range.
* Incorporate arbitrary hardening mechanisms into SAM construction, sample self-consistently.
* Should there be an additional dimension of parameter space for galaxy properties?  This way
  variance in scaling relations is incorporated directly before calculating realizations.
* Allow SAM class to take M-sigma in addition to M-Mbulge.
* Should GW calculations be moved to the `gravwaves.py` module?

References
----------
* [Sesana2008]_ Sesana, Vecchio, Colacino 2008.
* [Chen2019]_ Chen, Sesana, Conselice 2019.

"""

import abc

# import numba
import numpy as np

import kalepy as kale

import holodeck as holo
from holodeck import cosmo, utils, log
from holodeck.constants import GYR, SPLC, MSOL, MPC, NWTG
from holodeck import relations, gravwaves

_AGE_UNIVERSE_GYR = cosmo.age(0.0).to('Gyr').value  # [Gyr]  ~ 13.78

REDZ_SCALE_LOG = True
REDZ_SAMPLE_VOLUME = True


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


class GSMF_Schechter(_Galaxy_Stellar_Mass_Function):
    """Single Schechter Function - Galaxy Stellar Mass Function.

    This is density per unit log10-interval of stellar mass, i.e. $Phi = dn / d\\log_{10}(M)$

    See: [Chen2019]_ Eq.9 and enclosing section.

    """

    def __init__(self, phi0=-2.77, phiz=-0.27, mref0=1.74e11*MSOL, mrefz=0.0, alpha0=-1.24, alphaz=-0.03):
        self._phi0 = phi0         # - 2.77  +/- [-0.29, +0.27]  [1/Mpc^3]
        self._phiz = phiz         # - 0.27  +/- [-0.21, +0.23]  [1/Mpc^3]
        self._mref0 = mref0       # +11.24  +/- [-0.17, +0.20]  Msol
        self._mrefz = mrefz       #  0.0                        Msol    # noqa
        self._alpha0 = alpha0     # -1.24   +/- [-0.16, +0.16]
        self._alphaz = alphaz     # -0.03   +/- [-0.14, +0.16]
        return

    def __call__(self, mstar, redz):
        """Return the number-density of galaxies at a given stellar mass.

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
            Number-density of galaxies in units of [Mpc^-3]

        """
        phi = self._phi_func(redz)
        mref = self._mref_func(redz)
        alpha = self._alpha_func(redz)
        xx = mstar / mref
        # [Chen2019]_ Eq.8
        rv = np.log(10.0) * phi * np.power(xx, 1.0 + alpha) * np.exp(-xx)
        return rv

    def _phi_func(self, redz):
        """See: [Chen2019]_ Eq.9
        """
        return np.power(10.0, self._phi0 + self._phiz * redz)

    def _mref_func(self, redz):
        """See: [Chen2019]_ Eq.10 - NOTE: added `redz` term
        """
        return self._mref0 + self._mrefz * redz

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
    def __call__(self, mtot, mrat, redz):
        """Return the fraction of galaxies in pairs of the given parameters.

        Parameters
        ----------
        mtot : scalar or ndarray,
            Total-mass of combined system, units of [grams].
        mrat : scalar or ndarray,
            Mass-ratio of the system (m2/m1 <= 1.0), dimensionless.
        redz : scalar or ndarray,
            Redshift.

        Returns
        -------
        rv : scalar or ndarray,
            Galaxy pair fraction, in dimensionless units.

        """
        return


class GPF_Power_Law(_Galaxy_Pair_Fraction):
    """Galaxy Pair Fraction - Single Power-Law
    """

    def __init__(self, frac_norm_allq=0.025, frac_norm=None, mref=1.0e11*MSOL,
                 malpha=0.0, zbeta=0.8, qgamma=0.0, obs_conv_qlo=0.25):
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

        self._mref = mref   # NOTE: this is `a * M_0 = 1e11 Msol` in papers
        return

    def __call__(self, mtot, mrat, redz):
        """Return the fraction of galaxies in pairs of the given parameters.

        Parameters
        ----------
        mtot : scalar or ndarray,
            Total-mass of combined system, units of [grams].
        mrat : scalar or ndarray,
            Mass-ratio of the system (m2/m1 <= 1.0), dimensionless.
        redz : scalar or ndarray,
            Redshift.

        Returns
        -------
        rv : scalar or ndarray,
            Galaxy pair fraction, in dimensionless units.

        """
        # convert from total-mass to primary-mass
        mpri = utils.m1m2_from_mtmr(mtot, mrat)[0]
        f0p = self._frac_norm
        am0 = self._mref
        aa = self._malpha
        bb = self._zbeta
        gg = self._qgamma
        rv = f0p * np.power(mpri/am0, aa) * np.power(1.0 + redz, bb) * np.power(mrat, gg)
        return rv


# ----    Galaxy Merger Time    ----


class _Galaxy_Merger_Time(abc.ABC):
    """Galaxy Merger Time base class, used to model merger timescale of galaxy pairs.
    """

    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        return

    @abc.abstractmethod
    def __call__(self, mtot, mrat, redz):
        """Return the galaxy merger time for the given parameters.

        Parameters
        ----------
        mtot : scalar or ndarray,
            Total-mass of combined system, units of [grams].
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

    def zprime(self, mtot, mrat, redz, **kwargs):
        """Return the redshift after merger (i.e. input `redz` delayed by merger time).
        """
        tau0 = self(mtot, mrat, redz)  # sec
        age = cosmo.age(redz).to('s').value
        new_age = age + tau0

        if np.isscalar(new_age):
            if new_age < _AGE_UNIVERSE_GYR * GYR:
                redz_prime = cosmo.tage_to_z(new_age)
            else:
                redz_prime = -1

        else:
            redz_prime = -1.0 * np.ones_like(new_age)
            idx = (new_age < _AGE_UNIVERSE_GYR * GYR)
            redz_prime[idx] = cosmo.tage_to_z(new_age[idx])

        return redz_prime


class GMT_Power_Law(_Galaxy_Merger_Time):
    """Galaxy Merger Time - simple power law prescription
    """

    def __init__(self, time_norm=0.55*GYR, mref=7.2e10*MSOL, malpha=0.0, zbeta=-0.5, qgamma=0.0):
        # tau0  [sec]
        self._time_norm = time_norm   # +0.55  b/t [+0.1, +2.0]  [+0.1, +10.0]  values for [Gyr]
        self._malpha = malpha         # +0.0   b/t [-0.2, +0.2]  [-0.2, +0.2 ]
        self._zbeta = zbeta           # -0.5   b/t [-2.0, +1.0]  [-3.0, +1.0 ]
        self._qgamma = qgamma         # +0.0   b/t [-0.2, +0.2]  [-0.2, +0.2 ]

        # [Msol]  NOTE: this is `b * M_0 = 0.4e11 Msol / h0` in [Chen2019]_
        self._mref = mref
        return

    def __call__(self, mtot, mrat, redz):
        """Return the galaxy merger time for the given parameters.

        Parameters
        ----------
        mtot : (N,) array_like[scalar]
            Total mass of each binary, converted to primary-mass (used in literature equations).
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
        mpri = utils.m1m2_from_mtmr(mtot, mrat)[0]   # [grams]
        tau0 = self._time_norm                       # [sec]
        bm0 = self._mref                             # [grams]
        aa = self._malpha
        bb = self._zbeta
        gg = self._qgamma
        mtime = tau0 * np.power(mpri/bm0, aa) * np.power(1.0 + redz, bb) * np.power(mrat, gg)
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
        self, mtot=[2.75e5*MSOL, 1.0e11*MSOL, 46], mrat=[0.02, 1.0, 50], redz=[0.01, 6.0, 61],
        shape=None,
        gsmf=GSMF_Schechter, gpf=GPF_Power_Law, gmt=GMT_Power_Law, mmbulge=relations.MMBulge_MM2013
    ):
        """

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

        # Sanitize input classes/instances
        gsmf = utils._get_subclass_instance(gsmf, None, _Galaxy_Stellar_Mass_Function)
        gpf = utils._get_subclass_instance(gpf, None, _Galaxy_Pair_Fraction)
        gmt = utils._get_subclass_instance(gmt, None, _Galaxy_Merger_Time)
        mmbulge = utils._get_subclass_instance(mmbulge, None, relations._MMBulge_Relation)

        # NOTE: Create a copy of input args to make sure they aren't overwritten (in-place)
        mtot = [mt for mt in mtot]
        mrat = [mt for mt in mrat]
        redz = [mt for mt in redz]

        # Redefine shape of grid (i.e. number of bins in each parameter)
        if shape is not None:
            if np.isscalar(shape):
                shape = [shape+ii for ii in range(3)]

            shape = np.asarray(shape)
            if not kale.utils.isinteger(shape) or (shape.size != 3) or np.any(shape <= 1):
                raise ValueError(f"`shape` ({shape}) must be an integer, or (3,) iterable of integers, larger than 1!")

            # mtot
            if shape[0] is not None:
                mtot[2] = shape[0]
            # mrat
            if shape[1] is not None:
                mrat[2] = shape[1]
            # redz
            if shape[2] is not None:
                redz[2] = shape[2]

        # NOTE: the spacing (log vs lin) is important.  e.g. in integrating from differential-number to (total) number
        self.mtot = np.logspace(*np.log10(mtot[:2]), mtot[2])
        self.mrat = np.linspace(*mrat)

        if REDZ_SCALE_LOG:
            if redz[0] <= 0.0:
                err = f"With `REDZ_SCALE_LOG={REDZ_SCALE_LOG}` redshift lower bound must be non-zero ({redz})!"
                log.exception(err)
                raise ValueError(err)
            self.redz = np.logspace(*np.log10(redz[:2]), redz[2])
        else:
            self.redz = np.linspace(*redz)

        self._gsmf = gsmf             #: Galaxy Stellar-Mass Function (`_Galaxy_Stellar_Mass_Function` instance)
        self._gpf = gpf               #: Galaxy Pair Fraction (`_Galaxy_Pair_Fraction` instance)
        self._gmt = gmt               #: Galaxy Merger Time (`_Galaxy_Merger_Time` instance)
        self._mmbulge = mmbulge       #: Mbh-Mbulge relation (`relations._MMBulge_Relation` instance)

        self._density = None          #: Binary comoving number-density
        self._grid = None             #: Domain of population: total-mass, mass-ratio, and redshift
        return

    @property
    def edges(self):
        """The grid edges defining the domain (list of: [`mtot`, `mrat`, `redz`])
        """
        return [self.mtot, self.mrat, self.redz]

    @property
    def grid(self):
        """Grid of parameter space over which binary population is defined, ndarray (3, M, Q, Z).
        """
        if self._grid is None:
            self._grid = np.meshgrid(*self.edges, indexing='ij')
        return self._grid

    @property
    def shape(self):
        """Shape of the parameter space domain (number of edges in each dimension), (3,) tuple
        """
        shape = [len(ee) for ee in self.edges]
        return tuple(shape)

    def mass_stellar(self):
        """Calculate stellar masses for each MBH based on the M-MBulge relation.

        Returns
        -------
        masses : (N, 2) ndarray of scalar,
            Galaxy total stellar masses for all MBH. [:, 0] is primary, [:, 1] is secondary [grams].

        """
        # total-mass, mass-ratio ==> (M1, M2)
        masses = utils.m1m2_from_mtmr(self.mtot[:, np.newaxis], self.mrat[np.newaxis, :])
        # BH-masses to stellar-masses
        masses = self._mmbulge.mstar_from_mbh(masses, scatter=False)
        return masses

    @property
    def density(self):
        err = "DEPRECATION `Semi_Analytic_Model.density` ==> `Semi_Analytic_Model.static_binary_density`"
        log.exception(err)
        raise AttributeError(err)

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
        This function effectively calculates Eq.21 & 5 of [Chen2019]_; or equivalently, Eq. 6 of [Sesana2008]_.

        """
        if self._density is None:
            dens = np.zeros(self.shape)

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

            zprime = self._gmt.zprime(mstar_tot, mstar_rat, redz)
            # find valid entries (M, Q, Z)
            idx = (zprime > 0.0)
            if ~np.any(idx):
                utils.print_stats(stack=False, print_func=log.error,
                                  mstar_tot=mstar_tot, mstar_rat=mstar_rat, redz=redz)
                raise RuntimeError("No `zprime` values are greater than zero!")

            # these are now 1D arrays of the valid indices
            mstar_pri, mstar_rat, mstar_tot, redz = [ee[idx] for ee in [mstar_pri, mstar_rat, mstar_tot, redz]]

            # ---- Get Galaxy Merger Rate  [Chen2019] Eq.5
            # ! BUG: CHECK Whether primary vs total-mass is being used correctly !
            # `gsmf` returns [1/Mpc^3]   `dtdz` returns [sec]
            dens[idx] = self._gsmf(mstar_pri, redz) * self._gpf(mstar_tot, mstar_rat, redz) * cosmo.dtdz(redz)
            # convert from 1/dlog10(M_star-pri) to 1/dM_star-tot
            dlogten_mstar_pri__dmstar_tot = 1.0 / (mstar_tot * np.log(10.0) * (1.0 + mstar_rat))
            # `gmt` returns [sec]
            dens[idx] *= dlogten_mstar_pri__dmstar_tot / self._gmt(mstar_tot, mstar_rat, redz)
            # now `dens` is  ``dn_gal / [dMstar_tot dq_gal dz]``  with units of [Mpc^-3 gram^-1]

            # ---- Convert to MBH Binary density

            # so far we have ``dn_gal / [dM_gal dq_gal dz]``
            # dn / [dM dq dz] = (dn_gal / [dM_gal dq_gal dz]) * (dM_gal/dM_bh) * (dq_gal / dq_bh)
            dqgal_dqbh = 1.0     # conversion from galaxy mrat to MBH mrat
            # dMs/dMbh
            # dmstar_dmbh = 1.0 / self._mmbulge.dmbh_dmstar(mstar_tot)   # [unitless]
            dmstar_dmbh = self._mmbulge.dmstar_dmbh(mstar_tot)   # [unitless]
            mbh_tot = self._mmbulge.mbh_from_mstar(mstar_tot, scatter=False)  # [gram]
            # Eq.21, now [gram^-1 Mpc^-3]
            dens[idx] *= dqgal_dqbh * dmstar_dmbh
            # Convert from 1/dM to 1/dlog10(M), units are now [Mpc^-3] {lose [1/gram] because now 1/dlog10(M)}
            dens[idx] *= mbh_tot * np.log(10.0)

            self._density = dens

        return self._density

    def dynamic_binary_number(self, hard, fobs_orb=None, sepa=None, limit_merger_time=None):
        """Calculate the differential number of binaries (per bin-volume, per log-freq interval).

        #! BUG: `limit_merger_time` should compare merger time to the redshift of each bin !#

        d^4 N / [dlog10(M) dq dz dln(X)    <===    d^3 n / dlog10(M) dq dz

        The value returned is `d^4 N / [dlog10(M) dq dz dln(X)]`, where X is either:
        *   separation (a)                         ::  if `sepa`     is passed in, or
        *   observer-frame orbital-frequency (f_o) ::  if `fobs_orb` is passed in.

        Parameters
        ----------
        hard : `holodeck.evolution._Hardening`
            Hardening instance for calculating binary evolution rate.
        fobs_orb : ArrayLike
            observer-frame orbital-frequency in [1/sec].
            NOTE: either `fobs_orb` or `sepa` must be provided (and not both).
        sepa : ArrayLike
            Rest-frame binary separation in [cm].
            NOTE: either `fobs_orb` or `sepa` must be provided (and not both).
        limit_merger_time : None or scalar,
            Maximum allowed merger time in [sec].  If `None`, no maximum is imposed.

        Returns
        -------
        edges : (4,) list of 1darrays of scalar
            A list containing the edges along each dimension.  These are:
            {total-mass, mass-ratio, redshift, observer-frame orbital-frequency}
        dnum : (M, Q, Z, F) ndarray of scalar
            Differential number of binaries at the grid edges.  Units are dimensionless number of binaries.

        Notes
        -----
        This function is effectively Eq.6 of [Sesana2008]_.

        d^2 N / dz dln(f_r) = (dn/dz) * (dt/d ln f_r) * (dz/dt) * (dVc/dz)
                            = (dn/dz) * (f_r / [df_r/dt]) * 4 pi c D_c^2 (1+z)
                            = `dens`  *      `tau`        *   `cosmo_fact`

        d^2 N / dz dln(a)   = (dn/dz) * (dz/dt) * (dt/d ln a) * (dVc/dz)
                            = (dn/dz) * (a / [da/dt]) * 4 pi c D_c^2 (1+z)
                            = `dens`  *      `tau`        *   `cosmo_fact`

        To-Do
        -----
        *   Use the `utils.lambda_factor_freq` function instead of calculating it manually.  Be
            careful about units (Mpc).  Need to adjust the function to accept either sepa or freq
            as arguments and outputs.

        """

        if (fobs_orb is None) == (sepa is None):
            err = "one (and only one) of `fobs_orb` or `sepa` must be provided!"
            log.exception(err)
            raise ValueError(err)

        if fobs_orb is not None:
            fobs_orb = np.asarray(fobs_orb)
            xsize = fobs_orb.size
            edges = self.edges + [fobs_orb, ]
        else:
            xsize = len(sepa)
            edges = self.edges + [sepa, ]

        # shape: (M, Q, Z)
        dens = self.static_binary_density   # d3n/[dz dlog10(M) dq]  units: [Mpc^-3]

        # (Z,) comoving-distance in [Mpc]
        dc = cosmo.comoving_distance(self.redz).to('Mpc').value

        # (Z,) this is `(dVc/dz) * (dz/dt)` in units of [Mpc^3/s]
        cosmo_fact = 4 * np.pi * (SPLC/MPC) * np.square(dc) * (1.0 + self.redz)

        # (M, Q) calculate chirp-mass
        mchirp = utils.m1m2_from_mtmr(self.mtot[:, np.newaxis], self.mrat[np.newaxis, :])
        mchirp = utils.chirp_mass(*mchirp)
        # (M, Q, 1, 1) make shape broadcastable for later calculations
        mchirp = mchirp[..., np.newaxis, np.newaxis]

        # (M*Q*Z,) 1D arrays of each total-mass, mass-ratio, and redshift
        mt, mr, rz = [gg.ravel() for gg in self.grid]

        # Make sure we have both `frst_orb` and binary separation `sa`; shapes (X, M*Q*Z)
        if fobs_orb is not None:
            # Convert from observer-frame orbital freq, to rest-frame orbital freq
            # (X, M*Q*Z)
            frst_orb = fobs_orb[:, np.newaxis] * (1.0 + rz[np.newaxis, :])
            sa = utils.kepler_sepa_from_freq(mt[np.newaxis, :], frst_orb)
        else:
            sa = sepa[:, np.newaxis]
            # (X, M*Q*Z), this is the _orbital_ frequency (not GW), and in rest-frame
            frst_orb = utils.kepler_freq_from_sepa(mt[np.newaxis, :], sa)

        # (X, M*Q*Z), hardening rate, negative values, units of [cm/sec]
        dadt = hard.dadt(mt[np.newaxis, :], mr[np.newaxis, :], sa)

        # Calculate `tau = dt/dlnf_r = f_r / (df_r/dt)`
        if fobs_orb is not None:
            # dfdt is positive (increasing frequency)
            dfdt, frst_orb = utils.dfdt_from_dadt(dadt, sa, frst_orb=frst_orb)
            tau = frst_orb / dfdt
        else:
            # recall: dadt is negative (decreasing separation), units of [cm/sec]
            tau = - sa / dadt

        # ---------------------
        if (limit_merger_time is True):
            log.debug("limiting tau to < galaxy merger time")
            mstar = self.mass_stellar()[:, :, :, np.newaxis]
            ms_rat = mstar[1] / mstar[0]
            mstar = mstar.sum(axis=0)   # total mass [grams]
            gmt = self._gmt(mstar, ms_rat, self.redz[np.newaxis, np.newaxis, :])  # [sec]
            bads = (tau > gmt[..., np.newaxis])
            tau[bads] = 0.0
            log.debug(f"tau/GYR={utils.stats(tau/GYR)}, bads={np.count_nonzero(bads)/bads.size:.2e}")

        elif (limit_merger_time in [None, False]):
            pass

        elif utils.isnumeric(limit_merger_time):
            log.debug(f"limiting tau to < {limit_merger_time/GYR:.2f} Gyr")
            bads = (tau > limit_merger_time)
            tau[bads] = 0.0
            log.debug(f"tau/GYR={utils.stats(tau/GYR)}, bads={np.count_nonzero(bads)/bads.size:.2e}")

        else:
            err = f"`limit_merger_time` ({type(limit_merger_time)}) must be boolean or scalar!"
            log.exception(err)
            raise ValueError(err)

        # convert `tau` to the correct shape, note that moveaxis MUST happen _before_ reshape!
        # (X, M*Q*Z) ==> (M*Q*Z, X)
        tau = np.moveaxis(tau, 0, -1)
        # (M*Q*Z, X) ==> (M, Q, Z, X)
        tau = tau.reshape(dens.shape + (xsize,))

        # (M, Q, Z) units: [1/s] i.e. number per second
        dnum = dens * cosmo_fact
        # (M, Q, Z, X) units: [] unitless, i.e. number
        dnum = dnum[..., np.newaxis] * tau

        return edges, dnum

    def gwb(self, fobs_gw_edges, hard=holo.evolution.Hard_GW, realize=False):
        """Calculate the (smooth/semi-analytic) GWB at the given observed GW-frequencies.

        Parameters
        ----------
        fobs_gw : (F,) array_like of scalar,
            Observer-frame GW-frequencies. [1/sec]
            These are the frequency bin edges, which are integrated across to get the number of binaries in each
            frequency bin.
        hard : holodeck.evolution._Hardening class or instance
            Hardening mechanism to apply over the range of `fobs_gw`.
        realize : bool or int,
            Whether to construct a Poisson 'realization' (discretization) of the SAM distribution.
            Realizations approximate the finite-source effects of a realistic population.

        Returns
        -------
        hc : (F,[R,]) ndarray of scalar
            Dimensionless, characteristic strain at each frequency.
            If `realize` is an integer with value R, then R realizations of the GWB are returned,
            such that `hc` has shape (F,R,).

        """

        squeeze = True
        fobs_gw_edges = np.atleast_1d(fobs_gw_edges)
        if np.isscalar(fobs_gw_edges) or np.size(fobs_gw_edges) == 1:
            err = "GWB can only be calculated across bins of frequency, `fobs_gw_edges` must provide bin edges!"
            log.exception(err)
            raise ValueError(err)

        fobs_gw_cents = kale.utils.midpoints(fobs_gw_edges)
        # ---- Get the differential-number of binaries for each bin
        # convert to orbital-frequency (from GW-frequency)
        fobs_orb_edges = fobs_gw_edges / 2.0
        fobs_orb_cents = fobs_gw_cents / 2.0

        # `dnum` is  ``d^4 N / [dlog10(M) dq dz dln(f)]``
        # `dnum` has shape (M, Q, Z, F)  for mass, mass-ratio, redshift, frequency
        #! NOTE: using frequency-bin _centers_ produces more accurate results than frequency-bin _edges_ !#
        edges, dnum = self.dynamic_binary_number(hard, fobs_orb=fobs_orb_cents)
        # edges, dnum = self.dynamic_binary_number(hard, fobs_orb=fobs_orb_edges)
        edges[-1] = fobs_orb_edges
        log.debug(f"dnum: {utils.stats(dnum)}")

        # "integrate" within each bin (i.e. multiply by bin volume)
        # NOTE: `freq` should also be integrated to get proper poisson sampling!
        #       after poisson calculation, need to convert back to dN/dlogf
        #       to get proper characteristic strain measurement
        #! doing  ``dn/dlnf * Delta(ln[f])``  seems to be more accurate than trapz over log(freq) !#
        # number = utils._integrate_grid_differential_number(edges, dnum, freq=True)
        number = utils._integrate_grid_differential_number(edges, dnum, freq=False)
        number = number * np.diff(np.log(fobs_gw_edges))
        log.debug(f"number: {utils.stats(number)}")
        log.debug(f"number.sum(): {number.sum():.4e}")

        # ---- Get the GWB spectrum from number of binaries over grid
        hc = gravwaves._gws_from_number_grid_integrated(edges, number, realize)
        if squeeze:
            hc = hc.squeeze()

        return hc

    def gwb_ideal(self, fobs_gw):
        """Calculate the idealized, continuous GWB amplitude.

        Calculation follows [Phinney2001]_ (Eq.5) or equivalently [Enoki+Nagashima-2007] (Eq.3.6).
        This calculation assumes a smooth, continuous population of binaries that are purely GW driven.
        * There are no finite-number effects.
        * There are no environmental or non-GW driven evolution effects.
        * There is no coalescence of binaries cutting them off at high-frequencies.

        """

        const = ((4.0 * np.pi) / (3 * SPLC**2))
        # (M, Q)
        mc = utils.chirp_mass_mtmr(self.mtot[:, np.newaxis], self.mrat[np.newaxis, :])
        mc = np.power(NWTG * mc, 5.0/3.0)
        # (Z,)
        rz = np.power(1 + self.redz, -1.0/3.0)
        # (F,)
        fogw = np.power(np.pi * fobs_gw, -4.0/3.0)

        # d^3 n / [dlog10(M) dq dz] in units of [Mpc^-3]
        ndens = self.static_binary_density / (MPC**3)

        integ = ndens * mc[..., np.newaxis] * rz[np.newaxis, np.newaxis, :]
        integ = utils.trapz(integ, np.log10(self.mtot), axis=0, cumsum=False)
        integ = utils.trapz(integ, self.mrat, axis=1, cumsum=False)
        integ = utils.trapz(integ, self.redz, axis=2, cumsum=False)

        gwb = const * fogw * np.sum(integ)
        gwb = np.sqrt(gwb)

        return gwb


# ===============================
# ====    Utility Methods    ====
# ===============================


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
