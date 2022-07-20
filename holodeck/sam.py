"""Holodeck - Semi Analytic Modeling submodule

References
----------
* Chen, Sesana, Conselice 2019 = [Chen19]
  Constraining astrophysical observables of galaxy and supermassive black hole binary mergers
  using pulsar timing arrays
  https://ui.adsabs.harvard.edu/abs/2019MNRAS.488..401C/abstract

To-Do
-----
* Check that _GW_ frequencies and _orbital_ frequencies are being used in the correct places.
  Check `number_at_gw_fobs` and related methods.
* Change mass-ratios and redshifts (1+z) to log-space; expand q parameter range.
* Incorporate arbitrary hardening mechanisms into SAM construction, sample self-consistently.
* When using `sample_outliers` check whether the density (used for intrabin sampling) should be
  the log(dens) instead of just `dens`.
* Should there be an additional dimension of parameter space for galaxy properties?  This way
  variance in scaling relations is incorporated directly before calculating realizations.

"""

import abc
import inspect

import numba
import numpy as np

import kalepy as kale

import holodeck as holo
from holodeck import cosmo, utils, log
from holodeck.constants import GYR, SPLC, MSOL, MPC
from holodeck import relations

_AGE_UNIVERSE_GYR = cosmo.age(0.0).to('Gyr').value  # [Gyr]  ~ 13.78

REDZ_SCALE_LOG = True
REDZ_SAMPLE_VOLUME = True


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

    This is density per unit log10-interval of stellar mass, i.e. Phi = dn / dlog10(M)

    See: [Chen19] Eq.9 and enclosing section

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

        See: [Chen19] Eq.8

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
        # [Chen19] Eq.8
        rv = np.log(10.0) * phi * np.power(xx, 1.0 + alpha) * np.exp(-xx)
        return rv

    def _phi_func(self, redz):
        """See: [Chen19] Eq.9
        """
        return np.power(10.0, self._phi0 + self._phiz * redz)

    def _mref_func(self, redz):
        """See: [Chen19] Eq.10 - NOTE: added `redz` term
        """
        return self._mref0 + self._mrefz * redz

    def _alpha_func(self, redz):
        """See: [Chen19] Eq.11
        """
        return self._alpha0 + self._alphaz * redz


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

        # normalization corresponds to f0-prime in [Chen19]
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

        # [Msol]  NOTE: this is `b * M_0 = 0.4e11 Msol / h0` in [Chen19]
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
        # self, mtot=[2.75e5*MSOL, 1.0e11*MSOL, 11], mrat=[0.02, 1.0, 10], redz=[0.0, 6.0, 9],
        shape=None,
        gsmf=GSMF_Schechter, gpf=GPF_Power_Law, gmt=GMT_Power_Law, mmbulge=relations.MMBulge_MM13
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

        if inspect.isclass(gsmf):
            gsmf = gsmf()
        elif not isinstance(gsmf, _Galaxy_Stellar_Mass_Function):
            raise ValueError("`gsmf` must be an instance or subclass of `_Galaxy_Stellar_Mass_Function`!")

        if inspect.isclass(gpf):
            gpf = gpf()
        elif not isinstance(gpf, _Galaxy_Pair_Fraction):
            raise ValueError("`gpf` must be an instance or subclass of `_Galaxy_Pair_Fraction`!")

        if inspect.isclass(gmt):
            gmt = gmt()
        elif not isinstance(gmt, _Galaxy_Merger_Time):
            raise ValueError("`gmt` must be an instance or subclass of `_Galaxy_Merger_Time`!")

        if inspect.isclass(mmbulge):
            mmbulge = mmbulge()
        elif not isinstance(mmbulge, relations._MMBulge_Relation):
            raise ValueError("`mmbulge` must be an instance or subclass of `_MMBulge_Relation`!")

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
                utils.error(err)
            self.redz = np.logspace(*np.log10(redz[:2]), redz[2])
        else:
            self.redz = np.linspace(*redz)

        self._gsmf = gsmf
        self._gpf = gpf
        self._gmt = gmt
        self._mmbulge = mmbulge

        self._density = None
        self._grid = None
        return

    @property
    def edges(self):
        return [self.mtot, self.mrat, self.redz]

    @property
    def grid(self):
        if self._grid is None:
            self._grid = np.meshgrid(*self.edges, indexing='ij')
        return self._grid

    @property
    def shape(self):
        shape = [len(ee) for ee in self.edges]
        return tuple(shape)

    def mass_stellar(self):
        """Calculate stellar masses for each MBH based on the M-MBulge relation.

        Returns
        -------
        masses : (N, 2) ndarray of scalar,
            Galaxy total stellar masses for all MBH.  [:, 0] is primary, [:, 1] is secondary.

        """
        # total-mass, mass-ratio ==> (M1, M2)
        masses = utils.m1m2_from_mtmr(self.mtot[:, np.newaxis], self.mrat[np.newaxis, :])
        # BH-masses to stellar-masses
        masses = self._mmbulge.mstar_from_mbh(masses, scatter=False)
        return masses

    @property
    def density(self):
        """The number-density of binaries in each bin, 'd3n/[dz dlog10M dq]' in units of [Mpc^-3].

        This is calculated once and cached.

        Returns
        -------
        density : (M, Q, Z) ndarray of float
            Number density of binaries, per unit redshift, mass-ratio, and log10 of mass.  Units of [Mpc^-3].

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
            mstar_pri, mstar_rat, mstar_tot, redz = utils.expand_broadcastable(*args)

            zprime = self._gmt.zprime(mstar_tot, mstar_rat, redz)
            # find valid entries (M, Q, Z)
            idx = (zprime > 0.0)
            if ~np.any(idx):
                utils.print_stats(stack=False, print_func=log.error,
                                  mstar_tot=mstar_tot, mstar_rat=mstar_rat, redz=redz)
                raise RuntimeError("No `zprime` values are greater than zero!")

            # these are now 1D arrays of the valid indices
            mstar_pri, mstar_rat, mstar_tot, redz = [ee[idx] for ee in [mstar_pri, mstar_rat, mstar_tot, redz]]

            # ---- Get Galaxy Merger Rate  [Chen19] Eq.5
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

    def diff_num_from_hardening(self, hard, fobs=None, sepa=None, limit_merger_time=None):
        """Calculate the differential number of binaries (per bin-volume, per log-freq interval).

        The value returned is `d^4 N / [dlog10(M) dq dz dln(X)]`, where X is either
        separation (a) or frequency (f_r), depending on whether `sepa` or `fobs` is passed in.

        Parameters
        ----------
        hard : instance
        fobs : observed frequency in [1/sec]
        sepa : rest-frame separation in [cm]
        limit_merger_time : None or scalar,
            Maximum allowed merger time in [sec]

        Returns
        -------
        edges : (4,) list of 1darrays of scalar
            A list containing the edges along each dimension.
        dnum : (M, Q, Z, F) ndarray of scalar
            Differential number of binaries.

        Notes
        -----

        d^2 N / dz dln(f_r) = (dn/dz) * (dt/d ln f_r) * (dz/dt) * (dVc/dz)
                            = (dn/dz) * (f_r / [df_r/dt]) * 4 pi c D_c^2 (1+z)

        d^2 N / dz dln(a)   = (dn/dz) * (dz/dt) * (dt/d ln a) * (dVc/dz)
                            = (dn/dz) * (a / [da/dt]) * 4 pi c D_c^2 (1+z)

        """
        if (fobs is None) == (sepa is None):
            utils.error("one (and only one) of `fobs` or `sepa` must be provided!")

        if fobs is not None:
            fobs = np.asarray(fobs)
            xsize = fobs.size
            edges = self.edges + [fobs, ]
        else:
            xsize = len(sepa)
            edges = self.edges + [sepa, ]

        # shape: (M, Q, Z)
        dens = self.density   # d3n/[dz dlog10(M) dq]  units: [Mpc^-3]

        # (Z,) comoving-distance in [Mpc]
        dc = cosmo.comoving_distance(self.redz).to('Mpc').value

        # [Mpc^3/s] this is `(dVc/dz) * (dz/dt)`
        cosmo_fact = 4 * np.pi * (SPLC/MPC) * np.square(dc) * (1.0 + self.redz)

        # (M, Q)
        mchirp = utils.m1m2_from_mtmr(self.mtot[:, np.newaxis], self.mrat[np.newaxis, :])
        mchirp = utils.chirp_mass(*mchirp)
        # (M, Q, 1, 1)
        mchirp = mchirp[..., np.newaxis, np.newaxis]

        # (M*Q*Z,)
        mt, mr, rz = [gg.ravel() for gg in self.grid]

        if fobs is not None:
            # Convert from obs-GW freq, to rest-frame _orbital_ freq
            # (F, M*Q*Z)
            fr = fobs[:, np.newaxis] * (1.0 + rz[np.newaxis, :]) / 2.0
            sa = utils.kepler_sepa_from_freq(mt[np.newaxis, :], fr)
        else:
            sa = sepa[:, np.newaxis]
            # NOTE: `fr` is the _orbital_ frequency (not GW), and in rest-frame
            fr = utils.kepler_freq_from_sepa(mt[np.newaxis, :], sa)

        # recall: these are negative (decreasing separation)  [cm/sec]
        dadt = hard.dadt(mt[np.newaxis, :], mr[np.newaxis, :], sa)

        # Calculate `tau = dt/dlnf_r = f_r / (df_r/dt)`
        if fobs is not None:
            # dfdt is positive (increasing frequency)
            dfdt, fr = utils.dfdt_from_dadt(dadt, sa, freq_orb=fr)
            tau = fr / dfdt
        else:
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
            utils.error(err)

        # convert `tau` to the correct shape, note that moveaxis MUST happen _before_ reshape!
        # (F, M*Q*Z) ==> (M*Q*Z, F)
        tau = np.moveaxis(tau, 0, -1)
        # (M*Q*Z, F) ==> (M, Q, Z, F)
        tau = tau.reshape(dens.shape + (xsize,))

        # (M, Q, Z) units: [1/s] i.e. number per second
        dnum = dens * cosmo_fact
        # (M, Q, Z, F) units: [] unitless, i.e. number
        dnum = dnum[..., np.newaxis] * tau

        return edges, dnum

    def gwb(self, fobs, hard=holo.evolution.Hard_GW, realize=False):
        """Calculate the (smooth/semi-analytic) GWB at the given observed frequencies.

        Parameters
        ----------
        fobs : (F,) array_like of scalar, [1/sec]
            Observed GW frequencies.
        hard : holodeck.evolution._Hardening class or instance
            Hardening mechanism to apply over the range of `fobs`.
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
        fobs = np.atleast_1d(fobs)
        if np.isscalar(fobs) or np.size(fobs) == 1:
            err = "single values of `fobs` are not allowed, can only calculated GWB within some bin of frequency!"
            err += "  e.g. ``fobs = 1/YR; fobs = [0.9*fobs, 1.1*fobs]``"
            utils.error(err)

        # ---- Get the differential-number of binaries for each bin
        # this is  ``d^4 n / [dlog10(M) dq dz dln(f_r)]``
        # `dnum` has shape (M, Q, Z, F)  for mass, mass-ratio, redshift, frequency
        edges, dnum = self.diff_num_from_hardening(hard, fobs=fobs)

        # "integrate" within each bin (i.e. multiply by bin volume)
        # NOTE: `freq` should also be integrated to get proper poisson sampling!
        #       after poisson calculation, need to convert back to dN/dlogf
        #       to get proper characteristic strain measurement
        number = _integrate_differential_number(edges, dnum, freq=True)

        # ---- Get the GWB spectrum from number of binaries over grid
        hc = _gws_from_number_grid_integrated(edges, dnum, number, realize)
        if squeeze:
            hc = hc.squeeze()

        return hc


def _integrate_differential_number(edges, dnum, freq=False):
    """Integrate the differential number-density of binaries over the given grid (edges).

    NOTE: the `edges` provided MUST all be in linear space, mass is converted to ``log10(M)``
          and frequency is converted to ``ln(f)``.
    NOTE: the density `dnum` MUST correspond to `dn/ [dlog10(M) dq dz dln(f)]`

    Parameters
    ----------
    edges : _type_
    dnum : _type_
    freq : bool, optional
        Whether or not to also integrate the frequency dimension.

    Returns
    -------
    number : ndarray
        Number of binaries in each bin of mass, mass-ratio, redshift, frequency.
        NOTE: if `freq=False`, then `number` corresponds to `dN/dln(f)`, the number of binaries
              per log-interval of frequency.

    """
    # ---- integrate from differential-number to number per bin
    # integrate over dlog10(M)
    number = holo.utils.trapz(dnum, np.log10(edges[0]), axis=0, cumsum=False)
    # integrate over mass-ratio
    number = holo.utils.trapz(number, edges[1], axis=1, cumsum=False)
    # integrate over redshift
    number = holo.utils.trapz(number, edges[2], axis=2, cumsum=False)
    # integrate over frequency (if desired)
    if freq:
        number = holo.utils.trapz(number, np.log(edges[3]), axis=3, cumsum=False)

    return number


def sample_sam_with_hardening(
        sam, hard,
        fobs=None, sepa=None, sample_threshold=10.0, cut_below_mass=None, limit_merger_time=None,
        **sample_kwargs
):
    """Discretize Semi-Analytic Model into sampled binaries assuming the given binary hardening rate.

    fobs in units of [1/yr]
    sepa in units of [pc]

    Returns
    -------
    vals : (4, S) ndarray of scalar
        Parameters of sampled binaries.  Four parameters are:
        * mtot : total mass of binary (m1+m2) in [grams]
        * mrat : mass ratio of binary (m2/m1 <= 1)
        * redz : redshift of binary
        * fobs / sepa : observed-frequency (GW) [1/s] or binary separation [cm]
    weights : (S,) ndarray of scalar
        Weights of each sample point.
    edges : (4,) of list of scalars
        Edges of parameter-space grid for each of above parameters (mtot, mrat, redz, fobs)
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
    # edges: Mtot [grams], mrat (q), redz (z), {fobs (f) [1/s] OR sepa (a) [cm]}
    edges, dnum = sam.diff_num_from_hardening(hard, fobs=fobs, sepa=sepa, limit_merger_time=limit_merger_time)

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
    mass = _integrate_differential_number(edges_integrate, dnum, freq=True)

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


def sampled_gws_from_sam(sam, fobs, hard=holo.evolution.Hard_GW, **kwargs):
    """

    Parameters
    ----------
    fobs : (F,) array_like of scalar,
        Target frequencies of interest in units of [1/yr]

    """
    vals, weights, edges, dens, mass = sample_sam_with_hardening(sam, hard, fobs=fobs, **kwargs)
    gff, gwf, gwb = _gws_from_samples(vals, weights, fobs)
    return gff, gwf, gwb


def _strains_from_samples(vals, redz=True):
    """

    NOTE: this assumes that vales[3] is the observer-frame GW frequency.

    Parameters
    ----------
    vals : _type_
    redz : bool, optional

    Returns
    -------
    _type_
    """

    mc = utils.chirp_mass(*utils.m1m2_from_mtmr(vals[0], vals[1]))

    rz = vals[2]
    dc = cosmo.comoving_distance(rz).cgs.value

    fo = vals[3]
    frst = utils.frst_from_fobs(fo, rz)
    # convert from GW frequency to orbital (divide by 2.0)
    hs = utils.gw_strain_source(mc, dc, frst/2.0)
    return hs, fo


def _gws_from_samples(vals, weights, fobs):
    """

    Parameters
    ----------
    vals : (4, N) ndarray of scalar,
        Arrays of binary parameters.
        * vals[0] : mtot [grams]
        * vals[1] : mrat []
        * vals[2] : redz []
        * vals[3] : fobs [1/sec]
    weights : (N,) array of scalar,
    fobs : (F,) array of scalar,
        Target observer-frame frequencies to calculate GWs at.  Units of [1/sec].

    """
    hs, fo = _strains_from_samples(vals)
    gff, gwf, gwb = gws_from_sampled_strains(fobs, fo, hs, weights)
    return gff, gwf, gwb


@numba.njit
def gws_from_sampled_strains(fobs, fo, hs, weights):
    """Calculate GW background/foreground from sampled GW strains.

    Parameters
    ----------
    fobs : (F,) array_like of scalar
        Frequency bins.
    fo : (S,) array_like of scalar
        Observed GW frequency of each binary sample.
    hs : (S,) array_like of scalar
        GW source strain (*not characteristic strain*) of each binary sample.
    weights : (S,) array_like of int
        Weighting factor for each binary.
        NOTE: the GW calculation is ill-defined if weights have fractional values
        (i.e. float values, instead of integral values; but the type itself doesn't matter)

    Returns
    -------
    gwf_freqs : (F,) ndarray of scalar
        GW frequency of foreground sources in each frequency bin.
    gwfore : (F,) ndarray of scalar
        Strain amplitude of foreground sources in each frequency bin.
    gwback : (F,) ndarray of scalar
        Strain amplitude of the background in each frequency bin.

    """

    # ---- Initialize
    num_samp = fo.size                 # number of binaries/samples
    num_freq = fobs.size - 1           # number of frequency bins (edges - 1)
    gwback = np.zeros(num_freq)        # store GWB characteristic strain
    gwfore = np.zeros(num_freq)        # store loudest binary characteristic strain, for each bin
    gwf_freqs = np.zeros(num_freq)     # store frequency of loudest binary, for each bin

    # ---- Sort input by frequency for faster iteration
    idx = np.argsort(fo)
    fo = np.copy(fo)[idx]
    hs = np.copy(hs)[idx]
    weights = np.copy(weights)[idx]

    # ---- Calculate GW background and foreground in each frequency bin
    ii = 0
    lo = fobs[ii]
    for ff in range(num_freq):
        # upper-bound to this frequency bin
        hi = fobs[ff+1]
        # number of GW cycles (f/df), for conversion to characteristic strain
        # cycles = 0.5 * (hi + lo) / (hi - lo)
        # cycles = 1.0 / np.diff(np.log([lo, hi]))
        cycles = 1.0 / (np.log(hi) - np.log(lo))
        # amplitude and frequency of the loudest source in this bin
        hmax = 0.0
        fmax = 0.0
        # iterate over all sources with frequencies below this bin's limit (right edge)
        while (fo[ii] < hi) and (ii < num_samp):
            # Store the amplitude and frequency of loudest source
            #    NOTE: loudest source could be a single-sample (weight==1) or from a weighted-bin (weight > 1)
            #          the max
            if (weights[ii] >= 1) and (hs[ii] > hmax):
                hmax = hs[ii]
                fmax = fo[ii]
            # if (weights[ii] > 1.0) and poisson:
            #     h2temp *= np.random.poisson(weights[ii])
            h2temp = weights[ii] * (hs[ii] ** 2)
            gwback[ff] += h2temp

            # increment binary/sample index
            ii += 1

        # subtract foreground source from background
        gwf_freqs[ff] = fmax
        gwback[ff] -= hmax**2
        # Convert to *characteristic* strain
        gwback[ff] = gwback[ff] * cycles      # hs^2 ==> hc^2  (squared, so cycles^1)
        gwfore[ff] = hmax * np.sqrt(cycles)   # hs ==> hc (not squared, so sqrt of cycles)
        lo = hi

    gwback = np.sqrt(gwback)
    return gwf_freqs, gwfore, gwback


def _gws_from_number_grid_centroids(edges, dnum, number, realize, integrate=True):
    """Calculate GWs based on a grid of number-of-binaries.

    NOTE: `_gws_from_number_grid_integrated()` should be more accurate, but this method better
          matches GWB from sampled (kale.sample_) populations!!

    The input number of binaries is `N` s.t.
        ``N = (d^4 N / [dlog10(M) dq dz dlogf] ) * dlog10(M) dq dz dlogf``
    The number `N` is evaluated on a 4d grid, specified by `edges`, i.e.
        ``N = N(M, q, z, f_r)``
    NOTE: the provided `number` must also summed/integrated over dlogf.
    To calculate characteristic strain, this function divides again by the dlogf term.

    Parameters
    ----------
    fobs : _type_
    edges : _type_
    dnum : _type_
    number : _type_

    """

    # # ---- find 'center-of-mass' of each bin (i.e. based on grid edges)
    # # (3, M', Q', Z')
    # # coms = self.grid
    # # ===> (3, M', Q', Z', 1)
    # coms = [cc[..., np.newaxis] for cc in grid]
    # # ===> (4, M', Q', Z', F)
    # coms = np.broadcast_arrays(*coms, fobs[np.newaxis, np.newaxis, np.newaxis, :])

    # # ---- find weighted bin centers
    # # get unweighted centers
    # cent = kale.utils.midpoints(dnum, log=False, axis=(0, 1, 2, 3))
    # # get weighted centers for each dimension
    # for ii, cc in enumerate(coms):
    #     coms[ii] = kale.utils.midpoints(dnum * cc, log=False, axis=(0, 1, 2, 3)) / cent
    # print(f"{kale.utils.jshape(edges)=}, {dnum.shape=}")
    coms = kale.utils.centroids(edges, dnum)

    # ---- calculate GW strain at bin centroids
    mc = utils.chirp_mass(*utils.m1m2_from_mtmr(coms[0], coms[1]))
    dc = cosmo.comoving_distance(coms[2]).cgs.value
    fr = utils.frst_from_fobs(coms[3], coms[2])
    # convert from GW frequency to orbital frequency (divide by 2.0)
    hs = utils.gw_strain_source(mc, dc, fr/2.0)

    dlogf = np.diff(np.log(edges[-1]))
    dlogf = dlogf[np.newaxis, np.newaxis, np.newaxis, :]
    if realize is True:
        number = np.random.poisson(number)
    elif realize in [None, False]:
        pass
    elif utils.isinteger(realize):
        shape = number.shape + (realize,)
        number = np.random.poisson(number[..., np.newaxis], size=shape)
        hs = hs[..., np.newaxis]
        dlogf = dlogf[..., np.newaxis]
    else:
        err = "`realize` ({}) must be one of {{True, False, integer}}!".format(realize)
        raise ValueError(err)

    number = number / dlogf
    hs = np.nan_to_num(hs)
    hc = number * np.square(hs)
    # (M',Q',Z',F) ==> (F,)
    if integrate:
        hc = np.sqrt(np.sum(hc, axis=(0, 1, 2)))

    return hc


def _gws_from_number_grid_integrated(edges, dnum, number, realize, integrate=True):

    grid = np.meshgrid(*edges, indexing='ij')

    # ---- calculate GW strain at bin centroids
    mc = utils.chirp_mass(*utils.m1m2_from_mtmr(grid[0], grid[1]))
    dc = cosmo.comoving_distance(grid[2]).cgs.value
    fr = utils.frst_from_fobs(grid[3], grid[2])
    # convert from GW frequency to orbital frequency (divide by 2.0)
    hs = utils.gw_strain_source(mc, dc, fr/2.0)

    integrand = dnum * (hs ** 2)
    # hc = _integrate_differential_number(edges, integrand, freq=False)
    hc = _integrate_differential_number(edges, integrand, freq=True)
    hc = hc / np.diff(np.log(edges[-1]))[np.newaxis, np.newaxis, np.newaxis, :]

    if realize is True:
        number = np.random.poisson(number) / number
    elif realize in [None, False]:
        pass
    elif utils.isinteger(realize):
        shape = number.shape + (realize,)
        number = np.random.poisson(number[..., np.newaxis], size=shape) / number[..., np.newaxis]
        hc = hc[..., np.newaxis] * np.nan_to_num(number)
    else:
        err = "`realize` ({}) must be one of {{True, False, integer}}!".format(realize)
        raise ValueError(err)

    # (M',Q',Z',F) ==> (F,)
    if integrate:
        hc = np.sqrt(np.sum(hc, axis=(0, 1, 2)))

    return hc
