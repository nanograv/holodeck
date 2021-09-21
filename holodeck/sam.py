"""

Chen, Sesana, Conselice 2019 = [Chen19]
Constraining astrophysical observables of galaxy and supermassive black hole binary mergers using pulsar timing arrays
https://ui.adsabs.harvard.edu/abs/2019MNRAS.488..401C/abstract


To-Do
-----
* [ ] Check that _GW_ frequencies and _orbital_ frequencies are being used in the correct places.
    Check `number_at_gw_fobs` and related methods.
* [ ] Expand SAM mass-ratios to wider range, change to log-space.

"""

import abc
import inspect

import numba
import numpy as np

import kalepy as kale

import holodeck as holo
from holodeck import cosmo, utils, log
from holodeck.constants import GYR, SPLC, MSOL, MPC

_AGE_UNIVERSE_GYR = cosmo.age(0.0).to('Gyr').value  # [Gyr]  ~ 13.78


class _Galaxy_Stellar_Mass_Function(abc.ABC):

    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        return

    @abc.abstractmethod
    def __call__(self, mstar, redz):
        """Return the number-density of galaxies at a given stellar mass, per log10 interval of stellar-mass.

        i.e. Phi = dn / dlog10(M)

        Arguments
        ---------
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
        self._phi0 = phi0         # - 2.77  +/- [-0.29, +0.27]
        self._phiz = phiz         # - 0.27  +/- [-0.21, +0.23]
        self._mref0 = mref0       # +11.24  +/- [-0.17, +0.20]  Msol
        self._mrefz = mrefz
        self._alpha0 = alpha0     # -1.24   +/- [-0.16, +0.16]
        self._alphaz = alphaz     # -0.03   +/- [-0.14, +0.16]
        return

    def __call__(self, mstar, redz):
        """Return the number-density of galaxies at a given stellar mass.

        See: [Chen19] Eq.8

        Arguments
        ---------
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

    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        return

    @abc.abstractmethod
    def __call__(self, mtot, mrat, redz):
        """Return the fraction of galaxies in pairs of the given parameters.

        Arguments
        ---------
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

        Arguments
        ---------
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

    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        return

    @abc.abstractmethod
    def __call__(self, mtot, mrat, redz):
        """Return the galaxy merger time for the given parameters.

        Arguments
        ---------
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
    """
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

        Arguments
        ---------
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
        # convert to primary mass
        mpri = utils.m1m2_from_mtmr(mtot, mrat)[0]   # [grams]
        tau0 = self._time_norm                       # [sec]
        bm0 = self._mref                             # [grams]
        aa = self._malpha
        bb = self._zbeta
        gg = self._qgamma
        mtime = tau0 * np.power(mpri/bm0, aa) * np.power(1.0 + redz, bb) * np.power(mrat, gg)
        mtime = mtime * GYR
        return mtime


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


class Semi_Analytic_Model:

    def __init__(
        self, mtot=[2.75e5*MSOL, 1.0e11*MSOL, 46], mrat=[0.02, 1.0, 50], redz=[0.0, 6.0, 61],
        gsmf=GSMF_Schechter, gpf=GPF_Power_Law, gmt=GMT_Power_Law, mmbulge=MMBulge_Simple
    ):

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
        elif not isinstance(mmbulge, _MMBulge_Relation):
            raise ValueError("`mmbulge` must be an instance or subclass of `_MMBulge_Relation`!")

        self.mtot = np.logspace(*np.log10(mtot[:2]), mtot[2])
        self.mrat = np.linspace(*mrat)
        self.redz = np.linspace(*redz)

        # self._dlog10m = np.diff(np.log10(self.mtot))[0]
        # self._dq = np.diff(self.mrat)[0]
        # self._dz = np.diff(self.redz)[0]

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
        # total-mass, mass-ratio ==> (M1, M2)
        masses = utils.m1m2_from_mtmr(self.mtot[:, np.newaxis], self.mrat[np.newaxis, :])
        # BH-masses to stellar-masses
        masses = self._mmbulge.mstar_from_mbh(masses)
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
            # convert `mstar_tot` into total mass: M = m1 + m2
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
            # NOTE: there is a `1/log(10)` here in the formalism, but it cancels with another one below  {*1}
            # `gsmf` returns [1/Mpc^3]   `dtdz` returns [sec]
            dens[idx] = self._gsmf(mstar_pri, redz) * self._gpf(mstar_tot, mstar_rat, redz) * cosmo.dtdz(redz)
            # `gmt` returns [sec]
            dens[idx] /= self._gmt(mstar_tot, mstar_rat, redz) * mstar_tot
            # now `dens` has units of [1/Mpc^3]

            # ---- Convert to MBH Binary density

            dqgal_dqbh = 1.0     # conversion from galaxy mrat to MBH mrat
            # convert from 1/dm to 1/dlog10(m)
            # NOTE: there is a `log(10)` in the formalism here, but it cancels with another one above  {*1}
            mterm = self._mmbulge.mbh_from_mstar(mstar_tot)  # [gram]
            # dMs/dMbh
            dmstar_dmbh = 1.0 / self._mmbulge.dmbh_dmstar(mstar_tot)   # [unitless]

            # Eq.21, now [Mpc^-3], lose [1/gram] because now 1/dlog10(M) instead of 1/dM
            dens[idx] *= dqgal_dqbh * dmstar_dmbh * mterm

            # multiply by bin sizes to convert dn/dMdqdz ==> dn/dz
            # dens *= self._dlog10m * self._dq
            self._density = dens

        return self._density

    def number_at_gw_fobs(self, fobs, limit_merger_time=None):
        """Convert from number-density to finite Number, per log-frequency interval

        Arguments
        ---------
        fobs : observed frequency in [1/s]

        d N / d ln f_r = (dn/dz) * (dz/dt) * (dt/d ln f_r) * (dVc/dz)
                       = (dn/dz) * (f_r / [df_r/dt]) * 4 pi c D_c^2 (1+z) * dz

        """
        edges = self.edges + [fobs, ]

        # shape: (M, Q, Z)
        dens = self.density   # dn/dz  units: [Mpc^-3]

        # (Z,) comoving-distance in Mpc
        dc = cosmo.comoving_distance(self.redz).to('Mpc').value

        # [Mpc^3/s]
        cosmo_fact = 4 * np.pi * (SPLC/MPC) * np.square(dc) * self._dz

        # (M, Q)
        mchirp = utils.m1m2_from_mtmr(self.mtot[:, np.newaxis], self.mrat[np.newaxis, :])
        mchirp = utils.chirp_mass(*mchirp)
        # (M, Q, 1, 1)
        mchirp = mchirp[..., np.newaxis, np.newaxis]
        # (Z, F) find rest-frame frequencies in Hz [1/sec]
        frst = fobs[np.newaxis, :] * (1.0 + self.redz[:, np.newaxis])
        # (1, 1, Z, F)
        frst = frst[np.newaxis, np.newaxis, :, :]
        # (M, Q, Z, F)
        tau = utils.gw_hardening_timescale(mchirp, frst)

        # ---------------------
        if (limit_merger_time is True):
            log.warning("limiting tau to < galaxy merger time")
            mstar = self.mass_stellar()[:, :, :, np.newaxis]
            ms_rat = mstar[1] / mstar[0]
            mstar = mstar.sum(axis=0)   # total mass [gram]
            gmt = self._gmt(mstar, ms_rat, self.redz[np.newaxis, np.newaxis, :])  # [sec]
            bads = (tau > gmt[..., np.newaxis])
            tau[bads] = 0.0
            log.info(f"tau/GYR={utils.stats(tau/GYR)}, bads={np.count_nonzero(bads)/bads.size:.2e}")

        elif (limit_merger_time not in [None, False]):
            log.warning(f"limiting tau to < {limit_merger_time/GYR:.2f} Gyr")
            bads = (tau > limit_merger_time)
            tau[bads] = 0.0
            log.info(f"tau/GYR={utils.stats(tau/GYR)}, bads={np.count_nonzero(bads)/bads.size:.2e}")

        # luminosity distance in [cm]
        dl = dc * (1.0 + self.redz) * MPC     # `dc` is in [Mpc]
        dl = dl[np.newaxis, np.newaxis, :, np.newaxis]
        dl[dl <= 0.0] = np.nan
        strain = utils.gw_strain_source(mchirp, dl, frst)
        strain = np.nan_to_num(strain)

        # (M, Q, Z) units: [1/s] i.e. number per second
        number = dens * cosmo_fact
        # (M, Q, Z, F) units: [] unitless, i.e. number
        number = number[..., np.newaxis] * tau

        return edges, number, strain

    def number_from_hardening(self, hard, fobs=None, sepa=None):
        """Convert from number-density to finite Number, per log-frequency interval

        Arguments
        ---------
        fobs : observed frequency in [1/yr]
        sepa : orbital separation in [pc]

        d N / d ln f_r = (dn/dz) * (dz/dt) * (dt/d ln f_r) * (dVc/dz)
                       = (dn/dz) * (f_r / [df_r/dt]) * 4 pi c D_c^2 (1+z) * dz

        d N / d ln a   = (dn/dz) * (dz/dt) * (dt/d ln a) * (dVc/dz)
                       = (dn/dz) * (a / [da/dt]) * 4 pi c D_c^2 (1+z) * dz

        """
        if (fobs is None) == (sepa is None):
            utils.error("one (and only one) of `fobs` or `sepa` must be provided!")

        if fobs is not None:
            xsize = len(fobs)
            edges = self.edges + [fobs, ]
        else:
            xsize = len(sepa)
            edges = self.edges + [sepa, ]

        # shape: (M, Q, Z)
        dens = self.density   # d3n/[dz dlog10(M) dq]  units: [Mpc^-3]

        # (Z,) comoving-distance in Mpc
        dc = cosmo.comoving_distance(self.redz).to('Mpc').value

        # [Mpc^3/s] this is `(dVc/dz) * (dz/dt)`
        cosmo_fact = 4 * np.pi * (SPLC/MPC) * np.square(dc) * (1.0 + self.redz)

        # (M, Q)
        mchirp = utils.m1m2_from_mtmr(self.mtot[:, np.newaxis], self.mrat[np.newaxis, :])
        mchirp = utils.chirp_mass(*mchirp) * MSOL   # convert to [grams]
        # (M, Q, 1, 1)
        mchirp = mchirp[..., np.newaxis, np.newaxis]

        mt, mr, rz = [gg.ravel() for gg in self.grid]
        mt = mt * MSOL

        if fobs is not None:
            # Convert from obs-GW freq, to rest-frame orbital freq
            fr = fobs[:, np.newaxis] * (1.0 + rz[np.newaxis, :]) / YR / 2.0
            sa = utils.kepler_sepa_from_freq(mt[np.newaxis, :], fr)
        else:
            sa = sepa[:, np.newaxis] * PC
            # NOTE: `fr` is the _orbital_ frequency (not GW)
            fr = utils.kepler_freq_from_sepa(mt[np.newaxis, :], sa)

        # recall: these are negative (decreasing separation)
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
            log.info("limiting tau to < galaxy merger time")
            mstar = self.mass_stellar()[:, :, :, np.newaxis]
            ms_rat = mstar[1] / mstar[0]
            mstar = mstar.sum(axis=0)   # total mass
            gmt = self._gmt(mstar, ms_rat, self.redz[np.newaxis, np.newaxis, :])  # [sec]
            bads = (tau > gmt[..., np.newaxis])
            tau[bads] = 0.0
            log.info(f"tau/GYR={utils.stats(tau/GYR)}, bads={np.count_nonzero(bads)/bads.size:.2e}")

        elif (limit_merger_time in [None, False]):
            pass

        elif utils.isnumeric(limit_merger_time):
            log.info(f"limiting tau to < {limit_merger_time:.2f} Gyr")
            bads = (tau/GYR > limit_merger_time)
            tau[bads] = 0.0
            log.info(f"tau/GYR={utils.stats(tau/GYR)}, bads={np.count_nonzero(bads)/bads.size:.2e}")

        else:
            err = f"`limit_merger_time` ({type(limit_merger_time)}) must be boolean or scalar!"
            utils.error(err)

        # convert `tau` to the correct shape, note that moveaxis MUST happen _before_ reshape!
        # (F, M*Q*Z) ==> (M*Q*Z, F)
        tau = np.moveaxis(tau, 0, -1)
        fr = np.moveaxis(fr, 0, -1)
        # (M*Q*Z, F) ==> (M, Q, Z, F)
        tau = tau.reshape(dens.shape + (xsize,))
        fr = fr.reshape(dens.shape + (xsize,))

        dc[dc <= 0.0] = np.nan
        dc = dc[np.newaxis, np.newaxis, :, np.newaxis]
        # Note: `gw_strain_source` uses *orbital* frequency
        # strain = utils.gw_strain_source(mchirp, dl, fr)
        strain = utils.gw_strain_source(mchirp, dc * MPC, fr)
        strain = np.nan_to_num(strain)

        # (M, Q, Z) units: [1/s] i.e. number per second
        number = dens * cosmo_fact
        # (M, Q, Z, F) units: [] unitless, i.e. number
        number = number[..., np.newaxis] * tau

        return edges, number, strain

    def gwb(self, fobs, hard=holo.evolution.Hard_GW, realize=False, **kwargs):
        """
        Arguments
        ---------
        fobs : units of [1/s]

        """

        squeeze = False
        if np.isscalar(fobs):
            fobs = np.atleast_1d(fobs)
            squeeze = True

        # `num` has shape (M, Q, Z, F)  for mass, mass-ratio, redshift, frequency
        # `num` is dN/d ln f_r = dN/d ln f
        edges, num, hs = self.number_from_hardening(hard, fobs=fobs)

        if realize is True:
            num = np.random.poisson(num)
        elif realize in [None, False]:
            pass
        elif utils.isinteger(realize):
            shape = num.shape + (realize,)
            num = np.random.poisson(num[..., np.newaxis], size=shape)
            hs = hs[..., np.newaxis]
        else:
            err = "`realize` ({}) must be one of {{True, False, integer}}!".format(realize)
            raise ValueError(err)

        hs = np.sqrt(np.sum(num*np.square(hs), axis=(0, 1, 2)))
        if squeeze:
            hs = hs.squeeze()
        return hs


def sample_sam(sam, fobs, sample_threshold=10.0, cut_below_mass=1e6*MSOL, limit_merger_time=2.0*GYR):
    """
    fobs in units of [1/sec]
    """
    # edges: Mtot [gram], mrat (q), redz (z), fobs (f) [1/sec]
    edges, number, _ = sam.number_at_gw_fobs(fobs, limit_merger_time=limit_merger_time)
    log_edges = [np.log10(edges[0]), edges[1], edges[2], np.log10(edges[3])]

    if cut_below_mass is not None:
        m2 = edges[0][:, np.newaxis] * edges[1][np.newaxis, :]
        bads = (m2 < cut_below_mass)
        number[bads] = 0.0

    vals, weights = kale.sample_outliers(log_edges, number, sample_threshold)
    vals[0] = 10.0 ** vals[0]
    vals[3] = 10.0 ** vals[3]

    if cut_below_mass is not None:
        bads = (vals[0] * vals[1] < cut_below_mass)
        vals = vals.T[~bads].T
        weights = weights[~bads]

    return vals, weights


def sample_sam_with_hardening(sam, hard, fobs=None, sepa=None, sample_threshold=10.0, cut_below_mass=1e6):
    """
    fobs in units of [1/yr]
    sepa in units of [pc]
    """

    # edges: Mtot [Msol], mrat (q), redz (z), {fobs (f) [1/yr] OR sepa (a) [pc]}
    edges, number, _ = sam.number_from_hardening(hard, fobs=fobs, sepa=sepa, limit_merger_time=limit_merger_time)
    log_edges = [np.log10(edges[0]), edges[1], edges[2], np.log10(edges[3])]

    if cut_below_mass is not None:
        m2 = edges[0][:, np.newaxis] * edges[1][np.newaxis, :]
        bads = (m2 < cut_below_mass)
        number[bads] = 0.0

    vals, weights = kale.sample_outliers(log_edges, number, sample_threshold)
    vals[0] = 10.0 ** vals[0]
    vals[3] = 10.0 ** vals[3]

    if cut_below_mass is not None:
        bads = (vals[0] * vals[1] < cut_below_mass)
        vals = vals.T[~bads].T
        weights = weights[~bads]

    return vals, weights, number


def _gws_from_samples(vals, weights, fobs):
    """

    Arguments
    ---------
    vals : (4, N) ndarray of scalar,
        Arrays of binary parameters in linear space.
        * vals[0] : mtot [grams]
        * vals[1] : mrat []
        * vals[2] : redz []
        * vals[3] : fobs [1/sec]
    weights : (N,) array of scalar,
    fobs : (F,) array of scalar,
        Target observer-frame frequencies to calculate GWs at.  Units of [1/sec].

    """
    mc = utils.chirp_mass(*utils.m1m2_from_mtmr(vals[0], vals[1]))
    dl = vals[2, :]
    frst = vals[3] * (1.0 + dl)
    dl = cosmo.luminosity_distance(dl).cgs.value
    hs = utils.gw_strain_source(mc, dl, frst)
    fo = vals[-1]
    del vals

    gff, gwf, gwb = gws_from_sampled_strains(fobs, fo, hs, weights)

    return gff, gwf, gwb


def sampled_gws_from_sam(sam, fobs, hard=holo.evolution.Hard_GW, **kwargs):
    """

    Arguments
    ---------
    fobs : (F,) array_like of scalar,
        Target frequencies of interest in units of [1/yr]

    """
    vals, weights = sample_sam_with_hardening(sam, hard, fobs=fobs, **kwargs)
    gff, gwf, gwb = _gws_from_samples(vals, weights, fobs)
    return gff, gwf, gwb


@numba.njit
def gws_from_sampled_strains(freqs, fobs, hs, weights):
    num_samp = fobs.size
    num_freq = freqs.size - 1
    gwback = np.zeros(num_freq)
    gwfore = np.zeros_like(gwback)
    gwf_freqs = np.zeros_like(gwback)

    idx = np.argsort(fobs)
    weights = weights[idx]
    fobs = fobs[idx]
    hs = hs[idx]

    ii = 0
    for ff in range(num_freq):
        hi = freqs[ff+1]
        fmax = 0.0
        hmax = 0.0
        while (fobs[ii] < hi) and (ii < num_samp):
            if hs[ii] > hmax:
                hmax = hs[ii]
                fmax = fobs[ii]
            h2temp = hs[ii] ** 2
            if weights[ii] > 1.0:
                h2temp *= np.random.poisson(weights[ii])
            gwback[ff] += h2temp
            ii += 1

        gwfore[ff] = hmax
        gwf_freqs[ff] = fmax
        gwback[ff] -= hmax**2

    gwback = np.sqrt(gwback)

    return gwf_freqs, gwfore, gwback
