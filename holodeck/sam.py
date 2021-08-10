"""

Chen, Sesana, Conselice 2019 = [Chen19]
Constraining astrophysical observables of galaxy and supermassive black hole binary mergers using pulsar timing arrays
https://ui.adsabs.harvard.edu/abs/2019MNRAS.488..401C/abstract


To-Do
-----
[ ] Check: pair-fraction `f` is used correctly (as opposed to f-prime, the mass-ratio independent version)
[ ] Check: mass-variable is consistently used (should be *primary* mass, from [Chen19] _not_ *total* mass)


"""

import logging

import numba
import numpy as np
# import scipy as sp

import kalepy as kale

from holodeck import cosmo, utils
from holodeck.constants import GYR, SPLC, MSOL, MPC

# [Chen19] Eq.16 this is `b * M_0`
_MERGER_TIME_MASS = (0.4 / cosmo.h) * 1.0e11   # Msol
_BULGE_MASS_FRAC = 0.615
_MMBULGE_MASS_REF = 1.0e11  # [Msol]
_AGE_UNIVERSE_GYR = cosmo.age(0.0).to('Gyr').value  # [Gyr]  ~ 13.78


class BP_Semi_Analytic:

    def __init__(
        self, mstar_pri=[8.5, 13, 46], mrat=[0.02, 1.0, 50], redz=[0.0, 6.0, 61],
        smf_phi0=None, smf_phi_z=None, log_smf_mass=None, smf_alpha0=None, smf_alpha_z=None,
        pair_frac_rate=None, pair_alpha=None, pair_beta=None, pair_gamma=None,
        merger_time_tau=None, merger_alpha=None, merger_beta=None, merger_gamma=None,
        log_mmbulge_mstar=None, mmbulge_alpha=None
    ):
        """
        galaxy stellar mass function
            smf_phi0, smf_phi_z: galaxy stellar mass function renormalisation rate
            log_smf_mass: log10 of scale mass
            smf_alpha0, smf_alpha_z: galaxy stellar mass function slope
        pair fraction:
            pair_frac_rate : rate
            pair_alpha : mass power law
            pair_beta  : redshift power law
            pair_gamma : mass ratio power law
        merger time scale:
            merger_time_tau : time scale,
            merger_alpha : mass power law,
            merger_beta : redshift power law,
            merger_gamma : mass ratio power law
        Mbh-Mstar Relation
            log_mmbulge_mstar :
            mmbulge_alpha :

        [-2.8,-0.2,11.25,-1.25,0.,0.025,0.,0.8,0.,1.,0.,-0.5,0.,8.25,1.]#,0.4,0.5,0.]

        """

        self.mstar_pri = np.logspace(*mstar_pri)
        self.mrat = np.linspace(*mrat)
        self.redz = np.linspace(*redz)

        # ---- Default Parameters
        # See: [Chen19] Table 1

        # galaxy stellar mass function
        if smf_phi0 is None:
            smf_phi0 = -2.77        # +/- [-0.29, +0.27]
        if smf_phi_z is None:
            smf_phi_z = -0.27       # +/- [-0.21, +0.23]
        if log_smf_mass is None:
            log_smf_mass = 11.24    # +/- [-0.17, +0.20]
        if smf_alpha0 is None:
            smf_alpha0 = -1.24      # +/- [-0.16, +0.16]
        if smf_alpha_z is None:
            smf_alpha_z = -0.03     # +/- [-0.14, +0.16]

        # pair fraction
        if pair_frac_rate is None:
            pair_frac_rate = 0.025  # b/t [0.02, 0.03]   [0.01, 0.05]
        if pair_alpha is None:
            pair_alpha = 0.0        # b/t [-0.2, +0.2]   [-0.5, +0.5]
        if pair_beta is None:
            pair_beta = 0.8         # b/t [0.6, 0.1]  [0.0, 2.0]
        if pair_gamma is None:
            pair_gamma = 0.0        # b/t [-0.2, +0.2]  [-0.2, +0.2]

        # merger time
        if merger_time_tau is None:
            merger_time_tau = 0.55      # b/t [0.1, 2.0]  [0.1, 10.0]
        if merger_alpha is None:
            merger_alpha = 0.0      # b/t [-0.2, +0.2]  [-0.2, +0.2]
        if merger_beta is None:
            merger_beta = -0.5      # b/t [-2.0, +1.0]  [-3.0, +1.0]
        if merger_gamma is None:
            merger_gamma = 0.0      # b/t [-0.2, +0.2]  [-0.2, +0.2]

        # Mbh--Mbulge
        if log_mmbulge_mstar is None:
            log_mmbulge_mstar = 8.17  # +/- [-0.32, +0.35]
        if mmbulge_alpha is None:
            mmbulge_alpha = 1.01      # +/- [-0.10, +0.08]

        # ----

        # galaxy stellar-mass function
        self.smf_phi0 = smf_phi0
        self.smf_phi_z = smf_phi_z
        self.smf_mass = 10.0 ** log_smf_mass    # `M_0` in [Chen19]
        self.smf_alpha0 = smf_alpha0
        self.smf_alpha_z = smf_alpha_z

        # galaxy pair fractions
        #     galaxy pair fraction normalization
        pow = pair_gamma + 1.0
        pair_norm = (self.mrat[-1]**pow - self.mrat[0]**pow) / pow
        pair_frac_rate = pair_frac_rate / pair_norm
        self.pair_frac_rate = pair_frac_rate    # `f_0'` in [Chen19]
        self.pair_alpha = pair_alpha
        self.pair_beta = pair_beta
        self.pair_gamma = pair_gamma

        # merger time-scale
        self.merger_time_tau = merger_time_tau    # `tau_0`  in [Chen19]
        self.merger_alpha = merger_alpha  # `alpha_tau`  in [Chen19]
        self.merger_beta = merger_beta    # `beta_tau`  in [Chen19]
        self.merger_gamma = merger_gamma  # `gamma_tau`  in [Chen19]

        # MBH-Mstar relation
        self.mmbulge_mstar = 10.0 ** log_mmbulge_mstar
        self.mmbulge_alpha = mmbulge_alpha

        # ---- Derived quantities

        self._beta_eff = pair_beta - merger_beta
        self._gamma_eff = pair_gamma - merger_gamma

        self.mstar_sec = self.mstar_pri[:, np.newaxis] * self.mrat[np.newaxis, :]

        def mbh_from_mstar(mstar):
            mbulge = mstar * _BULGE_MASS_FRAC
            mbh = SAM.mbh_from_mbulge(mbulge, self.mmbulge_mstar, self.mmbulge_alpha)
            return mbh

        self.mbh1 = np.log10(mbh_from_mstar(self.mstar_pri))
        self.mbh2 = np.log10(mbh_from_mstar(self.mstar_sec))
        self.mchirp = utils.chirp_mass(10.0**self.mbh1[:, np.newaxis], 10.0**self.mbh2)

        def grid_space(vals):
            delta = (vals.max() - vals.min()) / (vals.size - 1.0) / 2.0
            return delta

        self.mbh1_delta = grid_space(self.mbh1)
        self.mrat_delta = grid_space(self.mrat)
        self.redz_delta = grid_space(self.redz)

        self._edges = [10.0**self.mbh1, self.mrat, self.redz]
        self._dnbh = None
        self._dc_mpc = None
        return

    @property
    def _dist_com_mpc(self):
        """Comoving distance [Mpc] to `self.redz` redshift values.
        """
        if self._dc_mpc is None:
            zz = self.redz
            self._dc_mpc = cosmo.comoving_distance(zz).to('Mpc').value

        return self._dc_mpc

    @property
    def edges(self):
        return self._edges

    def merger_time(self, mass, mrat, redz):
        """
        """
        tau = SAM.merger_time(
            mass, mrat, redz,
            self.merger_time_tau, self.merger_alpha, self.merger_beta, self.merger_gamma
        )
        return tau

    def zprime(self, mass, mrat, redz):
        """
        redshift condition, need to improve cut at the age of the universe
        """
        tau0 = self.merger_time(mass, mrat, redz)
        age = cosmo.age(redz).to('Gyr').value
        new_age = age + tau0
        # see if this is a scalar
        try:
            if new_age < _AGE_UNIVERSE_GYR:
                redz_prime = cosmo.tage_to_z(new_age * GYR)
            else:
                redz_prime = -1

        # handle it as a vector
        except ValueError:
            redz_prime = -1.0 * np.ones_like(new_age)
            idx = (new_age < _AGE_UNIVERSE_GYR)
            redz_prime[idx] = cosmo.tage_to_z(new_age[idx] * GYR)

        return redz_prime

    def _dngal(self, mstar, mrat, redz):
        """Galaxy merger rate distribution.

        [Chen19] Eq.17 & Eq.18

        Arguments
        ---------
        mstar : galaxy stellar mass [Msol]
        mrat : galaxy merger mass-ratio
        redz : redshift


        # d3n/dM1dqdz from parameters, missing b^merger_alpha/a^pair_alpha
        # b = 0.4*h^-1, a = 1
        """
        phi0 = 10.0 ** (self.smf_phi0 + self.smf_phi_z * redz)    # [Chen19] Eq.9
        alpha0 = self.smf_alpha0 + self.smf_alpha_z * redz        # [Chen19] Eq.11
        alpha_eff = alpha0 + self.pair_alpha - self.merger_alpha   # [Chen19] Eq.18

        # [Chen19] Eq.18  ---  this should be [Mpc^-3 Msol^-1 Gyr^-1]
        neff = phi0 * self.pair_frac_rate / self.smf_mass / self.merger_time_tau
        neff *= (0.4 / cosmo.h) ** self.merger_alpha
        neff *= (1.0e11 / self.smf_mass) ** (self.merger_alpha - self.pair_alpha)

        mterm = ((mstar / self.smf_mass) ** alpha_eff) * np.exp(-mstar / self.smf_mass)
        # This has units of time, use [Gyr] to cancel out with `neff` term
        zterm = ((1.0 + redz) ** self._beta_eff) * cosmo.dtdz(redz) / GYR

        # Ends up as [Mpc^-3 Msol^-3], correct for dn/[dzdMdq]
        rv = neff * mterm * (mrat ** self._gamma_eff) * zterm
        return rv

    def _dnbh_dlog10m(self, mstar, mrat, redz):
        """

        This is   `d^3 n_BH / [dlog_10(m) dq dz]`

        [Chen19] Eq.21
        """

        mbulge = mstar * _BULGE_MASS_FRAC
        dmbulge_dmstar = _BULGE_MASS_FRAC
        dqgal_dqbh = 1.0     # conversion from galaxy mrat to MBH mrat
        dn_gal = self._dngal(mstar, mrat, redz)   # galaxy merger rate   [Mpc^-3 Msol^-1]
        # convert from 1/dm to 1/dlog10(m)
        mterm = SAM.mbh_from_mbulge(mbulge, self.mmbulge_mstar, self.mmbulge_alpha) * np.log(10.0)  # [Msol]
        # first get (dmbh/dmstar) = (dmbh/dmbulge) * (dmbulge/dmstar)
        dmstar_dmbh = SAM.dmbh_dmbulge(mbulge, self.mmbulge_mstar, self.mmbulge_alpha) * dmbulge_dmstar
        # invert to dmstar / dmbh
        dmstar_dmbh = 1.0 / dmstar_dmbh     # [unitless]

        # Eq.21, now [Mpc^-3], lose Msol^-1 because now 1/dlog10(M) instead of 1/dM
        rv = dn_gal * dmstar_dmbh * dqgal_dqbh * mterm
        return rv

    def dnbh(self):
        """
        This is `dnbh / [dlog10m dq dz]` multiplied by dq dlog10m, i.e.
        `dnbh / dz` for each 3D bin (m, mrat, redz), in units of [Mpc^-3]

        input 3 x 1d array M1,mrat,redz
        output 3d array (M1,mrat,redz) (galaxy mass, galaxy mass ratio, redshift) of values for function

        """
        if self._dnbh is None:

            shape = (len(self.mstar_pri), len(self.mrat), len(self.redz))
            dnbh = np.zeros(shape)
            # factor of 4 is because `delta` values are half of bin widths; [unitless]
            #     unitless because `mbh1_delta` is Delta log10(M) (i.e. unitless)
            bin_vol = 4.0 * self.mbh1_delta * self.mrat_delta

            # Convert from (N,) (M,) (L,) ==> (N,1,1) (1,M,1) (1,1,L)
            args = utils.broadcastable(self.mstar_pri, self.mrat, self.redz)
            # (N,M,L)
            zprime = self.zprime(*args)
            # find valid entries
            idx = (zprime > 0.0)

            # Broadcast arrays [e.g. (1,M,1) ==> (N,M,L)] and select valid entries
            args = [aa[idx] for aa in utils.expand_broadcastable(*args)]
            # Calculate number densities at valid entries
            dnbh[idx] = self._dnbh_dlog10m(*args)
            # multiply by bin sizes
            dnbh *= bin_vol

            self._dnbh = dnbh

        return self._dnbh

    def num_mbhb(self, fobs):
        """Convert from number-density to finite Number, per log-frequency interval

        d N / d ln f_r = (dn/dz) * (dz/dt) * (dt/d ln f_r) * (dVc/dz)
                       = (dn/dz) * (f_r / [df_r/dt]) * 4 pi c D_c^2 (1+z) * dz

        """
        edges_fobs = self.edges + [fobs, ]

        # shape: (m1, mrat, redz)
        dnbh = self.dnbh()   # dn/dz  units: [Mpc^-3]

        dc = self._dist_com_mpc    # comoving-distance in Mpc
        dz = self.redz_delta * 2.0    # `delta` is half of bin-width, so multiply by 2
        # print(f"dc={utils.stats(dc)}")
        # [Mpc^3/s]
        cosmo_fact = 4 * np.pi * (SPLC/MPC) * np.square(dc) * dz
        # print(f"cosmo_fact={utils.stats(cosmo_fact)}")

        # (m1, q)
        mchirp = self.mchirp * MSOL
        # (m1, q, 1, 1)
        mchirp = mchirp[..., np.newaxis, np.newaxis]
        # print(f"mchirp/Msol={utils.stats(mchirp/MSOL)}")
        # (z, f)
        frst = fobs[np.newaxis, :] * (1.0 + self.redz[:, np.newaxis])
        # (1, 1, z, f)
        frst = frst[np.newaxis, np.newaxis, :, :]
        # print(f"frst*yr={utils.stats(frst*YR)}")
        # (m1, q, z, f)
        tau = utils.gw_hardening_timescale(mchirp, frst)
        # print(f"tau/GYR={utils.stats(tau/GYR)}")

        TAU_LIMIT = None
        TAU_LIMIT = 2.0
        if (TAU_LIMIT is not None):
            logging.warning(f"WARNING: limiting tau to < {TAU_LIMIT:.2f} Gyr")
            bads = (tau/GYR > 2.0)
            tau[bads] = 0.0
            print(f"tau/GYR={utils.stats(tau/GYR)}, bads={np.count_nonzero(bads)/bads.size:.2e}")

        dl = dc * (1.0 + self.redz) * MPC
        dl = dl[np.newaxis, np.newaxis, :, np.newaxis]
        dl[dl <= 0.0] = np.nan
        hs_mbhb = utils.gw_strain_source(mchirp, dl, frst)
        # hs[dl <= 0.0] = 0.0
        hs_mbhb = np.nan_to_num(hs_mbhb)

        # (m1, q, z)
        num_mbhb = dnbh * cosmo_fact
        # (m1, q, z, f)
        num_mbhb = num_mbhb[..., np.newaxis] * tau
        return edges_fobs, num_mbhb, hs_mbhb


class SAM:

    def __init__(self):
        pass

    @classmethod
    def mbh_from_mbulge(cls, mbulge, amp, pow):
        """
        mass of the black hole Mstar-Mbulge relation without scattering
        """
        NORM = _MMBULGE_MASS_REF   # [Msol]
        mbh = scaling(amp, NORM, pow, mbulge)
        return mbh

    @classmethod
    def dmbh_dmbulge(cls, mbulge, amp, pow):
        """
        dMBH/dMG
        """
        NORM = _MMBULGE_MASS_REF   # [Msol]
        rv = scaling(amp, NORM, pow, mbulge)
        rv *= pow / mbulge
        return rv

    @classmethod
    def merger_time(cls, mass, redz, mrat, amp_time, alpha, beta, gamma):
        """
        tau - merger time scale
        [Chen19] Eq.16
        """
        mterm = (mass / _MERGER_TIME_MASS) ** alpha
        zterm = (1.0 + redz) ** beta
        qterm = mrat ** gamma
        return amp_time * mterm * zterm * qterm


def scaling(amp, norm, power, value):
    """General power-law scaling relation.

    Y = A * (X/X0) ^ g

    """
    return amp * np.power(value / norm, power)


def gwb_continuous(sam, freqs):
    """

    Arguments
    ---------
    freqs : (F,) array_like of scalar,
        Target frequencies of interest in units of [Hz] = [1/s]

    """
    freqs = np.atleast_1d(freqs)
    assert freqs.ndim == 1, "Invalid `freqs` given!  shape={}".format(np.shape(freqs))

    # shape: (m1, mrat, redz)
    dnbh = sam.dnbh()
    zz = sam.redz

    cosmo_fact = cosmo.comoving_distance(zz).to('Mpc').value

    dz = sam.redz_delta * 2.0    # `delta` is half of bin-width, so multiply by 2
    # Now [Mpc^3 / s]
    cosmo_fact = 4*np.pi*(SPLC/MPC) * np.square(cosmo_fact) * dz

    m1 = np.power(10.0, sam.mbh1[np.newaxis, :, np.newaxis]) * MSOL
    m2 = np.power(10.0, sam.mbh2[np.newaxis, :, :]) * MSOL
    fr = freqs[:, np.newaxis, np.newaxis, np.newaxis] / (1.0 + zz[np.newaxis, np.newaxis, np.newaxis, :])
    fterm = utils.gw_hardening_rate_dfdt(m1[..., np.newaxis], m2[..., np.newaxis], fr)
    # f / (df/dt)   [s]
    fterm = fr / fterm

    mc = sam.mchirp[np.newaxis, :, :, np.newaxis] * MSOL
    dl = cosmo.luminosity_distance(zz).cgs.value[np.newaxis, np.newaxis, np.newaxis, :]

    # [-] dimensionless strain
    hs = utils.gw_strain_source(mc, dl, fr)
    gwb = dnbh * (hs**2) * cosmo_fact * fterm
    gwb[:, :, :, 0] = 0.0
    return gwb


def gwb_discrete(sam, freqs, sample_threshold=10.0, cut_below_mass=1e6):
    """

    Arguments
    ---------
    freqs : (F,) array_like of scalar,
        Target frequencies of interest in units of [Hz] = [1/s]

    """
    edges_fobs, num_mbhb_fobs, _ = sam.num_mbhb(freqs)
    log_edges_fobs = [np.log10(edges_fobs[0]), edges_fobs[1], edges_fobs[2], np.log10(edges_fobs[3])]

    if cut_below_mass is not None:
        m2 = edges_fobs[0][:, np.newaxis] * edges_fobs[1][np.newaxis, :]
        bads = (m2 < cut_below_mass)
        num_mbhb_fobs[bads] = 0.0

    vals, weights = kale.sample_outliers(log_edges_fobs, num_mbhb_fobs, sample_threshold)

    if cut_below_mass is not None:
        bads = ((10.0 ** vals[0]) * vals[1] < cut_below_mass)
        vals = vals.T[~bads].T
        weights = weights[~bads]

    vals[0] = 10.0 ** vals[0]
    vals[1] = vals[1] * vals[0]
    mc = utils.chirp_mass(vals[0], vals[1])
    dl = vals[2, :]
    frst = (10.0 ** vals[3]) * (1.0 + dl)
    dl = cosmo.luminosity_distance(dl).cgs.value
    hs = utils.gw_strain_source(mc * MSOL, dl, frst)
    fo = vals[-1, :]
    del vals

    gff, gwf, gwb = gws_from_sampled_strains(log_edges_fobs[-1], fo, hs, weights)
    gff = (10.0 ** gff)

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
