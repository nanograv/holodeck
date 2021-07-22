"""

Chen, Sesana, Conselice 2019 = [Chen19]
Constraining astrophysical observables of galaxy and supermassive black hole binary mergers using pulsar timing arrays
https://ui.adsabs.harvard.edu/abs/2019MNRAS.488..401C/abstract

"""

import numpy as np
import scipy as sp
from holodeck import cosmo, utils
from holodeck.constants import GYR, NWTG, SPLC, MSOL, MPC
from holodeck.evolution import _Binary_Evolution


# [Chen19] Eq.16 this is `b * M_0`
_MERGER_TIME_MASS = (0.4 / cosmo.h) * 1.0e11   # Msol
_BULGE_MASS_FRAC = 0.615

_AGE_UNIVERSE_GYR = cosmo.age(0.0).to('Gyr').value  # [Gyr]  ~ 13.78

'''
def invE(z):
    OmegaM = 0.3
    Omegak = 0.0
    OmegaLambda = 0.7
    return 1.0 / np.sqrt(OmegaM * (1.0+z)**3.0 + Omegak * (1.0+z)**2.0 + OmegaLambda)


def dtdz(z):
    t0 = 14.0
    if z == -1.0:
        z = -0.99
    return t0 / (1.0 + z) * invE(z)
'''

class BP_Semi_Analytic(_Binary_Evolution):

    _SELF_CONSISTENT = False

    def __init__(
        self,
        smf_phi0=None, smf_phi_z=None, log_smf_mass=None, smf_alpha0=None, smf_alpha_z=None,
        pair_frac_rate=None, pair_alpha=None, pair_beta=None, pair_gamma=None,
        merger_time=None, merger_alpha=None, merger_beta=None, merger_gamma=None,
        log_mmrel_mstar=None, mmrel_alpha=None):
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
            merger_time : time scale,
            merger_alpha : mass power law,
            merger_beta : redshift power law,
            merger_gamma : mass ratio power law
        Mbh-Mstar Relation
            log_mmrel_mstar :
            mmrel_alpha :
        """

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
        if merger_time is None:
            merger_time = 0.55      # b/t [0.1, 2.0]  [0.1, 10.0]
        if merger_alpha is None:
            merger_alpha = 0.0      # b/t [-0.2, +0.2]  [-0.2, +0.2]
        if merger_beta is None:
            merger_beta = -0.5      # b/t [-2.0, +1.0]  [-3.0, +1.0]
        if merger_gamma is None:
            merger_gamma = 0.0      # b/t [-0.2, +0.2]  [-0.2, +0.2]

        # Mbh--Mbulge
        if log_mmrel_mstar is None:
            log_mmrel_mstar = 8.17  # +/- [-0.32, +0.35]
        if mmrel_alpha is None:
            mmrel_alpha = 1.01      # +/- [-0.10, +0.08]

        # ----

        # galaxy stellar-mass function
        self.smf_phi0 = smf_phi0
        self.smf_phi_z = smf_phi_z
        self.smf_mass = 10.0 ** log_smf_mass    # `M_0` in [Chen19]
        self.smf_alpha0 = smf_alpha0
        self.smf_alpha_z = smf_alpha_z

        # galaxy pair fractions
        self.pair_frac_rate = pair_frac_rate    # `f_0'` in [Chen19]  -- I think this is the primed version! [CHECK]
        self.pair_alpha = pair_alpha
        self.pair_beta = pair_beta
        self.pair_gamma = pair_gamma

        # merger time-scale
        self.merger_time = merger_time   # `tau_0`  in [Chen19]
        self.merger_alpha = merger_alpha # `alpha_tau`  in [Chen19]
        self.merger_beta = merger_beta   # `beta_tau`  in [Chen19]
        self.merger_gamma = merger_gamma # `gamma_tau`  in [Chen19]

        # MBH-Mstar relation
        self.mmrel_mstar = 10.0 ** log_mmrel_mstar
        self.mmrel_alpha = mmrel_alpha

        # mstar_pri: stellar mass of primary galaxy [Msol]
        # self.mstar_pri = np.logspace(9, 13, 41)
        self.mstar_pri = np.logspace(9, 13, 4)
        # self.mstar_pri_diff = (mstar_pri.max() - mstar_pri.min()) / (len(mstar_pri) - 1.0) / 2.0
        # mrat: mass ratio between galaxies
        # self.mrat = np.linspace(0.1, 1.0, 10)
        self.mrat = np.linspace(0.1, 1.0, 2)
        self.mrat_delta = (self.mrat.max() - self.mrat.min()) / (len(self.mrat) - 1.0) / 2.0
        # redz: redshift
        # self.redz = np.linspace(0.0, 2.0, 21)
        self.redz = np.linspace(0.0, 2.0, 4+1)[1:]
        # self.redz_delta = (redz.max() - redz.min()) / (len(redz) - 1.0) / 2.0

        self._beta_eff = pair_beta - merger_beta
        self._gamma_eff = pair_gamma - merger_gamma
        self.mbh1 = self.log_mbh_from_mbulge(self.mstar_pri)
        self.mbh1_delta = (self.mbh1.max() - self.mbh1.min()) / (len(self.mbh1) - 1.0) / 2.0

        self.mstar_sec = self.mstar_pri[:, np.newaxis] * self.mrat[np.newaxis, :]
        self.mbh2 = self.log_mbh_from_mbulge(self.mstar_sec)
        self.mchirp = utils.chirp_mass(10.0**self.mbh1[:, np.newaxis], 10.0**self.mbh2)
        # Mc, mrat, redshift array lengths
        # self.mergerrate = np.zeros((len(self.mstar_pri), len(self.mrat), len(self.redz)))

        self._dndz = None
        return

    def _init_step_zero(self):
        super()._init_step_zero()
        self.mergerrate = self.output() #for black hole chirp mass: self.grid()
        return

    def _take_next_step(self, step):
        return EVO.END

    def zprime(self, M1, q, zp):
        """
        redshift condition, need to improve cut at the age of the universe
        """
        tau0 = self.tau(M1, q, zp)
        age = cosmo.age(zp).to('Gyr').value
        new_age = age + tau0
        if new_age < _AGE_UNIVERSE_GYR:
            redz = cosmo.tage_to_z(new_age * GYR)
        else:
            redz = -1

        return redz

    def tau(self, mass, mrat, redz):
        """
        merger time scale
        [Chen19] Eq.16
        """
        # m0 = (0.4 / cosmo.h) * 1.0e11   # Msol
        mterm = (mass / _MERGER_TIME_MASS) ** self.merger_alpha
        zterm = (1.0 + redz) ** self.merger_beta
        qterm = mrat ** self.merger_gamma
        return self.merger_time * mterm * zterm * qterm

    def dngal_dzdmdq(self, mstar, qq, zz):
        """Galaxy merger rate distribution.

        [Chen19] Eq.17 & Eq.18

        Arguments
        ---------
        mstar : galaxy stellar mass [Msol]
        qq : galaxy merger mass-ratio
        zz : redshift


        # d3n/dM1dqdz from parameters, missing b^merger_alpha/a^pair_alpha
        # b = 0.4*h^-1, a = 1
        """
        phi0 = 10.0 ** (self.smf_phi0 + self.smf_phi_z * zz)    # [Chen19] Eq.9
        alpha0 = self.smf_alpha0 + self.smf_alpha_z * zz        # [Chen19] Eq.11
        alpha_eff = alpha0 + self.pair_alpha - self.merger_alpha   # [Chen19] Eq.18

        # [Chen19] Eq.18  ---  this should be [Mpc^-3 Msol^-1 Gyr^-1]
        neff = phi0 * self.pair_frac_rate / self.smf_mass / self.merger_time
        neff *= (0.4 / cosmo.h) ** self.merger_alpha
        neff *= (1.0e11 / self.smf_mass) ** (self.merger_alpha - self.pair_alpha)

        # mterm = self.smf_mass * ((mstar / self.smf_mass) ** alpha_eff) * np.exp(-mstar / self.smf_mass)
        # LZK 2021-07-22 : I don't think there should be the extra smf_mass in the above!
        mterm = ((mstar / self.smf_mass) ** alpha_eff) * np.exp(-mstar / self.smf_mass)
        # This has units of time, use [Gyr] to cancel out with `neff` term
        zterm = ((1.0 + zz) ** self._beta_eff) * cosmo.dtdz(zz) / GYR

        # Ends up as [Mpc^-3 Msol^-3], correct for dn/[dzdMdq]
        rv = neff * mterm * (qq ** self._gamma_eff) * zterm
        return rv

    def dmbh_dmgal(self, mstar):
        """
        dMBH/dMG
        """
        rv = (mstar / 1.0e11) ** (self.mmrel_alpha - 1.0)
        rv *= self.mmrel_mstar * self.mmrel_alpha / 1.0e11
        return rv

    def dnbh_dlog10m(self, mstar, q, z):
        """

        This is   `d^3 n_BH / [dlog_10(m) dq dz]`

        [Chen19] Eq.21
        """

        mbulge = mstar * _BULGE_MASS_FRAC
        dmb_dms = _BULGE_MASS_FRAC
        dqgal_dqbh = 1.0     # conversion from galaxy mrat to MBH mrat
        dn_gal = self.dngal_dzdmdq(mstar, q, z)   # galaxy merger rate   [Mpc^-3 Msol^-1]
        # convert from 1/dm to 1/dlog10(m)
        mterm = 10.0 ** self.log_mbh_from_mbulge(mbulge) * np.log(10.0)  # [Msol]
        dm_dmbh = 1.0 / (self.dmbh_dmgal(mbulge) * dmb_dms)     # [unitless]

        # Eq.21, now [Mpc^-3], lose Msol^-1 because now 1/dlog10(M) instead of 1/dM
        rv = dn_gal * dm_dmbh * dqgal_dqbh * mterm
        return rv

    def log_mbh_from_mbulge(self, mass):
        """
        mass of the black hole Mstar-Mbulge relation without scattering
        """
        mbh = np.log10(self.mmrel_mstar * (mass / 1.0e11)**self.mmrel_alpha)
        return mbh

    def dnbh_dz(self):
        """
        This is `dnbh / [dlog10m dq dz]` multiplied by dq dlog10m, i.e.
        `dnbh / dz` for each 3D bin (m, q, z)

        input 3 x 1d array M1,q,z
        output 3d array (M1,q,z) (galaxy mass, galaxy mass ratio, redshift) of values for function

        """
        if self._dndz is None:

            shape = (len(self.mstar_pri), len(self.mrat), len(self.redz))
            dndz = np.zeros(shape)
            # factor of 4 is because `delta` values are half of bin widths; [unitless]
            #     unitless because `mbh1_delta` is Delta log10(M) (i.e. unitless)
            bin_vol = 4.0 * self.mbh1_delta * self.mrat_delta

            for i, j, k in np.ndindex(*shape):
                z = self.zprime(self.mstar_pri[i], self.mrat[j], self.redz[k])
                if z <= 0.0:
                    dndz[i, j, k] = 0.0
                else:
                    dnbh = self.dnbh_dlog10m(self.mstar_pri[i], self.mrat[j], z)
                    dndz[i, j, k] = dnbh * bin_vol

            self._dndz = dndz

        return self._dndz

    def grid(self,n0=None,M1=None,M2=None):
        """
        input 3d array n0, 1d array MBH1, 2d array MBH2
        output 3d array (Mcbh,qbh,z) (black hole chirp mass,
        black hole mass ratio, redshift) of values for function
        """
        if n0 is None:
            n0 = self.dnbh_dz()
        if M1 is None:
            M1 = 10.0**self.mbh1
        if M2 is None:
            M2 = 10.0**self.mbh2
        Mcbh = np.linspace(5,11,30)
        qbh = np.linspace(0,1,10)
        Mcbhdiff = (Mcbh.max()-Mcbh.min())/(len(Mcbh)-1.0)/2.0
        qbhdiff = (qbh.max()-qbh.min())/(len(qbh)-1.0)/2.0
        output = np.zeros((len(Mcbh),len(qbh),len(self.redz)))
        Mc = np.zeros((len(M1),len(M2[0,:])))
        q = np.zeros((len(M1),len(M2[0,:])))
        for i,j in np.ndindex(len(M1),len(M2[0,:])):
            Mc[i,j] = np.log10(mchirp(M1[i],M2[i,j]))
            if M2[i,j] > M1[i]:
                q[i,j] = M1[i]/M2[i,j]
            else:
                q[i,j] = M2[i,j]/M1[i]
        for i,j in np.ndindex(len(M1),len(M2[0,:])):
            for i0,j0 in np.ndindex(len(Mcbh),len(qbh)):
                if abs(Mc[i,j]-Mcbh[i0]) < Mcbhdiff and abs(q[i,j]-qbh[j0]) < qbhdiff:
                    for k in range(len(self.redz)):
                        output[i0,j0,k] += n0[i,j,k]/1.3
                else:
                    pass
        return output

    def gwb_sa(self, freqs):
        gwb = None
        # m1, q, z
        dndz = self.dnbh_dz()
        zz = self.redz
        cosmo_fact = cosmo.comoving_distance(zz).to('Mpc').value
        # Now [Mpc^3 / s]
        cosmo_fact = 4*np.pi*(SPLC/MPC) * np.square(cosmo_fact)
        print(f"{cosmo_fact=}")

        m1 = np.power(10.0, self.mbh1[np.newaxis, :, np.newaxis]) * MSOL
        m2 = np.power(10.0, self.mbh2[np.newaxis, :, :]) * MSOL
        fr = freqs[:, np.newaxis, np.newaxis, np.newaxis] / (1.0 + zz[np.newaxis, np.newaxis, np.newaxis, :])
        print(f"{fr=}")
        fterm = utils.gw_hardening_rate_dfdt(m1[..., np.newaxis], m2[..., np.newaxis], fr)
        # f / (df/dt)   [s]
        fterm = fr / fterm
        print(f"{fterm=}")

        mc = self.mchirp[np.newaxis, :, :, np.newaxis] * MSOL
        print(f"{mc=}")
        dl = cosmo.luminosity_distance(zz).cgs.value[np.newaxis, np.newaxis, np.newaxis, :]
        print(f"{dl=}")

        # hs = (8.0 / np.sqrt(10.0)) * np.power(NWTG * mc, 5.0/3.0) / (dl * np.power(SPLC, 4))
        # hs *= np.power(2*np.pi*fr, 2.0/3.0)
        # [-] dimensionless strain
        hs = utils.gw_strain_source(mc, dl, fr)
        print(f"{hs=}")

        gwb = dndz * (hs**2) * cosmo_fact * fterm
        print(f"{gwb=}")

        return gwb
