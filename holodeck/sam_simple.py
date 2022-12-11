"""
"""

import numpy as np
import holodeck as holo
from holodeck import cosmo, utils
from holodeck.constants import MSOL, GYR, MPC, SPLC, NWTG


class Simple_SAM:

    def __init__(
        self,
        # Galaxy Stellar-Mass Function (GSMF)
        gsmf_phi0_const=-2.77,          # normalization [Phi_0]
        gsmf_phiz=-0.27,          # norm redshift dependence [Phi_I]
        gsmf_log10m0=11.24,       # log10 of reference mass/Msol [log10(M0)]
        gsmf_alpha0_const=-1.24,          # mass exponent constant [alpha_0]
        gsmf_alphaz=-0.03,          # mass exponent redshift dependent [alpha_I]
        # Galaxy Pair Fraction (GPF)
        gpf_norm=0.025,     # normalization over all mass-ratios (f0)  [0.02, 0.03]
        gpf_alpha=0.0,            # mass index (alpha_f)  [-0.2, +0.2]
        gpf_beta=0.8,             # redshift index (beta_f)  [0.6, 1.0]
        gpf_gamma=0.0,            # mass-ratio index (gamma_f)  [-0.2, +0.2]
        # Galaxy Merger Time (GMT)
        gmt_norm=0.55,        # time normalization [Gyr]  (tau_0)  [0.1, 2.0]
        gmt_alpha=0.0,        # mass index (alpha_tau) [-0.2, +0.2]
        gmt_beta=-0.5,        # mass-ratio index (beta_tau) [-2, +1]
        gmt_gamma=0.0,        # redshift index (gamma_tau) [-0.2, +0.2]

        # Galaxy--Black-hole Relation
        mbh_star_log10=8.17,
        alpha_mbh_star=1.01,
    ):
        mass_gal = np.logspace(6, 15, 102) * MSOL
        mrat_gal = np.logspace(-4, 0, 101)
        redz = np.linspace(0.0, 5.0, 100)

        # Convert input quantities
        gsmf_m0 = (10.0 ** gsmf_log10m0) * MSOL
        gmt_norm = gmt_norm * GYR
        mbh_star = (10.0 ** mbh_star_log10) * MSOL

        # Convert between GPF normalization over all mass-ratios, to normalization at each mass-ratio
        pow = gpf_gamma + 1.0
        qlo = 0.25
        qhi = 1.00
        _pair_norm = (qhi**pow - qlo**pow) / pow
        gpf_norm_prime = gpf_norm / _pair_norm

        # Calculate derived constants
        # phi0 = 10.0 ** (gsmf_phi0 + (gsmf_phiz * redz))
        # neff = (phi0 * gpf_norm_prime) / (gsmf_m0 * gmt_norm)
        # neff *= np.power(0.4/cosmo.h, gmt_alpha)
        # neff *= np.power((1e11 * MSOL) / gsmf_m0, gmt_alpha - gpf_alpha)

        # alpha_eff = gsmf_alpha0 + (gsmf_alphaz * redz) + gpf_alpha - gmt_alpha
        beta_eff = gpf_beta - gmt_beta
        gamma_eff = gpf_gamma - gmt_gamma

        # self._neff = neff
        # self._alpha_eff = alpha_eff
        self._beta_eff = beta_eff
        self._gamma_eff = gamma_eff
        self._gsmf_m0 = gsmf_m0
        self._mbh_star = mbh_star
        self._alpha_mbh_star = alpha_mbh_star

        self._gsmf_phi0_const = gsmf_phi0_const
        self._gsmf_phiz = gsmf_phiz
        self._gsmf_m0 = gsmf_m0
        self._gsmf_alpha0_const = gsmf_alpha0_const
        self._gsmf_alphaz = gsmf_alphaz

        self._gpf_norm = gpf_norm
        self._gpf_norm_prime = gpf_norm_prime
        self._gpf_alpha = gpf_alpha
        self._gpf_beta = gpf_beta
        self._gpf_gamma = gpf_gamma

        self._gmt_norm = gmt_norm
        self._gmt_alpha = gmt_alpha
        self._gmt_beta = gmt_beta
        self._gmt_gamma = gmt_gamma

        mbh = self.mgal_to_mbh(mass_gal)
        qbh = self.qgal_to_qbh(mrat_gal)

        self.mass_gal = mass_gal
        self.mrat_gal = mrat_gal
        self.mbh = mbh
        self.qbh = qbh
        self.redz = redz

        return

    def _gsmf_phi0(self, redz):
        phi0 = 10.0 ** (self._gsmf_phi0_const + (self._gsmf_phiz * redz))
        return phi0

    def _gsmf_alpha0(self, redz):
        alpha0 = self._gsmf_alpha0_const + (self._gsmf_alphaz * redz)
        return alpha0

    def _neff(self, redz):
        neff = (self._gsmf_phi0(redz) * self._gpf_norm_prime) / (self._gsmf_m0 * self._gmt_norm)
        neff *= np.power(0.4/cosmo.h, self._gmt_alpha)
        neff *= np.power((1e11 * MSOL) / self._gsmf_m0, self._gmt_alpha - self._gpf_alpha)
        return neff

    def _alpha_eff(self, redz):
        alpha_eff = self._gsmf_alpha0(redz) + self._gpf_alpha - self._gmt_alpha
        return alpha_eff

    def gsmf(self, mgal, redz):
        mm = (mgal / self._gsmf_m0)
        power = 1.0 + self._gsmf_alpha0(redz)
        rv = np.log(10.0) * self._gsmf_phi0(redz) * np.power(mm, power) * np.exp(-mm)
        return rv

    def gpf(self, mgal, qgal, redz):
        rv = self._gpf_norm_prime * np.power(mgal / (1e11 * MSOL), self._gpf_alpha)
        rv = rv * np.power(1.0 + redz, self._gpf_beta) * np.power(qgal, self._gpf_gamma)
        return rv

    def gmt(self, mgal, qgal, redz):
        m0 = 1e11 * MSOL * (0.4 / cosmo.h)
        rv = self._gmt_norm * np.power(mgal/m0, self._gmt_alpha)
        rv = rv * np.power(1.0 + redz, self._gmt_beta) * np.power(qgal, self._gmt_gamma)
        return rv

    def gwb(self, fobs_gw):
        # mg, qg, rz = np.broadcast_arrays(self.mass_gal, self.mrat_gal, self.redz)
        mg = self.mass_gal[:, np.newaxis, np.newaxis]
        qg = self.mrat_gal[np.newaxis, :, np.newaxis]
        rz = self.redz[np.newaxis, np.newaxis, :]

        mtot = self.mbh[:, np.newaxis, np.newaxis]
        mrat = self.qbh[np.newaxis, :, np.newaxis]
        ndens = self.ndens_bh(mg, qg, rz)

        gwb = gwb_ideal(fobs_gw, ndens, mtot, mrat, rz)

        return gwb

    def gwb_sam(self, fobs_gw, sam):
        # mg, qg, rz = np.broadcast_arrays(self.mass_gal, self.mrat_gal, self.redz)
        mg = self.mass_gal[:, np.newaxis, np.newaxis]
        qg = self.mrat_gal[np.newaxis, :, np.newaxis]
        rz = self.redz[np.newaxis, np.newaxis, :]

        mtot = self.mbh[:, np.newaxis, np.newaxis]
        mrat = self.qbh[np.newaxis, :, np.newaxis]
        ndens = sam._ndens_mbh(mg, qg, rz)

        gwb = gwb_ideal(fobs_gw, ndens, mtot, mrat, rz)

        return gwb

    def gwb_dlog10m(self, fobs_gw):
        # mg, qg, rz = np.broadcast_arrays(self.mass_gal, self.mrat_gal, self.redz)
        mg = self.mass_gal[:, np.newaxis, np.newaxis]
        qg = self.mrat_gal[np.newaxis, :, np.newaxis]
        rz = self.redz[np.newaxis, np.newaxis, :]

        mtot = self.mbh[:, np.newaxis, np.newaxis]
        mrat = self.qbh[np.newaxis, :, np.newaxis]
        _ndens = self.ndens_bh(mg, qg, rz)
        ndens_dlog10m = _ndens * mtot * np.log(10.0)

        gwb = gwb_ideal_dlog10m(fobs_gw, ndens_dlog10m, mtot, mrat, rz)

        return gwb

    def ndens_bh(self, mass_gal, mrat_gal, redz):
        """Number density of MBH mergers [Chen+2019] Eq. 21

        This is

            (d^3 n_mbh / [dz' dM_bh dq_bh]) =
                (d^3 n_gal / [dz' dM dq]) * (dM_gal / dM_bh) * (dq_gal / dq_bh)

        """
        nd = self.ndens_galaxy(mass_gal, mrat_gal, redz)
        nd = nd / self.dmbh_dmgal(mass_gal) / self.dqbh_dqgal(mrat_gal)
        return nd

    def _ndens_galaxy_check(self, mass_gal, mrat_gal, redz):
        gsmf = self.gsmf(mass_gal, redz)
        gpf = self.gpf(mass_gal, mrat_gal, redz)
        gmt = self.gmt(mass_gal, mrat_gal, redz)

        nd = gsmf * gpf / (gmt * np.log(10.0) * mass_gal)
        nd = nd * cosmo.dtdz(redz) / MPC**3
        return nd

    def ndens_galaxy(self, mass_gal, mrat_gal, redz):
        """Number density of galaxy mergers [Chen+2019] Eq. 17

        This is  ``d^3 n_gal / [dM dq dz']``
        """
        neff = self._neff(redz)
        alpha = self._alpha_eff(redz)
        beta = self._beta_eff
        gamma = self._gamma_eff
        m0 = self._gsmf_m0

        mm = mass_gal / m0
        nd = neff * np.power(mm, alpha) * np.exp(-mm)
        nd = nd * np.power(1.0 + redz, beta)
        nd = nd * np.power(mrat_gal, gamma)
        nd = nd * cosmo.dtdz(redz) / MPC**3

        # Currently, we have  `` d^3 n / dM_pri dq dz' ``
        # convert from dM_pri to dM=dM_tot  ::  ``dM_tot = (1+q)*dM_pri``
        nd = nd / (1.0 + mrat_gal)

        return nd

    def dqbh_dqgal(self, mrat_gal):
        alpha_star = self._alpha_mbh_star
        dq = alpha_star * np.power(mrat_gal, alpha_star - 1.0)
        return dq

    def dmbh_dmgal(self, mgal):
        """
        dM_bh / dM_gal = (dM_bh / dM_bulge) * (dM_bulge / dM_gal)
        """
        alpha_mbh_star = self._alpha_mbh_star
        mbulge = self.mgal_to_mbulge(mgal)
        mbh = self.mbulge_to_mbh(mbulge)
        dm = alpha_mbh_star * mbh / mgal
        return dm

    def mgal_to_mbulge(self, mgal):
        """Convert from total galaxy mass to galaxy bulge mass.  [Chen+2019] Eq.19
        """
        # mm = mgal / MSOL
        # mbulge = 0.615 * np.ones_like(mm)
        # hi = (mm > 1e10)
        # mbulge[hi] = mbulge[hi] + np.sqrt(6.9)
        return mgal * 0.615

    def mbulge_to_mbh(self, mbulge):
        mstar = self._mbh_star
        alpha_star = self._alpha_mbh_star
        mref = 1.0e11 * MSOL
        mbh = mstar * np.power(mbulge / mref, alpha_star)
        return mbh

    def mgal_to_mbh(self, mgal):
        mbulge = self.mgal_to_mbulge(mgal)
        mbh = self.mbulge_to_mbh(mbulge)
        return mbh

    def qgal_to_qbh(self, qgal):
        qbh = np.power(qgal, self._alpha_mbh_star)
        return qbh


def gwb_ideal(fobs_gw, ndens, mtot, mrat, redz):

    const = ((4.0 * np.pi) / (3 * SPLC**2))
    mc = utils.chirp_mass_mtmr(mtot, mrat)
    mc = np.power(NWTG * mc, 5.0/3.0)
    rz = np.power(1 + redz, -1.0/3.0)
    fogw = np.power(np.pi * fobs_gw, -4.0/3.0)

    integ = ndens * mc * rz
    for ax, xx in enumerate([mtot, mrat, redz]):
        integ = np.moveaxis(integ, ax, 0)
        xx = np.moveaxis(xx, ax, 0)
        integ = 0.5 * (integ[:-1] + integ[1:]) * np.diff(xx, axis=0)
        integ = np.moveaxis(integ, 0, ax)

    gwb = const * fogw * np.sum(integ)
    gwb = np.sqrt(gwb)
    return gwb


def gwb_ideal_dlog10m(fobs_gw, ndens_dlog10m, mtot, mrat, redz):

    const = ((4.0 * np.pi) / (3 * SPLC**2))
    mc = utils.chirp_mass_mtmr(mtot, mrat)
    mc = np.power(NWTG * mc, 5.0/3.0)
    rz = np.power(1 + redz, -1.0/3.0)
    fogw = np.power(np.pi * fobs_gw, -4.0/3.0)

    integ = ndens_dlog10m * mc * rz
    for ax, xx in enumerate([np.log10(mtot), mrat, redz]):
        integ = np.moveaxis(integ, ax, 0)
        xx = np.moveaxis(xx, ax, 0)
        integ = 0.5 * (integ[:-1] + integ[1:]) * np.diff(xx, axis=0)
        integ = np.moveaxis(integ, 0, ax)

    gwb = const * fogw * np.sum(integ)
    gwb = np.sqrt(gwb)
    return gwb

