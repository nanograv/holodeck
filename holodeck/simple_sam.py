"""
"""

import numpy as np
from holodeck import cosmo, utils
from holodeck.constants import MSOL, GYR, MPC, SPLC, NWTG


class Simple_SAM:

    def __init__(
        self,
        size=100,
        # Galaxy Stellar-Mass Function (GSMF)
        gsmf_phi0_const=-2.77,    # normalization [Phi_0]   -2.77, -0.29, +0.27
        gsmf_phiz=-0.27,          # norm redshift dependence [Phi_I]  -0.27, -0.21, +0.23
        gsmf_log10m0=11.24,       # log10 of reference mass/Msol [log10(M0)]  11.24, -0.17, +0.20
        gsmf_alpha0_const=-1.24,          # mass exponent constant [alpha_0]   -1.24, -0.16, +0.16
        gsmf_alphaz=-0.03,          # mass exponent redshift dependent [alpha_I]  -0.03, -0.14, +0.16
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
        mbh_star_log10=8.17,   # 8.17, -0.32, +0.35
        alpha_mbh_star=1.01,   # 1.01, -0.10, +0.08
    ):
        mass_gal = np.logspace(6, 15, size) * MSOL
        mrat_gal = np.logspace(-4, 0, 41)
        redz = np.linspace(0.0, 5.0, 42)
        self._size = size

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

        mbh_pri = self.mgal_to_mbh(mass_gal)
        qbh = self.qgal_to_qbh(mrat_gal)
        mbh = mbh_pri[:, np.newaxis] * (1.0 + qbh[np.newaxis, :])

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

    def gwb_sam(self, fobs_gw, sam, dlog10=True):
        # NOTE: dlog10M performs MUCH better than dM
        # mg, qg, rz = np.broadcast_arrays(self.mass_gal, self.mrat_gal, self.redz)
        mg = self.mass_gal[:, np.newaxis, np.newaxis]
        qg = self.mrat_gal[np.newaxis, :, np.newaxis]
        rz = self.redz[np.newaxis, np.newaxis, :]

        mtot = self.mbh[:, :, np.newaxis]
        mrat = self.qbh[np.newaxis, :, np.newaxis]
        ndens = sam._ndens_mbh(mg, qg, rz) / (MPC**3)

        # convert from dn/dlog10(M) ==> dn/dM
        if not dlog10:
            ndens = ndens / (np.log(10.0) * mtot)

        gwb = gwb_ideal(fobs_gw, ndens, mtot, mrat, rz, dlog10=dlog10)
        # gwb = gwb_ideal_dlog10m(fobs_gw, ndens_dlog10m, mtot, mrat, rz)

        return gwb

    def gwb_ideal(self, fobs_gw, dlog10=True):
        # NOTE: dlog10M performs MUCH better than dM
        mg = self.mass_gal[:, np.newaxis, np.newaxis]
        qg = self.mrat_gal[np.newaxis, :, np.newaxis]
        rz = self.redz[np.newaxis, np.newaxis, :]
        mtot = self.mbh[:, :, np.newaxis]
        mrat = self.qbh[np.newaxis, :, np.newaxis]

        ndens = self.ndens_mbh(mg, qg, rz, dlog10=dlog10) / (MPC**3)
        gwb = gwb_ideal(fobs_gw, ndens, mtot, mrat, rz, dlog10=dlog10)
        return gwb

    def ndens_mbh(self, mass_gal, mrat_gal, redz, dlog10=True):
        """Number density of MBH mergers [Chen+2019] Eq. 21

        (d^3 n_mbh / [dlog10(M_bh) dq_bh dz']) =
            (d^3 n_gal / [dlog10(M_gal) dq_gal dz'])
             * (M_bh/M_gal-pri) * (dM_bh-pri/dM_bh) * (dM_gal-pri / dM_bh-pri) * (dq_gal / dq_bh)

        (d^3 n_mbh / [dM_bh dq_bh dz']) =
            (d^3 n_gal / [dM_gal-pri    dq_gal dz'])
                                * (dM_bh-pri/dM_bh) * (dM_gal-pri / dM_bh-pri) * (dq_gal / dq_bh)

        """
        nd = self.ndens_galaxy(mass_gal, mrat_gal, redz, dlog10=dlog10)
        dmgal_dmbh__pri = 1.0 / self.dmbh_dmgal(mass_gal)
        dqgal_dqbh = 1.0 / self.dqbh_dqgal(mrat_gal)
        mrat_mbh = self.qgal_to_qbh(mrat_gal)
        nd = nd * dmgal_dmbh__pri * dqgal_dqbh / (1.0 + mrat_mbh)
        if dlog10:
            mbh_pri = self.mgal_to_mbh(mass_gal)
            mbh = mbh_pri * (1.0 + mrat_mbh)
            nd = (mbh / mass_gal) * nd

        return nd

    def _ndens_galaxy_check(self, mass_gal, mrat_gal, redz, dlog10=True):
        gsmf = self.gsmf(mass_gal, redz)
        gpf = self.gpf(mass_gal, mrat_gal, redz)
        gmt = self.gmt(mass_gal, mrat_gal, redz)

        nd = gsmf * gpf / gmt
        nd = nd * cosmo.dtdz(redz)
        # This is  d^3 n / [dlog10(M_gal) dq_gal dz]

        if not dlog10:
            # convert from d/dlog10(M) ==> d/dM_pri
            nd = nd / (np.log(10.0) * mass_gal)
            # convert from d/dM_pri ==> d/dM_tot
            nd = nd / (1.0 + mrat_gal)

        return nd

    def ndens_galaxy(self, mass_gal, mrat_gal, redz, dlog10=True):
        """Number density of galaxy mergers [Chen+2019] Eq. 17

        This is  ``d^3 n_gal / [dlog10(M) dq dz']``   [Mpc^-3]

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
        nd = nd * cosmo.dtdz(redz)    # / MPC**3

        # Currently, we have  `` d^3 n / dM_pri dq dz' ``

        # convert from 1/dM ==> 1/dlog10(M) = ln(10) * M * 1/dM
        if dlog10:
            nd = nd * mass_gal * np.log(10.0)

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


def gwb_ideal(fobs_gw, ndens, mtot, mrat, redz, dlog10=False):

    const = ((4.0 * np.pi) / (3 * SPLC**2))
    mc = utils.chirp_mass_mtmr(mtot, mrat)
    mc = np.power(NWTG * mc, 5.0/3.0)
    rz = np.power(1 + redz, -1.0/3.0)
    fogw = np.power(np.pi * fobs_gw, -4.0/3.0)

    integ = ndens * mc * rz
    arguments = [mtot, mrat, redz]
    if dlog10:
        arguments[0] = np.log10(arguments[0])

    for ax, xx in enumerate(arguments):
        integ = np.moveaxis(integ, ax, 0)
        xx = np.moveaxis(xx, ax, 0)
        integ = 0.5 * (integ[:-1] + integ[1:]) * np.diff(xx, axis=0)
        integ = np.moveaxis(integ, 0, ax)

    gwb = const * fogw * np.sum(integ)
    gwb = np.sqrt(gwb)
    return gwb