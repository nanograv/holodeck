"""
"""

import numpy as np
from holodeck import cosmo, utils, gravwaves
from holodeck.constants import MSOL, GYR, MPC

_AGE_UNIVERSE_GYR = cosmo.age(0.0).to('Gyr').value  # [Gyr]  ~ 13.78


class Simple_SAM:

    def __init__(
        self,
        size=61,

        # Galaxy Stellar-Mass Function (GSMF)
        gsmf_phi0_const=-2.77,    # normalization [Phi_0]   -2.77, -0.29, +0.27
        gsmf_phiz=-0.27,          # norm redshift dependence [Phi_I]  -0.27, -0.21, +0.23
        gsmf_log10m0=11.24,       # log10 of reference mass/Msol [log10(M0)]  11.24, -0.17, +0.20
        gsmf_alpha0_const=-1.24,  # mass exponent constant [alpha_0]   -1.24, -0.16, +0.16
        gsmf_alphaz=-0.03,        # mass exponent redshift dependent [alpha_I]  -0.03, -0.14, +0.16
        # Galaxy Pair Fraction (GPF)
        gpf_norm=0.025,           # normalization over all mass-ratios (f0)  [0.02, 0.03]
        gpf_alpha=0.0,            # mass index (alpha_f)  [-0.2, +0.2]
        gpf_beta=0.8,             # redshift index (beta_f)  [0.6, 1.0]
        gpf_gamma=0.0,            # mass-ratio index (gamma_f)  [-0.2, +0.2]
        # Galaxy Merger Time (GMT)
        gmt_norm=0.55 * GYR,  # time normalization [Gyr]  (tau_0)  [0.1, 2.0]
        gmt_alpha=0.0,        # mass index (alpha_tau) [-0.2, +0.2]
        gmt_beta=-0.5,        # mass-ratio index (beta_tau) [-2, +1]
        gmt_gamma=0.0,        # redshift index (gamma_tau) [-0.2, +0.2]

        # Galaxy--Black-hole Relation
        mbh_star_log10=8.17,   # 8.17, -0.32, +0.35
        alpha_mbh_star=1.01,   # 1.01, -0.10, +0.08

        mass_gal=None,         # stellar-mass of the _primary_ galaxy (NOT the total M1+M2 mass)
        mrat_gal=None,         # stellar-mass ratio between galaxies
        redz=None,
    ):
        if mass_gal is None:
            # mass_gal = np.logspace(7.1, 14, size) * MSOL
            mass_gal = np.logspace(6.78438, 13.71506, size) * MSOL
        if mrat_gal is None:
            # mrat_gal = np.logspace(-4, 0, 41)
            # mtot=(1.0e4*MSOL, 1.0e11*MSOL, 61), mrat=(1e-3, 1.0, 81)
            mrat_gal = np.logspace(-3, 0, 81)
        if redz is None:
            # redz = np.linspace(0.0, 5.0, 42)
            redz = np.logspace(*np.log10([1e-3, 10.0]), 101)
        self._size = size

        # Convert input quantities
        gsmf_m0 = (10.0 ** gsmf_log10m0) * MSOL
        # gmt_norm = gmt_norm * GYR
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
        self._mbh_star = mbh_star
        self._mbh_star_log10 = mbh_star_log10
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

        self.mass_gal = mass_gal    #: primary-galaxy stellar-mass
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

    def _zprime(self, mgal, qgal, redz):
        tau0 = self.gmt(mgal, qgal, redz)  # sec
        age = cosmo.age(redz).to('s').value
        new_age = age + tau0
        redz_prime = -1.0 * np.ones_like(new_age)
        idx = (new_age < _AGE_UNIVERSE_GYR * GYR)
        redz_prime[idx] = cosmo.tage_to_z(new_age[idx])
        return redz_prime

    def gwb_sam(self, fobs_gw, sam, dlog10=True, sum=True, redz_prime=True):
        """GW background semi-analytic model

        Parameters
        ----------
        fobs_gw : float
            Observer-frame GW-frequency in units of [1/sec].  This is a single, float value.
        sam : `Semi_Analytic_Model` instance
            Binary population to sample
        dlog10 : boolean, optional
        sum : boolean, optional
        redz_prime : boolean, optional
            Galaxy-merger redshift

        Returns
        -------
        gwb : (F,) ndarray
            GW Background: the ideal characteristic strain of the GWB in each frequency bin.
            Does not include the strain from the loudest binary in each bin (`gwf`).

        Notes
        -----
        dlog_{10}M has higher performance than dM
        """

        # mg, qg, rz = np.broadcast_arrays(self.mass_gal, self.mrat_gal, self.redz)

        mg = self.mass_gal[:, np.newaxis, np.newaxis]    # this is _primary_ galaxy
        qg = self.mrat_gal[np.newaxis, :, np.newaxis]
        rz = self.redz[np.newaxis, np.newaxis, :]

        mtot = self.mbh[:, :, np.newaxis]
        mrat = self.qbh[np.newaxis, :, np.newaxis]
        ndens = sam._ndens_mbh(mg, qg, rz) / (MPC**3)

        # convert from initial galaxy-merger redshift, to after galaxy merger-time
        if redz_prime:
            rz = self._zprime(mg, qg, rz)
            print(f"{self} :: {utils.stats(rz)=}")

        # convert from dn/dlog10(M) ==> dn/dM
        if not dlog10:
            ndens = ndens / (np.log(10.0) * mtot)

        gwb = gravwaves.gwb_ideal(fobs_gw, ndens, mtot, mrat, rz, dlog10=dlog10, sum=sum)

        return gwb

    def gwb_ideal(self, fobs_gw, dlog10=True, sum=True, redz_prime=True):
        # NOTE: dlog10M performs MUCH better than dM
        mg = self.mass_gal[:, np.newaxis, np.newaxis]    #: this is primary galaxy stellar mass
        qg = self.mrat_gal[np.newaxis, :, np.newaxis]
        redz = self.redz[np.newaxis, np.newaxis, :]
        ndens = self.ndens_mbh(mg, qg, redz, dlog10=dlog10) / (MPC**3)

        # convert from initial galaxy-merger redshift, to after galaxy merger-time
        if redz_prime:
            redz = self._zprime(mg, qg, redz)
            print(f"{self} :: {utils.stats(redz)=}")

        mtot = self.mbh[:, :, np.newaxis]
        mrat = self.qbh[np.newaxis, :, np.newaxis]
        gwb = gravwaves.gwb_ideal(fobs_gw, ndens, mtot, mrat, redz, dlog10=dlog10, sum=sum)
        return gwb

    def _integrated_ndens_mbh(self):
        mg = self.mass_gal[:, np.newaxis, np.newaxis]
        qg = self.mrat_gal[np.newaxis, :, np.newaxis]
        rz = self.redz[np.newaxis, np.newaxis, :]
        mtot = self.mbh[:, :, np.newaxis]
        mrat = self.qbh[np.newaxis, :, np.newaxis]

        # this is [1/Mpc^3]
        integ = self.ndens_mbh(mg, qg, rz, dlog10=True)
        arguments = [np.log10(mtot), mrat, rz]

        for ax, xx in enumerate(arguments):
            integ = np.moveaxis(integ, ax, 0)
            xx = np.moveaxis(xx, ax, 0)
            integ = 0.5 * (integ[:-1] + integ[1:]) * np.diff(xx, axis=0)
            integ = np.moveaxis(integ, 0, ax)

        integ = integ.sum()
        return integ

    def ndens_mbh(self, mass_gal, mrat_gal, redz, dlog10=True):
        """Number density of MBH mergers [Chen+2019] Eq. 21

        mass_gal : primary galaxy stellar mass

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

        Parameters
        ----------
        mass_gal : primary galaxy stellar mass

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
