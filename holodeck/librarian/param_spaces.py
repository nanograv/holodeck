"""Parameter-Space definitions for holodeck libraries.
"""

import holodeck as holo
from holodeck.constants import GYR, PC, MSOL
from holodeck.librarian.libraries import _Param_Space, PD_Uniform


class _PS_Astro_Strong(_Param_Space):
    """SAM Model with strongly astrophysically-motivated parameters.

    This model uses a double-Schechter GSMF, an Illustris-derived galaxy merger rate, a Kormendy+Ho
    M-MBulge relationship, and a phenomenology binary evolution model.

    """

    DEFAULTS = dict(
        hard_time=3.0,          # [Gyr]
        hard_sepa_init=1e4,     # [pc]
        hard_rchar=10.0,       # [pc]
        hard_gamma_inner=-1.0,
        hard_gamma_outer=+2.5,

        # Galaxy stellar-mass Function (GSMF_Double_Schechter)
        # Parameters are based on `double-schechter.ipynb` conversions from [Leja2020]_
        gsmf_log10_phi_one_z0=-2.383,    # - 2.383 ± 0.028
        gsmf_log10_phi_one_z1=-0.264,    # - 0.264 ± 0.072
        gsmf_log10_phi_one_z2=-0.107,    # - 0.107 ± 0.031
        gsmf_log10_phi_two_z0=-2.818,    # - 2.818 ± 0.050
        gsmf_log10_phi_two_z1=-0.368,    # - 0.368 ± 0.070
        gsmf_log10_phi_two_z2=+0.046,    # + 0.046 ± 0.020
        gsmf_log10_mstar_z0=+10.767,     # +10.767 ± 0.026
        gsmf_log10_mstar_z1=+0.124,      # + 0.124 ± 0.045
        gsmf_log10_mstar_z2=-0.033,      # - 0.033 ± 0.015
        gsmf_alpha_one=-0.28,            # - 0.280 ± 0.070
        gsmf_alpha_two=-1.48,            # - 1.480 ± 0.150

        gmr_norm0_log10 = -2.2287,       # -2.2287 ± 0.0045    A0 [log10(A*Gyr)]
        gmr_normz = +2.4644,             # +2.4644 ± 0.0128    eta
        gmr_malpha0 = +0.2241,           # +0.2241 ± 0.0038    alpha0
        gmr_malphaz = -1.1759,           # -1.1759 ± 0.0316    alpha1
        gmr_mdelta0 = +0.7668,           # +0.7668 ± 0.0202    delta0
        gmr_mdeltaz = -0.4695,           # -0.4695 ± 0.0440    delta1
        gmr_qgamma0 = -1.2595,           # -1.2595 ± 0.0026    beta0
        gmr_qgammaz = +0.0611,           # +0.0611 ± 0.0021    beta1
        gmr_qgammam = -0.0477,           # -0.0477 ± 0.0013    gamma

        # From [KH2013]_
        mmb_mamp=0.49e9,        # 0.49e9 + 0.06 - 0.05  [Msol]
        mmb_plaw=1.17,          # 1.17 ± 0.08
        mmb_scatter_dex=0.28,
    )

    @classmethod
    def _init_sam(cls, sam_shape, params):
        log10_phi_one = [
            params['gsmf_log10_phi_one_z0'],
            params['gsmf_log10_phi_one_z1'],
            params['gsmf_log10_phi_one_z2'],
        ]
        log10_phi_two = [
            params['gsmf_log10_phi_two_z0'],
            params['gsmf_log10_phi_two_z1'],
            params['gsmf_log10_phi_two_z2'],
        ]
        log10_mstar = [
            params['gsmf_log10_mstar_z0'],
            params['gsmf_log10_mstar_z1'],
            params['gsmf_log10_mstar_z2'],
        ]
        gsmf = holo.sams.GSMF_Double_Schechter(
            log10_phi1=log10_phi_one,
            log10_phi2=log10_phi_two,
            log10_mstar=log10_mstar,
            alpha1=params['gsmf_alpha_one'],
            alpha2=params['gsmf_alpha_two'],
        )

        # Illustris Galaxy Merger Rate
        gmr = holo.sams.GMR_Illustris(
            norm0_log10=params['gmr_norm0_log10'],
            normz=params['gmr_normz'],
            malpha0=params['gmr_malpha0'],
            malphaz=params['gmr_malphaz'],
            mdelta0=params['gmr_mdelta0'],
            mdeltaz=params['gmr_mdeltaz'],
            qgamma0=params['gmr_qgamma0'],
            qgammaz=params['gmr_qgammaz'],
            qgammam=params['gmr_qgammam'],
        )

        mmbulge = holo.relations.MMBulge_KH2013(
            mamp=params['mmb_mamp']*MSOL,
            mplaw=params['mmb_plaw'],
            scatter_dex=params['mmb_scatter_dex'],
        )

        sam = holo.sams.Semi_Analytic_Model(
            gsmf=gsmf, gmr=gmr, mmbulge=mmbulge, shape=sam_shape,
        )
        return sam

    @classmethod
    def _init_hard(cls, sam, params):
        hard = holo.hardening.Fixed_Time_2PL_SAM(
            sam,
            params['hard_time']*GYR,
            sepa_init=params['hard_sepa_init']*PC,
            rchar=params['hard_rchar']*PC,
            gamma_inner=params['hard_gamma_inner'],
            gamma_outer=params['hard_gamma_outer'],
        )
        return hard


class PS_Astro_Strong_Hard_Only(_PS_Astro_Strong):

    def __init__(self, log, nsamples=None, sam_shape=None, seed=None):
        parameters = [
            PD_Uniform("hard_time", 0.1, 11.0, default=3.0),   # [Gyr]
            PD_Uniform("hard_gamma_inner", -1.5, +0.0, default=-1.0),
        ]
        _Param_Space.__init__(
            self, parameters,
            log=log, nsamples=nsamples, sam_shape=sam_shape, seed=seed,
        )
        return

_param_spaces_dict = {
    'PS_Astro_Strong_Hard_Only': PS_Astro_Strong_Hard_Only,
}

