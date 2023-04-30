"""
"""

import holodeck as holo
from holodeck.constants import PC, GYR
from holodeck.librarian import (
    _Param_Space,
    PD_Normal, PD_Uniform,
    # _Parameter_Space, _LHS_Parameter_Space,
    # PD_Uniform_Log,
)


class PS_Broad_Uniform_01(_Param_Space):

    def __init__(self, log, nsamples, sam_shape, seed):
        super().__init__(
            log, nsamples, sam_shape, seed,

            hard_time=PD_Uniform(0.1, 12.0),   # [Gyr]
            hard_gamma_inner=PD_Uniform(-1.5, +0.0),

            gsmf_phi0=PD_Uniform(-3.5, -1.5),           # [log10(#/Mpc^3)]
            gsmf_mchar0_log10=PD_Uniform(10.5, 12.0),   # [log10(Msol)]
            gsmf_alpha0=PD_Uniform(-2.0, -0.5),

            gpf_zbeta=PD_Uniform(-0.5, +2.5),
            gpf_qgamma=PD_Uniform(-1.5, +1.5),

            gmt_norm=PD_Uniform(0.1, +10.0),    # [Gyr]
            gmt_zbeta=PD_Uniform(-2.0, +1.0),

            mmb_amp_log10=PD_Uniform(+7.5, +9.5),   # [log10(Msol)]
            mmb_plaw=PD_Uniform(+0.5, +2.0),
            mmb_scatter=PD_Uniform(+0.0, +0.6),
        )

    @classmethod
    def model_for_params(cls, params, sam_shape=None):

        # Other parameters are guesses
        hard_time = params['hard_time'] * GYR
        hard_rchar = 10.0 * PC
        hard_gamma_outer = +2.5
        hard_sepa_init = 1e4 * PC

        # Parameters are based on `sam-parameters.ipynb` fit to [Tomczak+2014]
        gsmf_phiz = -0.6
        gsmf_mcharz = 0.11
        gsmf_alphaz = -0.03

        gpf_frac_norm_allq = 0.025
        gpf_malpha = 0.0
        gpf_max_frac = 1.0

        gmt_norm = params['gmt_norm'] * GYR
        gmt_malpha = 0.0
        gmt_qgamma = -1.0   # Boylan-Kolchin+2008

        gsmf = holo.sam.GSMF_Schechter(
            phi0=params['gsmf_phi0'],
            phiz=gsmf_phiz,
            mchar0_log10=params['gsmf_mchar0_log10'],
            mcharz=gsmf_mcharz,
            alpha0=params['gsmf_alpha0'],
            alphaz=gsmf_alphaz,
        )
        gpf = holo.sam.GPF_Power_Law(
            frac_norm_allq=gpf_frac_norm_allq,
            malpha=gpf_malpha,
            qgamma=params['gpf_qgamma'],
            zbeta=params['gpf_zbeta'],
            max_frac=gpf_max_frac,
        )
        gmt = holo.sam.GMT_Power_Law(
            time_norm=gmt_norm,
            malpha=gmt_malpha,
            qgamma=gmt_qgamma,
            zbeta=params['gmt_zbeta'],
        )
        mmbulge = holo.relations.MMBulge_KH2013(
            mamp_log10=params['mmb_amp_log10'],
            mplaw=params['mmb_plaw'],
            scatter_dex=params['mmb_scatter'],
        )

        kw = {} if sam_shape is None else dict(shape=sam_shape)
        sam = holo.sam.Semi_Analytic_Model(
            gsmf=gsmf, gpf=gpf, gmt=gmt, mmbulge=mmbulge,
            **kw
        )
        hard = holo.hardening.Fixed_Time.from_sam(
            sam,
            hard_time,
            sepa_init=hard_sepa_init,
            rchar=hard_rchar,
            gamma_sc=params['hard_gamma_inner'],
            gamma_df=hard_gamma_outer,
            progress=False,
        )

        return sam, hard


class PS_Broad_Uniform_01_GW(_Param_Space):

    def __init__(self, log, nsamples, sam_shape, seed):
        super().__init__(
            log, nsamples, sam_shape, seed,

            gsmf_phi0=PD_Uniform(-3.5, -1.5),
            gsmf_mchar0_log10=PD_Uniform(10.5, 12.0),   # [log10(Msol)]
            gsmf_alpha0=PD_Uniform(-2.0, -0.5),

            gpf_zbeta=PD_Uniform(-0.5, +2.5),
            gpf_qgamma=PD_Uniform(-1.5, +1.5),

            gmt_norm=PD_Uniform(0.1, +10.0),    # [Gyr]
            gmt_zbeta=PD_Uniform(-2.0, +1.0),

            mmb_amp_log10=PD_Uniform(+7.5, +9.5),   # [log10(Msol)]
            mmb_plaw=PD_Uniform(+0.5, +2.0),
            mmb_scatter=PD_Uniform(+0.0, +0.6),
        )

    @classmethod
    def model_for_params(cls, params, sam_shape=None):

        # Parameters are based on `sam-parameters.ipynb` fit to [Tomczak+2014]
        gsmf_phiz = -0.6
        gsmf_mcharz = 0.11
        gsmf_alphaz = -0.03

        gpf_frac_norm_allq = 0.025
        gpf_malpha = 0.0
        gpf_max_frac = 1.0

        gmt_norm = params['gmt_norm'] * GYR
        gmt_malpha = 0.0
        gmt_qgamma = -1.0   # Boylan-Kolchin+2008

        gsmf = holo.sam.GSMF_Schechter(
            phi0=params['gsmf_phi0'],
            phiz=gsmf_phiz,
            mchar0_log10=params['gsmf_mchar0_log10'],
            mcharz=gsmf_mcharz,
            alpha0=params['gsmf_alpha0'],
            alphaz=gsmf_alphaz,
        )
        gpf = holo.sam.GPF_Power_Law(
            frac_norm_allq=gpf_frac_norm_allq,
            malpha=gpf_malpha,
            qgamma=params['gpf_qgamma'],
            zbeta=params['gpf_zbeta'],
            max_frac=gpf_max_frac,
        )
        gmt = holo.sam.GMT_Power_Law(
            time_norm=gmt_norm,
            malpha=gmt_malpha,
            qgamma=gmt_qgamma,
            zbeta=params['gmt_zbeta'],
        )
        mmbulge = holo.relations.MMBulge_KH2013(
            mamp_log10=params['mmb_amp_log10'],
            mplaw=params['mmb_plaw'],
            scatter_dex=params['mmb_scatter'],
        )

        kw = {} if sam_shape is None else dict(shape=sam_shape)
        sam = holo.sam.Semi_Analytic_Model(
            gsmf=gsmf, gpf=gpf, gmt=gmt, mmbulge=mmbulge,
            ZERO_DYNAMIC_STALLED_SYSTEMS=False,
            ZERO_GMT_STALLED_SYSTEMS=True,
            **kw
        )

        hard = holo.hardening.Hard_GW()

        return sam, hard


class PS_Astro_01(PS_Broad_Uniform_01):

    def __init__(self, log, nsamples, sam_shape, seed):
        super(PS_Broad_Uniform_01, self).__init__(
            log, nsamples, sam_shape, seed,

            hard_time=PD_Uniform(0.1, 12.0),   # [Gyr]
            hard_gamma_inner=PD_Uniform(-1.5, +0.5),

            # from `sam-parameters.ipynb` fits to [Tomczak+2014] with 4x stdev values
            gsmf_phi0=PD_Normal(-2.56, +0.4),
            gsmf_mchar0_log10=PD_Normal(10.9, 0.4),   # [log10(Msol)]
            gsmf_alpha0=PD_Normal(-1.2, +0.2),

            gpf_zbeta=PD_Normal(+0.8, +0.4),
            gpf_qgamma=PD_Normal(+0.5, +0.3),

            gmt_norm=PD_Uniform(0.2, 5.0),    # [Gyr]
            gmt_zbeta=PD_Uniform(-2.0, +0.0),

            mmb_amp_log10=PD_Uniform(+7.5, +9.5),   # [log10(Msol)]
            mmb_plaw=PD_Normal(+1.2, +0.2),
            mmb_scatter=PD_Normal(+0.32, +0.15),
        )


class PS_Astro_01_GW(PS_Broad_Uniform_01_GW):

    def __init__(self, log, nsamples, sam_shape, seed):
        super(PS_Broad_Uniform_01_GW, self).__init__(
            log, nsamples, sam_shape, seed,

            # from `sam-parameters.ipynb` fits to [Tomczak+2014] with 4x stdev values
            gsmf_phi0=PD_Normal(-2.56, +0.4),
            gsmf_mchar0_log10=PD_Normal(10.9, 0.4),   # [log10(Msol)]
            gsmf_alpha0=PD_Normal(-1.2, +0.2),

            gpf_zbeta=PD_Normal(+0.8, +0.4),
            gpf_qgamma=PD_Normal(+0.5, +0.3),

            gmt_norm=PD_Uniform(0.2, 5.0),    # [Gyr]
            gmt_zbeta=PD_Uniform(-2.0, +0.0),

            mmb_amp_log10=PD_Uniform(+7.5, +9.5),   # [log10(Msol)]
            mmb_plaw=PD_Normal(+1.2, +0.2),
            mmb_scatter=PD_Normal(+0.32, +0.15),
        )


class PS_Astro_02(PS_Broad_Uniform_01):

    def __init__(self, log, nsamples, sam_shape, seed):
        super(PS_Broad_Uniform_01, self).__init__(
            log, nsamples, sam_shape, seed,

            hard_time=PD_Uniform(0.1, 12.0),   # [Gyr]
            hard_gamma_inner=PD_Uniform(-1.5, +0.5),

            # from `sam-parameters.ipynb` fits to [Tomczak+2014] with 4x stdev values
            gsmf_phi0=PD_Normal(-2.56, 0.4),
            gsmf_mchar0_log10=PD_Normal(10.9, 0.4),   # [log10(Msol)]
            gsmf_alpha0=PD_Normal(-1.2, 0.2),

            gpf_zbeta=PD_Normal(+0.8, 0.4),
            gpf_qgamma=PD_Normal(+0.5, 0.3),

            gmt_norm=PD_Uniform(0.2, 5.0),    # [Gyr]
            gmt_zbeta=PD_Uniform(-2.0, +0.0),

            mmb_amp_log10=PD_Normal(+8.6, 0.2),   # [log10(Msol)]
            mmb_plaw=PD_Normal(+1.2, 0.2),
            mmb_scatter=PD_Normal(+0.32, 0.15),
        )


class PS_Astro_02_GW(PS_Broad_Uniform_01_GW):

    def __init__(self, log, nsamples, sam_shape, seed):
        super(PS_Broad_Uniform_01_GW, self).__init__(
            log, nsamples, sam_shape, seed,

            # from `sam-parameters.ipynb` fits to [Tomczak+2014] with 4x stdev values
            gsmf_phi0=PD_Normal(-2.56, +0.4),
            gsmf_mchar0_log10=PD_Normal(10.9, 0.4),   # [log10(Msol)]
            gsmf_alpha0=PD_Normal(-1.2, +0.2),

            gpf_zbeta=PD_Normal(+0.8, +0.4),
            gpf_qgamma=PD_Normal(+0.5, +0.3),

            gmt_norm=PD_Uniform(0.2, 5.0),    # [Gyr]
            gmt_zbeta=PD_Uniform(-2.0, +0.0),

            mmb_amp_log10=PD_Normal(+8.6, 0.2),   # [log10(Msol)]
            mmb_plaw=PD_Normal(+1.2, +0.2),
            mmb_scatter=PD_Normal(+0.32, +0.15),
        )


class PS_Astro_Tight_02(PS_Broad_Uniform_01):

    def __init__(self, log, nsamples, sam_shape, seed):
        super(PS_Broad_Uniform_01, self).__init__(
            log, nsamples, sam_shape, seed,

            hard_time=PD_Uniform(0.2, 10.0),   # [Gyr]
            hard_gamma_inner=PD_Uniform(-1.5, -0.5),

            # from `sam-parameters.ipynb` fits to [Tomczak+2014] with 1x stdev values
            gsmf_phi0=PD_Normal(-2.56, 0.1),
            gsmf_mchar0_log10=PD_Normal(10.9, 0.1),   # [log10(Msol)]
            gsmf_alpha0=PD_Normal(-1.2, 0.05),

            gpf_zbeta=PD_Normal(+0.8, 0.4),
            gpf_qgamma=PD_Normal(+0.5, 0.3),

            gmt_norm=PD_Uniform(0.2, 5.0),    # [Gyr]
            gmt_zbeta=PD_Uniform(-2.0, +0.0),

            mmb_amp_log10=PD_Normal(+8.6, 0.1),   # [log10(Msol)]
            mmb_plaw=PD_Normal(+1.2, 0.1),
            mmb_scatter=PD_Normal(+0.32, 0.1),
        )


class PS_Astro_Tight_02_GW(PS_Broad_Uniform_01_GW):

    def __init__(self, log, nsamples, sam_shape, seed):
        super(PS_Broad_Uniform_01_GW, self).__init__(
            log, nsamples, sam_shape, seed,

            # from `sam-parameters.ipynb` fits to [Tomczak+2014] with 4x stdev values
            gsmf_phi0=PD_Normal(-2.56, 0.1),
            gsmf_mchar0_log10=PD_Normal(10.9, 0.1),   # [log10(Msol)]
            gsmf_alpha0=PD_Normal(-1.2, 0.05),

            gpf_zbeta=PD_Normal(+0.8, 0.4),
            gpf_qgamma=PD_Normal(+0.5, 0.3),

            gmt_norm=PD_Uniform(0.2, 5.0),    # [Gyr]
            gmt_zbeta=PD_Uniform(-2.0, +0.0),

            mmb_amp_log10=PD_Normal(+8.6, 0.1),   # [log10(Msol)]
            mmb_plaw=PD_Normal(+1.2, 0.1),
            mmb_scatter=PD_Normal(+0.32, 0.1),
        )


class PS_Astro_Tight_03(PS_Broad_Uniform_01):

    def __init__(self, log, nsamples, sam_shape, seed):
        super(PS_Broad_Uniform_01, self).__init__(
            log, nsamples, sam_shape, seed,

            hard_time=PD_Uniform(1.0, 8.0),   # [Gyr]
            hard_gamma_inner=PD_Uniform(-1.5, -0.5),

            # from `sam-parameters.ipynb` fits to [Tomczak+2014] with 1x stdev values
            gsmf_phi0=PD_Normal(-2.56, 0.1),
            gsmf_mchar0_log10=PD_Normal(10.9, 0.1),   # [log10(Msol)]
            gsmf_alpha0=PD_Normal(-1.2, 0.05),

            gpf_zbeta=PD_Normal(+0.8, 0.4),
            gpf_qgamma=PD_Normal(+0.5, 0.3),

            gmt_norm=PD_Uniform(0.2, 5.0),    # [Gyr]
            gmt_zbeta=PD_Uniform(-2.0, +0.0),

            mmb_amp_log10=PD_Normal(+8.6, 0.1),   # [log10(Msol)]
            mmb_plaw=PD_Normal(+1.2, 0.1),
            mmb_scatter=PD_Normal(+0.32, 0.1),
        )

