"""
"""

import holodeck as holo
from holodeck.constants import PC, GYR, MSOL
from holodeck.librarian import (
    _Param_Space, PD_Uniform, PD_Piecewise_Uniform_Density,
    # PD_Normal,
    # PD_Uniform_Log,
)


# ! BAD -- error in ZERO_DYNAMIC_STALLED_SYSTEMS & ZERO_GMT_STALLED_SYSTEMS parameters !#
class PS_Broad_Uniform_02(_Param_Space):

    def __init__(self, log, nsamples, sam_shape, seed):
        raise RuntimeError(f"THERE WAS AN ERROR IN THIS PARAMETER SPACE")
        super().__init__(
            log, nsamples, sam_shape, seed,

            hard_time=PD_Uniform(0.2, 10.0),   # [Gyr]

            gsmf_phi0=PD_Uniform(-3.5, -1.5),
            gsmf_mchar0_log10=PD_Uniform(10.5, 12.0),   # [log10(Msol)]

            mmb_amp_log10=PD_Uniform(+7.5, +9.5),   # [log10(Msol)]
            mmb_scatter=PD_Uniform(+0.0, +0.6),
        )

    @classmethod
    def model_for_params(cls, params, sam_shape=None):

        hard_time = params['hard_time'] * GYR
        hard_gamma_inner = -1.0
        hard_rchar = 10.0 * PC
        hard_gamma_outer = +2.5
        hard_sepa_init = 1e4 * PC

        # Parameters are based on `sam-parameters.ipynb` fit to [Tomczak+2014]
        gsmf_phiz = -0.6
        gsmf_mcharz = 0.11
        gsmf_alpha0 = -1.21
        gsmf_alphaz = -0.03

        gpf_frac_norm_allq = 0.025
        gpf_malpha = 0.0
        gpf_qgamma = 0.0
        gpf_zbeta = 1.0
        gpf_max_frac = 1.0

        gmt_norm = 0.5 * GYR
        gmt_malpha = 0.0
        gmt_qgamma = -1.0   # Boylan-Kolchin+2008
        gmt_zbeta = -0.5

        mmb_plaw = 1.10   # average MM2013 and KH2013

        gsmf = holo.sam.GSMF_Schechter(
            phi0=params['gsmf_phi0'],
            phiz=gsmf_phiz,
            mchar0_log10=params['gsmf_mchar0_log10'],
            mcharz=gsmf_mcharz,
            alpha0=gsmf_alpha0,
            alphaz=gsmf_alphaz,
        )
        gpf = holo.sam.GPF_Power_Law(
            frac_norm_allq=gpf_frac_norm_allq,
            malpha=gpf_malpha,
            qgamma=gpf_qgamma,
            zbeta=gpf_zbeta,
            max_frac=gpf_max_frac,
        )
        gmt = holo.sam.GMT_Power_Law(
            time_norm=gmt_norm,
            malpha=gmt_malpha,
            qgamma=gmt_qgamma,
            zbeta=gmt_zbeta,
        )
        mmbulge = holo.relations.MMBulge_KH2013(
            mamp_log10=params['mmb_amp_log10'],
            mplaw=mmb_plaw,
            scatter_dex=params['mmb_scatter'],
        )

        kw = {} if sam_shape is None else dict(shape=sam_shape)
        sam = holo.sam.Semi_Analytic_Model(
            gsmf=gsmf, gpf=gpf, gmt=gmt, mmbulge=mmbulge,
            ZERO_DYNAMIC_STALLED_SYSTEMS=False,
            ZERO_GMT_STALLED_SYSTEMS=True,
            **kw
        )

        hard = holo.hardening.Fixed_Time.from_sam(
            sam,
            hard_time,
            sepa_init=hard_sepa_init,
            rchar=hard_rchar,
            gamma_sc=hard_gamma_inner,
            gamma_df=hard_gamma_outer,
            progress=False,
        )

        return sam, hard


class PS_Broad_Uniform_02B(_Param_Space):
    """This class fixes an issue with PS_Broad_Uniform_02 where the parameters:

    `ZERO_DYNAMIC_STALLED_SYSTEMS` was set to False (should be True), and
    `ZERO_GMT_STALLED_SYSTEMS` was set to True (should be False, but doesn't really matter)

    These are fixed in this class!

    """

    def __init__(self, log, nsamples, sam_shape, seed):
        super().__init__(
            log, nsamples, sam_shape, seed,

            hard_time=PD_Uniform(0.2, 10.0),   # [Gyr]

            gsmf_phi0=PD_Uniform(-3.5, -1.5),
            gsmf_mchar0_log10=PD_Uniform(10.5, 12.0),   # [log10(Msol)]

            mmb_amp_log10=PD_Uniform(+7.5, +9.5),   # [log10(Msol)]
            mmb_scatter=PD_Uniform(+0.0, +0.6),
        )

    @classmethod
    def model_for_params(cls, params, sam_shape=None):

        hard_time = params['hard_time'] * GYR
        hard_gamma_inner = -1.0
        hard_rchar = 10.0 * PC
        hard_gamma_outer = +2.5
        hard_sepa_init = 1e4 * PC

        # Parameters are based on `sam-parameters.ipynb` fit to [Tomczak+2014]
        gsmf_phiz = -0.6
        gsmf_mcharz = 0.11
        gsmf_alpha0 = -1.21
        gsmf_alphaz = -0.03

        gpf_frac_norm_allq = 0.025
        gpf_malpha = 0.0
        gpf_qgamma = 0.0
        gpf_zbeta = 1.0
        gpf_max_frac = 1.0

        gmt_norm = 0.5 * GYR
        gmt_malpha = 0.0
        gmt_qgamma = -1.0   # Boylan-Kolchin+2008
        gmt_zbeta = -0.5

        mmb_plaw = 1.10   # average MM2013 and KH2013

        gsmf = holo.sam.GSMF_Schechter(
            phi0=params['gsmf_phi0'],
            phiz=gsmf_phiz,
            mchar0_log10=params['gsmf_mchar0_log10'],
            mcharz=gsmf_mcharz,
            alpha0=gsmf_alpha0,
            alphaz=gsmf_alphaz,
        )
        gpf = holo.sam.GPF_Power_Law(
            frac_norm_allq=gpf_frac_norm_allq,
            malpha=gpf_malpha,
            qgamma=gpf_qgamma,
            zbeta=gpf_zbeta,
            max_frac=gpf_max_frac,
        )
        gmt = holo.sam.GMT_Power_Law(
            time_norm=gmt_norm,
            malpha=gmt_malpha,
            qgamma=gmt_qgamma,
            zbeta=gmt_zbeta,
        )
        mmbulge = holo.relations.MMBulge_KH2013(
            mamp_log10=params['mmb_amp_log10'],
            mplaw=mmb_plaw,
            scatter_dex=params['mmb_scatter'],
        )

        kw = {} if sam_shape is None else dict(shape=sam_shape)
        sam = holo.sam.Semi_Analytic_Model(
            gsmf=gsmf, gpf=gpf, gmt=gmt, mmbulge=mmbulge,
            ZERO_DYNAMIC_STALLED_SYSTEMS=True,
            ZERO_GMT_STALLED_SYSTEMS=False,
            **kw
        )

        hard = holo.hardening.Fixed_Time.from_sam(
            sam,
            hard_time,
            sepa_init=hard_sepa_init,
            rchar=hard_rchar,
            gamma_sc=hard_gamma_inner,
            gamma_df=hard_gamma_outer,
            progress=False,
        )

        return sam, hard


class PS_Broad_Uniform_02C(_Param_Space):
    """Change the hard_gamma_outer default parameter.
    """

    def __init__(self, log, nsamples, sam_shape, seed):
        super().__init__(
            log, nsamples, sam_shape, seed,

            hard_time=PD_Uniform(0.2, 10.0),   # [Gyr]

            gsmf_phi0=PD_Uniform(-3.5, -1.5),
            gsmf_mchar0_log10=PD_Uniform(10.5, 12.0),   # [log10(Msol)]

            mmb_amp_log10=PD_Uniform(+7.5, +9.5),   # [log10(Msol)]
            mmb_scatter=PD_Uniform(+0.0, +0.6),
        )

    @classmethod
    def model_for_params(cls, params, sam_shape=None):

        hard_time = params['hard_time'] * GYR
        hard_gamma_inner = -1.0
        hard_rchar = 10.0 * PC
        hard_gamma_outer = +1.0
        hard_sepa_init = 1e4 * PC

        # Parameters are based on `sam-parameters.ipynb` fit to [Tomczak+2014]
        gsmf_phiz = -0.6
        gsmf_mcharz = 0.11
        gsmf_alpha0 = -1.21
        gsmf_alphaz = -0.03

        gpf_frac_norm_allq = 0.025
        gpf_malpha = 0.0
        gpf_qgamma = 0.0
        gpf_zbeta = 1.0
        gpf_max_frac = 1.0

        gmt_norm = 0.5 * GYR
        gmt_malpha = 0.0
        gmt_qgamma = -1.0   # Boylan-Kolchin+2008
        gmt_zbeta = -0.5

        mmb_plaw = 1.10   # average MM2013 and KH2013

        gsmf = holo.sam.GSMF_Schechter(
            phi0=params['gsmf_phi0'],
            phiz=gsmf_phiz,
            mchar0_log10=params['gsmf_mchar0_log10'],
            mcharz=gsmf_mcharz,
            alpha0=gsmf_alpha0,
            alphaz=gsmf_alphaz,
        )
        gpf = holo.sam.GPF_Power_Law(
            frac_norm_allq=gpf_frac_norm_allq,
            malpha=gpf_malpha,
            qgamma=gpf_qgamma,
            zbeta=gpf_zbeta,
            max_frac=gpf_max_frac,
        )
        gmt = holo.sam.GMT_Power_Law(
            time_norm=gmt_norm,
            malpha=gmt_malpha,
            qgamma=gmt_qgamma,
            zbeta=gmt_zbeta,
        )
        mmbulge = holo.relations.MMBulge_KH2013(
            mamp_log10=params['mmb_amp_log10'],
            mplaw=mmb_plaw,
            scatter_dex=params['mmb_scatter'],
        )

        kw = {} if sam_shape is None else dict(shape=sam_shape)
        sam = holo.sam.Semi_Analytic_Model(
            gsmf=gsmf, gpf=gpf, gmt=gmt, mmbulge=mmbulge,
            ZERO_DYNAMIC_STALLED_SYSTEMS=True,
            ZERO_GMT_STALLED_SYSTEMS=False,
            **kw
        )

        hard = holo.hardening.Fixed_Time.from_sam(
            sam,
            hard_time,
            sepa_init=hard_sepa_init,
            rchar=hard_rchar,
            gamma_sc=hard_gamma_inner,
            gamma_df=hard_gamma_outer,
            progress=False,
        )

        return sam, hard


class PS_Broad_Uniform_02_GW(_Param_Space):

    def __init__(self, log, nsamples, sam_shape, seed):
        super().__init__(
            log, nsamples, sam_shape, seed,

            gsmf_phi0=PD_Uniform(-3.5, -1.5),
            gsmf_mchar0_log10=PD_Uniform(10.5, 12.0),   # [log10(Msol)]

            mmb_amp_log10=PD_Uniform(+7.5, +9.5),   # [log10(Msol)]
            mmb_scatter=PD_Uniform(+0.0, +0.6),
        )

    @classmethod
    def model_for_params(cls, params, sam_shape=None):

        # Parameters are based on `sam-parameters.ipynb` fit to [Tomczak+2014]
        gsmf_phiz = -0.6
        gsmf_mcharz = 0.11
        gsmf_alpha0 = -1.21
        gsmf_alphaz = -0.03

        gpf_frac_norm_allq = 0.025
        gpf_malpha = 0.0
        gpf_qgamma = 0.0
        gpf_zbeta = 1.0
        gpf_max_frac = 1.0

        gmt_norm = 0.5 * GYR
        gmt_malpha = 0.0
        gmt_qgamma = -1.0   # Boylan-Kolchin+2008
        gmt_zbeta = -0.5

        mmb_plaw = 1.10   # average MM2013 and KH2013

        gsmf = holo.sam.GSMF_Schechter(
            phi0=params['gsmf_phi0'],
            phiz=gsmf_phiz,
            mchar0_log10=params['gsmf_mchar0_log10'],
            mcharz=gsmf_mcharz,
            alpha0=gsmf_alpha0,
            alphaz=gsmf_alphaz,
        )
        gpf = holo.sam.GPF_Power_Law(
            frac_norm_allq=gpf_frac_norm_allq,
            malpha=gpf_malpha,
            qgamma=gpf_qgamma,
            zbeta=gpf_zbeta,
            max_frac=gpf_max_frac,
        )
        gmt = holo.sam.GMT_Power_Law(
            time_norm=gmt_norm,
            malpha=gmt_malpha,
            qgamma=gmt_qgamma,
            zbeta=gmt_zbeta,
        )
        mmbulge = holo.relations.MMBulge_KH2013(
            mamp_log10=params['mmb_amp_log10'],
            mplaw=mmb_plaw,
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


# ! BAD -- error in ZERO_DYNAMIC_STALLED_SYSTEMS & ZERO_GMT_STALLED_SYSTEMS parameters !#
class PS_Broad_Uniform_03(_Param_Space):

    def __init__(self, log, nsamples, sam_shape, seed):
        raise RuntimeError(f"THERE WAS AN ERROR IN THIS PARAMETER SPACE")
        super().__init__(
            log, nsamples, sam_shape, seed,

            hard_time=PD_Uniform(0.2, 10.0),   # [Gyr]

            gsmf_mchar0_log10=PD_Uniform(10.5, 12.0),   # [log10(Msol)]

            mmb_amp_log10=PD_Uniform(+7.5, +9.5),   # [log10(Msol)]
            mmb_scatter=PD_Uniform(+0.0, +0.6),
        )

    @classmethod
    def model_for_params(cls, params, sam_shape=None):

        hard_time = params['hard_time'] * GYR
        hard_gamma_inner = -1.0
        hard_rchar = 10.0 * PC
        hard_gamma_outer = +2.5
        hard_sepa_init = 1e4 * PC

        # Parameters are based on `sam-parameters.ipynb` fit to [Tomczak+2014]
        gsmf_phi0 = -2.57
        gsmf_phiz = -0.6
        gsmf_mcharz = 0.11
        gsmf_alpha0 = -1.21
        gsmf_alphaz = -0.03

        gpf_frac_norm_allq = 0.025
        gpf_malpha = 0.0
        gpf_qgamma = 0.0
        gpf_zbeta = 1.0
        gpf_max_frac = 1.0

        gmt_norm = 0.5 * GYR
        gmt_malpha = 0.0
        gmt_qgamma = -1.0   # Boylan-Kolchin+2008
        gmt_zbeta = -0.5

        mmb_plaw = 1.10   # average MM2013 and KH2013

        gsmf = holo.sam.GSMF_Schechter(
            phi0=gsmf_phi0,
            phiz=gsmf_phiz,
            mchar0_log10=params['gsmf_mchar0_log10'],
            mcharz=gsmf_mcharz,
            alpha0=gsmf_alpha0,
            alphaz=gsmf_alphaz,
        )
        gpf = holo.sam.GPF_Power_Law(
            frac_norm_allq=gpf_frac_norm_allq,
            malpha=gpf_malpha,
            qgamma=gpf_qgamma,
            zbeta=gpf_zbeta,
            max_frac=gpf_max_frac,
        )
        gmt = holo.sam.GMT_Power_Law(
            time_norm=gmt_norm,
            malpha=gmt_malpha,
            qgamma=gmt_qgamma,
            zbeta=gmt_zbeta,
        )
        mmbulge = holo.relations.MMBulge_KH2013(
            mamp_log10=params['mmb_amp_log10'],
            mplaw=mmb_plaw,
            scatter_dex=params['mmb_scatter'],
        )

        kw = {} if sam_shape is None else dict(shape=sam_shape)
        sam = holo.sam.Semi_Analytic_Model(
            gsmf=gsmf, gpf=gpf, gmt=gmt, mmbulge=mmbulge,
            ZERO_DYNAMIC_STALLED_SYSTEMS=False,
            ZERO_GMT_STALLED_SYSTEMS=True,
            **kw
        )

        hard = holo.hardening.Fixed_Time.from_sam(
            sam,
            hard_time,
            sepa_init=hard_sepa_init,
            rchar=hard_rchar,
            gamma_sc=hard_gamma_inner,
            gamma_df=hard_gamma_outer,
            progress=False,
        )

        return sam, hard


class PS_Broad_Uniform_03B(_Param_Space):
    """This class fixes an issue with PS_Broad_Uniform_03 where the parameters:

    `ZERO_DYNAMIC_STALLED_SYSTEMS` was set to False (should be True), and
    `ZERO_GMT_STALLED_SYSTEMS` was set to True (should be False, but doesn't really matter)

    These are fixed in this class!

    """

    def __init__(self, log, nsamples, sam_shape, seed):
        super().__init__(
            log, nsamples, sam_shape, seed,

            hard_time=PD_Uniform(0.2, 10.0),   # [Gyr]

            gsmf_mchar0_log10=PD_Uniform(10.5, 12.0),   # [log10(Msol)]

            mmb_amp_log10=PD_Uniform(+7.5, +9.5),   # [log10(Msol)]
            mmb_scatter=PD_Uniform(+0.0, +0.6),
        )

    @classmethod
    def model_for_params(cls, params, sam_shape=None):

        hard_time = params['hard_time'] * GYR
        hard_gamma_inner = -1.0
        hard_rchar = 10.0 * PC
        hard_gamma_outer = +2.5
        hard_sepa_init = 1e4 * PC

        # Parameters are based on `sam-parameters.ipynb` fit to [Tomczak+2014]
        gsmf_phi0 = -2.57
        gsmf_phiz = -0.6
        gsmf_mcharz = 0.11
        gsmf_alpha0 = -1.21
        gsmf_alphaz = -0.03

        gpf_frac_norm_allq = 0.025
        gpf_malpha = 0.0
        gpf_qgamma = 0.0
        gpf_zbeta = 1.0
        gpf_max_frac = 1.0

        gmt_norm = 0.5 * GYR
        gmt_malpha = 0.0
        gmt_qgamma = -1.0   # Boylan-Kolchin+2008
        gmt_zbeta = -0.5

        mmb_plaw = 1.10   # average MM2013 and KH2013

        gsmf = holo.sam.GSMF_Schechter(
            phi0=gsmf_phi0,
            phiz=gsmf_phiz,
            mchar0_log10=params['gsmf_mchar0_log10'],
            mcharz=gsmf_mcharz,
            alpha0=gsmf_alpha0,
            alphaz=gsmf_alphaz,
        )
        gpf = holo.sam.GPF_Power_Law(
            frac_norm_allq=gpf_frac_norm_allq,
            malpha=gpf_malpha,
            qgamma=gpf_qgamma,
            zbeta=gpf_zbeta,
            max_frac=gpf_max_frac,
        )
        gmt = holo.sam.GMT_Power_Law(
            time_norm=gmt_norm,
            malpha=gmt_malpha,
            qgamma=gmt_qgamma,
            zbeta=gmt_zbeta,
        )
        mmbulge = holo.relations.MMBulge_KH2013(
            mamp_log10=params['mmb_amp_log10'],
            mplaw=mmb_plaw,
            scatter_dex=params['mmb_scatter'],
        )

        kw = {} if sam_shape is None else dict(shape=sam_shape)
        sam = holo.sam.Semi_Analytic_Model(
            gsmf=gsmf, gpf=gpf, gmt=gmt, mmbulge=mmbulge,
            ZERO_DYNAMIC_STALLED_SYSTEMS=True,
            ZERO_GMT_STALLED_SYSTEMS=False,
            **kw
        )

        hard = holo.hardening.Fixed_Time.from_sam(
            sam,
            hard_time,
            sepa_init=hard_sepa_init,
            rchar=hard_rchar,
            gamma_sc=hard_gamma_inner,
            gamma_df=hard_gamma_outer,
            progress=False,
        )

        return sam, hard


class PS_Broad_Uniform_03_GW(_Param_Space):

    def __init__(self, log, nsamples, sam_shape, seed):
        super().__init__(
            log, nsamples, sam_shape, seed,

            gsmf_phi0=PD_Uniform(-3.5, -1.5),
            gsmf_mchar0_log10=PD_Uniform(10.5, 12.0),   # [log10(Msol)]

            mmb_amp_log10=PD_Uniform(+7.5, +9.5),   # [log10(Msol)]
            mmb_scatter=PD_Uniform(+0.0, +0.6),
        )

    @classmethod
    def model_for_params(cls, params, sam_shape=None):

        # Parameters are based on `sam-parameters.ipynb` fit to [Tomczak+2014]
        gsmf_phi0 = -2.57
        gsmf_phiz = -0.6
        gsmf_mcharz = 0.11
        gsmf_alpha0 = -1.21
        gsmf_alphaz = -0.03

        gpf_frac_norm_allq = 0.025
        gpf_malpha = 0.0
        gpf_qgamma = 0.0
        gpf_zbeta = 1.0
        gpf_max_frac = 1.0

        gmt_norm = 0.5 * GYR
        gmt_malpha = 0.0
        gmt_qgamma = -1.0   # Boylan-Kolchin+2008
        gmt_zbeta = -0.5

        mmb_plaw = 1.10   # average MM2013 and KH2013

        gsmf = holo.sam.GSMF_Schechter(
            phi0=gsmf_phi0,
            phiz=gsmf_phiz,
            mchar0_log10=params['gsmf_mchar0_log10'],
            mcharz=gsmf_mcharz,
            alpha0=gsmf_alpha0,
            alphaz=gsmf_alphaz,
        )
        gpf = holo.sam.GPF_Power_Law(
            frac_norm_allq=gpf_frac_norm_allq,
            malpha=gpf_malpha,
            qgamma=gpf_qgamma,
            zbeta=gpf_zbeta,
            max_frac=gpf_max_frac,
        )
        gmt = holo.sam.GMT_Power_Law(
            time_norm=gmt_norm,
            malpha=gmt_malpha,
            qgamma=gmt_qgamma,
            zbeta=gmt_zbeta,
        )
        mmbulge = holo.relations.MMBulge_KH2013(
            mamp_log10=params['mmb_amp_log10'],
            mplaw=mmb_plaw,
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


class PS_Broad_Uniform_04(PS_Broad_Uniform_02B):
    """Expand the mmb_scatter parameter from PS_Broad_Uniform_02B
    """

    def __init__(self, log, nsamples, sam_shape, seed):
        super(PS_Broad_Uniform_02B, self).__init__(
            log, nsamples, sam_shape, seed,

            hard_time=PD_Uniform(0.1, 11.0),   # [Gyr]

            gsmf_phi0=PD_Uniform(-3.5, -1.5),
            gsmf_mchar0_log10=PD_Uniform(10.5, 12.5),   # [log10(Msol)]

            mmb_amp_log10=PD_Uniform(+7.5, +9.5),   # [log10(Msol)]
            mmb_scatter=PD_Uniform(+0.0, +1.2),
        )


class PS_Simple_2Par_01(_Param_Space):
    """Updated version of the old 2Par and 2Par_Wider parameter-spaces.
    """

    def __init__(self, log, nsamples, sam_shape, seed):
        super().__init__(
            log, nsamples, sam_shape, seed,

            hard_time=PD_Uniform(0.1, 12.0),   # [Gyr]
            gsmf_phi0=PD_Uniform(-3.5, -1.5),
        )

    @classmethod
    def model_for_params(cls, params, sam_shape=None):

        hard_time = params['hard_time'] * GYR
        hard_gamma_inner = -1.0
        hard_rchar = 10.0 * PC
        hard_gamma_outer = +2.5
        hard_sepa_init = 1e4 * PC

        # Parameters are based on `sam-parameters.ipynb` fit to [Tomczak+2014]
        gsmf_phi0 = params['gsmf_phi0']     # -2.57
        gsmf_phiz = -0.6
        gsmf_mchar0_log10 = 11.24,
        gsmf_mcharz = 0.11
        gsmf_alpha0 = -1.21
        gsmf_alphaz = -0.03

        gpf_frac_norm_allq = 0.025
        gpf_malpha = 0.0
        gpf_qgamma = 0.0
        gpf_zbeta = 1.0
        gpf_max_frac = 1.0

        gmt_norm = 0.5 * GYR
        gmt_malpha = 0.0
        gmt_qgamma = -1.0   # Boylan-Kolchin+2008
        gmt_zbeta = -0.5

        # averages of MM2013 and KH2013
        mmb_amp_log10 = 8.575            # [log10(Msol)]
        mmb_plaw = 1.10
        mmb_scatter = 0.31

        gsmf = holo.sam.GSMF_Schechter(
            phi0=gsmf_phi0,
            phiz=gsmf_phiz,
            mchar0_log10=gsmf_mchar0_log10,
            mcharz=gsmf_mcharz,
            alpha0=gsmf_alpha0,
            alphaz=gsmf_alphaz,
        )
        gpf = holo.sam.GPF_Power_Law(
            frac_norm_allq=gpf_frac_norm_allq,
            malpha=gpf_malpha,
            qgamma=gpf_qgamma,
            zbeta=gpf_zbeta,
            max_frac=gpf_max_frac,
        )
        gmt = holo.sam.GMT_Power_Law(
            time_norm=gmt_norm,
            malpha=gmt_malpha,
            qgamma=gmt_qgamma,
            zbeta=gmt_zbeta,
        )
        mmbulge = holo.relations.MMBulge_KH2013(
            mamp_log10=mmb_amp_log10,
            mplaw=mmb_plaw,
            scatter_dex=mmb_scatter,
        )

        kw = {} if sam_shape is None else dict(shape=sam_shape)
        sam = holo.sam.Semi_Analytic_Model(
            gsmf=gsmf, gpf=gpf, gmt=gmt, mmbulge=mmbulge,
            ZERO_DYNAMIC_STALLED_SYSTEMS=True,
            ZERO_GMT_STALLED_SYSTEMS=False,
            **kw
        )

        hard = holo.hardening.Fixed_Time.from_sam(
            sam,
            hard_time,
            sepa_init=hard_sepa_init,
            rchar=hard_rchar,
            gamma_sc=hard_gamma_inner,
            gamma_df=hard_gamma_outer,
            progress=False,
        )

        return sam, hard


class PS_Simple_2Par_02(_Param_Space):

    def __init__(self, log, nsamples, sam_shape, seed):
        super().__init__(
            log, nsamples, sam_shape, seed,

            mmb_amp_log10=PD_Uniform(+7.5, +9.5),   # [log10(Msol)]
            mmb_scatter=PD_Uniform(+0.0, +1.2),
        )

    @classmethod
    def model_for_params(cls, params, sam_shape=None):

        hard_time = 3.0 * GYR
        hard_gamma_inner = -1.0
        hard_rchar = 10.0 * PC
        hard_gamma_outer = +2.5
        hard_sepa_init = 1e4 * PC

        # Parameters are based on `sam-parameters.ipynb` fit to [Tomczak+2014]
        gsmf_phi0 = -2.57
        gsmf_phiz = -0.6
        gsmf_mchar0_log10 = 11.24,
        gsmf_mcharz = 0.11
        gsmf_alpha0 = -1.21
        gsmf_alphaz = -0.03

        gpf_frac_norm_allq = 0.025
        gpf_malpha = 0.0
        gpf_qgamma = 0.0
        gpf_zbeta = 1.0
        gpf_max_frac = 1.0

        gmt_norm = 0.5 * GYR
        gmt_malpha = 0.0
        gmt_qgamma = -1.0   # Boylan-Kolchin+2008
        gmt_zbeta = -0.5

        # averages of MM2013 and KH2013
        mmb_amp_log10 = params['mmb_amp_log10']   # 8.575            # [log10(Msol)]
        mmb_plaw = 1.10
        mmb_scatter = params['mmb_scatter']  # 0.31

        gsmf = holo.sam.GSMF_Schechter(
            phi0=gsmf_phi0,
            phiz=gsmf_phiz,
            mchar0_log10=gsmf_mchar0_log10,
            mcharz=gsmf_mcharz,
            alpha0=gsmf_alpha0,
            alphaz=gsmf_alphaz,
        )
        gpf = holo.sam.GPF_Power_Law(
            frac_norm_allq=gpf_frac_norm_allq,
            malpha=gpf_malpha,
            qgamma=gpf_qgamma,
            zbeta=gpf_zbeta,
            max_frac=gpf_max_frac,
        )
        gmt = holo.sam.GMT_Power_Law(
            time_norm=gmt_norm,
            malpha=gmt_malpha,
            qgamma=gmt_qgamma,
            zbeta=gmt_zbeta,
        )
        mmbulge = holo.relations.MMBulge_KH2013(
            mamp_log10=mmb_amp_log10,
            mplaw=mmb_plaw,
            scatter_dex=mmb_scatter,
        )

        kw = {} if sam_shape is None else dict(shape=sam_shape)
        sam = holo.sam.Semi_Analytic_Model(
            gsmf=gsmf, gpf=gpf, gmt=gmt, mmbulge=mmbulge,
            ZERO_DYNAMIC_STALLED_SYSTEMS=True,
            ZERO_GMT_STALLED_SYSTEMS=False,
            **kw
        )

        hard = holo.hardening.Fixed_Time.from_sam(
            sam,
            hard_time,
            sepa_init=hard_sepa_init,
            rchar=hard_rchar,
            gamma_sc=hard_gamma_inner,
            gamma_df=hard_gamma_outer,
            progress=False,
        )

        return sam, hard


class PS_Simple_2Par_02B(_Param_Space):
    """Change the SAM mtot grid
    """
    
    def __init__(self, log, nsamples, sam_shape, seed):
        super().__init__(
            log, nsamples, sam_shape, seed,

            mmb_amp_log10=PD_Uniform(+7.5, +9.5),   # [log10(Msol)]
            mmb_scatter=PD_Uniform(+0.0, +1.2),
        )

    @classmethod
    def model_for_params(cls, params, sam_shape=None):

        hard_time = 3.0 * GYR
        hard_gamma_inner = -1.0
        hard_rchar = 10.0 * PC
        hard_gamma_outer = +2.5
        hard_sepa_init = 1e4 * PC

        # Parameters are based on `sam-parameters.ipynb` fit to [Tomczak+2014]
        gsmf_phi0 = -2.57
        gsmf_phiz = -0.6
        gsmf_mchar0_log10 = 11.24,
        gsmf_mcharz = 0.11
        gsmf_alpha0 = -1.21
        gsmf_alphaz = -0.03

        gpf_frac_norm_allq = 0.025
        gpf_malpha = 0.0
        gpf_qgamma = 0.0
        gpf_zbeta = 1.0
        gpf_max_frac = 1.0

        gmt_norm = 0.5 * GYR
        gmt_malpha = 0.0
        gmt_qgamma = -1.0   # Boylan-Kolchin+2008
        gmt_zbeta = -0.5

        # averages of MM2013 and KH2013
        mmb_amp_log10 = params['mmb_amp_log10']   # 8.575            # [log10(Msol)]
        mmb_plaw = 1.10
        mmb_scatter = params['mmb_scatter']  # 0.31

        gsmf = holo.sam.GSMF_Schechter(
            phi0=gsmf_phi0,
            phiz=gsmf_phiz,
            mchar0_log10=gsmf_mchar0_log10,
            mcharz=gsmf_mcharz,
            alpha0=gsmf_alpha0,
            alphaz=gsmf_alphaz,
        )
        gpf = holo.sam.GPF_Power_Law(
            frac_norm_allq=gpf_frac_norm_allq,
            malpha=gpf_malpha,
            qgamma=gpf_qgamma,
            zbeta=gpf_zbeta,
            max_frac=gpf_max_frac,
        )
        gmt = holo.sam.GMT_Power_Law(
            time_norm=gmt_norm,
            malpha=gmt_malpha,
            qgamma=gmt_qgamma,
            zbeta=gmt_zbeta,
        )
        mmbulge = holo.relations.MMBulge_KH2013(
            mamp_log10=mmb_amp_log10,
            mplaw=mmb_plaw,
            scatter_dex=mmb_scatter,
        )

        # kw = {} if sam_shape is None else dict(shape=sam_shape)
        sam = holo.sam.Semi_Analytic_Model(
            mtot=(1.0e4*MSOL, 1.0e12*MSOL, 111),
            gsmf=gsmf, gpf=gpf, gmt=gmt, mmbulge=mmbulge,
            ZERO_DYNAMIC_STALLED_SYSTEMS=True,
            ZERO_GMT_STALLED_SYSTEMS=False,
        )

        hard = holo.hardening.Fixed_Time.from_sam(
            sam,
            hard_time,
            sepa_init=hard_sepa_init,
            rchar=hard_rchar,
            gamma_sc=hard_gamma_inner,
            gamma_df=hard_gamma_outer,
            progress=False,
        )

        return sam, hard


class PS_Generic(_Param_Space):
    """
    """

    #! This is included as an example:
    # def __init__(self, log, nsamples, sam_shape, seed):
    #     super(_Param_Space, self).__init__(
    #         log, nsamples, sam_shape, seed,
    #         hard_time=PD_Uniform(0.1, 11.0),   # [Gyr]
    #         gsmf_phi0=PD_Uniform(-3.5, -1.5),
    #         gsmf_mchar0_log10=PD_Uniform(10.5, 12.5),   # [log10(Msol)]
    #         mmb_amp_log10=PD_Uniform(+7.5, +9.5),   # [log10(Msol)]
    #         mmb_scatter=PD_Uniform(+0.0, +1.2),
    #     )

    @classmethod
    def model_for_params(cls, params, sam_shape=None, new_def_params={}):
        """Construct a model (SAM and hardening instances) from the given parameters.

        Arguments
        ---------
        params : dict
            Key-value pairs for sam/hardening parameters.  Each item much match expected parameters
            that are set in the `defaults` dictionary.
        sam_shape : None  or  int  or  (3,) int
        new_def_params : dict
            Key-value pairs to override default parameters.  This should be used for subclassing,
            so that this entire method does not need to be re-written.
            For example, to set a new default value, use something like:
            `super().model_for_params(params, sam_shape=sam_shape, new_def_params=dict(hard_rchar=1*PC))`

        Returns
        -------
        sam : `holodeck.sam.Semi_Analytic_Model` instance
        hard : `holodeck.hardening._Hardening` instance

        """

        # ---- Set default parameters

        defaults = dict(
            hard_time=3.0,               # [Gyr]
            hard_gamma_inner=-1.0,
            hard_rchar=10.0,             # [pc]
            hard_gamma_outer=+2.5,
            hard_sepa_init=1e4,       # [pc]

            # Parameters are based on `sam-parameters.ipynb` fit to [Tomczak+2014]
            gsmf_phi0=-2.77,
            gsmf_phiz=-0.6,
            gsmf_mchar0_log10=11.24,
            gsmf_mcharz=0.11,
            gsmf_alpha0=-1.21,
            gsmf_alphaz=-0.03,

            gpf_frac_norm_allq=0.025,
            gpf_malpha=0.0,
            gpf_qgamma=0.0,
            gpf_zbeta=1.0,
            gpf_max_frac=1.0,

            gmt_norm=0.5,      # [Gyr]
            gmt_malpha=0.0,
            gmt_qgamma=-1.0,        # Boylan-Kolchin+2008
            gmt_zbeta=-0.5,

            mmb_mamp_log10=8.69,
            mmb_plaw=1.10,          # average MM2013 and KH2013
            mmb_scatter_dex=0.3,
        )

        # ---- Update default parameters with input parameters

        if len(params) < 1:
            err = "No `params` included in call to `model_for_params`!"
            raise ValueError(err)

        # Update parameters specified in sub-classes

        for kk, vv in new_def_params.items():
            if kk not in defaults:
                err = f"`new_def_params` has key '{kk}' not found in defaults!  ({defaults.keys()})!"
                raise ValueError(err)
            defaults[kk] = vv

        # Update parameters passes in using the `params` dict, typically from LHC sampling

        for kk, vv in params.items():
            if kk not in defaults:
                err = f"`params` has key '{kk}' not found in defaults!  ({defaults.keys()})!"
                raise ValueError(err)
            if kk in new_def_params:
                err = f"`params` has key '{kk}' which is also in `new_def_params`!  ({new_def_params.keys()})!"
                raise ValueError(err)

            defaults[kk] = vv

        # ---- Construct SAM and hardening model

        gsmf = holo.sam.GSMF_Schechter(
            phi0=defaults['gsmf_phi0'],
            phiz=defaults['gsmf_phiz'],
            mchar0_log10=defaults['gsmf_mchar0_log10'],
            mcharz=defaults['gsmf_mcharz'],
            alpha0=defaults['gsmf_alpha0'],
            alphaz=defaults['gsmf_alphaz'],
        )
        gpf = holo.sam.GPF_Power_Law(
            frac_norm_allq=defaults['gpf_frac_norm_allq'],
            malpha=defaults['gpf_malpha'],
            qgamma=defaults['gpf_qgamma'],
            zbeta=defaults['gpf_zbeta'],
            max_frac=defaults['gpf_max_frac'],
        )
        gmt = holo.sam.GMT_Power_Law(
            time_norm=defaults['gmt_norm']*GYR,
            malpha=defaults['gmt_malpha'],
            qgamma=defaults['gmt_qgamma'],
            zbeta=defaults['gmt_zbeta'],
        )
        mmbulge = holo.relations.MMBulge_KH2013(
            mamp_log10=defaults['mmb_mamp_log10'],
            mplaw=defaults['mmb_plaw'],
            scatter_dex=defaults['mmb_scatter_dex'],
        )

        sam = holo.sam.Semi_Analytic_Model(
            gsmf=gsmf, gpf=gpf, gmt=gmt, mmbulge=mmbulge,
            ZERO_DYNAMIC_STALLED_SYSTEMS=True,
            ZERO_GMT_STALLED_SYSTEMS=False,
            shape=sam_shape,
        )

        hard = holo.hardening.Fixed_Time.from_sam(
            sam,
            defaults['hard_time']*GYR,
            sepa_init=defaults['hard_sepa_init']*PC,
            rchar=defaults['hard_rchar']*PC,
            gamma_sc=defaults['hard_gamma_inner'],
            gamma_df=defaults['hard_gamma_outer'],
            progress=False,
        )

        return sam, hard


# ==============================================================================
# ====    Uniform 05    ====
# ==============================================================================


class _PS_Uniform_05(PS_Generic):

    def __init__(self, log, nsamples, sam_shape, seed):
        super().__init__(
            log, nsamples, sam_shape, seed,
            hard_time=PD_Uniform(0.1, 11.0),   # [Gyr]
            gsmf_phi0=PD_Uniform(-3.5, -1.5),
            gsmf_mchar0_log10=PD_Uniform(10.5, 12.5),   # [log10(Msol)]
            mmb_mamp_log10=PD_Uniform(+7.5, +9.5),   # [log10(Msol)]
            mmb_scatter_dex=PD_Uniform(+0.0, +1.2),
        )


class PS_Uniform_05A(_PS_Uniform_05):

    @classmethod
    def model_for_params(cls, params, sam_shape=None, new_def_params={}):
        # NOTE: these should be the same as the default case, just duplicating them here for clarity
        new_def_params = dict(
            hard_gamma_inner=-1.0,
            hard_rchar=10.0,            # [pc]
            hard_gamma_outer=+2.5,
        )
        return super().model_for_params(params, sam_shape=sam_shape, new_def_params=new_def_params)


class PS_Uniform_05B(_PS_Uniform_05):

    @classmethod
    def model_for_params(cls, params, sam_shape=None, new_def_params={}):
        # NOTE: these should be the same as the default case, just duplicating them here for clarity
        new_def_params = dict(
            hard_gamma_inner=-1.0,
            hard_rchar=100.0,               # [pc]
            hard_gamma_outer=+2.5,
        )
        return super().model_for_params(params, sam_shape=sam_shape, new_def_params=new_def_params)


class PS_Uniform_05C(_PS_Uniform_05):

    @classmethod
    def model_for_params(cls, params, sam_shape=None, new_def_params={}):
        # NOTE: these should be the same as the default case, just duplicating them here for clarity
        new_def_params = dict(
            hard_gamma_inner=-1.0,
            hard_rchar=10.0,            # [pc]
            hard_gamma_outer=+1.0,
        )
        return super().model_for_params(params, sam_shape=sam_shape, new_def_params=new_def_params)


class PS_Uniform_05D(_PS_Uniform_05):

    @classmethod
    def model_for_params(cls, params, sam_shape=None, new_def_params={}):
        # NOTE: these should be the same as the default case, just duplicating them here for clarity
        new_def_params = dict(
            hard_gamma_inner=-1.0,
            hard_rchar=100.0,            # [pc]
            hard_gamma_outer=+1.0,
        )
        return super().model_for_params(params, sam_shape=sam_shape, new_def_params=new_def_params)


class PS_Uniform_05E(_PS_Uniform_05):

    @classmethod
    def model_for_params(cls, params, sam_shape=None, new_def_params={}):
        # NOTE: these should be the same as the default case, just duplicating them here for clarity
        new_def_params = dict(
            hard_gamma_inner=0.0,
            hard_rchar=10.0,            # [pc]
            hard_gamma_outer=+1.0,
        )
        return super().model_for_params(params, sam_shape=sam_shape, new_def_params=new_def_params)


class PS_Uniform_05F(_PS_Uniform_05):

    @classmethod
    def model_for_params(cls, params, sam_shape=None, new_def_params={}):
        # NOTE: these should be the same as the default case, just duplicating them here for clarity
        new_def_params = dict(
            hard_gamma_inner=0.0,
            hard_rchar=10.0,            # [pc]
            hard_gamma_outer=+2.5,
        )
        return super().model_for_params(params, sam_shape=sam_shape, new_def_params=new_def_params)


# ==============================================================================
# ====    Uniform 06    ====
# ==============================================================================


class PS_Uniform_06(PS_Generic):
    """New version of PS_Broad_Uniform_04 after updating redshifts used in the GWB calculation.
    """

    def __init__(self, log, nsamples, sam_shape, seed):
        super().__init__(
            log, nsamples, sam_shape, seed,
            hard_time=PD_Uniform(0.1, 11.0),   # [Gyr]
            gsmf_phi0=PD_Uniform(-3.5, -1.5),
            gsmf_mchar0_log10=PD_Uniform(10.5, 12.5),   # [log10(Msol)]
            mmb_mamp_log10=PD_Uniform(+7.5, +9.5),   # [log10(Msol)]
            mmb_scatter_dex=PD_Uniform(+0.0, +1.2),
        )


class PS_Uniform_06B(PS_Generic):
    """NON-Uniform spacing in `hard_time`
    """

    def __init__(self, log, nsamples, sam_shape, seed):
        super().__init__(
            log, nsamples, sam_shape, seed,
            # hard_time=PD_Uniform(0.1, 11.0),   # [Gyr]
            hard_time=PD_Piecewise_Uniform_Density([0.1, 1.0, 2.0, 9.0, 11.0], [2.5, 1.5, 1.0, 1.5]),   # [Gyr]
            gsmf_phi0=PD_Uniform(-3.5, -1.5),
            gsmf_mchar0_log10=PD_Uniform(10.5, 12.5),   # [log10(Msol)]
            mmb_mamp_log10=PD_Uniform(+7.5, +9.5),   # [log10(Msol)]
            mmb_scatter_dex=PD_Uniform(+0.0, +1.2),
        )


class PS_Uniform_06C(PS_Generic):
    """NON-Uniform spacing in all parameters.
    """

    def __init__(self, log, nsamples, sam_shape, seed):
        super().__init__(
            log, nsamples, sam_shape, seed,
            # hard_time=PD_Uniform(0.1, 11.0),   # [Gyr]
            hard_time=PD_Piecewise_Uniform_Density([0.1, 1.0, 2.0, 9.0, 11.0], [2.5, 1.5, 1.0, 1.5]),   # [Gyr]
            # gsmf_phi0=PD_Uniform(-3.5, -1.5),
            gsmf_phi0=PD_Piecewise_Uniform_Density([-3.5, -3.0, -2.0, -1.5], [2.0, 1.0, 2.0]),
            # gsmf_mchar0_log10=PD_Uniform(10.5, 12.5),   # [log10(Msol)]
            gsmf_mchar0_log10=PD_Piecewise_Uniform_Density([10.5, 11.0, 12.0, 12.5], [2.0, 1.0, 2.0]),   # [log10(Msol)]
            # mmb_mamp_log10=PD_Uniform(+7.5, +9.5),   # [log10(Msol)]
            mmb_mamp_log10=PD_Piecewise_Uniform_Density([7.5, 8.0, 9.0, 9.5], [1.5, 1.0, 2.0]),   # [log10(Msol)]
            # mmb_scatter_dex=PD_Uniform(+0.0, +1.2),
            mmb_scatter_dex=PD_Piecewise_Uniform_Density([0.0, 0.2, 1.0, 1.2], [1.5, 1.0, 2.0]),
        )
