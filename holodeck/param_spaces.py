"""
"""

import holodeck as holo
from holodeck.constants import PC, GYR, MSOL
from holodeck.librarian import (
    _Param_Space, PD_Uniform, PD_Piecewise_Uniform_Density,
    # PD_Normal,
    # PD_Uniform_Log,
)


class PS_Generic_1(_Param_Space):
    """
    """

    DEFAULTS = dict(
        hard_time=3.0,          # [Gyr]
        hard_gamma_inner=-1.0,
        hard_rchar=10.0,        # [pc]
        hard_gamma_outer=+2.5,
        hard_sepa_init=1e4,     # [pc]

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

        gmt_norm=0.5,           # [Gyr]
        gmt_malpha=0.0,
        gmt_qgamma=-1.0,        # Boylan-Kolchin+2008
        gmt_zbeta=-0.5,

        mmb_mamp_log10=8.69,
        mmb_plaw=1.10,          # average MM2013 and KH2013
        mmb_scatter_dex=0.3,
    )

    # ! This is included as an example:
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

        # ---- Update default parameters with input parameters

        settings = cls.DEFAULTS.copy()

        if len(params) < 1:
            err = "No `params` included in call to `model_for_params`!"
            raise ValueError(err)

        # Update parameters specified in sub-classes

        for kk, vv in new_def_params.items():
            if kk not in settings:
                err = f"`new_def_params` has key '{kk}' not found in settings!  ({settings.keys()})!"
                raise ValueError(err)
            settings[kk] = vv

        # Update parameters passes in using the `params` dict, typically from LHC sampling

        for kk, vv in params.items():
            if kk not in settings:
                err = f"`params` has key '{kk}' not found in settings!  ({settings.keys()})!"
                raise ValueError(err)
            if kk in new_def_params:
                err = f"`params` has key '{kk}' which is also in `new_def_params`!  ({new_def_params.keys()})!"
                raise ValueError(err)

            settings[kk] = vv

        # ---- Construct SAM and hardening model

        sam = cls._init_sam(sam_shape, settings)
        hard = cls._init_hard(sam, settings)

        return sam, hard

    @classmethod
    def _init_sam(cls, sam_shape, settings):
        gsmf = holo.sam.GSMF_Schechter(
            phi0=settings['gsmf_phi0'],
            phiz=settings['gsmf_phiz'],
            mchar0_log10=settings['gsmf_mchar0_log10'],
            mcharz=settings['gsmf_mcharz'],
            alpha0=settings['gsmf_alpha0'],
            alphaz=settings['gsmf_alphaz'],
        )
        gpf = holo.sam.GPF_Power_Law(
            frac_norm_allq=settings['gpf_frac_norm_allq'],
            malpha=settings['gpf_malpha'],
            qgamma=settings['gpf_qgamma'],
            zbeta=settings['gpf_zbeta'],
            max_frac=settings['gpf_max_frac'],
        )
        gmt = holo.sam.GMT_Power_Law(
            time_norm=settings['gmt_norm']*GYR,
            malpha=settings['gmt_malpha'],
            qgamma=settings['gmt_qgamma'],
            zbeta=settings['gmt_zbeta'],
        )
        mmbulge = holo.relations.MMBulge_KH2013(
            mamp_log10=settings['mmb_mamp_log10'],
            mplaw=settings['mmb_plaw'],
            scatter_dex=settings['mmb_scatter_dex'],
        )

        sam = holo.sam.Semi_Analytic_Model(
            gsmf=gsmf, gpf=gpf, gmt=gmt, mmbulge=mmbulge,
            ZERO_DYNAMIC_STALLED_SYSTEMS=True,
            ZERO_GMT_STALLED_SYSTEMS=False,
            shape=sam_shape,
        )
        return sam

    @classmethod
    def _init_hard(cls, sam, settings):
        hard = holo.hardening.Fixed_Time.from_sam(
            sam,
            settings['hard_time']*GYR,
            sepa_init=settings['hard_sepa_init']*PC,
            rchar=settings['hard_rchar']*PC,
            gamma_sc=settings['hard_gamma_inner'],
            gamma_df=settings['hard_gamma_outer'],
            progress=False,
        )
        return hard


class PS_Generic_2(PS_Generic_1):
    """
    """

    @classmethod
    def _init_hard(cls, sam, settings):
        # hard = holo.hardening.Fixed_Time_2PL.from_sam(
        hard = holo.hardening.Fixed_Time_2PL_SAM(
            sam,
            settings['hard_time']*GYR,
            sepa_init=settings['hard_sepa_init']*PC,
            rchar=settings['hard_rchar']*PC,
            gamma_inner=settings['hard_gamma_inner'],
            gamma_outer=settings['hard_gamma_outer'],
        )
        return hard

    @classmethod
    def _init_sam(cls, sam_shape, settings):
        gsmf = holo.sam.GSMF_Schechter(
            phi0=settings['gsmf_phi0'],
            phiz=settings['gsmf_phiz'],
            mchar0_log10=settings['gsmf_mchar0_log10'],
            mcharz=settings['gsmf_mcharz'],
            alpha0=settings['gsmf_alpha0'],
            alphaz=settings['gsmf_alphaz'],
        )
        gpf = holo.sam.GPF_Power_Law(
            frac_norm_allq=settings['gpf_frac_norm_allq'],
            malpha=settings['gpf_malpha'],
            qgamma=settings['gpf_qgamma'],
            zbeta=settings['gpf_zbeta'],
            max_frac=settings['gpf_max_frac'],
        )
        gmt = holo.sam.GMT_Power_Law(
            time_norm=settings['gmt_norm']*GYR,
            malpha=settings['gmt_malpha'],
            qgamma=settings['gmt_qgamma'],
            zbeta=settings['gmt_zbeta'],
        )
        mmbulge = holo.relations.MMBulge_KH2013(
            mamp_log10=settings['mmb_mamp_log10'],
            mplaw=settings['mmb_plaw'],
            scatter_dex=settings['mmb_scatter_dex'],
        )

        sam = holo.sam.Semi_Analytic_Model(
            gsmf=gsmf, gpf=gpf, gmt=gmt, mmbulge=mmbulge,
            shape=sam_shape,
        )
        return sam


# ==============================================================================
# ====    Uniform 05    ====
# ==============================================================================


class _PS_Uniform_05(PS_Generic_1):

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


class PS_Uniform_06(PS_Generic_1):
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


class PS_Uniform_06B(PS_Generic_1):
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


class PS_Uniform_06C(PS_Generic_1):
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


# ==============================================================================
# ====    Uniform 07 / PS_Generic_2    ====
# ==============================================================================


class PS_Uniform_07A(PS_Generic_2):
    """Use `Fixed_Time_2PL` (in `PS_Generic_2`) instead of `Fixed_Time`
    """

    def __init__(self, log, nsamples, sam_shape, seed):
        super().__init__(
            log, nsamples, sam_shape, seed,
            hard_time=PD_Uniform(0.1, 11.0),   # [Gyr]
            gsmf_phi0=PD_Uniform(-3.5, -1.5),
            gsmf_mchar0_log10=PD_Uniform(10.5, 12.5),   # [log10(Msol)]
            mmb_mamp_log10=PD_Uniform(+7.6, +9.0),   # [log10(Msol)]
            mmb_scatter_dex=PD_Uniform(+0.0, +0.9),
        )

    @classmethod
    def model_for_params(cls, params, sam_shape=None, new_def_params={}):
        # NOTE: these should be the same as the default case, just duplicating them here for clarity
        new_def_params = dict(
            hard_gamma_inner=-1.0,
            hard_rchar=100.0,               # [pc]
            hard_gamma_outer=+1.5,
            hard_sepa_init=1e4,     # [pc]
        )
        return super().model_for_params(params, sam_shape=sam_shape, new_def_params=new_def_params)


class PS_Uniform_07B(PS_Generic_2):
    """Use `Fixed_Time_2PL` (in `PS_Generic_2`) instead of `Fixed_Time`
    """

    def __init__(self, log, nsamples, sam_shape, seed):
        super().__init__(
            log, nsamples, sam_shape, seed,
            hard_time=PD_Uniform(0.1, 11.0),   # [Gyr]
            gsmf_phi0=PD_Uniform(-3.5, -1.5),
            gsmf_mchar0_log10=PD_Uniform(10.5, 12.5),   # [log10(Msol)]
            mmb_mamp_log10=PD_Uniform(+7.6, +9.0),   # [log10(Msol)]
            mmb_scatter_dex=PD_Uniform(+0.0, +0.9),
        )

    @classmethod
    def model_for_params(cls, params, sam_shape=None, new_def_params={}):
        # NOTE: these should be the same as the default case, just duplicating them here for clarity
        new_def_params = dict(
            hard_gamma_inner=-1.0,
            hard_rchar=100.0,               # [pc]
            hard_gamma_outer=+2.5,
            hard_sepa_init=1e4,     # [pc]
        )
        return super().model_for_params(params, sam_shape=sam_shape, new_def_params=new_def_params)


class PS_Uniform_07C(PS_Generic_2):
    """Use `Fixed_Time_2PL` (in `PS_Generic_2`) instead of `Fixed_Time`
    """

    def __init__(self, log, nsamples, sam_shape, seed):
        super().__init__(
            log, nsamples, sam_shape, seed,
            hard_time=PD_Uniform(0.1, 11.0),   # [Gyr]
            gsmf_phi0=PD_Uniform(-3.5, -1.5),
            gsmf_mchar0_log10=PD_Uniform(10.5, 12.5),   # [log10(Msol)]
            mmb_mamp_log10=PD_Uniform(+7.6, +9.0),   # [log10(Msol)]
            mmb_scatter_dex=PD_Uniform(+0.0, +0.9),
        )

    @classmethod
    def model_for_params(cls, params, sam_shape=None, new_def_params={}):
        # NOTE: these should be the same as the default case, just duplicating them here for clarity
        new_def_params = dict(
            hard_gamma_inner=-1.0,
            hard_rchar=30.0,               # [pc]
            hard_gamma_outer=+1.5,
            hard_sepa_init=3e3,     # [pc]
        )
        return super().model_for_params(params, sam_shape=sam_shape, new_def_params=new_def_params)


class PS_Uniform_07_GW(PS_Generic_2):
    """Use `Fixed_Time_2PL` (in `PS_Generic_2`) instead of `Fixed_Time`
    """

    def __init__(self, log, nsamples, sam_shape, seed):
        super().__init__(
            log, nsamples, sam_shape, seed,
            gsmf_phi0=PD_Uniform(-3.5, -1.5),
            gsmf_mchar0_log10=PD_Uniform(10.5, 12.5),   # [log10(Msol)]
            mmb_mamp_log10=PD_Uniform(+7.6, +9.0),   # [log10(Msol)]
            mmb_scatter_dex=PD_Uniform(+0.0, +0.9),
        )

    @classmethod
    def _init_hard(cls, sam, settings):
        hard = holo.hardening.Hard_GW()
        return hard
