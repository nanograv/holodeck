"""
"""

import holodeck as holo
from holodeck.constants import PC, GYR, MSOL
from holodeck.librarian import (
    _Param_Space, PD_Uniform, PD_Piecewise_Uniform_Density,
    PD_Normal,
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
    #         mmb_mamp_log10=PD_Uniform(+7.5, +9.5),   # [log10(Msol)]
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
        # hard = holo.hardening.Fixed_Time_2PL.from_sam(   # OLD
        hard = holo.hardening.Fixed_Time_2PL_SAM(     # NEW
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
        defs = dict(
            hard_gamma_inner=-1.0,
            hard_rchar=100.0,               # [pc]
            hard_gamma_outer=+1.5,
            hard_sepa_init=1e4,     # [pc]
        )
        defs.update(new_def_params)
        return super().model_for_params(params, sam_shape=sam_shape, new_def_params=defs)


class PS_Uniform_07A_Rot(PS_Generic_2):
    """Same as PS_Uniform_07A, but adding a hardening-rate power-law rotation.
    """

    # `DEFAULTS` must have a copy of all settings that are used, so make a copy and expand it
    DEFAULTS = PS_Generic_2.DEFAULTS.copy()
    DEFAULTS['hard_gamma_rot'] = 0.0

    def __init__(self, log, nsamples, sam_shape, seed):
        super().__init__(
            log, nsamples, sam_shape, seed,
            hard_time=PD_Uniform(0.1, 11.0),   # [Gyr]
            gsmf_phi0=PD_Uniform(-3.5, -1.5),
            gsmf_mchar0_log10=PD_Uniform(10.5, 12.5),   # [log10(Msol)]
            mmb_mamp_log10=PD_Uniform(+7.6, +9.0),   # [log10(Msol)]
            mmb_scatter_dex=PD_Uniform(+0.0, +0.9),
            hard_gamma_rot=PD_Uniform(-0.5, 0.5),
        )

    @classmethod
    def _init_hard(cls, sam, settings):
        rotation = settings['hard_gamma_rot']
        gamma_inner = settings['hard_gamma_inner'] + rotation
        gamma_outer = settings['hard_gamma_outer'] + rotation
        hard = holo.hardening.Fixed_Time_2PL_SAM(
            sam,
            settings['hard_time']*GYR,
            sepa_init=settings['hard_sepa_init']*PC,
            rchar=settings['hard_rchar']*PC,
            gamma_inner=gamma_inner,
            gamma_outer=gamma_outer,
        )
        return hard

    @classmethod
    def model_for_params(cls, params, sam_shape=None, new_def_params={}):
        # NOTE: these should be the same as the default case, just duplicating them here for clarity
        defs = dict(
            hard_gamma_inner=-1.0,
            hard_rchar=100.0,               # [pc]
            hard_gamma_outer=+1.5,
            hard_sepa_init=1e4,     # [pc]
        )
        defs.update(new_def_params)
        return super().model_for_params(params, sam_shape=sam_shape, new_def_params=defs)


class PS_Uniform_07A_Rot_Test(PS_Generic_2):
    """Same as PS_Uniform_07A, but adding a hardening-rate power-law rotation.
    """

    # `DEFAULTS` must have a copy of all settings that are used, so make a copy and expand it
    DEFAULTS = PS_Generic_2.DEFAULTS.copy()
    DEFAULTS['hard_gamma_rot'] = 0.0

    def __init__(self, log, nsamples, sam_shape, seed):
        super().__init__(
            log, nsamples, sam_shape, seed,
            hard_time=PD_Uniform(0.1, 11.0),   # [Gyr]
            hard_gamma_rot=PD_Uniform(-0.5, 0.5),
        )

    @classmethod
    def _init_hard(cls, sam, settings):
        rotation = settings['hard_gamma_rot']
        gamma_inner = settings['hard_gamma_inner'] + rotation
        gamma_outer = settings['hard_gamma_outer'] + rotation
        hard = holo.hardening.Fixed_Time_2PL_SAM(
            sam,
            settings['hard_time']*GYR,
            sepa_init=settings['hard_sepa_init']*PC,
            rchar=settings['hard_rchar']*PC,
            gamma_inner=gamma_inner,
            gamma_outer=gamma_outer,
        )
        return hard

    @classmethod
    def model_for_params(cls, params, sam_shape=None, new_def_params={}):
        # NOTE: these should be the same as the default case, just duplicating them here for clarity
        defs = dict(
            hard_gamma_inner=-1.0,
            hard_rchar=100.0,               # [pc]
            hard_gamma_outer=+1.5,
            hard_sepa_init=1e4,     # [pc]
        )
        defs.update(new_def_params)
        return super().model_for_params(params, sam_shape=sam_shape, new_def_params=defs)


class PS_Uniform_07A_Rot_2(PS_Generic_2):
    """Change `hard_gamma_rot` relative to PS_Uniform_07A_Rot
    """

    # `DEFAULTS` must have a copy of all settings that are used, so make a copy and expand it
    DEFAULTS = PS_Generic_2.DEFAULTS.copy()
    DEFAULTS['hard_gamma_rot'] = 0.0

    def __init__(self, log, nsamples, sam_shape, seed):
        super().__init__(
            log, nsamples, sam_shape, seed,
            hard_time=PD_Uniform(0.1, 11.0),   # [Gyr]
            gsmf_phi0=PD_Uniform(-3.5, -1.5),
            gsmf_mchar0_log10=PD_Uniform(10.5, 12.5),   # [log10(Msol)]
            mmb_mamp_log10=PD_Uniform(+7.6, +9.0),   # [log10(Msol)]
            mmb_scatter_dex=PD_Uniform(+0.0, +0.9),
            hard_gamma_rot=PD_Uniform(0.0, 1.0),
        )

    @classmethod
    def _init_hard(cls, sam, settings):
        rotation = settings['hard_gamma_rot']
        gamma_inner = settings['hard_gamma_inner'] + rotation
        gamma_outer = settings['hard_gamma_outer'] + rotation
        hard = holo.hardening.Fixed_Time_2PL_SAM(
            sam,
            settings['hard_time']*GYR,
            sepa_init=settings['hard_sepa_init']*PC,
            rchar=settings['hard_rchar']*PC,
            gamma_inner=gamma_inner,
            gamma_outer=gamma_outer,
        )
        return hard

    @classmethod
    def model_for_params(cls, params, sam_shape=None, new_def_params={}):
        # NOTE: these should be the same as the default case, just duplicating them here for clarity
        defs = dict(
            hard_gamma_inner=-1.0,
            hard_rchar=100.0,               # [pc]
            hard_gamma_outer=+1.5,
            hard_sepa_init=1e4,     # [pc]
        )
        defs.update(new_def_params)
        return super().model_for_params(params, sam_shape=sam_shape, new_def_params=defs)


class PS_Uniform_07A_Rot_Test_2(PS_Generic_2):
    """Change `hard_gamma_rot` relative to PS_Uniform_07A_Rot_Test
    """

    # `DEFAULTS` must have a copy of all settings that are used, so make a copy and expand it
    DEFAULTS = PS_Generic_2.DEFAULTS.copy()
    DEFAULTS['hard_gamma_rot'] = 0.0

    def __init__(self, log, nsamples, sam_shape, seed):
        super().__init__(
            log, nsamples, sam_shape, seed,
            hard_time=PD_Uniform(0.1, 11.0),   # [Gyr]
            hard_gamma_rot=PD_Uniform(0.0, 1.0),
        )

    @classmethod
    def _init_hard(cls, sam, settings):
        rotation = settings['hard_gamma_rot']
        gamma_inner = settings['hard_gamma_inner'] + rotation
        gamma_outer = settings['hard_gamma_outer'] + rotation
        hard = holo.hardening.Fixed_Time_2PL_SAM(
            sam,
            settings['hard_time']*GYR,
            sepa_init=settings['hard_sepa_init']*PC,
            rchar=settings['hard_rchar']*PC,
            gamma_inner=gamma_inner,
            gamma_outer=gamma_outer,
        )
        return hard

    @classmethod
    def model_for_params(cls, params, sam_shape=None, new_def_params={}):
        # NOTE: these should be the same as the default case, just duplicating them here for clarity
        defs = dict(
            hard_gamma_inner=-1.0,
            hard_rchar=100.0,               # [pc]
            hard_gamma_outer=+1.5,
            hard_sepa_init=1e4,     # [pc]
        )
        defs.update(new_def_params)
        return super().model_for_params(params, sam_shape=sam_shape, new_def_params=defs)


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
        defs = dict(
            hard_gamma_inner=-1.0,
            hard_rchar=100.0,               # [pc]
            hard_gamma_outer=+2.5,
            hard_sepa_init=1e4,     # [pc]
        )
        defs.update(new_def_params)
        return super().model_for_params(params, sam_shape=sam_shape, new_def_params=defs)


class PS_Uniform_07B_Rot(PS_Generic_2):
    """Same as PS_Uniform_07B, but adding a hardening-rate power-law rotation.
    """

    # `DEFAULTS` must have a copy of all settings that are used, so make a copy and expand it
    DEFAULTS = PS_Generic_2.DEFAULTS.copy()
    DEFAULTS['hard_gamma_rot'] = 0.0

    def __init__(self, log, nsamples, sam_shape, seed):
        super().__init__(
            log, nsamples, sam_shape, seed,
            hard_time=PD_Uniform(0.1, 11.0),   # [Gyr]
            gsmf_phi0=PD_Uniform(-3.5, -1.5),
            gsmf_mchar0_log10=PD_Uniform(10.5, 12.5),   # [log10(Msol)]
            mmb_mamp_log10=PD_Uniform(+7.6, +9.0),   # [log10(Msol)]
            mmb_scatter_dex=PD_Uniform(+0.0, +0.9),
            hard_gamma_rot=PD_Uniform(-0.5, 0.5),
        )

    @classmethod
    def _init_hard(cls, sam, settings):
        rotation = settings['hard_gamma_rot']
        gamma_inner = settings['hard_gamma_inner'] + rotation
        gamma_outer = settings['hard_gamma_outer'] + rotation
        hard = holo.hardening.Fixed_Time_2PL_SAM(
            sam,
            settings['hard_time']*GYR,
            sepa_init=settings['hard_sepa_init']*PC,
            rchar=settings['hard_rchar']*PC,
            gamma_inner=gamma_inner,
            gamma_outer=gamma_outer,
        )
        return hard

    @classmethod
    def model_for_params(cls, params, sam_shape=None, new_def_params={}):
        # NOTE: these should be the same as the default case, just duplicating them here for clarity
        defs = dict(
            hard_gamma_inner=-1.0,
            hard_rchar=100.0,               # [pc]
            hard_gamma_outer=+2.5,
            hard_sepa_init=1e4,     # [pc]
        )
        defs.update(new_def_params)
        return super().model_for_params(params, sam_shape=sam_shape, new_def_params=defs)


class PS_Uniform_07B_Rot_Test(PS_Generic_2):
    """Same as PS_Uniform_07A, but adding a hardening-rate power-law rotation.
    """

    # `DEFAULTS` must have a copy of all settings that are used, so make a copy and expand it
    DEFAULTS = PS_Generic_2.DEFAULTS.copy()
    DEFAULTS['hard_gamma_rot'] = 0.0

    def __init__(self, log, nsamples, sam_shape, seed):
        super().__init__(
            log, nsamples, sam_shape, seed,
            hard_time=PD_Uniform(0.1, 11.0),   # [Gyr]
            hard_gamma_rot=PD_Uniform(-0.5, 0.5),
        )

    @classmethod
    def _init_hard(cls, sam, settings):
        rotation = settings['hard_gamma_rot']
        gamma_inner = settings['hard_gamma_inner'] + rotation
        gamma_outer = settings['hard_gamma_outer'] + rotation
        hard = holo.hardening.Fixed_Time_2PL_SAM(
            sam,
            settings['hard_time']*GYR,
            sepa_init=settings['hard_sepa_init']*PC,
            rchar=settings['hard_rchar']*PC,
            gamma_inner=gamma_inner,
            gamma_outer=gamma_outer,
        )
        return hard

    @classmethod
    def model_for_params(cls, params, sam_shape=None, new_def_params={}):
        # NOTE: these should be the same as the default case, just duplicating them here for clarity
        defs = dict(
            hard_gamma_inner=-1.0,
            hard_rchar=100.0,               # [pc]
            hard_gamma_outer=+2.5,
            hard_sepa_init=1e4,     # [pc]
        )
        defs.update(new_def_params)
        return super().model_for_params(params, sam_shape=sam_shape, new_def_params=defs)


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
        defs = dict(
            hard_gamma_inner=-1.0,
            hard_rchar=30.0,               # [pc]
            hard_gamma_outer=+1.5,
            hard_sepa_init=3e3,     # [pc]
        )
        defs.update(new_def_params)
        return super().model_for_params(params, sam_shape=sam_shape, new_def_params=defs)


class PS_Uniform_07D(PS_Generic_2):
    """Same as PS_Uniform_07A with a rotation of -0.5
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
        defs = dict(
            hard_gamma_inner=-1.0 - 0.5,
            hard_rchar=100.0,               # [pc]
            hard_gamma_outer=+1.5 - 0.5,
            hard_sepa_init=1e4,     # [pc]
        )
        defs.update(new_def_params)
        return super().model_for_params(params, sam_shape=sam_shape, new_def_params=defs)


class PS_Uniform_07E(PS_Generic_2):
    """Same as PS_Uniform_07A with a rotation of +0.5
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
        defs = dict(
            hard_gamma_inner=-1.0 + 0.5,
            hard_rchar=100.0,               # [pc]
            hard_gamma_outer=+1.5 + 0.5,
            hard_sepa_init=1e4,     # [pc]
        )
        defs.update(new_def_params)
        return super().model_for_params(params, sam_shape=sam_shape, new_def_params=defs)


class PS_Uniform_07F(PS_Generic_2):
    """Same as PS_Uniform_07B with a rotation of -0.5
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
        defs = dict(
            hard_gamma_inner=-1.0 - 0.5,
            hard_rchar=100.0,               # [pc]
            hard_gamma_outer=+2.5 - 0.5,
            hard_sepa_init=1e4,     # [pc]
        )
        defs.update(new_def_params)
        return super().model_for_params(params, sam_shape=sam_shape, new_def_params=defs)


class PS_Uniform_07G(PS_Generic_2):
    """Same as PS_Uniform_07B with a rotation of +0.5
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
        defs = dict(
            hard_gamma_inner=-1.0 + 0.5,
            hard_rchar=100.0,               # [pc]
            hard_gamma_outer=+2.5 + 0.5,
            hard_sepa_init=1e4,     # [pc]
        )
        defs.update(new_def_params)
        return super().model_for_params(params, sam_shape=sam_shape, new_def_params=defs)


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


class PS_Uniform_08A(PS_Generic_2):
    """`PS_Uniform_07A` with varying gamma_outer
    """

    def __init__(self, log, nsamples, sam_shape, seed):
        super().__init__(
            log, nsamples, sam_shape, seed,
            hard_time=PD_Uniform(0.1, 11.0),   # [Gyr]
            gsmf_phi0=PD_Uniform(-3.5, -1.5),
            gsmf_mchar0_log10=PD_Uniform(10.5, 12.5),   # [log10(Msol)]
            mmb_mamp_log10=PD_Uniform(+7.6, +9.0),   # [log10(Msol)]
            mmb_scatter_dex=PD_Uniform(+0.0, +0.9),
            hard_gamma_outer=PD_Uniform(+1.5, +3.0),
        )

    @classmethod
    def model_for_params(cls, params, sam_shape=None, new_def_params={}):
        # NOTE: these should be the same as the default case, just duplicating them here for clarity
        defs = dict(
            hard_gamma_inner=-1.0,
            hard_rchar=100.0,               # [pc]
            hard_sepa_init=1e4,     # [pc]
        )
        defs.update(new_def_params)
        return super().model_for_params(params, sam_shape=sam_shape, new_def_params=defs)


'''
#! BAD
class PS_Uniform_08B(PS_Generic_2):
    """`PS_Uniform_07A` with varying gamma_outer
    """

    def __init__(self, log, nsamples, sam_shape, seed):
        super().__init__(
            log, nsamples, sam_shape, seed,
            hard_time=PD_Uniform(0.1, 11.0),   # [Gyr]
            gsmf_phi0=PD_Uniform(-3.5, -1.5),
            gsmf_mchar0_log10=PD_Uniform(10.5, 12.5),   # [log10(Msol)]
            mmb_mamp_log10=PD_Uniform(+7.6, +9.0),   # [log10(Msol)]
            mmb_scatter_dex=PD_Uniform(+0.0, +0.9),
            hard_gamma_inner=PD_Uniform(+0.5, +2.0),
        )

    @classmethod
    def model_for_params(cls, params, sam_shape=None, new_def_params={}):
        # NOTE: these should be the same as the default case, just duplicating them here for clarity
        defs = dict(
            hard_gamma_outer=+1.5,
            hard_rchar=100.0,               # [pc]
            hard_sepa_init=1e4,     # [pc]
        )
        defs.update(new_def_params)
        return super().model_for_params(params, sam_shape=sam_shape, new_def_params=defs)


#! BAD
class PS_Uniform_08C(PS_Generic_2):
    """`PS_Uniform_07A` with varying gamma_outer
    """

    def __init__(self, log, nsamples, sam_shape, seed):
        super().__init__(
            log, nsamples, sam_shape, seed,
            hard_time=PD_Uniform(0.1, 11.0),   # [Gyr]
            gsmf_phi0=PD_Uniform(-3.5, -1.5),
            gsmf_mchar0_log10=PD_Uniform(10.5, 12.5),   # [log10(Msol)]
            mmb_mamp_log10=PD_Uniform(+7.6, +9.0),   # [log10(Msol)]
            mmb_scatter_dex=PD_Uniform(+0.0, +0.9),
            hard_gamma_inner=PD_Uniform(+0.5, +2.0),
        )

    @classmethod
    def model_for_params(cls, params, sam_shape=None, new_def_params={}):
        # NOTE: these should be the same as the default case, just duplicating them here for clarity
        defs = dict(
            hard_gamma_outer=+2.0,
            hard_rchar=100.0,               # [pc]
            hard_sepa_init=1e4,     # [pc]
        )
        defs.update(new_def_params)
        return super().model_for_params(params, sam_shape=sam_shape, new_def_params=defs)


#! BAD
class PS_Uniform_08D(PS_Generic_2):
    """`PS_Uniform_07A` with varying gamma_outer
    """

    def __init__(self, log, nsamples, sam_shape, seed):
        super().__init__(
            log, nsamples, sam_shape, seed,
            hard_time=PD_Uniform(0.1, 11.0),   # [Gyr]
            gsmf_phi0=PD_Uniform(-3.5, -1.5),
            gsmf_mchar0_log10=PD_Uniform(10.5, 12.5),   # [log10(Msol)]
            mmb_mamp_log10=PD_Uniform(+7.6, +9.0),   # [log10(Msol)]
            mmb_scatter_dex=PD_Uniform(+0.0, +0.9),
            hard_gamma_inner=PD_Uniform(+0.5, +2.0),
        )

    @classmethod
    def model_for_params(cls, params, sam_shape=None, new_def_params={}):
        # NOTE: these should be the same as the default case, just duplicating them here for clarity
        defs = dict(
            hard_gamma_outer=+2.5,
            hard_rchar=100.0,               # [pc]
            hard_sepa_init=1e4,     # [pc]
        )
        defs.update(new_def_params)
        return super().model_for_params(params, sam_shape=sam_shape, new_def_params=defs)
'''


class PS_Uniform_08E(PS_Generic_2):
    """`PS_Uniform_07A` with varying gamma_outer
    """

    def __init__(self, log, nsamples, sam_shape, seed):
        super().__init__(
            log, nsamples, sam_shape, seed,
            hard_time=PD_Uniform(0.1, 11.0),   # [Gyr]
            gsmf_phi0=PD_Uniform(-3.5, -1.5),
            gsmf_mchar0_log10=PD_Uniform(10.5, 12.5),   # [log10(Msol)]
            mmb_mamp_log10=PD_Uniform(+7.6, +9.0),   # [log10(Msol)]
            mmb_scatter_dex=PD_Uniform(+0.0, +0.9),
            hard_gamma_outer=PD_Uniform(+1.5, +3.0),
        )

    @classmethod
    def model_for_params(cls, params, sam_shape=None, new_def_params={}):
        # NOTE: these should be the same as the default case, just duplicating them here for clarity
        defs = dict(
            hard_gamma_inner=-1.5,
            hard_rchar=100.0,               # [pc]
            hard_sepa_init=1e4,     # [pc]
        )
        defs.update(new_def_params)
        return super().model_for_params(params, sam_shape=sam_shape, new_def_params=defs)


'''
class PS_Uniform_08F(PS_Generic_2):
    """`PS_Uniform_08A` but with hard_gamma_inner=-1.5 instead of -1.0
    """

    def __init__(self, log, nsamples, sam_shape, seed):
        super().__init__(
            log, nsamples, sam_shape, seed,
            hard_time=PD_Uniform(0.1, 11.0),   # [Gyr]
            gsmf_phi0=PD_Uniform(-3.5, -1.5),
            gsmf_mchar0_log10=PD_Uniform(10.5, 12.5),   # [log10(Msol)]
            mmb_mamp_log10=PD_Uniform(+7.6, +9.0),   # [log10(Msol)]
            mmb_scatter_dex=PD_Uniform(+0.0, +0.9),
            hard_gamma_outer=PD_Uniform(+1.5, +3.0),
        )

    @classmethod
    def model_for_params(cls, params, sam_shape=None, new_def_params={}):
        # NOTE: these should be the same as the default case, just duplicating them here for clarity
        defs = dict(
            hard_gamma_inner=-1.5,
            hard_rchar=100.0,               # [pc]
            hard_sepa_init=1e4,     # [pc]
        )
        defs.update(new_def_params)
        return super().model_for_params(params, sam_shape=sam_shape, new_def_params=defs)
'''


class PS_Uniform_09A(PS_Generic_2):
    """`PS_Uniform_07A` with varying gamma_inner (gamma_outer=+1.5, same as 07A)
    """

    def __init__(self, log, nsamples, sam_shape, seed):
        super().__init__(
            log, nsamples, sam_shape, seed,
            hard_time=PD_Uniform(0.1, 11.0),   # [Gyr]
            gsmf_phi0=PD_Uniform(-3.5, -1.5),
            gsmf_mchar0_log10=PD_Uniform(10.5, 12.5),   # [log10(Msol)]
            mmb_mamp_log10=PD_Uniform(+7.6, +9.0),   # [log10(Msol)]
            mmb_scatter_dex=PD_Uniform(+0.0, +0.9),
            hard_gamma_inner=PD_Uniform(-1.5, +0.0),
        )

    @classmethod
    def model_for_params(cls, params, sam_shape=None, new_def_params={}):
        # NOTE: these should be the same as the default case, just duplicating them here for clarity
        defs = dict(
            hard_gamma_outer=+1.5,
            hard_rchar=100.0,               # [pc]
            hard_sepa_init=1e4,     # [pc]
        )
        defs.update(new_def_params)
        return super().model_for_params(params, sam_shape=sam_shape, new_def_params=defs)


class PS_Uniform_09B(PS_Generic_2):
    """`PS_Uniform_07B` with varying gamma_inner (gamma_outer=+2.5, same as 07B)
    """

    def __init__(self, log, nsamples, sam_shape, seed):
        super().__init__(
            log, nsamples, sam_shape, seed,
            hard_time=PD_Uniform(0.1, 11.0),   # [Gyr]
            gsmf_phi0=PD_Uniform(-3.5, -1.5),
            gsmf_mchar0_log10=PD_Uniform(10.5, 12.5),   # [log10(Msol)]
            mmb_mamp_log10=PD_Uniform(+7.6, +9.0),   # [log10(Msol)]
            mmb_scatter_dex=PD_Uniform(+0.0, +0.9),
            hard_gamma_inner=PD_Uniform(-1.5, +0.0),
        )

    @classmethod
    def model_for_params(cls, params, sam_shape=None, new_def_params={}):
        # NOTE: these should be the same as the default case, just duplicating them here for clarity
        defs = dict(
            hard_gamma_outer=+2.5,
            hard_rchar=100.0,               # [pc]
            hard_sepa_init=1e4,     # [pc]
        )
        defs.update(new_def_params)
        return super().model_for_params(params, sam_shape=sam_shape, new_def_params=defs)


# ==============================================================================
# ====    New-Astro-02 / PS_Generic_2    ====
# ==============================================================================


class PS_New_Astro_02A(PS_Generic_2):
    """New version of 'PS_Astro_02' using PS_Generic_2 base.
    """

    def __init__(self, log, nsamples, sam_shape, seed):
        super().__init__(
            log, nsamples, sam_shape, seed,

            hard_time=PD_Uniform(0.1, 11.0),   # [Gyr]
            hard_gamma_inner=PD_Uniform(-1.5, +0.5),

            # from `sam-parameters.ipynb` fits to [Tomczak+2014] with 4x stdev values
            gsmf_phi0=PD_Normal(-2.56, 0.4),
            gsmf_mchar0_log10=PD_Normal(10.9, 0.4),   # [log10(Msol)]
            gsmf_alpha0=PD_Normal(-1.2, 0.2),

            gpf_zbeta=PD_Normal(+0.8, 0.4),
            gpf_qgamma=PD_Normal(+0.5, 0.3),

            gmt_norm=PD_Uniform(0.2, 5.0),    # [Gyr]
            gmt_zbeta=PD_Uniform(-2.0, +0.0),

            mmb_mamp_log10=PD_Normal(+8.6, 0.2),   # [log10(Msol)]
            mmb_plaw=PD_Normal(+1.2, 0.2),
            mmb_scatter_dex=PD_Normal(+0.32, 0.15),

            # hard_time=PD_Uniform(0.1, 11.0),   # [Gyr]
            # gsmf_phi0=PD_Uniform(-3.5, -1.5),
            # gsmf_mchar0_log10=PD_Uniform(10.5, 12.5),   # [log10(Msol)]
            # mmb_mamp_log10=PD_Uniform(+7.6, +9.0),   # [log10(Msol)]
            # mmb_scatter_dex=PD_Uniform(+0.0, +0.9),
        )

    @classmethod
    def model_for_params(cls, params, sam_shape=None, new_def_params={}):
        # NOTE: these should be the same as the default case, just duplicating them here for clarity
        defs = dict(
            hard_rchar=100.0,               # [pc]
            hard_gamma_outer=+1.5,
            hard_sepa_init=1e4,     # [pc]
        )
        defs.update(new_def_params)
        return super().model_for_params(params, sam_shape=sam_shape, new_def_params=defs)


class PS_New_Astro_02B(PS_Generic_2):
    """New version of 'PS_Astro_02' using PS_Generic_2 base.
    """

    def __init__(self, log, nsamples, sam_shape, seed):
        super().__init__(
            log, nsamples, sam_shape, seed,

            hard_time=PD_Uniform(0.1, 11.0),   # [Gyr]
            hard_gamma_inner=PD_Uniform(-1.5, +0.5),

            # from `sam-parameters.ipynb` fits to [Tomczak+2014] with 4x stdev values
            gsmf_phi0=PD_Normal(-2.56, 0.4),
            gsmf_mchar0_log10=PD_Normal(10.9, 0.4),   # [log10(Msol)]
            gsmf_alpha0=PD_Normal(-1.2, 0.2),

            gpf_zbeta=PD_Normal(+0.8, 0.4),
            gpf_qgamma=PD_Normal(+0.5, 0.3),

            gmt_norm=PD_Uniform(0.2, 5.0),    # [Gyr]
            gmt_zbeta=PD_Uniform(-2.0, +0.0),

            mmb_mamp_log10=PD_Normal(+8.6, 0.2),   # [log10(Msol)]
            mmb_plaw=PD_Normal(+1.2, 0.2),
            mmb_scatter_dex=PD_Normal(+0.32, 0.15),

            # hard_time=PD_Uniform(0.1, 11.0),   # [Gyr]
            # gsmf_phi0=PD_Uniform(-3.5, -1.5),
            # gsmf_mchar0_log10=PD_Uniform(10.5, 12.5),   # [log10(Msol)]
            # mmb_mamp_log10=PD_Uniform(+7.6, +9.0),   # [log10(Msol)]
            # mmb_scatter_dex=PD_Uniform(+0.0, +0.9),
        )

    @classmethod
    def model_for_params(cls, params, sam_shape=None, new_def_params={}):
        # NOTE: these should be the same as the default case, just duplicating them here for clarity
        defs = dict(
            hard_rchar=100.0,               # [pc]
            hard_gamma_outer=+2.5,
            hard_sepa_init=1e4,     # [pc]
        )
        defs.update(new_def_params)
        return super().model_for_params(params, sam_shape=sam_shape, new_def_params=defs)


class PS_New_Astro_02_GW(PS_Generic_2):
    """New version of 'PS_Astro_02' using PS_Generic_2 base.
    """

    def __init__(self, log, nsamples, sam_shape, seed):
        super().__init__(
            log, nsamples, sam_shape, seed,

            # from `sam-parameters.ipynb` fits to [Tomczak+2014] with 4x stdev values
            gsmf_phi0=PD_Normal(-2.56, 0.4),
            gsmf_mchar0_log10=PD_Normal(10.9, 0.4),   # [log10(Msol)]
            gsmf_alpha0=PD_Normal(-1.2, 0.2),

            gpf_zbeta=PD_Normal(+0.8, 0.4),
            gpf_qgamma=PD_Normal(+0.5, 0.3),

            gmt_norm=PD_Uniform(0.2, 5.0),    # [Gyr]
            gmt_zbeta=PD_Uniform(-2.0, +0.0),

            mmb_mamp_log10=PD_Normal(+8.6, 0.2),   # [log10(Msol)]
            mmb_plaw=PD_Normal(+1.2, 0.2),
            mmb_scatter_dex=PD_Normal(+0.32, 0.15),

            # hard_time=PD_Uniform(0.1, 11.0),   # [Gyr]
            # gsmf_phi0=PD_Uniform(-3.5, -1.5),
            # gsmf_mchar0_log10=PD_Uniform(10.5, 12.5),   # [log10(Msol)]
            # mmb_mamp_log10=PD_Uniform(+7.6, +9.0),   # [log10(Msol)]
            # mmb_scatter_dex=PD_Uniform(+0.0, +0.9),
        )

    @classmethod
    def _init_hard(cls, sam, settings):
        hard = holo.hardening.Hard_GW()
        return hard
