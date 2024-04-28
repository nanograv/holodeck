"""'Classic' parameter spaces used in the NANOGrav 15yr analysis.
"""

from holodeck.constants import PC, GYR
from holodeck.librarian.libraries import _Param_Space, PD_Uniform, PD_Normal
from holodeck import sams, hardening, host_relations


class _PS_Classic_Phenom(_Param_Space):
    """Base class for classic phenomenological parameter space used in 15yr analysis.
    """

    DEFAULTS = dict(
        hard_time=3.0,          # [Gyr]
        hard_sepa_init=1e4,     # [pc]
        hard_rchar=100.0,       # [pc]
        hard_gamma_inner=-1.0,
        hard_gamma_outer=+2.5,

        # Parameters are based on `sam-parameters.ipynb` fit to [Tomczak+2014]
        gsmf_phi0_log10=-2.77,
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

    @classmethod
    def _init_sam(cls, sam_shape, params):
        gsmf = sams.GSMF_Schechter(
            phi0=params['gsmf_phi0_log10'],
            phiz=params['gsmf_phiz'],
            mchar0_log10=params['gsmf_mchar0_log10'],
            mcharz=params['gsmf_mcharz'],
            alpha0=params['gsmf_alpha0'],
            alphaz=params['gsmf_alphaz'],
        )
        gpf = sams.GPF_Power_Law(
            frac_norm_allq=params['gpf_frac_norm_allq'],
            malpha=params['gpf_malpha'],
            qgamma=params['gpf_qgamma'],
            zbeta=params['gpf_zbeta'],
            max_frac=params['gpf_max_frac'],
        )
        gmt = sams.GMT_Power_Law(
            time_norm=params['gmt_norm']*GYR,
            malpha=params['gmt_malpha'],
            qgamma=params['gmt_qgamma'],
            zbeta=params['gmt_zbeta'],
        )
        mmbulge = host_relations.MMBulge_KH2013(
            mamp_log10=params['mmb_mamp_log10'],
            mplaw=params['mmb_plaw'],
            scatter_dex=params['mmb_scatter_dex'],
        )

        sam = sams.Semi_Analytic_Model(
            gsmf=gsmf, gpf=gpf, gmt=gmt, mmbulge=mmbulge,
            shape=sam_shape,
        )
        return sam

    @classmethod
    def _init_hard(cls, sam, params):
        hard = hardening.Fixed_Time_2PL_SAM(
            sam,
            params['hard_time']*GYR,
            sepa_init=params['hard_sepa_init']*PC,
            rchar=params['hard_rchar']*PC,
            gamma_inner=params['hard_gamma_inner'],
            gamma_outer=params['hard_gamma_outer'],
        )
        return hard


class PS_Classic_Phenom_Uniform(_PS_Classic_Phenom):
    """Classic 5D phenomenological, uniform parameter space used in 15yr analysis.

    Previously called the `PS_Uniform_09B` parameter space, or 'phenom-uniform'.

    """

    def __init__(self, log=None, nsamples=None, sam_shape=None, seed=None):
        parameters = [
            PD_Uniform("gsmf_phi0_log10", -3.5, -1.5),
            PD_Uniform("gsmf_mchar0_log10", 10.5, 12.5),   # [log10(Msol)]
            PD_Uniform("mmb_mamp_log10", +7.5, +9.5),      # [log10(Msol)]
            PD_Uniform("mmb_scatter_dex", +0.0, +1.2),
            PD_Uniform("hard_time", 0.1, 11.0),            # [Gyr]
            PD_Uniform("hard_gamma_inner", -1.5, +0.0),
        ]

        super().__init__(
            parameters,
            log=log, nsamples=nsamples, sam_shape=sam_shape, seed=seed,
        )


class PS_Classic_Phenom_Astro_Extended(_PS_Classic_Phenom):
    """Classic 12D phenomenological, uniform parameter space used in 15yr analysis.

    Previously called the `PS_New_Astro_02B` parameter space, or 'phenom-astro+extended'.

    """

    def __init__(self, log=None, nsamples=None, sam_shape=None, seed=None):
        parameters = [
            PD_Uniform("hard_time", 0.1, 11.0),   # [Gyr]
            PD_Uniform("hard_gamma_inner", -1.5, +0.5),

            # from `sam-parameters.ipynb` fits to [Tomczak+2014] with 4x stdev values
            PD_Normal("gsmf_phi0", -2.56, 0.4),
            PD_Normal("gsmf_mchar0_log10", 10.9, 0.4),   # [log10(Msol)]
            PD_Normal("gsmf_alpha0", -1.2, 0.2),

            PD_Normal("gpf_zbeta", +0.8, 0.4),
            PD_Normal("gpf_qgamma", +0.5, 0.3),

            PD_Uniform("gmt_norm", 0.2, 5.0),    # [Gyr]
            PD_Uniform("gmt_zbeta", -2.0, +0.0),

            PD_Normal("mmb_mamp_log10", +8.6, 0.2),   # [log10(Msol)]
            PD_Normal("mmb_plaw", +1.2, 0.2),
            PD_Normal("mmb_scatter_dex", +0.32, 0.15),
        ]
        super().__init__(
            parameters,
            log=log, nsamples=nsamples, sam_shape=sam_shape, seed=seed,
        )


class _PS_Classic_GWOnly(_Param_Space):
    """Base class for classic GW-Only parameter space used in 15yr analysis.
    """

    DEFAULTS = dict(
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

    def __init__(self, log=None, nsamples=None, sam_shape=None, seed=None):
        parameters = [
            PD_Uniform("gsmf_phi0", -3.5, -1.5),
            PD_Uniform("gsmf_mchar0_log10", 10.5, 12.5),   # [log10(Msol)]
            PD_Uniform("mmb_mamp_log10", +7.5, +9.5),   # [log10(Msol)]
            PD_Uniform("mmb_scatter", +0.0, +1.2),
        ]
        super().__init__(
            parameters,
            log=log, nsamples=nsamples, sam_shape=sam_shape, seed=seed,
        )
        return

    @classmethod
    def _init_sam(cls, sam_shape, params):
        gsmf = sams.GSMF_Schechter(
            phi0=params['gsmf_phi0'],
            phiz=params['gsmf_phiz'],
            mchar0_log10=params['gsmf_mchar0_log10'],
            mcharz=params['gsmf_mcharz'],
            alpha0=params['gsmf_alpha0'],
            alphaz=params['gsmf_alphaz'],
        )
        gpf = sams.GPF_Power_Law(
            frac_norm_allq=params['gpf_frac_norm_allq'],
            malpha=params['gpf_malpha'],
            qgamma=params['gpf_qgamma'],
            zbeta=params['gpf_zbeta'],
            max_frac=params['gpf_max_frac'],
        )
        gmt = sams.GMT_Power_Law(
            time_norm=params['gmt_norm']*GYR,
            malpha=params['gmt_malpha'],
            qgamma=params['gmt_qgamma'],
            zbeta=params['gmt_zbeta'],
        )
        mmbulge = host_relations.MMBulge_KH2013(
            mamp_log10=params['mmb_mamp_log10'],
            mplaw=params['mmb_plaw'],
            scatter_dex=params['mmb_scatter_dex'],
        )

        sam = sams.Semi_Analytic_Model(
            gsmf=gsmf, gpf=gpf, gmt=gmt, mmbulge=mmbulge,
            shape=sam_shape,
        )
        return sam

    @classmethod
    def _init_hard(cls, sam, params):
        hard = hardening.Hard_GW()
        return hard


class PS_Classic_GWOnly_Uniform(_PS_Classic_GWOnly):
    """Classic 4D GW-Only, uniform parameter space used in 15yr analysis.

    Previously called the `PS_Uniform_07_GW` parameter space, or 'gw-only'.

    """

    def __init__(self, log=None, nsamples=None, sam_shape=None, seed=None):
        parameters = [
            PD_Uniform("gsmf_phi0", -3.5, -1.5),
            PD_Uniform("gsmf_mchar0_log10", 10.5, 12.5),   # [log10(Msol)]
            PD_Uniform("mmb_mamp_log10", +7.5, +9.5),   # [log10(Msol)]
            PD_Uniform("mmb_scatter", +0.0, +1.2),
        ]
        _Param_Space.__init__(
            self, parameters,
            log=log, nsamples=nsamples, sam_shape=sam_shape, seed=seed,
        )
        return


class PS_Classic_GWOnly_Astro_Extended(_PS_Classic_GWOnly):
    """Classic 10D GW-Only, uniform parameter space used in 15yr analysis.

    Previously called the `PS_New_Astro_02_GW` parameter space, or 'gw-only+extended'.

    """

    def __init__(self, log=None, nsamples=None, sam_shape=None, seed=None):
        parameters = [
            # from `sam-parameters.ipynb` fits to [Tomczak+2014] with 4x stdev values
            PD_Normal("gsmf_phi0", -2.56, 0.4),
            PD_Normal("gsmf_mchar0_log10", 10.9, 0.4),   # [log10(Msol)]
            PD_Normal("gsmf_alpha0", -1.2, 0.2),

            PD_Normal("gpf_zbeta", +0.8, 0.4),
            PD_Normal("gpf_qgamma", +0.5, 0.3),

            PD_Uniform("gmt_norm", 0.2, 5.0),    # [Gyr]
            PD_Uniform("gmt_zbeta", -2.0, +0.0),

            PD_Normal("mmb_mamp_log10", +8.6, 0.2),   # [log10(Msol)]
            PD_Normal("mmb_plaw", +1.2, 0.2),
            PD_Normal("mmb_scatter_dex", +0.32, 0.15),
        ]
        _Param_Space.__init__(
            self, parameters,
            log=log, nsamples=nsamples, sam_shape=sam_shape, seed=seed,
        )
        return


class PS_Test(_PS_Classic_Phenom):
    """Simple test parameter space in 2D.
    """

    def __init__(self, log=None, nsamples=None, sam_shape=None, seed=None):

        parameters = [
            PD_Uniform("mmb_mamp_log10", +7.5, +9.5),   # [log10(Msol)]
            PD_Uniform("hard_time", 0.1, 11.0),         # [Gyr]
        ]

        _Param_Space.__init__(
            self, parameters,
            log=log, nsamples=nsamples, sam_shape=sam_shape, seed=seed,
        )
        return


_param_spaces_dict = {
    "PS_Test": PS_Test,
    "PS_Classic_Phenom_Uniform": PS_Classic_Phenom_Uniform,    # PS_Uniform_09B
    "PS_Classic_Phenom_Astro_Extended": PS_Classic_Phenom_Astro_Extended,
    "PS_Classic_GWOnly_Uniform": PS_Classic_GWOnly_Uniform,
    "PS_Classic_GWOnly_Astro_Extended": PS_Classic_GWOnly_Astro_Extended,
}

