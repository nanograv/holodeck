"""
"""

import holodeck as holo
from holodeck.constants import MSOL, PC, GYR
from holodeck.librarian import (
    _Parameter_Space, _LHS_Parameter_Space, _Param_Space,
    PD_Normal, PD_Uniform, PD_Uniform_Log
)


class PS_Broad_Uniform_01(_Param_Space):

    def __init__(self, log, nsamples, sam_shape, seed):
        super().__init__(
            log, nsamples, sam_shape, seed,

            hard_time=PD_Uniform(0.1, 12.0),   # [Gyr]
            hard_gamma_inner=PD_Uniform(-1.5, +0.0),
            # hard_rchar=[+0.0, +4.0],   # [log10(pc)]
            # hard_gamma_outer=[+2.0, +3.0],

            gsmf_phi0=PD_Uniform(-3.5, -1.5),
            # gsmf_phiz =[-1.5, +0.5],
            gsmf_mchar0_log10=PD_Uniform(10.5, 12.0),   # [log10(Msol)]
            gsmf_alpha0=PD_Uniform(-2.0, -0.5),

            # gpf_malpha=[-1.0, +1.0],
            gpf_zbeta=PD_Uniform(-0.5, +2.5),
            gpf_qgamma=PD_Uniform(-1.5, +1.5),

            gmt_norm=PD_Uniform(0.1, +10.0),    # [Gyr]
            # gmt_malpha=[-1.0, +1.0],
            gmt_zbeta=PD_Uniform(-2.0, +1.0),
            # gmt_qgamma=[-1.0, +1.0],

            mmb_amp_log10=PD_Uniform(+7.5, +9.5),   # [log10(Msol)]
            mmb_plaw=PD_Uniform(+0.5, +2.0),
            mmb_scatter=PD_Uniform(+0.0, +0.6),
        )

    def model_for_number(self, num):
        params = self.param_dict(num)
        self._log.debug(f"params {num}:: {params}")

        hard_time = params['hard_time'] * GYR
        gmt_norm = params['gmt_norm'] * GYR

        gsmf = holo.sam.GSMF_Schechter(
            phi0=params['gsmf_phi0'],
            mchar0_log10=params['gsmf_mchar0_log10'],
            alpha0=params['gsmf_alpha0'],
        )
        gpf = holo.sam.GPF_Power_Law(
            qgamma=params['gpf_qgamma'],
            zbeta=params['gpf_zbeta'],
        )
        gmt = holo.sam.GMT_Power_Law(
            time_norm=gmt_norm,
            zbeta=params['gmt_zbeta'],
        )
        mmbulge = holo.relations.MMBulge_KH2013(
            mamp_log10=params['mmb_amp_log10'],
            mplaw=params['mmb_plaw'],
            scatter_dex=params['mmb_scatter'],
        )

        sam = holo.sam.Semi_Analytic_Model(
            gsmf=gsmf, gpf=gpf, gmt=gmt, mmbulge=mmbulge,
            shape=self.sam_shape
        )
        hard = holo.hardening.Fixed_Time.from_sam(
            sam,
            hard_time,
            gamma_sc=params['hard_gamma_inner'],
            progress=False,
        )

        return sam, hard


class PS_Astro_01(PS_Broad_Uniform_01):

    def __init__(self, log, nsamples, sam_shape, seed):
        super(PS_Broad_Uniform_01, self).__init__(
            log, nsamples, sam_shape, seed,

            hard_time=PD_Uniform(0.1, 12.0),   # [Gyr]
            hard_gamma_inner=PD_Uniform(-1.5, +0.5),
            # hard_rchar=[+0.0, +4.0],   # [log10(pc)]
            # hard_gamma_outer=[+2.0, +3.0],

            # from `sam-parameters.ipynb` fits to [Tomczak+2014] with 4x stdev values
            gsmf_phi0=PD_Normal(-2.56, +0.4),
            # gsmf_phiz =[-1.5, +0.5],
            gsmf_mchar0_log10=PD_Normal(10.9, 0.4),   # [log10(Msol)]
            gsmf_alpha0=PD_Normal(-1.2, +0.2),

            # gpf_norm=
            # gpf_malpha=[-1.0, +1.0],
            gpf_zbeta=PD_Normal(+0.8, +0.4),
            gpf_qgamma=PD_Normal(+0.5, +0.3),

            gmt_norm=PD_Uniform(0.2, 5.0),    # [Gyr]
            # gmt_malpha=[-1.0, +1.0],
            gmt_zbeta=PD_Uniform(-2.0, +0.0),
            # gmt_qgamma=[-1.0, +1.0],

            mmb_amp_log10=PD_Uniform(+7.5, +9.5),   # [log10(Msol)]
            mmb_plaw=PD_Normal(+1.2, +0.2),
            mmb_scatter=PD_Normal(+0.32, +0.15),
        )

    def model_for_number(self, num):
        params = self.param_dict(num)
        self._log.debug(f"params {num}:: {params}")

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

        sam = holo.sam.Semi_Analytic_Model(
            gsmf=gsmf, gpf=gpf, gmt=gmt, mmbulge=mmbulge,
            shape=self.sam_shape
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


class PS_Test_Uniform(_Param_Space):

    def __init__(self, log, nsamples, sam_shape, seed):
        super().__init__(
            log, nsamples, sam_shape, seed,
            hard_time=PD_Uniform(-2.0, 1.1),
            gsmf_phi0=PD_Uniform(-3.5, -1.5),

            gsmf_mchar0=PD_Uniform(10.75, 11.75),   # [log10(Msol)]
            gpf_qgamma=PD_Uniform(0.0, +2.0),
            gmt_zbeta=PD_Uniform(-2.0, +2.0),

            mmb_amp=PD_Uniform(+8.0, +9.0),   # [log10(Msol)]
            mmb_scatter=PD_Uniform(+0.0, +0.6),
        )
        return

    def model_for_number(self, num):
        params = self.param_dict(num)

        self._log.debug(f"params {num}:: {params}")

        hard_time = (10.0 ** params['hard_time']) * GYR

        gsmf = holo.sam.GSMF_Schechter(phi0=params['gsmf_phi0'], mchar0_log10=params['gsmf_mchar0'])
        gpf = holo.sam.GPF_Power_Law(qgamma=params['gpf_qgamma'])
        gmt = holo.sam.GMT_Power_Law(zbeta=params['gmt_zbeta'])
        mmbulge = holo.relations.MMBulge_KH2013(mamp_log10=params['mmb_amp'], scatter_dex=params['mmb_scatter'])

        sam = holo.sam.Semi_Analytic_Model(
            gsmf=gsmf, gpf=gpf, gmt=gmt, mmbulge=mmbulge,
            shape=self.sam_shape
        )
        hard = holo.hardening.Fixed_Time.from_sam(
            sam, hard_time,
            progress=False
        )
        return sam, hard


class PS_Test_Normal(PS_Test_Uniform):

    def __init__(self, log, nsamples, sam_shape, seed):
        super(PS_Test_Uniform, self).__init__(
            log, nsamples, sam_shape, seed,
            hard_time=PD_Normal(-0.45, 0.775),
            gsmf_phi0=PD_Normal(-2.5, 0.5),

            gsmf_mchar0=PD_Normal(11.25, 0.25),   # [log10(Msol)]
            gpf_qgamma=PD_Normal(1.0, 0.5),
            gmt_zbeta=PD_Normal(0.0, 1.0),

            mmb_amp=PD_Normal(8.5, 0.25),   # [log10(Msol)]
            mmb_scatter=PD_Normal(0.3, 0.15),
        )
        return


# ==============================================================================
# ====    OLD    ====
# ==============================================================================


class Parameter_Space_Hard04(_Parameter_Space):

    _PARAM_NAMES = [
        'hard_time',
        'hard_gamma_inner',
        'hard_gamma_outer',
        'hard_rchar',
        'gsmf_phi0',
        'mmb_amp',
    ]

    def __init__(self, log, nsamples, sam_shape):
        super().__init__(
            log, nsamples, sam_shape,
            hard_time=[-1.0, +1.0, 5],   # [log10(Gyr)]
            hard_gamma_inner=[-1.5, -0.5, 5],
            hard_gamma_outer=[+2.0, +3.0, 5],
            hard_rchar=[1.0, 3.0, 5],
            gsmf_phi0=[-3.0, -2.0, 5],
            mmb_amp=[0.1e9, 1.0e9, 5],
        )

    def sam_for_lhsnumber(self, lhsnum):
        param_grid = self.params_for_lhsnumber(lhsnum)

        time, gamma_inner, gamma_outer, rchar, gsmf_phi0, mmb_amp = param_grid
        time = (10.0 ** time) * GYR
        rchar = (10.0 ** rchar) * PC
        mmb_amp = mmb_amp*MSOL

        gsmf = holo.sam.GSMF_Schechter(phi0=gsmf_phi0)
        gpf = holo.sam.GPF_Power_Law()
        gmt = holo.sam.GMT_Power_Law()
        mmbulge = holo.relations.MMBulge_KH2013(mamp=mmb_amp)

        sam = holo.sam.Semi_Analytic_Model(
            gsmf=gsmf, gpf=gpf, gmt=gmt, mmbulge=mmbulge,
            shape=self.sam_shape
        )
        hard = holo.hardening.Fixed_Time.from_sam(
            sam, time, rchar=rchar, gamma_sc=gamma_inner, gamma_df=gamma_outer,
            exact=True, progress=False
        )
        return sam, hard


class LHS_Parameter_Space_Hard04(_LHS_Parameter_Space):

    _PARAM_NAMES = [
        'hard_time',
        'hard_gamma_inner',
        'hard_gamma_outer',
        'hard_rchar',
        'gsmf_phi0',
        'mmb_amp',
    ]

    def __init__(self, log, nsamples, sam_shape, lhs_sampler, seed):
        super().__init__(
            log, nsamples, sam_shape, lhs_sampler, seed,
            hard_time=[-1.0, +1.0],   # [log10(Gyr)]
            hard_gamma_inner=[-1.5, -0.5],
            hard_gamma_outer=[+2.0, +3.0],
            hard_rchar=[1.0, 3.0],
            gsmf_phi0=[-3.0, -2.0],
            mmb_amp=[0.1e9, 1.0e9],
        )

    def sam_for_lhsnumber(self, lhsnum):
        param_grid = self.params_for_lhsnumber(lhsnum)

        time, gamma_inner, gamma_outer, rchar, gsmf_phi0, mmb_amp = param_grid
        time = (10.0 ** time) * GYR
        rchar = (10.0 ** rchar) * PC
        mmb_amp = mmb_amp*MSOL

        gsmf = holo.sam.GSMF_Schechter(phi0=gsmf_phi0)
        gpf = holo.sam.GPF_Power_Law()
        gmt = holo.sam.GMT_Power_Law()
        mmbulge = holo.relations.MMBulge_KH2013(mamp=mmb_amp)

        sam = holo.sam.Semi_Analytic_Model(
            gsmf=gsmf, gpf=gpf, gmt=gmt, mmbulge=mmbulge,
            shape=self.sam_shape
        )
        hard = holo.evolution.Fixed_Time.from_sam(
            sam, time, rchar=rchar, gamma_sc=gamma_inner, gamma_df=gamma_outer,
            exact=True, progress=False
        )
        return sam, hard


class LHS_PSpace_Eccen_01(_LHS_Parameter_Space):

    _PARAM_NAMES = [
        'eccen_init',
        'gsmf_phi0',
        'gpf_zbeta',
        'mmb_amp',
    ]

    SEPA_INIT = 1.0 * PC

    def __init__(self, log, nsamples, sam_shape, lhs_sampler, seed):
        super().__init__(
            log, nsamples, sam_shape, lhs_sampler, seed,
            eccen_init=[0.0, +0.975],
            gsmf_phi0=[-3.0, -2.0],
            gpf_zbeta=[+0.0, +2.0],
            mmb_amp=[0.1e9, 1.0e9],
        )

    def sam_for_lhsnumber(self, lhsnum):
        param_grid = self.params_for_lhsnumber(lhsnum)

        eccen, gsmf_phi0, gpf_zbeta, mmb_amp = param_grid
        mmb_amp = mmb_amp*MSOL

        # favor higher values of eccentricity instead of uniformly distributed
        eccen = eccen ** (1.0/5.0)

        gsmf = holo.sam.GSMF_Schechter(phi0=gsmf_phi0)
        gpf = holo.sam.GPF_Power_Law(zbeta=gpf_zbeta)
        gmt = holo.sam.GMT_Power_Law()
        mmbulge = holo.relations.MMBulge_KH2013(mamp=mmb_amp)

        sam = holo.sam.Semi_Analytic_Model(
            gsmf=gsmf, gpf=gpf, gmt=gmt, mmbulge=mmbulge,
            shape=self.sam_shape
        )

        sepa_evo, eccen_evo = holo.sam.evolve_eccen_uniform_single(sam, eccen, self.SEPA_INIT, DEF_ECCEN_NUM_STEPS)

        return sam, sepa_evo, eccen_evo


class PSpace_Big_Circ_01(_LHS_Parameter_Space):

    _PARAM_NAMES = [
        'hard_time',
        'hard_rchar',
        'hard_gamma_inner',
        'hard_gamma_outer',

        'gsmf_phi0',
        'gsmf_phiz',
        'gsmf_alpha0',
        'gpf_malpha',
        'gpf_zbeta',
        'gpf_qgamma',

        'gmt_malpha',
        'gmt_zbeta',
        'gmt_qgamma',
        'mmb_amp',
        'mmb_plaw',
    ]

    def __init__(self, log, nsamples, sam_shape, lhs_sampler, seed):
        super().__init__(
            log, nsamples, sam_shape, lhs_sampler, seed,
            hard_time =[-2.0, +2.0],   # [log10(Gyr)]
            hard_rchar=[+0.0, +4.0],   # [log10(pc)]
            hard_gamma_inner=[-1.5, +0.0],
            hard_gamma_outer=[+2.0, +3.0],

            gsmf_phi0 =[-3.5, -1.5],
            gsmf_phiz =[-1.5, +0.5],
            gsmf_alpha0=[-2.5, -0.5],
            gpf_malpha=[-1.0, +1.0],
            gpf_zbeta =[-0.5, +2.5],
            gpf_qgamma=[-1.0, +1.0],

            gmt_malpha=[-1.0, +1.0],
            gmt_zbeta =[-3.0, +2.0],
            gmt_qgamma=[-1.0, +1.0],
            mmb_amp   =[+7.0, +10.0],   # [log10(Msol)]
            mmb_plaw  =[+0.25, +2.5],
        )

    def sam_for_lhsnumber(self, lhsnum):
        param_grid = self.params_for_lhsnumber(lhsnum)

        hard_time, hard_rchar, gamma_inner, gamma_outer, \
            gsmf_phi0, gsmf_phiz, gsmf_alpha0, \
            gpf_malpha, gpf_zbeta, gpf_qgamma, \
            gmt_malpha, gmt_zbeta, gmt_qgamma, \
            mmb_amp, mmb_plaw = param_grid

        mmb_amp = (10.0 ** mmb_amp) * MSOL
        hard_time = (10.0 ** hard_time) * GYR
        hard_rchar = (10.0 ** hard_rchar) * PC

        gsmf = holo.sam.GSMF_Schechter(phi0=gsmf_phi0, phiz=gsmf_phiz, alpha0=gsmf_alpha0)
        gpf = holo.sam.GPF_Power_Law(malpha=gpf_malpha, qgamma=gpf_qgamma, zbeta=gpf_zbeta)
        gmt = holo.sam.GMT_Power_Law(malpha=gmt_malpha, qgamma=gmt_qgamma, zbeta=gmt_zbeta)
        mmbulge = holo.relations.MMBulge_KH2013(mamp=mmb_amp, mplaw=mmb_plaw)

        sam = holo.sam.Semi_Analytic_Model(
            gsmf=gsmf, gpf=gpf, gmt=gmt, mmbulge=mmbulge,
            shape=self.sam_shape
        )
        hard = holo.hardening.Fixed_Time.from_sam(
            sam, hard_time, rchar=hard_rchar, gamma_sc=gamma_inner, gamma_df=gamma_outer,
            exact=True, progress=False
        )
        return sam, hard


class PS_Circ_01(_LHS_Parameter_Space):

    _PARAM_NAMES = [
        'hard_time',
        'hard_gamma_inner',

        'gsmf_phi0',
        # 'gsmf_phiz',
        'gsmf_mchar0',
        # 'gsmf_mcharz',
        'gsmf_alpha0',
        # 'gsmf_alphaz',

        # 'gpf_malpha',
        'gpf_zbeta',
        'gpf_qgamma',

        'gmt_norm',
        # 'gmt_malpha',
        'gmt_zbeta',
        # 'gmt_qgamma',

        'mmb_amp',
        'mmb_plaw',
        'mmb_scatter',
    ]

    def __init__(self, log, nsamples, sam_shape, lhs_sampler, seed):
        super().__init__(
            log, nsamples, sam_shape, lhs_sampler, seed,

            hard_time=[-2.0, +1.12],   # [log10(Gyr)]
            hard_gamma_inner=[-1.5, +0.0],
            # hard_rchar=[+0.0, +4.0],   # [log10(pc)]
            # hard_gamma_outer=[+2.0, +3.0],

            gsmf_phi0=[-3.5, -1.5],
            # gsmf_phiz =[-1.5, +0.5],
            gsmf_mchar0=[10.0, 12.5],   # [log10(Msol)]
            gsmf_alpha0=[-2.5, -0.5],

            # gpf_malpha=[-1.0, +1.0],
            gpf_zbeta=[-0.5, +2.5],
            gpf_qgamma=[-1.5, +1.5],

            gmt_norm=[0.1, +10.0],    # [Gyr]
            # gmt_malpha=[-1.0, +1.0],
            gmt_zbeta=[-3.0, +2.0],
            # gmt_qgamma=[-1.0, +1.0],

            mmb_amp=[+7.0, +10.0],   # [log10(Msol)]
            mmb_plaw=[+0.25, +2.5],
            mmb_scatter=[+0.0, +0.6],
        )

    def sam_for_lhsnumber(self, lhsnum):
        param_grid = self.params_for_lhsnumber(lhsnum)
        self.log.debug("params at {lhsnum}:: {param_grid}")

        hard_time, hard_gamma_inner, \
            gsmf_phi0, gsmf_mchar0, gsmf_alpha0, \
            gpf_zbeta, gpf_qgamma, \
            gmt_norm, gmt_zbeta, \
            mmb_amp, mmb_plaw, mmb_scatter = param_grid

        mmb_amp = (10.0 ** mmb_amp) * MSOL
        hard_time = (10.0 ** hard_time) * GYR
        gmt_norm = gmt_norm * GYR

        gsmf = holo.sam.GSMF_Schechter(phi0=gsmf_phi0, mchar0_log10=gsmf_mchar0, alpha0=gsmf_alpha0)
        gpf = holo.sam.GPF_Power_Law(qgamma=gpf_qgamma, zbeta=gpf_zbeta)
        gmt = holo.sam.GMT_Power_Law(time_norm=gmt_norm, zbeta=gmt_zbeta)
        mmbulge = holo.relations.MMBulge_KH2013(mamp=mmb_amp, mplaw=mmb_plaw, scatter_dex=mmb_scatter)

        sam = holo.sam.Semi_Analytic_Model(
            gsmf=gsmf, gpf=gpf, gmt=gmt, mmbulge=mmbulge,
            shape=self.sam_shape
        )
        hard = holo.hardening.Fixed_Time.from_sam(
            sam, hard_time, gamma_sc=hard_gamma_inner,
            progress=False
        )
        return sam, hard

