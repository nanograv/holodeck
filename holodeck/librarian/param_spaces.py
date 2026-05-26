"""Parameter-Space definitions for holodeck libraries."""

import numpy as np
import holodeck as holo
from holodeck.constants import GYR, PC, MSOL
from holodeck.librarian.lib_tools import _Param_Space, PD_Uniform, PD_Normal, PD_Uniform_Log, PD_MVNormal


# Define a new Parameter-Space class by subclassing the base class:
# :py:class:`holodeck.librarian.libaries._Param_Space`.  The names of all parameter-space subclasses
# should typically be prefixed by `PS_` to denote that they are parameter-spaces.
class PS_Test(_Param_Space):
    """Simple Test Parameter Space: SAM with strongly astrophysically-motivated parameters.

    This model uses a double-Schechter GSMF, an Illustris-derived galaxy merger rate, a Kormendy+Ho
    M-MBulge relationship, and a phenomenology binary evolution model.

    """

    # The `DEFAULTS` attribute is a dictionary of default parameter values.  These are automatically
    # copied over to the `params` arguments that are passed into the `_init_sam` and `_init_hard`
    # methods.  Specifying these is strongly recommended to ensure that parameters are set
    # consistently, by setting them explicitly.
    # Notice that each group of parameters is named with a common prefix, e.g. 'hard_' or 'gsmf_'.
    # This is not required, but simply used to more easily organize/identify parameters.
    DEFAULTS = dict(
        hard_time=3.0,  # [Gyr]
        hard_sepa_init=1e4,  # [pc]
        hard_rchar=100.0,  # [pc]
        hard_gamma_inner=-1.0,
        hard_gamma_outer=+2.5,
        # Galaxy stellar-mass Function (``GSMF_Schechter``)
        gsmf_phi0_log10=-2.77,
        gsmf_phiz=-0.6,
        gsmf_mchar0_log10=11.24,
        gsmf_mcharz=0.11,
        gsmf_alpha0=-1.21,
        gsmf_alphaz=-0.03,
        # Galaxy merger rate (``GMR_Illustris``)
        # Parameters are taken directly from [Rodriguez-Gomez2015]_
        gmr_norm0_log10=-2.2287,  # -2.2287 ± 0.0045    A0 [log10(A*Gyr)]
        gmr_normz=+2.4644,  # +2.4644 ± 0.0128    eta
        gmr_malpha0=+0.2241,  # +0.2241 ± 0.0038    alpha0
        gmr_malphaz=-1.1759,  # -1.1759 ± 0.0316    alpha1
        gmr_mdelta0=+0.7668,  # +0.7668 ± 0.0202    delta0
        gmr_mdeltaz=-0.4695,  # -0.4695 ± 0.0440    delta1
        gmr_qgamma0=-1.2595,  # -1.2595 ± 0.0026    beta0
        gmr_qgammaz=+0.0611,  # +0.0611 ± 0.0021    beta1
        gmr_qgammam=-0.0477,  # -0.0477 ± 0.0013    gamma
        # M-MBulge Relationship (``MMBulge_KH2013``)
        # From [KH2013]_
        mmb_mamp=0.49e9,  # 0.49e9 + 0.06 - 0.05  [Msol]
        mmb_plaw=1.17,  # 1.17 ± 0.08
        mmb_scatter_dex=0.28,  # no uncertainties given
    )

    # The initialization method should typically include the below arguments, which are passed
    # directly to the parent/super ``_Param_Space`` constructor.  These arguments could be passed
    # along using ``**kwargs``, but they are included explicitly for clarity and for convenience
    # when examining the function signature.
    def __init__(self, log=None, nsamples=None, sam_shape=None, seed=None, **kwargs):
        # NOTE: this is where the parameter-space is actually defined:
        parameters = [
            # The names of the parameters passed to the parameter distribution constructors
            # MUST match the variable names expected in the parameter space methods (``_init_sam``
            # and ``_init_hard``).
            PD_Uniform("hard_time", 0.1, 11.0, default=3.0),  # [Gyr]
            # For the "hard_time" and "hard_gamma_inner" parameters, default values are specified
            # explicitly because the fiducial parameters are not the central/average values.  If no
            # ``default`` value was specified, then the central/average value would be used, for
            # example :math:`(11.0 + 0.1)/2 = 5.5` in the case of "hard_time" above.
            PD_Uniform("hard_gamma_inner", -1.5, +0.0, default=-1.0),
            # This specifies a normal distribution with the given mean and standard-deviation.  In
            # this case, no ``default`` value is specified, so the value returned from an input of
            # 0.5 will be used which, for a normal distribution, is by definition the mean (0.49e9
            # in this case).
            PD_Normal("mmb_mamp", 0.49e9, 0.055e9),
        ]
        # Call the parent/super constructor, passing in these parameters to define the domain of
        # the parameter space.
        _Param_Space.__init__(self, parameters, log=log, nsamples=nsamples, sam_shape=sam_shape, seed=seed, **kwargs)
        return

    # Define the function which actually constructs the SAM, using a dictionary of model parameters.
    # This is not intended as an API function, but an internal method used to build the SAM model.
    # The call signature of this function should *not* be changed, and the function must always
    # return a single object: the instance of :py:class:`holodeck.sams.sam.Semi_Analytic_Model`.
    def _init_sam(self, sam_shape, params):

        # Schechter Galaxy Stellar-Mass Function
        gsmf = holo.sams.GSMF_Schechter(
            phi0=params["gsmf_phi0_log10"],
            phiz=params["gsmf_phiz"],
            mchar0_log10=params["gsmf_mchar0_log10"],
            mcharz=params["gsmf_mcharz"],
            alpha0=params["gsmf_alpha0"],
            alphaz=params["gsmf_alphaz"],
        )

        # Illustris Galaxy Merger Rate
        gmr = holo.sams.GMR_Illustris(
            norm0_log10=params["gmr_norm0_log10"],
            normz=params["gmr_normz"],
            malpha0=params["gmr_malpha0"],
            malphaz=params["gmr_malphaz"],
            mdelta0=params["gmr_mdelta0"],
            mdeltaz=params["gmr_mdeltaz"],
            qgamma0=params["gmr_qgamma0"],
            qgammaz=params["gmr_qgammaz"],
            qgammam=params["gmr_qgammam"],
        )

        # Notice that a unit-conversion is being performed here, in the M-Mbulge constructor.  The
        # parameter space is defined such that the normalization is in units of solar-masses, while
        # the M-Mbulge class itself is defined such that the normalization is in units of grams.
        # This is a very easy place to make mistakes, and so any/all units (and particularly unit
        # conversions) should be described carefully in docstrings.
        mmbulge = holo.host_relations.MMBulge_KH2013(
            mamp=params["mmb_mamp"] * MSOL,
            mplaw=params["mmb_plaw"],
            scatter_dex=params["mmb_scatter_dex"],
        )

        sam = holo.sams.Semi_Analytic_Model(
            gsmf=gsmf,
            gmr=gmr,
            mmbulge=mmbulge,
            shape=sam_shape,
            log=self._log,
        )
        return sam

    # Define the function which constructs the hardening model, used by the SAM.
    # This is not intended as an API function, but an internal method used to build the SAM model.
    # The call signature of this function should *not* be changed, and the function must always
    # return a single object: an instance of a subclass of
    # :py:class:`holodeck.hardening._Hardening`.
    def _init_hard(self, sam, params):
        hard = holo.hardening.Fixed_Time_2PL_SAM(
            sam,
            params["hard_time"] * GYR,
            sepa_init=params["hard_sepa_init"] * PC,
            rchar=params["hard_rchar"] * PC,
            gamma_inner=params["hard_gamma_inner"],
            gamma_outer=params["hard_gamma_outer"],
        )
        return hard


class _PS_Astro_Strong(_Param_Space):
    """SAM Model with strongly astrophysically-motivated parameters.

    This model uses a double-Schechter GSMF, an Illustris-derived galaxy merger rate, a Kormendy+Ho
    M-MBulge relationship, and a phenomenology binary evolution model.

    """

    __version__ = "0.3"

    DEFAULTS = dict(
        # Hardening model (phenom 2PL)
        hard_time=3.0,  # [Gyr]
        hard_sepa_init=1e4,  # [pc]
        hard_rchar=10.0,  # [pc]
        hard_gamma_inner=-1.0,
        hard_gamma_outer=+2.5,
        # Galaxy stellar-mass Function (``GSMF_Double_Schechter``)
        # Parameters are based on `double-schechter.ipynb` conversions from [Leja2020]_
        gsmf_log10_phi_one_z0=-2.383,  # - 2.383 ± 0.028
        gsmf_log10_phi_one_z1=-0.264,  # - 0.264 ± 0.072
        gsmf_log10_phi_one_z2=-0.107,  # - 0.107 ± 0.031
        gsmf_log10_phi_two_z0=-2.818,  # - 2.818 ± 0.050
        gsmf_log10_phi_two_z1=-0.368,  # - 0.368 ± 0.070
        gsmf_log10_phi_two_z2=+0.046,  # + 0.046 ± 0.020
        gsmf_log10_mstar_z0=+10.767,  # +10.767 ± 0.026
        gsmf_log10_mstar_z1=+0.124,  # + 0.124 ± 0.045
        gsmf_log10_mstar_z2=-0.033,  # - 0.033 ± 0.015
        gsmf_alpha_one=-0.28,  # - 0.280 ± 0.070
        gsmf_alpha_two=-1.48,  # - 1.480 ± 0.0150
        # Galaxy merger rate (``GMR_Illustris``)
        # Parameters are taken directly from [Rodriguez-Gomez2015]_
        gmr_norm0_log10=-2.2287,  # -2.2287 ± 0.0045    A0 [log10(A*Gyr)]
        gmr_normz=+2.4644,  # +2.4644 ± 0.0128    eta
        gmr_malpha0=+0.2241,  # +0.2241 ± 0.0038    alpha0
        gmr_malphaz=-1.1759,  # -1.1759 ± 0.0316    alpha1
        gmr_mdelta0=+0.7668,  # +0.7668 ± 0.0202    delta0
        gmr_mdeltaz=-0.4695,  # -0.4695 ± 0.0440    delta1
        gmr_qgamma0=-1.2595,  # -1.2595 ± 0.0026    beta0
        gmr_qgammaz=+0.0611,  # +0.0611 ± 0.0021    beta1
        gmr_qgammam=-0.0477,  # -0.0477 ± 0.0013    gamma
        # M-MBulge Relationship (``MMBulge_KH2013``)
        # From [KH2013]_
        mmb_mamp_log10=8.69,  # 8.69±0.05  [log10(M/Msol)]  approx uncertainties!
        mmb_plaw=1.17,  # 1.17 ± 0.08
        mmb_scatter_dex=0.28,  # no uncertainties given
        # bulge fraction
        bf_frac_lo=0.4,
        bf_frac_hi=0.8,
        bf_mstar_crit=11.0,  # [log10(M_star/M_Sol)]
        bf_width_dex=1.0,  # [dex]
    )

    def _init_sam(self, sam_shape, params):
        log10_phi_one = [
            params["gsmf_log10_phi_one_z0"],
            params["gsmf_log10_phi_one_z1"],
            params["gsmf_log10_phi_one_z2"],
        ]
        log10_phi_two = [
            params["gsmf_log10_phi_two_z0"],
            params["gsmf_log10_phi_two_z1"],
            params["gsmf_log10_phi_two_z2"],
        ]
        log10_mstar = [
            params["gsmf_log10_mstar_z0"],
            params["gsmf_log10_mstar_z1"],
            params["gsmf_log10_mstar_z2"],
        ]
        gsmf = holo.sams.GSMF_Double_Schechter(
            log10_phi1=log10_phi_one,
            log10_phi2=log10_phi_two,
            log10_mstar=log10_mstar,
            alpha1=params["gsmf_alpha_one"],
            alpha2=params["gsmf_alpha_two"],
        )

        # Illustris Galaxy Merger Rate
        gmr = holo.sams.GMR_Illustris(
            norm0_log10=params["gmr_norm0_log10"],
            normz=params["gmr_normz"],
            malpha0=params["gmr_malpha0"],
            malphaz=params["gmr_malphaz"],
            mdelta0=params["gmr_mdelta0"],
            mdeltaz=params["gmr_mdeltaz"],
            qgamma0=params["gmr_qgamma0"],
            qgammaz=params["gmr_qgammaz"],
            qgammam=params["gmr_qgammam"],
        )

        # Mbh-MBulge relationship (and bulge-fractions)
        bulge_frac = holo.host_relations.BF_Sigmoid(
            bulge_frac_lo=params["bf_frac_lo"],
            bulge_frac_hi=params["bf_frac_hi"],
            mstar_char_log10=params["bf_mstar_crit"],
            width_dex=params["bf_width_dex"],
        )
        mmbulge = holo.host_relations.MMBulge_KH2013(
            mamp_log10=params["mmb_mamp_log10"],
            mplaw=params["mmb_plaw"],
            scatter_dex=params["mmb_scatter_dex"],
            bulge_frac=bulge_frac,
        )

        sam = holo.sams.Semi_Analytic_Model(
            gsmf=gsmf,
            gmr=gmr,
            mmbulge=mmbulge,
            shape=sam_shape,
            log=self._log,
        )
        return sam

    def _init_hard(self, sam, params):
        hard = holo.hardening.Fixed_Time_2PL_SAM(
            sam,
            params["hard_time"] * GYR,
            sepa_init=params["hard_sepa_init"] * PC,
            rchar=params["hard_rchar"] * PC,
            gamma_inner=params["hard_gamma_inner"],
            gamma_outer=params["hard_gamma_outer"],
        )
        return hard


class PS_Astro_Strong_All(_PS_Astro_Strong):
    def __init__(self, log=None, nsamples=None, sam_shape=None, seed=None):
        parameters = [
            # Hardening model (phenom 2PL)
            PD_Uniform("hard_time", 0.1, 11.0, default=3.0),  # [Gyr]
            PD_Uniform("hard_gamma_inner", -2.0, +0.0, default=-1.0),
            PD_Uniform("hard_rchar", 2.0, 20.0, default=10.0),  # [pc]
            # GSMF
            PD_Normal("gsmf_log10_phi_one_z0", -2.383, 0.028),  # - 2.383 ± 0.028
            PD_Normal("gsmf_log10_phi_one_z1", -0.264, 0.072),  # - 0.264 ± 0.072
            PD_Normal("gsmf_log10_phi_one_z2", -0.107, 0.031),  # - 0.107 ± 0.031
            PD_Normal("gsmf_log10_phi_two_z0", -2.818, 0.050),  # - 2.818 ± 0.050
            PD_Normal("gsmf_log10_phi_two_z1", -0.368, 0.070),  # - 0.368 ± 0.070
            PD_Normal("gsmf_log10_phi_two_z2", +0.046, 0.020),  # + 0.046 ± 0.020
            PD_Normal("gsmf_log10_mstar_z0", +10.767, 0.026),  # +10.767 ± 0.026
            PD_Normal("gsmf_log10_mstar_z1", +0.124, 0.045),  # + 0.124 ± 0.045
            PD_Normal("gsmf_log10_mstar_z2", -0.033, 0.015),  # - 0.033 ± 0.015
            PD_Normal("gsmf_alpha_one", -0.28, 0.070),  # - 0.280 ± 0.070
            PD_Normal("gsmf_alpha_two", -1.48, 0.0150),  # - 1.480 ± 0.0150
            # GMR
            PD_Normal("gmr_norm0_log10", -2.2287, 0.0045),  # -2.2287 ± 0.0045    A0 [log10(A*Gyr)]
            PD_Normal("gmr_normz", +2.4644, 0.0128),  # +2.4644 ± 0.0128    eta
            PD_Normal("gmr_malpha0", +0.2241, 0.0038),  # +0.2241 ± 0.0038    alpha0
            PD_Normal("gmr_malphaz", -1.1759, 0.0316),  # -1.1759 ± 0.0316    alpha1
            PD_Normal("gmr_mdelta0", +0.7668, 0.0202),  # +0.7668 ± 0.0202    delta0
            PD_Normal("gmr_mdeltaz", -0.4695, 0.0440),  # -0.4695 ± 0.0440    delta1
            PD_Normal("gmr_qgamma0", -1.2595, 0.0026),  # -1.2595 ± 0.0026    beta0
            PD_Normal("gmr_qgammaz", +0.0611, 0.0021),  # +0.0611 ± 0.0021    beta1
            PD_Normal("gmr_qgammam", -0.0477, 0.0013),  # -0.0477 ± 0.0013    gamma
            # MMBulge
            # From [KH2013]_
            PD_Normal("mmb_mamp_log10", 8.69, 0.05),  # 8.69 ± 0.05  [log10(M/Msol)]
            PD_Normal("mmb_plaw", 1.17, 0.08),  # 1.17 ± 0.08
            # Extra
            PD_Normal("mmb_scatter_dex", 0.28, 0.05),  # no uncertainties given
            PD_Uniform("bf_frac_lo", 0.1, 0.4),
            PD_Uniform("bf_frac_hi", 0.6, 1.0),
            PD_Uniform("bf_width_dex", 0.5, 1.5),  # [dex]
        ]
        _Param_Space.__init__(
            self,
            parameters,
            log=log,
            nsamples=nsamples,
            sam_shape=sam_shape,
            seed=seed,
        )
        return


class PS_Astro_Strong_Hard(_PS_Astro_Strong):
    def __init__(self, log=None, nsamples=None, sam_shape=None, seed=None):
        parameters = [
            # Hardening model (phenom 2PL)
            PD_Uniform("hard_time", 0.1, 11.0, default=3.0),  # [Gyr]
            PD_Uniform("hard_gamma_inner", -2.0, +0.0, default=-1.0),
            PD_Uniform("hard_rchar", 2.0, 20.0, default=10.0),  # [pc]
        ]
        _Param_Space.__init__(
            self,
            parameters,
            log=log,
            nsamples=nsamples,
            sam_shape=sam_shape,
            seed=seed,
        )
        return


class PS_Astro_Strong_Hard_All(_PS_Astro_Strong):
    def __init__(self, log=None, nsamples=None, sam_shape=None, seed=None):
        parameters = [
            # Hardening model (phenom 2PL)
            PD_Uniform("hard_time", 0.1, 11.0, default=3.0),  # [Gyr]
            PD_Uniform("hard_gamma_inner", -2.0, +0.0, default=-1.0),
            PD_Uniform("hard_rchar", 2.0, 20.0, default=10.0),  # [pc]
            PD_Uniform_Log("hard_sepa_init", 1e3, 1e4, default=1e4),  # [pc]
            PD_Uniform("hard_gamma_outer", 0.0, +2.5, default=0.0),
        ]
        _Param_Space.__init__(
            self,
            parameters,
            log=log,
            nsamples=nsamples,
            sam_shape=sam_shape,
            seed=seed,
        )
        return


class PS_Astro_Strong_GSMF(_PS_Astro_Strong):
    def __init__(self, log=None, nsamples=None, sam_shape=None, seed=None):
        parameters = [
            # GSMF
            PD_Normal("gsmf_log10_phi_one_z0", -2.383, 0.028),  # - 2.383 ± 0.028
            PD_Normal("gsmf_log10_phi_one_z1", -0.264, 0.072),  # - 0.264 ± 0.072
            PD_Normal("gsmf_log10_phi_one_z2", -0.107, 0.031),  # - 0.107 ± 0.031
            PD_Normal("gsmf_log10_phi_two_z0", -2.818, 0.050),  # - 2.818 ± 0.050
            PD_Normal("gsmf_log10_phi_two_z1", -0.368, 0.070),  # - 0.368 ± 0.070
            PD_Normal("gsmf_log10_phi_two_z2", +0.046, 0.020),  # + 0.046 ± 0.020
            PD_Normal("gsmf_log10_mstar_z0", +10.767, 0.026),  # +10.767 ± 0.026
            PD_Normal("gsmf_log10_mstar_z1", +0.124, 0.045),  # + 0.124 ± 0.045
            PD_Normal("gsmf_log10_mstar_z2", -0.033, 0.015),  # - 0.033 ± 0.015
            PD_Normal("gsmf_alpha_one", -0.28, 0.070),  # - 0.280 ± 0.070
            PD_Normal("gsmf_alpha_two", -1.48, 0.0150),  # - 1.480 ± 0.0150
        ]
        _Param_Space.__init__(
            self,
            parameters,
            log=log,
            nsamples=nsamples,
            sam_shape=sam_shape,
            seed=seed,
        )
        return


# --- GSMF Covariance Data (derived from Leja+2020 as seen in notebooks/dev/covariant-double-schecter.ipynb) ---
# Perhaps this should be moved to the data directory?
GSMF_COV_NAMES = [
    "gsmf_log10_phi_one_z0",
    "gsmf_log10_phi_one_z1",
    "gsmf_log10_phi_one_z2",
    "gsmf_log10_phi_two_z0",
    "gsmf_log10_phi_two_z1",
    "gsmf_log10_phi_two_z2",
    "gsmf_log10_mstar_z0",
    "gsmf_log10_mstar_z1",
    "gsmf_log10_mstar_z2",
    "gsmf_alpha_one",
    "gsmf_alpha_two",
]
GSMF_COV_MEANS = [
    -2.38605866,
    -0.25897199,
    -0.10951613,
    -2.82070514,
    -0.36816392,
    0.04568619,
    10.76498848,
    0.12976555,
    -0.03443286,
    -0.27756761,
    -1.47916569,
]
# ruff: disable[E501]
GSMF_COV_MATRIX = [
    [
        9.97362541e-04,
        -1.75952495e-03,
        6.64667123e-04,
        -1.22588632e-04,
        -1.38383795e-04,
        6.72750271e-05,
        -3.75077937e-04,
        6.82393902e-04,
        -2.27547904e-04,
        -1.74718494e-04,
        -1.22584103e-04,
    ],
    [
        -1.75952495e-03,
        3.86310686e-03,
        -1.54843880e-03,
        2.76580123e-04,
        3.38194509e-04,
        -1.62399202e-04,
        5.36023989e-04,
        -1.46778730e-03,
        5.52098977e-04,
        8.21823185e-04,
        1.88617607e-04,
    ],
    [
        6.64667123e-04,
        -1.54843880e-03,
        7.17015559e-04,
        -1.53369714e-04,
        -1.22257443e-04,
        7.12455172e-05,
        -1.27727417e-04,
        4.70667715e-04,
        -1.99297686e-04,
        -3.96605202e-04,
        -9.11970213e-05,
    ],
    [
        -1.22588632e-04,
        2.76580123e-04,
        -1.53369714e-04,
        1.30829541e-03,
        -1.78028307e-04,
        -1.94597386e-05,
        -4.73316265e-04,
        2.89730040e-05,
        2.48570545e-05,
        2.19588739e-03,
        5.31183997e-04,
    ],
    [
        -1.38383795e-04,
        3.38194509e-04,
        -1.22257443e-04,
        -1.78028307e-04,
        4.57300656e-04,
        -1.70579413e-04,
        2.30730634e-04,
        -5.33099365e-04,
        1.94166589e-04,
        1.12878278e-04,
        7.91474258e-06,
    ],
    [
        6.72750271e-05,
        -1.62399202e-04,
        7.12455172e-05,
        -1.94597386e-05,
        -1.70579413e-04,
        7.46084434e-05,
        -5.32932948e-05,
        2.01144553e-04,
        -8.02847917e-05,
        -1.86917626e-04,
        -3.93673897e-05,
    ],
    [
        -3.75077937e-04,
        5.36023989e-04,
        -1.27727417e-04,
        -4.73316265e-04,
        2.30730634e-04,
        -5.32932948e-05,
        5.32167282e-04,
        -5.89018509e-04,
        1.76566274e-04,
        -7.98939131e-04,
        -1.24156129e-04,
    ],
    [
        6.82393902e-04,
        -1.46778730e-03,
        4.70667715e-04,
        2.89730040e-05,
        -5.33099365e-04,
        2.01144553e-04,
        -5.89018509e-04,
        1.36062137e-03,
        -4.76115334e-04,
        -4.36423012e-04,
        -9.24872624e-05,
    ],
    [
        -2.27547904e-04,
        5.52098977e-04,
        -1.99297686e-04,
        2.48570545e-05,
        1.94166589e-04,
        -8.02847917e-05,
        1.76566274e-04,
        -4.76115334e-04,
        1.84634729e-04,
        2.22527764e-04,
        4.31135674e-05,
    ],
    [
        -1.74718494e-04,
        8.21823185e-04,
        -3.96605202e-04,
        2.19588739e-03,
        1.12878278e-04,
        -1.86917626e-04,
        -7.98939131e-04,
        -4.36423012e-04,
        2.22527764e-04,
        4.57272842e-03,
        9.05757285e-04,
    ],
    [
        -1.22584103e-04,
        1.88617607e-04,
        -9.11970213e-05,
        5.31183997e-04,
        7.91474258e-06,
        -3.93673897e-05,
        -1.24156129e-04,
        -9.24872624e-05,
        4.31135674e-05,
        9.05757285e-04,
        2.46586887e-04,
    ],
]
# ruff: enable[E501]


class PS_Astro_Strong_Covariant_GSMF(_PS_Astro_Strong):
    def __init__(self, log=None, nsamples=None, sam_shape=None, seed=None):
        parameters = [
            PD_MVNormal(GSMF_COV_NAMES, GSMF_COV_MEANS, np.array(GSMF_COV_MATRIX)),
        ]

        _Param_Space.__init__(
            self,
            parameters,
            log=log,
            nsamples=nsamples,
            sam_shape=sam_shape,
            seed=seed,
        )
        return


class PS_Astro_Strong_GMR(_PS_Astro_Strong):
    def __init__(self, log=None, nsamples=None, sam_shape=None, seed=None):
        parameters = [
            # GMR
            PD_Normal("gmr_norm0_log10", -2.2287, 0.0045),  # -2.2287 ± 0.0045    A0 [log10(A*Gyr)]
            PD_Normal("gmr_normz", +2.4644, 0.0128),  # +2.4644 ± 0.0128    eta
            PD_Normal("gmr_malpha0", +0.2241, 0.0038),  # +0.2241 ± 0.0038    alpha0
            PD_Normal("gmr_malphaz", -1.1759, 0.0316),  # -1.1759 ± 0.0316    alpha1
            PD_Normal("gmr_mdelta0", +0.7668, 0.0202),  # +0.7668 ± 0.0202    delta0
            PD_Normal("gmr_mdeltaz", -0.4695, 0.0440),  # -0.4695 ± 0.0440    delta1
            PD_Normal("gmr_qgamma0", -1.2595, 0.0026),  # -1.2595 ± 0.0026    beta0
            PD_Normal("gmr_qgammaz", +0.0611, 0.0021),  # +0.0611 ± 0.0021    beta1
            PD_Normal("gmr_qgammam", -0.0477, 0.0013),  # -0.0477 ± 0.0013    gamma
        ]
        _Param_Space.__init__(
            self,
            parameters,
            log=log,
            nsamples=nsamples,
            sam_shape=sam_shape,
            seed=seed,
        )
        return


class PS_Astro_Strong_MMBulge_BFrac(_PS_Astro_Strong):
    def __init__(self, log=None, nsamples=None, sam_shape=None, seed=None):
        parameters = [
            # MMbulge - from [KH2013]_
            PD_Normal("mmb_mamp_log10", 8.69, 0.05),  # 8.69 ± 0.05  [log10(M/Msol)]
            PD_Normal("mmb_plaw", 1.17, 0.08),  # 1.17 ± 0.08
            PD_Normal("mmb_scatter_dex", 0.28, 0.05),  # no uncertainties given
            PD_Uniform("bf_frac_lo", 0.1, 0.4),
            PD_Uniform("bf_frac_hi", 0.6, 1.0),
            PD_Uniform("bf_width_dex", 0.5, 1.5),  # [dex]
        ]
        _Param_Space.__init__(
            self,
            parameters,
            log=log,
            nsamples=nsamples,
            sam_shape=sam_shape,
            seed=seed,
        )
        return


class PS_Astro_Strong_MMBulge(_PS_Astro_Strong):
    def __init__(self, log=None, nsamples=None, sam_shape=None, seed=None):
        parameters = [
            # MMbulge - from [KH2013]_
            PD_Normal("mmb_mamp_log10", 8.69, 0.05),  # 8.69 ± 0.05  [log10(M/Msol)]
            PD_Normal("mmb_plaw", 1.17, 0.08),  # 1.17 ± 0.08
            PD_Normal("mmb_scatter_dex", 0.28, 0.05),  # no uncertainties given
        ]
        _Param_Space.__init__(
            self,
            parameters,
            log=log,
            nsamples=nsamples,
            sam_shape=sam_shape,
            seed=seed,
        )
        return


class PS_Test_Astro_Strong_Covariant_MMBulge(_PS_Astro_Strong):
    """
    Test Parameter Space derived from PS_Astro_Strong, demonstrating a
    Multivariate Normal distribution for MMBulge parameters.

    - mmb_mamp_log10 and mmb_plaw are strongly coupled (rho=0.9).
    - mmb_scatter_dex remains independent.
    """

    def __init__(self, log=None, nsamples=None, sam_shape=None, seed=None):

        # --- 1. Define the parameters using a mix of univariate and multivariate distributions ---
        parameters = [
            # 1. Multivariate Distribution: mmb_mamp_log10 and mmb_plaw
            PD_MVNormal(
                names=("mmb_mamp_log10", "mmb_plaw"),
                means=[8.69, 1.17],
                cov=[
                    [0.0025, 0.0036],  # Variance for mamp (0.05^2), Covariance with plaw (0.9*0.05*0.08)
                    [0.0036, 0.0064],  # Covariance with mamp, Variance for plaw (0.08^2)
                ],
            ),
            # 2. Univariate Distribution: mmb_scatter_dex (Independent)
            PD_Normal("mmb_scatter_dex", 0.28, 0.05),
        ]

        # --- 2. Initialize the base class (_Param_Space) ---
        # The base class handles flattening the names and sampling 3 dimensions in total.
        _Param_Space.__init__(
            self,
            parameters,
            log=log,
            nsamples=nsamples,
            sam_shape=sam_shape,
            seed=seed,
        )
        return


class PS_Astro_Strong_Covariant_All(_PS_Astro_Strong):
    def __init__(self, log=None, nsamples=None, sam_shape=None, seed=None):
        parameters = [
            # Hardening model (phenom 2PL) ## Kayhan thinks this should be updated to Laura's hardening method
            PD_Uniform("hard_time", 0.1, 11.0, default=3.0),  # [Gyr]
            PD_Uniform("hard_gamma_inner", -2.0, +0.0, default=-1.0),
            PD_Uniform("hard_rchar", 2.0, 20.0, default=10.0),  # [pc]
            # GSMF
            PD_MVNormal(GSMF_COV_NAMES, GSMF_COV_MEANS, np.array(GSMF_COV_MATRIX)),
            # GMR ## This could become covariant, but Kayhan will have to look harder at the papers
            PD_Normal("gmr_norm0_log10", -2.2287, 0.0045),  # -2.2287 ± 0.0045    A0 [log10(A*Gyr)]
            PD_Normal("gmr_normz", +2.4644, 0.0128),  # +2.4644 ± 0.0128    eta
            PD_Normal("gmr_malpha0", +0.2241, 0.0038),  # +0.2241 ± 0.0038    alpha0
            PD_Normal("gmr_malphaz", -1.1759, 0.0316),  # -1.1759 ± 0.0316    alpha1
            PD_Normal("gmr_mdelta0", +0.7668, 0.0202),  # +0.7668 ± 0.0202    delta0
            PD_Normal("gmr_mdeltaz", -0.4695, 0.0440),  # -0.4695 ± 0.0440    delta1
            PD_Normal("gmr_qgamma0", -1.2595, 0.0026),  # -1.2595 ± 0.0026    beta0
            PD_Normal("gmr_qgammaz", +0.0611, 0.0021),  # +0.0611 ± 0.0021    beta1
            PD_Normal("gmr_qgammam", -0.0477, 0.0013),  # -0.0477 ± 0.0013    gamma
            # MMBulge
            # From [KH2013]_
            PD_Normal("mmb_mamp_log10", 8.69, 0.05),  # 8.69 ± 0.05  [log10(M/Msol)]
            PD_Normal("mmb_plaw", 1.17, 0.08),  # 1.17 ± 0.08
            # Extra
            PD_Normal("mmb_scatter_dex", 0.28, 0.05),  # no uncertainties given
            PD_Uniform("bf_frac_lo", 0.1, 0.4),
            PD_Uniform("bf_frac_hi", 0.6, 1.0),
            PD_Uniform("bf_width_dex", 0.5, 1.5),  # [dex]
        ]
        _Param_Space.__init__(
            self,
            parameters,
            log=log,
            nsamples=nsamples,
            sam_shape=sam_shape,
            seed=seed,
        )
        return


_param_spaces_dict = {
    "PS_Test": PS_Test,
    "PS_Astro_Strong_All": PS_Astro_Strong_All,
    "PS_Astro_Strong_Hard": PS_Astro_Strong_Hard,
    "PS_Astro_Strong_Hard_All": PS_Astro_Strong_Hard_All,
    "PS_Astro_Strong_GSMF": PS_Astro_Strong_GSMF,
    "PS_Astro_Strong_GMR": PS_Astro_Strong_GMR,
    "PS_Astro_Strong_MMBulge": PS_Astro_Strong_MMBulge,
    "PS_Astro_Strong_MMBulge_BFrac": PS_Astro_Strong_MMBulge_BFrac,
    "PS_Test_Astro_Strong_Covariant_MMBulge": PS_Test_Astro_Strong_Covariant_MMBulge,
    "PS_Astro_Strong_Covariant_All": PS_Astro_Strong_Covariant_All,
    "PS_Astro_Strong_Covariant_GSMF": PS_Astro_Strong_Covariant_GSMF,
}
