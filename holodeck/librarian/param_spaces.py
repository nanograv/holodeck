"""Parameter-Space definitions for holodeck libraries.
"""

import holodeck as holo
from holodeck.constants import GYR, PC, MSOL
from holodeck.librarian.libraries import _Param_Space, PD_Uniform, PD_Normal


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
        hard_time=3.0,          # [Gyr]
        hard_sepa_init=1e4,     # [pc]
        hard_rchar=100.0,       # [pc]
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
        gmr_norm0_log10=-2.2287,         # -2.2287 ± 0.0045    A0 [log10(A*Gyr)]
        gmr_normz=+2.4644,               # +2.4644 ± 0.0128    eta
        gmr_malpha0=+0.2241,             # +0.2241 ± 0.0038    alpha0
        gmr_malphaz=-1.1759,             # -1.1759 ± 0.0316    alpha1
        gmr_mdelta0=+0.7668,             # +0.7668 ± 0.0202    delta0
        gmr_mdeltaz=-0.4695,             # -0.4695 ± 0.0440    delta1
        gmr_qgamma0=-1.2595,             # -1.2595 ± 0.0026    beta0
        gmr_qgammaz=+0.0611,             # +0.0611 ± 0.0021    beta1
        gmr_qgammam=-0.0477,             # -0.0477 ± 0.0013    gamma

        # M-MBulge Relationship (``MMBulge_KH2013``)
        # From [KH2013]_
        mmb_mamp=0.49e9,                 # 0.49e9 + 0.06 - 0.05  [Msol]
        mmb_plaw=1.17,                   # 1.17 ± 0.08
        mmb_scatter_dex=0.28,            # no uncertainties given
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
            PD_Uniform("hard_time", 0.1, 11.0, default=3.0),   # [Gyr]
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
        _Param_Space.__init__(
            self, parameters,
            log=log, nsamples=nsamples, sam_shape=sam_shape, seed=seed,
            **kwargs
        )
        return

    # Define the function which actually constructs the SAM, using a dictionary of model parameters.
    # This is not intended as an API function, but an internal method used to build the SAM model.
    # The call signature of this function should *not* be changed, and the function must always
    # return a single object: the instance of :py:class:`holodeck.sams.sam.Semi_Analytic_Model`.
    def _init_sam(self, sam_shape, params):

        # Schechter Galaxy Stellar-Mass Function
        gsmf = holo.sams.GSMF_Schechter(
            phi0=params['gsmf_phi0_log10'],
            phiz=params['gsmf_phiz'],
            mchar0_log10=params['gsmf_mchar0_log10'],
            mcharz=params['gsmf_mcharz'],
            alpha0=params['gsmf_alpha0'],
            alphaz=params['gsmf_alphaz'],
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

        # Notice that a unit-conversion is being performed here, in the M-Mbulge constructor.  The
        # parameter space is defined such that the normalization is in units of solar-masses, while
        # the M-Mbulge class itself is defined such that the normalization is in units of grams.
        # This is a very easy place to make mistakes, and so any/all units (and particularly unit
        # conversions) should be described carefully in docstrings.
        mmbulge = holo.host_relations.MMBulge_KH2013(
            mamp=params['mmb_mamp']*MSOL,
            mplaw=params['mmb_plaw'],
            scatter_dex=params['mmb_scatter_dex'],
        )

        sam = holo.sams.Semi_Analytic_Model(
            gsmf=gsmf, gmr=gmr, mmbulge=mmbulge, shape=sam_shape,
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
            params['hard_time']*GYR,
            sepa_init=params['hard_sepa_init']*PC,
            rchar=params['hard_rchar']*PC,
            gamma_inner=params['hard_gamma_inner'],
            gamma_outer=params['hard_gamma_outer'],
        )
        return hard


class _PS_Astro_Strong(_Param_Space):
    """SAM Model with strongly astrophysically-motivated parameters.

    This model uses a double-Schechter GSMF, an Illustris-derived galaxy merger rate, a Kormendy+Ho
    M-MBulge relationship, and a phenomenology binary evolution model.

    """

    __version__ = "0.1"

    DEFAULTS = dict(
        # Hardening model (phenom 2PL)
        hard_time=3.0,          # [Gyr]
        hard_sepa_init=1e4,     # [pc]
        hard_rchar=10.0,        # [pc]
        hard_gamma_inner=-1.0,
        hard_gamma_outer=+2.5,

        # Galaxy stellar-mass Function (``GSMF_Double_Schechter``)
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

        # Galaxy merger rate (``GMR_Illustris``)
        # Parameters are taken directly from [Rodriguez-Gomez2015]_
        gmr_norm0_log10=-2.2287,         # -2.2287 ± 0.0045    A0 [log10(A*Gyr)]
        gmr_normz=+2.4644,               # +2.4644 ± 0.0128    eta
        gmr_malpha0=+0.2241,             # +0.2241 ± 0.0038    alpha0
        gmr_malphaz=-1.1759,             # -1.1759 ± 0.0316    alpha1
        gmr_mdelta0=+0.7668,             # +0.7668 ± 0.0202    delta0
        gmr_mdeltaz=-0.4695,             # -0.4695 ± 0.0440    delta1
        gmr_qgamma0=-1.2595,             # -1.2595 ± 0.0026    beta0
        gmr_qgammaz=+0.0611,             # +0.0611 ± 0.0021    beta1
        gmr_qgammam=-0.0477,             # -0.0477 ± 0.0013    gamma

        # M-MBulge Relationship (``MMBulge_KH2013``)
        # From [KH2013]_
        mmb_mamp=0.49e9,                 # 0.49e9 + 0.06 - 0.05  [Msol]
        mmb_plaw=1.17,                   # 1.17 ± 0.08
        mmb_scatter_dex=0.28,            # no uncertainties given
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

        mmbulge = holo.host_relations.MMBulge_KH2013(
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


class PS_Astro_Strong_All(_PS_Astro_Strong):

    def __init__(self, log=None, nsamples=None, sam_shape=None, seed=None):
        parameters = [
            # Hardening model (phenom 2PL)
            PD_Uniform("hard_time", 0.1, 11.0, default=3.0),   # [Gyr]
            PD_Uniform("hard_gamma_inner", -1.5, +0.0, default=-1.0),

            # GSMF
            PD_Normal('gsmf_log10_phi_one_z0', -2.383, 0.028),    # - 2.383 ± 0.028
            PD_Normal('gsmf_log10_phi_one_z1', -0.264, 0.072),    # - 0.264 ± 0.072
            PD_Normal('gsmf_log10_phi_one_z2', -0.107, 0.031),    # - 0.107 ± 0.031
            PD_Normal('gsmf_log10_phi_two_z0', -2.818, 0.050),    # - 2.818 ± 0.050
            PD_Normal('gsmf_log10_phi_two_z1', -0.368, 0.070),    # - 0.368 ± 0.070
            PD_Normal('gsmf_log10_phi_two_z2', +0.046, 0.020),    # + 0.046 ± 0.020
            PD_Normal('gsmf_log10_mstar_z0', +10.767, 0.026),     # +10.767 ± 0.026
            PD_Normal('gsmf_log10_mstar_z1', +0.124, 0.045),      # + 0.124 ± 0.045
            PD_Normal('gsmf_log10_mstar_z2', -0.033, 0.015),      # - 0.033 ± 0.015
            PD_Normal('gsmf_alpha_one', -0.28, 0.070),            # - 0.280 ± 0.070
            PD_Normal('gsmf_alpha_two', -1.48, 0.150),            # - 1.480 ± 0.150

            # GMR
            PD_Normal('gmr_norm0_log10', -2.2287, 0.0045),        # -2.2287 ± 0.0045    A0 [log10(A*Gyr)]
            PD_Normal('gmr_normz', +2.4644, 0.0128),              # +2.4644 ± 0.0128    eta
            PD_Normal('gmr_malpha0', +0.2241, 0.0038),            # +0.2241 ± 0.0038    alpha0
            PD_Normal('gmr_malphaz', -1.1759, 0.0316),            # -1.1759 ± 0.0316    alpha1
            PD_Normal('gmr_mdelta0', +0.7668, 0.0202),            # +0.7668 ± 0.0202    delta0
            PD_Normal('gmr_mdeltaz', -0.4695, 0.0440),            # -0.4695 ± 0.0440    delta1
            PD_Normal('gmr_qgamma0', -1.2595, 0.0026),            # -1.2595 ± 0.0026    beta0
            PD_Normal('gmr_qgammaz', +0.0611, 0.0021),            # +0.0611 ± 0.0021    beta1
            PD_Normal('gmr_qgammam', -0.0477, 0.0013),            # -0.0477 ± 0.0013    gamma

            # From [KH2013]_
            PD_Normal('mmb_mamp', 0.49e9, 0.055e9),               # 0.49e9 + 0.06 - 0.05  [Msol]
            PD_Normal('mmb_plaw', 1.17, 0.08),                    # 1.17 ± 0.08
        ]
        _Param_Space.__init__(
            self, parameters,
            log=log, nsamples=nsamples, sam_shape=sam_shape, seed=seed,
        )
        return


class PS_Astro_Strong_Hard(_PS_Astro_Strong):

    def __init__(self, log=None, nsamples=None, sam_shape=None, seed=None):
        parameters = [
            # Hardening model (phenom 2PL)
            PD_Uniform("hard_time", 0.1, 11.0, default=3.0),   # [Gyr]
            PD_Uniform("hard_gamma_inner", -1.5, +0.0, default=-1.0),
        ]
        _Param_Space.__init__(
            self, parameters,
            log=log, nsamples=nsamples, sam_shape=sam_shape, seed=seed,
        )
        return


class PS_Astro_Strong_GSMF(_PS_Astro_Strong):

    def __init__(self, log=None, nsamples=None, sam_shape=None, seed=None):
        parameters = [
            # GSMF
            PD_Normal('gsmf_log10_phi_one_z0', -2.383, 0.028),    # - 2.383 ± 0.028
            PD_Normal('gsmf_log10_phi_one_z1', -0.264, 0.072),    # - 0.264 ± 0.072
            PD_Normal('gsmf_log10_phi_one_z2', -0.107, 0.031),    # - 0.107 ± 0.031
            PD_Normal('gsmf_log10_phi_two_z0', -2.818, 0.050),    # - 2.818 ± 0.050
            PD_Normal('gsmf_log10_phi_two_z1', -0.368, 0.070),    # - 0.368 ± 0.070
            PD_Normal('gsmf_log10_phi_two_z2', +0.046, 0.020),    # + 0.046 ± 0.020
            PD_Normal('gsmf_log10_mstar_z0', +10.767, 0.026),     # +10.767 ± 0.026
            PD_Normal('gsmf_log10_mstar_z1', +0.124, 0.045),      # + 0.124 ± 0.045
            PD_Normal('gsmf_log10_mstar_z2', -0.033, 0.015),      # - 0.033 ± 0.015
            PD_Normal('gsmf_alpha_one', -0.28, 0.070),            # - 0.280 ± 0.070
            PD_Normal('gsmf_alpha_two', -1.48, 0.150),            # - 1.480 ± 0.150
        ]
        _Param_Space.__init__(
            self, parameters,
            log=log, nsamples=nsamples, sam_shape=sam_shape, seed=seed,
        )
        return


class PS_Astro_Strong_GMR(_PS_Astro_Strong):

    def __init__(self, log=None, nsamples=None, sam_shape=None, seed=None):
        parameters = [
            # GMR
            PD_Normal('gmr_norm0_log10', -2.2287, 0.0045),        # -2.2287 ± 0.0045    A0 [log10(A*Gyr)]
            PD_Normal('gmr_normz', +2.4644, 0.0128),              # +2.4644 ± 0.0128    eta
            PD_Normal('gmr_malpha0', +0.2241, 0.0038),            # +0.2241 ± 0.0038    alpha0
            PD_Normal('gmr_malphaz', -1.1759, 0.0316),            # -1.1759 ± 0.0316    alpha1
            PD_Normal('gmr_mdelta0', +0.7668, 0.0202),            # +0.7668 ± 0.0202    delta0
            PD_Normal('gmr_mdeltaz', -0.4695, 0.0440),            # -0.4695 ± 0.0440    delta1
            PD_Normal('gmr_qgamma0', -1.2595, 0.0026),            # -1.2595 ± 0.0026    beta0
            PD_Normal('gmr_qgammaz', +0.0611, 0.0021),            # +0.0611 ± 0.0021    beta1
            PD_Normal('gmr_qgammam', -0.0477, 0.0013),            # -0.0477 ± 0.0013    gamma
        ]
        _Param_Space.__init__(
            self, parameters,
            log=log, nsamples=nsamples, sam_shape=sam_shape, seed=seed,
        )
        return


class PS_Astro_Strong_MMBulge(_PS_Astro_Strong):

    def __init__(self, log=None, nsamples=None, sam_shape=None, seed=None):
        parameters = [
            # MMbulge - from [KH2013]_
            PD_Normal('mmb_mamp', 0.49e9, 0.055e9),               # 0.49e9 + 0.06 - 0.05  [Msol]
            PD_Normal('mmb_plaw', 1.17, 0.08),                    # 1.17 ± 0.08
        ]
        _Param_Space.__init__(
            self, parameters,
            log=log, nsamples=nsamples, sam_shape=sam_shape, seed=seed,
        )
        return


_param_spaces_dict = {
    'PS_Astro_Strong_All': PS_Astro_Strong_All,
    'PS_Astro_Strong_Hard': PS_Astro_Strong_Hard,
    'PS_Astro_Strong_GSMF': PS_Astro_Strong_GSMF,
    'PS_Astro_Strong_GMR': PS_Astro_Strong_GMR,
    'PS_Astro_Strong_MMBulge': PS_Astro_Strong_MMBulge,
}

