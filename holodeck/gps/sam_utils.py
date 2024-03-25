#!/usr/bin/env python3
"""Utilities for SAMs."""
import holodeck as holo
from holodeck.constants import MSOL, GYR, PC
import abc
import sys

# Default parameters
DEF_SAM_SHAPE = 50
DEF_NUM_REALS = 100
DEF_NUM_FBINS = 40
DEF_PTA_DUR = 16.03  # [yrs]

DEF_ECCEN_NUM_STEPS = 123
DEF_ECCEN_NHARMS = 100


class SamModel(abc.ABC):
    """An abstract class for GWB calculations using SAMs.

    Attributes
    ----------
    param_names : list
        Parameters expected for SAM

    Methods
    -------
    sam_for_params
        Construct SAM for given parameter input
    validate_params
        Check that supplied parameters match those expected for the SAM


    Examples
    --------
    FIXME: Add docs.


    """

    _PARAM_NAMES = []

    def __init__(self, param_names=_PARAM_NAMES):
        self.param_names = param_names

    @abc.abstractstaticmethod
    def sam_for_params(env_pars, sam_shape):
        """Abstract method to be implemented in each child class.

        Given an input dictionary `env_pars` of form parameter:value, return a
        SAM and hardening model.

        Parameters
        ----------
        env_pars : dict
            Dictionary of form parameter:value
        sam_shape : int
            Shape of SAM grid

        Examples
        --------
        FIXME: Add docs.


        """
        pass

    def validate_params(self, env_pars):
        """Validate the supplied parameters.

        Parameters
        ----------
        env_pars : dict
            Dictionary of form parameter:value

        Raises
        ------
        KeyError
            If the supplied parameters are not the expected parameters, raise an
            error

        Examples
        --------
        FIXME: Add docs.

        """
        # Sometimes the libraries log the parameter names as byte string instead of a normal string
        pars = [
            par.decode("ascii") if isinstance(par, bytes) else par
            for par in env_pars.keys()
        ]

        if set(pars) != set(self.param_names):
            raise KeyError(f"These parameters are not valid.\n Expected {self.param_names} and got {list(env_pars.keys())}.")


class Hard04(SamModel):

    _PARAM_NAMES = [
        'hard_time',
        'hard_gamma_inner',
        'hard_gamma_outer',
        'hard_rchar',
        'gsmf_phi0',
        'mmb_amp',
    ]

    def __init__(self, param_names=_PARAM_NAMES):
        super().__init__(param_names=param_names)

    def sam_for_params(self, env_pars, sam_shape):
        """
        Parameters
        ----------
        env_pars : dict
            A dictionary containing the environmental parameters. The keys are:
            * time: The time at which to calculate the hardening.
            * gamma_inner: The inner slope of the hardening profile.
            * gamma_outer: The outer slope of the hardening profile.
            * rchar: The characteristic radius of the hardening profile.
            * gsmf_phi0: The normalization of the galaxy stellar mass function (GSMF).
            * mmb_amp: The amplitude of the bulge mass-to-light ratio relation.
        sam_shape : tuple
            The expected shape of the SAM.

        Returns
        -------
        """
        self.validate_params(env_pars)
        time, gamma_inner, gamma_outer, rchar, gsmf_phi0, mmb_amp = env_pars.values(
        )
        time = (10.0**time) * GYR
        rchar = (10.0**rchar) * PC
        mmb_amp = (10**mmb_amp) * MSOL

        gsmf = holo.sam.GSMF_Schechter(phi0=gsmf_phi0)
        gpf = holo.sam.GPF_Power_Law()
        gmt = holo.sam.GMT_Power_Law()
        mmbulge = holo.host_relations.MMBulge_KH2013(mamp=mmb_amp)

        sam = holo.sam.Semi_Analytic_Model(gsmf=gsmf,
                                           gpf=gpf,
                                           gmt=gmt,
                                           mmbulge=mmbulge,
                                           shape=sam_shape)
        hard = holo.hardening.Fixed_Time.from_sam(sam,
                                                  time,
                                                  rchar=rchar,
                                                  gamma_sc=gamma_inner,
                                                  gamma_df=gamma_outer,
                                                  exact=True,
                                                  progress=False)
        return sam, hard


class Eccen01(SamModel):

    _PARAM_NAMES = [
        'eccen_init',
        'gsmf_phi0',
        'gpf_zbeta',
        'mmb_amp',
    ]

    SEPA_INIT = 1.0 * PC

    def __init__(self, param_names=_PARAM_NAMES):
        super().__init__(param_names=param_names)

    def sam_for_params(self, env_pars, sam_shape):
        """Creates a Semi-Analytic Model (SAM) from the given environmental parameters and shape.

        Parameters
        ----------
        env_pars : dict
            A dictionary containing the environmental parameters. The keys are:
            * eccen: The eccentricity of the SAM.
            * gsmf_phi0: The normalization of the galaxy stellar mass function (GSMF).
            * gpf_zbeta: The redshift slope of the galaxy power spectrum (GPF).
            * mmb_amp: The amplitude of the bulge mass-to-light ratio relation.

        sam_shape : tuple
            The expected shape of the SAM.

        Returns
        -------
        sam : holo.sam.Semi_Analytic_Model
            The created SAM.
        sepa_evo : holo.sam.SEPA_Evol
            The evolution of the SAM's SEPA.
        eccen_evo : holo.sam.ECCEN_Evol
            The evolution of the SAM's eccentricity.
        """
        self.validate_params(env_pars)

        eccen, gsmf_phi0, gpf_zbeta, mmb_amp = env_pars.values()
        mmb_amp = mmb_amp * MSOL

        # favor higher values of eccentricity instead of uniformly distributed
        eccen = eccen**(1.0 / 5.0)

        gsmf = holo.sam.GSMF_Schechter(phi0=gsmf_phi0)
        gpf = holo.sam.GPF_Power_Law(zbeta=gpf_zbeta)
        gmt = holo.sam.GMT_Power_Law()
        mmbulge = holo.host_relations.MMBulge_KH2013(mamp=mmb_amp)

        sam = holo.sam.Semi_Analytic_Model(gsmf=gsmf,
                                           gpf=gpf,
                                           gmt=gmt,
                                           mmbulge=mmbulge,
                                           shape=sam_shape)

        sepa_evo, eccen_evo = holo.sam.evolve_eccen_uniform_single(
            sam, eccen, self.SEPA_INIT, DEF_ECCEN_NUM_STEPS)

        return sam, sepa_evo, eccen_evo


class BigCirc01(SamModel):

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

    SEPA_INIT = 1.0 * PC

    def __init__(self, param_names=_PARAM_NAMES):
        super().__init__(param_names=param_names)

    def sam_for_params(self, env_pars, sam_shape):
        self.validate_params(env_pars)

        hard_time, hard_rchar, gamma_inner, gamma_outer, \
            gsmf_phi0, gsmf_phiz, gsmf_alpha0, \
            gpf_malpha, gpf_zbeta, gpf_qgamma, \
            gmt_malpha, gmt_zbeta, gmt_qgamma, \
            mmb_amp, mmb_plaw = env_pars.values()

        mmb_amp = (10.0**mmb_amp) * MSOL
        hard_time = (10.0**hard_time) * GYR
        hard_rchar = (10.0**hard_rchar) * PC

        gsmf = holo.sam.GSMF_Schechter(phi0=gsmf_phi0,
                                       phiz=gsmf_phiz,
                                       alpha0=gsmf_alpha0)
        gpf = holo.sam.GPF_Power_Law(malpha=gpf_malpha,
                                     qgamma=gpf_qgamma,
                                     zbeta=gpf_zbeta)
        gmt = holo.sam.GMT_Power_Law(malpha=gmt_malpha,
                                     qgamma=gmt_qgamma,
                                     zbeta=gmt_zbeta)
        mmbulge = holo.host_relations.MMBulge_KH2013(mamp=mmb_amp, mplaw=mmb_plaw)

        sam = holo.sam.Semi_Analytic_Model(gsmf=gsmf,
                                           gpf=gpf,
                                           gmt=gmt,
                                           mmbulge=mmbulge,
                                           shape=sam_shape)
        hard = holo.hardening.Fixed_Time.from_sam(sam,
                                                  hard_time,
                                                  rchar=hard_rchar,
                                                  gamma_sc=gamma_inner,
                                                  gamma_df=gamma_outer,
                                                  exact=True,
                                                  progress=False)
        return sam, hard


class PS_2Par_Circ_01(SamModel):

    _PARAM_NAMES = ['hard_time', 'gsmf_phi0']

    def __init__(self, param_names=_PARAM_NAMES):
        super().__init__(param_names=param_names)

    def sam_for_params(self, env_pars, sam_shape):
        self.validate_params(env_pars)
        time, gsmf_phi0 = env_pars.values()

        time = (10.0**time) * GYR

        gsmf = holo.sam.GSMF_Schechter(phi0=gsmf_phi0)
        gpf = holo.sam.GPF_Power_Law()
        gmt = holo.sam.GMT_Power_Law()
        mmbulge = holo.host_relations.MMBulge_KH2013()

        sam = holo.sam.Semi_Analytic_Model(gsmf=gsmf,
                                           gpf=gpf,
                                           gmt=gmt,
                                           mmbulge=mmbulge,
                                           shape=sam_shape)
        hard = holo.hardening.Fixed_Time.from_sam(sam, time, progress=False)
        return sam, hard


class PS_Circ_01(SamModel):

    _PARAM_NAMES = [
        'hard_time',
        'hard_gamma_inner',
        'gsmf_phi0',
        'gsmf_mchar0',
        'gsmf_alpha0',
        'gpf_zbeta',
        'gpf_qgamma',
        'gmt_norm',
        'gmt_zbeta',
        'mmb_amp',
        'mmb_plaw',
        'mmb_scatter',
    ]

    def __init__(self, param_names=_PARAM_NAMES):
        super().__init__(param_names=param_names)

    def sam_for_params(self, env_pars, sam_shape):
        self.validate_params(env_pars)

        hard_time, hard_gamma_inner, \
            gsmf_phi0, gsmf_mchar0, gsmf_alpha0, \
            gpf_zbeta, gpf_qgamma, \
            gmt_norm, gmt_zbeta, \
            mmb_amp, mmb_plaw, mmb_scatter = env_pars.values()

        mmb_amp = (10.0**mmb_amp) * MSOL
        hard_time = (10.0**hard_time) * GYR
        gmt_norm = gmt_norm * GYR

        gsmf = holo.sam.GSMF_Schechter(phi0=gsmf_phi0,
                                       mchar0_log10=gsmf_mchar0,
                                       alpha0=gsmf_alpha0)
        gpf = holo.sam.GPF_Power_Law(qgamma=gpf_qgamma, zbeta=gpf_zbeta)
        gmt = holo.sam.GMT_Power_Law(time_norm=gmt_norm, zbeta=gmt_zbeta)
        mmbulge = holo.host_relations.MMBulge_KH2013(mamp=mmb_amp,
                                                mplaw=mmb_plaw,
                                                scatter_dex=mmb_scatter)

        sam = holo.sam.Semi_Analytic_Model(gsmf=gsmf,
                                           gpf=gpf,
                                           gmt=gmt,
                                           mmbulge=mmbulge,
                                           shape=sam_shape)
        hard = holo.hardening.Fixed_Time.from_sam(sam,
                                                  hard_time,
                                                  gamma_sc=hard_gamma_inner,
                                                  progress=False)
        return sam, hard


class Broad_Uniform_02B(SamModel):
    _PARAM_NAMES = [
        'hard_time',
        'gsmf_phi0',
        'gsmf_mchar0_log10',
        'mmb_amp_log10',
        'mmb_scatter',
    ]

    def __init__(self, param_names=_PARAM_NAMES):
        super().__init__(param_names=param_names)

    def sam_for_params(self, env_pars, sam_shape):
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


        self.validate_params(env_pars)

        hard_time, gsmf_phi0, gsmf_mchar0_log10, mmb_amp_log10, mmb_scatter = env_pars.values(
        )
        hard_time *= GYR

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
        mmbulge = holo.host_relations.MMBulge_KH2013(
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
