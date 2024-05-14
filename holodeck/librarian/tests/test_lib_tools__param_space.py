"""Test holodeck/librarian/lib_tools.py: the parameter space base class, and parameter distributions.
"""

import numpy as np

# Import package, test suite, and other packages as needed
import holodeck as holo
# import pytest  # noqa

from holodeck import sams, host_relations, hardening, librarian
from holodeck.constants import GYR
from holodeck.librarian import (
    param_spaces_dict
)
from holodeck.librarian.lib_tools import (
    _Param_Space, PD_Uniform,
)

MMB_MAMP_LOG10_EXTR = [+7.5, +9.5]
GSMF_PHI0_LOG10_EXTR = [-2.5, -3.0]


class PS_Test_Wout_Defaults(_Param_Space):
    """Simple test parameter space in 2D, NOT including default parameters (`DEFAULTS`).
    """

    def __init__(self, log=None, nsamples=None, sam_shape=None, seed=None):
        parameters = [
            PD_Uniform("mmb_mamp_log10", *MMB_MAMP_LOG10_EXTR),   # [log10(Msol)]
            PD_Uniform("gsmf_phi0_log10", *GSMF_PHI0_LOG10_EXTR),
        ]
        super().__init__(
            parameters,
            log=log, nsamples=nsamples, sam_shape=sam_shape, seed=seed,
        )
        return

    @classmethod
    def _init_sam(cls, sam_shape, params):
        gsmf = sams.GSMF_Schechter(
            phi0=params['gsmf_phi0_log10'],
        )
        gpf = sams.GPF_Power_Law()
        gmt = sams.GMT_Power_Law()
        mmbulge = host_relations.MMBulge_KH2013(
            mamp_log10=params['mmb_mamp_log10'],
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


class PS_Test_With_Defaults(_Param_Space):
    """Simple test parameter space in 2D, NOT including default parameters (`DEFAULTS`).
    """

    DEFAULTS = dict(
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

    def __init__(self, log=None, nsamples=None, sam_shape=None, seed=None):
        parameters = [
            PD_Uniform("mmb_mamp_log10", *MMB_MAMP_LOG10_EXTR),   # [log10(Msol)]
            PD_Uniform("gsmf_phi0_log10", *GSMF_PHI0_LOG10_EXTR),
        ]
        super().__init__(
            parameters,
            log=log, nsamples=nsamples, sam_shape=sam_shape, seed=seed,
        )
        return

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
        hard = hardening.Hard_GW()
        return hard


def _run_pspace_no_nsamples(Param_Space_Class):
    pspace = Param_Space_Class(holo.log)

    # Without setting `nsamples` in the initializer, there shouldn't be any samples chosen
    assert pspace._uniform_samples is None, f"{pspace._uniform_samples=} should be `None`!"
    assert pspace.param_samples is None, f"{pspace.param_samples=} should be `None`!"

    # Make sure there are two parameters
    assert pspace.nparameters == 2, f"Expected 2 parameters, but found `{pspace.nparameters=}`"
    assert len(pspace._parameters) == 2, f"Expected {len(pspace._parameters)=} to equal 2!"
    assert len(pspace.param_names) == 2, f"Expected {len(pspace.param_names)=} to equal 2!"
    for ii, (pn, test) in enumerate(zip(pspace.param_names, ['mmb_mamp_log10', 'gsmf_phi0_log10'])):
        assert pn == test, f"Parameter name {ii} = '{pn}' does not match input '{test}'!"

    # Make sure parameters's extrema match input values
    extr = pspace.extrema
    for name, ex, input in zip(pspace.param_names, extr, [MMB_MAMP_LOG10_EXTR, GSMF_PHI0_LOG10_EXTR]):
        for ii in range(2):
            assert np.isclose(ex[ii], input[ii]), f"{name} extrema {ii} is {ex[ii]}, does not match input {input[ii]}!"

    return


def _run_pspace_nsamples(Param_Space_Class):
    NSAMPLES = 10
    SAM_SHAPE = 11
    pspace = Param_Space_Class(holo.log, nsamples=NSAMPLES, sam_shape=SAM_SHAPE)

    # Make sure shapes match expectations
    shape = (NSAMPLES, 2)
    assert pspace._uniform_samples.shape == shape, f"{pspace._uniform_samples.shape=} should be `{shape}`!"
    assert pspace.param_samples.shape == shape, f"{pspace.param_samples.shape=} should be `{shape}`!"
    samps = pspace._uniform_samples
    assert np.all((0.0 < samps) & (samps < 1.0)), "`uniform_samples` are not between (0.0, 1.0)!  "

    # Make sure there are two parameters
    assert pspace.nparameters == 2, f"Expeted 2 parameters, but found `{pspace.nparameters=}`"
    assert len(pspace._parameters) == 2, f"Expeted {len(pspace._parameters)=} to equal 2!"
    assert len(pspace.param_names) == 2, f"Expeted {len(pspace.param_names)=} to equal 2!"
    for ii, (pn, test) in enumerate(zip(pspace.param_names, ['mmb_mamp_log10', 'gsmf_phi0_log10'])):
        assert pn == test, f"Parameter name {ii} = '{pn}' does not match input '{test}'!"

    # Make sure parameters's extrema match input values
    extr = pspace.extrema
    for name, ex, input in zip(pspace.param_names, extr, [MMB_MAMP_LOG10_EXTR, GSMF_PHI0_LOG10_EXTR]):
        for ii in range(2):
            assert np.isclose(ex[ii], input[ii]), f"{name} extrema {ii} is {ex[ii]}, does not match input {input[ii]}!"

    # Load model, check it
    sam, hard = pspace.model_for_sample_number(0)
    _check_sam_hard(sam, hard, SAM_SHAPE)

    SAMP = 0
    params = pspace.param_dict(SAMP)
    for ii, (pn, name) in enumerate(zip(params.keys(), pspace.param_names)):
        assert pn == name, f"Parameter {ii} does not match between returned params {pn} and param name {name}!"
        assert params[pn] == pspace.param_samples[SAMP][ii]

    return


def _check_sam_hard(sam, hard, sam_shape):
    shape = (sam_shape, sam_shape, sam_shape)
    assert sam.shape == shape, f"{sam.shape=} does not match input {shape}!"
    assert isinstance(hard, hardening.Hard_GW)

    # Make sure model runs
    import holodeck.librarian.lib_tools  # noqa
    data = librarian.lib_tools.run_model(sam, hard)
    assert data is not None

    return


def test__ps_test_wout_defaults():
    pspace_class = PS_Test_Wout_Defaults
    _run_pspace_no_nsamples(pspace_class)
    _run_pspace_nsamples(pspace_class)

    params = dict(
        mmb_mamp_log10=np.mean(MMB_MAMP_LOG10_EXTR),
        gsmf_phi0_log10=np.mean(GSMF_PHI0_LOG10_EXTR),
    )

    # class (not instance) should also be able to generate model
    SAM_SHAPE = 11
    sam, hard = pspace_class().model_for_params(params, sam_shape=SAM_SHAPE)
    _check_sam_hard(sam, hard, SAM_SHAPE)

    return


def test__ps_test_with_defaults():
    pspace_class = PS_Test_With_Defaults
    _run_pspace_no_nsamples(pspace_class)
    _run_pspace_nsamples(pspace_class)

    params = dict(
        mmb_mamp_log10=np.mean(MMB_MAMP_LOG10_EXTR),
        gsmf_phi0_log10=np.mean(GSMF_PHI0_LOG10_EXTR),
    )

    SAM_SHAPE = 11
    sam, hard = pspace_class().model_for_params(params, sam_shape=SAM_SHAPE)
    _check_sam_hard(sam, hard, SAM_SHAPE)

    return


def _check_ps_basics(name, pspace_class):
    # all default arguments
    space = pspace_class()

    param_names = space.param_names
    params = space._parameters
    nparams = space.nparameters
    assert len(param_names) == nparams
    assert len(params) == nparams

    # check defaults
    def_params = space.default_params()
    for ii, (name, dp) in enumerate(def_params.items()):
        assert dp == params[ii].default

    np.random.seed(12345)
    for ii in range(nparams):
        # Choose random normalized values for each parameter
        # must convert to list so that we can replace values with `None`
        norm_vals = np.random.uniform(0.0, 1.0, size=nparams).tolist()
        # Set parameter `ii` to default value (`None`)
        norm_vals[ii] = None
        dist_vals = space.normalized_params(norm_vals)
        for jj, (pnam, pval) in enumerate(dist_vals.items()):
            # make sure default value is set
            if jj == ii:
                def_val = params[jj].default
                err = f"default was not set for {jj}: `{pnam}`!  value={pval}, def={def_val}"
                assert pval == def_val, err
            # make sure other values are set
            else:
                truth = params[jj](norm_vals[jj])
                err = f"normalized param was not set for {jj}: `{pnam}`!  value={pval} {truth=}"
                assert pval == truth, err

    return

def test_basics_all_param_spaces():
    """Run `_check_ps_basics` on all registered parameters-space classes.
    """
    for name, pspace_class in param_spaces_dict.items():
        print(f"Checking '{name}' ({pspace_class})")
        _check_ps_basics(name, pspace_class)

    return