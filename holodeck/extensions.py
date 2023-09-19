"""Custom holodeck extensions not supported as part of the core package.
"""

from typing import Any
import holodeck as holo
from holodeck import log, cosmo
from holodeck.constants import MSOL, GYR
import numpy as np
import kalepy as kale
from sams import cyutils as sam_cyutils


class Realizer:

    def __init__(self, fobs_orb_edges, resample=None, lifetime=2*GYR, dens=None, **mmbulge_kwargs):
        pop = holo.population.Pop_Illustris()
        if resample is not None:
            mod_resamp = holo.population.PM_Resample(resample=resample)
            pop.modify(mod_resamp)
            log.info(f"Resampling population by {resample}")

        if dens is not None:
            mod_dens = holo.population.PM_Density(factor=dens)
            pop.modify(mod_dens)
            log.info(f"Modifying population density by {dens}")

        if len(mmbulge_kwargs):
            log.info(f"Modifying population masses with params: {mmbulge_kwargs}")
            mmbulge = holo.relations.MMBulge_KH2013(**mmbulge_kwargs)
            mod_mm13 = holo.population.PM_Mass_Reset(mmbulge, scatter=True)
            mt, _ = holo.utils.mtmr_from_m1m2(pop.mass)
            log.debug(f"mass bef = {holo.utils.stats(mt/MSOL)}")
            pop.modify(mod_mm13)
            mt, _ = holo.utils.mtmr_from_m1m2(pop.mass)
            log.debug(f"mass aft = {holo.utils.stats(mt/MSOL)}")

        fixed = holo.hardening.Fixed_Time.from_pop(pop, lifetime)
        evo = holo.evolution.Evolution(pop, fixed)
        evo.evolve()

        self._pop = pop
        self._evo = evo

        names, vals, weights = evo._sample_universe__at_values_weights(fobs_orb_edges)
        self._fobs_orb_edges = fobs_orb_edges
        self._vals_names = names
        self._binary_vals = vals
        self._binary_weights = weights
        return

    def __call__(self, down_sample=None):
        evo = self._evo

        fobs_orb_edges = self._fobs_orb_edges
        names = self._vals_names
        vals = self._binary_vals
        weights = self._binary_weights
        samples = evo._sample_universe__resample(fobs_orb_edges, vals, weights, down_sample)
        return names, samples


class Realizer_SAM:
    def __init__(
            self, fobs_orb_edges, sam=None, hard=None, params=None, 
            pspace=holo.param_spaces.PS_Uniform_09B(holo.log, nsamples=1, sam_shape=None, seed=None)):
        """Construct a Realizer for a given semi-analytic model and hardening model,
        or build this model using params and a pspace.

        Parameters
        ----------
        sam : Semi_Analytic_Model object or None
            Semi-analytic model instance, if not using pspace.
        hard : Fixed_Time_2PL_SAM object, GW_Only object, None
            Hardening model instance, if not using pspace.
        params : dict or None
            Parameters for a given parameter space, if sam is not provided.
        pspace : _Param_Space object
            Parameter space.
        
        """

        # check that ('sam' and 'hard') OR 'params' is provided
        if params is not None:
            if sam is not None or hard is not None:
                err = "Only 'params' or ('sam' and 'hard') should be provided."
                raise ValueError(err)
            sam, hard = pspace.model_for_params(params=params, sam_shape=pspace.sam_shape,)
        else:
            if sam is None or hard is None:
                err = "'params' or ('sam' and 'hard') must be provided."
                raise ValueError(err)
            
        self._sam = sam
        self._hard = hard
        self._fobs_orb_edges = fobs_orb_edges

    def __call__(self, nreals=100):
        """ Calculate samples and weights for an entire semi-analytic population.
        
        """

        sam = self._sam
        hard = self._hard
        fobs_orb_edges = self.fobs_orb_edges
        names = ['mtot', 'mrat', 'redz', 'fobs']

        fobs_orb_cents = kale.utils.midpoints(fobs_orb_edges)
        fobs_cents = 2.0 * fobs_orb_cents
        fobs_edges = 2.0 * fobs_orb_edges


        # ---- Calculate number of binaries in each bin

        redz_final, diff_num = sam_cyutils.dynamic_binary_number_at_fobs(
            fobs_orb_cents, sam, hard, cosmo
        )

        edges = [sam.mtot, sam.mrat, sam.redz, fobs_orb_edges]
        number = sam_cyutils.integrate_differential_number_3dx1d(edges, diff_num) # weights


            

def realizer_single_sources(params, nreals, nloudest, nfreqs=40, log10=False,
               pspace = holo.param_spaces.PS_Uniform_09B(holo.log, nsamples=1, sam_shape=None, seed=None)):
    """ Like Realizer but using single sources from a SAM instead of Illustris populations

    Parameters
    ----------
    params : dict
        model parameters for parameter space
    nreal : int
        number of realizations
    nloudest : int
        number of loudest sources to use

    Returns
    -------
    names : names of parameters
    samples : [R, 4, nloudest*nreals] NDarray
        mtot, mrat, redz, and fobs of each source in log space

    sspar is in shape [F,R,L]
    
    """
    fobs_cents, fobs_edges = holo.utils.pta_freqs(num=nfreqs)
    
    sam, hard = pspace.model_for_params(params=params, sam_shape=None,)
    _, _, sspar, bgpar = sam.gwb(
        fobs_edges, hard=hard, nreals=nreals, nloudest=nloudest, params=True)
    
    vals_names=['mtot', 'mrat', 'redz', 'fobs']
    fobs = np.repeat(fobs_cents, nreals*nloudest).reshape(nfreqs, nreals, nloudest)
    mtot = sspar[0] # g
    mrat = sspar[1]
    redz = sspar[3] # final redshift, not initial

    samples = []
    for par in [mtot, mrat, redz, fobs]: # starts in shape F,R,L
        par = np.swapaxes(par, 0, 1) # R,F,L
        par = par.reshape(nreals, nfreqs*nloudest) # R, F*L
        if log10:    
            samples.append(np.log10(par))
        else:
            samples.append(par)
    samples = np.array(samples) # 4,R,F*L
    samples = np.swapaxes(samples, 0,1) # R,4,F*L

    return vals_names, samples