"""Custom holodeck extensions not supported as part of the core package.
"""

import holodeck as holo
from holodeck import log, cosmo, gravwaves
from holodeck.constants import MSOL, GYR
import numpy as np
import kalepy as kale
from holodeck.sams import cyutils as sam_cyutils


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
        
            
        NOTE: To match the Realizer above I could initialize with weights and whatnot, then
        possibly use the same resample/downsample function.
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

    def __call__(self, nreals=100, clean=False):
        """ Calculate samples and weights for an entire semi-analytic population.
        
        Parameters
        ----------
        nreals : int
            Number of realizations
        clean : boolean
            Whether or not to make a samples array for every realization 
            and clean weights==zero bins from each array

        Returns
        -------
        names : array of strings
            Names of the parameters returned in samples
        samples : array of R or 4 NDarrays
            if clean: R arrays of 4 x N_clean NDarrays [R,] x [4,N_clean] for each realization
            else: NDarrays for mass, ratio, redshift, and frequency [4,M*Q*Z*F]
        weights : array of R NDarrays 
            array of number of sources per sample bin for R
            If clean, the shape is [R,] arrays of len N_clean for each realizations, with zero values removed.
            Otherwise, the shape is [R, M*Q*Z*F]

        """

        sam = self._sam
        hard = self._hard
        fobs_orb_edges = self._fobs_orb_edges

        fobs_orb_cents = kale.utils.midpoints(fobs_orb_edges)
        fobs = 2.0 * fobs_orb_cents


        # ---- Calculate number of binaries in each bin

        redz, diff_num = sam_cyutils.dynamic_binary_number_at_fobs(
            fobs_orb_cents, sam, hard, cosmo
        )

        edges = [sam.mtot, sam.mrat, sam.redz, fobs_orb_edges]
        number = sam_cyutils.integrate_differential_number_3dx1d(edges, diff_num) # fractional number per bin

        samples = get_samples_from_edges(edges, redz, number.shape, flatten=True)
        names = ['mtot', 'mrat', 'redz', 'fobs']
        number = number.flatten()
        shape = (number.size, nreals)
        weights = gravwaves.poisson_as_needed(number[..., np.newaxis] * np.ones(shape)).reshape(shape)

        

        if clean:
            nonzero_samples = []
            nonzero_weights = []
            for rr in range(nreals):
                nonzero = weights[:,rr]!=0
                mtot = samples[0][nonzero]
                mrat = samples[1][nonzero]
                redz = samples[2][nonzero]
                fobs = samples[3][nonzero]
                nonzero_samples.append([mtot, mrat, redz, fobs])
                nonzero_weights.append(weights[:,rr][nonzero])
    
            weights = nonzero_weights
            samples = nonzero_samples

        return names, samples, weights
            

def get_samples_from_edges(edges, redz, number_shape, flatten=True):
    """ Get the sample parameters for every bin center and return as flattened arrays.

    Parameters
    ----------
    edges : array of [M+1,], [Q+1,], [Z+1,], and [F+1,] NDarrays
        Edges for mtot, mrat, redz, fobs_orb_edges 
    redz : [M+1, Q+1, Z+1, F+1] NDarray
        Final redshifts
    number_shape : array
        Shape [M,Q,Z,F]
    flatten : boolean
        Whether or not to flatten each sample array
    
    Returns
    -------
    samples : array of 4 flattened [M*Q*Z*F,] NDarrays
    
    """
    
     # ---- Find bin center properties
    mtot = kale.utils.midpoints(edges[0]) #: total mass
    mrat = kale.utils.midpoints(edges[1]) #: mass ratio
    fobs_orb_cents = kale.utils.midpoints(edges[3]) 
    fobs = 2.0 * fobs_orb_cents           #: gw fobs

    for dd in range(3):
        redz = np.moveaxis(redz, dd, 0)
        redz = kale.utils.midpoints(redz, axis=0) # get final redz at bin centers
        redz = np.moveaxis(redz, 0, dd)
    sel = (redz > 0.0) # identify emitting sources
    redz[~sel] = -1.0 # set all other redshifts to zero
    redz[redz<0] = -1.0

    # get bin shape
    nmtot = len(mtot)
    nmrat = len(mrat)
    nredz = redz.shape[2]
    nfobs = len(fobs)
    # if np.any([number_shape[0] != nmtot, number_shape[1]!=nmrat, number_shape[2]!=nredz, number_shape[3]!=nfobs]):
    #     err = f"Parameter bin shape [{nmtot=}, {nmrat=}, {nredz=}, {nfobs=}] does not match {number_shape=}."
    #     raise ValueError(err)
    print(f"Parameter bin shape [{nmtot=}, {nmrat=}, {nredz=}, {nfobs=}] should match {number_shape=}.")

    # Reshape arrays to [M,Q,Z,F]
    mtot = np.repeat(mtot, nmrat*nredz*nfobs).reshape(nmtot, nmrat, nredz, nfobs)

    mrat = np.repeat(mrat, nmtot*nredz*nfobs).reshape(nmrat, nmtot, nredz, nfobs) # Q,M,Z,F
    mrat = np.swapaxes(mrat, 0, 1) # M,Q,Z,F

    fobs = np.repeat(fobs, nmrat*nredz*nmtot).reshape(nfobs, nmrat, nredz, nmtot) # F,Q,Z,M
    fobs = np.swapaxes(fobs, 0, 3) # M,Q,Z,F

    # check shapes again
    if np.any([mtot.shape != number_shape, 
                mrat.shape != number_shape,
                redz.shape != number_shape,
                fobs.shape != number_shape]):
        err = f"Sample shapes don't all match number! {mtot.shape=}, {mrat.shape=}, {redz.shape=}, {fobs.shape=}"
        raise ValueError(err)

    if flatten:
        samples = [mtot.flatten(), mrat.flatten(), redz.flatten(), fobs.flatten()]
    else:
        samples = [mtot, mrat, redz, fobs]

    return samples


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