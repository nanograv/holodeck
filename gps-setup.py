#!/usr/bin/env python3
# coding: utf-8

# # Overview
# This notebook will walk you through how to set-up a GP from any given bank of spectra. 
# 
# The GPs come from the python package `george` and we "train" them using the package `emcee`. 
# 
# Once the GP is trained, we export it as a pickle object to then use with PTA data.

from __future__ import division

from multiprocessing import Pool, cpu_count
import os

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import h5py,time
import scipy.signal as ssig

from holodeck.constants import YR
import george
import george.kernels as kernels
import emcee, corner, pickle

from pathlib import Path


# # Load Spectra
# 
#     The first step is to load the bank of spectra. 
#     Make sure to double check that dimensionality of the parameter space, and get the parameter limits.


# Start with Spectra from Luke
spectra_file = Path('./spec_libraries/hard04b_n1000_g100_s40_r50_f40/sam-lib_hard04b_2023-01-23_01_n1000_g100_s40_r50_f40.hdf5')
spectra = h5py.File(spectra_file, 'r')

# ## Compute the mean and std from all spectra realizations
#     At each point in parameter space, we need to find the mean value and the
#     standard deviation from all of the realizations that we have.


## NOTE - Only need to train GP on number of frequencies in PTA analysis !
gwb_spectra = spectra['gwb'][:,:30,:]**2

# Find all of the zeros and set them to be h_c = 1e-20
low_ind = np.where(gwb_spectra < 1e-40)
gwb_spectra[low_ind] = 1e-40


# Find mean over 100 realizations
mean = np.log10(np.mean(gwb_spectra, axis=-1))

# Smooth Mean Spectra
## NOTE FOR LUKE - HOW MUCH SMOOTHING DO WE WANT TO DO ?
smooth_mean = ssig.savgol_filter(mean, 7, 3)

# Find std
err = np.std(np.log10(gwb_spectra), axis=-1)

if np.any(np.isnan(err)):
    print('Got a NAN issue')

# # Train GP
# 
#     The next step is to set up the GP class.
#     Things to note:
#         - need to make sure that the GP has the same dimensionality as the parameter space from the spectra.
#         - the GPs work better when they are trained on zero-mean data, so it's very important that we
#         remove the mean values for the spectra at each frequency, BUT these values HAVE TO BE SAVED, because
#         they are required to extract meaningful information back out of the GP once it is trained!


# Define a GP class containing the kernel parameter priors and a log-likelihood

class gaussproc(object):
    
    def __init__(self, x, y, yerr=None, par_dict = None):

        self.x = x
        self.y = y
        self.yerr = yerr
        self.par_dict = par_dict
        
        # The number of GP parameters is one more than the number of spectra parameters.
        self.pmax = np.array([20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0]) # sampling ranges
        self.pmin = np.array([-20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0])
        self.emcee_flatchain = None
        self.emcee_flatlnprob = None
        self.emcee_kernel_map = None
    
    def lnprior(self, p):
    
        logp = 0.
    
        if np.all(p <= self.pmax) and np.all(p >= self.pmin):
            logp = np.sum(np.log(1/(self.pmax-self.pmin)))
        else:
            logp = -np.inf

        return logp

    def lnlike(self, p):

        # Update the kernel and compute the lnlikelihood.
        a, tau = np.exp(p[0]), np.exp(p[1:])
        
        lnlike = 0.0
        try:
            gp = george.GP(a * kernels.ExpSquaredKernel(tau,ndim=len(tau)))
            #gp = george.GP(a * kernels.Matern32Kernel(tau))
            gp.compute(self.x , self.yerr)
            
            lnlike = gp.lnlikelihood(self.y, quiet=True)
        except np.linalg.LinAlgError:
            lnlike = -np.inf
        
        return lnlike
    
    def lnprob(self, p):
        
        return self.lnprior(p) + self.lnlike(p)


## Load in the spectra data!

# The "y" data are the means and errors for the spectra at each point in parameter space
yobs = smooth_mean.copy() #mean.copy()
yerr = err.copy()
GP_freqs = spectra['fobs'][:30].copy()
GP_freqs *= YR

## Find mean in each frequency bin (remove it before analyzing with the GP) ##
# This allows the GPs to oscillate around zero, where they are better behaved.
yobs_mean = np.mean(yobs,axis=0)
# MAKE SURE TO SAVE THESE VALUES - THE GP IS USELESS WITHOUT THEM !
np.save('./Luke_Spectra_MEANS.npy', yobs_mean)

yobs -= yobs_mean[None,:]


# ### Note on saving the means
# I think that this .npy file is not needed, the means are saved as an attribute
# of the `gaussproc` objects in the `gp_george` list



pars = spectra['parameters'].attrs['ordered_parameters']

## The "x" data are the actual parameter values
xobs = np.zeros((spectra['gwb'].shape[0], len(pars)))

# [gsmf_phi0, hard_gamma_inner, hard_gamma_outer, hard_rchar, hard_time, mmb_amp]
for ii in range((spectra['gwb'].shape[0])):
    for k, par in enumerate(pars):
        xobs[ii,k] = spectra['sample_params'][ii,k]
        
# Put mmb_amp in logspace
xobs[:, -1] = np.log10(xobs[:, -1])



# Instanciate a list of GP kernels and models [one for each frequency]

gp_george = []
k = []

# Create the parameter dictionary for the gp objects
par_dict = dict()
for ind, par in enumerate(pars):
    par_dict[par] = {"min": np.min(xobs[:, ind]),
                     "max": np.max(xobs[:, ind])}

for freq_ind in range(len(GP_freqs)):
    
    gp_george.append(gaussproc(xobs,yobs[:,freq_ind],yerr[:,freq_ind], par_dict))
    k.append( 1.0 * kernels.ExpSquaredKernel([2.0,2.0,2.0,2.0,2.0,2.0],ndim=6) )
    num_kpars = len(k[freq_ind])



# Sample the posterior distribution of the kernel parameters 
# to find MAP value for each frequency. 

# THIS WILL TAKE A WHILE... (~ 5 min per frequency on 18 cores)

sampler = [0.0]*len(GP_freqs)
nwalkers, ndim = 36, num_kpars
for freq_ind in range(len(GP_freqs)):
    # Parellize emcee with nwalkers //2 or the maximum number of processors available, whichever is smaller
    with Pool(min(nwalkers // 2, cpu_count()) ) as pool:
        t_start = time.time()

        # Set up the sampler.
        sampler[freq_ind] = emcee.EnsembleSampler(nwalkers, ndim, gp_george[freq_ind].lnprob, pool=pool)

        # Initialize the walkers.
        p0 = [np.log([1.,1.,1.,1.,1.,1., 1.]) + 1e-4 * np.random.randn(ndim)
              for i in range(nwalkers)]

        print(freq_ind, "Running burn-in")
        p0, lnp, _ = sampler[freq_ind].run_mcmc(p0, int(750))
        sampler[freq_ind].reset()

        print(freq_ind, "Running second burn-in")
        p = p0[np.argmax(lnp)]
        p0 = [p + 1e-8 * np.random.randn(ndim) for i in range(nwalkers)]
        p0, _, _ = sampler[freq_ind].run_mcmc(p0, int(750))
        sampler[freq_ind].reset()

        print(freq_ind, "Running production")
        p0, _, _ = sampler[freq_ind].run_mcmc(p0, int(1500))

        print('Completed in {} min'.format((time.time()-t_start)/60.) , '\n')



# ## Save training information

## Populate the GP class with the details of the kernel
## MAP values for each frequency.

for ii in range(len(GP_freqs)):
    
    gp_george[ii].chain = None 
    gp_george[ii].lnprob = None 
    
    gp_george[ii].kernel_map = sampler[ii].flatchain[np.argmax(sampler[ii].flatlnprobability)] 

    # add-in mean yobs (freq) values
    gp_george[ii].mean_spectra = yobs_mean[ii]


## Save the trained GP as a pickle to be used with PTA data!
gp_file = "trained_gp_" + spectra_file.stem + ".pkl"
with open(gp_file, "wb") as gpf:
    pickle.dump(gp_george, gpf)
