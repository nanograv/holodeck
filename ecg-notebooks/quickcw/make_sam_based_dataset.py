import os
import glob
import shutil
import json

import numpy as np
import matplotlib.pyplot as plt

import libstempo.toasim as LT
import libstempo.plot as LP

import libstempo_add_catalog_of_cws as LT_catalog

from holodeck.constants import MSOL, PC, YR, MPC, GYR, SPLC
from holodeck import log, cosmo, utils, plot
from holodeck import extensions as holo_extensions


####################################################################################
#
# Input parameters
#
####################################################################################
#number of realizations to produce
N_real = 100

#path to directory where par and tim files will be saved
#savedir = "../Test_datasets_12p5yr_based_2real/"
#savedir = "../Test_datasets_15yr_based_2real/"
#savedir = "../Test_datasets_15yr_based_100real/"
#savedir = "../Test_datasets_15yr_based_1real_debug_wn_only/"
#savedir = "../Test_datasets_15yr_based_1real_v2/"
#savedir = "../Test_datasets_15yr_based_100real_v2/"
#savedir = "../Test_datasets_15yr_based_100real_v3/"
#savedir = "../Test_datasets_15yr_based_100real_v4/"
savedir = "../Test_datasets_15yr_based_100real_v10/"

#path to par files used for the dataset
#parpath = '../12p5yr_stripped_pars/'
parpath = '../stripped_pars_15yr_v1p1/'

#path to json file with pulsar summary data made a la Atro4Cast
#summary_data_json = '../psr_sumdata_v2.json' #'../psr_sumdata_12p5yr_based_15yr.json'
summary_data_json = '../psr_sumdata_15yr_v1p1.json'

#path to json with RN values
#rn_json = '../channelized_12p5yr_v3_full_noisedict.json'
rn_json = '../v1p1_all_dict.json'

#observational timespan and minimum time resolution
#used to set lower and upper boundary on frequency for simulating binaries
#TODO: should just get T_obs from the summary data
T_obs = 16.03*YR
# T_min = 1/24.0*YR

# Choose binary population parameters
RESAMP = 2.0       # resample initial population for smoother statistics
TIME = 3.0 * GYR   # lifetime of systems between formation and coalescence
DENS = 2.0         # change the density of binaries in the universe by this factor
MAMP = 1e9 * MSOL  # normalization of the MBH-Galaxy mass relation

####################################################################################
#
# Setup
#
####################################################################################
#make directory to save datasets to
os.mkdir(savedir)

#get list of par files
parfiles = sorted(glob.glob(parpath + '/*.par'))
#reduce number of psrs for testing
#parfiles = parfiles[:5]
print(len(parfiles))
print(parfiles)

#copy parfiles to output directory to have them at the same place
for p in parfiles:
    shutil.copy(p, savedir+p.split('/')[-1])

#open dictionary with pulsar summary data
with open(summary_data_json) as fp:
    psr_sumdata = json.load(fp)

#make fake pulsars based on parfiles and summary data
psrs = []
for i, p in enumerate(parfiles):
    psrname = p.split('/')[-1].split('.')[0]
    print(psrname)
    
    t = np.array(psr_sumdata[psrname]['toas']) / 86400.0
    toaerrs = np.array(psr_sumdata[psrname]['toaerrs']) / 1e-6
    
    fake_pulsar = LT.fakepulsar(p,obstimes=t,toaerr=toaerrs)#, flags='-pta PPTA')
    psrs.append(fake_pulsar)


####################################################################################
#
# Generate SAM-based population
#
####################################################################################
#set up frequency bin centers
F_bin_centers, F_bins = utils.pta_freqs()

n_bins = F_bin_centers.size
# print(f"{n_bins=}")

# #get bin edges from bin centers
# F_bins = [F_bin_centers[0]*2/3,]
# for iii in range(F_bin_centers.size-1):
#     F_bins.append((F_bin_centers[iii+1]+F_bin_centers[iii])/2)

# F_bins.append(F_bins[-1]+F_bin_centers[-1]-F_bin_centers[-2])

# F_bins = np.array(F_bins)

#set up realizer object used to create realizations of a binary population (need to use orbital frequency instead of GW)
params = {'hard_time': 2.3580737294474514, 
          'gsmf_phi0': -2.012540540307903, 
          'gsmf_mchar0_log10': 11.358074707612774, 
          'mmb_mamp_log10': 8.87144417474846, 
          'mmb_scatter_dex': 0.027976545572248435, 
          'hard_gamma_inner': -0.38268820924239666}
nn, samples = holo_extensions.realizer(params, nreals=1. nloudest=1000)
samples = samples.squeeze() # just using 1 realization


####################################################################################
#
# Do GWB + outlier injections over multiple realizations of the population and noise
#
####################################################################################
for i in range(N_real):
    print(i)

    #sample binary parameters from population
    # nn, samples = realizer()
    #nn, samples = realizer(down_sample=50) #optional downsampling for quick testing
    print(samples.shape)

    units = [1.99e+33, 1, 3.17e-08]

    # TODO figure out if these should be in log space or not!!!
    #Mtots = samples[0,:]/units[0] #solar mass
    Mtots = samples[0,:] #cgs
    Mrs = samples[1,:]
    MCHs = utils.chirp_mass(*utils.m1m2_from_mtmr(Mtots, Mrs)) #cgs

    REDZs = samples[2,:]/units[1]

    FOs = samples[3,:] #Hz

    print(MCHs)
    print(REDZs)
    print(FOs)

    #make weights array with ones (included so we can support non-unit weights)
    weights = np.ones(FOs.size)

    #make vals array containing chirp mass, mass ratio, redshift, and observer frame GW frequency for each binary
    vals = np.array([Mtots, Mrs, REDZs, FOs])
    
    #reset psr objects so they have zero residuals
    for psr in psrs:
        for I in range(3):
            psr.stoas[:] -= psr.residuals() / 86400.0
            psr.formbats()

    #Add WN (only efac needed for epoch-averaged TOAs)
    for psr in psrs:
        #print("WN")
        #print(psr.name)
        LT.add_efac(psr, efac=1.00, seed=1_000_000+i)
        #psr.fit()

    #Add per-psr RN
    with open(rn_json, 'r') as f:
        noisedict = json.load(f)
    for psr in psrs:
        #print("PSR RN")
        #print(psr.name)
        A = 10**noisedict[psr.name+"_red_noise_log10_A"]
        gamma = noisedict[psr.name+"_red_noise_gamma"]
        LT.add_rednoise(psr, A, gamma, components=30, seed=1848_1919+i)
        #psr.fit()

    #Add population of BBHs
    inj_return = LT_catalog.add_gwb_plus_outlier_cws(psrs,
                                                     vals, weights, F_bins, T_obs,
                                                     outlier_per_bin=1_000, seed=1994+i)
    f_centers, free_spec, outlier_fo, outlier_hs, outlier_mc, outlier_dl, random_gwthetas, random_gwphis, random_phases, random_psis, random_incs = inj_return
    #save simulated dataset to tim files
    real_dir = savedir+"real{0:03d}/".format(i)
    os.mkdir(real_dir)
    
    np.savez(real_dir+"simulation_info.npz", free_spec=free_spec, f_centers=f_centers, F_bins=F_bins,
                                             outlier_fo=outlier_fo, outlier_hs=outlier_hs,
                                             outlier_mc=outlier_mc, outlier_dl=outlier_dl,
                                             random_gwthetas=random_gwthetas, random_gwphis=random_gwphis,
                                             random_phases=random_phases, random_psis=random_psis, random_incs=random_incs)

    for j in range(len(psrs)):
        #print("CWs")
        print(psrs[j].name)
        #no need to fit here since we fit after adding each signal
        psrs[j].fit()
        psrs[j].savetim(real_dir + 'fake_{0}.tim'.format(psrs[j].name))
        #psrs[j].savepar(real_dir + 'fake_{0}.par'.format(psrs[j].name))

    #for iii, psr in enumerate(psrs):
    #    plt.figure(iii)
    #    LP.plotres(psr)
    #    plt.savefig(psr.name+"_injection_debug.png")

