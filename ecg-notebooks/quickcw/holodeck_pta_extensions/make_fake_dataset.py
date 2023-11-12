import os
import glob
import shutil
import json

import numpy as np
from datetime import datetime
from tqdm import tqdm
import argparse

import libstempo.toasim as LT
import libstempo.plot as LP

import libstempo_add_catalog_of_cws as LT_catalog

from holodeck.constants import MSOL, PC, YR, MPC, GYR, SPLC
from holodeck import utils
from holodeck import extensions as holo_extensions
import holodeck as holo


####################################################################################
# Hardcoded Paths
####################################################################################
def get_paths(): # UPDATE ACCORDINGLY
    #path to par files used for the dataset
    parpath = '/Users/emigardiner/GWs/holodeck/output/holodeck_extension_15yr_stuff/stripped_pars_15yr_v1p1/'

    #path to json file with pulsar summary data made a la Astro4Cast
    summary_data_json = '/Users/emigardiner/GWs/holodeck/output/holodeck_extension_15yr_stuff/psr_sumdata_15yr_v1p1.json'

    #path to json with RN values
    rn_json = '/Users/emigardiner/GWs/holodeck/output/holodeck_extension_15yr_stuff/v1p1_all_dict.json'

    #path to save folder, to be appended with save_dir name where par and tim files will be saved
    save_dir = "/Users/emigardiner/GWs/holodeck/output/holodeck_extension_15yr_stuff/"

    path_dict = dict(parpath=parpath, summary_data_json=summary_data_json, rn_json=rn_json, save_dir=save_dir)
    return path_dict



####################################################################################
# Default Arguments
####################################################################################
DEF_NREALS = 3
DEF_NPSRS = None
DEF_DEBUG = True
VERSION = 1

### SAM data
DEF_SHAPE = None

####################################################################################
# Set up argparser
####################################################################################
def _setup_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--illustris', action='store_true', dest='illustris_flag', default=False,
                        help='whether to use Illustris populations instead of SAM')
    parser.add_argument('-r', '--nreals', action='store', dest='nreals', type=int, default=DEF_NREALS, 
                        help='number of realizations')
    
    parser.add_argument('-p', '--npsrs', action='store', dest='npsrs', type=int, default=DEF_NPSRS, 
                        help='number of pulsars to use if testing, else None')
    parser.add_argument('--debug', action='store_true', dest='debug', default=DEF_DEBUG,
                        help='whether to print steps')
    
    parser.add_argument('--pkl', '--tim_to_pkl', action='store_true', dest='pkl_flag', default=False,
                        help='whether to convert psr data to a pickle file')
    
    ### SAM data
    parser.add_argument('--shape', action='store', dest='shape', type=int, default=DEF_SHAPE,
                        help='sam shape')
    
    args = parser.parse_args()
    return args

####################################################################################
# Main
####################################################################################
def main():

    ### Time
    start = datetime.now()
    date = start.strftime("%Y_%m_%d")
    print(f'----- Starting at {start} -----\n{date}')

    ### Set up args
    args = _setup_argparse()
    ILLUSTRIS_FLAG = args.illustris_flag # whether to use Illustris or SAM for initial population
    N_REAL = args.nreals      # number of realizations to produce
    N_PSRS = args.npsrs      # use fewer pulsars for testing, otherwise set to None
    debug = args.debug     # whether to print steps
    SHAPE = args.shape
    print(f"{ILLUSTRIS_FLAG=}, {N_PSRS=}, {N_REAL=}, {N_PSRS=}, {SHAPE=}")

    ### Set up directory paths
    path_dict = get_paths()
    parpath = path_dict['parpath']
    summary_data_json = path_dict['summary_data_json'] 
    rn_json = path_dict['rn_json']               
    save_dir = path_dict['save_dir']

    ### Set up directory where par and tim files will be saved
    save_flag = 'pop' if ILLUSTRIS_FLAG else 'sam'
    save_dir += f"{date}_{save_flag}_datasets_15yr_based_r{N_REAL:03d}"
    if N_PSRS is not None:
        save_dir += f"_p{N_PSRS:02d}"
    save_dir += f"_v{VERSION:02d}/"

    ####################################################################################
    #
    # Input parameters
    #
    ####################################################################################

    #observational timespan and minimum time resolution
    #used to set lower and upper boundary on frequency for simulating binaries
    #TODO: should just get T_obs from the summary data
    T_obs = 16.03*YR
    # T_min = 1/24.0*YR

    # Choose binary population parameters

    # --- if using Illustris you'll need:
    RESAMP = 2.0       # resample initial population for smoother statistics
    TIME = 3.0 * GYR   # lifetime of systems between formation and coalescence
    DENS = 2.0         # change the density of binaries in the universe by this factor
    MAMP = 1e9 * MSOL  # normalization of the MBH-Galaxy mass relation

    # --- if using SAM you'll need:
    # TODO add Pspace and params to argparse arguments
    PARAMS = {'hard_time': 2.3580737294474514, 
            'gsmf_phi0': -2.012540540307903, 
            'gsmf_mchar0_log10': 11.358074707612774, 
            'mmb_mamp_log10': 8.87144417474846, 
            'mmb_scatter_dex': 0.027976545572248435, 
            'hard_gamma_inner': -0.38268820924239666} 
    PSPACE = holo.param_spaces.PS_Uniform_09B(holo.log, nsamples=1, sam_shape=SHAPE, seed=None)


    ####################################################################################
    #
    # Setup
    #
    ####################################################################################
    #make directory to save datasets to
    if os.path.exists(save_dir) is False:
        os.mkdir(save_dir)
    else:
        print(f"Overwriting {save_dir=}")

    #get list of par files
    parfiles = sorted(glob.glob(parpath + '/*.par'))
    #reduce number of psrs for testing
    if N_PSRS is not None:
        parfiles = parfiles[:N_PSRS]
    print(f"{len(parfiles)=}")
    print(f"{parfiles=}")

    #copy parfiles to output directory to have them at the same place
    for p in parfiles:
        shutil.copy(p, save_dir+p.split('/')[-1])

    #open dictionary with pulsar summary data
    with open(summary_data_json) as fp:
        psr_sumdata = json.load(fp)

    #make fake pulsars based on parfiles and summary data
    psrs = []
    for i, p in enumerate(parfiles):
        psrname = p.split('/')[-1].split('.')[0]
        print(f"{psrname=}")
        
        t = np.array(psr_sumdata[psrname]['toas']) / 86400.0
        toaerrs = np.array(psr_sumdata[psrname]['toaerrs']) / 1e-6

        fake_pulsar = LT.fakepulsar(p,obstimes=t,toaerr=toaerrs)#, flags='-pta PPTA')
        psrs.append(fake_pulsar)



    ####################################################################################
    #
    # Generate Illustris-based or SAM-based population
    #
    ####################################################################################
    #set up frequency bin centers
    F_bin_centers, F_bins_edges = utils.pta_freqs()

    n_bins = F_bin_centers.size
    print(f"{n_bins=}")

    # set up realizer object used to create realizations of a binary population for a specific model
    # (need to use orbital frequency instead of GW)
    if ILLUSTRIS_FLAG:
        realizer = holo_extensions.Realizer(F_bin_centers/2, resample=RESAMP, lifetime=TIME, mamp=MAMP, dens=DENS)
    else:
        realizer = holo_extensions.Realizer_SAM(
            fobs_orb_edges=F_bins_edges/2.0, params=PARAMS, pspace=PSPACE)



    ####################################################################################
    #
    # Do GWB + outlier injections over multiple realizations of the population and noise
    #
    ####################################################################################

    if ILLUSTRIS_FLAG is False:
        print(f'-- runtime: {datetime.now()-start}')
        print('realizing samples')
        names, real_samples, real_weights = realizer(nreals=N_REAL, clean=True)
        print(f'-- runtime: {datetime.now()-start}')

    for rr in tqdm(range(N_REAL)):
        print(f"--- Realization: {rr}/{N_REAL}")
        print(f'-- runtime: {datetime.now()-start}')

        #sample binary parameters from population
        if ILLUSTRIS_FLAG:
            # nn, samples = realizer()
            nn, samples = realizer(down_sample=50) #optional downsampling for quick testing
            print(samples.shape)
        else:
            samples = real_samples[rr]
            print(f"{len(samples[0])=}")

        units = [1.99e+33, 1, 3.17e-08]


        if ILLUSTRIS_FLAG:
            #Mtots = samples[0,:]/units[0] #solar mass
            Mtots = 10**samples[0,:] #cgs
            Mrats = 10**samples[1,:]
            # MCHs = utils.chirp_mass(*utils.m1m2_from_mtmr(Mtots, Mrs)) #cgs
            REDZs = 10**samples[2,:]/units[1]
            FOs = 10**samples[3,:] #Hz

            #make weights array with ones (included so we can support non-unit weights)
            weights = np.ones(FOs.size)
        else:
            Mtots = samples[0] #cgs
            Mrats = samples[1]
            REDZs = samples[2]/units[1] # dimensionless
            FOs = samples[3]  #Hz
            
            weights = real_weights[rr] # np.ones(FOs.size)

        # print(f"MCHs: {MCHs.shape=}, {utils.stats(MCHs)}")
        print(f"REDZs: {REDZs.shape=}, {utils.stats(REDZs)}")
        print(f"FOs: {FOs.shape=}, {utils.stats(FOs)}")


        #make vals array containing total mass, mass ratio, redshift, and observer frame GW frequency for each binary
        vals = np.array([Mtots, Mrats, REDZs, FOs]) # should this not be Mchirps?
        print(f"{vals.shape=}")
        
        #reset psr objects so they have zero residuals
        print('Resetting pulsars')
        for psr in psrs:
            for I in range(3):
                psr.stoas[:] -= psr.residuals() / 86400.0
                psr.formbats()

        #Add WN (only efac needed for epoch-averaged TOAs)
        print('Adding white noise')
        for psr in psrs:
            #print("WN")
            #print(psr.name)
            LT.add_efac(psr, efac=1.00, seed=1_000_000+rr)
            #psr.fit()

        #Add per-psr RN
        print('Adding per-pulsar red noise')
        with open(rn_json, 'r') as f:
            noisedict = json.load(f)
        for psr in psrs:
            #print("PSR RN")
            #print(psr.name)
            A = 10**noisedict[psr.name+"_red_noise_log10_A"]
            gamma = noisedict[psr.name+"_red_noise_gamma"]
            LT.add_rednoise(psr, A, gamma, components=30, seed=1848_1919+rr)
            #psr.fit()

        #Add population of BBHs
        print('Adding population of BBHs')
        inj_return = LT_catalog.add_gwb_plus_outlier_cws(
            psrs, vals, weights, F_bins_edges, T_obs,
            outlier_per_bin=1_000, seed=1994+rr)
        f_centers, free_spec, outlier_fo, outlier_hs, outlier_mc, outlier_dl, random_gwthetas, random_gwphis, random_phases, random_psis, random_incs = inj_return
        #save simulated dataset to tim files
        real_dir = save_dir+"real{0:03d}/".format(rr)
        os.mkdir(real_dir)
        
        print('Saving simulation info')
        np.savez(real_dir+"simulation_info.npz", free_spec=free_spec, f_centers=f_centers, F_bins=F_bins_edges,
                                                outlier_fo=outlier_fo, outlier_hs=outlier_hs,
                                                outlier_mc=outlier_mc, outlier_dl=outlier_dl,
                                                random_gwthetas=random_gwthetas, random_gwphis=random_gwphis,
                                                random_phases=random_phases, random_psis=random_psis, random_incs=random_incs)


        print('Saving fake pulsar timfiles')
        for j in range(len(psrs)):
            # print("CWs")
            # print(psrs[j].name)
            #no need to fit here since we fit after adding each signal
            psrs[j].fit()
            psrs[j].savetim(real_dir + 'fake_{0}.tim'.format(psrs[j].name))
            #psrs[j].savepar(real_dir + 'fake_{0}.par'.format(psrs[j].name))

        #for iii, psr in enumerate(psrs):
        #    plt.figure(iii)
        #    LP.plotres(psr)
        #    plt.savefig(psr.name+"_injection_debug.png")

    end = datetime.now()
    print(f'----- Started at {start} -----')
    print(f'----- Ended fake data production at {end} -----')
    print(f'-- runtime: {end-start}')

    ####################################################################################
    #
    # Pickle tim files
    #
    ####################################################################################
    if args.pkl_flag:
        print(f'Converting tim files to pkl file')
        import pickle
        from enterprise.pulsar import Pulsar
        for rr in range(N_REAL):
            real_dir = save_dir+"real{0:03d}/".format(rr)
            #get list of par files
            timfiles = sorted(glob.glob(real_dir + '/*.tim'))
            parfiles = sorted(glob.glob(save_dir + '/*.par'))
            if N_PSRS is not None:
                timfiles = timfiles[:N_PSRS]
                parfiles = parfiles[:N_PSRS]
            assert len(timfiles)==len(parfiles), "mismatch between number of par and tim files!"
            print(f"{timfiles=}\n{parfiles=}")

            # load each psr
            psrs = []
            for par, tim in zip(parfiles, timfiles):
                print(f"{tim=}")
                psr = Pulsar(par, tim)
                psrs.append(psr)

            # pickle and save
            pkl_file = real_dir + f'data_15yr_fake_r{RR:03d}_of_{N_REAL:03d}_v01.pkl'
            print(pkl_file)
            with open(pkl_file, 'wb') as file: 
                # A new file will be created 
                pickle.dump(psrs, file) 


        end = datetime.now()
        print(f'----- Started at {start} -----')
        print(f'----- Ended file pickling at {end} -----')
        print(f'-- runtime: {end-start}')

if __name__ == "__main__":
    main()