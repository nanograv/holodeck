"""Compare multiple discrete MBH Binary Populations (from cosmological hydrodynamic simulations)."""

import os
import h5py
import numpy as np
import holodeck as holo
from holodeck import utils, log, _PATH_DATA, cosmo, discrete
from holodeck.constants import PC, MSOL, YR, MPC, GYR, SPLC
import matplotlib.pyplot as plt

class Discrete:
    
    def __init__(self, freqs, freqs_edges, attrs=(None,None,'k',1.0), lbl=None, fixed_sepa=None, 
                 tau=1.0*YR, nreals=500, mod_mmbulge=False, rescale_mbulge=False, allow_mbh0=False, 
                 skip_evo=False, use_mstar_tot_as_mbulge=False, nloudest=10):

        self.attrs = attrs
        self.freqs = freqs
        self.freqs_edges = freqs_edges
        self.lbl = lbl
        self.fname = self.attrs[0]
        self.basepath = self.attrs[1]
        self.color = self.attrs[2]
        self.lw = self.attrs[3]
        self.fixed_sepa = fixed_sepa
        self.tau = tau
        self.nreals = nreals
        self.mod_mmbulge = mod_mmbulge
        self.allow_mbh0 = allow_mbh0
        self.use_mstar_tot_as_mbulge = use_mstar_tot_as_mbulge
        self.nloudest = nloudest
        
        print(f"\nCreating Discrete_Pop class instance '{self.lbl}' with tau={self.tau}, fixed_sepa={self.fixed_sepa}")
        print(f" fname={self.fname}")
        self.pop = discrete.population.Pop_Illustris(fname=self.fname, basepath=self.basepath, 
                                                     fixed_sepa=self.fixed_sepa, allow_mbh0=self.allow_mbh0,
                                                     use_mstar_tot_as_mbulge=self.use_mstar_tot_as_mbulge)
        print(f"{self.pop.sepa.min()=}, {self.pop.sepa.max()=}, {self.pop.sepa.shape=}")
        print(f"{self.pop.mstar_tot.min()=}, {self.pop.mstar_tot.max()=}, {self.pop.mstar_tot.shape=}")

        # apply modifiers if requested
        if self.mod_mmbulge == True:
            print(f"before mass mod: {self.pop.mass.min()=}, {self.pop.mass.max()=}, {self.pop.mass.shape=}")
            print(f"before mass mod: {self.pop.mbulge.min()=}, {self.pop.mbulge.max()=}, {self.pop.mbulge.shape=}")
            old_mass = self.pop.mass
            old_mbulge = self.pop.mbulge
            old_mrat = self.pop.mass[:,1]/self.pop.mass[:,0]
            old_mrat[old_mrat>1] = 1/old_mrat[old_mrat>1]
            
            print(f"before mass mod: mass ratio m2/m1: {old_mrat.min()=}, {old_mrat.max()=}, {old_mrat.shape=}")
            ## self.mmbulge = holo.relations.MMBulge_KH2013() # deprecated
            self.mmbulge = holo.host_relations.MMBulge_KH2013()
            self.mod_KH2013 = discrete.population.PM_Mass_Reset(self.mmbulge, scatter=True, 
                                                                rescale_mbulge=rescale_mbulge)
            self.pop.modify(self.mod_KH2013)

            # ---- Added for debugging change in mass ratios 5/15/25 - LB ----#
            new_mrat = self.pop.mass[:,1]/self.pop.mass[:,0]
            new_mrat[new_mrat>1] = 1/new_mrat[new_mrat>1]

            print(f"after mass mod: {self.pop.mass.min()=}, {self.pop.mass.max()=}, {self.pop.mass.shape=}")
            print(f"after mass mod: {self.pop.mbulge.min()=}, {self.pop.mbulge.max()=}, {self.pop.mbulge.shape=}")
            print(f"after mass mod: mass ratio m2/m1: {new_mrat.min()=}, {new_mrat.max()=}, {new_mrat.shape=}")

            mrat_increase_factor = new_mrat / old_mrat
            mass_increase_factor = self.pop.mass / old_mass
            mbulge_increase_factor = self.pop.mbulge / old_mbulge
            test_old_mass_fac = self.pop._mass / old_mass
            print(f"after mass mod: {mrat_increase_factor.min()=}, {mrat_increase_factor.max()=}, {np.median(mrat_increase_factor)=}")
            print(f"after mass mod: {mass_increase_factor.min()=}, {mass_increase_factor.max()=}, {np.median(mass_increase_factor)=}")
            print(f"after mass mod: {mbulge_increase_factor.min()=}, {mbulge_increase_factor.max()=}, {np.median(mbulge_increase_factor)=}")
            print(f"after mass mod: {test_old_mass_fac.min()=}, {test_old_mass_fac.max()=}, {np.median(test_old_mass_fac)=}")

            ix_low_mrat = np.where(mrat_increase_factor<0.25)[0]
            print(f"{ix_low_mrat.size=}")
            #print(f"{mrat_increase_factor[ix_low_mrat]}")
            #for i in range(ix_low_mrat.size):
            #    print(f"mrat inc fac: {mrat_increase_factor[ix_low_mrat[i]]:.3g}, " 
            #          f"mbh old: {old_mass[ix_low_mrat[i],0]/MSOL:.3g}, {old_mass[ix_low_mrat[i],1]/MSOL:.3g}, ", 
            #          f"mbh new: {self.pop.mass[ix_low_mrat[i],0]/MSOL:.3g}, {self.pop.mass[ix_low_mrat[i],1]/MSOL:.3g}, ", 
            #          f"mbulge: {self.pop.mbulge[ix_low_mrat[i],0]/MSOL:.3g}, {self.pop.mbulge[ix_low_mrat[i],1]/MSOL:.3g}")

            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('q')
            plt.ylabel('Mtot [Msun]')
            old_mtot = old_mass[:,0] + old_mass[:,1]
            new_mtot = self.pop.mass[:,0] + self.pop.mass[:,1]
            himass_mrat_increased = 0
            himass_mrat_decreased = 0
            mrat_increased = 0
            mrat_decreased = 0
            himass_count = 0
            for i in range(old_mrat.size):
                if new_mrat[i]>old_mrat[i]:
                    mrat_increased += 1
                else:
                    mrat_decreased += 1
                    
                if np.max([old_mass[i,0],old_mass[i,1],self.pop.mass[i,0],self.pop.mass[i,1]])>1e8*MSOL:
                    himass_count += 1
                    if new_mrat[i]>old_mrat[i]:
                        col='r'
                        himass_mrat_increased += 1 
                    else: 
                        col='k'
                        himass_mrat_decreased += 1
                    #plt.plot([old_mass[i,0]/MSOL,self.pop.mass[i,0]/MSOL], [old_mass[i,1]/MSOL,self.pop.mass[i,1]/MSOL],alpha=0.2)
                    plt.plot([old_mrat[i],new_mrat[i]], [old_mtot[i]/MSOL,new_mtot[i]/MSOL],alpha=0.3, lw=0.5, color=col)
                    plt.plot([new_mrat[i]], [new_mtot[i]/MSOL],alpha=0.3, marker='.', ms=2, color=col)
            print(f"{old_mrat.size=}, {mrat_increased=}, {mrat_decreased=}")
            print(f"{himass_count=}, {himass_mrat_increased=}, {himass_mrat_decreased=}")
            plt.show()
            # ---------------------------------------------------------------------------- #

            #print(f"{self.pop.sepa.min()=}, {self.pop.sepa.max()=}, {self.pop.sepa.shape=}")

        if skip_evo == False:
            # create a fixed-total-time hardening mechanism
            print(f"modeling fixed-total-time hardening...")
            self.fixed = holo.hardening.Fixed_Time_2PL.from_pop(self.pop, self.tau)
            print(f"{self.pop.sepa.min()=}, {self.pop.sepa.max()=}, {self.pop.sepa.shape=}")

            # Create an evolution instance using population and hardening mechanism
            print(f"creating evolution instance and evolving it...")
            self.evo = discrete.evolution.Evolution(self.pop, self.fixed)
            print(f"{self.evo._sample_volume=}")
            print(f"{self.pop.sepa.min()=}, {self.pop.sepa.max()=}, {self.pop.sepa.shape=}")
            print(f"{self.pop.scafa.min()=}, {self.pop.scafa.max()=}, {self.pop.scafa.shape=}")
            print(f"# with scafa=1 in pop: {self.pop.scafa[self.pop.scafa==1.0].size}")
            print(f"{self.evo.sepa.min()=}, {self.evo.sepa.max()=}, {self.evo.sepa.shape=}")
            print(f"{self.evo.scafa.min()=}, {self.evo.scafa.max()=}, {self.evo.scafa.shape=}")
            print(f"{self.evo.scafa[:,-1].min()=}, {self.evo.scafa[:,-1].max()=}, {self.evo.scafa[:,-1].shape=}")
            print(f"{self.evo.scafa[:,0].min()=}, {self.evo.scafa[:,0].max()=}, {self.evo.scafa[:,0].shape=}")
            # evolve binary population
            self.evo.evolve()
            coal = self.evo.coal
            print(f"{coal.shape=}, {coal[coal].shape}")
            print(f"{self.evo.mass.shape=}")
            print(f"{self.evo.sepa.min()=}, {self.evo.sepa.max()=}, {self.evo.sepa.shape=}")
            print(f"{self.evo.scafa[:,-1].min()=}, {self.evo.scafa[:,-1].max()=}, {self.evo.scafa[:,-1].shape=}")
            print(f"{self.evo.scafa[:,0].min()=}, {self.evo.scafa[:,0].max()=}, {self.evo.scafa[:,0].shape=}")
            print(f"{self.evo.scafa[coal,-1].min()=}, {self.evo.scafa[coal,-1].max()=}, {self.evo.scafa[coal,-1].shape=}")
            print(f"{self.evo.scafa[coal,0].min()=}, {self.evo.scafa[coal,0].max()=}, {self.evo.scafa[coal,0].shape=}")
            print(f"{self.evo.scafa[~coal,-1].min()=}, {self.evo.scafa[~coal,-1].max()=}, {self.evo.scafa[~coal,-1].shape=}")
            print(f"{self.evo.scafa[~coal,0].min()=}, {self.evo.scafa[~coal,0].max()=}, {self.evo.scafa[~coal,0].shape=}")
            tol=1.0e-8
            ix_a1_at_t0 = np.where(np.abs(self.evo.scafa[:,0]-1.0)<tol)[0]
            print(f"{ix_a1_at_t0.shape=}")
            ix_a1 = np.where(np.abs(self.evo.scafa[:,-1]-1.0)<tol)[0]
            print(f"{ix_a1.shape=}")
            #print(f"# with scafa=1 in first evo step: {self.evo.scafa[self.evo.scafa[:,0]==1.0,0].size}")
            #print(f"# with scafa=1 in last evo step: {self.evo.scafa[self.evo.scafa[:,-1]==1.0,-1].size}")

            print(f"\nsepa (coal) init step [PC]: min={self.evo.sepa[coal,0].min()/PC:.6g}, "
                  f"max={self.evo.sepa[coal,0].max()/PC:.6g}, med={np.median(self.evo.sepa[coal,0])/PC:.6g}")
            print(f"sepa (~coal) init step [PC]: min={self.evo.sepa[~coal,0].min()/PC:.6g}, "
                  f"max={self.evo.sepa[~coal,0].max()/PC:.6g}, med={np.median(self.evo.sepa[~coal,0])/PC:.6g}")
            print(f"sepa (coal) final step [PC]: min={self.evo.sepa[coal,-1].min()/PC:.6g}, "
                  f"max={self.evo.sepa[coal,-1].max()/PC:.6g}, med={np.median(self.evo.sepa[coal,-1])/PC:.6g}")
            print(f"sepa (~coal) final step [PC]: min={self.evo.sepa[~coal,-1].min()/PC:.6g}, "
                  f"max={self.evo.sepa[~coal,-1].max()/PC:.6g}, med={np.median(self.evo.sepa[~coal,-1])/PC:.6g}")

            print(f"\nscafa (coal) init step: min={self.evo.scafa[coal,0].min():.6g}, "
                  f"max={self.evo.scafa[coal,0].max():.6g}, med={np.median(self.evo.scafa[coal,0]):.6g}")
            print(f"scafa (~coal) init step: min={self.evo.scafa[~coal,0].min():.6g}, "
                  f"max={self.evo.scafa[~coal,0].max():.6g}, med={np.median(self.evo.scafa[~coal,0]):.6g}")
            print(f"scafa (coal) final step: min={self.evo.scafa[coal,-1].min():.6g}, "
                  f"max={self.evo.scafa[coal,-1].max():.6g}, med={np.median(self.evo.scafa[coal,-1]):.6g}")
            print(f"scafa (~coal) final step: min={self.evo.scafa[~coal,-1].min():.6g}, "
                  f"max={self.evo.scafa[~coal,-1].max():.6g}, med={np.median(self.evo.scafa[~coal,-1]):.6g}")

            print(f"\ntlook (coal) init step: min={self.evo.tlook[coal,0].min()/GYR:.6g}, "
                  f"max={self.evo.tlook[coal,0].max()/GYR:.6g}, med={np.median(self.evo.tlook[coal,0])/GYR:.6g}")
            print(f"tlook (~coal) init step: min={self.evo.tlook[~coal,0].min()/GYR:.6g}, "
                  f"max={self.evo.tlook[~coal,0].max()/GYR:.6g}, med={np.median(self.evo.tlook[~coal,0])/GYR:.6g}")
            print(f"tlook (coal) final step: min={self.evo.tlook[coal,-1].min()/GYR:.6g}, "
                  f"max={self.evo.tlook[coal,-1].max()/GYR:.6g}, med={np.median(self.evo.tlook[coal,-1])/GYR:.6g}")
            print(f"tlook (~coal) final step: min={self.evo.tlook[~coal,-1].min()/GYR:.6g}, "
                  f"max={self.evo.tlook[~coal,-1].max()/GYR:.6g}, med={np.median(self.evo.tlook[~coal,-1])/GYR:.6g}")
            dtlook = (self.evo.tlook[:,0]-self.evo.tlook[:,-1])/GYR
            print(f"dtlook (coal): min={dtlook[coal].min():.6g}, max={dtlook[coal].max():.6g}, "
                  f"med={np.median(dtlook[coal]):.6g}")
            print(f"dtlook (~coal): min={dtlook[~coal].min():.6g}, max={dtlook[~coal].max():.6g}, "
                  f"med={np.median(dtlook[~coal]):.6g}")

            print(f"\nmass1 (coal) init step [MSOL]: min={self.evo.mass[coal,0,0].min()/MSOL:.6g}, "
                  f"max={self.evo.mass[coal,0,0].max()/MSOL:.6g}, med={np.median(self.evo.mass[coal,0,0])/MSOL:.6g}")
            print(f"mass1 (~coal) init step [MSOL]: min={self.evo.mass[~coal,0,0].min()/MSOL:.6g}, "
                  f"max={self.evo.mass[~coal,0,0].max()/MSOL:.6g}, med={np.median(self.evo.mass[~coal,0,0])/MSOL:.6g}")

            print(f"\nmass2 (coal) init step [MSOL]: min={self.evo.mass[coal,0,1].min()/MSOL:.6g}, "
                  f"max={self.evo.mass[coal,0,1].max()/MSOL:.6g}, med={np.median(self.evo.mass[coal,0,1])/MSOL:.6g}")
            print(f"mass2 (~coal) init step [MSOL]: min={self.evo.mass[~coal,0,1].min()/MSOL:.6g}, "
                  f"max={self.evo.mass[~coal,0,1].max()/MSOL:.6g}, med={np.median(self.evo.mass[~coal,0,1])/MSOL:.6g}")

            #print(f"\ninit/final evo vals for 'coal at t0' binaries:")
            #for ii in range(len(ix_coal_at_t0)):
            #    print(f"sepa [PC]: {self.evo.sepa[ix_coal_at_t0[ii],0]/PC:.6g}, {self.evo.sepa[ix_coal_at_t0[ii],-1]/PC:.6g}, "
            #          f"scafa: {self.evo.scafa[ix_coal_at_t0[ii],0]:.6g}, {self.evo.scafa[ix_coal_at_t0[ii],-1]:.6g}, "
            #          f"m1: {self.evo.mass[ix_coal_at_t0[ii],0,0]:.6g}, {self.evo.mass[ix_coal_at_t0[ii],-1,0]:.6g}, "
            #          f"m2: {self.evo.mass[ix_coal_at_t0[ii],0,1]:.6g}, {self.evo.mass[ix_coal_at_t0[ii],-1,1]:.6g}")
            
            ## create GWB
            self.gwb = holo.gravwaves.GW_Discrete(self.evo, self.freqs, nreals=self.nreals) #, nloudest=self.nloudest)
            self.gwb.emit(nloudest=self.nloudest)

    def sim_mass_resolution(self):
        # Baryonic mass resolution for each simulation, in Msun
        mres_baryon = {
            'Ill-1': 1.3e6,
            'TNG50-1': 8.4e4,
            'TNG50-2': 6.8e5,
            'TNG50-3': 5.4e6,
            'TNG100-1': 1.4e6,
            'TNG100-2': 1.1e7,
            'TNG300-1': 1.1e7,
        }
        self.mres_baryon = None
        for k in mres_baryon.keys():
            if k in self.lbl: 
                self.mres_baryon = mres_baryon[k]
                break
        if self.mres_baryon is None:
            raise ValueError(f"{self.lbl=} has no match for mres_baryon.keys().")

    
    def load_sim_gsmf_file(self, basePath=_PATH_DATA):

        for s in ['Ill-1', 'TNG50-1', 'TNG100-1', 'TNG300-1']:
            if s in self.lbl: 
                print(f"{s=}")
                try:                
                    gsmf_fpath = os.path.join(basePath, f"gsmf_all_snaps_Nmin1_{s}.hdf5")
                    print(f"{gsmf_fpath=}")
                    f = h5py.File(gsmf_fpath,"r")
                except:
                    try:
                        gsmf_fpath = os.path.join(_PATH_DATA, f"gsmf_all_snaps_Nmin1_{s}.hdf5")
                        print(f"{gsmf_fpath=}")
                        f = h5py.File(gsmf_fpath,"r")
                    except:
                        raise Exception(f"Could not open GSMF file {gsmf_fpath}.")

        if not hasattr(self, "mhist_bins"): self.mhist_bins = {}
        if not hasattr(self, "gsmf"): self.gsmf = {}
        if not hasattr(self, "mhist"): self.mhist = {}
        
        return f
        
    def calc_sim_gsmf_from_snaps(self, req_z, req_binsize=0.05, verbose=False): 

        f = self.load_sim_gsmf_file(self.basepath)

        box_vol_mpc = f.attrs['box_volume_mpc']
        snapnums = f.attrs['SnapshotNums']
        scalefacs = f.attrs['SnapshotScaleFacs']
        zsnaps = 1.0 / scalefacs - 1.0
    
        diff = np.abs(zsnaps-req_z)
        snapNum = snapnums[diff==diff.min()][0]
        zsnap = zsnaps[diff==diff.min()][0]
        if verbose or (diff.min()>0.01):
            print(f"{req_z=}, {snapNum=}, {zsnap=}, {diff.min()=}")

        dlgm_orig = f.attrs['LogMassBinWidth']
        mbin_edges_orig = np.array(f['StellarMassBinEdges'])
        nbins_orig = mbin_edges_orig.size - 1
        mhist_all_snaps = np.array(f['StellarMassHistograms'])
    
        mhist_snap_orig = mhist_all_snaps[:,(snapnums==snapNum)].flatten()
        if verbose: print(f"{mhist_snap_orig.shape=}, {mbin_edges_orig.shape=}")
        if mhist_snap_orig.size != nbins_orig:
            print('whoops')
            return

        if req_binsize < dlgm_orig:
            raise ValueError(f"{req_binsize=} requested, but min allowed is {dlgm_orig=}")
        if int(req_binsize/dlgm_orig) > nbins_orig/2:
            raise ValueError(f"{req_binsize=} requested, but max allowed is {dlgm_orig*nbins_orig/2=}")

        ncomb = int(req_binsize/dlgm_orig)
        dlgm = dlgm_orig * ncomb
        mbin_edges = mbin_edges_orig[::ncomb]
        nbins = mbin_edges.size
        if ncomb > 1:
            mbin_edges = np.append(mbin_edges, mbin_edges[-1]+dlgm)
            mhist_snap = np.zeros((nbins))
            if verbose: print(f"{mbin_edges.size=}")
            for i in range(mbin_edges.size-1):
                mhist_snap[i] = mhist_snap_orig[i*ncomb:i*ncomb+ncomb].sum()
            if verbose:
                print(f"{mbin_edges_orig=}")
                print(f"{mbin_edges=}")
        else:
            if verbose:
                print(f"WARNING: {req_binsize=}, {ncomb=}; retaining original binsize {dlgm_orig=}")
            assert mbin_edges.all() == mbin_edges_orig.all() and dlgm == dlgm_orig, "Error in setting ncomb=1!"
            mhist_snap = mhist_snap_orig
        
        if verbose:
            print(f"{mhist_all_snaps.shape=}, {mhist_all_snaps.min()=}, {mhist_all_snaps.max()=}")
            print(f"{mhist_snap.shape=}, {mhist_snap.min()=}, {mhist_snap.max()=}")
            print(f"{snapnums=}")
            print(f"{dlgm=}, {mbin_edges.shape=}")
            print(f"{mbin_edges=}")

        ##gsmf = mhist_snap / dlgm / np.log(10) / box_vol_mpc  # dex^-1 Mpc^-3
        gsmf = mhist_snap / dlgm / box_vol_mpc  # dex^-1 Mpc^-3

        self.mhist_bins[req_z] = mbin_edges[:-1]+0.5*dlgm
        self.gsmf[req_z] = gsmf
        self.mhist[req_z] = mhist_snap
        
        #return mbin_edges[:-1]+0.5*dlgm, gsmf, mhist_snap #mbin_edges, dlgm



def create_dpops(tau=1.0, fsa=1.0e4, mod_mmbulge=True, nreals=500, inclIll=True, inclOldIll=False, 
                 inclT50=True, inclT300=True, inclRescale=False, allow_mbh0=False, skip_evo=False,
                 fsa_only=False, use_mstar_tot_as_mbulge=False, nloudest=10, fpath=_PATH_DATA):
    
    assert ((fsa is not None) or (not fsa_only)), f"{fsa_only=} and {fsa=}; no dpops to generate."
    
    # ---- Set the fixed binary lifetime
    print(f"Setting inspiral timescale tau = {tau} Gyr.")
    tau = tau * GYR
    
    # ---- Define the GWB frequencies
    freqs, freqs_edges = utils.pta_freqs()

    # ---- Initialize return variables
    all_dpops = []
    tng_dpops = []

    # ---- (Optionally) set the fixed initial binary separation & initialize fsa return vars
    if fsa is not None:
        print(f"Setting fixed init binary sep = {fsa} pc.")
        fsa = fsa * PC
        all_fsa_dpops = []
        tng_fsa_dpops = []
        
    # ---- Define dpop attributes: (filename, filepath, plot color, plot linewidth)    
    #tpath = '/orange/lblecha/IllustrisTNG/Runs/'
    #ipath = '/orange/lblecha/Illustris/'
    dpop_attrs = {
        #'Ill-1-N010-bh0' : ('galaxy-mergers_Illustris-1_gas-000_dm-000_star-010_bh-000.hdf5', fpath, 'darkgreen', 1.5),
        #'Ill-1-bh0' : ('galaxy-mergers_Illustris-1_gas-100_dm-100_star-100_bh-000.hdf5', fpath, 'g', 1.5),
        'Ill-1' : ('galaxy-mergers_Illustris-1_gas-100_dm-100_star-100_bh-001.hdf5', fpath, 'g', 2.5),
        #'TNG50-1-N100' : ('galaxy-mergers_TNG50-1_gas-100_dm-100_star-100_bh-001.hdf5',  fpath, 'darkred', 4),
        #'TNG50-1-N100-bh0' : ('galaxy-mergers_TNG50-1_gas-100_dm-100_star-100_bh-000.hdf5', fpath, 'darkred', 3),
        #'TNG50-1-bh0' : ('galaxy-mergers_TNG50-1_gas-800_dm-800_star-800_bh-000.hdf5', fpath, 'r', 2.5),
        'TNG50-1' : ('galaxy-mergers_TNG50-1_gas-800_dm-800_star-800_bh-001.hdf5', fpath, 'r', 3.5),
        #'TNG50-2' : ('galaxy-mergers_TNG50-2_gas-100_dm-100_star-100_bh-001.hdf5', fpath, 'orange', 2.5),
        #'TNG50-3' : ('galaxy-mergers_TNG50-3_gas-012_dm-012_star-012_bh-001.hdf5', fpath, 'y', 1.5),
        #'TNG100-1-N010-bh0' : ('galaxy-mergers_TNG100-1_gas-000_dm-000_star-010_bh-000.hdf5', fpath, 'darkblue', 2.5),
        #'TNG100-1-bh0' : ('galaxy-mergers_TNG100-1_gas-100_dm-100_star-100_bh-000.hdf5', fpath, 'b', 1.5),
        'TNG100-1' : ('galaxy-mergers_TNG100-1_gas-100_dm-100_star-100_bh-001.hdf5', fpath, 'b', 2.5),
        #'TNG100-2' : ('galaxy-mergers_TNG100-2_gas-012_dm-012_star-012_bh-001.hdf5', fpath, 'c', 1.5),
        #'TNG300-1-bh0' : ('galaxy-mergers_TNG300-1_gas-012_dm-012_star-012_bh-000.hdf5', fpath, 'm', 1.0),
        'TNG300-1' : ('galaxy-mergers_TNG300-1_gas-012_dm-012_star-012_bh-001.hdf5', fpath, 'm', 1.5),
        #'TNG300-1-N100' : ('galaxy-mergers_TNG300-1_gas-100_dm-100_star-100_bh-001.hdf5', fpath, 'pink', 1.5),
        #'TNG300-1-N100-bh0' : ('galaxy-mergers_TNG300-1_gas-100_dm-100_star-100_bh-000.hdf5', fpath, 'pink', 1)
        ##---'oldIll' : (None, 'brown', 2.5),
        ##---'Ill-1-nomprog' : ('galaxy_merger_files_with_no_mprog/galaxy-mergers_Illustris-1_gas-100_dm-100_star-100_bh-001.hdf5', 
        ##---                 ipath+'Illustris-1/output/', 'g', 2.5),
        ##---'TNG100-1-nomprog' : ('galaxy_merger_files_with_no_mprog/galaxy-mergers_TNG100-1_gas-100_dm-100_star-100_bh-001.hdf5', 
        ##---                      tpath+'TNG100-1/output/', 'b', 2.5),
        ##---'TNG100-1-bh0-nomprog' : ('galaxy_merger_files_with_no_mprog/galaxy-mergers_TNG100-1_gas-100_dm-100_star-100_bh-000.hdf5', 
        ##---                          tpath+'TNG100-1/output/', 'b', 1.5),
        ##---'TNG100-1-N012-bh0' : ('galaxy_merger_files_with_no_mprog/galaxy-mergers_TNG100-1_gas-012_dm-012_star-012_bh-000.hdf5', 
        ##---                       tpath+'TNG100-1/output/', 'darkblue', 2.5),
        ### dont use this file; it has at least one merger remnant with mbulge=0. prob need to rerun with Ngas=10
        ### ('galaxy-mergers_Illustris-1_gas-000_dm-010_star-010_bh-000.hdf5', 'darkgreen', 1.5)
    }
    
    # ---- Loop thru dict and create dpops
    for l in dpop_attrs.keys():
        if ('Ill' in l) and (not inclIll): 
            continue
        if (l == 'oldIll') and (not inclOldIll):
            continue
        if ('TNG50' in l) and (not inclT50): 
            continue
        if ('TNG300' in l) and (not inclT300): 
                continue

        if not fsa_only:
            #if '-bh0' not in l:
            dp = Discrete(freqs, freqs_edges, lbl=l, tau=tau, fixed_sepa=None, nreals=nreals,
                          allow_mbh0=allow_mbh0, skip_evo=skip_evo, attrs=dpop_attrs[l],
                          use_mstar_tot_as_mbulge=use_mstar_tot_as_mbulge ,nloudest=nloudest)

            all_dpops = all_dpops + [dp]
            if 'Ill' not in l: 
                tng_dpops = tng_dpops + [dp]
            #else:
            #    print(f"Skipping run {l} with bh0")

        if fsa is not None:

            lbl='fsa-mm-'+l if mod_mmbulge else 'fsa-'+l
            dp_fsa = Discrete(freqs, freqs_edges, lbl='fsa-mm-'+l, tau=tau, fixed_sepa=fsa, nreals=nreals,
                              allow_mbh0=allow_mbh0, skip_evo=skip_evo, attrs=dpop_attrs[l], 
                              mod_mmbulge=mod_mmbulge, use_mstar_tot_as_mbulge=use_mstar_tot_as_mbulge, nloudest=nloudest)

            all_fsa_dpops = all_fsa_dpops + [dp_fsa]
            if 'Ill' not in l: 
                tng_fsa_dpops = tng_fsa_dpops + [dp_fsa]
            
            if ('TNG300' in l) and (inclT300) and (inclRescale):
                rescale_dp_fsa = Discrete(freqs, freqs_edges, lbl='rescale-fsa-mm-'+l,tau=tau, fixed_sepa=fsa, 
                                          nreals=nreals, allow_mbh0=allow_mbh0, skip_evo=skip_evo, attrs=dpop_attrs[l],
                                          mod_mmbulge=True, use_mstar_tot_as_mbulge=use_mstar_tot_as_mbulge, rescale_mbulge=True, nloudest=nloudest)
                tng_fsa_dpops = tng_fsa_dpops + [rescale_dp_fsa]

        print(f"{l} dpop_attrs: {dpop_attrs[l][0]} {dpop_attrs[l][1]} {dpop_attrs[l][2]} {dpop_attrs[l][3]}")

    
    if fsa is not None:

        return all_dpops, tng_dpops, all_fsa_dpops, tng_fsa_dpops

    else:
        
        return all_dpops, tng_dpops