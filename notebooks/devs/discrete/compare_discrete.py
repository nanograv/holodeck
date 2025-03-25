"""Compare multiple discrete MBH Binary Populations (from cosmological hydrodynamic simulations)."""

import numpy as np
import holodeck as holo
from holodeck import utils, log, _PATH_DATA, cosmo, discrete
from holodeck.constants import PC, MSOL, YR, MPC, GYR, SPLC


class Discrete:
    
    def __init__(self, freqs, freqs_edges, attrs=(None,None,'k',1.0), lbl=None, fixed_sepa=None, 
                 tau=1.0*YR, nreals=500, mod_mmbulge=False, rescale_mbulge=False, allow_mbh0=False, 
                 skip_evo=False, use_mstar_tot_as_mbulge=False):

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
            print(f"after mass mod: {self.pop.mbulge.min()=}, {self.pop.mbulge.max()=}, {self.pop.mbulge.shape=}")
            self.mmbulge = holo.relations.MMBulge_KH2013()
            self.mod_KH2013 = discrete.population.PM_Mass_Reset(self.mmbulge, scatter=True, 
                                                                rescale_mbulge=rescale_mbulge)
            self.pop.modify(self.mod_KH2013)
            print(f"after mass mod: {self.pop.mass.min()=}, {self.pop.mass.max()=}, {self.pop.mass.shape=}")
            print(f"after mass mod: {self.pop.mbulge.min()=}, {self.pop.mbulge.max()=}, {self.pop.mbulge.shape=}")
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
            self.gwb = holo.gravwaves.GW_Discrete(self.evo, self.freqs, nreals=self.nreals)
            self.gwb.emit()



def create_dpops(tau=1.0, fsa=1.0e4, mod_mmbulge=True, nreals=500, inclIll=True, inclOldIll=False, 
                 inclT50=True, inclT300=True, inclRescale=False, allow_mbh0=False, skip_evo=False,
                 fsa_only=False, use_mstar_tot_as_mbulge=False):
    
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
        
    # ---- Define dpop attributes: (filename, plot color, plot linewidth)
    tpath = '/orange/lblecha/IllustrisTNG/Runs/'
    ipath = '/orange/lblecha/Illustris/'
    dpop_attrs = {
        # dont use this file; it has at least one merger remnant with mbulge=0. prob need to rerun with Ngas=10
        ### ('galaxy-mergers_Illustris-1_gas-000_dm-010_star-010_bh-000.hdf5', 'darkgreen', 1.5), 
        #'TNG50-1-N100' : ('galaxy-mergers_TNG50-1_gas-100_dm-100_star-100_bh-001.hdf5', 
        #                  tpath+'TNG50-1/output/', 'darkred', 4),
        #'TNG50-1-N100-bh0' : ('galaxy-mergers_TNG50-1_gas-100_dm-100_star-100_bh-000.hdf5', 
        #                      tpath+'TNG50-1/output/', 'darkred', 3),
        #'TNG50-1' : ('galaxy-mergers_TNG50-1_gas-800_dm-800_star-800_bh-001.hdf5', 
        #             tpath+'TNG50-1/output/', 'r', 3.5),
        #'TNG50-1-bh0' : ('galaxy-mergers_TNG50-1_gas-800_dm-800_star-800_bh-000.hdf5', 
        #                 tpath+'TNG50-1/output/', 'r', 2.5),
        #'TNG50-2' : ('galaxy-mergers_TNG50-2_gas-100_dm-100_star-100_bh-001.hdf5', 
        #             tpath+'TNG50-2/output/', 'orange', 2.5),
        #'TNG50-3' : ('galaxy-mergers_TNG50-3_gas-012_dm-012_star-012_bh-001.hdf5', 
        #             tpath+'TNG50-3/output/', 'y', 1.5),
        ##'oldIll' : (None, 'brown', 2.5),
        #---'Ill-nomprog' : ('galaxy_merger_files_with_no_mprog/galaxy-mergers_Illustris-1_gas-100_dm-100_star-100_bh-001.hdf5', 
        #---                 ipath+'Illustris-1/output/', 'g', 2.5),
        #'Ill-N010-bh0' : ('galaxy-mergers_Illustris-1_gas-000_dm-000_star-010_bh-000.hdf5', 
        #                  ipath+'Illustris-1/output/', 'darkgreen', 1.5),
        #'Ill-bh0' : ('galaxy-mergers_Illustris-1_gas-100_dm-100_star-100_bh-000.hdf5', 
        #             ipath+'Illustris-1/output/', 'g', 1.5),
        #'Ill' : ('galaxy-mergers_Illustris-1_gas-100_dm-100_star-100_bh-001.hdf5', 
        #         ipath+'Illustris-1/output/', 'g', 2.5),
        #'TNG100-1-N010-bh0' : ('galaxy-mergers_TNG100-1_gas-000_dm-000_star-010_bh-000.hdf5', 
        #                       tpath+'TNG100-1/output/', 'darkblue', 2.5),
        #'TNG100-1-bh0' : ('galaxy-mergers_TNG100-1_gas-100_dm-100_star-100_bh-000.hdf5', 
        #                  tpath+'TNG100-1/output/', 'b', 1.5),
        'TNG100-1' : ('galaxy-mergers_TNG100-1_gas-100_dm-100_star-100_bh-001.hdf5', 
                      tpath+'TNG100-1/output/', 'b', 2.5),
        #---'TNG100-1-nomprog' : ('galaxy_merger_files_with_no_mprog/galaxy-mergers_TNG100-1_gas-100_dm-100_star-100_bh-001.hdf5', 
        #---                      tpath+'TNG100-1/output/', 'b', 2.5),
        #---'TNG100-1-bh0-nomprog' : ('galaxy_merger_files_with_no_mprog/galaxy-mergers_TNG100-1_gas-100_dm-100_star-100_bh-000.hdf5', 
        #---                          tpath+'TNG100-1/output/', 'b', 1.5),
        #---'TNG100-1-N012-bh0' : ('galaxy_merger_files_with_no_mprog/galaxy-mergers_TNG100-1_gas-012_dm-012_star-012_bh-000.hdf5', 
        #---                       tpath+'TNG100-1/output/', 'darkblue', 2.5),
        #'TNG100-2' : ('galaxy-mergers_TNG100-2_gas-012_dm-012_star-012_bh-001.hdf5', 
        #              tpath+'TNG100-1/output/', 'c', 1.5),
        #'TNG300-1' : ('galaxy-mergers_TNG300-1_gas-012_dm-012_star-012_bh-001.hdf5', tpath+'TNG300-1/output/', 'm', 1.5),
        #'TNG300-1-bh0' : ('galaxy-mergers_TNG300-1_gas-012_dm-012_star-012_bh-000.hdf5', tpath+'TNG300-1/output/', 'm', 1.0),
        #'TNG300-1-N100' : ('galaxy-mergers_TNG300-1_gas-100_dm-100_star-100_bh-001.hdf5', tpath+'TNG300-1/output/', 'pink', 1.5),
        #'TNG300-1-N100-bh0' : ('galaxy-mergers_TNG300-1_gas-100_dm-100_star-100_bh-000.hdf5', tpath+'TNG300-1/output/', 'pink', 1)
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
                          use_mstar_tot_as_mbulge=use_mstar_tot_as_mbulge)

            all_dpops = all_dpops + [dp]
            if 'Ill' not in l: 
                tng_dpops = tng_dpops + [dp]
            #else:
            #    print(f"Skipping run {l} with bh0")

        if fsa is not None:

            lbl='fsa-mm-'+l if mod_mmbulge else 'fsa-'+l
            dp_fsa = Discrete(freqs, freqs_edges, lbl='fsa-mm-'+l, tau=tau, fixed_sepa=fsa, nreals=nreals,
                              allow_mbh0=allow_mbh0, skip_evo=skip_evo, attrs=dpop_attrs[l], 
                              mod_mmbulge=mod_mmbulge, use_mstar_tot_as_mbulge=use_mstar_tot_as_mbulge)

            all_fsa_dpops = all_fsa_dpops + [dp_fsa]
            if 'Ill' not in l: 
                tng_fsa_dpops = tng_fsa_dpops + [dp_fsa]
            
            if ('TNG300' in l) and (inclT300) and (inclRescale):
                rescale_dp_fsa = Discrete(freqs, freqs_edges, lbl='rescale-fsa-mm-'+l,tau=tau, fixed_sepa=fsa, 
                                          nreals=nreals, allow_mbh0=allow_mbh0, skip_evo=skip_evo, attrs=dpop_attrs[l],
                                          mod_mmbulge=True, use_mstar_tot_as_mbulge=use_mstar_tot_as_mbulge, rescale_mbulge=True)
                tng_fsa_dpops = tng_fsa_dpops + [rescale_dp_fsa]

        print(f"{l} dpop_attrs: {dpop_attrs[l][0]} {dpop_attrs[l][1]} {dpop_attrs[l][2]} {dpop_attrs[l][3]}")

    
    if fsa is not None:

        return all_dpops, tng_dpops, all_fsa_dpops, tng_fsa_dpops

    else:
        
        return all_dpops, tng_dpops