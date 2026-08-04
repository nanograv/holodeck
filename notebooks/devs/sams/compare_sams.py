"""Compare multiple MBH binary SAMS."""

import os
import sys
import h5py
import numpy as np
import pickle

import astropy as ap
import scipy as sp
import scipy.stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import tqdm.notebook as tqdm

import kalepy as kale
import kalepy.utils
import kalepy.plot

import holodeck as holo
import holodeck.sams
import holodeck.gravwaves
from holodeck import cosmo, utils, plot, discrete, sams, host_relations, _PATH_DATA
from holodeck import utils, log
from holodeck.constants import MSOL, PC, YR, MPC, GYR, SPLC, NWTG, SCHW

from pathlib import Path


# ---- Define filepath containing simulation galaxy merger data files ----#
# ---- (if using files not in _PATH_DATA) ---- #
_HOME_PATH = Path('~/').expanduser()
p = os.path.join(_HOME_PATH, 'cosmo_sim_merger_data')
if os.path.exists(p):
    _SIM_MERGER_PATH = p
else:
    p = os.path.join(_HOME_PATH, 'nanograv/cosmo_sim_merger_data')
    if os.path.exists(p):
        _SIM_MERGER_PATH = p
    else:
        _SIM_MERGER_PATH = _PATH_DATA
#print(f"{_SIM_MERGER_PATH=}")
# ------------------------------------------------------------------------ #

class Test_SAM:
    
    def __init__(self, model_type='old', hard_type='fixed2PL',
                 nreals=10, nloud=5, gridshape=None, gpf_flag=0,
                 galaxy_pars_type=None, skip_evo=False, bfrac=None, 
                 hard_t=None, hard_ai=None, hard_rc=None, 
                 hard_nin=None, hard_nout=None, gsmf_flag = None,
                 mmbulge=None, var_value=None, tout_default=None,
                 nuin_default=None, dadt_default=None, 
                 alphgw_default=None, betagw_default=None, 
                 rgw9_default=None,
                 alphch_default=None, rch9_default=None,
                 gwb_params=False,
                 mtrange=None):

        if hard_type not in ('fixed2PL','fixedOuter'):
            raise ValueError(f"{hard_type=} note defined, must be 'fixed2PL' or 'fixedOuter'.")
        
        self.model_type = model_type
        self.hard_type = hard_type
        self.nreals = nreals
        self.nloud = nloud
        self.gpf_flag = gpf_flag
        
        self.skip_evo = skip_evo
        self.bfrac = None
        print(f"\nCreating Test_SAM class instance with model_type={self.model_type}")

        if model_type == 'grid':
            if hard_type != 'fixed2PL':
                raise ValueError("model_type='grid' valid only with hard_type='fixed2PL.")
                
            if None in (hard_t, hard_ai, hard_rc, hard_nin, hard_nout, gsmf_flag):
                raise ValueError(f"cannot set grid elem for sam params if any keyword is None.")

            if mtrange is not None:
                log.warning("Ignoring keyword `mtrange` for model_type=='grid'.")
                
            self.set_sam_params_grid(tau=hard_t, ai=hard_ai, rc=hard_rc, 
                                     nin=hard_nin, nout=hard_nout, mf = gsmf_flag)
            if gpf_flag:
                self.sam = sams.Semi_Analytic_Model(gpf = sams.GPF_Power_Law(),
                                                    shape = gridshape)
            else: 
                self.sam = sams.Semi_Analytic_Model(gpf = None, shape = gridshape)

        else:
            print(f"in TestSAM, defaults: {tout_default=} {nuin_default=} {dadt_default=} "
                  f"{alphgw_default=} {rgw9_default=} {alphch_default=} {rch9_default=}")
            
            self.set_sam_params_manual(gpf_flag=gpf_flag, tau=hard_t, var_value=var_value, 
                                       galaxy_pars_type=galaxy_pars_type,
                                       tout_default=tout_default,
                                       nuin_default=nuin_default, dadt_default=dadt_default,
                                       alphgw_default=alphgw_default, betagw_default=betagw_default,
                                       rgw9_default=rgw9_default,
                                       alphch_default=alphch_default, rch9_default=rch9_default)

            if 'new_hardening' in self.model_type:
                print(f"pars: {self.PARS['hard_outer_time']=} {self.PARS['hard_dadt_rchar']=} "
                      f"{self.PARS['hard_rchar_9']=} {self.PARS['hard_alpha_char']=} "
                      f"{self.PARS['hard_r_gw_crit_9']=} {self.PARS['hard_alpha_gw_crit']=}"
                      f"{self.PARS['hard_beta_gw_crit']=}")

            self.nfreqs = self.PARS['freqs'].size
            print(f"{self.nfreqs=}")
            
            if gpf_flag:
                print(f"creating SAM using GPF with {self.model_type=}...")
                self.lbl = model_type+'_gpf'
            else:
                print(f"creating SAM using GMR with {self.model_type=}...")
                self.lbl = model_type+'_gmr'

            if mtrange is not None:
                if  len(mtrange) != 2:
                    raise ValueError("len(mtrange) must be 2")
                _mt = (10.0**mtrange[0]*MSOL, 10.0**mtrange[1]*MSOL, gridshape)
                print(f"in Test_SAM: {_mt=}")
                log.info(f'{_mt=}')
                self.sam = sams.Semi_Analytic_Model(gsmf = self.PARS['gsmf'], 
                                                    gpf = self.PARS['gpf'],
                                                    gmt = self.PARS['gmt'],
                                                    gmr = self.PARS['gmr'],
                                                    mmbulge = self.PARS['mmbulge'],
                                                    shape = gridshape,
                                                    mtot=_mt)
            else:
                self.sam = sams.Semi_Analytic_Model(gsmf = self.PARS['gsmf'], 
                                                    gpf = self.PARS['gpf'],
                                                    gmt = self.PARS['gmt'],
                                                    gmr = self.PARS['gmr'],
                                                    mmbulge = self.PARS['mmbulge'],
                                                    shape = gridshape)
        
        print(f"    ...calculating hardening for GPF SAM with {self.model_type=}")
        if self.hard_type == 'fixed2PL':
            self.hard = holo.hardening.Fixed_Time_2PL_SAM(self.sam, self.PARS['hard_time']*GYR, 
                                                          sepa_init = self.PARS['hard_sepa_init']*PC,
                                                          rchar = self.PARS['hard_rchar']*PC,
                                                          gamma_inner = self.PARS['hard_gamma_inner'],
                                                          gamma_outer = self.PARS['hard_gamma_outer']
                                                         )
        else:
            print(
                f"{self.PARS['hard_dadt_rchar']=}, {self.PARS['hard_rchar_9']=}, "
                f"{self.PARS['hard_alpha_char']=} {self.PARS['hard_r_gw_crit_9']=}, "
                f"{self.PARS['hard_alpha_gw_crit']=}, {self.PARS['hard_beta_gw_crit']=}, "
                f"{self.PARS['hard_nu_inner']=}, {self.PARS['hard_inner_time']=}"
            ) 
            self.hard = holo.hardening.FixedOuterTime_InnerPL_SAM(self.sam, 
                                                                  inner_model_type = self.PARS['hard_inner_model_type'],
                                                                  outer_time = self.PARS['hard_outer_time']*GYR, 
                                                                  rchar_9 = self.PARS['hard_rchar_9']*PC,
                                                                  alpha_char = self.PARS['hard_alpha_char'],
                                                                  nu_inner = self.PARS['hard_nu_inner'],
                                                                  gw_crit_units= self.PARS['hard_gw_crit_units'],
                                                                  r_gw_crit_9 = self.PARS['hard_r_gw_crit_9'],
                                                                  alpha_gw_crit = self.PARS['hard_alpha_gw_crit'],
                                                                  beta_gw_crit = self.PARS['hard_beta_gw_crit'],
                                                                  dadt_rchar = self.PARS['hard_dadt_rchar'],
                                                                  inner_time = self.PARS['hard_inner_time']
                                                                 )
        
        ### ***NOTE*** gwb() allows for including pars of loud sources (unlike gwb_new()):
        if 'new_hardening' in self.model_type:
            if np.any(self.hard._params_allowed==False):
                print("    ...skipping gwb for SAM with invalid hardening model(s):")
                print(f"outer time={self.PARS['hard_outer_time']} dadt(rchar)={self.PARS['hard_dadt_rchar']} "
                      f"alpha,gw={self.PARS['hard_alpha_gw_crit']}, beta,gw={self.PARS['hard_beta_gw_crit']}, "
                      f"rgw9={self.PARS['hard_r_gw_crit_9']} "
                      f"alpha,ch={self.PARS['hard_alpha_char']}, rchar9={self.PARS['hard_rchar_9']}")
                mt, mr, = np.broadcast_arrays(
                    self.sam.mtot[:, np.newaxis],
                    self.sam.mrat[np.newaxis, :]
                )
                print(f"min/max bad mt={np.log10(mt[self.hard._params_allowed==False].min()/MSOL)}/"
                      f"{np.log10(mt[self.hard._params_allowed==False].max()/MSOL)}")
                print(f"min/max bad mr={mr[self.hard._params_allowed==False].min()}/"
                      f"{mr[self.hard._params_allowed==False].max()}")
                self.gwb_sam = None
            else:
                print("    ...creating gwb for SAM")
                if gwb_params:
                    self.gwb_sam = self.sam.gwb(self.PARS['freqs_edges'], self.hard,
                                                realize=self.PARS['NREALS'], 
                                                loudest=self.PARS['NLOUD'], params=True)
                else:
                    # use the faster version if we don't need the binary param info 
                    # from the gwb calculation
                    self.gwb_sam = self.sam.gwb_new(self.PARS['freqs_edges'], self.hard,
                                                    realize=self.PARS['NREALS'])

        else:
            log.warning(f"No check is performed on allowed param space for old hardening type "
                        f"{self.model_type}! GWB results should be interpreted with caution!")
            print("    ...creating gwb for SAM")
            if gwb_params:
                    self.gwb_sam = self.sam.gwb(self.PARS['freqs_edges'], self.hard,
                                                realize=self.PARS['NREALS'], 
                                                loudest=self.PARS['NLOUD'], params=True)
            else:
                # use the faster version if we don't need the binary param info 
                # from the gwb calculation
                self.gwb_sam = self.sam.gwb_new(self.PARS['freqs_edges'], self.hard,
                                                realize=self.PARS['NREALS'])  

    
    def set_sam_params_manual(self, gpf_flag=0, tau=None, galaxy_pars_type=None, var_value=None, 
                              tout_default=None, dadt_default=None, nuin_default=None, 
                              alphgw_default=None, betagw_default=None, rgw9_default=None, 
                              alphch_default=None, rch9_default=None):
        
        modlist = [
            'old', 'old_2s', 'old_rc100', 'ph15', 'astr', 'astr_nuo0', 'astr_rc100',
            'ph15_nuivar', 'ph15_muvar', 'ph15_epsmuvar', 'ph15_phivar', 'ph15_Mphivar',
            'astr_rc100_nuivar', 'astr_rc100_muvar', 'astr_rc100_epsmuvar',
            'astr_rc100_phi10var', 'astr_rc100_phi20var', 
            'astr_rc100_Mc0var', 'astr_rc100_Mc1var', 'astr_rc100_Mc2var',
            'new_hardening_type0_toutvar', 'new_hardening_type0_nuivar', 
            'new_hardening_type0_rgw9var', 'new_hardening_type0_alphgwvar', 
            'new_hardening_type0_betagwvar', 'new_hardening_type0_rch9var',
            'new_hardening_type1_toutvar', 'new_hardening_type1_dadtvar', 
            'new_hardening_type1_rgw9var', 'new_hardening_type1_alphgwvar',
            'new_hardening_type1_rch9var'
        ]

        if self.model_type not in modlist:
            raise ValueError(f"{self.model_type=} is not defined. Options are {[m for m in modlist]}.")
        
        # ---- Define the key model params ----
        
        if 'new_hardening' not in self.model_type:
            # ---- OLD HARDENING MODELS ----
            
            if tau is None:
                raise ValueError("must choose a numerical value of keyword tau (in Gyr)!")

            if alphch_default is not None:    
                raise ValueError()
                
            log.warning(f"for {self.model_type=}, {rch9_default=} will be interpreted as "
                        f"the constant rchar value for all MBHB masses")

            self.PARS = dict(
                desc='old hardening template',
                hard_sepa_init=1e4,     # [pc]
                mmbulge = host_relations.MMBulge_KH2013,
                gpf = None,
                gmt = None,
                gmr = None
            )
            if gpf_flag:
                self.PARS['gpf'] = sams.components.GPF_Power_Law()
                self.PARS['gmt'] = sams.components.GMT_Power_Law()
            else:
                self.PARS['gmr'] = sams.components.GMR_Illustris()

            # define the fixed inspiral timescale from input keyword
            self.PARS["hard_time"] = tau    # [Gyr]

            # ---- Set params for individual models
            
            if self.model_type == 'old':
                self.PARS['desc']='LB old'
                self.PARS['hard_rchar']=10.0      # [pc]
                self.PARS['hard_gamma_inner']=-1.0
                self.PARS['hard_gamma_outer']=+1.5
                self.PARS['gsmf'] = holo.sams.GSMF_Schechter()
                
            elif self.model_type == 'old_2s':
                self.PARS['desc']='LB old + 2-Schechter'
                self.PARS['hard_rchar']=10.0        # [pc]
                self.PARS['hard_gamma_inner']=-1.0
                self.PARS['hard_gamma_outer']=+1.5
                self.PARS['gsmf'] = holo.sams.GSMF_Double_Schechter()
                
            elif self.model_type == 'old_rc100':
                self.PARS['desc']='LB old + rchar=100'
                self.PARS['hard_rchar']=100.0        # [pc]
                self.PARS['hard_gamma_inner']=-1.0
                self.PARS['hard_gamma_outer']=+1.5
                self.PARS['gsmf'] = holo.sams.GSMF_Schechter()
                
            elif self.model_type == 'ph15':
                self.PARS['desc']='15yr Phenom'
                self.PARS['hard_rchar']=100.0        # [pc]
                self.PARS['hard_gamma_inner']=-1.0
                self.PARS['hard_gamma_outer']=+2.5
                self.PARS['gsmf'] = holo.sams.GSMF_Schechter()
                
            elif self.model_type == 'astr':
                self.PARS['desc']='Astro Strong'
                self.PARS['hard_rchar']=10.0        # [pc]
                self.PARS['hard_gamma_inner']=-1.0
                self.PARS['hard_gamma_outer']=+2.5
                self.PARS['gsmf'] = holo.sams.GSMF_Double_Schechter()
                
            elif self.model_type == 'astr_nuo0':
                self.PARS['desc']='Astro Strong w/ nu_outer=0'
                self.PARS['hard_rchar']=10.0        # [pc]
                self.PARS['hard_gamma_inner']=-1.0
                self.PARS['hard_gamma_outer']=0.0
                self.PARS['gsmf'] = holo.sams.GSMF_Double_Schechter()

            elif self.model_type == 'astr_rc100':
                self.PARS['desc']='Astro Strong w/ rchar=100'
                self.PARS['hard_rchar']=100.0        # [pc]
                self.PARS['hard_gamma_inner']=-1.0
                self.PARS['hard_gamma_outer']=+2.5
                self.PARS['gsmf'] = holo.sams.GSMF_Double_Schechter()

            elif self.model_type == 'ph15_nuivar':
                # gamma_inner hardening PL index, varied for NG15, Model A, & Model B
                if var_value is None:
                    raise ValueError(f'{var_value=} invalid for {self.model_type=}')
                else:
                    self.PARS['desc']='15yr Phenom w/ nu_inner varied'
                    self.PARS['hard_rchar']=100.0        # [pc]
                    self.PARS['hard_gamma_inner']=var_value
                    self.PARS['hard_gamma_outer']=+2.5
                    self.PARS['gsmf'] = holo.sams.GSMF_Schechter()

            elif self.model_type == 'ph15_muvar':
                # MMbulge norm, varied for NG15 and Model A, fixed for Model B
                raise NotImplementedError()

            elif self.model_type == 'ph15_epsmuvar':
                # MMbulge scatter, varied for NG15 and Model A, fixed for Model B
                raise NotImplementedError()

            elif self.model_type == 'ph15_phivar':
                # z=0 norm for Chen19 GSMF schechter func, varied for NG15
                # analogous to astr_rc100_phii0var for Model A & B
                self.PARS['desc']='15yr Phenom w/ phivar'
                self.PARS['hard_rchar']=100.0        # [pc]
                self.PARS['hard_gamma_inner']=-1.0
                self.PARS['hard_gamma_outer']=+2.5
                self.PARS['gsmf'] = holo.sams.GSMF_Schechter(phi0=var_value)

            elif self.model_type == 'ph15_Mphivar':
                # z=0 Mchar for Chen19 GSMF schechter func, varied for NG15
                # analogous to astr_Mc0var for Model A & B
                self.PARS['desc']='15yr Phenom w/ Mphivar'
                self.PARS['hard_rchar']=100.0        # [pc]
                self.PARS['hard_gamma_inner']=-1.0
                self.PARS['hard_gamma_outer']=+2.5
                self.PARS['gsmf'] = holo.sams.GSMF_Schechter(mchar0_log10=var_value)

            elif self.model_type == 'astr_rc100_nuivar':
                # gamma_inner hardening PL index, varied for NG15, Model A, & Model B
                if var_value is None:
                    raise ValueError(f'{var_value=} invalid for {self.model_type=}')
                else:
                    self.PARS['desc']='Astro Strong w/ rchar=100 & w/ nu_inner varied'
                    self.PARS['hard_rchar']=100.0        # [pc]
                    self.PARS['hard_gamma_inner']=var_value
                    self.PARS['hard_gamma_outer']=+2.5
                    self.PARS['gsmf'] = holo.sams.GSMF_Double_Schechter()

            elif self.model_type == 'astr_rc100_muvar':
                # MMbulge norm, varied for NG15 and Model A, fixed for Model B
                raise NotImplementedError()

            elif self.model_type == 'astr_rc100_epsmuvar':
                # MMbulge scatter, varied for NG15 and Model A, fixed for Model B
                raise NotImplementedError()
            
            elif self.model_type == 'astr_rc100_phi10var':
                # z=0 norm for 1st Leja20 GSMF schechter func
                # varied for Model A and Model B, analogous to ph15_phivar for NG15
                self.PARS['desc']='Astro Strong w/ rchar=100 & w phi10 varied'
                self.PARS['hard_rchar']=100.0        # [pc]
                self.PARS['hard_gamma_inner']=-1.0
                self.PARS['hard_gamma_outer']=+2.5
                self.PARS['gsmf'] = holo.sams.GSMF_Double_Schechter(log10_phi1=[var_value, -0.264, -0.107])

            elif self.model_type == 'astr_rc100_phi20var':
                # z=0 norm for 2nd Leja20 GSMF schechter func
                # varied for Model A and Model B, analogous to ph15_phivar for NG15
                self.PARS['desc']='Astro Strong w/ rchar=100 & w phi20 varied'
                self.PARS['hard_rchar']=100.0        # [pc]
                self.PARS['hard_gamma_inner']=-1.0
                self.PARS['hard_gamma_outer']=+2.5
                self.PARS['gsmf'] = holo.sams.GSMF_Double_Schechter(log10_phi2=[var_value, -0.368, +0.046])

            elif self.model_type == 'astr_rc100_Mc0var':
                # z=0 Mchar for Leja20 GSMF schechter func
                # varied for Model A and Model B, analogous to phi15_Mphivar for NG15
                self.PARS['desc']='Astro Strong w/ rchar=100 & w Mchar0 varied'
                self.PARS['hard_rchar']=100.0        # [pc]
                self.PARS['hard_gamma_inner']=-1.0
                self.PARS['hard_gamma_outer']=+2.5
                self.PARS['gsmf'] = holo.sams.GSMF_Double_Schechter(log10_mstar=[var_value, +0.124, -0.033])

            elif self.model_type == 'astr_rc100_Mc1var':
                # linear z-evol term for Mchar for Leja20 GSMF schechter func
                # varied for Model A and Model B, doesn't exist in NG15
                self.PARS['desc']='Astro Strong w/ rchar=100 & w Mchar1 varied'
                self.PARS['hard_rchar']=100.0        # [pc]
                self.PARS['hard_gamma_inner']=-1.0
                self.PARS['hard_gamma_outer']=+2.5
                self.PARS['gsmf'] = holo.sams.GSMF_Double_Schechter(log10_mstar=[+10.767, var_value, -0.033])

            elif self.model_type == 'astr_rc100_Mc2var':
                # quadratic z-evol term for Mchar for Leja20 GSMF schechter func
                # varied for Model A and Model B, doesn't exist in NG15         
                self.PARS['desc']='Astro Strong w/ rchar=100 & w Mchar2 varied'
                self.PARS['hard_rchar']=100.0        # [pc]
                self.PARS['hard_gamma_inner']=-1.0
                self.PARS['hard_gamma_outer']=+2.5
                self.PARS['gsmf'] = holo.sams.GSMF_Double_Schechter(log10_mstar=[+10.767, +0.124, var_value])

            if galaxy_pars_type == 'NG15':
                if self.model_type not in ['old', 'old_rc100', 'ph15', 'ph15_nuivar']:
                    raise ValueError(f"{self.model_type=} is incompatible with 'NG15' approx-posterior galaxy pars")
                
                log.warning(f"OVERRIDING gsmf, gpf, gmt, & mmbulge values for {self.model_type=} with NG15 posterior approx values!!")

                if self.PARS['hard_gamma_outer'] != +2.5:
                    log.warning(f"OVERRIDING hard_gamma_outer value for {self.model_type=} with NG15 value (+2.5)!!")
                    self.PARS['hard_gamma_outer'] = +2.5
                if self.PARS['hard_rchar'] != 100.0:
                    log.warning(f"OVERRIDING hard_rchar value for {self.model_type=} with NG15 value (100.0)!!")
                    self.PARS['hard_rchar'] = 100.0        # [pc]

                self.PARS['hard_gamma_inner'] = -0.5
                log.warning(f"OVERRIDING hard_gamma_inner value for {self.model_type=} with NG15 value (-0.5)!!")
                
                # note: phi0 and mchar0 are just chosen here to be *roughly* approximate to 
                # peak values from the NG15 posteriors
                self.PARS['gsmf'] = holo.sams.GSMF_Schechter(phi0=-2.0, phiz=-0.60, 
                                                             mchar0_log10=11.5, mcharz=+0.11, 
                                                             alpha0=-1.21, alphaz=-0.03)
                # note: mamp_log10 and scatter_dex are just chosen to be the mean values for the
                # 'astrophysical priors' from NG15, since the posteriors are ultimately prior dominated
                self.PARS['mmbulge'] = host_relations.MMBulge_Standard(mamp_log10=8.6, mplaw=+1.10,
                                                                       scatter_dex=0.32, bulge_mfrac=+0.615)
                if gpf_flag:
                    # note that frac_norm = frac_norm_allq/(1.0-0.25) = +0.033 is the NG15Astro value
                    self.PARS['gpf'] = sams.components.GPF_Power_Law(frac_norm_allq=+0.025, 
                                                                     malpha=0.0, zbeta=+1.0,
                                                                     qgamma=0.0)
                    self.PARS['gmt'] = sams.components.GMT_Power_Law(time_norm=0.5*GYR,
                                                                     malpha=0.0, zbeta=-0.5, 
                                                                     qgamma=-1.0)
                    self.PARS['gmr'] = None
                else:
                    raise ValueError("`galaxy_pars_type`='NG15' requires `gpf_flag`=1.")                    

        
            # ---- End of old hardening models
        
        else:
            
            # ---- NEW HARDENING MODELS ----
            # set default 'new_hardening' parameter values
            self.PARS = dict(
                desc='new_hardening template',
                hard_inner_model_type=None,       # must set value for each model below
                hard_outer_time=tout_default,     # [Gyr]
                hard_rchar_9=rch9_default,        # [pc]
                hard_alpha_char=alphch_default,
                hard_nu_inner=nuin_default,
                hard_gw_crit_units='rg',
                hard_r_gw_crit_9=rgw9_default, 
                hard_alpha_gw_crit=alphgw_default, 
                hard_beta_gw_crit=betagw_default, 
                hard_dadt_rchar=None, 
                hard_inner_time=None,
                gpf = None,
                gmt = None,
                gmr = None
            )

            # set default galaxy parameter values
            if galaxy_pars_type == 'dev-defaults':
                self.PARS['gsmf'] = holo.sams.GSMF_Double_Schechter()
                self.PARS['mmbulge'] = host_relations.MMBulge_KH2013()                
                if gpf_flag:
                    self.PARS['gpf'] = sams.components.GPF_Power_Law()
                    self.PARS['gmt'] = sams.components.GMT_Power_Law()
                else:
                    self.PARS['gmr'] = sams.components.GMR_Illustris()
                
            elif galaxy_pars_type == 'NG15':
                # note: phi0 and mchar0 are just chosen here to be *roughly* approximate to 
                # peak values from the NG15 posteriors
                self.PARS['gsmf'] = holo.sams.GSMF_Schechter(phi0=-2.0, phiz=-0.60, 
                                                             mchar0_log10=11.5, mcharz=+0.11, 
                                                             alpha0=-1.21, alphaz=-0.03)
                # note: mamp_log10 and scatter_dex are just chosen to be the mean values for the
                # 'astrophysical priors' from NG15, since the posteriors are ultimately prior dominated
                self.PARS['mmbulge'] = host_relations.MMBulge_Standard(mamp_log10=8.6, mplaw=+1.10,
                                                                       scatter_dex=0.32, bulge_mfrac=+0.615)
                if gpf_flag:
                    # note that frac_norm = frac_norm_allq/(1.0-0.25) = +0.033 is the NG15Astro value
                    self.PARS['gpf'] = sams.components.GPF_Power_Law(frac_norm_allq=+0.025, 
                                                                     malpha=0.0, zbeta=+1.0,
                                                                     qgamma=0.0)
                    self.PARS['gmt'] = sams.components.GMT_Power_Law(time_norm=0.5*GYR,
                                                                     malpha=0.0, zbeta=-0.5, 
                                                                     qgamma=-1.0)
                    self.PARS['gmr'] = None
                else:
                    raise ValueError("`galaxy_pars_type`='NG15' requires `gpf_flag`=1.")                    
            
            elif galaxy_pars_type == 'dev-defaults-KH13evol':
                # Will use GMR + double schechter + MMbulge with alpha_z=1.0
                self.PARS['gsmf'] = holo.sams.GSMF_Double_Schechter()
                self.PARS['mmbulge'] = host_relations.MMBulge_Redshift_KH2013(z_plaw=1.0)                
                if gpf_flag:
                    self.PARS['gpf'] = sams.components.GPF_Power_Law()
                    self.PARS['gmt'] = sams.components.GMT_Power_Law()
                else:
                    self.PARS['gmr'] = sams.components.GMR_Illustris()
                
            else: 
                raise ValueError(f'{galaxy_pars_type=} not defined')

            # ---- Set values for individual models
            type0_desc = 'set inner hardening using nu_inner, r_gw_crit_9, and alpha_gw_crit'
            type1_desc = 'set inner hardening using dadt_rchar, r_gw_crit_9, and alpha_gw_crit'
            
            if self.model_type == 'new_hardening_type0_toutvar':
                self.PARS['desc'] = type0_desc
                self.PARS['hard_inner_model_type'] = 0
                self.PARS['hard_outer_time'] = var_value

            elif self.model_type == 'new_hardening_type0_nuivar':
                self.PARS['desc'] = type0_desc
                self.PARS['hard_inner_model_type'] = 0
                self.PARS['hard_nu_inner'] = var_value

            elif self.model_type == 'new_hardening_type0_rgw9var':
                self.PARS['desc'] = type0_desc
                self.PARS['hard_inner_model_type'] = 0
                self.PARS['hard_r_gw_crit_9'] = var_value

            elif self.model_type == 'new_hardening_type0_alphgwvar':
                self.PARS['desc'] = type0_desc
                self.PARS['hard_inner_model_type'] = 0
                self.PARS['hard_alpha_gw_crit'] = var_value

            elif self.model_type == 'new_hardening_type0_betagwvar':
                self.PARS['desc'] = type0_desc
                self.PARS['hard_inner_model_type'] = 0
                self.PARS['hard_beta_gw_crit'] = var_value

            elif self.model_type == 'new_hardening_type0_rch9var':
                self.PARS['desc'] = type0_desc
                self.PARS['hard_inner_model_type'] = 0
                self.PARS['hard_rchar_9'] = var_value

            elif self.model_type == 'new_hardening_type1_toutvar':
                self.PARS['desc'] = type1_desc
                self.PARS['hard_inner_model_type'] = 1
                self.PARS['hard_outer_time'] = var_value

            elif self.model_type == 'new_hardening_type1_dadtvar':
                self.PARS['desc'] = type1_desc
                self.PARS['hard_inner_model_type'] = 1
                self.PARS['hard_dadt_rchar'] = var_value

            elif self.model_type == 'new_hardening_type1_rgw9var':
                self.PARS['desc'] = type1_desc
                self.PARS['hard_inner_model_type'] = 1
                self.PARS['hard_r_gw_crit_9'] = var_value

            elif self.model_type == 'new_hardening_type1_alphgwvar':
                self.PARS['desc'] = type1_desc
                self.PARS['hard_inner_model_type'] = 1
                self.PARS['hard_alpha_gw_crit'] = var_value

            elif self.model_type == 'new_hardening_type1_rch9var':
                self.PARS['desc'] = type1_desc
                self.PARS['hard_inner_model_type'] = 1
                self.PARS['hard_rchar_9'] = var_value
            # ---- End of new hardening models ----
            


        # add other params that will stay the same between model types
        freqs, freqs_edges = utils.pta_freqs()
        self.PARS["freqs"] = freqs
        self.PARS["freqs_edges"] = freqs_edges
        self.PARS["NREALS"] = self.nreals
        self.PARS["NLOUD"] = self.nloud

        print(f"Set SAM params manually for {self.model_type=}.")
    

    def set_sam_params_grid(self, tau=None, ai=None, rc=None, 
                            nin=None, nout=None, mf = None):
        """Create sam and hard instance with specified params. 
        
        Outdated (works only with FixedTime_2PL hardening)."""
        
        if mf == 1:
            _gsmf = holo.sams.GSMF_Schechter()
        elif mf == 2:
            _gsmf = holo.sams.GSMF_Double_Schechter()
        else:
            raise ValueError(f"invalid gsmf_flag: {mf}. must be 1 or 2.")

        self.PARS = dict(
            desc=(f't{tau}ai{ai}rc{rc}nin{nin}nout{nout}gsmf{mf}'),
            hard_time=tau,          # [Gyr]
            hard_sepa_init=ai,     # [pc]
            hard_rchar=rc,        # [pc]
            hard_gamma_inner=nin,
            hard_gamma_outer=nout,
            gsmf = _gsmf
        )
        
        # add other params that will stay the same between model types
        freqs, freqs_edges = utils.pta_freqs()
        self.PARS["freqs"] = freqs
        self.PARS["freqs_edges"] = freqs_edges
        self.PARS["NREALS"] = self.nreals
        self.PARS["NLOUD"] = self.nloud

        print(f"Set SAM param grid elem for {self.PARS['desc']}.")
        

        
def create_sams(nreals=5, nloud=5, gridshape=None,
                fpath=_PATH_DATA, suite_type='grid', hard_type='fixed2PL',
                sparse_param_sweep=False, _gal_pars_type=None,
                _gwb_params=True,
                ai=None, tau=None, gsmf=None, gpfflag=None, _tout_default=None,
                _nuin_default=None, _dadt_default=None, 
                _alphgw_default=None, _betagw_default=None, _rgw9_default=None, 
                _alphch_default=None, _rch9_default=None, 
                _mtrange=None,
                pickle_sams=True, pickle_name=None, pickle_name_extra=None):

    all_sams = []

    if nloud > 0 and not _gwb_params:
        # setting up this code to use function gwb() if nloud>0 and gwb_new() if nloud=0. 
        # gwb_new() is faster but doesnt return info about loudest single sources.
        log.warning("Cannot currently set nloud>0 and _gwb_params=False. "
                    "Overriding to set _gwb_params=True "
                    "(will use slower `gwb()` function intead of `gwb_new()`.")
        _gwb_params = True
    
    if suite_type == 'grid':
        if hard_type != 'fixed2PL':
            raise ValueError(f"only hard_type='fixed2PL' allowed for suite_type='grid'.")
            
        for t in [1.0,3.0]:
            for rc in [10.0, 100.0]:
                for nin in [-2.0, -0.5]:
                    for nout in [0.0, +2.5]:
                        #for mf in [1,2]:

                        s = Test_SAM(hard_type=hard_type, model_type='grid', 
                                     gwb_params=_gwb_params,
                                     nreals=nreals, nloud=nloud, 
                                     hard_t=t, hard_ai=ai, hard_rc=rc, 
                                     hard_nin=nin, hard_nout=nout,
                                     gsmf_flag = gsmf, gpf_flag=gpfflag)
                        all_sams = all_sams + [s]
                            
        if pickle_sams and pickle_name is None:
            gpf_str = ['gmr', 'gpf']
            pickle_name = f'wide_grid_{gpf_str[gpfflag]}_ai{ai}_{gsmf}schech'

    elif suite_type == 'manual':
        test_model_list = [
                           'old',
                           'old_2s', 
                           'old_rc100', 
                           'ph15', 
                           'astr', 
                           'astr_nuo0', 
                           'astr_rc100',
                          ]
    
        for mod in test_model_list:
            if tau is None:
                print(f'WARNING: keyword `tau` set to None. Assigning hard tscale tau=1.0 Gyr.')
                tau = 1.0
                
            print(f'Creating test SAM for model_type {mod} with gpf_flag={gpfflag} & tau={tau}.')
            s = Test_SAM(hard_type=hard_type, model_type=mod, gwb_params=_gwb_params,
                         nreals=nreals, nloud=nloud, gpf_flag=gpfflag, hard_t=tau)

            all_sams = all_sams + [s]
                
        if pickle_sams and pickle_name is None:
            gpf_str = ['gmr', 'gpf']
            pickle_name = f'manual_moddefs_{gpf_str[gpfflag]}_tau{tau}'
            
    elif suite_type == 'old_new_mods_compare':
        test_model_list = [
                           'ph15',
                           'astr_rc100'
                          ]
        if gpfflag is not None:
            print(f"WARNING: overriding keyword {gpfflag=} for {suite_type=}.")
            gpfflag = None
        gpf_list = [1, 0]

        for i,mod in enumerate(test_model_list):
            if tau is None:
                print(f'WARNING: keyword `tau` set to None. Assigning hard tscale tau=5.55 Gyr.')
                tau = 5.55
                
            print(f'Creating test SAM for model_type {mod} with gpf_flag={gpf_list[i]} & tau={tau}.')
            s = Test_SAM(hard_type=hard_type, model_type=mod, gwb_params=_gwb_params,
                         nreals=nreals, nloud=nloud, 
                         gpf_flag=gpf_list[i], hard_t=tau)

            all_sams = all_sams + [s]
                
        if pickle_sams and pickle_name is None:
            pickle_name = f'old_new_mods_compare_tau{tau}'

    elif suite_type == 'model_a_varied':
        # varying tau, nu_inner, z=0 GSMF pars, MMBulge pars
        # fixed nu_outer, rchar, GSMF evol pars
               
        if gpfflag is not None:
            print(f"WARNING: overriding keyword {gpfflag=} for {suite_type=}. Setting to 0.")
        gpfflag = 0
        if tau is not None:
            print(f'WARNING: overriding keyword {tau=}. Varying from 0.1 to 11.0 Gyr.')
            tau = None
            
        varied_values = dict(
            astr_rc100=[0.1, 5.55, 11.0],
            astr_rc100_nuivar=[],
            astr_rc100_muvar=[], 
            astr_rc100_epsmuvar=[],
            astr_rc100_phi10var=[],
            astr_rc100_phi20var=[],
            astr_rc100_Mc0var=[]
        )

        print(f'{gpfflag=}')
        for m in varied_values.keys():
            for i in range(len(varied_values[m])):
                if m=='astr_rc100':
                    print(f'Creating test SAM for model_type {m} w/ tau={tau_vals[i]}.')
                    s = Test_SAM(hard_type=hard_type, model_type=m, gwb_params=_gwb_params,
                                 nreals=nreals, nloud=nloud, 
                                 gpf_flag=gpfflag, hard_t=varied_values[m][i])
                else:
                    print(f'Creating test SAM for model_type {m} w/ {tau_vals[1]=} & {varied_values[m][i]=}.')
                    s = Test_SAM(hard_type=hard_type, model_type=m, gwb_params=_gwb_params,
                                 nreals=nreals, nloud=nloud, 
                                 gpf_flag=gpfflag, var_value=varied_values[m][i])

                all_sams = all_sams + [s]
                
        if pickle_sams and pickle_name is None:
            pickle_name = suite_type
    
    elif suite_type == 'model_b_varied':
        # varying tau, nu_inner, z=0 GSMF pars, GSMF evol pars
        # fixed nu_outer, rchar, MMBulge pars 

        if gpfflag is not None:
            print(f"WARNING: overriding keyword {gpfflag=} for {suite_type=}. Setting to 0.")
        gpfflag = 0    
        if tau is not None:
            print(f'WARNING: overriding keyword {tau=}. Varying from 0.1 to 11.0 Gyr.')
            tau = None
            
        varied_values = dict(
            astr_rc100=[0.1, 5.55, 11.0],
            astr_rc100_nuivar=[],
            astr_rc100_phi10var=[],
            astr_rc100_phi20var=[],
            astr_rc100_Mc0var=[],
            astr_rc100_Mc1var=[],
            astr_rc100_Mc2var=[]
        )

        print(f'{gpfflag=}')
        for m in varied_values.keys():
            for i in range(len(varied_values[m])):
                if m=='astr_rc100':
                    print(f'Creating test SAM for model_type {m} w/ tau={tau_vals[i]}.')
                    s = Test_SAM(hard_type=hard_type, model_type=m, gwb_params=_gwb_params,
                                 nreals=nreals, nloud=nloud, 
                                 gpf_flag=gpfflag, hard_t=varied_values[m][i])
                else:
                    print(f'Creating test SAM for model_type {m} w/ {tau_vals[1]=} & {varied_values[m][i]=}.')
                    s = Test_SAM(hard_type=hard_type, model_type=m, gwb_params=_gwb_params,
                                 nreals=nreals, nloud=nloud, 
                                 gpf_flag=gpfflag, var_value=varied_values[m][i])

                all_sams = all_sams + [s]
                
        if pickle_sams and pickle_name is None:
            pickle_name = suite_type


    elif suite_type == 'phenom15_varied':
        # varying tau, nu_inner, z=0 GSMF pars,  MMBulge pars 
        # fixed nu_outer, rchar, GSMF evol pars
        raise NotImplementedError
        
        test_model_list = [
                           'ph15', 'ph15_nuivar', 'ph15_muvar', 'ph15_epsmuvar', 
                           'ph15_phivar', 'ph15_Mphivar'
                          ]
        if gpfflag is not None:
            print(f"WARNING: overriding keyword {gpfflag=} for {suite_type=}. Setting to 1.")
        gpfflag = 1

    elif suite_type == 'ph15_ng15':
        # set everything to NG15 defualt values
        # and approx posterior values for NG15 varied pars
        # but tau will still be allowed to be set manually here

        test_model_list = ['ph15']
        if not gpfflag:
            log.warning(f"OVERRIDING {gpfflag=} for {suite_type=}. Setting to 1.")
            gpfflag = 1

        for i,mod in enumerate(test_model_list):
            if tau is None:
                log.warning(f'keyword `tau` set to None. Assigning hard tscale tau=5.55 Gyr.')
                tau = 5.55
                
            print(f'Creating test SAM for model_type {mod} with tau={tau}.')
            s = Test_SAM(hard_type=hard_type, model_type=mod, gwb_params=_gwb_params,
                         nreals=nreals, nloud=nloud, 
                         gridshape=gridshape, gpf_flag=gpfflag, galaxy_pars_type=_gal_pars_type, hard_t=tau)
            all_sams = all_sams + [s]
                
        if pickle_sams and pickle_name is None:
            pickle_name = f'ph15_ng15_tau{tau}'
        
    elif 'new_hardening' in suite_type:

        if hard_type != 'fixedOuter':
            raise ValueError(f"{suite_type=} requires hard_type='fixedOuter'.")
            
        #if gpfflag is not None:
        #    print(f"WARNING: overriding keyword {gpfflag=} for {suite_type=}. Setting to 0.")
        #gpfflag = 0    
        print(f'{gpfflag=}')
        if tau is not None:
            print(f'WARNING: overriding keyword {tau=}, setting to `None` for new hardening.')
            tau = None

        if sparse_param_sweep:
            varied_values = dict(
                tout=np.logspace(-1, 1, 3).tolist(),
                nui=np.arange(-1, 3).tolist(),
                dadt=(-1.0*np.logspace(3, 9, 3)).tolist(),
                rgw9=np.logspace(1.5, 3.5, 5).tolist(),
                alphgw=np.linspace(-0.5, 0.0, 3).tolist(),
                betagw=np.linspace(-0.5, 0.5, 3).tolist(),
                rch9=np.logspace(-1.0, 2.0, 4).tolist(),
                alphch=np.linspace(-1.0, 0.0, 4).tolist()
            )
        else:
            varied_values = dict(
                tout=np.logspace(-1, 1, 5).tolist(),
                nui=np.arange(-1, 2.5, 0.5).tolist(),
                dadt=(-1.0*np.logspace(3, 9, 7)).tolist(),
                rgw9=np.logspace(1.25, 3.75, 11).tolist(), #fid model is invalid for <~ 10^1.5Rg, no GWs for >10^3.5Rg.
                alphgw=np.linspace(-0.5, 0.25, 7).tolist(), 
                betagw=np.linspace(-0.5, 0.5, 5).tolist(),
                rch9=np.logspace(-1.0, 2.0, 7).tolist(),
                # includes -2/3 and -1/2:
                alphch=np.linspace(-1.0, 0.0, 7).tolist()
        )
        
        varName = [m for m in varied_values.keys() if m in suite_type]
        if len(varName)==1:         
            varName = varName[0]
        else:
            raise ValueError(f"No unique var name match found for {suite_type=} in {varied_values.keys()=}")
        
        for i in range(len(varied_values[varName])):
            log.info(
                f"Creating test SAM for {suite_type=}, {varName=}, var_value={varied_values[varName][i]}."
                f"Defaults params are: {_tout_default=} {_nuin_default=} {_dadt_default=} "
                f"{_alphgw_default=} {_betagw_default=} {_rgw9_default=} {_alphch_default=} {_rch9_default=}"
            )
            s = Test_SAM(hard_type=hard_type, model_type=suite_type, gwb_params=_gwb_params,
                         nreals=nreals, nloud=nloud, 
                         gridshape=gridshape, gpf_flag=gpfflag, 
                         galaxy_pars_type=_gal_pars_type,
                         tout_default=_tout_default, nuin_default=_nuin_default, dadt_default=_dadt_default, 
                         alphgw_default=_alphgw_default, betagw_default=_betagw_default, rgw9_default=_rgw9_default, 
                         alphch_default=_alphch_default, rch9_default=_rch9_default,
                         var_value=varied_values[varName][i], hard_t=tau,
                         mtrange=_mtrange)

            all_sams = all_sams + [s]

        print(f"{len(all_sams)=}")
        if pickle_sams:
            if pickle_name is None:
                pickle_name = suite_type
            if pickle_name_extra is not None:
                pickle_name += pickle_name_extra

    else:
        raise ValueError(f"{suite_type=} not defined.")

    
    if pickle_sams:
        nfreqs = all_sams[0].PARS['freqs'].size
        if _mtrange is not None:
            pickle_name += f"_M{_mtrange[0]}-{_mtrange[1]}"
        pkl_fname = f"test_sam_nfreqs{nfreqs}_nreals{nreals}_nloud{nloud}_{pickle_name}.pkl"

        print(f"creating pkl file: {pkl_fname}")
        with open(f"{fpath}/{pkl_fname}", "wb") as f:
            pickle.dump(all_sams, f)

    return all_sams 


if __name__ == '__main__':

    raise NotImplementedError("Cannot run `compare_sams.py` from the command line. Only old code lives here.")
    
    suite_type = 'manual'
    #suite_type = 'grid'
    
    if suite_type == 'grid':
    
        # ---- 'GRID' SETUP ----# 
        if len(sys.argv)>3:
            NREALS = int(sys.argv[1])
            NLOUD = int(sys.argv[2])
            GPFFLAG = int(sys.argv[3])
            GSMF = int(sys.argv[4])
            if len(sys.argv)>5:
                print("Too many command line args ({sys.argv}).")
                sys.exit()

        else:
            print("expecting 4 command line args: NREALS, NLOUD, GPFFLAG, GSMF.")
            sys.exit()

        tmp = create_sams(nreals=NREALS, nloud=NLOUD, fpath=_PATH_DATA, 
                          suite_type='grid', ai=1e4, gsmf=GSMF, gpfflag=GPFFLAG,
                          pickle_sams=True, pickle_name=None)

    elif suite_type == 'manual':

        # ---- 'MANUAL' SETUP ----#         
        if len(sys.argv)>3:
            NREALS = int(sys.argv[1])
            NLOUD = int(sys.argv[2])
            GPFFLAG = int(sys.argv[3])
            TAU = float(sys.argv[4])
            if len(sys.argv)>5:
                print("Too many command line args ({sys.argv}).")
                sys.exit()

        else:
            print("expecting 4 command line args: NREALS, NLOUD, GPFFLAG, TAU.")
            sys.exit()

        tmp = create_sams(nreals=NREALS, nloud=NLOUD, fpath=_PATH_DATA, 
                          suite_type='manual', gpfflag=GPFFLAG, tau=TAU,
                          pickle_sams=True, pickle_name=None)

    else:
        raise ValueError("`suite_type` must be 'grid' or 'manual'.")


def load_sams_from_pkl(nloud=1, nreals=10, nfreqs=40, gpf_flag=False, tau=1.0,
                       data_dir=_SIM_MERGER_PATH, subdir=None, fname_type='manual_moddefs'):

    if fname_type=='manual_moddefs':
        samtype = 'gpf' if gpf_flag else 'gmr'
        sam_pkl_fname=f'test_sam_nfreqs{nfreqs}_nreals{nreals}_nloud{nloud}_manual_moddefs_{samtype}_tau{tau}.pkl'
    elif fname_type=='old_new_mods_compare':
        sam_pkl_fname = f'test_sam_nfreqs{nfreqs}_nreals{nreals}_nloud{nloud}_old_new_mods_compare_tau5.55.pkl'
    elif 'new_hardening' in fname_type:
        sam_pkl_fname = f'test_sam_nfreqs{nfreqs}_nreals{nreals}_nloud{nloud}_{fname_type}.pkl'
    elif fname_type=='ph15_ng15':
        sam_pkl_fname = f'test_sam_nfreqs{nfreqs}_nreals{nreals}_nloud{nloud}_{fname_type}_tau{tau}.pkl'
    else:
        raise ValueError(f'{fname_type=} not defined.')

    if subdir is not None:
        fpath = '/'.join((data_dir, subdir, sam_pkl_fname))
    else:
        fpath = '/'.join((data_dir, sam_pkl_fname))
    with open(fpath, "rb") as f:
        print(f'unpickling SAM data: {sam_pkl_fname}')
        sams = pickle.load(f)
        #sam, hard, gwb_new_sam, gwb_sam, freqs, freqs_edges = sam_data
        #sam, hard, gwb_sam, MODEL_PARS = sam_data
   
    #return sam, hard, gwb_new_sam, gwb_sam, freqs, freqs_edges
    #return sam, hard, gwb_sam, MODEL_PARS
    return sams

def calc_sam_dadt_from_pkl(sam, nloud=1, nreals=10, nfreqs=40, 
                           num_steps=100, verbose=False):

    # TO DO: make allowed param check generic once coded up for Fixed_Time_2PL_SAM
    
    #log.info(f"tout={sam.hard._outer_time:.4g} "
    #         f"rchar9={sam.hard._rchar_9:.4g} alpha_char={sam.hard._alpha_char:.4g} "
    #         f"dadt_rchar={sam.hard._dadt_rchar} nuin={sam.hard._nu_inner} "
    #         f"rgw9={sam.hard._r_gw_crit_9:.4g} alpha_gw={sam.hard._alpha_gw_crit:.4g}")
    
    if isinstance(sam.hard, holo.hardening.Fixed_Time_2PL_SAM):
        print(f"before defining radii: {sam.sam.mtot.shape=} {sam.sam.mrat.shape=} {sam.sam.redz.shape=}")
        # () start from the hardening model's initial separation
        rmax = sam.hard._sepa_init
        # (M,) end at the ISCO
        rmin = utils.rad_isco(sam.sam.mtot)
        # rmin = hard._TIME_TOTAL_RMIN * np.ones_like(sam.mtot)
        # Choose steps for each binary, log-spaced between rmin and rmax
        extr = np.log10([rmax * np.ones_like(rmin), rmin])
        radii = np.linspace(0.0, 1.0, num_steps)[np.newaxis, :]
        # (M, X)
        radii = extr[0][:, np.newaxis] + (extr[1] - extr[0])[:, np.newaxis] * radii
        radii = 10.0 ** radii
        # (M, Q, Z, X)
        mt, mr, rz, rads = np.broadcast_arrays(
            sam.sam.mtot[:, np.newaxis, np.newaxis, np.newaxis],
            sam.sam.mrat[np.newaxis, :, np.newaxis, np.newaxis],
            sam.sam.redz[np.newaxis, np.newaxis, :, np.newaxis],
            radii[:, np.newaxis, np.newaxis, :]
        )

        # old: (X, M*Q*Z) --- `Fixed_Time.dadt` will only accept this shape
        # new: (M, Q, Z, X) is shape of input and output arrays
        dadt = sam.hard.dadt(mt, mr, rads)

        return sam.sam, sam.hard, rads, dadt, sam.gwb_sam
        
    elif isinstance(sam.hard, holo.hardening.FixedOuterTime_InnerPL_SAM):

        if np.any(sam.hard._params_allowed==False):
            warn = "Skipping dadt calculation for sam with invalid params."
            log.warning(warn)
            return (None,)*8           
            
        # () start from the hardening model's initial separation
        rmax = sam.hard._rchar_9 * (sam.sam.mtot/(1.0e9*MSOL))**(sam.hard._alpha_char+1)
        # (M,) end at the ISCO
        rmin = utils.rad_isco(sam.sam.mtot)
        # Choose steps for each binary, log-spaced between rmin and rmax
        extr = np.log10([rmax * np.ones_like(rmin), rmin])
        radii = np.linspace(0.0, 1.0, num_steps)[np.newaxis, :]
        # (M, X)
        radii = extr[0][:, np.newaxis] + (extr[1] - extr[0])[:, np.newaxis] * radii
        radii = 10.0 ** radii
        # (M, Q, Z, X)
        mt, mr, rz, rads = np.broadcast_arrays(
            sam.sam.mtot[:, np.newaxis, np.newaxis, np.newaxis],
            sam.sam.mrat[np.newaxis, :, np.newaxis, np.newaxis],
            sam.sam.redz[np.newaxis, np.newaxis, :, np.newaxis],
            radii[:, np.newaxis, np.newaxis, :]
        )

        if verbose:
            print(sam.model_type)
            print(f"{sam.PARS['hard_outer_time']=}")
            print(sam.sam.mtot.shape, sam.sam.mrat.shape, sam.sam.redz.shape)
            #print(sam.gwb_sam[0].shape,sam.gwb_sam[1].shape,sam.gwb_sam[2].shape,sam.gwb_sam[3].shape)
            print(f"{sam.hard._rchar_9=} {sam.hard._rchar_9/PC=}")
            print(rmin.shape, rmin.min(), rmin.max())
            print(np.log10(sam.hard._rchar_9), np.log10(rmin), num_steps)

        dadt, agw_crit, rz_char, rz_final = sam.hard.dadt(mt, mr, rz, rads)

        return sam.sam, sam.hard, rads, dadt, agw_crit, rz_char, rz_final, sam.gwb_sam

    else:
        raise ValueError("`calc_sam_dadt_from_pkl` requires hard_type='2PL' or 'InnerPL'")
        return np.nan


def calc_cumulative_thard(_sam_data, astart, astop, fixedTime='total'):

    if fixedTime not in ['total','outer']:
        raise ValueError(f"keyword `fixedTime` must be 'total' or 'outer'.")

    if fixedTime=='total':
            if len(_sam_data) == 4:
                _sam, _hard, _rads, _dadt = _sam_data
            elif len(_sam_data) == 5:
                _sam, _hard, _rads, _dadt, _gwb = _sam_data
            else:
                raise ValueError(f"sam_data has unexpected length {len(_sam_data)}. must be 4 or 5 for `fixedTime`='total'.")
    else:
        if len(_sam_data) == 7:
            _sam, _hard, _rads, _dadt, _agw_crit, _rz_char, _rz_final = _sam_data
        elif len(_sam_data) == 8:
            _sam, _hard, _rads, _dadt, _agw_crit, _rz_char, _rz_final, _gwb = _sam_data
        else:
            raise ValueError(f"sam_data has unexpected length {len(_sam_data)}. must be 7 or 8 for `fixedTime`='outer'")


    idx_avals = np.where((_rads[0,0,0,:]<=astart)&(_rads[0,0,0,:]>=astop))[0]
    
    _tevo = -utils.trapz_loglog(-1.0 / _dadt[:,:,0,idx_avals], _rads[:,:,0,idx_avals], 
                                axis=2, cumsum=True)
    return _tevo


def calc_total_tau_inner(_hard, mt, mr):

    # mt and mr should both have shape (Nmtot, Nmrat)

    #print(f"{mt.min()/MSOL=} {mt.max()/MSOL=} {mr.min()=} {mr.max()=}")
    m9 = mt / (1.0e9*MSOL)
    eta_norm = mr / np.square(1 + mr) * 4    
    rchar = _hard._rchar_9 * m9**(_hard._alpha_char+1)
    risco = utils.rad_isco(mt)
    if _hard._gw_crit_units=='rg':
        r_gw_crit_9_in_cm = _hard._r_gw_crit_9*utils.gravitational_radius(1.0e9*MSOL)
    else:
        r_gw_crit_9_in_cm = _hard._r_gw_crit_9*PC
    rgw_crit = r_gw_crit_9_in_cm * m9**(_hard._alpha_gw_crit+1) * eta_norm**_hard._beta_gw_crit
    
    gw_const = 16 * NWTG*NWTG*NWTG / (5 * SPLC**5) * eta_norm * mt*mt*mt   

    sepa_arr = np.linspace(risco,rchar,500)
    x = sepa_arr / rgw_crit
    integrand = x*x*x / (1 + x**(4-_hard._nu_inner))
    #print(f"{gw_const.min()=} {gw_const.max()=}")
    #print(f"min/max prefactor: {np.min(rgw_crit**3 / gw_const)} {np.max(rgw_crit**3 / gw_const)}")
    #print(f"min/max integrand: {integrand.min()} {integrand.max()}")
    #print(f"{rgw_crit.min()=} {rgw_crit.max()=}")
    #print(f"sepa_arr min/max: {sepa_arr.min()/PC}[pc] {sepa_arr.max()/PC}[pc], x min/max: {x.min()} {x.max()}")
    #print(f"{mt.shape=} {mr.shape=} {integrand.shape=}")
    tau_inner = (np.square(np.square(rgw_crit)) / gw_const) * np.trapz(integrand, x,axis=0)
    #print(f"{tau_inner.shape=}")
    
    return tau_inner, rgw_crit
    
def calc_aGW_for_Fixed_Time_2PL(_hard, _sam):
    """Calculate (circular) binary separation of transition from 'inner' power-law hardening to GW regime"""

    _mt, _mr = np.broadcast_arrays(
        _sam.mtot[:, np.newaxis],
        _sam.mrat[np.newaxis, :]
    )
    
    _m1, _m2 = utils.m1m2_from_mtmr(_mt, _mr)
    dadt_gw_const = - (64/5.0) * NWTG**3 / SPLC**5 * _m1 * _m2 * _mt
    dadt_innerPL_const = - _hard._norm * _hard._rchar**(_hard._gamma_inner-1)
    
    aGW = ( dadt_gw_const / dadt_innerPL_const ) ** (1.0/(4.0-_hard._gamma_inner))
    
    return aGW

def sepa_emit(mtot, fgw):
    """
    Semimajor axis of a (circular) binary with total mass mtot emitting GWs at frequency fgw
    
    Assumes mtot in cgs units and fgw in Hz, returns separation in cm.
    """
    return ( NWTG * mtot / (fgw * np.pi) **2 )**(1.0/3)

def plot_dadt(sam_data, pars, gwcrit_units='pc', fixedTime='total', 
              max_m_to_plot=4, max_q_to_plot=4, manual_mt_to_plot=None,
              dadt_idx_to_plot=None, nsams_total=None, idx_fiducial=0,
              var_name=None, extra_panels=False, 
              pubstyle=False, twopanel=False, onepanel=False, Tobs_yr = 100.0,
              model_labels=None, cmap_arr=None, cmap_name=None, maxc=1,
              time_ylim=None, rate_ylim=None,
              vary_linestyle=None, vary_lw_for_mass=True, color_by_mass=False,
              show_legends=True, shading=0.7,
              fname_extra='', fpath='', save=False, verbose=False):

    if pubstyle and extra_panels:
        raise ValueError("must set keyword `pubstyle`=False if `extra_panels`=True")

    if twopanel and onepanel:
        raise ValueError("only one of keywords `twopanel` and `onepanel` can be True.")
    
    plt.rc('axes', titlesize=12)     # Title size
    plt.rc('axes', labelsize=12)     # X and Y label size
    plt.rc('xtick', labelsize=10)    # X-axis tick size
    plt.rc('ytick', labelsize=10)    # Y-axis tick size
    plt.rc('legend', fontsize=10)    # Legend size

    tHubble = 13.7e9 * YR

    valid_models = []
    for i_sam,sd in enumerate(sam_data):
        if sd[1] is not None:
            valid_models += [i_sam] 
    if len(valid_models) == 0:
        log.warning("No elements in `sam_data` had valid hardening models. Nothing to plot.")
        return
    if nsams_total is None:
        nsams_total = len(valid_models)
    
    if gwcrit_units == 'pc':
        #xlim=[1e-8,1e5]
        xlim=[10**-6,100]
        xlbl = r'Binary separation $a$ [pc]'
    elif gwcrit_units == 'rg':
        #xlim=[1,1e13]
        if onepanel:
            xlim=[10**0.5,10**6]
        else:
            xlim=[10**0.5,10**9]
        xlbl = r'Binary separation $a$ [${\rm R_g}$]'
    else:
        raise ValueError(f"invalid keyword {gwcrit_units=}. must be 'pc' or 'rg'.")
    
    if fixedTime not in ['total','outer']:
        raise ValueError(f"keyword `fixedTime` must be 'total' or 'outer'.")

    if model_labels is not None and len(model_labels)!=len(sam_data):
        raise ValueError(f"model_labels must be None or equal in length to sam_data.")
        
    # Make the plot
    if extra_panels:
        fig = plt.figure(figsize=(12,7))
        first_plot_index = 231
    else:
        if pubstyle:
            if twopanel:
                fig, (ax_dadt,ax_tau) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=[4.5,5])
                first_plot_index = 211
            elif onepanel:
                fig, ax_dadt = plt.subplots(nrows=1, ncols=1, figsize=[6.5,4])
                first_plot_index = 111                
            else:
                fig, (ax_t,ax_dadt,ax_tau) = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=[4.5,6])
                first_plot_index = 311
        else:
            fig = plt.figure(figsize=(12,4))
            first_plot_index = 131
            

    if pubstyle:
        kwargs = dict(xscale='log', yscale='log', xlim=xlim)
    else:
        kwargs = dict(xscale='log', yscale='log', xlim=xlim, xlabel=xlbl)

    if time_ylim is None:
        time_ylim = [0.15,1e14]
    if rate_ylim is None:
        rate_ylim = [0.15,2e11]

    if not onepanel:
        if not twopanel:
            ax_t.set(ylabel=r'$t_{\rm hard}$ [yr]',ylim=time_ylim, **kwargs)
            ax_t.xaxis.set_inverted(True)
        if pubstyle:
            ax_tau.set(ylabel=r'$\tau_{in+gw}$ [yr]', xlabel=xlbl, 
                    ylim=time_ylim, **kwargs)
        else:
            ax_tau.set(ylabel=r'$\tau_{in+gw}$ [yr]', **kwargs)
        ax_tau.xaxis.set_inverted(True)
        ax_dadt.set(ylabel=r'$|da/dt|$ [cm/s]', ylim=rate_ylim, **kwargs)
    else:
        ax_dadt.set(ylabel=r'$|da/dt|$ [cm/s]', xlabel=xlbl, 
                    ylim=rate_ylim, **kwargs)
    ax_dadt.xaxis.set_inverted(True)

    if extra_panels:
        ax4 = fig.add_subplot(first_plot_index+3, **kwargs)
        ax4.xaxis.set_inverted(True)
        ax4.set(ylabel=r's (hardening parameter) ')

        ax5 = fig.add_subplot(first_plot_index+4)
        ax5.set(xscale='log',yscale='log',xlabel=r'<s_inner> [pc Myr]$^{-1}$',
                ylabel='cumulative time [yr]')

    #if cmap_arr is None:
    #    cm = plot._get_cmap('viridis')
    #    colors = cm(np.linspace(0, 1, nsams_total))
    #    #cm = plot._get_cmap('tab20b')
    #    #cm2 = plot._get_cmap('tab20c')
    #    #cm3 = plot._get_cmap('tab20')
    #    #if len(sam_data)<=14:
    #    #    colors= np.vstack([cm(np.arange(0,20,4)),
    #    #                      cm2(np.arange(0,16,4)),
    #    #                      cm3(np.array([6,12,16,18,10]))]).reshape(14,4)
    #    #elif len(sam_data)>14 and len(sam_data)<=60:
    #    #    colors= np.vstack(cm(np.arange(20)),
    #    #                      cm2(np.arange(20)),
    #    #                      cm3(np.arange(20))).reshape(60,4)
    #    #else:
    #    #    raise ValueError()
    #    #cmap_arr = ['Blues', 'Oranges',  'Greens', 'Reds', 'Purples', 'GnBu',
    #    #            'RdPu', 'YlGnBu', 'YlGn','PuBuGn', 'OrRd', 'PuRd', 'YlOrRd', 'BuPu',
    #    #            'PuBu', 'YlGnBu_r']*5        
    #    #cmap_arr = ['Blues', 'Oranges',  'Greens', 'Reds', 'Purples',
    #    #            'Greys', 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
    #    #            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']*2
    #elif len(cmap)>1:
    #    log.warning(f'using only index 0 element of cmap_arr with length {len(cmap_arr)}.')
    #    cm = plot._get_cmap(cmap_arr[0])
    #    #colors = cmap(np.linspace(0.3, 1, max_to_plot+1))
    #    #colors = cmap(np.linspace(0.2, 1, max_to_plot+1))
    #    colors = cm(np.linspace(0.2, 1, nsams_total))
    #    #lw = np.arange(0.5,max_to_plot+1, 0.6)

    if not isinstance(cmap_arr, list):
        if cmap_arr is None:
            if cmap_name is None:
                cmap_arr = ['viridis']
            else:
                cmap_arr = [cmap_name]
        elif isinstance(cmap_arr, str):
            cmap_arr = [cmap_arr]
        else:
            raise ValueError("keyword `cmap_arr` must be of type None, str, or list.")

    colors = None
    if color_by_mass:
        if len(cmap_arr) == nsams_total:
            colors = np.zeros((nsams_total, max_m_to_plot+1, 4))
            for n in range(nsams_total):
                cm = plot._get_cmap(cmap_arr[n])
                colors[n,::] = cm(np.linspace(0.3, 1, max_m_to_plot+1))
        else:
            if len(cmap_arr) > 1:
                print(f"{cmap_arr=}")
                log.warning(f'using only index 0 element of {cmap_arr=}.')
            cm = plot._get_cmap(cmap_arr[0])
            colors = cm(np.linspace(0, maxc, max_m_to_plot+1))
    else:
        if len(cmap_arr) > 1:
            log.warning(f'using only index 0 element of {cmap_arr=}.')
        cm = plot._get_cmap(cmap_arr[0])
        colors = cm(np.linspace(0, maxc, nsams_total))
            
    
    if dadt_idx_to_plot is not None:
        if len(dadt_idx_to_plot) != len(sam_data):
            raise ValueError("length mismatch: dadt_idx_to_plot and sam_data.")
    else:
        dadt_idx_to_plot = np.arange(len(sam_data))
    if vary_linestyle=='mods':
        ls_arr = ['-',(0, (5, 1)),(0, (3, 1, 1, 1))]*12
    elif vary_linestyle=='q':
        ls_arr = [':','-','-.','--']*9
    elif vary_linestyle==None: 
        ls_arr = ['-']
    else: 
        raise ValueError(f"invalid value for keyword {vary_linestyle=}. must be 'mods', 'q', or None.")
        
    #fgw_ls = ['-','-.',':']
    mlhandles = []
    flhandles = []
    qlhandles = []
    plhandles = []    
    nu_mrk = ['s','^','o','*']

    if vary_lw_for_mass:
        lw = np.arange(1,max_m_to_plot+1, 1)
    else:
        if max_q_to_plot<=3:
            lw = np.arange(1,max_m_to_plot+1, 2)
        else:
            lw = np.arange(0.7,max_q_to_plot+1, 0.5)
            

    if model_labels is None:
        if var_name=='hard_r_gw_crit_9':
            ptitle=r'log$_{10}$(a$_{\rm GW,9}/{\rm R_g})$'
        elif var_name=='hard_nu_inner':
            ptitle = r'$\nu_{\rm in}$'
        elif var_name=='hard_alpha_gw_crit':
            ptitle = r'$\alpha_{\rm GW}$'
        elif var_name=='hard_beta_gw_crit':
            ptitle = r'$\beta_{\rm GW}$'
        elif var_name=='hard_rchar_9':
            ptitle = r'a$_{\rm char,9}$/pc'
        elif var_name=='hard_outer_time':
            ptitle = r'${\rm \tau_{out}/Gyr}$'           
        else:
            raise ValueError(f"invalid value of {var_name=}")
    else:
        ptitle = 'Model type'

    log.info(f"{len(sam_data)=}")
    for n,sd in enumerate(sam_data):
        
        if fixedTime=='total':
            if len(sd) == 4:
                sam, hard, rads, dadt = sd
            elif len(sd) == 5:
                sam, hard, rads, dadt, gwb = sd
            else:
                raise ValueError(f"sam_data has unexpected length {len(sd)}. "
                                 f"must be 4 or 5 for `fixedTime`='total'.")
                
            agw = calc_aGW_for_Fixed_Time_2PL(hard, sam)
            
        else: 
            if sd[1]==None:
                log.warning("skipping dadt plot for NoneType hardening data.")
                continue

            # fixedTime == 'outer'
            if len(sd) == 7:
                sam, hard, rads, dadt, agw, rzch, rzf = sd
            elif len(sd) == 8:
                sam, hard, rads, dadt, agw, rzch, rzf, gwb = sd
            else:
                raise ValueError(f"sam_data has unexpected length {len(sd)}. "
                                 f"must be 7 or 8 for `fixedTime`='outer'")
        
        mt_nskip = int((sam.mtot.size-1)/(max_m_to_plot-1)) if sam.mtot.size>max_m_to_plot else 1
        if manual_mt_to_plot is not None:
            ii = []
            for xx in manual_mt_to_plot:
                diff = np.abs(sam.mtot-10.0**xx*MSOL)
                ii += [np.where(diff==diff.min())[0].item()]
                if xx == 9:
                    idx_9 = np.where(diff==diff.min())[0].item()
            print(f"{ii=}, {sam.mtot[ii]}")
            idx_mt = np.array(ii)
        else:
            idx_mt = np.arange(0,sam.mtot.size,mt_nskip) 

        mr_nskip = int((sam.mrat.size-1)/(max_q_to_plot-1)) if sam.mrat.size>max_q_to_plot else 1
        zord = 2 if (pubstyle and n==valid_models[0]) else 1.9

        times_evo = calc_cumulative_thard(sd, rads[0,0,0,0], rads[0,0,0,-1],fixedTime=fixedTime)
        #times_evo = -utils.trapz_loglog(-1.0 / dadt[:,:,0,:], rads[:,:,0,:], axis=2, cumsum=True)
        if fixedTime=='outer':
            mt, mr, = np.broadcast_arrays(
                sam.mtot[:, np.newaxis],
                sam.mrat[np.newaxis, :]
            )    
            #tau_inner, rgw_crit = calc_total_tau_inner(hard, mt, mr)
            risco = utils.rad_isco(sam.mtot)
            #print(f"min/max tau_inner: {np.min(tau_inner/YR):.4g} {np.max(tau_inner/YR):.4g}")

        
        freqs, freqs_edges = utils.pta_freqs()

        if verbose:
            print(f"*** in plotting function: {rads.min()=} {rads.max()=}")
            print(f"{mt_nskip=}, {mr_nskip=}")
            print(f"{rads[0,0,0,0]=}, {rads[0,0,0,-1]=}")
            print(freqs_edges.min(),freqs_edges.max())
            if n==0:
                print(f"Mtot={sam.mtot/MSOL}, q={sam.mrat}, redz={sam.redz}")
            if fixedTime == 'outer':
                log.info(
                    f"hard pars : tout={hard._outer_time:.4g} "
                    f"dadt={hard._dadt_rchar} nuin={hard._nu_inner} "
                    f"rchar9={hard._rchar_9:.4g} alphch={hard._alpha_char:.4g}  "
                    f"rgw9={hard._r_gw_crit_9:.4g} "
                    f"alphgw={hard._alpha_gw_crit:.4g} betagw={hard._beta_gw_crit:.4g} "
                )

        i_plot = 0
        for i in idx_mt: #np.arange(0,sam.mtot.size,mt_nskip):
            
            if gwcrit_units == 'pc':
                dunits = PC
            elif gwcrit_units == 'rg':
                dunits = NWTG * sam.mtot[i] / SPLC**2
            
            #frst_min = utils.frst_from_fobs(freqs_edges.min(), sam.redz.min())
            frst_min = 1 / (Tobs_yr*YR)
            sepa_obs_max = sepa_emit(sam.mtot[i],frst_min) / dunits

            #ax2.plot([sepa_em,sepa_em],[1e-3,1e11],ls=fgw_ls[fi],color=colors[i])
            #if i_plot==max_to_plot-1 and n==len(sam_data)-1:
            #    flmi,= ax1.plot([sepa_obs_max,sepa_obs_max],[1e-4,1e10],ls=':',
            #                     alpha=0.7,color='k', #lw=0,
            #                     label=r'$f_{\rm obs}$'+f'=1/({Tobs_yr:g} yr)')
                    
            #a_late = sepa_emit(sam.mtot[i], 1.0/(30*YR)) 
            #times_early = calc_cumulative_thard(sd, rads[0,0,0,0], a_late,fixedTime=fixedTime)
            #times_late = calc_cumulative_thard(sd, a_late, rads[0,0,0,-1],fixedTime=fixedTime)
            #ax2.plot([a_late/dunits, a_late/dunits], [1e-3,1e11], color=colors[i_plot], alpha=0.1)

            j_plot=0
            for j in np.arange(0,sam.mrat.size,mr_nskip):
                
                #this_lw=lw[i_plot]+2 if n==idx_fiducial else lw[i_plot]
                
                this_shading = 1.0 if n==idx_fiducial else shading 
                if color_by_mass:
                    this_lw = lw[j_plot]
                    if len(colors.shape)>2:
                        this_color = colors[n,i_plot,:]
                    else:
                        this_color = colors[i_plot]
                else:
                    this_lw = lw[i_plot]
                    this_color = colors[dadt_idx_to_plot[n]]
                if vary_linestyle=='mods':
                    this_ls = ls_arr[n]
                    zord += n*0.05
                elif vary_linestyle=='q':
                    this_ls = ls_arr[j_plot]
                else:
                    this_ls = ls_arr[0]
                
                if j_plot==max_q_to_plot-1 and n==idx_fiducial:
                    if not onepanel:
                        if not twopanel:
                            # linestyle=(0, (3, 1, 1, 1)) is densely dash-dotted
                            flmi, = ax_t.plot([sepa_obs_max,sepa_obs_max],time_ylim, color=this_color,
                                             #alpha=0.7,color=colors[i_plot], lw=2,
                                             alpha=this_shading, lw=this_lw, ls='-',
                                             label=r'$f_{\rm obs}$'+f'=1/({Tobs_yr:g} yr)',zorder=zord-0.2)
                            #if i_plot==max_to_plot-1 and n==len(sam_data)-1:
                            if i_plot==max_m_to_plot-1 and n==idx_fiducial: #valid_models[0]:
                                flhandles += [flmi]
                        ax_tau.plot([sepa_obs_max,sepa_obs_max],time_ylim, color=this_color,
                                     #alpha=0.7,color=colors[i_plot], lw=2,zorder=zord-0.1)
                                     alpha=this_shading,lw=this_lw,ls='-',zorder=zord-0.2)
                    if onepanel and i==idx_9:
                        ax_dadt.plot([sepa_obs_max,sepa_obs_max],rate_ylim,  color='k',
                                     alpha=this_shading,lw=this_lw+3,ls='-',zorder=zord-0.25)
                    ax_dadt.plot([sepa_obs_max,sepa_obs_max],rate_ylim, color=this_color,
                             #alpha=0.7,color=colors[i_plot], lw=2,zorder=zord-0.1)
                             alpha=this_shading,lw=this_lw,ls='-',zorder=zord-0.2)

                
                #ax1.plot(rads[i,j,0,:]/dunits, -rads[i,j,0,:]/dadt[i,j,0,:]/YR, 
                #         alpha=shading, color='k', lw=lw[i_plot]+1, 
                #         label=f'{np.log10(sam.mtot[i]/MSOL):.2g}',zorder=zord)
                if twopanel or onepanel:
                    if j_plot==max_q_to_plot-1 and n==idx_fiducial: ##valid_models[0]: #n==len(sam_data)-1:
                        lm,= ax_dadt.plot(rads[i,j,0,:]/dunits, -dadt[i,j,0,:], 
                                          #alpha=shading, color=colors[i_plot], lw=lw[j_plot], 
                                          alpha=this_shading, color=this_color,  
                                          lw=this_lw, ls=this_ls,
                                          label=f'{np.log10(sam.mtot[i]/MSOL):.2g}',zorder=zord)    
                        mlhandles += [lm]
                    
                else:
                    lm,= ax_t.plot(rads[i,j,0,:]/dunits, -rads[i,j,0,:]/dadt[i,j,0,:]/YR, 
                                   #alpha=shading, color=colors[i_plot], lw=lw[j_plot], 
                                   alpha=this_shading, color=this_color, 
                                   lw=this_lw, ls=this_ls,
                                   label=f'{np.log10(sam.mtot[i]/MSOL):.2g}',zorder=zord)
                                   #label=f'M={sam.mtot[i]/MSOL:.2g}'+r'${\rm M_{\odot}}$')
                    #if j_plot==max_to_plot-1 and n==len(sam_data)-1:
                    if j_plot==max_q_to_plot-1 and n==idx_fiducial: ##valid_models[0]: #n==len(sam_data)-1:
                        mlhandles += [lm]

                if onepanel and i==idx_9:
                    ax_dadt.plot(rads[i,j,0,:]/dunits, -dadt[i,j,0,:],  color='k',
                                 alpha=this_shading,lw=this_lw+3,ls='-',zorder=zord-0.15)
                lq,= ax_dadt.plot(rads[i,j,0,:]/dunits, -dadt[i,j,0,:], 
                              #alpha=shading, color=colors[i_plot], lw=lw[j_plot], 
                              alpha=this_shading, color=this_color,  
                              lw=this_lw, ls=this_ls,
                              label=f'{np.log10(sam.mrat[j]):.2g}',zorder=zord)
                              #label=fr'$q={sam.mrat[j]:.2g}$')
                #if i_plot==max_to_plot-1 and n==len(sam_data)-1:
                if i_plot==max_m_to_plot-1 and n==idx_fiducial:  #valid_models[0]:
                    qlhandles += [lq]

                if var_name is not None:
                    if model_labels is None:
                        if var_name=='hard_r_gw_crit_9':
                            plbl = f"{np.log10(pars[n][var_name]):.2g}"
                        else:
                            plbl = f"{pars[n][var_name]:.2g}"
                    else:
                        plbl = model_labels[n]
                    if not onepanel:
                        lp, = ax_tau.plot(rads[i,j,0,:-1]/dunits, times_evo[i,j,:]/YR, 
                                          #alpha=shading, color=colors[i_plot], lw=lw[j_plot], 
                                          alpha=this_shading, color=this_color, 
                                          lw=this_lw, ls=this_ls,
                                          label = plbl,zorder=zord)
                        if i_plot==max_m_to_plot-1 and j_plot==max_q_to_plot-1:
                            plhandles += [lp]
                    if fixedTime == 'outer':
                        if not twopanel and not onepanel:
                            ax_t.plot(xlim, [(tHubble)/YR,(tHubble)/YR], ls="--",lw=1.2, color='k')
                        if not onepanel:
                            ax_tau.plot(xlim, [(tHubble)/YR,(tHubble)/YR], ls="--",lw=1.2,color='k')
                else:
                    if not onepanel:
                        ax_tau.plot(rads[i,j,0,:-1]/dunits, times_evo[i,j,:]/YR, 
                                    #alpha=shading, color=colors[i_plot], lw=lw[j_plot],zorder=zord)
                                    alpha=shading, color=this_color, 
                                    lw=this_lw, ls=this_ls, zorder=zord)
                
                j_plot += 1

                if extra_panels:
                    hard_param_s = -dadt[i,j,0,:]/rads[i,j,0,:]**2*PC*YR*1.0e6 # 1/(pc*Myr)
                    avg_hard_param_s = hard_param_s.mean()
                
                    if j_plot==max_to_plot and n==len(sam_data)-1:
                        ax4.plot(rads[i,j,0,:]/dunits, hard_param_s, 
                                 #alpha=shading, color=colors[i_plot], lw=lw[j_plot],
                                 alpha=shading, color=this_color, 
                                 lw=this_lw, ls=this_ls,
                                 label=f'mtot={sam.mtot[i]/MSOL:.2g}',zorder=zord)
                    else:
                        ax4.plot(rads[i,j,0,:]/dunits, hard_param_s, 
                                 #alpha=shading, color=colors[i_plot], lw=lw[j_plot],label=None,zorder=zord)
                                 alpha=shading, color=this_color, 
                                 lw=this_lw,ls=this_ls,
                                 label=None,zorder=zord)
                    ax5.scatter(avg_hard_param_s, times_evo[i,j,-1]/YR)

            i_plot += 1
        
        if fixedTime=='total':
            if not twopanel and not onepanel:
                ax_t.plot(xlim, [hard._target_time/YR, hard._target_time/YR], '--', color='darkgray')
                if gwcrit_units=='pc': ax_t.plot([hard._rchar/dunits, hard._rchar/dunits], [1e-2,1e10],'k--')
            if gwcrit_units=='pc': ax_dadt.plot([hard._rchar/dunits, hard._rchar/dunits], [10,1e11],'k--')
            if not onepanel:
                ax_tau.plot(xlim, [tHubble/YR, tHubble/YR], 'k:')
                
    ax_dadt.fill_between(xlim, [SPLC,SPLC], 
                         [rate_ylim[1],rate_ylim[1]], hatch='XXXX', facecolor='none',
                         edgecolor='darkgray',lw=1)
    if show_legends:
        if twopanel or onepanel:
            leg2 = ax_dadt.legend(handles=qlhandles, loc='lower left',title=r"log$_{\rm 10}(q)$",
                                  labelspacing=0.2) #,handlelength=1.5)
            ax_dadt.add_artist(leg2)    
            ax_dadt.legend(handles=mlhandles, loc='upper left',title=r"log$_{\rm 10}(M/M_{\odot})$",
                           labelspacing=0.2) #,handlelength=1.5)
        else:
            #leg2 = ax_t.legend(handles=flhandles, loc='upper left')
            ax_t.legend(handles=mlhandles, loc='lower left',title=r"log$_{\rm 10}(M/M_{\odot})$",
                        labelspacing=0.2) #,handlelength=1.5)
            #ax_t.add_artist(leg2)
            ax_dadt.legend(handles=qlhandles, loc='lower left',title=r"log$_{\rm 10}(q)$",
                           labelspacing=0.2) #,handlelength=1.5)
        if not onepanel:
            ax_tau.legend(handles=plhandles, loc='lower left',title=ptitle,
                          labelspacing=0.2) #,handlelength=1.5)

    if not pubstyle:
        if fixedTime=='total':
            plt.suptitle(f'ai={hard._sepa_init/PC:.2g}pc, '
                         f'rc={hard._rchar/PC:.2g}pc,'
                         f' nu_in={hard._gamma_inner}, nu_out={hard._gamma_outer}\n'
                         f'Mtot=({sam.mtot.min()/MSOL:.2g},{sam.mtot.max()/MSOL:.2g})Msun, '
                         f'q=({sam.mrat.min():.2g},{sam.mrat.max():.2g})')
        else:
            plt.suptitle(fname_extra+"\n"
                         f"tout={hard._outer_time:.2g}, "
                         f"rch9={hard._rchar_9:.2g}pc, "
                         f"alphch={hard._alpha_char:.2g}, "
                         f"rgw9={hard._r_gw_crit_9:.2g}{hard._gw_crit_units}, "
                         f"alphgw={hard._alpha_gw_crit:.2g}, "
                         f"betagw={hard._beta_gw_crit:.2g}, "
                         f"dadt={hard._dadt_rchar}, "
                         f"nuin={hard._nu_inner}")

    if pubstyle:
        if twopanel:
            bottom=0.1
        elif onepanel:
            bottom=0.2
        else:
            bottom=0.08
        fig.subplots_adjust(hspace=0,top=0.98,right=0.98,left=0.15,bottom=bottom) #,top=0.85, right=0.95)
    else:
        fig.subplots_adjust(wspace=0.3,top=0.85, right=0.95)
        
    if save:
        print(f"{fpath=}")
        fig.savefig(f'{fpath}/dadt_{fname_extra}_{gwcrit_units}.png', dpi=300)
    
    return
