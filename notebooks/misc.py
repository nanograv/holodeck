#Need py38 environment for this to run

import sys
import os
import sys
import logging
import warnings
import numpy as np
import astropy as ap
import scipy as sp
import scipy.stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import tqdm.notebook as tqdm

import kalepy as kale
import kalepy.utils
import kalepy.plot

import holodeck as holo
#import holodeck.sam
from holodeck import cosmo, utils, plot
from holodeck import hardening as hard
from holodeck import accretion
from holodeck.constants import MSOL, PC, YR, MPC, GYR

def generate_evol(keys, f_edd = 0.1, eccen_init=0.01, nsteps=100, acc_test = False, ecc_test=False, f_edd_test=False, **kwargs):
	# ---- Create initial population
	evol_dict = {}

	for var in list(locals().keys()):
		evol_dict['%s' %var] = locals()['%s' %var]

	pop = holo.population.Pop_Illustris()
	size = pop.size
	eccen = np.ones(size) * eccen_init
	pop = holo.population.Pop_Illustris(eccen=eccen)

	# ---- Apply population modifiers
	redz = cosmo.a_to_z(pop.scafa)

	hards_no_eccen = [
		hard.Hard_GW,
		hard.Sesana_Scattering(),
		hard.Dynamical_Friction_NFW(attenuate=False),
		]

	hards_eccen = [
		hard.Hard_GW,
		hard.CBD_Torques(),
		hard.Sesana_Scattering(),
		hard.Dynamical_Friction_NFW(attenuate=False),
		]

	if ecc_test:
		for key in keys:
			if key == 'no_doteb':
				label = 'no CBD'
				hards = hards_no_eccen
				acc = None
				evol_dict['color_' + key] = sns.color_palette('colorblind')[3]
			if key == 'doteb':
				label = 'CBD (S23a,b)'
				hards = hards_eccen
				acc = accretion.Accretion(accmod='Siwek22', f_edd = f_edd, subpc=True, evol_mass=True)
				evol_dict['color_' + key] = sns.color_palette('colorblind')[2]
			if key == 'no_acc_doteb':
				label = 'CBD torques (S23a,b), \n (no mass evol.)'
				hards = hards_eccen
				acc = accretion.Accretion(accmod='Siwek22', f_edd = f_edd, subpc=True, evol_mass=False)
				evol_dict['color_' + key] = sns.color_palette('colorblind')[1]
			if key == 'acc_no_doteb':
				label = 'CBD accretion, \n (no CBD torques)'
				hards = hards_no_eccen
				acc = accretion.Accretion(accmod='Siwek22', f_edd = f_edd, subpc=True, evol_mass=True)
				evol_dict['color_' + key] = sns.color_palette('colorblind')[4]

			evo = holo.evolution.Evolution(pop, hards, nsteps = nsteps, debug=True, acc=acc)
			evo.evolve()
			evol_dict[key] = evo
			evol_dict['label_%s' %key] = label

		if 'labels' in kwargs:
			labels=kwargs['labels']
			for key,label in zip(keys,labels):
				evol_dict['label_%s' %key] = label

	if acc_test:
		edd_lims = [1,10]
		acc_instance_no_edd_lim = accretion.Accretion(accmod='Siwek22', f_edd = f_edd, subpc=True, edd_lim=None)
		acc_insts = []
		keys = []
		labels = []
		for edd_lim in edd_lims:
			acc_instance_edd_lim = accretion.Accretion(accmod='Siwek22', f_edd = f_edd, subpc=True, edd_lim=edd_lim)
			acc_insts.append(acc_instance_edd_lim)
			keys.append('lim=%d' %edd_lim)
			labels.append(r'$\rm{f_{\rm Edd,2} = } %d$' %edd_lim)
		acc_insts.append(acc_instance_no_edd_lim)
		keys.append('no_lim')
		labels.append(r'$\rm{f_{\rm Edd,2} = \infty}$')

		for key,label,hards,acc in zip(keys,labels,[hards_eccen]*len(acc_insts),acc_insts):
			evo = holo.evolution.Evolution(pop, hards, nsteps = nsteps, debug=True, acc=acc)
			evo.evolve()
			evol_dict[key] = evo
			evol_dict['label_%s' %key] = label

		palette = sns.color_palette('colorblind')
		for i,key in enumerate(keys):
			evol_dict['color_' + key] = palette[i]

	if f_edd_test:
		keys = []
		if 'f_edds' in kwargs:
			f_edds = kwargs['f_edds']
		else:
			f_edds = [1.e-3, 1.e-2, 0.1, 1.0]
		
		for j,f_edd in enumerate(f_edds):
			key = 'f_edd=%.4f' %f_edd
			if 'labels' in kwargs:
				label = kwargs['labels'][j]
			else:
				label = r'$f_{\rm Edd} = %.3f$' %f_edd
			hards = hards_eccen
			acc = accretion.Accretion(accmod='Siwek22', f_edd = f_edd, subpc=True, evol_mass=True)
			evol_dict['color_' + key] = sns.color_palette('colorblind')[j]
			evo = holo.evolution.Evolution(pop, hards, nsteps = nsteps, debug=True, acc=acc)
			evo.evolve()
			evol_dict[key] = evo
			evol_dict['label_%s' %key] = label
			keys.append(key)

	return(keys, evol_dict)