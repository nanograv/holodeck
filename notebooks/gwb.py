""" ASSESS THE IMPACT OF CBD MODELS ON GWB """
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import holodeck as holo
from holodeck import utils
from holodeck import plot as hplot
from holodeck.constants import MSOL, PC, YR, MPC, GYR, MYR
import seaborn as sns
palette = sns.color_palette('colorblind') 

def plot(keys, evol_dict, nreals=30, fname='gwb', plot_nanograv23=True, savefig=True, **kwargs):
	fobs, _ = utils.pta_freqs(num=40)

	lw = 5
	figwidth = 20
	figheight = 12
	ticksize = 20
	tickwidth = 4
	alpha = 1.0

	ncols = 1
	nrows = 1
	figwidth *= ncols
	figheight *= nrows

	fs = 60 * (1 + 0.8*(nrows-1))
	# fig, ax = plt.subplots(nrows,ncols,figsize=(figwidth, figheight), sharex='col', sharey='row')

	fig, ax = holo.plot.figax(figsize=[figwidth, figheight])
	ax.grid(True, alpha=0.25)
	xx = fobs #* 1e9

	for key in keys:
		evol = evol_dict[key]
		gwb = holo.gravwaves.GW_Discrete(evol, fobs, nreals=nreals)
		gwb.emit(eccen=True)
		median_gwb = np.median(gwb.both, axis=-1)

		if 'ls_%s' %key in evol_dict.keys():
			ls = evol_dict['ls_%s' %key]
		else:
			ls = '-'

		cc, = ax.plot(xx, median_gwb, label=evol_dict['label_%s' %key], \
						color=evol_dict['color_%s' %key], linewidth=lw, \
						linestyle=ls)
		conf = np.percentile(gwb.both, [25, 75], axis=-1)
		ax.fill_between(xx, *conf, color=cc.get_color(), alpha=0.1)

		twin_ax = hplot._twin_yr(ax, nano=False, fs=fs, label=False)

		plt.setp(twin_ax.get_xticklabels(which='both'), fontsize=fs, rotation=0)
		plt.setp(twin_ax.get_yticklabels(which='both'), fontsize=fs)
		twin_ax.tick_params(axis='both', which='major', direction='inout', size=ticksize, width=tickwidth)
		twin_ax.tick_params(axis='both', which='minor', direction='inout', size=0.7*ticksize, width=0.7*tickwidth)

		plt.setp(ax.get_xticklabels(which='both'), fontsize=fs, rotation=0)
		plt.setp(ax.get_yticklabels(which='both'), fontsize=fs)
		ax.tick_params(axis='both', which='major', direction='inout', size=ticksize, width=tickwidth)
		ax.tick_params(axis='both', which='minor', direction='inout', size=0.7*ticksize, width=0.7*tickwidth)

	if plot_nanograv23:
		f_det = 1./YR
		amp_det = 2.4*10**(-15)
		err_det = np.array([[0.6*10**(-15),0.7*10**(-15)]]).T
		ax.errorbar(f_det, amp_det, yerr=err_det, color = 'royalblue', \
					marker='*', markersize=4*lw, mew=lw, label='NANOGrav\n(2023)')

	twin_ax.set_xlabel(r'$f_{\rm GW} \ [\rm{yr}^{-1}]$', fontsize=fs)
	ax.set_xlabel(r'$f_{\rm GW} \ [\rm{Hz}]$', fontsize=fs)
	if 'leg_fs' in kwargs:
		leg_fs = kwargs['leg_fs']
	else:
		leg_fs = 0.4*fs
	if 'leg_loc' in kwargs:
		leg_loc = kwargs['leg_loc']
	else:
		leg_loc = 'lower center'
	ax.legend(fontsize=leg_fs, ncol=max(1,int(len(keys)/2.)+1), loc=leg_loc)
	fig.tight_layout()
	if savefig:
		plt.savefig('../paper/figures/' + fname + ".png", facecolor='w', bbox_inches='tight')
	else:
		plt.show()

