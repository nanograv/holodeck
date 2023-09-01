from holodeck import utils
import numpy as np
from holodeck.constants import MSOL, PC, YR, MPC, GYR, NWTG, MYR
from holodeck import gravwaves as gw
from holodeck import utils, cosmo
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import weighted

palette = sns.color_palette('colorblind')

xlabels = {'mrat': r'$q_{\rm b}$',\
		   'eccen': r'$e_{\rm b}$', \
		   'mtot': r'$M_{\rm b}/M_{\odot}$', \
           'mchirp': r'$\mathcal{M}/M_{\odot}$',\
		   'sepa': r'$a_{\rm b}/\rm{PC}$',\
		   'fobs': r'$f_{\rm GW}\ [\rm Hz]$',\
		   'redz': r'$z$'}

class evolution_xt:
	""" Create a new evolution object. 
		Use the old evolution object to map Illustris
		population to a representitive 
		sample of binaries in our lightcone, yielding many more binaries """
	def __init__(self, evol,fobs_orb_edges):
		names, samples, vals, weights = evol.sample_universe(fobs_orb_edges)
		#names are the names of the quantities that are interpolated
		# names = ['mtot', 'mrat', 'redz', 'fobs', 'eccen', 'sepa']
		#vals are the values of the above mentioned quantities for all binaries, per frequency bin
		#weights are the _rates_ at which each binary occurs in the light cone.
		#e.g. if the value is ~10, we might expect to see 10 such binaries 
		#at the given frequency bin in the observable Universe.
		#if the value is 1.e-7 (this happens), we hardly see any. 
		#But this is at any given _instance_, i.e. seeing a 'snapshot' of the Universe!
		#In reality, over 5 years observing time, LISA/LSST may see such objects, 
		#but since they move through their frequency bins quickly, we don't see them at a given snapshot.

		name_inds = {
		'mtot': 0,
		'mrat': 1,
		'redz': 2,
		'fobs': 3,
		'eccen': 4,
		'dadt': 5,
		'dedt': 6
		}

		self.vals = vals
		self.name_inds = name_inds
		self.mass = utils.m1m2_from_mtmr(vals[name_inds['mtot']], vals[name_inds['mrat']]).T #needs to be an array of shape (N,2)
		self.mtot = vals[name_inds['mtot']]
		self.mrat = vals[name_inds['mrat']]
		self.fobs = vals[name_inds['fobs']] # in Hz
		self.freq_orb_obs = vals[name_inds['fobs']] # in Hz, duplicate from line above. just adding this because original evolution file uses this name
		self.eccen = vals[name_inds['eccen']]
		self.forb_rest = utils.frst_from_fobs(vals[name_inds['fobs']], vals[name_inds['redz']]) * 1./2.
		self.sepa = ((NWTG * vals[name_inds['mtot']])/(4*np.pi**2 * self.forb_rest**2))**(1./3.) # in cm
		self.weights = weights
		self.norm_weights = self.weights/sum(self.weights)
		self.redz = vals[name_inds['redz']]
		self.scafa = cosmo.z_to_a(self.redz)
		self.tlook = cosmo.z_to_tlbk(vals[name_inds['redz']])
		self.mchirp = utils.chirp_mass(*self.mass.T)
		self.dadt = vals[name_inds['dadt']] 
		self.dedt = vals[name_inds['dedt']]

		#then pass the evolution_xt object into put.bin_evol(keys,evol_dict,param,param_bins,beyond_z0=False,**kwargs)
		#and pass the resulting binary_stats_binned into median_gwb_lsst_lisa.py
		#but need to WEIGHT the bins somehow: pass weights to bin_evol() and bin those too

		#for 'histograms', write new functions
		return

	def prob_func(self, param, param_bins, window_param=None, windows=None, **kwargs):
		""" Returns probability of finding binaries in each param_bin.
			param can be: ['mtot', 'mrat', 'redz', 'fobs', 'eccen', 'sepa']
			Can additionally divide initial data into bin, if windows given 
			window_param can be: ['mtot', 'mrat', 'redz', 'fobs', 'eccen', 'sepa'] """

		if window_param is not None:
			#get boolean arrays for each window
			window_inds = []
			for window in windows:
				w_param = eval("self.%s" %window_param)#self.vals[self.name_inds[window_param]]
				w_ind_lower = w_param >= window[0]
				w_ind_upper = w_param <= window[1]
				w_ind = w_ind_lower & w_ind_upper
				window_inds.append(w_ind)
		else:
			#initialize boolean array with all True values
			w_param = self.vals[0]
			window_inds = []
			w_ind = [True for l in range(len(self.vals[0]))]
			window_inds.append(w_ind)

		""" Initialize return array """
		population_stats = {}
		all_perc = [32,50,68]
		for i,w_ind in enumerate(window_inds):
			for perc in all_perc:
				key_w = 'window%d_perc%d' %(i,perc)
				population_stats[key_w] = []
			key_exp = 'window%d_exp' %i
			population_stats[key_exp] = []
			key_bin = 'window%d_bin' %i
			population_stats[key_bin] = []
			key_bin_widths = 'window%d_logbinwidths' %i
			population_stats[key_bin_widths] = []

		for i,w_ind in enumerate(window_inds):			
			#get binaries that appear in this window
			param_vals = eval("self.%s[w_ind]" %param)#self.vals[self.name_inds[param]][w_ind]
			param_bin_inds = np.digitize(param_vals, param_bins, right=False)
			#expectation value for each binary at a given time, i.e. how many binaries we expect to see in a random snapshot of the Universe
			weights_window = self.weights[w_ind] 

			for j in range(0,len(param_bins)):
				#indices where binaries are in current bin
				ind_bin = np.where(param_bin_inds == j)
				#values in this bin
				param_vals_in_bin = param_vals[ind_bin]
				#weights of binaries in this bin (e.g. how many binaries of each kind we expect)
				weights_in_bin = weights_window[ind_bin]
				weights_in_bin_norm = weights_in_bin/sum(weights_in_bin)

				if len(param_vals_in_bin) >= 1:
					key_bin = 'window%d_bin' %i
					key_bin_widths = 'window%d_logbinwidths' %i
					population_stats[key_bin].append(param_bins[j])
					bin_width_j = np.abs(np.log10(param_bins[j-1]) - np.log10(param_bins[j]))
					population_stats[key_bin_widths].append(bin_width_j)

					for perc in all_perc:
						param_perc = weighted.quantile(param_vals_in_bin, weights_in_bin_norm, perc/100.)
						key_w = 'window%d_perc%d' %(i,perc)
						population_stats[key_w].append(param_perc)

					key_exp = 'window%d_exp' %i
					population_stats[key_exp].append(sum(weights_in_bin))


		return(population_stats)

def adjust_axes(fig, show_grid=False, **kwargs):
	if 'fs' in kwargs:
		fs = kwargs['fs']
	if 'figwidth' in kwargs:
		figwidth = kwargs['figwidth']
	if 'figheight' in kwargs:
		figheight = kwargs['figheight']
	if 'ticksize' in kwargs:
		ticksize = kwargs['ticksize']
	if 'tickwidth' in kwargs:
		tickwidth = kwargs['tickwidth']
	if 'linewidth' in kwargs:
		linewidth = kwargs['linewidth']

	if 'axes' in kwargs:
		axes = kwargs['axes']
	else:
		axes = fig.axes

	""" FORMAT ALL AXES WITH SPECIFIC AXIS TICKS, ADDING GRIDS, AND APPLYTING TIGHT LAYOUT """
	for i, ax in enumerate(axes):
		ax.tick_params(axis='both', which='major', direction='inout', size=ticksize, width=tickwidth)
		ax.tick_params(axis='both', which='minor', direction='inout', size=0.7*ticksize, width=0.7*tickwidth)
		if show_grid:
			ax.grid(which='both', color='k', linestyle='-', linewidth=0.5)
		plt.setp(ax.get_xticklabels(which='both'), fontsize=0.8*fs)
		plt.setp(ax.get_yticklabels(which='both'), fontsize=fs)
		#plt.tight_layout()
		mpl.rcParams['axes.labelsize'] = fs
		mpl.rcParams['axes.titlesize'] = fs

	return(fig)
	""" FORMAT ALL AXES WITH SPECIFIC AXIS TICKS, ADDING GRIDS, AND APPLYING TIGHT LAYOUT """




def plot(keys, evol_dict, params, all_param_bins, fobs_orb_edges, \
		 window_param=None, windows=None, window_labels=None, \
		 all_in_one_ax=False, show_median=False, pdf=True, fname='fig7', savefig=True):

	lw = 5
	figwidth = 20
	figheight = 12
	ticksize = 20
	tickwidth = 4
	alpha = 1.0
	
	# fname += '_binparam'
	# for param in params:
	# 	fname+= '_%s' %(param)
	# fname += '_windowparam_%s' %window_param
	
	ncols = len(params)
	figwidth *= len(params)
	if all_in_one_ax:
		nrows = 1
	else:
		if windows is not None:
			figheight *= len(windows)
			nrows = len(windows)
		else:
			nrows = 1

	fs = 40 * (1 + 0.4*(nrows-1)) * (1 + 0.4*(ncols-1))
	fig, axes = plt.subplots(nrows,ncols,figsize=(figwidth, figheight), sharex='col')
	axes = np.reshape(axes, (nrows,ncols))

	legend_handles = {}

	for key in keys:
		evol = evol_dict[key]
		evol_xt = evolution_xt(evol, fobs_orb_edges)
		color=evol_dict['color_' + key]

		for i,(param,param_bins) in enumerate(zip(params,all_param_bins)):
			pop_stats = evol_xt.prob_func(param, param_bins, \
										 window_param=window_param, \
										 windows=windows)
			if windows is None:
				n_windows = 1
			else:
				n_windows = len(windows)

			for j in range(nrows):
				ax = axes[j][i]
				if i == ncols-1 and ncols > 1:
					ax.yaxis.tick_right()
					ax.yaxis.set_label_position("right")
				if all_in_one_ax:
					# hndls=[]
					for n_window,window_label in zip(range(n_windows),window_labels):
						key_exp = 'window%d_exp' %n_window
						key_bin = 'window%d_bin' %n_window
						key_bin_width = 'window%d_logbinwidths' %n_window
						if param == 'mtot':
							norm_fac = MSOL
						else:
							norm_fac = 1
						if show_median:
							weights_norm = pop_stats[key_exp]/sum(pop_stats[key_exp])
							med = weighted.quantile(np.array(pop_stats[key_bin]),np.array(weights_norm), 0.5)
							#window_label += ', median=%.2e' %med
							ax.axvline(x=med/norm_fac, linestyle='--', \
								linewidth=lw, color=palette[n_window], alpha = alpha)
						if pdf:
							val_to_plot = pop_stats[key_exp]/np.array(pop_stats[key_bin_width])
							ylabel = r'$\mathcal{N}/\delta \log_{10}(\rm bin)$'
						else:
							val_to_plot = pop_stats[key_exp]
							ylabel = r'$\mathcal{N}$'
						
						ax.loglog(np.array(pop_stats[key_bin])/norm_fac, pop_stats[key_exp],\
									label = window_label, \
									color = palette[n_window], \
									linewidth = lw, \
									markersize=4*lw, alpha = alpha)
						weights_norm = np.array(pop_stats[key_exp])/sum(pop_stats[key_exp])
						med = weighted.quantile(np.array(pop_stats[key_bin]),np.array(weights_norm), 0.5)
						# vl, = ax.plot(np.NaN, np.NaN,  '--', label='median=%.2e' %med, linewidth=lw, color=palette[n_window], alpha = alpha)
						# hndls.append(vl)
				else:
					ax.set_title(window_labels[j], fontsize=fs)
					key_exp = 'window%d_exp' %j
					key_bin = 'window%d_bin' %j
					key_bin_width = 'window%d_logbinwidths' %j
					label = evol_dict['label_%s' %key]
					if param == 'mtot':
							norm_fac = MSOL
					else:
						norm_fac = 1
					
					if show_median:
						weights_norm = pop_stats[key_exp]/sum(pop_stats[key_exp])
						med = weighted.quantile(np.array(pop_stats[key_bin]),np.array(weights_norm), 0.5)
						#label += ', median=%.2e' %med
						l = ax.axvline(x = med/norm_fac, linestyle='--', linewidth=lw, color=color, alpha = alpha) #ymin=0, ymax=1, 
					if pdf:
						val_to_plot = pop_stats[key_exp]/np.array(pop_stats[key_bin_width])
						ylabel = r'$\mathcal{N}/\delta \log_{10}(\rm bin)$'
					else:
						val_to_plot = pop_stats[key_exp]
						ylabel = r'$\mathcal{N}$'

					ax.loglog(np.array(pop_stats[key_bin])/norm_fac, val_to_plot,\
								label = label, \
								color=color, linewidth = lw, \
								markersize=4*lw, alpha = alpha)

				if param == params[0]:
					ax.set_ylabel(ylabel, fontsize=1*fs)

			ax.set_xlabel(xlabels[param], fontsize=1*fs)
			if all_in_one_ax:
				if param == params[-1]:
					ax.legend(fontsize=0.8*fs)
			else:
				if show_median:
					if key == keys[-1]:
						ax.plot(np.NaN, np.NaN,  '--', label='median' %med, linewidth=lw, color='gray', alpha = alpha)
				if param == params[-1]:
					ax.legend(fontsize=0.8*fs)

	fig = adjust_axes(fig, show_grid=False, ticksize=ticksize, tickwidth=tickwidth, fs=fs, axes=np.array(axes).flat)
	plt.tight_layout()
	if savefig:
		plt.savefig('../paper/figures/' + fname + ".png", facecolor='w')
	else:
		plt.show()

