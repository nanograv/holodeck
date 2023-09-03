import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import h5py


from holodeck import plot, detstats
import holodeck.single_sources as sings
from holodeck.constants import YR, MSOL, MPC
import holodeck as holo

import hasasia.sim as hsim



NPSRS = 50
SIGMA = 3.55e-6
NSKIES = 25
THRESH = 0.5
DUR = holo.librarian.DEF_PTA_DUR

def detect_pspace_model(data, dur=DUR,
                        npsrs=NPSRS, sigma=SIGMA, nskies=NSKIES, thresh=THRESH):
    fobs_cents = data['fobs_cents']
    hc_ss = data['hc_ss']
    hc_bg = data['hc_bg']
    dsdata = detstats.detect_pspace_model(fobs_cents, hc_ss, hc_bg, 
                        npsrs, sigma, nskies, thresh)
    return dsdata


def draw_sample_text(fig, params, param_names, 
                     xx=0.1, yy=-0.025, fontsize=10, color='k'):
    text = ''
    for pp, name in enumerate(param_names):
        text = text+"'%s'=%.2e, " % (name, params[name])
        # if pp == int(len(param_names)/2):
        #     text = text+'\n'
    fig.text(xx, yy, text, fontsize=fontsize, color=color, alpha=0.75)


############################################################
#### Draw hc vs par
############################################################

def draw_hc_vs_par(ax, xx_ss=None, yy_ss=None, xx_bg=None, yy_bg=None, color_ss='r', color_bg='k', fast_ss=True, 
                   show_medians = False, show_ci=False, show_reals=True):
    if show_reals:
        if (xx_ss is not None) and (yy_ss is not None):
            if fast_ss:
                ax.scatter(xx_ss.flatten(), yy_ss.flatten(), marker='o', s=15, alpha=0.1, color=color_ss)
            else:
                colors = cm.rainbow(np.linspace(0,1,len(yy_ss[0])))
                for rr in range(len(yy_ss[0])):
                    ax.scatter(xx_ss[:,rr,:].flatten(), yy_ss[:,rr,:].flatten(), marker='o', s=10, alpha=0.1, color=colors[rr])
        if (xx_bg is not None) and (yy_bg is not None):
            ax.scatter(xx_bg.flatten(), yy_bg.flatten(), marker='x', s=15, alpha=0.1, color=color_bg)
    # if show_medians:
    #     if (xx_ss is not None) and (yy_ss is not None):
    #         ax.plot(np.median(xx_ss, axis=())
    #     if (xx_bg is not None) and (yy_bg is not None):




def plot_hc_vs_binpars(fobs_cents, hc_ss, hc_bg, sspar, bgpar, params, param_names, color_ss='r', color_bg='k', fast_ss=True):
    """ plot mtot (0), mrat (1), redz_init (2), dc_final (4), sepa_final(5), angs_final(6)"""
    colors = cm.rainbow(np.linspace(0,1,len(hc_ss[0])))
    idx = [0,1,2,4,5,6]

    labels = sings.par_labels[idx]
    units = sings.par_units[idx]
    xx_ss = sspar[idx]*units[:,np.newaxis,np.newaxis,np.newaxis]
    xx_bg = bgpar[idx]*units[:,np.newaxis,np.newaxis]
    print(f"{xx_ss.shape=}")
    print(f"{xx_bg.shape=}")

    yy_ss = hc_ss
    yy_bg = hc_bg
    print(f"{yy_ss.shape=}")
    print(f"{yy_bg.shape=}")


    fig, axs = holo.plot.figax(
        nrows=2, ncols=3, sharey=True, figsize=(12,6))
    for ax in axs[:,0]:
        ax.set_ylabel(holo.plot.LABEL_CHARACTERISTIC_STRAIN)
    for ii, ax in enumerate(axs.flatten()):
        ax.set_xlabel(labels[ii])
        draw_hc_vs_par(ax, xx_ss[ii], yy_ss, xx_bg[ii], yy_bg, color_ss, color_bg, fast_ss, colors)
        
    draw_sample_text(fig, params, param_names, xx=0.1, yy=-0.05, fontsize=12)
    fig.tight_layout()
    return fig


###################################################################
#### Draw Par vs Frequency
###################################################################

def draw_par_vs_freq(
    ax, xx, xx_ss, yy_ss, xx_bg, yy_bg, 
    color_ss, color_bg, ls_bg='-',
    show_bg_median=True, show_bg_ci=True, 
    show_ss_err=True, show_ss_reals=True):

        if show_bg_median:
            ax.plot(xx, np.median(yy_bg, axis=-1), color=color_bg, linestyle=ls_bg, alpha=0.75)
        if show_bg_ci:
            for pp in [50, 98]:
                conf = np.percentile(yy_bg, [50-pp/2, 50+pp/2], axis=-1)
                ax.fill_between(xx, *conf, color=color_bg, alpha=0.25)
        if show_ss_err:
            ax.errorbar(xx, np.median(yy_ss[:,:,0], axis=-1), yerr=np.std(yy_ss[:,:,0], axis=-1),
                    color=color_ss, alpha=0.5, marker='o', markersize=5, capsize=5)
        if show_ss_reals:
            ax.scatter(xx_ss[...,0].flatten(), yy_ss[:,:,0].flatten(), color=color_ss,  alpha=0.1, s=20)


def draw_snr_vs_freqs(ax, xx, snr_ss, snr_bg, 
             color_ss, color_bg, ls_ss, ls_bg, lw_ss, lw_bg, 
             colors, fast, nfreqs, nreals, nskies, nloudest,):
    
    xx_snr = np.repeat(xx, nreals*nskies*nloudest).reshape(nfreqs, nreals, nskies, nloudest)
    if fast:
        ax.scatter(xx_snr.flatten(), snr_ss.flatten(), marker='o', alpha=0.05, s=5, color=color_ss)
        for rr in range(nreals):
            ax.axhline(snr_bg[rr], ls=ls_bg, lw=lw_bg, alpha=0.1, color=color_bg)
    else:
        xx_skies = np.repeat(xx, nskies).reshape(nfreqs, nskies)
        for rr in range(nreals):
            for ll in range(nloudest):
                edgecolor = 'k' if ll==0 else None
                ax.scatter(xx_skies.flatten(), snr_ss[:,rr,:,ll].flatten(), marker='o', s=10, alpha=0.1, color=colors[rr], edgecolor=edgecolor)
            ax.axhline(snr_bg[rr], ls=ls_bg, lw=lw_bg, alpha=0.1, color=color_bg) 

def draw_dp_vs_freqs(ax, dp_ss, dp_bg,
            color_ss, color_bg, ls_ss, ls_bg, lw_ss, lw_bg,
            nreals, nskies):
    for rr in range(nreals):
        ax.axhline(dp_bg[rr], ls=ls_bg, lw=lw_bg, alpha=0.25, color=color_bg)
        for ss in range(nskies):
            ax.axhline(dp_ss[rr,ss], ls=ls_ss, lw=lw_ss, alpha=0.1, color=color_ss)

def plot_everything_vs_freqs(fobs_cents, hc_ss, hc_bg, sspar, bgpar, dp_ss, dp_bg, snr_ss, snr_bg,
                             params, param_names,
                             color_ss='r', color_bg='k', ls_ss = ':', ls_bg = '--', lw_ss = 2, lw_bg = 2, 
                             fast=True, show_reals=True):
    """ plot mtot (0), mrat (1), redz_init (2), 
    dc_final (4), sepa_final(5), angs_final(6),
    hs, snr, dp"""
    colors = cm.rainbow(np.linspace(0,1,len(hc_ss[0])))
    idx = [0,1,2,4,5,6]
    shape = snr_ss.shape
    nfreqs, nreals, nskies, nloudest = shape[0], shape[1], shape[2], shape[3]
    xx = fobs_cents*YR
    xx_ss = np.repeat(xx, nreals*nloudest).reshape(nfreqs, nreals, nloudest)
    xx_bg = np.repeat(xx, nreals).reshape(nfreqs, nreals)

    labels = np.append(sings.par_labels[idx], 
                       np.array([plot.LABEL_CHARACTERISTIC_STRAIN, 'SNR', 'Detection Probability']))
    units = sings.par_units[idx]
    yy_ss = sspar[idx]*units[:,np.newaxis,np.newaxis,np.newaxis] # shape 6, F,R,L
    yy_ss = np.append(yy_ss, hc_ss).reshape(7, nfreqs, nreals, nloudest) # shape 7, F,R,L
    yy_bg = bgpar[idx]*units[:,np.newaxis,np.newaxis] # shape 6,F,R
    yy_bg = np.append(yy_bg, hc_bg).reshape(7, nfreqs, nreals) # shape 7,F,R
    print(f"{yy_ss.shape=}")
    print(f"{yy_bg.shape=}")

    print(f"{snr_ss.shape=}")
    print(f"{dp_ss.shape=}")


    fig, axs = holo.plot.figax(
        nrows=3, ncols=3, sharex=True, figsize=(15,10))
    for ax in axs[-1]:
        ax.set_xlabel(holo.plot.LABEL_GW_FREQUENCY_YR)

    # plot all pars and hs
    for ii,ax in enumerate(axs.flatten()[:7]):
        print('plotting', labels[ii])
        ax.set_ylabel(labels[ii])
        draw_par_vs_freq(ax, xx, xx_ss, yy_ss[ii], xx_bg, yy_bg[ii], 
                         color_ss, color_bg)
        
    # plot snr, snr_ss = (F,R,L), snr_bg = (R)
    ii=7
    ax = axs.flatten()[ii]
    ax.set_ylabel(labels[ii])
    print('plotting', labels[ii])

    draw_snr_vs_freqs(ax, xx, snr_ss, snr_bg, 
             color_ss, color_bg, ls_ss, ls_bg, lw_ss, lw_bg,
             colors, fast, nfreqs, nreals, nskies, nloudest,)
    
    # plot detection probability, dp_ss = (R,S), dp_bg = (R)
    ii=8
    ax = axs.flatten()[ii]
    ax.set_ylabel(labels[ii])
    print('plotting', labels[ii])

    draw_dp_vs_freqs(ax, dp_ss, dp_bg,
            color_ss, color_bg, ls_ss, ls_bg, lw_ss, lw_bg,
            nreals, nskies)
    
    draw_sample_text(fig, params, param_names, xx=0.1, yy=-0.05, fontsize=12)    
    fig.tight_layout()
    return fig

def plot_everything_vs_freqs_from_data(data, fast=True, color_ss='r', color_bg='k'):
    dsdata = detect_pspace_model(data['fobs_cents'], data['hc_ss'], data['hc_bg'])
    sspar = sings.all_sspars(data['fobs_cents'], data['sspar'])
    fig=plot_everything_vs_freqs(
        data['fobs_cents'], data['hc_ss'], data['hc_bg'], sspar, data['bgpar'],
        dsdata['dp_ss'], dsdata['dp_bg'], dsdata['snr_ss'], dsdata['snr_bg'],
        color_ss=color_ss, color_bg=color_bg, fast=fast, )
    return fig



###################################################################
#### Plot 3 Functions
###################################################################
def draw_everything_model(fig, axs, fobs_cents, hc_ss, hc_bg, sspar, bgpar, dp_ss, dp_bg, snr_ss, snr_bg,
                          color_ss='r', color_bg='k', ls_ss = ':', ls_bg = '--', lw_ss = 2, lw_bg = 2, 
                          fast=True, show_reals=True): 
    colors = cm.rainbow(np.linspace(0,1,len(hc_ss[0])))
    idx = [0,1,2,4,5,6]
    shape = snr_ss.shape
    nfreqs, nreals, nskies, nloudest = shape[0], shape[1], shape[2], shape[3]
    xx = fobs_cents*YR
    xx_ss = np.repeat(xx, nreals*nloudest).reshape(nfreqs, nreals, nloudest)
    xx_bg = np.repeat(xx, nreals).reshape(nfreqs, nreals)

    units = sings.par_units[idx]
    yy_ss = sspar[idx]*units[:,np.newaxis,np.newaxis,np.newaxis] # shape 6, F,R,L
    yy_ss = np.append(yy_ss, hc_ss).reshape(7, nfreqs, nreals, nloudest) # shape 7, F,R,L
    yy_bg = bgpar[idx]*units[:,np.newaxis,np.newaxis] # shape 6,F,R
    yy_bg = np.append(yy_bg, hc_bg).reshape(7, nfreqs, nreals) # shape 7,F,R

    # plot all pars and hs
    for ii,ax in enumerate(axs.flatten()[:7]):
        draw_par_vs_freq(ax, xx, xx_ss, yy_ss[ii], xx_bg, yy_bg[ii], 
                        color_ss, color_bg,)
        
    # plot snr, snr_ss = (F,R,L), snr_bg = (R)
    ii=7
    ax = axs.flatten()[ii]
    draw_snr_vs_freqs(ax, xx, snr_ss, snr_bg, 
            color_ss, color_bg, ls_ss, ls_bg, lw_ss, lw_bg,
            colors, fast, nfreqs, nreals, nskies, nloudest,)
    
    # plot detection probability, dp_ss = (R,S), dp_bg = (R)
    ii=8
    ax = axs.flatten()[ii]
    draw_dp_vs_freqs(ax, dp_ss, dp_bg,
            color_ss, color_bg, ls_ss, ls_bg, lw_ss, lw_bg,
            nreals, nskies)
    
    return fig


def plot_three_models(
        data, params, hard_name, shape, target_param, filename, param_names,
        datcolor_ss = np.array(['limegreen', 'cornflowerblue', 'tomato']),
        datcolor_bg = np.array(['#003300', 'darkblue', 'darkred']),
        datlw = np.array([3,4,5]),
        dattext_yy = np.array([-0.02, -0.05, -0.08]), save_dir=None, save_append=''):
    
    fobs_cents = data[0]['fobs_cents']
    fig, axs = holo.plot.figax(
        nrows=3, ncols=3, sharex=True, figsize=(11.25,7.5))

    idx = [0,1,2,4,5,6]
    labels = np.append(sings.par_labels[idx], 
                        np.array([plot.LABEL_CHARACTERISTIC_STRAIN, 
                                  'SNR', 'Detection Probability']))
        
    for ax in axs[-1]:
        ax.set_xlabel(holo.plot.LABEL_GW_FREQUENCY_YR)
    for ii,ax in enumerate(axs.flatten()):
        ax.set_ylabel(labels[ii])


    for ii, dat in enumerate(data):
        print(f'on dat {ii}')
        dsdat = detect_pspace_model(dat)
        sspar = sings.all_sspars(fobs_cents, dat['sspar'])
        fig = draw_everything_model(fig, axs, fobs_cents, dat['hc_ss'], dat['hc_bg'], 
                                    sspar, dat['bgpar'], dsdat['dp_ss'], dsdat['dp_bg'],
                                    dsdat['snr_ss'], dsdat['snr_bg'], 
                                    color_ss=datcolor_ss[ii], color_bg=datcolor_bg[ii],
                                    lw_bg = datlw[ii], lw_ss = datlw[ii]) 
        draw_sample_text(fig, params[ii], param_names, color=datcolor_bg[ii], 
                         yy=dattext_yy[ii], xx=0, fontsize=12)
    fig.suptitle("%s, %s, Varying '%s'" % (hard_name, str(shape), target_param))
    fig.tight_layout()
    if save_dir is not None:
        str_shape = 's%d_%d_%d' % (shape[0], shape[1], shape[2])
        filename = save_dir+'/%s_allvsfreqs%s_%s.png' % (target_param, save_append, str_shape)  
        fig.savefig(filename, dpi=100, bbox_inches='tight')

    return fig


def draw_hs_vs_binpars(fig, axs, hc_ss, hc_bg, sspar, bgpar, color_ss='r', color_bg='k', fast_ss=True,):
    """ plot mtot (0), mrat (1), redz_init (2), dc_final (4), sepa_final(5), angs_final(6)"""
    colors = cm.rainbow(np.linspace(0,1,len(hc_ss[0])))
    idx = [0,1,2,4,5,6]

    labels = sings.par_labels[idx]
    units = sings.par_units[idx]
    xx_ss = sspar[idx]*units[:,np.newaxis,np.newaxis,np.newaxis]
    xx_bg = bgpar[idx]*units[:,np.newaxis,np.newaxis]

    yy_ss = hc_ss
    yy_bg = hc_bg

    for ax in axs[:,0]:
        ax.set_ylabel(holo.plot.LABEL_CHARACTERISTIC_STRAIN)
    for ii, ax in enumerate(axs.flatten()):
        ax.set_xlabel(labels[ii])
        draw_hc_vs_par(ax, xx_ss[ii], yy_ss, xx_bg[ii], yy_bg, color_ss, color_bg, fast_ss, colors)
        
    return fig

def plot_three_hs_vs_binpars(data, params,
                             hard_name, shape, target_param, filename, param_names,
                             datcolor_ss = np.array(['limegreen', 'cornflowerblue', 'tomato']),
                             datcolor_bg = np.array(['#003300', 'darkblue', 'darkred']),
                             dattext_yy = np.array([-0.02, -0.05, -0.08]), save_dir=None, save_append='',):
    fobs_cents = data[0]['fobs_cents']
    
    fig, axs = holo.plot.figax(
        nrows=2, ncols=3, sharey=True, figsize=(9,4)
        )
    
    for ii, dat in enumerate(data):
        sspar = sings.all_sspars(fobs_cents, dat['sspar'])
        
        draw_hs_vs_binpars(fig, axs, dat['hc_ss'], dat['hc_bg'], sspar, dat['bgpar'], fast_ss=True,
                        color_bg = datcolor_bg[ii], color_ss=datcolor_ss[ii])
        draw_sample_text(fig, params[ii], param_names, color=datcolor_bg[ii], 
                         yy=dattext_yy[ii], xx=0, fontsize=9.5)
    fig.suptitle("%s, %s, Varying '%s'" % (hard_name, str(shape), target_param))
    fig.tight_layout()
    if save_dir is not None:
        str_shape = 's%d_%d_%d' % (shape[0], shape[1], shape[2])
        filename = save_dir+'/%s_hsvsbinpars%s_%s.png' % (target_param, save_append, str_shape)     
        fig.savefig(filename, dpi=100, bbox_inches='tight')

    return fig

# fig = plot_three_hs_vs_binpars(data_hard_time, params_hard_time, 
#                                hard_name, sam.shape, target_param)
