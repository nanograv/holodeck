"""Plotting module.
"""

import numpy as np
# import matplotlib as mpl
import matplotlib.pyplot as plt

import kalepy as kale

from holodeck import cosmo, utils, observations
from holodeck.constants import MSOL, PC, YR


def figax(figsize=[12, 6], ncols=1, nrows=1, sharex=False, sharey=False, squeeze=True,
          scale=None, xscale='log', xlabel='', xlim=None, yscale='log', ylabel='', ylim=None,
          left=None, bottom=None, right=None, top=None, hspace=None, wspace=None,
          widths=None, heights=None, grid=True, **kwargs):

    if scale is not None:
        xscale = scale
        yscale = scale

    scales = [xscale, yscale]
    for ii in range(2):
        if scales[ii].startswith('lin'):
            scales[ii] = 'linear'

    xscale, yscale = scales

    if (widths is not None) or (heights is not None):
        gridspec_kw = dict()
        if widths is not None:
            gridspec_kw['width_ratios'] = widths
        if heights is not None:
            gridspec_kw['height_ratios'] = heights
        kwargs['gridspec_kw'] = gridspec_kw

    fig, axes = plt.subplots(figsize=figsize, squeeze=False, ncols=ncols, nrows=nrows,
                             sharex=sharex, sharey=sharey, **kwargs)

    plt.subplots_adjust(
        left=left, bottom=bottom, right=right, top=top, hspace=hspace, wspace=wspace)

    if ylim is not None:
        shape = (nrows, ncols, 2)
        if np.shape(ylim) == (2,):
            ylim = np.array(ylim)[np.newaxis, np.newaxis, :]
    else:
        shape = (nrows, ncols,)

    ylim = np.broadcast_to(ylim, shape)

    if xlim is not None:
        shape = (nrows, ncols, 2)
        if np.shape(xlim) == (2,):
            xlim = np.array(xlim)[np.newaxis, np.newaxis, :]
    else:
        shape = (nrows, ncols)

    xlim = np.broadcast_to(xlim, shape)
    _, xscale, xlabel = np.broadcast_arrays(axes, xscale, xlabel)
    _, yscale, ylabel = np.broadcast_arrays(axes, yscale, ylabel)

    for idx, ax in np.ndenumerate(axes):
        ax.set(xscale=xscale[idx], xlabel=xlabel[idx], yscale=yscale[idx], ylabel=ylabel[idx])
        if xlim[idx] is not None:
            ax.set_xlim(xlim[idx])
        if ylim[idx] is not None:
            ax.set_ylim(ylim[idx])

        if grid is True:
            ax.set_axisbelow(True)
            ax.grid(True, which='major', axis='both', c='0.6', zorder=2, alpha=0.4)
            ax.grid(True, which='minor', axis='both', c='0.8', zorder=2, alpha=0.4)

    if squeeze:
        axes = np.squeeze(axes)
        if np.ndim(axes) == 0:
            axes = axes[()]

    return fig, axes


def plot_bin_pop(pop):
    mt, mr = utils.mtmr_from_m1m2(pop.mass)
    redz = cosmo.a_to_z(pop.scafa)
    data = [mt/MSOL, mr, pop.sepa/PC, 1+redz]
    data = [np.log10(dd) for dd in data]
    reflect = [None, [None, 0], None, [0, None]]
    labels = [r'M/M_\odot', 'q', r'a/\mathrm{{pc}}', '1+z']
    labels = [r'${{\log_{{10}}}} \left({}\\right)$'.format(ll) for ll in labels]

    if pop.eccen is not None:
        data.append(pop.eccen)
        reflect.append([0.0, 1.0])
        labels.append('e')

    kde = kale.KDE(data, reflect=reflect)
    corner = kale.Corner(kde, labels=labels, figsize=[8, 8])
    corner.plot_data(kde)
    return corner


def plot_mbh_scaling_relations(pop, fname=None, color='r'):
    fig, ax = plt.subplots(figsize=[8, 5])

    #   ====    Plot McConnell+Ma-2013 Data    ====
    handles = []
    names = []
    if fname is not None:
        hh = _draw_mm13_data(ax, fname)
        handles.append(hh)
        names.append('McConnell+Ma')

    #   ====    Plot MBH Merger Data    ====
    hh, nn = _draw_pop_masses(ax, pop, color)
    handles = handles + hh
    names = names + nn
    ax.legend(handles, names)

    return fig


def _draw_mm13_data(ax, fname):
    data = observations.load_mcconnell_ma_2013(fname)
    data = {kk: data[kk] if kk == 'name' else np.log10(data[kk]) for kk in data.keys()}
    key = 'mbulge'
    mass = data['mass']
    yy = mass[:, 1]
    yerr = np.array([yy - mass[:, 0], mass[:, 2] - yy])
    vals = data[key]
    if np.ndim(vals) == 1:
        xx = vals
        xerr = None
    elif vals.shape[1] == 2:
        xx = vals[:, 0]
        xerr = vals[:, 1]
    elif vals.shape[1] == 3:
        xx = vals[:, 1]
        xerr = np.array([xx-vals[:, 0], vals[:, 2]-xx])
    else:
        raise ValueError()

    idx = (xx > 0.0) & (yy > 0.0)
    if xerr is not None:
        xerr = xerr[:, idx]
    ax.errorbar(xx[idx], yy[idx], xerr=xerr, yerr=yerr[:, idx], fmt='none', zorder=10)
    handle = ax.scatter(xx[idx], yy[idx], zorder=10)
    ax.set(ylabel='MBH Mass', xlabel=key)

    return handle


def _draw_pop_masses(ax, pop, color='r', nplot=3e3):
    xx = pop.mbulge.flatten() / MSOL
    yy_list = [pop.mass]
    names = ['new']
    if hasattr(pop, '_mass'):
        yy_list.append(pop._mass)
        names.append('old')

    colors = [color, '0.5']
    handles = []
    if xx.size > nplot:
        cut = np.random.choice(xx.size, int(nplot), replace=False)
        print("Plotting {:.1e}/{:.1e} data-points".format(nplot, xx.size))
    else:
        cut = slice(None)

    for ii, yy in enumerate(yy_list):
        yy = yy.flatten() / MSOL
        data = np.log10([xx[cut], yy[cut]])
        kale.plot.dist2d(
            data, ax=ax, color=colors[ii], hist=False, contour=True,
            median=True, mask_dense=True,
        )
        hh, = plt.plot([], [], color=colors[ii])
        handles.append(hh)

    return handles, names


def _twin_hz(ax, nano=True, fs=12):
    tw = ax.twiny()
    xlim = np.array(ax.get_xlim()) / YR
    if nano:
        label = "nHz"
        xlim *= 1e9
    else:
        label = "Hz"

    label = fr"frequency $[\mathrm{{{label}}}]$"
    tw.set(xlim=xlim, xscale='log')
    tw.set_xlabel(label, fontsize=fs)
    return


def plot_gwb(gwb, color=None, uniform=False, nreals=5):
    fig, ax = plt.subplots(figsize=[10, 5])
    ax.set(xscale='log', xlabel=r'frequency $[\mathrm{yr}^{-1}]$',
           yscale='log', ylabel=r'characteristic strain $[\mathrm{h}_c]$')
    ax.grid(True)

    if uniform:
        color = ax._get_lines.get_next_color()

    _draw_gwb_sample(ax, gwb, color=color, num=nreals)
    _draw_gwb_conf(ax, gwb, color=color)
    _draw_plaw(ax, gwb.freqs*YR, f0=1, color='k')

    _twin_hz(ax, nano=True, fs=12)
    return fig


def _draw_gwb_sample(ax, gwb, num=10, back=True, fore=True, color=None):
    back_flag = back
    fore_flag = fore
    back = gwb.back
    fore = gwb.fore

    freqs = gwb.freqs * YR
    pl = dict(alpha=0.5, color=color, lw=0.8)
    plsel = dict(alpha=0.85, color=color, lw=1.6)
    sc = dict(alpha=0.25, s=20, fc=color, lw=0.0, ec='none')
    scsel = dict(alpha=0.50, s=40, ec='k', fc=color, lw=1.0)

    cut = np.random.choice(back.shape[1], num, replace=False)
    sel = cut[0]
    cut = cut[1:]

    color_gen = None
    color_sel = None
    if back_flag:
        hands_gen = ax.plot(freqs, back[:, cut], **pl)
        hands_sel, = ax.plot(freqs, back[:, sel], **plsel)
        color_gen = [hh.get_color() for hh in hands_gen]
        color_sel = hands_sel.get_color()

    if color is None:
        sc['fc'] = color_gen
        scsel['fc'] = color_sel

    if fore_flag:
        yy = fore[:, cut]
        xx = freqs[:, np.newaxis] * np.ones_like(yy)
        dx = np.diff(freqs)
        dx = np.concatenate([[dx[0]], dx])[:, np.newaxis]

        dx *= 0.2
        xx += np.random.normal(0, dx, np.shape(xx))
        # xx += np.random.uniform(-dx, dx, np.shape(xx))
        xx = np.clip(xx, freqs[0]*0.75, None)
        ax.scatter(xx, yy, **sc)

        yy = fore[:, sel]
        xx = freqs
        ax.scatter(xx, yy, **scsel)

    return


def _draw_gwb_conf(ax, gwb, **kwargs):
    conf = [0.25, 0.50, 0.75]
    freqs = gwb.freqs * YR
    back = gwb.back
    kwargs.setdefault('alpha', 0.5)
    kwargs.setdefault('lw', 0.5)
    conf = np.percentile(back, 100*np.array(conf), axis=-1)
    ax.fill_between(freqs, conf[0], conf[-1], **kwargs)
    kwargs['alpha'] = 1.0 - 0.5*(1.0 - kwargs['alpha'])
    ax.plot(freqs, conf[1], **kwargs)
    return


def _draw_plaw(ax, freqs, amp=1e-15, f0=1/YR, **kwargs):
    kwargs.setdefault('alpha', 0.25)
    kwargs.setdefault('color', 'k')
    kwargs.setdefault('ls', '--')
    plaw = amp * np.power(freqs/f0, -2/3)
    return ax.plot(freqs, plaw, **kwargs)


def plot_evo(evo, freqs):
    fig, ax = plt.subplots(figsize=[10, 5])
    ax.set(xlabel='Obs Orb Freq [1/yr]', xscale='log', yscale='log')
    tw = ax.twiny()
    tw.set(xlim=1e9*np.array([freqs[0], freqs[-1]]), xscale='log', xlabel='Freq [nHz]')

    data = evo.at('fobs', freqs)

    def _draw_vals_conf(ax, xx, name, color=None, units=1.0):
        if color is None:
            color = ax._get_lines.get_next_color()
        nn = name.split(' ')[0]
        vals = data[nn]
        if vals is None:
            return None, None
        ax.set_ylabel(name, color=color)
        ax.tick_params(axis='y', which='both', colors=color)
        vals = np.percentile(vals, [25, 50, 75], axis=0) / units
        h1 = ax.fill_between(xx, vals[0], vals[-1], alpha=0.25, color=color)
        h2, = ax.plot(xx, vals[1], alpha=0.75, lw=2.0, color=color)
        return (h1, h2), nn

    handles = []
    labels = []

    name = 'sepa [pc]'
    hh, nn = _draw_vals_conf(ax, freqs*YR, name, 'blue', units=PC)
    handles.append(hh)
    labels.append(nn)

    name = 'eccen'
    tw = ax.twinx()
    hh, nn = _draw_vals_conf(tw, freqs*YR, name, 'green')
    if hh is not None:
        handles.append(hh)
        labels.append(nn)

    ax.legend(handles, labels)
    return fig
