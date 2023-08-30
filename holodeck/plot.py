"""Plotting module.

Provides convenience methods for generating standard plots and components using `matplotlib`.

"""

import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import kalepy as kale

import holodeck as holo
from holodeck import cosmo, utils, observations, log
from holodeck.constants import MSOL, PC, YR

FIGSIZE = 6
FONTSIZE = 13
GOLDEN_RATIO = (np.sqrt(5) - 1) / 2

mpl.style.use('default')   # avoid dark backgrounds from dark theme vscode
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.15
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.family"] = "serif"
plt.rcParams["legend.handlelength"] = 1.5
plt.rcParams["lines.solid_capstyle"] = 'round'
# plt.rcParams["font.size"] = FONTSIZE
# plt.rcParams["legend.fontsize"] = FONTSIZE*0.8
# mpl.rcParams['xtick.labelsize'] = FONTSIZE*0.8
# mpl.rcParams['ytick.labelsize'] = FONTSIZE*0.8

LABEL_GW_FREQUENCY_YR = r"GW Frequency $[\mathrm{yr}^{-1}]$"
LABEL_GW_FREQUENCY_HZ = r"GW Frequency $[\mathrm{Hz}]$"
LABEL_GW_FREQUENCY_NHZ = r"GW Frequency $[\mathrm{nHz}]$"
LABEL_SEPARATION_PC = r"Binary Separation $[\mathrm{pc}]$"
LABEL_CHARACTERISTIC_STRAIN = r"GW Characteristic Strain"
LABEL_HARDENING_TIME = r"Hardening Time $[\mathrm{Gyr}]$"
LABEL_CLC0 = r"$C_\ell / C_0$"

PARAM_KEYS = {
    'hard_time': r"phenom $\tau_f$",
    'hard_gamma_inner': r"phenom $\nu_\mathrm{inner}$",
    'hard_gamma_outer': r"phenom $\nu_\mathrm{outer}$",
    'hard_gamma_rot' : r"phenom $\nu_{\mathrm{rot}}$",
    'gsmf_phi0': r"GSMF $\psi_0$",
    'gsmf_mchar0_log10': r"GSMF $m_{\psi,0}$",
    'gsmf_alpha0': r"GSMF $\alpha_{\psi,0}$",
    'gpf_zbeta': r"GPF $\beta_{p,z}$",
    'gpf_qgamma': r"GPF $\gamma_{p,0}$",
    'gmt_norm': r"GMT $T_0$",
    'gmt_zbeta': r"GMT $\beta_{t,z}$",
    'mmb_mamp_log10': r"MMB $\mu$",
    'mmb_plaw': r"MMB $\alpha_{\mu}$",
    'mmb_scatter_dex': r"MMB $\epsilon_{\mu}$",
}

LABEL_DPRATIO = r"$\langle N_\mathrm{SS} \rangle / \mathrm{DP}_\mathrm{BG}$"
LABEL_EVSS = r"$\langle N_\mathrm{SS} \rangle$"
LABEL_DPBG = r"$\mathrm{DP}_\mathrm{BG}$"

COLORS_MPL = plt.rcParams['axes.prop_cycle'].by_key()['color']


class MidpointNormalize(mpl.colors.Normalize):
    """
    Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

    e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
    """

    def __init__(self, vmin=None, vmax=None, midpoint=0.0, clip=False):
        super().__init__(vmin, vmax, clip)
        self.midpoint = midpoint
        return

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))

    def inverse(self, value):
        # x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        y, x = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


class MidpointLogNormalize(mpl.colors.LogNorm):

    def __init__(self, vmin=None, vmax=None, midpoint=0.0, clip=False):
        super().__init__(vmin, vmax, clip)
        self.midpoint = midpoint
        return

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        vals = utils.interp(value, x, y, xlog=True, ylog=False)
        # return np.ma.masked_array(vals, np.isnan(value))
        return vals

    def inverse(self, value):
        y, x = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        vals = utils.interp(value, x, y, xlog=False, ylog=True)
        # return np.ma.masked_array(vals, np.isnan(value))
        return vals


def figax_single(height=None, **kwargs):
    mpl.style.use('default')   # avoid dark backgrounds from dark theme vscode
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.15
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["legend.handlelength"] = 1.5
    plt.rcParams["lines.solid_capstyle"] = 'round'
    plt.rcParams["font.size"] = FONTSIZE
    plt.rcParams["legend.fontsize"] = FONTSIZE*0.8
    mpl.rcParams['xtick.labelsize'] = FONTSIZE*0.8
    mpl.rcParams['ytick.labelsize'] = FONTSIZE*0.8

    if height is None:
        height = FIGSIZE * GOLDEN_RATIO
    figsize_single = [FIGSIZE, height]
    adjust_single = dict(left=0.15, bottom=0.15, right=0.95, top=0.95)

    kwargs.setdefault('figsize', figsize_single)
    for kk, vv in adjust_single.items():
        kwargs.setdefault(kk, vv)

    return figax(**kwargs)


def figax_double(height=None, **kwargs):
    mpl.style.use('default')   # avoid dark backgrounds from dark theme vscode
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.15
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["legend.handlelength"] = 1.5
    plt.rcParams["lines.solid_capstyle"] = 'round'
    plt.rcParams["font.size"] = FONTSIZE
    plt.rcParams["legend.fontsize"] = FONTSIZE*0.8
    mpl.rcParams['xtick.labelsize'] = FONTSIZE*0.8
    mpl.rcParams['ytick.labelsize'] = FONTSIZE*0.8

    if height is None:
        height = 2 * FIGSIZE * GOLDEN_RATIO

    figsize_double = [2*FIGSIZE, height]
    adjust_double = dict(left=0.10, bottom=0.10, right=0.98, top=0.95)

    kwargs.setdefault('figsize', figsize_double)
    for kk, vv in adjust_double.items():
        kwargs.setdefault(kk, vv)

    return figax(**kwargs)


def figax(figsize=[7, 5], ncols=1, nrows=1, sharex=False, sharey=False, squeeze=True,
          scale=None, xscale='log', xlabel='', xlim=None, yscale='log', ylabel='', ylim=None,
          left=None, bottom=None, right=None, top=None, hspace=None, wspace=None,
          widths=None, heights=None, grid=True, **kwargs):
    """Create matplotlib figure and axes instances.

    Convenience function to create fig/axes using `plt.subplots`, and quickly modify standard
    parameters.

    Parameters
    ----------
    figsize : (2,) list, optional
        Figure size in inches.
    ncols : int, optional
        Number of columns of axes.
    nrows : int, optional
        Number of rows of axes.
    sharex : bool, optional
        Share xaxes configuration between axes.
    sharey : bool, optional
        Share yaxes configuration between axes.
    squeeze : bool, optional
        Remove dimensions of length (1,) in the `axes` object.
    scale : [type], optional
        Axes scaling to be applied to all x/y axes.  One of ['log', 'lin'].
    xscale : str, optional
        Axes scaling for xaxes ['log', 'lin'].
    xlabel : str, optional
        Label for xaxes.
    xlim : [type], optional
        Limits for xaxes.
    yscale : str, optional
        Axes scaling for yaxes ['log', 'lin'].
    ylabel : str, optional
        Label for yaxes.
    ylim : [type], optional
        Limits for yaxes.
    left : [type], optional
        Left edge of axes space, set using `plt.subplots_adjust()`, as a fraction of figure.
    bottom : [type], optional
        Bottom edge of axes space, set using `plt.subplots_adjust()`, as a fraction of figure.
    right : [type], optional
        Right edge of axes space, set using `plt.subplots_adjust()`, as a fraction of figure.
    top : [type], optional
        Top edge of axes space, set using `plt.subplots_adjust()`, as a fraction of figure.
    hspace : [type], optional
        Height space between axes if multiple rows are being used.
    wspace : [type], optional
        Width space between axes if multiple columns are being used.
    widths : [type], optional
    heights : [type], optional
    grid : bool, optional
        Add grid lines to axes.

    Returns
    -------
    fig : `matplotlib.figure.Figure`
        New matplotlib figure instance containing axes.
    axes : [ndarray] `matplotlib.axes.Axes`
        New matplotlib axes, either a single instance or an ndarray of axes.

    """

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
            # ax.grid(True, which='major', axis='both', c='0.6', zorder=2, alpha=0.4)
            # ax.grid(True, which='minor', axis='both', c='0.8', zorder=2, alpha=0.4)
            # ax.grid(True, which='major', axis='both', c='0.6', zorder=2, alpha=0.4)
            # ax.grid(True, which='minor', axis='both', c='0.8', zorder=2, alpha=0.4)

    if squeeze:
        axes = np.squeeze(axes)
        if np.ndim(axes) == 0:
            axes = axes[()]

    return fig, axes


def smap(args=[0.0, 1.0], cmap=None, log=False, norm=None, midpoint=None,
         under='0.8', over='0.8', left=None, right=None):
    """Create a colormap from a scalar range to a set of colors.

    Parameters
    ----------
    args : scalar or array_like of scalar
        Range of valid scalar values to normalize with
    cmap : None, str, or ``matplotlib.colors.Colormap`` object
        Colormap to use.
    log : bool
        Logarithmic scaling
    norm : None or `matplotlib.colors.Normalize`
        Normalization to use.
    under : str or `None`
        Color specification for values below range.
    over : str or `None`
        Color specification for values above range.
    left : float {0.0, 1.0} or `None`
        Truncate the left edge of the colormap to this value.
        If `None`, 0.0 used (if `right` is provided).
    right : float {0.0, 1.0} or `None`
        Truncate the right edge of the colormap to this value
        If `None`, 1.0 used (if `left` is provided).

    Returns
    -------
    smap : ``matplotlib.cm.ScalarMappable``
        Scalar mappable object which contains the members:
        `norm`, `cmap`, and the function `to_rgba`.

    """
    # _DEF_CMAP = 'viridis'
    _DEF_CMAP = 'Spectral'

    if cmap is None:
        if midpoint is not None:
            cmap = 'bwr'
        else:
            cmap = _DEF_CMAP

    cmap = _get_cmap(cmap)

    # Select a truncated subsection of the colormap
    if (left is not None) or (right is not None):
        if left is None:
            left = 0.0
        if right is None:
            right = 1.0
        cmap = _cut_cmap(cmap, left, right)

    if under is not None:
        cmap.set_under(under)
    if over is not None:
        cmap.set_over(over)

    if norm is None:
        norm = _get_norm(args, midpoint=midpoint, log=log)
    else:
        log = isinstance(norm, mpl.colors.LogNorm)

    # Create scalar-mappable
    smap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    # Bug-Fix something something
    smap._A = []
    # Allow `smap` to be used to construct colorbars
    smap.set_array([])
    # Store type of mapping
    smap.log = log

    return smap


def _get_norm(data, midpoint=None, log=False):
    """
    """
    # Determine minimum and maximum
    if np.size(data) == 1:
        min = 0
        max = np.int(data) - 1
    elif np.size(data) == 2:
        min, max = data
    else:
        try:
            min, max = utils.minmax(data, filter=log)
        except:
            err = f"Input `data` ({type(data)}) must be an integer, (2,) of scalar, or ndarray of scalar!"
            log.exception(err)
            raise ValueError(err)

    # print(f"{min=}, {max=}")

    # Create normalization
    if log:
        if midpoint is None:
            norm = mpl.colors.LogNorm(vmin=min, vmax=max)
        else:
            norm = MidpointLogNormalize(vmin=min, vmax=max, midpoint=midpoint)
    else:
        if midpoint is None:
            norm = mpl.colors.Normalize(vmin=min, vmax=max)
        else:
            # norm = MidpointNormalize(vmin=min, vmax=max, midpoint=midpoint)
            norm = MidpointNormalize(vmin=min, vmax=max, midpoint=midpoint)
            # norm = mpl.colors.TwoSlopeNorm(vmin=min, vcenter=midpoint, vmax=max)

    return norm


def _cut_cmap(cmap, min=0.0, max=1.0, n=100):
    """Select a truncated subset of the given colormap.

    Code from: http://stackoverflow.com/a/18926541/230468
    """
    name = f"trunc({cmap.name},{min:.2f},{max:.2f})"
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(name, cmap(np.linspace(min, max, n)))
    return new_cmap


def _get_cmap(cmap):
    """Retrieve a colormap with the given name if it is not already a colormap.
    """
    if isinstance(cmap, mpl.colors.Colormap):
        return cmap

    try:
        return mpl.cm.get_cmap(cmap).copy()
    except Exception as err:
        log.error(f"Could not load colormap from `{cmap}` : {err}")
        raise


def _get_hist_steps(xx, yy, yfilter=None):
    """Convert from bin-edges and histogram heights, to specifications for step lines.

    Parameters
    ----------
    xx : array_like
        Independence variable representing bin-edges.  Size (N,)
    yy : array_like
        Dependence variable representing histogram amplitudes.  Size (N-1,)
    yfilter : None, bool, callable

    Returns
    -------
    xnew : array (N,)
        x-values
    ynew : array (N,)
        y-values

    """
    size = len(xx) - 1
    if size != len(yy):
        err = f"Length of `xx` ({len(xx)}) should be length of `yy` ({len(yy)}) + 1!"
        log.exception(err)
        raise ValueError(err)

    xnew = [[xx[ii], xx[ii+1]] for ii in range(xx.size-1)]
    ynew = [[yy[ii], yy[ii]] for ii in range(xx.size-1)]
    xnew = np.array(xnew).flatten()
    ynew = np.array(ynew).flatten()

    if yfilter not in [None, False]:
        if yfilter is True:
            idx = (ynew > 0.0)
        elif callable(yfilter):
            idx = yfilter(ynew)
        else:
            raise ValueError()

        xnew = xnew[idx]
        ynew = ynew[idx]

    return xnew, ynew


def draw_hist_steps(ax, xx, yy, yfilter=None, **kwargs):
    return ax.plot(*_get_hist_steps(xx, yy, yfilter=yfilter), **kwargs)


def draw_gwb(ax, xx, gwb, nsamp=10, color=None, label=None, **kwargs):
    if color is None:
        color = ax._get_lines.get_next_color()

    kw_plot = kwargs.pop('plot', {})
    kw_plot.setdefault('color', color)
    hh = draw_med_conf(ax, xx, gwb, plot=kw_plot, **kwargs)
    if (nsamp is not None) and (nsamp > 0):
        nsamp_max = gwb.shape[1]
        idx = np.random.choice(nsamp_max, np.min([nsamp, nsamp_max]), replace=False)
        for ii in idx:
            ax.plot(xx, gwb[:, ii], color=color, alpha=0.25, lw=1.0, ls='-')

    return hh


def draw_ss_and_gwb(ax, xx, hc_ss, gwb, nsamp=10,
                    color=None, cmap = cm.rainbow,
                    sslabel=None, bglabel=None, **kwargs):
    if color is None:
        color = ax._get_lines.get_next_color()

    kw_plot = kwargs.get('plot', {})
    kw_plot.setdefault('color', color)
    # hh = draw_med_conf(ax, xx, gwb, plot=kw_plot, **kwargs)

    if (nsamp is not None) and (nsamp > 0):
        nsamp_max = gwb.shape[1]
        nsize = np.min([nsamp, nsamp_max])
        colors = cmap(np.linspace(0,1,nsize))
        ci = 0
        idx = np.random.choice(nsamp_max, nsize, replace=False)
        for ii in idx:
            if(ii==0):
                label=bglabel
            else: label=None
            cc = colors[ci] if color is None else color
            ax.plot(xx, gwb[:, ii], color=cc, alpha=0.25, lw=1.0, ls='-')
            for ll in range(len(hc_ss[0,0])):
                if(ll==0):
                    edgecolor='k'
                    if(ii==0): label=sslabel # first source of first realization
                    else: label=None
                else:
                    edgecolor=None
                    label=None
                ax.scatter(xx, hc_ss[:, ii, ll], color=cc, alpha=0.25,
                           edgecolor=edgecolor, label=label)
            ci+=1

    # return hh


def plot_gwb(fobs, gwb, hc_ss=None, bglabel=None, sslabel=None, **kwargs):
    xx = fobs * YR
    fig, ax = figax(
        xlabel=LABEL_GW_FREQUENCY_YR,
        ylabel=LABEL_CHARACTERISTIC_STRAIN
    )
    if(hc_ss is not None):
        draw_ss_and_gwb(ax, xx, hc_ss, gwb, sslabel=sslabel,
                        bglabel=bglabel, **kwargs)
    else:
        draw_gwb(ax, xx, gwb, **kwargs)
    _twin_hz(ax)
    return fig


def plot_bg_ss(fobs, bg, ss=None, bglabel=None, sslabel=None,
             xlabel=LABEL_GW_FREQUENCY_YR, ylabel=LABEL_CHARACTERISTIC_STRAIN, **kwargs):
    """ Can plot strain or power spectral density, just need to set ylabel accordingly
    """
    xx = fobs * YR
    fig, ax = figax(
        xlabel=xlabel,
        ylabel=ylabel
    )
    if(ss is not None):
        draw_ss_and_gwb(ax, xx, ss, bg, sslabel=sslabel,
                        bglabel=bglabel, **kwargs)
    else:
        draw_gwb(ax, xx, bg, **kwargs)
    _twin_hz(ax)
    return fig


def draw_sspars_and_bgpars(axs, xx, sspar, bgpar, nsamp=10, cmap=cm.rainbow_r, color = None, label=None, **kwargs):
    # if color is None:
    #     color = axs[0,0]._get_lines.get_next_color()

    # kw_plot = kwargs.get('plot', {})
    # kw_plot.setdefault('color', color)


    m_bg = bgpar[0,:,:]/MSOL # bg avg masses in solar masses
    m_ss = sspar[0,:,:,:]/MSOL # ss masses in solar masses
    # mm_med = draw_med_conf(axs[0,0], xx, m_bg, plot=kw_plot, **kwargs)

    q_bg = bgpar[1,:,:] # bg avg ratios
    q_ss = sspar[1,:,:,:] # ss ratios
    # qq_med = draw_med_conf(axs[0,1], xx, q_bg, plot=kw_plot, **kwargs)

    di_bg = holo.cosmo.comoving_distance(bgpar[2,:,:]).value # bg avg distances in Mpc
    di_ss = holo.cosmo.comoving_distance(sspar[2,:,:,:]).value # ss distances in Mpc


    df_bg = holo.cosmo.comoving_distance(bgpar[3,:,:]).value # bg avg distances in Mpc
    df_ss = holo.cosmo.comoving_distance(sspar[3,:,:,:]).value # ss distances in Mpc
    # dd_med = draw_med_conf(axs[1,0], xx, d_bg, plot=kw_plot, **kwargs)

    # hh_med = draw_med_conf(axs[1,1], xx, hc_bg, plot=kw_plot, **kwargs)

    if (nsamp is not None) and (nsamp > 0):
        nsamp_max = bgpar.shape[2]
        nsize = np.min([nsamp, nsamp_max])
        colors = cmap(np.linspace(0,1,nsize))
        ci = 0
        idx = np.random.choice(nsamp_max, nsize, replace=False)
        for ii in idx:
            # background
            axs[0,0].plot(xx, m_bg[:,ii], color=colors[ci], alpha=0.25, lw=1.0, ls='-') # masses (upper left)
            axs[0,1].plot(xx, q_bg[:,ii], color=colors[ci], alpha=0.25, lw=1.0, ls='-') # ratios (upper right)
            axs[1,0].plot(xx, di_bg[:,ii], color=colors[ci], alpha=0.25, lw=1.0, ls='-') # initial distances (lower left)
            axs[1,1].plot(xx, df_bg[:, ii], color=colors[ci], alpha=0.25, lw=1.0, ls='-') # final distances (lower right)

            # single sources
            for ll in range(sspar.shape[-1]):
                if(ll==0): edgecolor='k'
                else: edgecolor=None
                axs[0,0].scatter(xx, m_ss[:, ii, ll], color=colors[ci], alpha=0.25,
                           edgecolor=edgecolor) # ss masses (upper left)
                axs[0,1].scatter(xx, q_ss[:, ii, ll], color=colors[ci], alpha=0.25,
                           edgecolor=edgecolor) # ss ratios (upper right)
                axs[1,0].scatter(xx, di_ss[:, ii, ll], color=colors[ci], alpha=0.25,
                           edgecolor=edgecolor) # ss intial distances (lower left)
                axs[1,1].scatter(xx, df_ss[:, ii, ll], color=colors[ci], alpha=0.25,
                           edgecolor=edgecolor) # ss final distances (lower left)
            ci +=1
    # return mm_med, qq_med, dd_med, hh_med


def plot_pars(fobs, sspar, bgpar, **kwargs):
    xx= fobs * YR
    fig, axs = figax(figsize = (11,6), ncols=2, nrows=2, sharex = True)
    axs[0,0].set_ylabel('Total Mass $M/M_\odot$')
    axs[0,1].set_ylabel('Mass Ratio $q$')
    axs[1,0].set_ylabel('Initial Comoving Distance $d_c$ (Mpc)')
    axs[1,1].set_ylabel('Final Comoving Distance $d_c$ (Mpc)')

    axs[1,0].set_xlabel(LABEL_GW_FREQUENCY_YR)
    axs[1,1].set_xlabel(LABEL_GW_FREQUENCY_YR)
    draw_sspars_and_bgpars(axs, xx, sspar, bgpar, color='pink')
    fig.tight_layout()
    return fig


def scientific_notation(val, man=1, exp=0, dollar=True):
    """Convert a scalar into a string with scientific notation (latex formatted).

    Arguments
    ---------
    val : scalar
        Numerical value to convert.
    man : int or `None`
        Precision of the mantissa (decimal points); or `None` for omit mantissa.
    exp : int or `None`
        Precision of the exponent (decimal points); or `None` for omit exponent.
    dollar : bool
        Include dollar-signs ('$') around returned expression.

    Returns
    -------
    rv_str : str
        Scientific notation string using latex formatting.

    """
    if val == 0.0:
        rv_str = "$"*dollar + "0.0" + "$"*dollar
        return rv_str

    # get log10 exponent
    val_exp = np.floor(np.log10(np.fabs(val)))
    # get mantissa (positive/negative is still included here)
    val_man = val / np.power(10.0, val_exp)

    val_man = np.around(val_man, man)
    if val_man >= 10.0:
        val_man /= 10.0
        val_exp += 1

    # Construct Mantissa String
    # --------------------------------
    str_man = "{0:.{1:d}f}".format(val_man, man)

    # If the mantissa is '1' (or '1.0' or '1.00' etc), dont write it
    if str_man == "{0:.{1:d}f}".format(1.0, man):
        str_man = ""

    # Construct Exponent String
    # --------------------------------
    str_exp = "10^{{ {:d} }}".format(int(val_exp))

    # Put them together
    # --------------------------------
    rv_str = "$"*dollar + str_man
    if len(str_man) and len(str_exp):
        rv_str += " \\times"
    rv_str += str_exp + "$"*dollar

    return rv_str


def _draw_plaw(ax, freqs, amp=1e-15, f0=1/YR, **kwargs):
    kwargs.setdefault('alpha', 0.25)
    kwargs.setdefault('color', 'k')
    kwargs.setdefault('ls', '--')
    plaw = amp * np.power(np.asarray(freqs)/f0, -2/3)
    return ax.plot(freqs, plaw, **kwargs)


def _twin_hz(ax, nano=True, fs=10, **kw):
    tw = ax.twiny()
    tw.grid(False)
    xlim = np.array(ax.get_xlim()) / YR
    if nano:
        label = LABEL_GW_FREQUENCY_NHZ
        xlim *= 1e9
    else:
        label = LABEL_GW_FREQUENCY_HZ

    tw.set(xlim=xlim, xscale=ax.get_xscale())
    tw.set_xlabel(label, fontsize=fs, **kw)
    return tw


def _twin_yr(ax, nano=True, fs=10, label=True, **kw):
    tw = ax.twiny()
    tw.grid(False)
    xlim = np.array(ax.get_xlim()) * YR
    if nano:
        xlim /= 1e9

    tw.set(xlim=xlim, xscale=ax.get_xscale())
    if label:
        tw.set_xlabel(LABEL_GW_FREQUENCY_YR, fontsize=fs, **kw)
    return tw


def draw_med_conf(ax, xx, vals, fracs=[0.50, 0.90], weights=None, plot={}, fill={}, filter=False):
    plot.setdefault('alpha', 0.75)
    fill.setdefault('alpha', 0.2)
    percs = np.atleast_1d(fracs)
    assert np.all((0.0 <= percs) & (percs <= 1.0))

    # center the target percentages into pairs around 50%, e.g.  68 ==> [16,84]
    inter_percs = [[0.5-pp/2, 0.5+pp/2] for pp in percs]
    # Add the median value (50%)
    inter_percs = [0.5, ] + np.concatenate(inter_percs).tolist()
    # Get percentiles; they go along the last axis
    if filter:
        rv = [
            kale.utils.quantiles(vv[vv > 0.0], percs=inter_percs, weights=weights)
            for vv in vals
        ]
        rv = np.asarray(rv)
    else:
        rv = kale.utils.quantiles(vals, percs=inter_percs, weights=weights, axis=-1)

    med, *conf = rv.T
    # plot median
    hh, = ax.plot(xx, med, **plot)

    # Reshape confidence intervals to nice plotting shape
    # 2*P, X ==> (P, 2, X)
    conf = np.array(conf).reshape(len(percs), 2, xx.size)

    kw = dict(color=hh.get_color())
    kw.update(fill)
    fill = kw

    # plot each confidence interval
    for lo, hi in conf:
        gg = ax.fill_between(xx, lo, hi, **fill)

    return (hh, gg)

def draw_med_conf_color(ax, xx, vals, fracs=[0.50, 0.90], weights=None, plot={}, fill={}, 
                        filter=False, color=None, linestyle='-'):
    plot.setdefault('alpha', 0.75)
    fill.setdefault('alpha', 0.2)
    percs = np.atleast_1d(fracs)
    assert np.all((0.0 <= percs) & (percs <= 1.0))

    # center the target percentages into pairs around 50%, e.g.  68 ==> [16,84]
    inter_percs = [[0.5-pp/2, 0.5+pp/2] for pp in percs]
    # Add the median value (50%)
    inter_percs = [0.5, ] + np.concatenate(inter_percs).tolist()
    # Get percentiles; they go along the last axis
    if filter:
        rv = [
            kale.utils.quantiles(vv[vv > 0.0], percs=inter_percs, weights=weights)
            for vv in vals
        ]
        rv = np.asarray(rv)
    else:
        rv = kale.utils.quantiles(vals, percs=inter_percs, weights=weights, axis=-1)

    med, *conf = rv.T
    
    # plot median
    if color is not None:
        hh, = ax.plot(xx, med, color=color, linestyle=linestyle, **plot)
    else:
        hh, = ax.plot(xx, med, **plot)

    # Reshape confidence intervals to nice plotting shape
    # 2*P, X ==> (P, 2, X)
    conf = np.array(conf).reshape(len(percs), 2, xx.size)

    kw = dict(color=hh.get_color())
    kw.update(fill)
    fill = kw

    # plot each confidence interval
    for lo, hi in conf:
        gg = ax.fill_between(xx, lo, hi, **fill)

    return (hh, gg)


def smooth_spectra(xx, gwb, smooth=(20, 4), interp=100):
    assert np.shape(xx) == (np.shape(gwb)[0],)

    if len(smooth) != 2:
        err = f"{smooth=} must be a (2,) of float specifying the filter-window size and polynomial-order!!"
        raise ValueError(err)

    xnew = kale.utils.spacing(xx, 'log', num=int(interp))
    rv = [utils.interp(xnew, xx, vv) for vv in gwb.T]
    rv = sp.signal.savgol_filter(rv, *smooth, axis=-1)

    med, *conf = rv

    # Reshape confidence intervals to nice plotting shape
    # 2*P, X ==> (P, 2, X)
    npercs = np.shape(conf)[0] // 2
    conf = np.array(conf).reshape(npercs, 2, xnew.size)
    return xnew, med, conf


def get_med_conf(vals, fracs, weights=None, axis=-1):
    percs = np.atleast_1d(fracs)
    assert np.all((0.0 <= percs) & (percs <= 1.0))

    # center the target percentages into pairs around 50%, e.g.  68 ==> [16,84]
    inter_percs = [[0.5-pp/2, 0.5+pp/2] for pp in percs]
    # Add the median value (50%)
    inter_percs = [0.5, ] + np.concatenate(inter_percs).tolist()
    # Get percentiles; they go along the last axis
    rv = kale.utils.quantiles(vals, percs=inter_percs, weights=weights, axis=axis)
    return rv


def draw_smooth_med_conf(ax, xx, vals, smooth=(10, 4), interp=100, fracs=[0.50, 0.90], weights=None, plot={}, fill={}):
    plot.setdefault('alpha', 0.5)
    fill.setdefault('alpha', 0.2)

    rv = get_med_conf(vals, fracs, weights, axis=-1)
    xnew, med, conf = smooth_spectra(xx, rv, smooth=smooth, interp=interp)

    # plot median
    hh, = ax.plot(xnew, med, **plot)

    # plot each confidence interval
    for lo, hi in conf:
        gg = ax.fill_between(xnew, lo, hi, color=hh.get_color(), **fill)

    return (hh, gg)


def violins(ax, xx, yy, zz, width, **kwargs):
    assert np.ndim(xx) == 1
    if np.ndim(yy) == 1:
        yy = [yy] * len(xx)

    assert np.ndim(yy) == 2
    assert np.shape(yy) == np.shape(zz)
    if np.shape(yy)[0] != xx.size:
        if np.shape(yy)[1] == xx.size:
            yy = yy.T
            zz = zz.T
    assert np.shape(xx)[0] == xx.size
    assert np.shape(zz)[0] == xx.size

    for ii in range(xx.size):
        usey = yy[ii]
        usez = zz[ii]
        handle = violin(ax, xx[ii], usey, usez, width, **kwargs)

    return handle


def violin(ax, xx, yy, zz, width, median_log10=False, side='both', clip_pdf=None,
           median={}, line={}, fill={}, **kwargs):
    assert np.ndim(xx) == 0
    assert np.shape(xx) == np.shape(width)
    assert np.ndim(yy) == 1
    assert yy.shape == zz.shape
    valid_sides = ['l', 'r', 'b']
    if side[0] not in valid_sides:
        raise ValueError(f"{side=} must begin with one of {valid_sides}!")

    if line is not None:
        line_def = dict(alpha=0.5, lw=0.5, color='k')
        line_def.update(kwargs)
        line_def.update(line)
        line = line_def

    if fill is not None:
        fill_def = dict(alpha=0.25, lw=0.0)
        fill_def.update(kwargs)
        fill_def.update(fill)
        fill = fill_def

    if clip_pdf is not None:
        assert np.ndim(clip_pdf) == 0
        assert clip_pdf < 1.0

    zz = zz / zz.max()

    if median is True:
        median = {}
    if median is False:
        median = None

    if median is not None:
        if median_log10:
            dy = np.diff(np.log10(yy))
        else:
            dy = np.diff(yy)
        cdf = 0.5 * (zz[1:] + zz[:-1]) * dy
        cdf = np.concatenate([[0.0, ], cdf])
        cdf = np.cumsum(cdf)
        med = np.interp([0.5], cdf/cdf.max(), yy)

    if clip_pdf is not None:
        idx = zz > clip_pdf
        yy = yy[idx]
        zz = zz[idx]

    xl = xx * np.ones_like(yy)
    xr = xx * np.ones_like(yy)
    left_flag = side.startswith('l') or side.startswith('b')
    right_flag = side.startswith('r') or side.startswith('b')
    if left_flag:
        xl = xl - zz * width
    if right_flag:
        xr = xr + zz * width

    handle = []
    if line is not None:
        h1, = ax.plot(xl, yy, **line)
        ax.plot(xr, yy, **line)
        handle.append(h1)

    if fill is not None:
        h2 = ax.fill_betweenx(yy, xl, xr, **fill)
        handle.append(h2)

    if median is not None:
        kwargs = dict(line)
        kwargs['lw'] = 1.0
        kwargs.update(median)
        mwid = kwargs.pop('width', 0.5)
        ll = xx
        rr = xx
        if left_flag:
            ll = ll - width * mwid
        if right_flag:
            rr = rr + width * mwid
        ax.plot([ll, rr], [med, med], **kwargs)

    handle = handle[0] if len(handle) == 1 else tuple(handle)
    return handle


class Corner:

    _LIMITS_STRETCH = 0.1

    def __init__(self, ndim, origin='tl', rotate=True, axes=None, labels=None, limits=None, **kwargs):

        origin = kale.plot._parse_origin(origin)

        # -- Construct figure and axes
        if axes is None:
            fig, axes = kale.plot._figax(ndim, **kwargs)
            self.fig = fig
            if origin[0] == 1:
                axes = axes[::-1]
            if origin[1] == 1:
                axes = axes.T[::-1].T
        else:
            try:
                self.fig = axes[0, 0].figure
            except Exception as err:
                raise err

        self.origin = origin
        self.axes = axes

        last = ndim - 1
        if labels is None:
            labels = [''] * ndim

        for (ii, jj), ax in np.ndenumerate(axes):
            # Set upper-right plots to invisible
            if jj > ii:
                ax.set_visible(False)
                continue

            ax.grid(True)

            # Bottom row
            if ii == last:
                if rotate and (jj == last):
                    ax.set_ylabel(labels[jj])   # currently this is being reset to empty later, that's okay
                else:
                    ax.set_xlabel(labels[jj])

                # If vertical origin is the top
                if origin[0] == 1:
                    ax.xaxis.set_label_position('top')
                    ax.xaxis.set_ticks_position('top')

            # Non-bottom row
            else:
                ax.set_xlabel('')
                for tlab in ax.xaxis.get_ticklabels():
                    tlab.set_visible(False)

            # First column
            if jj == 0:
                # Not-first rows
                if ii != 0:
                    ax.set_ylabel(labels[ii])

                # If horizontal origin is the right
                if origin[1] == 1:
                    ax.yaxis.set_label_position('right')
                    ax.yaxis.set_ticks_position('right')

            # Not-first columns
            else:
                # if (jj != last) or (not rotate):
                ax.set_ylabel('')
                for tlab in ax.yaxis.get_ticklabels():
                    tlab.set_visible(False)

            # Diagonals
            if ii == jj:
                # not top-left
                if (ii != 0) and (origin[1] == 0):
                    ax.yaxis.set_label_position('right')
                    ax.yaxis.set_ticks_position('right')
                else:
                    ax.yaxis.set_label_position('left')
                    ax.yaxis.set_ticks_position('left')

        # If axes limits are given, set axes to them
        if limits is not None:
            limit_flag = False
            kale.plot._set_corner_axes_extrema(self.axes, limits, rotate)
        # Otherwise, prepare to calculate limits during plotting
        else:
            limits = [None] * ndim
            limit_flag = True

        # --- Store key parameters
        self.ndim = ndim
        self._rotate = rotate
        self._labels = labels
        self._limits = limits
        self._limit_flag = limit_flag

        return

    def plot(self, data, edges=None, weights=None, ratio=None, quantiles=None, sigmas=None, reflect=None,
             color=None, cmap=None, limit=None, dist1d={}, dist2d={}):

        if limit is None:
            limit = self._limit_flag

        # ---- Sanitize

        if np.ndim(data) != 2:
            err = "`data` (shape: {}) must be 2D with shape (parameters, data-points)!".format(
                np.shape(data))
            raise ValueError(err)

        axes = self.axes
        size = np.shape(data)[0]
        shp = np.shape(axes)
        if (np.ndim(axes) != 2) or (shp[0] != shp[1]) or (shp[0] != size):
            raise ValueError("`axes` (shape: {}) does not match data dimension {}!".format(shp, size))

        if ratio is not None:
            if np.ndim(ratio) != 2 or np.shape(ratio)[0] != size:
                err = f"`ratio` (shape: {np.shape(ratio)}) must be 2D with shape (parameters, data-points)!"
                raise ValueError(err)

        # ---- Set parameters

        last = size - 1
        rotate = self._rotate

        # Set default color or cmap as needed
        color, cmap = kale.plot._parse_color_cmap(ax=axes[0][0], color=color, cmap=cmap)

        edges = kale.utils.parse_edges(data, edges=edges)
        quantiles, _ = kale.plot._default_quantiles(quantiles=quantiles, sigmas=sigmas)

        # ---- Draw 1D Histograms & Carpets

        limits = [None] * size      # variable to store the data extrema
        for jj, ax in enumerate(axes.diagonal()):
            rot = (rotate and (jj == last))
            refl = reflect[jj] if reflect is not None else None
            rat = ratio[jj] if ratio is not None else None
            self.dist1d(
                ax, edges[jj], data[jj], weights=weights, ratio=rat, quantiles=quantiles, rotate=rot,
                color=color, reflect=refl, **dist1d
            )
            limits[jj] = kale.utils.minmax(data[jj], stretch=self._LIMITS_STRETCH)

        # ---- Draw 2D Histograms and Contours

        for (ii, jj), ax in np.ndenumerate(axes):
            if jj >= ii:
                continue
            rat = [ratio[jj], ratio[ii]] if ratio is not None else None
            handle = self.dist2d(
                ax, [edges[jj], edges[ii]], [data[jj], data[ii]], weights=weights, ratio=rat,
                color=color, cmap=cmap, quantiles=quantiles, **dist2d
            )

        # ---- calculate and set axes limits

        if limit:
            # Update any stored values
            for ii in range(self.ndim):
                self._limits[ii] = kale.utils.minmax(limits[ii], prev=self._limits[ii])

            # Set axes to limits
            kale.plot._set_corner_axes_extrema(self.axes, self._limits, self._rotate)

        return handle

    def dist1d(self, ax, edges, data, color=None, weights=None, ratio=None, probability=True, rotate=False,
               density=None, confidence=False, hist=None, carpet=True, quantiles=None,
               ls=None, alpha=None, reflect=None, **kwargs):

        if np.ndim(data) != 1:
            err = "Input `data` (shape: {}) is not 1D!".format(np.shape(data))
            raise ValueError(err)

        if ratio is not None and np.ndim(ratio) != 1:
            err = "`ratio` (shape: {}) is not 1D!".format(np.shape(ratio))
            raise ValueError(err)

        # Use `scatter` as the limiting-number of scatter-points
        #    To disable scatter, `scatter` will be set to `None`
        carpet = kale.plot._scatter_limit(carpet, "carpet")

        # set default color to next from axes' color-cycle
        if color is None:
            color = kale.plot._get_next_color(ax)

        # ---- Draw Components

        # Draw PDF from KDE
        handle = None     # variable to store a plotting 'handle' from one of the plotted objects
        if density is not False:
            kde = kale.KDE(data, weights=weights)

            # If histogram is also being plotted (as a solid line) use a dashed line
            if ls is None:
                _ls = '--' if hist else '-'
                _alpha = 0.8 if hist else 0.8
            else:
                _ls = ls
                _alpha = alpha

            # Calculate KDE density distribution for the given parameter
            kde_kwargs = dict(probability=probability, params=0, reflect=reflect)
            xx, yy = kde.density(**kde_kwargs)

            if ratio is not None:
                kde_ratio = kale.KDE(ratio, weights=weights)
                _, kde_ratio = kde_ratio.density(points=xx, **kde_kwargs)
                yy /= kde_ratio

            # rescale by value of density
            yy = yy * density
            # Plot
            if rotate:
                temp = xx
                xx = yy
                yy = temp

            handle, = ax.plot(xx, yy, color=color, ls=_ls, alpha=_alpha, **kwargs)

        # Draw Histogram
        if hist:
            if alpha is None:
                _alpha = 0.5 if density else 0.8
            else:
                _alpha = alpha

            _, _, hh = self.hist1d(
                ax, data, edges=edges, weights=weights, ratio=ratio, color=color,
                density=True, probability=probability, joints=True, rotate=rotate,
                ls=ls, alpha=_alpha, **kwargs
            )
            if handle is None:
                handle = hh

        # Draw Contours and Median Line
        if confidence:
            if ratio is not None:
                raise NotImplementedError("`confidence` with `ratio` is not implemented!")
            hh = kale.plot._confidence(data, ax=ax, color=color, quantiles=quantiles, rotate=rotate)
            if handle is None:
                handle = hh

        # Draw Carpet Plot
        if carpet is not None:
            if ratio is not None:
                raise NotImplementedError("`confidence` with `carpet` is not implemented!")
            hh = kale.plot._carpet(data, weights=weights, ax=ax, color=color, rotate=rotate, limit=carpet)
            if handle is None:
                handle = hh

        return handle

    def hist1d(self, ax, data, edges=None, weights=None, ratio=None, density=False, probability=False,
            renormalize=False, joints=True, positive=True, rotate=False, **kwargs):

        hist_kwargs = dict(density=density, probability=probability)
        # Calculate histogram
        hist, edges = kale.utils.histogram(data, bins=edges, weights=weights, **hist_kwargs)

        if ratio is not None:
            hist_ratio, _ = kale.utils.histogram(data, bins=edges, **hist_kwargs)
            hist /= hist_ratio

        # Draw
        rv = kale.plot.draw_hist1d(
            ax, edges, hist,
            renormalize=renormalize, joints=joints, positive=positive, rotate=rotate,
            **kwargs
        )
        return hist, edges, rv

    def dist2d(
        self, ax, edges, data, weights=None, ratio=None, quantiles=None, sigmas=None,
        color=None, cmap=None, smooth=None, upsample=None, pad=True, outline=True,
        median=False, scatter=True, contour=True, hist=True, mask_dense=None, mask_below=True, mask_alpha=0.9
    ):

        if np.ndim(data) != 2 or np.shape(data)[0] != 2:
            err = f"`data` (shape: {np.shape(data)}) must be 2D with shape (parameters, data-points)!"
            raise ValueError(err)

        if ratio is not None:
            if np.ndim(ratio) != 2 or np.shape(ratio)[0] != 2:
                err = f"`ratio` (shape: {np.shape(ratio)}) must be 2D with shape (parameters, data-points)!"
                raise ValueError(err)

        # Set default color or cmap as needed
        color, cmap = kale.plot._parse_color_cmap(ax=ax, color=color, cmap=cmap)

        # Use `scatter` as the limiting-number of scatter-points
        #    To disable scatter, `scatter` will be set to `None`
        scatter = kale.plot._scatter_limit(scatter, "scatter")

        # Default: if either hist or contour is being plotted, mask over high-density scatter points
        if mask_dense is None:
            mask_dense = (scatter is not None) and (hist or contour)

        # Calculate histogram
        edges = kale.utils.parse_edges(data, edges=edges)
        hist_kwargs = dict(bins=edges, density=True)
        hh, *_ = np.histogram2d(*data, weights=weights, **hist_kwargs)

        if ratio is not None:
            hh_ratio, *_ = np.histogram2d(*ratio, **hist_kwargs)
            hh /= hh_ratio
            hh = np.nan_to_num(hh)

        _, levels, quantiles = kale.plot._dfm_levels(hh, quantiles=quantiles, sigmas=sigmas)
        if mask_below is True:
            mask_below = levels.min()

        handle = None

        # ---- Draw Scatter Points
        if (scatter is not None):
            handle = kale.plot.draw_scatter(ax, *data, color=color, zorder=5, limit=scatter)

        # ---- Draw Median Lines (cross-hairs style)
        if median:
            if ratio:
                raise NotImplementedError("`median` is not impemented with `ratio`!")

            for dd, func in zip(data, [ax.axvline, ax.axhline]):
                # Calculate value
                if weights is None:
                    med = np.median(dd)
                else:
                    med = kale.utils.quantiles(dd, percs=0.5, weights=weights)

                # Load path_effects
                out_pe = kale.plot._get_outline_effects() if outline else None
                # Draw
                func(med, color=color, alpha=0.25, lw=1.0, zorder=40, path_effects=out_pe)

        cents, hh_prep = kale.plot._prep_hist(edges, hh, smooth, upsample, pad)

        # ---- Draw 2D Histogram
        if hist:
            _ee, _hh, handle = kale.plot.draw_hist2d(
                ax, edges, hh, mask_below=mask_below, cmap=cmap, zorder=10, shading='auto',
            )

        # ---- Draw Contours
        if contour:
            contour_cmap = cmap.reversed()
            # Narrow the range of contour colors relative to full `cmap`
            dd = 0.7 / 2
            nq = len(quantiles)
            if nq < 4:
                dd = nq*0.08
            contour_cmap = kale.plot._cut_colormap(contour_cmap, 0.5 - dd, 0.5 + dd)

            _ee, _hh, _handle = _contour2d(
                ax, cents, hh_prep, levels=levels, cmap=contour_cmap, zorder=20, outline=outline,
            )

            # hi = 1 if len(_handle.collections) > 0 else 0
            hi = -1
            handle = _handle.collections[hi]
            # for some reason the above handle is not showing up on legends... create a dummy line
            # to get a better handle
            col = handle.get_edgecolor()
            handle, = ax.plot([], [], color=col)

        # Mask dense scatter-points
        if mask_dense:
            # NOTE: levels need to be recalculated here!
            _, levels, quantiles = kale.plot._dfm_levels(hh_prep, quantiles=quantiles)
            span = [levels.min(), hh_prep.max()]
            mask_cmap = mpl.colors.ListedColormap('white')
            # Draw
            ax.contourf(*cents, hh_prep, span, cmap=mask_cmap, antialiased=True, zorder=9, alpha=mask_alpha)

        return handle

    def legend(self, handles, labels, index=None,
               loc=None, fancybox=False, borderaxespad=0, **kwargs):
        """
        """
        fig = self.fig

        # Set Bounding Box Location
        # ------------------------------------
        bbox = kwargs.pop('bbox', None)
        bbox = kwargs.pop('bbox_to_anchor', bbox)
        if bbox is None:
            if index is None:
                size = self.ndim
                if size in [2, 3, 4]:
                    index = (0, -1)
                    loc = 'lower left'
                elif size == 1:
                    index = (0, 0)
                    loc = 'upper right'
                elif size % 2 == 0:
                    index = size // 2
                    index = (size - index - 2, index + 1)
                    loc = 'lower left'
                else:
                    index = (size // 2) + 1
                    loc = 'lower left'
                    index = (size-index-1, index)

            bbox = self.axes[index].get_position()
            bbox = (bbox.x0, bbox.y0)
            kwargs['bbox_to_anchor'] = bbox
            kwargs.setdefault('bbox_transform', fig.transFigure)

        # Set other defaults
        leg = fig.legend(handles, labels, fancybox=fancybox,
                         borderaxespad=borderaxespad, loc=loc, **kwargs)
        return leg

    def target(self, targets, upper_limits=None, lower_limits=None, lw=1.0, fill_alpha=0.1, **kwargs):
        size = self.ndim
        axes = self.axes
        last = size - 1
        # labs = self._labels
        extr = self._limits

        # ---- check / sanitize arguments
        if len(targets) != size:
            err = "`targets` (shape: {}) must be shaped ({},)!".format(np.shape(targets), size)
            raise ValueError(err)

        if lower_limits is None:
            lower_limits = [None] * size
        if len(lower_limits) != size:
            err = "`lower_limits` (shape: {}) must be shaped ({},)!".format(np.shape(lower_limits), size)
            raise ValueError(err)

        if upper_limits is None:
            upper_limits = [None] * size
        if len(upper_limits) != size:
            err = "`upper_limits` (shape: {}) must be shaped ({},)!".format(np.shape(upper_limits), size)
            raise ValueError(err)

        # ---- configure settings
        kwargs.setdefault('color', 'red')
        kwargs.setdefault('alpha', 0.50)
        kwargs.setdefault('zorder', 20)
        line_kw = dict()
        line_kw.update(kwargs)
        line_kw['lw'] = lw
        span_kw = dict()
        span_kw.update(kwargs)
        span_kw['alpha'] = fill_alpha

        # ---- draw 1D targets and limits
        for jj, ax in enumerate(axes.diagonal()):
            if (self._rotate and (jj == last)):
                func = ax.axhline
                func_up = lambda xx: ax.axhspan(extr[jj][0], xx, **span_kw)
                func_lo = lambda xx: ax.axhspan(xx, extr[jj][1], **span_kw)
            else:
                func = ax.axvline
                func_up = lambda xx: ax.axvspan(extr[jj][0], xx, **span_kw)
                func_lo = lambda xx: ax.axvspan(xx, extr[jj][1], **span_kw)

            if targets[jj] is not None:
                func(targets[jj], **line_kw)
            if upper_limits[jj] is not None:
                func_up(upper_limits[jj])
            if lower_limits[jj] is not None:
                func_lo(lower_limits[jj])

        # ---- draw 2D targets and limits
        for (ii, jj), ax in np.ndenumerate(axes):
            if jj >= ii:
                continue
            for kk, func, func_lim in zip([ii, jj], [ax.axhline, ax.axvline], [ax.axhspan, ax.axvspan]):
                if targets[kk] is not None:
                    func(targets[kk], **line_kw)
                if upper_limits[kk] is not None:
                    func_lim(extr[kk][0], upper_limits[kk], **span_kw)
                if lower_limits[kk] is not None:
                    func_lim(lower_limits[kk], extr[kk][0], **span_kw)

        return


def _contour2d(ax, edges, hist, levels, outline=True, **kwargs):

    LW = 1.5

    alpha = kwargs.setdefault('alpha', 0.8)
    lw = kwargs.pop('linewidths', kwargs.pop('lw', LW))
    kwargs.setdefault('linestyles', kwargs.pop('ls', '-'))
    kwargs.setdefault('zorder', 10)

    # ---- Draw contours
    cont = ax.contour(*edges, hist, levels=levels, linewidths=lw, **kwargs)

    # ---- Add Outline path effect to contours
    if (outline is True):
        outline = kale.plot._get_outline_effects(2*lw, alpha=1 - np.sqrt(1 - alpha))
        plt.setp(cont.collections, path_effects=outline)

    return edges, hist, cont


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    '''
    https://stackoverflow.com/a/18926541
    '''
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

# =================================================================================================
# ====    Below Needs Review / Cleaning    ====
# =================================================================================================

'''
def plot_bin_pop(pop):
    mt, mr = utils.mtmr_from_m1m2(pop.mass)
    redz = cosmo.a_to_z(pop.scafa)
    data = [mt/MSOL, mr, pop.sepa/PC, 1+redz]
    data = [np.log10(dd) for dd in data]
    reflect = [None, [None, 0], None, [0, None]]
    labels = [r'M/M_\odot', 'q', r'a/\mathrm{{pc}}', '1+z']
    labels = [r'${{\log_{{10}}}} \left({}\right)$'.format(ll) for ll in labels]

    if pop.eccen is not None:
        data.append(pop.eccen)
        reflect.append([0.0, 1.0])
        labels.append('e')

    kde = kale.KDE(data, reflect=reflect)
    corner = kale.Corner(kde, labels=labels, figsize=[8, 8])
    corner.plot_data(kde)
    return corner


def plot_mbh_scaling_relations(pop, fname=None, color='r'):
    units = r"$[\log_{10}(M/M_\odot)]$"
    fig, ax = plt.subplots(figsize=[8, 5])
    ax.set(xlabel=f'Stellar Mass {units}', ylabel=f'BH Mass {units}')

    #   ====    Plot McConnell+Ma-2013 Data    ====
    handles = []
    names = []
    if fname is not None:
        hh = _draw_MM2013_data(ax, fname)
        handles.append(hh)
        names.append('McConnell+Ma')

    #   ====    Plot MBH Merger Data    ====
    hh, nn = _draw_pop_masses(ax, pop, color)
    handles = handles + hh
    names = names + nn
    ax.legend(handles, names)

    return fig


def _draw_MM2013_data(ax):
    data = observations.load_mcconnell_ma_2013()
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


def plot_gwb(gwb, color=None, uniform=False, nreals=5):
    """Plot a GW background from the given `Grav_Waves` instance.

    Plots samples, confidence intervals, power-law, and adds twin-Hz axis (x2).

    Parameters
    ----------
    gwb : `gravwaves.Grav_Waves` (subclass) instance

    Returns
    -------
    fig : `mpl.figure.Figure`
        New matplotlib figure instance.

    """

    fig, ax = figax(
        scale='log',
        xlabel=r'frequency $[\mathrm{yr}^{-1}]$',
        ylabel=r'characteristic strain $[\mathrm{h}_c]$'
    )

    if uniform:
        color = ax._get_lines.get_next_color()

    _draw_gwb_sample(ax, gwb, color=color, num=nreals)
    _draw_gwb_conf(ax, gwb, color=color)
    _draw_plaw(ax, gwb.freqs*YR, f0=1, color='0.5', lw=2.0, ls='--')

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
'''
