"""Plotting module.

Provides convenience methods for generating standard plots and components using `matplotlib`.

"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import kalepy as kale

from holodeck import cosmo, utils, observations, log
from holodeck.constants import MSOL, PC, YR

LABEL_GW_FREQUENCY_YR = "GW Frequency $[\mathrm{yr}^{-1}]$"
LABEL_CHARACTERISTIC_STRAIN = "GW Characteristic Strain"


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
            ax.grid(True, which='major', axis='both', c='0.6', zorder=2, alpha=0.4)
            ax.grid(True, which='minor', axis='both', c='0.8', zorder=2, alpha=0.4)

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


def set_axis_color(ax, axis='y', fs=None, color='k', side='right'):
    # Set tick colors and font-sizes
    kw = dict(labelsize=fs) if (fs is not None) else {}
    ax.tick_params(axis=axis, which='both', colors=color, **kw)

    if axis == 'x':
        ax.xaxis.label.set_color(color)
        offt = ax.get_xaxis().get_offset_text()
    else:
        ax.yaxis.label.set_color(color)
        offt = ax.get_yaxis().get_offset_text()

    # Set Spine colors
    if side is not None:
        ax.spines[side].set_color(color)
    offt.set_color(color)
    return ax


def set_axis_pos(ax, axis, pos, side):
    axis = axis.lower()

    if axis == 'x':
        offt = ax.get_xaxis().get_offset_text()

        offt.set_y(pos)
        ax.xaxis.set_label_position(side)
        ax.xaxis.set_ticks_position(side)

    elif axis == 'y':
        offt = ax.get_yaxis().get_offset_text()

        offt.set_x(pos)
        ax.yaxis.set_label_position(side)
        ax.yaxis.set_ticks_position(side)

    else:
        err = f"`axis` argument should be ['x', 'y'], got '{axis}'!"
        log.exception(err)
        raise ValueError(err)

    # Set Spine colors
    ax.set_frame_on(True)
    ax.spines[side].set_position(('axes', pos))
    ax.spines[side].set_visible(True)
    ax.patch.set_visible(False)

    return ax


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
    """Convert from

    Parameters
    ----------
    xx : array_like
        Independence variable representing bin-edges.  Size (N,)
    yy : array_like
        Dependence variable representing histogram amplitudes.  Size (N-1,)

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

    if yfilter is not None:
        if yfilter is True:
            idx = (ynew > 0.0)
        else:
            idx = yfilter(ynew)

        xnew = xnew[idx]
        ynew = ynew[idx]

    return xnew, ynew


def draw_hist_steps(ax, xx, yy, yfilter=None, **kwargs):
    return ax.plot(*_get_hist_steps(xx, yy, yfilter=yfilter), **kwargs)


def draw_gwb(ax, xx, gwb, nsamp=10, color=None, label=None, plot_kwargs={}):
    if color is None:
        color = ax._get_lines.get_next_color()

    mm, *conf = np.percentile(gwb, [50, 25, 75], axis=1)
    hh, = ax.plot(xx, mm, alpha=0.5, color=color, label=label, **plot_kwargs)
    ax.fill_between(xx, *conf, color=color, alpha=0.15)

    if (nsamp is not None) and (nsamp > 0):
        nsamp_max = gwb.shape[1]
        idx = np.random.choice(nsamp_max, np.min([nsamp, nsamp_max]), replace=False)
        for ii in idx:
            ax.plot(xx, gwb[:, ii], color=color, alpha=0.25, lw=1.0, ls='-')

    return hh


def plot_gwb(fobs, gwb, **kwargs):
    xx = fobs * YR
    fig, ax = figax(
        xlabel=LABEL_GW_FREQUENCY_YR,
        ylabel=LABEL_CHARACTERISTIC_STRAIN
    )
    draw_gwb(ax, xx, gwb, **kwargs)
    _twin_hz(ax)
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


def _twin_hz(ax, nano=True, fs=8, **kw):
    tw = ax.twiny()
    xlim = np.array(ax.get_xlim()) / YR
    if nano:
        label = "nHz"
        xlim *= 1e9
    else:
        label = "Hz"

    label = fr"GW Frequency $[\mathrm{{{label}}}]$"
    tw.set(xlim=xlim, xscale='log')
    tw.set_xlabel(label, fontsize=fs, **kw)
    return



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


def draw_med_conf(ax, xx, vals, percs=[25, 75], axis=-1, yfilter=None, **kwargs):
    kwargs.setdefault('alpha', 0.5)
    assert len(percs) == 2
    med, *conf = np.percentile(vals, [50, percs[0], percs[1]], axis=-1)
    if (xx.size,) == med.shape:
        hist_flag = False
    elif (xx.size-1,) == med.shape:
        hist_flag = True
    else:
        raise ValueError(f"Bad shapes!  {np.shape(xx)=}, {np.shape(vals)=}")

    if hist_flag:
        hh, = draw_hist_steps(ax, xx, med, yfilter=yfilter, **kwargs)
        _xx, lo = _get_hist_steps(xx, conf[0])
        _xx, hi = _get_hist_steps(xx, conf[1])
    else:
        hh, = ax.plot(xx, med, **kwargs)
        _xx = xx
        lo = conf[0]
        hi = conf[1]

    fill = ax.fill_between(_xx, lo, hi, alpha=0.2, color=hh.get_color())
    return (hh, fill)
'''

'''
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
'''

'''
def plot_evo(evo, freqs=None, sepa=None, ax=None):
    if (freqs is None) and (sepa is None):
        err = "Either `freqs` or `sepa` must be provided!"
        log.exception(err)
        raise ValueError(err)

    if freqs is not None:
        data = evo.at('fobs', freqs)
        xx = freqs * YR
        xlabel = 'GW Frequency [1/yr]'
    else:
        data = evo.at('sepa', sepa)
        xx = sepa / PC
        xlabel = 'Binary Separation [pc]'

    if ax is None:
        fig, ax = figax(xlabel=xlabel)
    else:
        fig = ax.get_figure()

    def _draw_vals_conf(ax, xx, vals, color=None, label=None):
        if color is None:
            color = ax._get_lines.get_next_color()
        if label is not None:
            ax.set_ylabel(label, color=color)
            ax.tick_params(axis='y', which='both', colors=color)
        # vals = np.percentile(vals, [25, 50, 75], axis=0) / units
        vals = utils.quantiles(vals, [0.25, 0.50, 0.75], axis=0).T
        h1 = ax.fill_between(xx, vals[0], vals[-1], alpha=0.25, color=color)
        h2, = ax.plot(xx, vals[1], alpha=0.75, lw=2.0, color=color)
        return (h1, h2)

    # handles = []
    # labels = []

    name = 'Hardening Time [yr]'
    vals = np.fabs(data['sepa'] / data['dadt']) / YR
    _draw_vals_conf(ax, xx, vals, label=name)
    # handles.append(hh)
    # labels.append(name)

    # name = 'eccen'
    # tw = ax.twinx()
    # hh, nn = _draw_vals_conf(tw, freqs*YR, name, 'green')
    # if hh is not None:
    #     handles.append(hh)
    #     labels.append(nn)

    # ax.legend(handles, labels)
    return fig
'''