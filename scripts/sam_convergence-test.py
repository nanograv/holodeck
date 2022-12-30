"""
"""

from pathlib import Path
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import kalepy as kale
import zcode.plot as zplot

import holodeck as holo
import holodeck.utils
import holodeck.plot
from holodeck.constants import YR

np.seterr(divide='ignore', invalid='ignore', over='ignore')
warnings.filterwarnings("ignore", category=UserWarning)

# Plotting settings
mpl.rc('font', **{'family': 'serif', 'sans-serif': ['Times'], 'size': 15})
mpl.rc('lines', solid_capstyle='round')
mpl.rc('mathtext', fontset='cm')
mpl.style.use('default')   # avoid dark backgrounds from dark theme vscode
plt.rcParams.update({'grid.alpha': 0.5})

PATH = "output/convergence"
PATH = Path(holo._PATH_ROOT).joinpath(PATH).resolve()
PATH.mkdir(parents=True, exist_ok=True)


def run_with_shape(fobs, num, nreals, sample_threshold, axis=None, **kw):
    poisson = kw.pop('poisson', False)
    kwargs = dict(poisson_outside=poisson, poisson_inside=poisson)
    kwargs.update(kw)

    gsmf = holo.sam.GSMF_Schechter()        # Galaxy Stellar-Mass Function (GSMF)
    gpf = holo.sam.GPF_Power_Law()          # Galaxy Pair Fraction         (GPF)
    gmt = holo.sam.GMT_Power_Law()          # Galaxy Merger Time           (GMT)
    mmbulge = holo.relations.MMBulge_Standard()     # M-MBulge Relation            (MMB)
    hard = holo.hardening.Hard_GW

    sam = holo.sam.Semi_Analytic_Model(gsmf=gsmf, gpf=gpf, gmt=gmt, mmbulge=mmbulge, shape=num)
    gwb_smooth = sam.gwb(fobs, realize=False)

    nbins = len(fobs) - 1
    gwb = np.zeros((nbins, nreals))
    gff = np.zeros_like(gwb)
    gwf = np.zeros_like(gwb)

    for rr in holo.utils.tqdm(range(nreals), leave=False):
        vals, weights, edges, dens, mass = holo.sam.sample_sam_with_hardening(
            sam, hard, fobs=fobs, sample_threshold=sample_threshold, **kwargs
        )
        gff[:, rr], gwf[:, rr], gwb[:, rr] = holo.sam._gws_from_samples(vals, weights, fobs)

    fig, ax = holo.plot.figax(
        figsize=[8, 4], xlabel='Frequency [yr]', ylabel='cStrain',
        left=0.1, right=0.98, bottom=0.15
    )

    axes = [ax, axis] if (axis is not None) else [ax]

    fcent = kale.utils.midpoints(fobs*YR)
    plaw = 1e-15 * np.power(fcent, -2.0/3.0)
    for ax in axes:
        ax.plot(fcent, gwb_smooth, 'k--', alpha=0.5, lw=2.0)
        ax.plot(fcent, plaw, 'k-', alpha=0.5, lw=1.0)

    percs = np.percentile(gwb, [50, 5, 95, 25, 75], axis=-1)
    yy = []
    for pp in percs:
        xx, _yy = holo.plot._get_hist_steps(fobs, pp)
        yy.append(_yy)

    xx *= YR
    for ax in axes:
        hh, = ax.plot(xx, yy[0], 'k-')
        for ii in range((len(percs)-1)//2):
            lo = 2*ii + 1
            hi = 2*ii + 2
            ax.fill_between(xx, yy[lo], yy[hi], color=hh.get_color(), alpha=0.2)

    yy = np.percentile(gwf, 75, axis=-1)
    xx = np.median(gff, axis=-1)
    for ax in axes:
        ax.scatter(xx*YR, yy, color='r', alpha=0.35, marker='x', s=10)

    label = (f"g$={num:04d}$", f"nr$={nreals:04d}$", f"thresh=${sample_threshold:+05.1f}$")
    fname = "_".join(label).replace('$', '').replace('=', '')
    fname = PATH.joinpath(fname)

    val_sample = np.median(gwb[0, :])
    val_smooth = gwb_smooth[0]
    sample_err = (val_sample - val_smooth) / val_smooth
    sample_off = (val_sample - plaw[0]) / plaw[0]
    smooth_off = (val_smooth - plaw[0]) / plaw[0]

    label = ", ".join(label)
    label = f"err={sample_err:+.5f} off={sample_off:+.5f} ({smooth_off:+.5f})" + "\n" + label
    for ax in axes:
        zplot.text(ax, label, loc='ll', fs=10, )
        holo.plot._twin_hz(ax, labelpad=-4)

    fname_fig = fname.parent / (fname.name + ".png")
    fig.savefig(fname_fig)
    fname_npz = fname.parent / (fname.name + ".npz")
    np.savez(fname_npz, fobs=fobs, gwb_smooth=gwb_smooth, gff=gff, gwf=gwf, gwb=gwb,
             sample_err=sample_err, sample_ff=sample_off, smooth_off=smooth_off)

    return fig, sample_err, sample_off, smooth_off


def main():

    fobs = holo.utils.nyquist_freqs(10.0*YR, 0.1*YR)
    # nreals = 30
    # thresholds = [3, 10, 30]
    # grids = [10, 40, 80, 120, 160]
    nreals = 10
    thresholds = [3, 5]
    grids = [10, 20]

    sample_errors = np.zeros((len(thresholds), len(grids)))
    sample_offsets = np.zeros_like(sample_errors)
    smooth_offsets = np.zeros_like(sample_errors)

    figsize = [6, 4]
    figsize[0] *= sample_errors.shape[1]
    figsize[1] *= sample_errors.shape[0]

    fig, axes = holo.plot.figax(
        figsize=figsize, nrows=sample_errors.shape[0], ncols=sample_errors.shape[1], sharex=True, sharey=True,
        xlabel='Frequency [yr]', ylabel='cStrain',
    )

    for ii, thresh in enumerate(holo.utils.tqdm(thresholds)):
        for jj, gr in enumerate(holo.utils.tqdm(grids, leave=False)):
            ax = axes[ii, jj]
            _, sample_err, sample_off, smooth_off = run_with_shape(
                fobs, num=gr, nreals=nreals, sample_threshold=thresh, axis=ax
            )
            plt.close(_)
            sample_errors[ii, jj] = sample_err
            sample_offsets[ii, jj] = sample_off
            smooth_offsets[ii, jj] = smooth_off

    g_str = [str(g) for g in grids]
    g_str = '-'.join(g_str)
    t_str = [str(t) for t in thresholds]
    t_str = '-'.join(t_str)
    fname = f"nr{nreals:03d}_ng{g_str}_t{t_str}"
    fig.savefig(PATH.joinpath(fname + ".png"))

    fig, ax = holo.plot.figax(xlabel='Grid Size', scale='lin')
    kw = dict(alpha=0.75, lw=1.0)
    lines = []
    labels = []
    for ii, thresh in enumerate(thresholds):
        label = 'sample error' if ii == 0 else None
        cc, = ax.plot(grids, sample_errors[ii, :], ls='-', label=label, **kw)
        lines.append(cc)
        labels.append(f"{thresh:+4.1f}")
        cc = cc.get_color()
        label = 'sample offset' if ii == 0 else None
        ax.plot(grids, sample_offsets[ii, :], ls='--', label=label, color=cc, **kw)
        label = 'smooth offset' if ii == 0 else None
        ax.plot(grids, smooth_offsets[ii, :], ls=':', label=label, color=cc, **kw)

    leg = ax.legend()
    zplot.legend(ax, lines, labels, prev=leg, loc='bl')
    fig.savefig(PATH.joinpath(fname + "_convergence.png"))
    return


if __name__ == "__main__":
    main()
