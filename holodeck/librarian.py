"""
"""

import abc
from pathlib import Path
from datetime import datetime
import psutil
import resource
import os

import h5py
import numpy as np
import scipy as sp
import scipy.optimize  # noqa
import matplotlib.pyplot as plt
import tqdm

from scipy.stats import qmc
import pyDOE

import holodeck as holo
import holodeck.single_sources as ss
from holodeck import log, utils
from holodeck.constants import YR


__version__ = "0.1.0"


class _Param_Space(abc.ABC):

    def __init__(self, log, nsamples, sam_shape, seed, **kwargs):

        log.debug(f"seed = {seed}")
        np.random.seed(seed)
        #! NOTE: this should be saved to output!!!
        random_state = np.random.get_state()
        log.debug(f"Random state is:\n{random_state}")

        names = list(kwargs.keys())
        ndims = len(names)
        if ndims == 0:
            err = f"No parameters passed to {self}!"
            log.exception(err)
            raise RuntimeError(err)

        dists = []
        for nam in names:
            val = kwargs[nam]

            if not isinstance(val, _Param_Dist):
                err = f"{nam}: {val} is not a `_Param_Dist` object!"
                log.exception(err)
                raise ValueError(err)

            try:
                vv = val(0.0)
                f"{vv:.4e}"
            except Exception as err:
                log.exception(f"Failed to call {val}(0.0)!")
                log.exception(err)
                raise err

            dists.append(val)

        # if strength = 2, then n must be equal to p**2, with p prime, and d <= p + 1
        LHS = qmc.LatinHypercube(d=ndims, centered=False, strength=1, seed=seed)
        # (S, D) - samples, dimensions
        samples = LHS.random(n=nsamples)
        params = np.zeros_like(samples)

        for ii, dist in enumerate(dists):
            params[:, ii] = dist(samples[:, ii])

        self._log = log
        self.names = names
        self.nsamples = nsamples
        self.ndims = ndims
        self.sam_shape = sam_shape
        self._seed = seed
        self._random_state = random_state
        self._samples = samples
        self._params = params

        return

    def params(self, num):
        return self._params[num]

    def param_dict(self, num):
        rv = {nn: pp for nn, pp in zip(self.names, self.params(num))}
        return rv

    def __call__(self, num):
        return self.model_for_number(num)

    @property
    def shape(self):
        return self._samples.shape

    def model_for_number(self, num):
        params = self.param_dict(num)
        self._log.debug(f"params {num} :: {params}")
        return self.model_for_params(params, self.sam_shape)

    @classmethod
    @abc.abstractmethod
    def model_for_params(cls, params):
        raise


class _Param_Dist(abc.ABC):

    def __init__(self, clip=None):
        if clip is not None:
            assert len(clip) == 2
        self._clip = clip
        return

    def __call__(self, xx):
        rv = self._dist_func(xx)
        if self._clip is not None:
            rv = np.clip(rv, *self._clip)
        return rv


class PD_Uniform(_Param_Dist):

    def __init__(self, lo, hi, **kwargs):
        super().__init__(**kwargs)
        self._lo = lo
        self._hi = hi
        self._dist_func = lambda xx: self._lo + (self._hi - self._lo) * xx
        return


class PD_Uniform_Log(_Param_Dist):

    def __init__(self, lo, hi, **kwargs):
        super().__init__(**kwargs)
        assert lo > 0.0 and hi > 0.0
        self._lo = np.log10(lo)
        self._hi = np.log10(hi)
        self._dist_func = lambda xx: np.power(10.0, self._lo + (self._hi - self._lo) * xx)
        return


class PD_Normal(_Param_Dist):

    def __init__(self, mean, stdev, clip=None, **kwargs):
        super().__init__(**kwargs)
        assert stdev > 0.0
        self._mean = mean
        self._stdev = stdev
        self._clip = clip
        self._dist = sp.stats.norm(loc=mean, scale=stdev)
        if clip is not None:
            if len(clip) != 2:
                err = f"{clip=} | `clip` must be (2,) values of lo and hi bounds at which to clip!"
                log.exception(err)
                raise ValueError(err)
            self._dist_func = lambda xx: np.clip(self._dist.ppf(xx), *clip)
        else:
            self._dist_func = lambda xx: self._dist.ppf(xx)
        return


class PD_Lin_Log(_Param_Dist):

    def __init__(self, lo, hi, crit, lofrac, **kwargs):
        """Distribute linearly below a cutoff, and then logarithmically above.

        Parameters
        ----------
        lo : float,
            lowest output value (in linear space)
        hi : float,
            highest output value (in linear space)
        crit : float,
            Location of transition from log to lin scaling.
        lofrac : float,
            Fraction of mass below the cutoff.

        """
        super().__init__(**kwargs)
        self._lo = lo
        self._hi = hi
        self._crit = crit
        self._lofrac = lofrac
        return

    def _dist_func(self, xx):
        lo = self._lo
        crit = self._crit
        lofrac = self._lofrac
        l10_crit = np.log10(crit)
        l10_hi = np.log10(self._hi)
        xx = np.atleast_1d(xx)
        yy = np.empty_like(xx)

        # select points below the cutoff
        loidx = (xx <= lofrac)
        # transform to linear-scaling between [lo, crit]
        yy[loidx] = lo + xx[loidx] * (crit - lo) / lofrac

        # select points above the cutoff
        hiidx = ~loidx
        # transform to log-scaling between [crit, hi]
        temp = l10_crit + (l10_hi - l10_crit) * (xx[hiidx] - lofrac) / (1 - lofrac)
        yy[hiidx] = np.power(10.0, temp)
        return yy


class PD_Log_Lin(_Param_Dist):

    def __init__(self, lo, hi, crit, lofrac, **kwargs):
        """Distribute logarithmically below a cutoff, and then linearly above.

        Parameters
        ----------
        lo : float,
            lowest output value (in linear space)
        hi : float,
            highest output value (in linear space)
        crit : float,
            Location of transition from log to lin scaling.
        lofrac : float,
            Fraction of mass below the cutoff.

        """
        super().__init__(**kwargs)
        self._lo = lo
        self._hi = hi
        self._crit = crit
        self._lofrac = lofrac
        return

    def _dist_func(self, xx):
        hi = self._hi
        crit = self._crit
        lofrac = self._lofrac
        l10_lo = np.log10(self._lo)
        l10_crit = np.log10(crit)

        xx = np.atleast_1d(xx)
        yy = np.empty_like(xx)

        # select points below the cutoff
        loidx = (xx <= lofrac)
        # transform to log-scaling between [lo, crit]
        temp = l10_lo + (l10_crit - l10_lo) * xx[loidx] / lofrac
        yy[loidx] = np.power(10.0, temp)

        # select points above the cutoff
        hiidx = ~loidx
        # transform to lin-scaling between [crit, hi]
        yy[hiidx] = crit + (hi - crit) * (xx[hiidx] - lofrac) / (1.0 - lofrac)
        return yy


def sam_lib_combine(path_output, log, debug=False):
    path_output = Path(path_output)
    log.info(f"Path output = {path_output}")

    regex = "lib_sams__p*.npz"
    files = sorted(path_output.glob(regex))
    num_files = len(files)
    log.info(f"\texists={path_output.exists()}, found {num_files} files")

    # ---- Make sure that no file numbers are missing
    all_exist = True
    log.info("Checking files")
    ii = 0
    for ii in tqdm.tqdm(range(num_files)):
        temp = path_output.joinpath(regex.replace('*', f"{ii:06d}"))
        exists = temp.exists()
        if not exists:
            all_exist = False
            break

    if num_files < 2:
        all_exist = False

    if not all_exist:
        err = f"Missing at least file number {ii} out of {num_files} files!"
        log.exception(err)
        raise ValueError(err)

    # ---- Check one example data file
    temp = files[0]
    data = np.load(temp, allow_pickle=True)
    log.info(f"Test file: {temp}\n\tkeys: {list(data.keys())}")
    fobs = data['fobs']
    fobs_edges = data['fobs_edges']
    nfreqs = fobs.size
    if ('fail' in data) or ('gwb' not in data):
        err = f"THIS IS A FAILED DATASET ({0}, {temp}).  LIBRARIAN HASNT BEEN UPDATED TO HANDLE THIS CASE!"
        log.exception(err)
        raise RuntimeError(err)

    temp_gwb = data['gwb'][:]
    assert np.ndim(temp_gwb) == 2
    _nfreqs, nreals = temp_gwb.shape
    assert nfreqs == _nfreqs
    all_sample_vals = data['samples']   # uniform [0.0, 1.0] samples in each dimension, converted to parameters
    all_param_vals = data['params']     # physical parameters
    param_names = data['param_names']
    nsamples = data['nsamples']
    pdim = data['pdim']
    lib_vers = str(data['librarian_version'])
    log.info(f"Sample file uses librarian.py version {lib_vers}")
    new_lib_vers = __version__
    if lib_vers != new_lib_vers:
        log.warning(f"Loaded file {temp} uses librarian.py version {lib_vers}, current version is {new_lib_vers}!")

    lib_vers = [lib_vers]
    assert all_sample_vals.shape == (nsamples, pdim)
    assert all_param_vals.shape == (nsamples, pdim)
    if num_files != nsamples:
        raise ValueError(f"nsamples={nsamples} but num_files={num_files} !!")

    fit_nbins = data['fit_nbins']
    fit_shape = data['fit_lamp'].shape
    fit_med_shape = data['fit_med_lamp'].shape
    fit_keys = ['fit_lamp', 'fit_plaw']
    fit_med_keys = ['fit_med_lamp', 'fit_med_plaw']
    for fk in fit_keys + fit_med_keys:
        log.info(f"\t{fk:>40s}: {data[fk].shape}")

    # ---- Store results from all files

    gwb_shape = [num_files, nfreqs, nreals]
    shape_names = ['params', 'freqs', 'reals']
    gwb = np.zeros(gwb_shape)
    sample_params = np.zeros((num_files, pdim))
    fit_shape = (num_files,) + fit_shape
    fit_med_shape = (num_files,) + fit_med_shape
    fit_data = {kk: np.zeros(fit_shape) for kk in fit_keys}
    fit_data.update({kk: np.zeros(fit_med_shape) for kk in fit_med_keys})

    log.info(f"Collecting data from {len(files)} files")
    good_samp = np.ones(nsamples, dtype=bool)
    all_nonzero = np.zeros(nsamples, dtype=bool)
    any_nonzero = np.zeros_like(all_nonzero)
    tot_nonzero = np.zeros((nsamples, nreals), dtype=bool)
    for ii, file in enumerate(tqdm.tqdm(files)):
        temp = np.load(file, allow_pickle=True)
        # When a processor fails for a given parameter, the output file is still created with the 'fail' key added
        if ('fail' in temp) or ('gwb' not in temp):
            msg = f"file {ii=:06d} is a failure file, setting values to NaN ({file})"
            log.warning(msg)
            gwb[ii, :, :] = np.nan
            for fk in fit_keys + fit_med_keys:
                fit_data[fk][ii, ...] = np.nan

            good_samp[ii] = False
            continue

        this_gwb = temp['gwb']
        all_nonzero[ii] = np.all(this_gwb > 0.0)
        any_nonzero[ii] = np.any(this_gwb > 0.0)
        tot_nonzero[ii, :] = np.all(this_gwb > 0.0, axis=0)

        # Make sure basic parameters match from this file to the test file
        assert ii == temp['pnum']
        assert np.allclose(fobs, temp['fobs'])
        assert np.allclose(fobs_edges, temp['fobs_edges'])
        check = str(temp['librarian_version'])
        if check not in lib_vers:
            log.warning("Mismatch in librarian.py version in saved files!  {ii} {file} with version {check}")
            lib_vers.append(check)

        # Make sure the individual parameters are consistent with the test file
        for jj, nam in enumerate(param_names):
            check = temp[nam]
            expect = all_param_vals[ii, jj]
            if not np.isclose(check, expect):
                err = f"Expected {expect} from all parameters [{ii}, {jj}={nam}], but got {check}!"
                log.exception(f"error in file {ii} {file}")
                log.exception(err)
                raise ValueError(err)

            sample_params[ii, jj] = check

        for fk in fit_keys + fit_med_keys:
            fit_data[fk][ii, ...] = temp[fk][...]

        # Store the GWB from this file
        gwb[ii, :, :] = temp['gwb'][...]
        if debug:
            break

    log.info(f"{utils.frac_str(~good_samp)} files are failures")
    log.info(f"GWB")
    log.info(f"\tall nonzero: {utils.frac_str(all_nonzero)} ")
    log.info(f"\tany nonzero: {utils.frac_str(any_nonzero)} ")
    log.info(f"\ttot nonzero: {utils.frac_str(tot_nonzero)} (realizations)")

    # ---- Save to concatenated output file ----
    out_filename = path_output.joinpath('sam_lib.hdf5')
    log.info(f"Writing collected data to file {out_filename}")
    with h5py.File(out_filename, 'w') as h5:
        h5.create_dataset('fobs', data=fobs)
        h5.create_dataset('fobs_edges', data=fobs_edges)
        h5.create_dataset('gwb', data=gwb)
        h5.create_dataset('sample_params', data=sample_params)
        for fk in fit_keys + fit_med_keys:
            h5.create_dataset(fk, data=fit_data[fk])
        h5.attrs['fit_nbins'] = fit_nbins
        h5.attrs['param_names'] = np.array(param_names).astype('S')
        h5.attrs['shape_names'] = np.array(shape_names).astype('S')
        h5.attrs['librarian_version'] = ", ".join(lib_vers)

    log.warning(f"Saved to {out_filename}, size: {holo.utils.get_file_size(out_filename)}")
    return


def fit_spectra(freqs, gwb, nbins=[5, 10, 15]):
    nf, nreals = np.shape(gwb)
    assert len(freqs) == nf

    num_snaps = len(nbins)
    fit_lamp = np.zeros((nreals, num_snaps))
    fit_plaw = np.zeros((nreals, num_snaps))
    fit_med_lamp = np.zeros((num_snaps))
    fit_med_plaw = np.zeros((num_snaps))
    for ii, num in enumerate(nbins):
        if (num is not None) and (num > nf):
            continue
        num = None if (num == 0) else num
        cut = slice(None, num)
        try:
            xx = freqs[cut]*YR
        except:
            print(f"{freqs=}, {nbins=}")
            raise

        yy = np.median(gwb, axis=-1)
        log10_amp, plaw = _fit_powerlaw(xx, yy[cut], min_val_frac=0.75)

        fit_med_lamp[ii] = log10_amp
        fit_med_plaw[ii] = plaw
        for rr in range(nreals):
            yy = gwb[:, rr]
            log10_amp, plaw = _fit_powerlaw(xx, yy[cut], min_val_frac=0.75)
            fit_lamp[rr, ii] = log10_amp
            fit_plaw[rr, ii] = plaw

    return nbins, fit_lamp, fit_plaw, fit_med_lamp, fit_med_plaw


def _fit_powerlaw(xx, yy, min_val_frac=0.75):
    if (min_val_frac is not None):
        idx = np.isfinite(yy) & (yy > 0.0)
        frac_val = np.count_nonzero(idx) / idx.size
        if frac_val > min_val_frac:
            xx = xx[idx]
            yy = yy[idx]
        else:
            xx = []
            yy = []

    if len(xx) >= 2:
        log10_amp, plaw = utils.fit_powerlaw(xx, yy)
    else:
        log10_amp, plaw = np.nan, np.nan

    return log10_amp, plaw


def get_gwb_fits_data(fobs_cents, gwb):
    # these values must match label construction!
    nbins = [5, 10, 15, 0]

    nbins, lamp, plaw, med_lamp, med_plaw = fit_spectra(fobs_cents, gwb, nbins=nbins)

    label = (
        f"log10(A10)={med_lamp[1]:.2f}, G10={med_plaw[1]:.4f}"
        " | "
        f"log10(A)={med_lamp[-1]:.2f}, G={med_plaw[-1]:.4f}"
    )

    fits_data = dict(
        fit_nbins=nbins, fit_lamp=lamp, fit_plaw=plaw, fit_med_lamp=med_lamp, fit_med_plaw=med_plaw, fit_label=label
    )
    return fits_data


def make_gwb_plot(fobs, gwb, fits_data):
    fig = holo.plot.plot_gwb(fobs, gwb)
    ax = fig.axes[0]

    if len(fits_data):
        xx = fobs * YR
        yy = 1e-15 * np.power(xx, -2.0/3.0)
        ax.plot(xx, yy, 'r-', alpha=0.5, lw=1.0, label="$10^{-15} \cdot f_\\mathrm{yr}^{-2/3}$")

        fits = get_gwb_fits_data(fobs, gwb)

        for ls, idx in zip([":", "--"], [1, -1]):
            med_lamp = fits['fit_med_lamp'][idx]
            med_plaw = fits['fit_med_plaw'][idx]
            yy = (10.0 ** med_lamp) * (xx ** med_plaw)
            label = fits['fit_nbins'][idx]
            label = 'all' if label in [0, None] else label
            ax.plot(xx, yy, color='k', ls=ls, alpha=0.5, lw=2.0, label=str(label) + " bins")

        label = fits['fit_label'].replace(" | ", "\n")
        fig.text(0.99, 0.99, label, fontsize=6, ha='right', va='top')

    return fig


def run_sam_at_pspace_num(args, space, pnum, path_output):
    log = args.log
    fname = f"lib_sams__p{pnum:06d}.npz"
    fname = Path(path_output, fname)
    beg = datetime.now()
    log.info(f"{pnum=} :: {fname=} beginning at {beg}")
    if fname.exists():
        log.warning(f"File {fname} already exists.")

    def log_mem():
        # results.ru_maxrss is KB on Linux, B on macos
        mem_max = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2)
        process = psutil.Process(os.getpid())
        mem_rss = process.memory_info().rss / 1024**3
        mem_vms = process.memory_info().vms / 1024**3
        log.info(f"Current memory usage: max={mem_max:.2f} GB, RSS={mem_rss:.2f} GB, VMS={mem_vms:.2f} GB")

    pta_dur = args.pta_dur * YR
    nfreqs = args.nfreqs
    hifr = nfreqs/pta_dur
    pta_cad = 1.0 / (2 * hifr)
    fobs_cents = holo.utils.nyquist_freqs(pta_dur, pta_cad)
    fobs_edges = holo.utils.nyquist_freqs_edges(pta_dur, pta_cad)
    log.info(f"Created {fobs_cents.size} frequency bins")
    log.info(f"\t[{fobs_cents[0]*YR}, {fobs_cents[-1]*YR}] [1/yr]")
    log.info(f"\t[{fobs_cents[0]*1e9}, {fobs_cents[-1]*1e9}] [nHz]")
    log_mem()
    assert nfreqs == fobs_cents.size

    try:
        log.debug("Selecting `sam` and `hard` instances")
        sam, hard = space(pnum)
        log_mem()
        log.debug(f"Calculating GWB for shape ({fobs_cents.size}, {args.nreals})")
        gwb = sam.gwb(fobs_edges, realize=args.nreals, hard=hard)
        log_mem()
        log.debug(f"{holo.utils.stats(gwb)=}")
        legend = space.param_dict(pnum)
        log.debug(f"Saving {pnum} to file")
        data = dict(fobs=fobs_cents, fobs_edges=fobs_edges, gwb=gwb)
        rv = True
    except Exception as err:
        log.exception("\n\n")
        log.exception("="*100)
        log.exception(f"`run_sam` FAILED on {pnum=}\n")
        log.exception(err)
        log.exception("="*100)
        log.exception("\n\n")
        rv = False
        legend = {}
        data = dict(fail=str(err))

    if rv:
        try:
            fits_data = get_gwb_fits_data(fobs_cents, gwb)
        except Exception as err:
            log.exception("Failed to load gwb fits data!")
            log.exception(err)
            fits_data = {}

    else:
        fits_data = {}

    meta_data = dict(
        pnum=pnum, pdim=space.ndims, nsamples=args.nsamples, librarian_version=__version__,
        param_names=space.names, params=space._params, samples=space._samples, # prob don't need these for all of them
    )

    np.savez(fname, **data, **meta_data, **fits_data, **legend)
    log.info(f"Saved to {fname}, size {holo.utils.get_file_size(fname)} after {(datetime.now()-beg)}")

    if rv:
        try:
            fname = fname.with_suffix('.png')
            fig = make_gwb_plot(fobs_cents, gwb, fits_data)
            fig.savefig(fname, dpi=300)
            log.info(f"Saved to {fname}, size {holo.utils.get_file_size(fname)}")
            plt.close('all')
        except Exception as err:
            log.exception("Failed to make gwb plot!")
            log.exception(err)

    return rv

def run_ss_at_pspace_num(args, space, pnum, path_output):
    log = args.log
    fname = f"lib_sams__p{pnum:06d}.npz"
    fname = Path(path_output, fname)
    beg = datetime.now()
    log.info(f"{pnum=} :: {fname=} beginning at {beg}")
    if fname.exists():
        log.warning(f"File {fname} already exists.")

    def log_mem():
        # results.ru_maxrss is KB on Linux, B on macos
        mem_max = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2)
        process = psutil.Process(os.getpid())
        mem_rss = process.memory_info().rss / 1024**3
        mem_vms = process.memory_info().vms / 1024**3
        log.info(f"Current memory usage: max={mem_max:.2f} GB, RSS={mem_rss:.2f} GB, VMS={mem_vms:.2f} GB")

    pta_dur = args.pta_dur * YR
    nfreqs = args.nfreqs
    hifr = nfreqs/pta_dur
    pta_cad = 1.0 / (2 * hifr)
    fobs_cents = holo.utils.nyquist_freqs(pta_dur, pta_cad)
    fobs_edges = holo.utils.nyquist_freqs_edges(pta_dur, pta_cad)
    log.info(f"Created {fobs_cents.size} frequency bins")
    log.info(f"\t[{fobs_cents[0]*YR}, {fobs_cents[-1]*YR}] [1/yr]")
    log.info(f"\t[{fobs_cents[0]*1e9}, {fobs_cents[-1]*1e9}] [nHz]")
    log_mem()
    assert nfreqs == fobs_cents.size

    try:
        log.debug("Selecting `sam` and `hard` instances")
        sam, hard = space(pnum)
        log_mem()

        ### HERE IS WHERE THINGS CHANGE FOR SS ###
        log.debug(f"Calculating SS and BG GWs for shape ({fobs_cents.size}, {args.nreals})")
        fobs_orb_edges = fobs_edges / 2.0 
        fobs_orb_cents = fobs_cents/ 2.0
        # edges
        edges, dnum = sam.dynamic_binary_number(holo.hardening.Hard_GW, fobs_orb=fobs_orb_cents, zero_stalled=False) # should the zero stalled option be part of the parameter space?
        edges[-1] = fobs_orb_edges
        # integrate for number
        number = utils._integrate_grid_differential_number(edges, dnum, freq=False)
        number = number * np.diff(np.log(fobs_edges))
        # gws
        hc_ss, hc_bg, sspar, bgpar = ss.ss_gws(edges, number, realize=args.nreals, 
                                               loudest = 5, params = True) # replace 5 with args.nloudest
        log_mem()
        log.debug(f"{holo.utils.stats(hc_ss)=}")
        legend = space.param_dict(pnum)
        log.debug(f"Saving {pnum} to file")
        data = dict(fobs=fobs_cents, fobs_edges=fobs_edges, 
                    hc_ss = hc_ss, hc_bg = hc_bg, sspar = sspar, bgpar = bgpar)
        ### EDITED UP TO HERE ###

        rv = True
    except Exception as err:
        log.exception("\n\n")
        log.exception("="*100)
        log.exception(f"`run_sam` FAILED on {pnum=}\n")
        log.exception(err)
        log.exception("="*100)
        log.exception("\n\n")
        rv = False
        legend = {}
        data = dict(fail=str(err))

    if rv:
        try:
            fits_data = get_gwb_fits_data(fobs_cents, hc_bg)
        except Exception as err:
            log.exception("Failed to load gwb fits data!")
            log.exception(err)
            fits_data = {}

    else:
        fits_data = {}

    meta_data = dict(
        pnum=pnum, pdim=space.ndims, nsamples=args.nsamples, librarian_version=__version__,
        param_names=space.names, params=space._params, samples=space._samples, # prob don't need these for all of them
    )

    np.savez(fname, **data, **meta_data, **fits_data, **legend)
    log.info(f"Saved to {fname}, size {holo.utils.get_file_size(fname)} after {(datetime.now()-beg)}")

    if rv:
        try:
            fname = fname.with_suffix('.png')
            fig = make_gwb_plot(fobs_cents, hc_bg, fits_data)
            fig.savefig(fname, dpi=300)
            log.info(f"Saved to {fname}, size {holo.utils.get_file_size(fname)}")
            plt.close('all')
        except Exception as err:
            log.exception("Failed to make gwb plot!")
            log.exception(err)

    return rv

'''
class _Parameter_Space(abc.ABC):

    _PARAM_NAMES = []

    def __init__(self, log, nsamples, sam_shape, **kwargs):

        self.log = log
        self.nsamples = nsamples
        self.sam_shape = sam_shape

        names = []
        params = []

        log.debug(f"Loading parameters: {self._PARAM_NAMES}")
        for par in self._PARAM_NAMES:
            if par not in kwargs:
                err = f"Parameter '{par}' missing from kwargs={kwargs}!"
                log.exception(err)
                raise ValueError(err)

            vv = kwargs.pop(par)
            msg = f"{par}: {vv}"
            log.debug(f"\t{msg}")
            try:
                vv = np.linspace(*vv)
            except Exception as err:
                log.exception(f"Failed to create spacing from: {msg} ({err})")
                raise

            names.append(par)
            params.append(vv)

        self.paramdimen = len(params)
        self.params = params
        self.names = names
        maxints = [tmparr.size for tmparr in params]

        sampleindxs = pyDOE.lhs(n=self.paramdimen, samples=nsamples, criterion='m')

        for i in range(self.paramdimen):
            sampleindxs[:, i] = np.floor(maxints[i] * sampleindxs[:, i])

        sampleindxs = sampleindxs.astype(int)
        log.debug(f"d={len(params)} samplelims={maxints} nsamples={nsamples}")
        self.sampleindxs = sampleindxs

        # self.param_grid = np.meshgrid(*params, indexing='ij')
        # self.shape = self.param_grid[0].shape
        self.shape = tuple([len(pp) for pp in params])
        self.size = np.product(self.shape)
        if self.size < nsamples:
            err = (
                f"There are only {self.size} gridpoints in parameter space but you are requesting "
                f"{nsamples} samples of them. They will be over-sampled!"
            )
            log.warning(err)

        return

    def number_to_index(self, num):
        idx = np.unravel_index(num, self.shape)
        return idx

    def lhsnumber_to_index(self, lhsnum):
        idx = tuple(self.sampleindxs[lhsnum])
        return idx

    def index_to_number(self, idx):
        num = np.ravel_multi_index(idx, self.shape)
        return num

    def params_at_index(self, index):
        assert len(index) == len(self.params)
        pars = [pp[ii] for pp, ii in zip(self.params, index)]
        return pars

    def param_dict_for_number(self, num):
        idx = self.number_to_index(num)
        pars = self.params_at_index(idx)
        rv = {nn: pp for nn, pp in zip(self.names, pars)}
        return rv

    def param_dict_for_lhsnumber(self, lhsnum):
        idx = self.lhsnumber_to_index(lhsnum)
        pars = self.params_at_index(idx)
        rv = {nn: pp for nn, pp in zip(self.names, pars)}
        return rv

    def params_for_number(self, num):
        idx = self.number_to_index(num)
        pars = self.params_at_index(idx)
        return pars

    def params_for_lhsnumber(self, lhsnum):
        idx = self.lhsnumber_to_index(lhsnum)
        pars = self.params_at_index(idx)
        return pars

    # @abc.abstractmethod
    def sam_for_number(self, num):
        raise
        return

    @abc.abstractmethod
    def sam_for_lhsnumber(self, lhsnum):
        return


class _LHS_Parameter_Space(_Parameter_Space):

    _PARAM_NAMES = []

    def __init__(self, log, nsamples, sam_shape, lhs_sampler='scipy', seed=None, **kwargs):

        self.log = log
        self.nsamples = nsamples
        self.sam_shape = sam_shape
        self.lhs_sampler = lhs_sampler
        self.seed = seed

        names = []
        param_ranges = []

        log.debug(f"Loading parameters: {self._PARAM_NAMES}")
        for par in self._PARAM_NAMES:
            if par not in kwargs:
                err = f"Parameter '{par}' missing from kwargs={kwargs}!"
                log.exception(err)
                raise ValueError(err)

            vv = kwargs.pop(par)
            msg = f"{par}: {vv}"
            log.debug(f"\t{msg}")
            if len(vv) > 3 or len(vv) < 2:
                err = f"Wanted 2 arguments in {par}, but got {len(vv)}: {vv}"
                log.exception(err)
                raise ValueError(err)
            elif len(vv) == 3:
                msg = f"Wanted 2 arguments in {par}, but got {len(vv)}: {vv}. I will assume you are using the NON-LHS initialization scheme. Bad scientist!  For LHS, give min and max values, not grid size. I will guess that the first two values are min and max values and ignore the third."
                log.warning(msg)
                vv = vv[0:2]

            names.append(par)
            param_ranges.append(vv)

        self.paramdimen = len(param_ranges)
        self.param_ranges = np.array(param_ranges)
        self.names = names
        self.params = np.zeros((self.nsamples, self.paramdimen))
        # Below is done out of laziness and backwards compatibility but should be deprecated
        self.sampleindxs = -1

        if self.seed is not None:
            log.info(f"Generated with random seed: {self.seed}")
        else:
            log.info(f"Did not use seed. Initializing random state explicitly for reproducibility.")
            np.random.seed(None)
            st0 = np.random.get_state()
            log.info(f"Random state is:\n{st0}")

        # do scipy LHS
        if self.lhs_sampler == 'scipy':
            LHS = qmc.LatinHypercube(d=self.paramdimen, centered=False, strength=1, seed=self.seed)
            # if strength = 2, then n must be equal to p**2, with p prime, and d <= p + 1
            sample_rvs = LHS.random(n=nsamples)
        # do pyDOE LHS
        elif self.lhs_sampler == 'pydoe':
            sample_rvs = pyDOE.lhs(n=self.paramdimen, samples=nsamples, criterion='m')
        else:
            err = f"unknown LHS sampler: {self.lhs_sampler}"
            log.exception(err)
            raise ValueError(err)

        for i in range(self.paramdimen):
            # Assume uniform sampling from min to max of each parameter
            self.params[:, i] = sample_rvs[:, i] * (self.param_ranges[i][1] - self.param_ranges[i][0]) + self.param_ranges[i][0]

    def param_dict_for_lhsnumber(self, num):
        rv = {nn: pp for nn, pp in zip(self.names, self.params[num, :])}
        return rv

    def params_for_lhsnumber(self, num):
        pars = self.params[num, :]
        return pars

    # Below are done out of laziness and backwards compatibility but should be deprecated
    def lhsnumber_to_index(self, pnum):
        return pnum
'''


def main():

    from argparse import ArgumentParser

    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest="subcommand")

    combine = subparsers.add_parser('combine', help='combine output files')
    combine.add_argument('path', default=None)
    combine.add_argument('--debug', '-d', action='store_true', default=False)

    args = parser.parse_args()
    log.debug(f"{args=}")

    if args.subcommand == 'combine':
        sam_lib_combine(args.path, log, args.debug)
    else:
        raise

    return


if __name__ == "__main__":
    main()
