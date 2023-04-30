"""
"""

import abc
from pathlib import Path
from datetime import datetime
import psutil
import resource
import os
import sys

import h5py
import numpy as np
import scipy as sp
import scipy.optimize  # noqa
import matplotlib.pyplot as plt
import tqdm

from scipy.stats import qmc

import holodeck as holo
from holodeck import log, utils
from holodeck.constants import YR


__version__ = "0.2.0"

FITS_NBINS_PLAW = [2, 3, 4, 5, 8, 9, 14]
FITS_NBINS_TURN = [4, 9, 14, 30]

FNAME_SIM_FILE = "lib_sams__p{pnum:06d}.npz"
PSPACE_FILE_SUFFIX = ".pspace.npz"


class _Param_Space(abc.ABC):

    _SAVED_ATTRIBUTES = ["sam_shape", "param_names", "_uniform_samples", "param_samples"]

    def __init__(self, log, nsamples, sam_shape, seed, **kwargs):
        log.debug(f"seed = {seed}")
        np.random.seed(seed)
        # NOTE: this should be saved to output
        random_state = np.random.get_state()
        # log.debug(f"Random state is:\n{random_state}")

        param_names = list(kwargs.keys())
        ndims = len(param_names)
        if ndims == 0:
            err = f"No parameters passed to {self}!"
            log.exception(err)
            raise RuntimeError(err)

        dists = []
        for nam in param_names:
            val = kwargs[nam]

            # if not isinstance(val, _Param_Dist):
            # NOTE: this is a hacky check to see if `val` inherits from `_Param_Dist`
            #       it does this just by checking the string names, but it should generally work.
            #       The motivation is to make this work more easily for changing classes and modules.
            mro = val.__class__.__mro__
            mro = [mm.__name__ for mm in mro]
            if holo.librarian._Param_Dist.__name__ not in mro:
                err = f"{nam}: {val} is not a `_Param_Dist` object!"
                log.exception(err)
                raise ValueError(err)

            dists.append(val)

        # if strength = 2, then n must be equal to p**2, with p prime, and d <= p + 1
        lhs = qmc.LatinHypercube(d=ndims, centered=False, strength=1, seed=seed)
        # (S, D) - samples, dimensions
        uniform_samples = lhs.random(n=nsamples)
        param_samples = np.zeros_like(uniform_samples)

        for ii, dist in enumerate(dists):
            param_samples[:, ii] = dist(uniform_samples[:, ii])

        self._log = log
        self.param_names = param_names
        self.sam_shape = sam_shape
        self.param_samples = param_samples
        self._dists = dists
        self._uniform_samples = uniform_samples
        self._seed = seed
        self._random_state = random_state
        return

    def save(self, path_output):
        """Save the generated samples and parameter-space info from this instance to an output file.

        This data can then be loaded using the `_Param_Space.from_save` method.

        Arguments
        ---------
        path_output : str
            Path in which to save file.  This must be an existing directory.

        Returns
        -------
        fname : str
            Output path including filename in which this parameter-space was saved.

        """
        log = self._log
        my_name = self.__class__.__name__
        vers = __version__

        # make sure `path_output` is a directory, and that it exists
        path_output = Path(path_output)
        if not path_output.exists() or not path_output.is_dir():
            err = f"save path {path_output} does not exist, or is not a directory!"
            log.exception(err)
            raise ValueError(err)

        fname = f"{my_name.lower()}{PSPACE_FILE_SUFFIX}"
        fname = path_output.joinpath(fname)
        log.debug(f"{my_name=} {vers=} {fname=}")

        data = {}
        for key in self._SAVED_ATTRIBUTES:
            data[key] = getattr(self, key)

        np.savez(
            fname, class_name=my_name, librarian_version=vers,
            **data,
        )

        log.info(f"Saved to {fname} size {utils.get_file_size(fname)}")
        return fname

    @classmethod
    def from_save(cls, fname, log):
        """Create a new _Param_Space instance loaded from the given file.

        Arguments
        ---------
        fname : str
            Filename containing parameter-space save information, generated form `_Param_Space.save`.

        Returns
        -------
        space : `_Param_Space` instance

        """
        log.debug(f"loading parameter space from {fname}")
        data = np.load(fname)

        # get the name of the parameter-space class from the file, and try to find this class in the
        # `holodeck.param_spaces` module
        class_name = data['class_name'][()]
        log.debug(f"loaded: {class_name=}, vers={data['librarian_version']}")
        pspace_class = getattr(holo.param_spaces, class_name, None)
        # if it is not found, default to the current class/subclass
        if pspace_class is None:
            log.warning(f"pspace file {fname} has {class_name=}, not found in `holo.param_spaces`!")
            pspace_class = cls

        # construct instance with dummy/temporary values (which will be overwritten)
        space = pspace_class(log, 10, 10, None)
        if class_name != space.__class__.__name__:
            err = "loaded class name '{class_name}' does not match this class name '{space.__name__}'!"
            log.warning(err)
            # raise RuntimeError(err)

        # Store loaded parameters into the parameter-space instance
        for key in space._SAVED_ATTRIBUTES:
            setattr(space, key, data[key][()])

        return space

    def params(self, samp_num):
        return self.param_samples[samp_num]

    def param_dict(self, samp_num):
        rv = {nn: pp for nn, pp in zip(self.param_names, self.params(samp_num))}
        return rv

    def __call__(self, samp_num):
        return self.model_for_number(samp_num)

    @property
    def shape(self):
        return self.param_samples.shape

    @property
    def nsamples(self):
        return self.shape[0]

    @property
    def npars(self):
        return self.shape[1]

    def model_for_number(self, samp_num):
        params = self.param_dict(samp_num)
        self._log.debug(f"params {samp_num} :: {params}")
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
        # self._dist_func = lambda xx: self._lo + (self._hi - self._lo) * xx
        return

    def _dist_func(self, xx):
        yy = self._lo + (self._hi - self._lo) * xx
        return yy


class PD_Uniform_Log(_Param_Dist):

    def __init__(self, lo, hi, **kwargs):
        super().__init__(**kwargs)
        assert lo > 0.0 and hi > 0.0
        self._lo = np.log10(lo)
        self._hi = np.log10(hi)
        # self._dist_func = lambda xx: np.power(10.0, self._lo + (self._hi - self._lo) * xx)
        return

    def _dist_func(self, xx):
        yy = np.power(10.0, self._lo + (self._hi - self._lo) * xx)
        return yy


class PD_Normal(_Param_Dist):

    def __init__(self, mean, stdev, clip=None, **kwargs):
        super().__init__(**kwargs)
        assert stdev > 0.0
        if clip is not None:
            if len(clip) != 2:
                err = f"{clip=} | `clip` must be (2,) values of lo and hi bounds at which to clip!"
                log.exception(err)
                raise ValueError(err)

        self._mean = mean
        self._stdev = stdev
        self._clip = clip
        self._dist = sp.stats.norm(loc=mean, scale=stdev)
        # if clip is not None:
        #     if len(clip) != 2:
        #         err = f"{clip=} | `clip` must be (2,) values of lo and hi bounds at which to clip!"
        #         log.exception(err)
        #         raise ValueError(err)
        #     self._dist_func = lambda xx: np.clip(self._dist.ppf(xx), *clip)
        # else:
        #     self._dist_func = lambda xx: self._dist.ppf(xx)

        return

    def _dist_func(self, xx):
        clip = self.clip
        if clip is not None:
            yy = np.clip(self._dist.ppf(xx), *clip)
        else:
            yy = self._dist.ppf(xx)
        return yy


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


def sam_lib_combine(path_output, log, path_sims=None, path_pspace=None):
    """

    Arguments
    ---------
    path_output : str or Path,
        Path to output directory where combined library will be saved.
    log : `logging.Logger`
        Logging instance.
    path_sims : str or None,
        Path to output directory containing simulation files.
        If `None` this is set to be the same as `path_output`.
    path_pspace : str or None,
        Path to file containing _Param_Space subclass instance.
        If `None` then `path_output` is searched for a `_Param_Space` save file.

    Returns
    -------
    out_filename : Path,
        Path to library output filename (typically ending with 'sam_lib.hdf5').

    """

    # ---- setup paths

    path_output = Path(path_output)
    log.info(f"Path output = {path_output}")
    # if dedicated simulation path is not given, assume same as general output path
    if path_sims is None:
        path_sims = path_output
    path_sims = Path(path_sims)
    log.info(f"Path sims = {path_sims}")

    # ---- load parameter space from save file

    if path_pspace is None:
        # look for parameter-space save files
        regex = "*" + PSPACE_FILE_SUFFIX   # "*.pspace.npz"
        files = sorted(path_output.glob(regex))
        num_files = len(files)
        msg = f"found {num_files} pspace.npz files in {path_output}"
        log.info(msg)
        if num_files != 1:
            log.exception(f"")
            log.exception(msg)
            log.exception(f"{files=}")
            log.exception(f"{regex=}")
            raise RuntimeError(f"{msg}")
        path_pspace = files[0]

    pspace = _Param_Space.from_save(path_pspace, log)
    log.info(f"loaded param space: {pspace}")
    param_names = pspace.param_names
    param_samples = pspace.param_samples
    nsamp, ndim = param_samples.shape
    log.debug(f"{nsamp=}, {ndim=}, {param_names=}")

    # ---- make sure all files exist; get shape information from files

    log.info(f"checking that all {nsamp} files exist")
    fobs, nreals, fit_data = _check_files_and_load_shapes(path_sims, nsamp)
    nfreqs = fobs.size
    log.debug(f"{nfreqs=}, {nreals=}")

    # ---- Store results from all files

    gwb = np.zeros((nsamp, nfreqs, nreals))
    gwb, fit_data, bad_files = _load_library_from_all_files(path_sims, gwb, fit_data, log)
    param_samples[bad_files] = 0.0
    if fit_data is None:
        msg = "`fit_data` is None, fits have failed.  Attempting to combine data anyway."
        log.error(msg)
        fit_data = {}

    # ---- Save to concatenated output file ----

    out_filename = path_output.joinpath('sam_lib.hdf5')
    log.info(f"Writing collected data to file {out_filename}")
    with h5py.File(out_filename, 'w') as h5:
        h5.create_dataset('fobs', data=fobs)
        h5.create_dataset('gwb', data=gwb)
        h5.create_dataset('sample_params', data=param_samples)
        for kk, vv in fit_data.items():
            h5.create_dataset(kk, data=vv)
        h5.attrs['param_names'] = np.array(param_names).astype('S')

    log.warning(f"Saved to {out_filename}, size: {holo.utils.get_file_size(out_filename)}")

    return out_filename


def _check_files_and_load_shapes(path_sims, nsamp):
    """Check that all `nsamp` files exist in the given path, and load info about array shapes.

    Arguments
    ---------
    path_sims : str
        Path in which individual simulation files can be found.
    nsamp : int
        Number of simulations/files that should be found.
        This should typically be loaded from the parameter-space object used to generate the library.

    Returns
    -------
    fobs : (F,) ndarray
        Observer-frame frequency bin centers at which GW signals are calculated.
    nreals : int
        Number of realizations in the output files.
    fit_data : dict
        Dictionary where each key is a fit-parameter in all of the output files.  The values are
        'ndarray's of the appropriate shapes to store fit-parameters from all files.
        The 0th dimension is always for the number-of-files.

    """

    fobs = None
    nreals = None
    fit_data = None
    for ii in tqdm.trange(nsamp):
        temp = _sim_fname(path_sims, ii)
        if not temp.exists():
            err = f"Missing at least file number {ii} out of {nsamp} files!  {temp}"
            log.exception(err)
            raise ValueError(err)

        # if we've already loaded all of the necessary info, then move on to the next file
        if (fobs is not None) and (nreals is not None) and (fit_data is not None):
            continue

        temp = np.load(temp)
        data_keys = temp.keys()

        if fobs is None:
            fobs = temp['fobs'][()]

        # find a file that has GWB data in it (not all of them do, if the file was a 'failure' file)
        if (nreals is None) and ('gwb' in data_keys):
            nreals = temp['gwb'].shape[-1]

        # find a file that has fits data in it (it's possible for the fits portion to fail by itself)
        # initialize arrays to store output data for all files
        if (fit_data is None) and np.any([kk.startswith('fit_') for kk in data_keys]):
            fit_data = {}
            for kk in data_keys:
                if not kk.startswith('fit_'):
                    continue

                vv = temp[kk]
                # arrays need to store values for 'nsamp' files
                shape = (nsamp,) + vv.shape
                fit_data[kk] = np.zeros(shape)

    if fit_data is not None:
        for kk, vv in fit_data.items():
            log.debug(f"\t{kk:>20s}: {vv.shape}")

    return fobs, nreals, fit_data


def _load_library_from_all_files(path_sims, gwb, fit_data, log):
    """Load data from all individual simulation files.

    Arguments
    ---------
    path_sims : str
        Path to find individual simulation files.
    gwb : (S, F, R) ndarray
        Array in which to store GWB data from all of 'S' files.
        S: num-samples/simulations,  F: num-frequencies,  R: num-realizations.
    fit_data : dict
        Dictionary of ndarrays in which to store fit-parameters.
    log : `logging.Logger`
        Logging instance.

    """

    nsamp = gwb.shape[0]
    log.info(f"Collecting data from {nsamp} files")
    bad_files = np.zeros(nsamp, dtype=bool)     #: track which files contain UN-useable data
    num_fits_failed = 0
    msg = None
    for pnum in tqdm.trange(nsamp):
        fname = _sim_fname(path_sims, pnum)
        temp = np.load(fname, allow_pickle=True)
        # When a processor fails for a given parameter, the output file is still created with the 'fail' key added
        if ('fail' in temp) or ('gwb' not in temp):
            msg = f"file {pnum=:06d} is a failure file, setting values to NaN ({fname})"
            log.warning(msg)
            # set all parameters to NaN for failure files.  Note that this is distinct from gwb=0.0 which can be real.
            gwb[pnum, :, :] = np.nan
            for fk in fit_data.keys():
                fit_data[fk][pnum, ...] = np.nan

            bad_files[pnum] = True
            continue

        # store the GWB from this file
        gwb[pnum, :, :] = temp['gwb'][...]

        # store all of the fit data
        fits_bad = False
        for fk in fit_data.keys():
            try:
                fit_data[fk][pnum, ...] = temp[fk][...]
            except Exception as err:
                # only count the first time it fails
                if not fits_bad:
                    num_fits_failed += 1
                fits_bad = True
                fit_data[fk][pnum, ...] = np.nan
                msg = str(err)

        if fits_bad:
            log.warning(f"Missing fit keys in file {pnum} = {fname.name}")

    if num_fits_failed > 0:
        log.warning(f"Missing fit keys in {num_fits_failed}/{nsamp} = {num_fits_failed/nsamp:.2e} files!")
        log.warning(msg)

    log.info(f"{utils.frac_str(bad_files)} files are failures")

    return gwb, fit_data, bad_files


def _fit_spectra(freqs, psd, nbins, nfit_pars, fit_func, min_nfreq_valid):
    nfreq, nreals = np.shape(psd)
    assert len(freqs) == nfreq

    def fit_if_all_finite(xx, yy):
        if np.any(~np.isfinite(yy)):
            pars = [np.nan] * nfit_pars
        else:
            sel = (yy > 0.0)
            if np.count_nonzero(sel) < min_nfreq_valid:
                pars = [np.nan] * nfit_pars
            else:
                pars = fit_func(xx[sel], yy[sel])
        return pars

    nfreq_bins = len(nbins)
    fit_pars = np.zeros((nfreq_bins, nreals, nfit_pars))
    fit_med_pars = np.zeros((nfreq_bins, nfit_pars))
    for ii, num in enumerate(nbins):
        if num > nfreq:
            raise ValueError(f"Cannot fit for {num=} bins, data has {nfreq=} frequencies!")

        num = None if (num == 0) else num
        cut = slice(None, num)
        xx = freqs[cut]

        # fit the median spectra
        yy = np.median(psd, axis=-1)[cut]
        fit_med_pars[ii] = fit_if_all_finite(xx, yy)

        # fit each realization of the spectra
        for rr in range(nreals):
            yy = psd[cut, rr]
            fit_pars[ii, rr, :] = fit_if_all_finite(xx, yy)

    return nbins, fit_pars, fit_med_pars


def fit_spectra_plaw(freqs, psd, nbins):
    fit_func = lambda xx, yy: utils.fit_powerlaw_psd(xx, yy, 1/YR)
    nfit_pars = 2
    min_nfreq_valid = 2
    return _fit_spectra(freqs, psd, nbins, nfit_pars, fit_func, min_nfreq_valid)


def fit_spectra_turn(freqs, psd, nbins):
    fit_func = lambda xx, yy: utils.fit_turnover_psd(xx, yy, 1/YR)
    min_nfreq_valid = 3
    nfit_pars = 4
    return _fit_spectra(freqs, psd, nbins, nfit_pars, fit_func, min_nfreq_valid)


def fit_spectra_plaw_hc(freqs, gwb, nbins):
    fit_func = lambda xx, yy: utils.fit_powerlaw(xx, yy)
    min_nfreq_valid = 2
    nfit_pars = 2
    return _fit_spectra(freqs, gwb, nbins, nfit_pars, fit_func, min_nfreq_valid)


def make_gwb_plot(fobs, gwb, fit_data):
    # fig = holo.plot.plot_gwb(fobs, gwb)
    psd = utils.char_strain_to_psd(fobs[:, np.newaxis], gwb)
    fig = holo.plot.plot_gwb(fobs, psd)
    ax = fig.axes[0]

    xx = fobs * YR
    yy = 1e-15 * np.power(xx, -2.0/3.0)
    ax.plot(xx, yy, 'k--', alpha=0.5, lw=1.0, label=r"$10^{-15} \cdot f_\mathrm{yr}^{-2/3}$")

    if len(fit_data) > 0:
        fit_nbins = fit_data['fit_plaw_nbins']
        med_pars = fit_data['fit_plaw_med']

        plot_nbins = [4, 14]

        for nbins in plot_nbins:
            idx = fit_nbins.index(nbins)
            pars = med_pars[idx]

            pars[0] = 10.0 ** pars[0]
            yy = holo.utils._func_powerlaw_psd(fobs, 1/YR, *pars)
            label = fit_nbins[idx]
            label = 'all' if label in [0, None] else f"{label:02d}"
            ax.plot(xx, yy, alpha=0.75, lw=1.0, label="plaw: " + str(label) + " bins", ls='--')

        fit_nbins = fit_data['fit_turn_nbins']
        med_pars = fit_data['fit_turn_med']

        plot_nbins = [14, 30]

        for nbins in plot_nbins:
            idx = fit_nbins.index(nbins)
            pars = med_pars[idx]

            pars[0] = 10.0 ** pars[0]
            zz = holo.utils._func_turnover_psd(fobs, 1/YR, *pars)
            label = fit_nbins[idx]
            label = 'all' if label in [0, None] else f"{label:02d}"
            ax.plot(xx, zz, alpha=0.75, lw=1.0, label="turn: " + str(label) + " bins")

    ax.legend(fontsize=6, loc='upper right')

    return fig


def run_sam_at_pspace_num(args, space, pnum):
    """Run the SAM simulation for sample-parameter `pnum` in the `space` parameter-space.

    Arguments
    ---------
    args : `argparse.ArgumentParser` instance
        Arguments from the `gen_lib_sams.py` script.
        NOTE: this should be improved.
    space : _Param_Space instance
        Parameter space from which to load `sam` and `hard` instances.
    pnum : int
        Which parameter-sample from `space` should be run.

    Returns
    -------
    rv : bool
        True if this simulation was successfully run.

    """

    log = args.log

    # ---- get output filename for this simulation, check if already exists

    sim_fname = _sim_fname(args.output_sims, pnum)
    beg = datetime.now()
    log.info(f"{pnum=} :: {sim_fname=} beginning at {beg}")
    if sim_fname.exists():
        log.info(f"File {sim_fname} already exists.  {args.recreate=}")
        # skip existing files unless we specifically want to recreate them
        if not args.recreate:
            return True

    # ---- Setup PTA frequencies

    pta_dur = args.pta_dur * YR
    nfreqs = args.nfreqs
    hifr = nfreqs/pta_dur
    pta_cad = 1.0 / (2 * hifr)
    fobs_cents = holo.utils.nyquist_freqs(pta_dur, pta_cad)
    fobs_edges = holo.utils.nyquist_freqs_edges(pta_dur, pta_cad)
    log.info(f"Created {fobs_cents.size} frequency bins")
    log.info(f"\t[{fobs_cents[0]*YR}, {fobs_cents[-1]*YR}] [1/yr]")
    log.info(f"\t[{fobs_cents[0]*1e9}, {fobs_cents[-1]*1e9}] [nHz]")
    _log_mem_usage(log)
    assert nfreqs == fobs_cents.size

    # ---- Calculate GWB from SAM

    try:
        log.debug("Selecting `sam` and `hard` instances")
        sam, hard = space(pnum)
        _log_mem_usage(log)
        log.debug(f"Calculating GWB for shape ({fobs_cents.size}, {args.nreals})")
        gwb = sam.gwb(fobs_edges, realize=args.nreals, hard=hard)
        _log_mem_usage(log)
        log.debug(f"{holo.utils.stats(gwb)=}")
        log.debug(f"Saving {pnum} to file")
        data = dict(fobs=fobs_cents, gwb=gwb)
        rv = True
    except Exception as err:
        log.exception(f"`run_sam` FAILED on {pnum=}\n")
        log.exception(err)
        rv = False
        data = dict(fail=str(err))

    # ---- Fit GWB spectra

    fit_data = {}
    if rv:
        log.info("calculating spectra fits")
        try:
            psd = utils.char_strain_to_psd(fobs_cents[:, np.newaxis], gwb)
            plaw_nbins, fit_plaw, fit_plaw_med = fit_spectra_plaw(fobs_cents, psd, FITS_NBINS_PLAW)
            turn_nbins, fit_turn, fit_turn_med = fit_spectra_turn(fobs_cents, psd, FITS_NBINS_TURN)

            fit_data = dict(
                fit_plaw_nbins=plaw_nbins, fit_plaw=fit_plaw, fit_plaw_med=fit_plaw_med,
                fit_turn_nbins=turn_nbins, fit_turn=fit_turn, fit_turn_med=fit_turn_med,
            )
        except Exception as err:
            log.error("Failed to load gwb fits data!")
            log.error(err)
            if "Number of calls to function has reached maxfev" in err.value:
                log.exception("fit did not converge.")
            else:
                log.exception(err)

    # ---- Save data to file

    np.savez(sim_fname, **data, **fit_data)
    log.info(f"Saved to {sim_fname}, size {holo.utils.get_file_size(sim_fname)} after {(datetime.now()-beg)}")

    # ---- Plot GWB spectra

    if rv and args.plot:
        log.info("generating spectra plots")
        try:
            plot_fname = args.output_plots.joinpath(sim_fname.name)
            plot_fname = plot_fname.with_suffix('.png')

            fig = make_gwb_plot(fobs_cents, gwb, fit_data)
            fig.savefig(plot_fname, dpi=100)
            log.info(f"Saved to {plot_fname}, size {holo.utils.get_file_size(plot_fname)}")
            plt.close('all')
        except Exception as err:
            log.exception("Failed to make gwb plot!")
            log.exception(err)

    return rv


def _sim_fname(path, pnum):
    temp = FNAME_SIM_FILE.format(pnum=pnum)
    temp = path.joinpath(temp)
    return temp


def _log_mem_usage(log):
    # results.ru_maxrss is KB on Linux, B on macos
    mem_max = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform.lower().startswith('darwin'):
        mem_max = (mem_max / 1024 ** 3)
    else:
        mem_max = (mem_max / 1024 ** 2)

    process = psutil.Process(os.getpid())
    mem_rss = process.memory_info().rss / 1024**3
    mem_vms = process.memory_info().vms / 1024**3

    msg = f"Current memory usage: max={mem_max:.2f} GB, RSS={mem_rss:.2f} GB, VMS={mem_vms:.2f} GB"
    if log is None:
        print(msg, flush=True)
    else:
        log.info(msg)

    return


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
        sam_lib_combine(args.path, log, path_sims=Path(args.path).joinpath('sims'))
    else:
        raise

    return


if __name__ == "__main__":
    main()
