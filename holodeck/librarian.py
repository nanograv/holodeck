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

import holodeck as holo
import holodeck.single_sources as ss
from holodeck import log, utils
from holodeck.constants import YR


__version__ = "0.1.1"

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
    gwb, fit_data = _load_library_from_all_files(path_sims, gwb, fit_data, log)

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
    good_file = np.ones(nsamp, dtype=bool)     #: track which files contain useable data
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

            good_file[pnum] = False
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

    log.info(f"{utils.frac_str(~good_file)} files are failures")

    return gwb, fit_data

def ss_lib_combine(path_output, log, get_pars, debug=False):
    path_output = Path(path_output)
    log.info(f"Path output = {path_output}")

    regex = "lib_ss__p*.npz"
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

    # ---- Find and check an example data file
    idx_template = 0
    temp = files[idx_template]
    data = np.load(temp, allow_pickle=True)
    while ('fail' in data) or ('hc_bg' not in data) or ('hc_ss' not in data):
        idx_template += 1
        try:
            temp = files[idx_template]
        except IndexError as err:
            log.error(err)
            log.exception("All {len(files)} files are failed! Cannot find a template!")
            raise # do we not specify what to raise, e.g. "raise IndexError(err)" since we already defined it?
        data = np.load(temp, allow_pickle=True)

    log.info(f"Test file: {temp}\n\tkeys: {list(data.keys())}")
    fobs = data['fobs']
    fobs_edges = data['fobs_edges']
    nfreqs = fobs.size

    temp_hc_ss = data['hc_ss'][:]
    temp_hc_bg = data['hc_bg'][:]
    assert np.ndim(temp_hc_bg) == 2
    assert np.ndim(temp_hc_ss) == 3
    if(get_pars):
        temp_sspar = data['sspar'][:]
        temp_bgpar = data['bgpar'][:]
        assert np.ndim(temp_sspar) == 4
        assert np.ndim(temp_bgpar) == 3
    _nfreqs, nreals, nloudest = temp_hc_ss.shape
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

    bg_shape = [num_files, nfreqs, nreals]
    bg_shape_names = ['params', 'freqs', 'reals']
    ss_shape = [num_files, nfreqs, nreals, nloudest]
    ss_shape_names = ['params', 'freqs', 'reals', 'loudest']
    hc_ss = np.zeros(ss_shape)
    hc_bg = np.zeros(bg_shape)
    if(get_pars):
        sspar = np.zeros([num_files, 3, nfreqs, nreals, nloudest])
        bgpar = np.zeros([num_files, 3, nfreqs, nreals])
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
        # NOTE: This is currently only checking for hc_bg, not hc_ss, sspar, or bgpar.
        if ('fail' in temp) or ('hc_bg' not in temp):
            msg = f"file {ii=:06d} is a failure file, setting values to NaN ({file})"
            log.warning(msg)
            hc_bg[ii, :, :] = np.nan
            for fk in fit_keys + fit_med_keys:
                fit_data[fk][ii, ...] = np.nan

            good_samp[ii] = False
            continue

        this_hc_bg = temp['hc_bg']
        all_nonzero[ii] = np.all(this_hc_bg > 0.0)
        any_nonzero[ii] = np.any(this_hc_bg > 0.0)
        tot_nonzero[ii, :] = np.all(this_hc_bg > 0.0, axis=0)

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

        # Store the hc_ss, hc_bg, sspar, bgpar from this file
        hc_ss[ii, :, :, :] = temp['hc_ss'][...]
        hc_bg[ii, :, :] = temp['hc_bg'][...]
        if(get_pars):
            sspar[ii, :, :, :, :] = temp['sspar'][...]
            bgpar[ii, :, :, :] = temp['bgpar'][...]
        if debug:
            break

    log.info(f"{utils.frac_str(~good_samp)} files are failures")
    log.info(f"GWB")
    log.info(f"\tall nonzero: {utils.frac_str(all_nonzero)} ")
    log.info(f"\tany nonzero: {utils.frac_str(any_nonzero)} ")
    log.info(f"\ttot nonzero: {utils.frac_str(tot_nonzero)} (realizations)")

    # ---- Save to concatenated output file ----
    out_filename = path_output.joinpath('ss_lib.hdf5')
    log.info(f"Writing collected data to file {out_filename}")
    with h5py.File(out_filename, 'w') as h5:
        h5.create_dataset('fobs', data=fobs)
        h5.create_dataset('fobs_edges', data=fobs_edges)
        h5.create_dataset('hc_ss', data=hc_ss)
        h5.create_dataset('hc_bg', data=hc_bg)
        if(get_pars):
            h5.create_dataset('sspar', data=sspar)
            h5.create_dataset('bgpar', data=bgpar)

        h5.create_dataset('sample_params', data=sample_params)
        for fk in fit_keys + fit_med_keys:
            h5.create_dataset(fk, data=fit_data[fk])
        h5.attrs['fit_nbins'] = fit_nbins
        h5.attrs['param_names'] = np.array(param_names).astype('S')
        h5.attrs['ss_shape_names'] = np.array(ss_shape_names).astype('S')
        h5.attrs['bg_shape_names'] = np.array(bg_shape_names).astype('S')
        h5.attrs['librarian_version'] = ", ".join(lib_vers)

    log.warning(f"Saved to {out_filename}, size: {holo.utils.get_file_size(out_filename)}")
    return


def ss_lib_combine(path_output, log, get_pars, debug=False):
    path_output = Path(path_output)
    log.info(f"Path output = {path_output}")

    regex = "lib_ss__p*.npz"
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

    # ---- Find and check an example data file
    idx_template = 0
    temp = files[idx_template]
    data = np.load(temp, allow_pickle=True)
    while ('fail' in data) or ('hc_bg' not in data) or ('hc_ss' not in data):
        idx_template += 1
        try:
            temp = files[idx_template]
        except IndexError as err:
            log.error(err)
            log.exception("All {len(files)} files are failed! Cannot find a template!")
            raise # do we not specify what to raise, e.g. "raise IndexError(err)" since we already defined it?
        data = np.load(temp, allow_pickle=True)

    log.info(f"Test file: {temp}\n\tkeys: {list(data.keys())}")
    fobs = data['fobs']
    fobs_edges = data['fobs_edges']
    nfreqs = fobs.size

    temp_hc_ss = data['hc_ss'][:]
    temp_hc_bg = data['hc_bg'][:]
    assert np.ndim(temp_hc_bg) == 2
    assert np.ndim(temp_hc_ss) == 3
    if(get_pars):
        temp_sspar = data['sspar'][:]
        temp_bgpar = data['bgpar'][:]
        assert np.ndim(temp_sspar) == 4
        assert np.ndim(temp_bgpar) == 3
    _nfreqs, nreals, nloudest = temp_hc_ss.shape
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

    bg_shape = [num_files, nfreqs, nreals]
    bg_shape_names = ['params', 'freqs', 'reals']
    ss_shape = [num_files, nfreqs, nreals, nloudest]
    ss_shape_names = ['params', 'freqs', 'reals', 'loudest']
    hc_ss = np.zeros(ss_shape)
    hc_bg = np.zeros(bg_shape)
    if(get_pars):
        sspar = np.zeros([num_files, 3, nfreqs, nreals, nloudest])
        bgpar = np.zeros([num_files, 3, nfreqs, nreals])
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
        # NOTE: This is currently only checking for hc_bg, not hc_ss, sspar, or bgpar.
        if ('fail' in temp) or ('hc_bg' not in temp):
            msg = f"file {ii=:06d} is a failure file, setting values to NaN ({file})"
            log.warning(msg)
            hc_bg[ii, :, :] = np.nan
            for fk in fit_keys + fit_med_keys:
                fit_data[fk][ii, ...] = np.nan

            good_samp[ii] = False
            continue

        this_hc_bg = temp['hc_bg']
        all_nonzero[ii] = np.all(this_hc_bg > 0.0)
        any_nonzero[ii] = np.any(this_hc_bg > 0.0)
        tot_nonzero[ii, :] = np.all(this_hc_bg > 0.0, axis=0)

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

        # Store the hc_ss, hc_bg, sspar, bgpar from this file
        hc_ss[ii, :, :, :] = temp['hc_ss'][...]
        hc_bg[ii, :, :] = temp['hc_bg'][...]
        if(get_pars):
            sspar[ii, :, :, :, :] = temp['sspar'][...]
            bgpar[ii, :, :, :] = temp['bgpar'][...]
        if debug:
            break

    log.info(f"{utils.frac_str(~good_samp)} files are failures")
    log.info(f"GWB")
    log.info(f"\tall nonzero: {utils.frac_str(all_nonzero)} ")
    log.info(f"\tany nonzero: {utils.frac_str(any_nonzero)} ")
    log.info(f"\ttot nonzero: {utils.frac_str(tot_nonzero)} (realizations)")

    # ---- Save to concatenated output file ----
    out_filename = path_output.joinpath('ss_lib.hdf5')
    log.info(f"Writing collected data to file {out_filename}")
    with h5py.File(out_filename, 'w') as h5:
        h5.create_dataset('fobs', data=fobs)
        h5.create_dataset('fobs_edges', data=fobs_edges)
        h5.create_dataset('hc_ss', data=hc_ss)
        h5.create_dataset('hc_bg', data=hc_bg)
        if(get_pars):
            h5.create_dataset('sspar', data=sspar)
            h5.create_dataset('bgpar', data=bgpar)

        h5.create_dataset('sample_params', data=sample_params)
        for fk in fit_keys + fit_med_keys:
            h5.create_dataset(fk, data=fit_data[fk])
        h5.attrs['fit_nbins'] = fit_nbins
        h5.attrs['param_names'] = np.array(param_names).astype('S')
        h5.attrs['ss_shape_names'] = np.array(ss_shape_names).astype('S')
        h5.attrs['bg_shape_names'] = np.array(bg_shape_names).astype('S')
        h5.attrs['librarian_version'] = ", ".join(lib_vers)

    log.warning(f"Saved to {out_filename}, size: {holo.utils.get_file_size(out_filename)}")
    return


def fit_spectra_plaw_hc(freqs, gwb, nbins):
    nfreq, nreals = np.shape(gwb)
    assert len(freqs) == nfreq

    def fit_if_all_finite(xx, yy):
        if np.any(~np.isfinite(yy)):
            pars = np.nan, np.nan
        else:
            pars = utils.fit_powerlaw(xx, yy)
        return pars

    nfreq_bins = len(nbins)
    fit_pars = np.zeros((nfreq_bins, nreals, 2))
    fit_med_pars = np.zeros((nfreq_bins, 2))
    for ii, num in enumerate(nbins):
        if num > nfreq:
            raise ValueError(f"Cannot fit for {num=} bins, data has {nfreq=} frequencies!")

        num = None if (num == 0) else num
        cut = slice(None, num)
        xx = freqs[cut]*YR

        # fit the median spectra
        yy = np.median(gwb, axis=-1)[cut]
        fit_med_pars[ii] = fit_if_all_finite(xx, yy)

        # fit each realization of the spectra
        for rr in range(nreals):
            yy = gwb[cut, rr]
            fit_pars[ii, rr, :] = fit_if_all_finite(xx, yy)

    return nbins, fit_pars, fit_med_pars


def fit_spectra_plaw(freqs, psd, nbins):
    nfreq, nreals = np.shape(psd)
    assert len(freqs) == nfreq

    def fit_if_all_finite(xx, yy):
        if np.any(~np.isfinite(yy)):
            pars = np.nan, np.nan
        else:
            pars = utils.fit_powerlaw_psd(xx, yy, 1/YR)
        return pars

    nfreq_bins = len(nbins)
    fit_pars = np.zeros((nfreq_bins, nreals, 2))
    fit_med_pars = np.zeros((nfreq_bins, 2))
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


def fit_spectra_turn(freqs, psd, nbins):
    nfreq, nreals = np.shape(psd)
    assert len(freqs) == nfreq

    def fit_if_all_finite(xx, yy):
        if np.any(~np.isfinite(yy)):
            pars = np.nan, np.nan
        else:
            pars = utils.fit_turnover_psd(xx, yy, 1/YR)
        return pars

    nfreq_bins = len(nbins)
    fit_pars = np.zeros((nfreq_bins, nreals, 4))
    fit_med_pars = np.zeros((nfreq_bins, 4))
    for ii, num in enumerate(nbins):
        if num > nfreq:
            raise ValueError(f"Cannot fit for {num=} bins, data has {nfreq=} frequencies!")

        num = None if (num == 0) else num
        cut = slice(None, num)
        xx = freqs[cut]

        # fit the median spectra
        yy = np.median(psd, axis=-1)[cut]
        fit_med_pars[ii, :] = fit_if_all_finite(xx, yy)

        # fit each realization of the spectra
        for rr in range(nreals):
            yy = psd[cut, rr]
            fit_pars[ii, rr, :] = fit_if_all_finite(xx, yy)

    return nbins, fit_pars, fit_med_pars


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

def get_hc_bg_fits_data(fobs_cents, gwb):
    # these values must match label construction!
    nbins = [5, 10, 15, 0]

    nbins, lamp, plaw, med_lamp, med_plaw = fit_spectra_plaw_hc(fobs_cents, gwb, nbins=nbins)

    label = (
        f"log10(A10)={med_lamp[1]:.2f}, G10={med_plaw[1]:.4f}"
        " | "
        f"log10(A)={med_lamp[-1]:.2f}, G={med_plaw[-1]:.4f}"
    )

    fits_data = dict(
        fit_nbins=nbins, fit_lamp=lamp, fit_plaw=plaw, fit_med_lamp=med_lamp, fit_med_plaw=med_plaw, fit_label=label
    )
    return fits_data

def make_ss_plot(fobs, hc_ss, hc_bg, fits_data):
    fig = holo.plot.plot_gwb(fobs, gwb=hc_bg, hc_ss=hc_ss)
    ax = fig.axes[0]

    if len(fits_data):
        xx = fobs * YR
        yy = 1e-15 * np.power(xx, -2.0/3.0)
        ax.plot(xx, yy, 'r-', alpha=0.5, lw=1.0, label="$10^{-15} \cdot f_\\mathrm{yr}^{-2/3}$")

        fits = get_hc_bg_fits_data(fobs, hc_bg)

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

def make_pars_plot(fobs, hc_ss, hc_bg, sspar, bgpar, fits_data):
    fig = holo.plot.plot_pars(fobs, hc_ss, hc_bg, sspar, bgpar)
    # add plaw and fits to hc plot
    ax = fig.axes[3]
    if len(fits_data):
        xx = fobs * YR
        yy = 1e-15 * np.power(xx, -2.0/3.0)
        ax.plot(xx, yy, 'r-', alpha=0.5, lw=1.0, label="$10^{-15} \cdot f_\\mathrm{yr}^{-2/3}$")

        fits = get_hc_bg_fits_data(fobs, hc_bg)

        for ls, idx in zip([":", "--"], [1, -1]):
            med_lamp = fits['fit_med_lamp'][idx]
            med_plaw = fits['fit_med_plaw'][idx]
            yy = (10.0 ** med_lamp) * (xx ** med_plaw)
            label = fits['fit_nbins'][idx]
            label = 'all' if label in [0, None] else label
            ax.plot(xx, yy, color='k', ls=ls, alpha=0.5, lw=2.0, label=str(label) + " bins")

        label = fits['fit_label'].replace(" | ", "\n")
        fig.text(0.93, 0.93, label, fontsize=6, ha='right', va='top')
        # fig.tight_layout()

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
            log.exception("Failed to load gwb fits data!")
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

def run_ss_at_pspace_num(args, space, pnum):
    """Run single source and background strain calculations for the SAM simulation 
    for sample-parameter `pnum` in the `space` parameter-space.

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
    get_pars = bool(args.get_pars)

    # ---- Calculate hc_ss, hc_bg, sspar, and bgpar from SAM

    try:
        log.debug("Selecting `sam` and `hard` instances")
        sam, hard = space(pnum)
        _log_mem_usage(log)

        log.debug(f"Calculating 'edges' and 'number' for this SAM.")
        fobs_orb_edges = fobs_edges / 2.0 
        fobs_orb_cents = fobs_cents/ 2.0
        # edges
        edges, dnum = sam.dynamic_binary_number(hard, fobs_orb=fobs_orb_cents) # should the zero stalled option be part of the parameter space?
        edges[-1] = fobs_orb_edges
        # integrate for number
        number = utils._integrate_grid_differential_number(edges, dnum, freq=False)
        number = number * np.diff(np.log(fobs_edges))  
        _log_mem_usage(log)
        
        if(get_pars):
            log.debug(f"Calculating 'hc_ss', 'hc_bg', 'sspar', and 'bgpar' for shape ({fobs_cents.size}, {args.nreals})")
            hc_ss, hc_bg, sspar, bgpar = ss.ss_gws(edges, number, realize=args.nreals, 
                                               loudest = args.nloudest, params = True)
        else:
            log.debug(f"Calculating 'hc_ss' and 'hc_bg' only for shape ({fobs_cents.size}, {args.nreals})")
            hc_ss, hc_bg = ss.ss_gws(edges, number, realize=args.nreals, 
                                               loudest = args.nloudest, params = False) 
        _log_mem_usage(log)
        log.debug(f"{holo.utils.stats(hc_ss)=}")
        log.debug(f"{holo.utils.stats(hc_bg)=}")
        if(get_pars):
            log.debug(f"{holo.utils.stats(sspar)=}")
            log.debug(f"{holo.utils.stats(bgpar)=}")

        log.debug(f"Saving {pnum} to file")
        if(get_pars):
            data = dict(fobs=fobs_cents, fobs_edges=fobs_edges, 
                    hc_ss = hc_ss, hc_bg = hc_bg, sspar = sspar, bgpar = bgpar)
        else:
            data = dict(fobs=fobs_cents, fobs_edges=fobs_edges, 
                    hc_ss = hc_ss, hc_bg = hc_bg)
        rv = True
    except Exception as err:
        log.exception(f"`run_ss` FAILED on {pnum=}\n")
        log.exception(err)
        rv = False
        data = dict(fail=str(err))

    # ---- Fit hc_bg spectra

    fit_data = {}
    if rv:
        log.info("calculating hc_bg spectra fits")
        try:
            psd = utils.char_strain_to_psd(fobs_cents[:, np.newaxis], hc_bg)
            plaw_nbins, fit_plaw, fit_plaw_med = fit_spectra_plaw(fobs_cents, psd, FITS_NBINS_PLAW)
            turn_nbins, fit_turn, fit_turn_med = fit_spectra_turn(fobs_cents, psd, FITS_NBINS_TURN)

            fit_data = dict(
                fit_plaw_nbins=plaw_nbins, fit_plaw=fit_plaw, fit_plaw_med=fit_plaw_med,
                fit_turn_nbins=turn_nbins, fit_turn=fit_turn, fit_turn_med=fit_turn_med,
            )
        except Exception as err:
            log.exception("Failed to load hc_bg fits data!")
            log.exception(err)

    # ---- Save data to file

    np.savez(sim_fname, **data, **fit_data)
    log.info(f"Saved to {sim_fname}, size {holo.utils.get_file_size(sim_fname)} after {(datetime.now()-beg)}")

    # STILL NEED TO EDIT BELOW HERE
    # ---- Plot GWB spectra

    if rv and args.plot:
        log.info("generating spectra plots")
        try:
            plot_fname = args.output_plots.joinpath(sim_fname.name)
            hc_fname = str(plot_fname.with_suffix('')) + "_strain.png"
            fig = make_ss_plot(fobs_cents, hc_ss, hc_bg, fit_data)
            fig.savefig(hc_fname, dpi=100)
            log.info(f"Saved to {hc_fname}, size {holo.utils.get_file_size(hc_fname)}")
            plt.close('all')
        except Exception as err:
            log.exception("Failed to make strain plot!")
            log.exception(err)
        if(get_pars):
            try:
                pars_fname = str(plot_fname.with_suffix('')) + "_pars.png"
                fig = make_pars_plot(fobs_cents, hc_ss, hc_bg, sspar, bgpar, fit_data)
                fig.savefig(pars_fname, dpi=100)
                log.info(f"Saved to {pars_fname}, size {holo.utils.get_file_size(pars_fname)}")
                plt.close('all')
            except Exception as err:
                log.exception("Failed to make pars plot!")
                log.exception(err)


    return rv


def _sim_fname(path, pnum):
    temp = FNAME_SIM_FILE.format(pnum=pnum)
    temp = path.joinpath(temp)
    return temp


def _log_mem_usage(log):
    # results.ru_maxrss is KB on Linux, B on macos
    mem_max = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2)
    process = psutil.Process(os.getpid())
    mem_rss = process.memory_info().rss / 1024**3
    mem_vms = process.memory_info().vms / 1024**3
    log.info(f"Current memory usage: max={mem_max:.2f} GB, RSS={mem_rss:.2f} GB, VMS={mem_vms:.2f} GB")
    return

# def run_ss_at_pspace_num(args, space, pnum, path_output):
#     log = args.log
#     fname = f"lib_ss__p{pnum:06d}.npz"
#     fname = Path(path_output, fname)
#     beg = datetime.now()
#     log.info(f"{pnum=} :: {fname=} beginning at {beg}")
#     if fname.exists():
#         log.warning(f"File {fname} already exists.")

#     def log_mem():
#         # results.ru_maxrss is KB on Linux, B on macos
#         mem_max = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2)
#         process = psutil.Process(os.getpid())
#         mem_rss = process.memory_info().rss / 1024**3
#         mem_vms = process.memory_info().vms / 1024**3
#         log.info(f"Current memory usage: max={mem_max:.2f} GB, RSS={mem_rss:.2f} GB, VMS={mem_vms:.2f} GB")

#     pta_dur = args.pta_dur * YR
#     nfreqs = args.nfreqs
#     hifr = nfreqs/pta_dur
#     pta_cad = 1.0 / (2 * hifr)
#     fobs_cents = holo.utils.nyquist_freqs(pta_dur, pta_cad)
#     fobs_edges = holo.utils.nyquist_freqs_edges(pta_dur, pta_cad)
#     log.info(f"Created {fobs_cents.size} frequency bins")
#     log.info(f"\t[{fobs_cents[0]*YR}, {fobs_cents[-1]*YR}] [1/yr]")
#     log.info(f"\t[{fobs_cents[0]*1e9}, {fobs_cents[-1]*1e9}] [nHz]")
#     log_mem()
#     assert nfreqs == fobs_cents.size
#     get_pars = bool(args.get_pars)

#     try:
#         log.debug("Selecting `sam` and `hard` instances")
#         sam, hard = space(pnum)
#         log_mem()

#         ### HERE IS WHERE THINGS CHANGE FOR SS ###
#         log.debug(f"Calculating SS and BG GWs for shape ({fobs_cents.size}, {args.nreals})")
#         fobs_orb_edges = fobs_edges / 2.0 
#         fobs_orb_cents = fobs_cents/ 2.0
#         # edges
#         edges, dnum = sam.dynamic_binary_number(hard, fobs_orb=fobs_orb_cents) # should the zero stalled option be part of the parameter space?
#         edges[-1] = fobs_orb_edges
#         # integrate for number
#         number = utils._integrate_grid_differential_number(edges, dnum, freq=False)
#         number = number * np.diff(np.log(fobs_edges))  
#         # gws
#         if(get_pars):
#             hc_ss, hc_bg, sspar, bgpar = ss.ss_gws(edges, number, realize=args.nreals, 
#                                                loudest = args.nloudest, params = True) 
#         else:
#             hc_ss, hc_bg = ss.ss_gws(edges, number, realize=args.nreals, 
#                                                loudest = args.nloudest, params = False) 
            
#         log_mem()
#         log.debug(f"{holo.utils.stats(hc_ss)=}")
#         legend = space.param_dict(pnum)
#         log.debug(f"Saving {pnum} to file")
#         if(get_pars):
#             data = dict(fobs=fobs_cents, fobs_edges=fobs_edges, 
#                     hc_ss = hc_ss, hc_bg = hc_bg, sspar = sspar, bgpar = bgpar)
#         else:
#             data = dict(fobs=fobs_cents, fobs_edges=fobs_edges, 
#                     hc_ss = hc_ss, hc_bg = hc_bg)
#         ### EDITED UP TO HERE ###

#         rv = True
#     except Exception as err:
#         log.exception("\n\n")
#         log.exception("="*100)
#         log.exception(f"`run_ss` FAILED on {pnum=}\n")
#         log.exception(err)
#         log.exception("="*100)
#         log.exception("\n\n")
#         rv = False
#         legend = {}
#         data = dict(fail=str(err))

#     if rv:
#         try:
#             fits_data = get_gwb_fits_data(fobs_cents, hc_bg)
#         except Exception as err:
#             log.exception("Failed to load hc_bg fits data!")
#             log.exception(err)
#             fits_data = {}

#     else:
#         fits_data = {}

#     meta_data = dict(
#         pnum=pnum, pdim=space.ndims, nsamples=args.nsamples, librarian_version=__version__,
#         param_names=space.names, params=space._params, samples=space._samples, # prob don't need these for all of them
#     )

#     np.savez(fname, **data, **meta_data, **fits_data, **legend)
#     log.info(f"Saved to {fname}, size {holo.utils.get_file_size(fname)} after {(datetime.now()-beg)}")

#     if rv:
#         try:
#             hname = str(fname.with_suffix('')) + "_strain.png"
#             fig = make_ss_plot(fobs_cents, hc_ss, hc_bg, fits_data)
#             fig.savefig(hname, dpi=300)
#             log.info(f"Saved to {hname}, size {holo.utils.get_file_size(hname)}")
#             plt.close('all')
#         except Exception as err:
#             log.exception("Failed to make strain plots!")
#             log.exception(err)
#         if(get_pars):
#             try:
#                 pname = str(fname.with_suffix('')) + "_pars.png"
#                 fig = make_pars_plot(fobs_cents, hc_ss, hc_bg, sspar, bgpar, fits_data)
#                 fig.savefig(pname, dpi=300)
#                 log.info(f"Saved to {pname}, size {holo.utils.get_file_size(pname)}")
#             except Exception as err:
#                 log.exception("Failed to make pars plots!")
#                 log.exception(err)
            
#     return rv

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
