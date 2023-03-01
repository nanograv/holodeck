"""
"""

import abc

import h5py
import numpy as np
import scipy as sp
import scipy.optimize  # noqa
import tqdm

from scipy.stats import qmc
import pyDOE

import holodeck as holo
from holodeck.constants import YR


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

            if not isinstance(val, holo.librarian._Param_Dist):
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
        uniform_samples = LHS.random(n=nsamples)
        samples = np.zeros_like(uniform_samples)

        for ii, dist in enumerate(dists):
            samples[:, ii] = dist(uniform_samples[:, ii])

        self._log = log
        self.names = names
        self.nsamples = nsamples
        self.ndims = ndims
        self.sam_shape = sam_shape
        self._seed = seed
        self._random_state = random_state
        self._uniform_samples = uniform_samples
        self._samples = samples

        return

    def params(self, num):
        return self._samples[num]

    def param_dict(self, num):
        rv = {nn: pp for nn, pp in zip(self.names, self.params(num))}
        return rv

    def __call__(self, num):
        return self.model_for_number(num)

    @property
    def shape(self):
        return self._samples.shape

    @abc.abstractmethod
    def model_for_number(self, num):
        raise


class _Param_Dist(abc.ABC):

    @abc.abstractmethod
    def __init__(self):
        return

    def __call__(self, xx):
        return self._dist_func(xx)


class PD_Uniform(_Param_Dist):

    def __init__(self, lo, hi):
        self._lo = lo
        self._hi = hi
        self._dist_func = lambda xx: self._lo + (self._hi - self._lo) * xx
        return


class PD_Uniform_Log(_Param_Dist):

    def __init__(self, lo, hi):
        self._lo = np.log10(lo)
        self._hi = np.log10(hi)
        self._dist_func = lambda xx: np.power(10.0, self._lo + (self._hi - self._lo) * xx)
        return


class PD_Normal(_Param_Dist):

    def __init__(self, mean, stdev):
        self._mean = mean
        self._stdev = stdev
        self._dist = sp.stats.norm(loc=mean, scale=stdev)
        self._dist_func = lambda xx: self._dist.ppf(xx)
        return


def sam_lib_combine(path_output, log, debug=False):
    log.info(f"Path output = {path_output}")

    regex = "lib_sams__p*.npz"
    files = sorted(path_output.glob(regex))
    num_files = len(files)
    log.info(f"\texists={path_output.exists()}, found {num_files} files")

    all_exist = True
    log.info("Checking files")
    for ii in tqdm.tqdm(range(num_files)):
        temp = path_output.joinpath(regex.replace('*', f"{ii:06d}"))
        exists = temp.exists()
        if not exists:
            all_exist = False
            break

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
    nreals = temp_gwb.shape[1]
    param_vals = data['params']
    param_names = data['names']
    lhs_grid = data['lhs_grid']
    try:
        pdim = data['pdim']
    except KeyError:
        pdim = 6

    try:
        nsamples = data['nsamples']
        if num_files != nsamples:
            raise ValueError(f"nsamples={nsamples} but num_files={num_files} !!")
    except KeyError:
        pass

    assert np.ndim(temp_gwb) == 2
    if temp_gwb.shape[0] != nfreqs:
        raise ValueError(f"temp_gwb.shape={temp_gwb.shape} but nfreqs={nfreqs}!!")
    if temp_gwb.shape[1] != nreals:
        raise ValueError(f"temp_gwb.shape={temp_gwb.shape} but nreals={nreals}!!")

    # ---- Store results from all files

    gwb_shape = [num_files, nfreqs, nreals]
    shape_names = list(param_names[:]) + ['freqs', 'reals']
    gwb = np.zeros(gwb_shape)
    sample_params = np.zeros((num_files, pdim))
    grid_idx = np.zeros((num_files, pdim), dtype=int)
    if lhs_grid.shape == ():   # not a gridded lhs parameter space
        if lhs_grid[()] == -1:   # using scipy LHS direct sampling
            log.info(f"Parameter Space Type is direct LHS")
            pspacetype = 'ungriddedlhs'
        else:
            err = f"Uknown parameter space type: {lhs_grid[()]}"
            log.exception(err)
            raise ValueError(err)
    else:
        log.info(f"Parameter Space Type is Gridded LHS")
        pspacetype = 'griddedlhs'

    log.info(f"Collecting data from {len(files)} files")
    for ii, file in enumerate(tqdm.tqdm(files)):
        temp = np.load(file, allow_pickle=True)
        if ('fail' in temp) or ('gwb' not in temp):
            err = f"THIS IS A FAILED DATASET ({ii}, {file}).  LIBRARIAN HASNT BEEN UPDATED TO HANDLE THIS CASE!"
            log.exception(err)
            raise RuntimeError(err)

        assert ii == temp['pnum']
        assert np.allclose(fobs, temp['fobs'])
        assert np.allclose(fobs_edges, temp['fobs_edges'])
        pars = [temp[nn][()] for nn in param_names]
        for jj, (pp, nn) in enumerate(zip(temp['params'], temp['names'])):
            assert np.allclose(pp, param_vals[jj])
            assert nn == param_names[jj]

        assert np.all(lhs_grid == temp['lhs_grid'])

        tt = temp['gwb'][:]
        assert np.shape(tt) == (nfreqs, nreals)
        gwb[ii] = tt
        sample_params[ii, :] = pars
        grid_idx[ii, :] = temp['lhs_grid_idx']
        if debug:
            break

    out_filename = path_output.joinpath('sam_lib.hdf5')
    log.info(f"Writing collected data to file {out_filename}")
    with h5py.File(out_filename, 'w') as h5:
        h5.create_dataset('fobs', data=fobs)
        h5.create_dataset('fobs_edges', data=fobs_edges)
        h5.create_dataset('gwb', data=gwb)
        h5.create_dataset('sample_params', data=sample_params)
        h5.attrs['param_names'] = np.array(param_names).astype('S')
        h5.attrs['shape_names'] = np.array(shape_names).astype('S')
        h5.attrs['parameter_space_type'] = pspacetype
        if pspacetype == 'griddedlhs':
            h5.create_dataset('lhs_grid', data=lhs_grid)
            h5.create_dataset('lhs_grid_indices', data=grid_idx)
            group = h5.create_group('parameters')
            group.attrs['ordered_parameters'] = param_names
            for pname, pvals in zip(param_names, param_vals):
                group.create_dataset(pname, data=pvals)

    log.warning(f"Saved to {out_filename}, size: {holo.utils.get_file_size(out_filename)}")
    return


def fit_powerlaw(freqs, hc, nbins, init=[-15.0, -2.0/3.0]):
    nbins = None if (nbins == 0) else nbins
    cut = slice(None, nbins)
    xx = freqs[cut] * YR
    yy = hc[cut]

    def powerlaw_fit(freqs, log10Ayr, gamma):
        zz = log10Ayr + gamma * freqs
        return zz

    popt, pcov = sp.optimize.curve_fit(powerlaw_fit, np.log10(xx), np.log10(yy), p0=init, maxfev=10000)

    amp = 10.0 ** popt[0]
    gamma = popt[1]
    return xx, amp, gamma


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

        xx, amp, plaw = fit_powerlaw(freqs, np.median(gwb, axis=-1), num)
        fit_med_lamp[ii] = np.log10(amp)
        fit_med_plaw[ii] = plaw
        for rr in range(nreals):
            xx, amp, plaw = fit_powerlaw(freqs, gwb[:, rr], num)
            fit_lamp[rr, ii] = np.log10(amp)
            fit_plaw[rr, ii] = plaw

    return nbins, fit_lamp, fit_plaw, fit_med_lamp, fit_med_plaw


def get_gwb_fits_data(fobs_cents, gwb):
    # these values must match label construction!
    nbins = [5, 10, 15, 0]

    nbins, lamp, plaw, med_lamp, med_plaw = holo.librarian.fit_spectra(fobs_cents, gwb, nbins=nbins)

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

        fits = holo.librarian.get_gwb_fits_data(fobs, gwb)

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


