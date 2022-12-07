"""
"""

import abc

import h5py
import numpy as np
import tqdm

from scipy.stats import qmc
import pyDOE

import holodeck as holo


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

        '''
        self.gsmf_phi0 = np.linspace(*gsmf_phi0)
        self.times = np.logspace(*np.log10(times[:2]), times[2])
        self.gpf_qgamma = np.linspace(*gpf_qgamma)
        self.hard_gamma_inner = np.linspace(*hard_gamma_inner)
        self.mmb_amp = np.linspace(*mmb_amp)
        self.mmb_plaw = np.linspace(*mmb_plaw)
        params = [
            self.gsmf_phi0,
            self.times,   # [Gyr]
            self.gpf_qgamma,
            self.hard_gamma_inner,
            self.mmb_amp,
            self.mmb_plaw
        ]
        self.names = [
            'gsmf_phi0',
            'times',
            'gpf_qgamma',
            'hard_gamma_inner',
            'mmb_amp',
            'mmb_plaw'
        ]
        '''

        self.paramdimen = len(params)
        self.params = params
        self.names = names
        maxints = [tmparr.size for tmparr in params]

        # do scipy LHS
        if False:
            LHS = qmc.LatinHypercube(d=self.paramdimen, centered=False, strength=1)
            # if strength = 2, then n must be equal to p**2, with p prime, and d <= p + 1
            sampleindxs = LHS.random(n=nsamples)

        # do pyDOE LHS
        else:
            sampleindxs = pyDOE.lhs(n=self.paramdimen, samples=nsamples, criterion='m')

        for i in range(self.paramdimen):
            sampleindxs[:, i] = np.floor(maxints[i] * sampleindxs[:, i])

        sampleindxs = sampleindxs.astype(int)
        log.debug(f"d={len(params)} samplelims={maxints} {nsamples=}")
        self.sampleindxs = sampleindxs

        self.param_grid = np.meshgrid(*params, indexing='ij')
        self.shape = self.param_grid[0].shape
        self.size = np.product(self.shape)
        if self.size < nsamples:
            err = (
                f"There are only {self.size} gridpoints in parameter space but you are requesting "
                f"{nsamples} samples of them. They will be over-sampled!"
            )
            log.warning(err)

        self.param_grid = np.moveaxis(self.param_grid, 0, -1)

        pass

    def number_to_index(self, num):
        idx = np.unravel_index(num, self.shape)
        return idx

    def lhsnumber_to_index(self, lhsnum):
        idx = tuple(self.sampleindxs[lhsnum])
        return idx

    def index_to_number(self, idx):
        num = np.ravel_multi_index(idx, self.shape)
        return num

    def param_dict_for_number(self, num):
        idx = self.number_to_index(num)
        pars = self.param_grid[idx]
        rv = {nn: pp for nn, pp in zip(self.names, pars)}
        return rv

    def param_dict_for_lhsnumber(self, lhsnum):
        idx = self.lhsnumber_to_index(lhsnum)
        pars = self.param_grid[idx]
        rv = {nn: pp for nn, pp in zip(self.names, pars)}
        return rv

    def params_for_number(self, num):
        idx = self.number_to_index(num)
        pars = self.param_grid[idx]
        return pars

    def params_for_lhsnumber(self, lhsnum):
        idx = self.lhsnumber_to_index(lhsnum)
        pars = self.param_grid[idx]
        return pars

    # @abc.abstractmethod
    def sam_for_number(self, num):
        raise
        return

    @abc.abstractmethod
    def sam_for_lhsnumber(self, lhsnum):
        return

class _LHS_Parameter_Space(abc.ABC):

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
                msg = f"Wanted 2 arguments in {par}, but got {len(vv)}: {vv}. I will assume you are using the NON-LHS initialization scheme. Bad scientist!  For LHS, give min and max values, not grid size."
                log.warning(msg)
                vv = vv[0:2]

            names.append(par)
            param_ranges.append(vv)

        '''
        self.gsmf_phi0 = np.linspace(*gsmf_phi0)
        self.times = np.logspace(*np.log10(times[:2]), times[2])
        self.gpf_qgamma = np.linspace(*gpf_qgamma)
        self.hard_gamma_inner = np.linspace(*hard_gamma_inner)
        self.mmb_amp = np.linspace(*mmb_amp)
        self.mmb_plaw = np.linspace(*mmb_plaw)
        params = [
            self.gsmf_phi0,
            self.times,   # [Gyr]
            self.gpf_qgamma,
            self.hard_gamma_inner,
            self.mmb_amp,
            self.mmb_plaw
        ]
        self.names = [
            'gsmf_phi0',
            'times',
            'gpf_qgamma',
            'hard_gamma_inner',
            'mmb_amp',
            'mmb_plaw'
        ]
        '''

        self.paramdimen = len(param_ranges)
        self.param_ranges = np.array(param_ranges)
        self.names = names
        self.params = np.zeros((self.nsamples,self.paramdimen))
        # Below are done out of laziness and backwards compatibility but should be deprecated
        self.sampleindxs = 0
        
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
        elif self.lhs_sampler == 'pydoe': # do pyDOE LHS
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

    # @abc.abstractmethod
    def sam_for_number(self, num):
        raise
        return

    @abc.abstractmethod
    def sam_for_lhsnumber(self, lhsnum):
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

    log.info(f"Collecting data from {len(files)} files")
    for ii, file in enumerate(tqdm.tqdm(files)):
        temp = np.load(file, allow_pickle=True)
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
        h5.create_dataset('lhs_grid', data=lhs_grid)
        h5.create_dataset('lhs_grid_indices', data=grid_idx)
        h5.attrs['param_names'] = np.array(param_names).astype('S')
        h5.attrs['shape_names'] = np.array(shape_names).astype('S')
        group = h5.create_group('parameters')
        for pname, pvals in zip(param_names, param_vals):
            group.create_dataset(pname, data=pvals)

    log.warning(f"Saved to {out_filename}, size: {holo.utils.get_file_size(out_filename)}")
    return

