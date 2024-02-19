"""Parameters and parameter spaces for holodeck libraries.
"""

import abc
import os
from pathlib import Path
import sys

import numpy as np
import scipy as sp
import scipy.stats

import holodeck as holo


class _Param_Space(abc.ABC):
    """Base class for generating holodeck libraries.  Defines the parameter space and settings.

    Libraries are generated over some parameter space defined by which parameters are being varied.

    """

    _SAVED_ATTRIBUTES = [
        "sam_shape", "param_names", "_uniform_samples", "param_samples",
    ]

    DEFAULTS = {}

    def __init__(self, parameters, log=None, nsamples=None, sam_shape=None, seed=None, random_state=None):
        """Construct a parameter-space instance.

        Arguments
        ---------
        parameters : list of `_Param_Dist` subclasses
        log : ``logging.Logger`` instance  or  ``None``
        nsamples : int or ``None``
            Number of samples to draw from the parameter-space.
        sam_shape : int or (3,) of int or ``None``
            Shape of the SAM grid (see :class:`~holodeck.sams.sam.Semi_Analytic_Model`).
        seed : int or None,
            Seed for the ``numpy`` random number generator.  Better to use ``random_state``.
        random_state : tuple,
            A tuple describing the state of the ``numpy`` random number generator.
        param_kwargs : dict
            Key-value pairs specifying the parameters for this model.  Each key must be the name of
            the parameter, and each value must be a
            :class:`~holodeck.librarian.libraries._Param_Dist`
            subclass instance with the desired distribution.

        Returns
        -------
        None

        """
        if log is None:
            log = holo.log
        log.debug(f"seed = {seed}")
        if random_state is None:
            np.random.seed(seed)
            random_state = np.random.get_state()
        else:
            np.random.set_state(random_state)

        try:
            npars = len(parameters)
            assert npars > 0
        except (TypeError, AssertionError) as err:
            msg = "`parameters` must be a list of `_Param_Dist` subclasses!"
            log.exception(msg)
            log.exception(err)
            raise err

        param_names = [param.name for param in parameters]

        param_names = []
        for param in parameters:
            name = param.name
            param_names.append(name)

            if not isinstance(param, _Param_Dist):
                err = f"{name}: {param} is not a `_Param_Dist` object!"
                log.exception(err)
                raise ValueError(err)

        if (nsamples is None) or (npars == 0):
            log.warning(f"{self}: {nsamples=} {npars=} - cannot generate parameter samples.")
            uniform_samples = None
            param_samples = None
        else:
            # if strength = 2, then n must be equal to p**2, with p prime, and d <= p + 1
            lhc = sp.stats.qmc.LatinHypercube(d=npars, strength=1, seed=seed)
            # (S, D) - samples, dimensions
            uniform_samples = lhc.random(n=nsamples)
            param_samples = np.zeros_like(uniform_samples)

            for ii, param in enumerate(parameters):
                param_samples[:, ii] = param(uniform_samples[:, ii])

        self._log = log
        self._npars = npars
        self._seed = seed
        self._random_state = random_state
        self.sam_shape = sam_shape
        self.param_names = param_names
        self.param_samples = param_samples
        self._parameters = parameters
        self._uniform_samples = uniform_samples
        return

    @classmethod
    def model_for_params(cls, params, sam_shape=None):
        """Construct a model (SAM and hardening instances) from the given parameters.

        Arguments
        ---------
        params : dict
            Key-value pairs for sam/hardening parameters.  Each item much match expected parameters
            that are set in the `defaults` dictionary.
        sam_shape : None  or  int  or  (3,) int

        Returns
        -------
        sam : :class:`holodeck.sam.Semi_Analytic_Model` instance
        hard : :class:`holodeck.hardening._Hardening` instance

        """

        # ---- Update default parameters with input parameters

        settings = cls.DEFAULTS.copy()
        for name, value in params.items():
            settings[name] = value

        # ---- Construct SAM and hardening model

        sam = cls._init_sam(sam_shape, settings)
        hard = cls._init_hard(sam, settings)

        return sam, hard

    @classmethod
    @abc.abstractmethod
    def _init_sam(cls, sam_shape, params):
        raise

    @classmethod
    @abc.abstractmethod
    def _init_hard(cls, sam, params):
        raise

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
        vers = holo.librarian.__version__

        # make sure `path_output` is a directory, and that it exists
        path_output = Path(path_output)
        if not path_output.exists() or not path_output.is_dir():
            err = f"save path {path_output} does not exist, or is not a directory!"
            log.exception(err)
            raise ValueError(err)

        fname = f"{my_name}{holo.librarian.PSPACE_FILE_SUFFIX}"
        fname = path_output.joinpath(fname)
        log.debug(f"{my_name=} {vers=} {fname=}")

        data = {}
        for key in self._SAVED_ATTRIBUTES:
            data[key] = getattr(self, key)

        np.savez(
            fname, class_name=my_name, librarian_version=vers,
            **data,
        )

        log.info(f"Saved to {fname} size {holo.utils.get_file_size(fname)}")
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
        data = np.load(fname, allow_pickle=True)

        # get the name of the parameter-space class from the file, and try to find this class in the
        # `holodeck.param_spaces` module
        class_name = data['class_name'][()]
        log.debug(f"loaded: {class_name=}, vers={data['librarian_version']}")
        # pspace_class = getattr(holo.param_spaces, class_name, None)
        pspace_class = holo.librarian.param_spaces.get(class_name, None)
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

    def param_dict(self, samp_num):
        rv = {nn: pp for nn, pp in zip(self.param_names, self.param_samples[samp_num])}
        return rv

    @property
    def extrema(self):
        extr = [dd.extrema for dd in self._parameters]
        return np.asarray(extr)

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def lib_shape(self):
        return self.param_samples.shape

    @property
    def nsamples(self):
        return self.lib_shape[0]

    @property
    def npars(self):
        return self._npars

    def model_for_sample_number(self, samp_num, sam_shape=None):
        if sam_shape is None:
            sam_shape = self.sam_shape
        params = self.param_dict(samp_num)
        self._log.debug(f"params {samp_num} :: {params}")
        return self.model_for_params(params, sam_shape)

    def normalized_params(self, vals):
        """Convert input values (uniform/linear) into parameters from the stored distributions.

        For example, if this parameter space has 2 dimensions, where the distributions are:

        0. 'value_a' is a uniform parameter from [-1.0, 1.0], and
        1. 'value_b' normal with mean 10.0 and stdev 1.0

        Then input values of ``[0.75, 0.5]`` are mapped to parameters ``[0.5, 10.0]``, which will be
        returned as ``{value_a: 0.5, value_b: 10.0}``.

        Arguments
        ---------
        vals : (P,) iterable of float,
            A list/iterable of `P` float values, matching the number of parameters (i.e. dimensions)
            in this parameter space.  Each value is passed to the corresponding distribution for
            that parameter.

        Returns
        -------
        params : dict,
            The resulting parameters in the form of key-value pairs where the keys are the parameter
            names, and the values are drawn from the correspinding distributions.

        """
        if np.ndim(vals) == 0:
            vals = self.npars * [vals]
        assert len(vals) == self.npars

        params = {}
        for ii, pname in enumerate(self.param_names):
            vv = vals[ii]                     # desired fractional parameter value [0.0, 1.0]
            ss = self._parameters[ii](vv)     # convert to actual parameter values
            params[pname] = ss                # store to dictionary

        return params

    def default_params(self):
        """Return a parameter dictionary with default values for each parameter.

        Returns
        -------
        params : dict,
            Key-value pairs where each key is the parameter name, and the value is the default value
            returned from the :class:`_Param_Dist` subclass.

        """
        params = {param.name: param.default for param in self._parameters}
        return params


class _Param_Dist(abc.ABC):
    """Parameter Distribution base-class for use in Latin HyperCube sampling.

    These classes are passed uniform random variables between [0.0, 1.0], and return parameters
    from the desired distribution.

    Subclasses are required to implement the ``_dist_func()`` function which accepts a float value
    from [0.0, 1.0] and returns the appropriate corresponding parameter, drawn from the desired
    distribution.  In practice, ``_dist_func()`` is usually the inverse cumulative-distribution for
    the desired distribution function.

    """

    def __init__(self, name, default=None, clip=None):
        if clip is not None:
            assert len(clip) == 2
        self._clip = clip
        self._name = name
        self._default = default
        return

    def __call__(self, xx):
        rv = self._dist_func(xx)
        if self._clip is not None:
            rv = np.clip(rv, *self._clip)
        return rv

    @abc.abstractmethod
    def _dist_func(self, *args, **kwargs):
        pass

    @property
    def extrema(self):
        return self(np.asarray([0.0, 1.0]))

    @property
    def name(self):
        return self._name

    @property
    def default(self):
        """Return the default parameter value.

        If a fixed value was set, it will be returned.  Otherwise the parameter value for a random
        uniform input value of 0.5 will be returned.

        """
        if self._default is not None:
            return self._default

        return self(0.5)


class PD_Uniform(_Param_Dist):

    def __init__(self, name, lo, hi, **kwargs):
        super().__init__(name, **kwargs)
        self._lo = lo
        self._hi = hi
        return

    def _dist_func(self, xx):
        yy = self._lo + (self._hi - self._lo) * xx
        return yy


class PD_Uniform_Log(_Param_Dist):

    def __init__(self, name, lo, hi, **kwargs):
        super().__init__(name, **kwargs)
        assert lo > 0.0 and hi > 0.0
        self._lo_log10 = np.log10(lo)
        self._hi_log10 = np.log10(hi)
        return

    def _dist_func(self, xx):
        yy = np.power(10.0, self._lo_log10 + (self._hi_log10 - self._lo_log10) * xx)
        return yy


class PD_Normal(_Param_Dist):
    """

    NOTE: use `clip` parameter to avoid extreme values.

    """

    def __init__(self, name, mean, stdev, clip=None, **kwargs):
        """

        Arguments
        ---------

        """
        assert stdev > 0.0
        super().__init__(name, clip=clip, **kwargs)
        self._mean = mean
        self._stdev = stdev
        self._frozen_dist = sp.stats.norm(loc=mean, scale=stdev)
        return

    def _dist_func(self, xx):
        yy = self._frozen_dist.ppf(xx)
        return yy


class PD_Lin_Log(_Param_Dist):

    def __init__(self, name, lo, hi, crit, lofrac, **kwargs):
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
        super().__init__(name, **kwargs)
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

    def __init__(self, name, lo, hi, crit, lofrac, **kwargs):
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
        super().__init__(name, **kwargs)
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


class PD_Piecewise_Uniform_Mass(_Param_Dist):

    def __init__(self, name, edges, weights, **kwargs):
        super().__init__(name, **kwargs)
        edges = np.asarray(edges)
        self._edges = edges
        weights = np.asarray(weights)
        self._weights = weights / weights.sum()
        assert edges.size == weights.size + 1
        assert np.ndim(edges) == 1
        assert np.ndim(weights) == 1
        assert np.all(np.diff(edges) > 0.0)
        assert np.all(weights > 0.0)
        return

    def _dist_func(self, xx):
        yy = np.zeros_like(xx)
        xlo = 0.0
        for ii, ww in enumerate(self._weights):
            ylo = self._edges[ii]
            yhi = self._edges[ii+1]

            xhi = xlo + ww
            sel = (xlo < xx) & (xx <= xhi)
            yy[sel] = ylo + (xx[sel] - xlo) * (yhi - ylo) / (xhi - xlo)

            xlo = xhi

        return yy


class PD_Piecewise_Uniform_Density(PD_Piecewise_Uniform_Mass):

    def __init__(self, name, edges, densities, **kwargs):
        dx = np.diff(edges)
        weights = dx * np.asarray(densities)
        super().__init__(name, edges, weights)
        return


def load_pspace_from_path(log, path, space_class=None):
    """Load a `_Param_Space` instance from the saved file in the given directory.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to directory containing save file.
        A single file matching `*.pspace.npz` is required in that directory.
        NOTE: the specific glob pattern is specified by `holodeck.librarian.PSPACE_FILE_SUFFIX` e.g. '.pspace.npz'
    space_class : `_Param_Space` subclass
        Class with which to call the `from_save()` method to load a new `_Param_Space` instance.

    Returns
    -------
    log : `logging.Logger`
    space : `_Param_Space` subclass instance
        An instance of the `space_class` class.
    space_fname : pathlib.Path
        File that `space` was loaded from.

    """
    path = Path(path)
    if not path.exists():
        raise RuntimeError(f"path {path} does not exist!")

    # If this is a directory, look for a pspace save file
    if path.is_dir():
        pattern = "*" + holo.librarian.PSPACE_FILE_SUFFIX
        space_fname = list(path.glob(pattern))
        if len(space_fname) != 1:
            raise FileNotFoundError(f"found {len(space_fname)} matches to {pattern} in output {path}!")

        space_fname = space_fname[0]

    # if this is a file, assume that it's already the pspace save file
    elif path.is_file():
        space_fname = path

    else:
        raise

    # Based on the `space_fname`, try to find a matching PS (parameter-space) in `holodeck.param_spaces`
    if space_class is None:
        space_class = _get_space_class_from_space_fname(space_fname)

    space = space_class.from_save(space_fname, log)
    return space, space_fname


def _get_space_class_from_space_fname(space_fname):
    # Based on the `space_fname`, try to find a matching PS (parameter-space) in `holodeck.param_spaces`
    space_name = space_fname.name.split(".")[0]
    space_class = holo.librarian.param_spaces_dict[space_name]
    return space_class


def _get_sim_fname(path, pnum):
    temp = holo.librarian.FNAME_SIM_FILE.format(pnum=pnum)
    temp = path.joinpath(temp)
    return temp


def get_sam_lib_fname(path, gwb_only):
    fname = 'sam_lib'
    if gwb_only:
        fname += "_gwb-only"
    lib_path = path.joinpath(fname).with_suffix(".hdf5")
    return lib_path


def get_fits_path(library_path):
    """Get the name of the spectral fits file, given a library file path.
    """
    fits_path = library_path.with_stem(library_path.stem + "_fits")
    fits_path = fits_path.with_suffix('.npz')
    return fits_path


def log_mem_usage(log):

    try:
        import resource
        # results.ru_maxrss is KB on Linux, B on macos
        mem_max = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macos systems
        if sys.platform.lower().startswith('darwin'):
            mem_max = (mem_max / 1024 ** 3)
        # linux systems
        else:
            mem_max = (mem_max / 1024 ** 2)
    except Exception:
        mem_max = np.nan

    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem_rss = process.memory_info().rss / 1024**3
        mem_vms = process.memory_info().vms / 1024**3
        msg = f"Current memory usage: max={mem_max:.2f} GB, RSS={mem_rss:.2f} GB, VMS={mem_vms:.2f} GB"
    except Exception:
        msg = "Unable to load either `resource` or `psutil`.  Try installing at least `psutil`."

    if log is None:
        print(msg, flush=True)
    else:
        log.info(msg)

    return


