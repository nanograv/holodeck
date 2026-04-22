"""Parameters and parameter spaces for holodeck libraries."""

import abc
import os
from pathlib import Path
import sys

import numpy as np
from numpy.linalg import LinAlgError, cholesky
import scipy as sp
import scipy.stats

import holodeck as holo
from holodeck import utils, cosmo, log
from holodeck.constants import YR
from holodeck.librarian import (
    DEF_NUM_FBINS,
    DEF_NUM_LOUDEST,
    DEF_NUM_REALS,
    DEF_PTA_DUR,
    FNAME_LIBRARY_COMBINED_FILE,
    FNAME_DOMAIN_COMBINED_FILE,
)

PARAM_NAMES__ERROR = [
    # ["gsmf_phi0", "Use `gsmf_phi0_log10` instead!"],
]

PARAM_NAMES_REPLACE = {
    "gsmf_phi0": ["gsmf_phi0_log10", None],
}

# __all__ = [
#     "_Param_space"
# ]


class _Param_Space(abc.ABC):
    """Base class for generating holodeck libraries.  Defines the parameter space and settings.

    Libraries are generated over some parameter space defined by which parameters are being varied.

    """

    __version__ = "0.0"

    _SAVED_ATTRIBUTES = ["sam_shape", "param_names", "_uniform_samples", "param_samples", "_nsamples", "_nparameters"]

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
            :class:`~holodeck.librarian.lib_tools._Param_Dist`
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

        self._parameters = parameters
        self._nparameters = 0
        self.param_names = []
        self._param_dist_names = []
        self._nsamples = nsamples
        self.sam_shape = sam_shape
        self._seed = seed
        self._random_state = random_state

        for pdist in self._parameters:
            pdist_name = pdist.name
            if not isinstance(pdist, _Param_Dist):
                err = f"Parameter distribution '{pdist_name}: {pdist}' is not a `_Param_Dist` subclass!"
                log.exception(err)
                raise ValueError(err)
            for pname, msg in PARAM_NAMES__ERROR:
                if pname != pdist_name:
                    continue
                err = f"Found '{pdist_name}' in parameters: {msg}"
                log.exception(err)
                raise ValueError(err)
            for pname, replace in PARAM_NAMES_REPLACE.items():
                if pname != pdist_name:
                    continue
                new_name, new_func = replace
                msg = f"Found '{pdist_name}' in parameters, should be '{new_name}'!"
                log.error(msg)
                # When there is no change in the parameter VALUES, we can just change the name
                if new_func is None:
                    pdist.name = new_name
                    msg = f"Replacing '{pdist_name}' ==> '{new_name}'!"
                    log.error(msg)
                else:
                    err = f"CANNOT replace '{pdist_name}' ==> '{new_name}'!"
                    log.exception(err)
                    raise ValueError(err)

            if isinstance(pdist_name, (list, tuple)):
                self.param_names.extend(pdist_name)
                self._nparameters += len(pdist_name)
            else:
                self.param_names.append(pdist_name)
                self._nparameters += 1
            self._param_dist_names.append(pdist_name)

        if (self._nsamples is None) or (self._nparameters == 0):
            log.info(f"{self}: {self._nsamples=} {self._nparameters=} - cannot generate parameter samples.")
            self._uniform_samples = None
            self.param_samples = None
        else:
            # if strength = 2, then n must be equal to p**2, with p prime, and d <= p + 1
            lhc = sp.stats.qmc.LatinHypercube(d=self._nparameters, strength=1, seed=self._seed)
            # (S, D) - samples, dimensions
            self._uniform_samples = lhc.random(n=self._nsamples)
            self.param_samples = self._transform_samples()

        self._log = log
        return

    def _transform_samples(self):
        """
        Transforms the uniform samples self._uniform_samples into distributed parameter samples.

        This method is updated to correctly pass the required number of uniform dimensions
        to each _Param_Dist object, whether it is univariate (1D) or multivariate (ND).
        """

        # Initialize the array to hold the final, transformed parameter samples
        param_samples = np.zeros((self._nsamples, self._nparameters))

        # Tracks the current column index in the uniform samples array
        current_idx = 0

        for pdist in self._parameters:
            pdist_name = pdist.name

            # Determine the number of dimensions/columns this distribution requires
            if isinstance(pdist_name, (list, tuple)):
                n_dims = len(pdist_name)
            else:
                n_dims = 1

            # Slice the correct block of uniform samples for this distribution
            # The shape will be (nsamples, n_dims)
            uniform_block = self._uniform_samples[:, current_idx : current_idx + n_dims]

            # Apply the distribution's transformation function
            # The __call__ method handles the conversion from uniform to the desired distribution (e.g., Normal)
            transformed_block = pdist(uniform_block)

            # Ensure the output is 2D and has the correct number of columns before insertion
            if transformed_block.ndim == 1:
                # Reshape (nsamples,) to (nsamples, 1) for univariate case
                transformed_block = transformed_block.reshape(-1, 1)

            # Insert the transformed values into the final array
            param_samples[:, current_idx : current_idx + n_dims] = transformed_block

            # Advance the index tracker
            current_idx += n_dims

        # The final samples array shape is (nsamples, n_total_parameters)
        return param_samples

    def model_for_params(self, params, sam_shape=None):
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
        log = self._log

        if sam_shape is None:
            sam_shape = self.sam_shape

        # ---- Update default parameters with input parameters

        settings = self.DEFAULTS.copy()
        for name, value in params.items():
            for pname, replace in PARAM_NAMES_REPLACE.items():
                if pname != name:
                    continue
                new_name, new_func = replace
                msg = f"Found '{name}' in parameters, should be '{new_name}'!"
                self._log.error(msg)
                name = new_name
                if new_func is None:
                    new_value = value
                else:
                    new_value = new_func(value)

                msg = f"Replacing '{name}' ==> '{new_name}' ({value} ==> {new_value})!"
                self._log.error(msg)
                value = new_value

            # Check parameter names to make sure they're not on the error list
            for pname, msg in PARAM_NAMES__ERROR:
                if pname != name:
                    continue
                err = f"Found '{name}' in parameters: {msg}"
                self._log.exception(err)
                raise ValueError(err)

            # Nowhere is it required that all parameters have values stored in the default settings (`DEFAULTS`).
            # For that reason, we don't raise an error if a passed parameter is not already in the defaults.
            # But a good parameter space class should probably always have a default value set.
            # An error may be raised in the future.
            if name not in settings:
                log.warning(
                    f"`params` has key '{name}' which is not already in the default settings!  "
                    f"Ensure '{name}' is consistent with this parameter space ({self})!"
                )

            settings[name] = value

        # ---- Construct SAM and hardening model

        sam = self._init_sam(sam_shape, settings)
        hard = self._init_hard(sam, settings)

        return sam, hard

    # @classmethod
    @abc.abstractmethod
    def _init_sam(self, sam_shape, params):
        """Initialize a :class:`holodeck.sams.sam.Semi_Analytic_Model` instance with given params.

        Arguments
        ---------
        sam_shape : None  or  int  or  (3,) tuple of int
            Shape of the SAM grid (M, Q, Z).  If:
            * `None` : use default values.
            * int : apply this size to each dimension of the grid.
            * (3,) of int : provide size for each dimension.
        params : dict
            Dictionary of parameters needed to initialize SAM model.

        Returns
        -------
        sam : :class:`holodeck.sams.sam.Semi_Analytic_Model` instance,
            Initialized SAM model instance.

        """
        raise

    # @classmethod
    @abc.abstractmethod
    def _init_hard(self, sam, params):
        """Initialize a :class:`holodeck.hardening._Hardening` subclass instance with given params.

        Arguments
        ---------
        sam : :class:`holodeck.sams.sam.Semi_Analytic_Model` instance,
            SAM model instance.
        params : dict
            Dictionary of parameters needed to initialize hardening model.

        Returns
        -------
        hard : :class:`holodeck.hardening._Hardening` subclass instance,
            Initialized hardening model instance.

        """
        raise

    def save(self, path_output):
        """Save the generated samples and parameter-space info from this instance to an output file.

        This data can then be loaded using the ``_Param_Space.from_save`` method.
        NOTE: existing save files with the same name will be overwritten!

        Arguments
        ---------
        path_output : str
            Path in which to save file.  This must be an existing directory.

        Returns
        -------
        fname : ``pathlib.Path`` object
            Output path including filename in which this parameter-space was saved.

        """
        log = self._log
        class_name = self.__class__.__name__
        class_vers = self.__version__
        vers = holo.librarian.__version__

        # make sure `path_output` is a directory, and that it exists
        path_output = Path(path_output)
        if not path_output.exists() or not path_output.is_dir():
            err = f"save path {path_output} does not exist, or is not a directory!"
            log.exception(err)
            raise ValueError(err)

        fname = f"{class_name}{holo.librarian.PSPACE_FILE_SUFFIX}"
        fname = path_output.joinpath(fname)
        log.debug(f"{class_name=} {vers=} {fname=}")

        data = {}
        for key in self._SAVED_ATTRIBUTES:
            data[key] = getattr(self, key)

        np.savez(
            fname,
            class_name=class_name,
            class_vers=class_vers,
            librarian_version=vers,
            **data,
        )

        log.info(f"Saved to {fname} size {utils.get_file_size(fname)}")
        return fname

    @classmethod
    def from_save(cls, fname, log=None):
        """Create a new _Param_Space instance loaded from the given save file.

        Arguments
        ---------
        fname : str
            Filename containing parameter-space save information, generated form `_Param_Space.save`.

        Returns
        -------
        space : `_Param_Space` subclass instance

        """
        if log is None:
            log = holo.log
        log.debug(f"loading parameter space from {fname}")
        data = np.load(fname, allow_pickle=True)

        # get the name of the parameter-space class from the file, and try to find this class in the
        # `holodeck.param_spaces` module
        class_name = data["class_name"][()]
        log.debug(f"loaded: {class_name=}, vers={data['librarian_version']}")
        pspace_class = holo.librarian.param_spaces_dict.get(class_name, None)
        # if it is not found, default to the current class/subclass
        if pspace_class is None:
            log.warning(f"pspace file {fname} has {class_name=}, not found in `holo.param_spaces_dict`!")
            pspace_class = cls

        # construct instance with dummy/temporary values (which will be overwritten)
        if np.all(data["param_samples"] == None):  # noqa: E711
            nsamples = None
            nparameters = None
        else:
            # print(f"{data['param_samples']=}")
            nsamples = data["param_samples"].shape[0]
            nparameters = data["param_samples"].shape[1]
        param_names = data["param_names"]
        space = pspace_class(nsamples=nsamples, log=log)
        if class_name != space.name:
            err = "loaded class name '{class_name}' does not match this class's name '{space.name}'!"
            log.warning(err)
        if not all([pname_load == pname_class for pname_load, pname_class in zip(param_names, space.param_names)]):
            err = f"Mismatch between loaded parameter names ({param_names}) and class parameter names ({space.param_names})!"
            log.exception(err)

            for pn in param_names:
                if pn not in space.param_names:
                    log.exception(f"parameter name `{pn}` in loaded data is not in class param names!")

            for pn in space.param_names:
                if pn not in param_names:
                    log.exception(f"parameter name `{pn}` in class param names is not in loaded parameter names!")

            raise RuntimeError(err)

        # Store loaded parameters into the parameter-space instance
        for key in space._SAVED_ATTRIBUTES:
            # Load from save data
            try:
                val = data[key][()]
            # Handle special elements that may not be saved in older files
            except KeyError:
                if key == "_nsamples":
                    val = nsamples
                elif key == "_nparameters":
                    val = nparameters
                else:
                    raise

            setattr(space, key, val)

        return space

    def param_dict(self, samp_num):
        rv = {nn: pp for nn, pp in zip(self.param_names, self.param_samples[samp_num])}
        return rv

    @property
    def extrema(self):
        """
        The combined extrema for all parameters governed by this parameter space.

        The result is a NumPy array of shape (N_total_params, 2), where N_total_params
        is the sum of all dimensions across all distribution objects.
        """
        # 1. Get the extrema for each distribution object (dd.extrema)
        extr_list = [dd.extrema for dd in self._parameters]

        # 2. Promote any 1D array (from univariate distributions, shape (2,))
        #    to 2D (shape (1, 2)) using np.atleast_2d, and stack them vertically.
        #    Multivariate distributions (shape (N, 2)) are unchanged by atleast_2d.
        extr_stacked = np.vstack([np.atleast_2d(e) for e in extr_list])

        return extr_stacked
        # extr = [dd.extrema for dd in self._parameters] # the old version
        # return np.asarray(extr)

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def lib_shape(self):
        return self.param_samples.shape

    @property
    def nsamples(self):
        return self._nsamples

    @property
    def nparameters(self):
        return self._nparameters

    def model_for_sample_number(self, samp_num, sam_shape=None):
        params = self.param_dict(samp_num)
        self._log.debug(f"params {samp_num} :: {params}")
        return self.model_for_params(params, sam_shape)

    def normalized_params(self, vals):
        """Convert input values (uniform [0.0, 1.0]) into parameters from the stored distributions.

        For example, if this parameter space has 2 dimensions, where the distributions are:

        0. 'value_a' is a uniform parameter from [-1.0, 1.0], and
        1. 'value_b' is normally distributed with mean 10.0 and stdev 1.0

        Then input values of ``[0.75, 0.5]`` are mapped to parameters ``[0.5, 10.0]``, which will be
        returned as ``{value_a: 0.5, value_b: 10.0}``.

        Arguments
        ---------
        vals : (P,) iterable of (float or `None`),
            A list/iterable of `P` values, matching the number of parameters (i.e. dimensions)
            in this parameter space.  If a value is `None`, then the default value for that
            parameter is obtained, otherwise the value is passed to the corresponding parameter
            distribution.

        Returns
        -------
        params : dict,
            The resulting parameters in the form of key-value pairs where the keys are the parameter
            names, and the values are drawn from the correspinding distributions.

        """
        if np.ndim(vals) == 0:
            vals = self.nparameters * [vals]
        assert len(vals) == self.nparameters
        all_finite = np.all([(vv is None) or np.isfinite(vv) for vv in vals])
        assert all_finite, f"Not all `vals` are finite!  {vals}"

        params = {}
        idx = 0
        for param in self._parameters:
            n_dims = param.n_params if param.n_params is not None else 1
            vv = vals[idx : idx + n_dims]
            idx += n_dims

            if all(v is None for v in vv):
                ss = param.default
            else:
                if any(v is None for v in vv):
                    raise ValueError(f"Cannot partially specify values for multivariate parameter {param.name}")
                if n_dims == 1:
                    ss = param(vv[0])  # convert from fractional to actual values
                else:
                    ss = param([vv])[0]

            if n_dims == 1:
                params[param.name] = ss  # store to dictionary
            else:
                for pname, s in zip(param.name, ss):
                    params[pname] = s

        return params

    def default_params(self):
        """Return a parameter dictionary with default values for each parameter.

        Returns
        -------
        params : dict,
            Key-value pairs where each key is the parameter name, and the value is the default value
            returned from the :class:`_Param_Dist` subclass.

        """
        params = {}
        for param in self._parameters:
            if isinstance(param.name, (list, tuple)):
                for pname, val in zip(param.name, param.default):
                    params[pname] = val
            else:
                params[param.name] = param.default
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
            clip = np.asarray(clip)

            # Check for Multivariate chip: (n, 2)
            if clip.ndim == 2 and clip.shape[1] == 2:
                pass
            elif clip.ndim == 1 and clip.shape[0] == 2:
                clip = clip.reshape(1, 2)
            else:
                raise ValueError(
                    "The 'clip' argument must be array-like with shape (n_dims, 2) for "
                    "multivariate distributions, or (2,) for single-parameter distributions: "
                    f"[lower_bound, upper_bound]. got {clip.shape}!"
                )
        self._clip = clip
        self._name = name
        self._default = default
        return

    def __call__(self, xx):
        xx = np.asarray(xx)
        n_dims = self.n_params

        if n_dims is not None:
            if xx.ndim != 2:
                raise ValueError(
                    f"Input samples 'xx' must be 2-dimensional (n_samples, n_dims) for this {n_dims}-parameter distribution! got {xx.ndim=}"
                )
            if xx.shape[1] != n_dims:
                raise ValueError(f"Input samples 'xx'has {xx.shape[1]} dimensions but distribution requires {n_dims} parameters.")
        rv = self._dist_func(xx)
        if self._clip is not None:
            n_dims_check = n_dims if n_dims is not None else self._clip.shape[0]
            if self._clip.shape[0] != n_dims_check:
                raise ValueError(
                    f"Internal error: The 'clip' array has {self._clip.shape[0]} rows, "
                    f"but the distribution requires {n_dims_check} parameters."
                )
            lower_bounds = self._clip[:, 0]
            upper_bounds = self._clip[:, 1]
            rv = np.clip(rv, lower_bounds, upper_bounds)
        return rv

    @abc.abstractmethod
    def _dist_func(self, *args, **kwargs):
        pass

    @property
    def n_params(self):
        """Return the number of parameters for this distribution.

        Returns
        -------
        n_params : int  or  `None`
            Number of parameters for this distribution.  If univariate (1D), then returns `None`.

        """
        if isinstance(self._name, (list, tuple)):
            return len(self._name)
        return None

    @property
    def extrema(self):
        """Return the extrema (min, max) of this parameter distribution."""
        n = self.n_params if self.n_params is not None else 1

        if self._clip is not None:
            result = self._clip
            if n == 1:
                result = result.squeeze()
            return result

        uniform_bounds = np.zeros((2, n))
        uniform_bounds[1, :] = 1.0

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            transformed_bounds = self(uniform_bounds)

        result = transformed_bounds.T

        if n == 1:
            result = result.squeeze()
        return result
        # if self.n_params is None or self.n_params == 1:
        #     return self(np.asarray([0.0, 1.0]))
        # else:

        #     transformed_bounds = self(uniform_bounds)
        #     result = transformed_bounds.T
        #     return result

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

        n = self.n_params
        if n is None:
            # Univariate case: pass a scalar, get a scalar back
            return self(0.5)
        # Multivariate case: __call__ requires a 2D array of shape (n_samples, n_dims).
        # Pass a single sample of all-0.5 values, then squeeze to a 1D array.
        return self(np.full((1, n), 0.5)).squeeze()


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
    """Normal/Gaussian parameter distribution with given mean and standard-deviation.

    NOTE: use `clip` parameter to avoid extreme values.

    """

    def __init__(self, name, mean, stdev, clip=None, **kwargs):
        """Initialize a `PD_Normal` instance.

        Arguments
        ---------
        name : str
            Name of the parameter being distributed.
        mean : float
            Mean / expectation value, center of normal distribution.
        stdev : float
            Standard deviation, width of normal distribution.
        clip : `None`  or  (2,) of float
            Bounds by which to restrict the resulting distribution.

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


class PD_MVNormal(_Param_Dist):
    """Multi-variate Normal/Gaussian parameter distribution with given means and covariance matrix.

    NOTE: use `clip` parameter to avoid extreme values.

    """

    def __init__(self, names, means, cov, clip=None, **kwargs):
        """Initialize a `PD_MVNormal` instance.

        Arguments
        ---------
        names: (n,) of str
            Names of the parameters being distributed.
        means : (n,) of float
            Mean / expectation value vector for the parameters. (Renamed from 'mean')
        cov : (n,n) of float
            Covariance matrix. Must be positive definite.
        clip : `None` or (n, 2) of float
            Bounds by which to restrict the resulting distribution. Must have
            shape (n, 2) where n is the number of parameters.

        """
        # Convert inputs to numpy arrays
        means = np.asarray(means)
        cov = np.asarray(cov)
        n_params = len(names)

        # --- SIZE AND DIMENSION CHECKS ---
        assert cov.ndim == 2, "Covariance matrix must be 2-dimensional."
        assert cov.shape[0] == cov.shape[1], "Covariance matrix must be square (n x n)."
        assert cov.shape[0] == n_params, (
            f"The number of names ({n_params}) must match the dimension of the covariance matrix ({cov.shape[0]})."
        )
        assert means.ndim == 1, "Mean vector must be 1-dimensional."
        assert len(means) == n_params, f"The length of the means vector ({len(means)}) must match the number of parameters ({n_params})."

        # --- VALIDITY CHECKS ---

        # 1. Positive Definiteness Check & Store Cholesky Factor (Single Call)
        # Cholesky decomposition only succeeds for Positive Definite matrices.
        try:
            self._L = cholesky(cov)  # Renamed to _L
        except LinAlgError:
            # If the decomposition fails, the matrix is not positive definite.
            raise AssertionError(
                "Covariance matrix must be positive definite (non-degenerate). Consider using `holodeck.utils.repair_covariance(cov)`."
            )

        # 2. Check for Symmetry (Physical requirement for covariance)
        assert np.allclose(cov, cov.T), "Covariance matrix must be symmetric."

        # Pass a single identifying name to the base class ### KG Note to self: probably want to have it be a tuple of all the names
        super().__init__(name=tuple(names), clip=clip, **kwargs)

        self._names = names
        self._cov = cov
        self._means = means  # Stored as _means

        # Create the frozen SciPy distribution (Accessed via sp.stats)
        self._frozen_dist = sp.stats.multivariate_normal(mean=means, cov=cov)

        return

    def _dist_func(self, xx):
        """
        Transforms uniform random numbers (xx) into Multivariate Normal samples.
        xx is expected to be a (n_samples, n_dims) array of uniform [0, 1] variables.
        """
        # 1. Transform uniform samples into standard normal samples (zz ~ N(0, I))
        # zz has mean 0 and covariance I (identity)
        zz = sp.stats.norm.ppf(xx)  # Accessed via sp.stats

        # 2. Apply the Cholesky factor (_L) and add the means vector (mu)
        # y = mu + _L @ z.T (then transpose back)
        # This transforms N(0, I) to N(mu, Sigma=_L*_L.T)
        yy = np.dot(self._L, zz.T).T + self._means  # Used _L and _means

        return yy

    @property
    def extrema(self):
        """Return the analytical extrema (min, max) of this parameter distribution.

        Multivariate Gaussians gracefully extend to infinity in all dimensions if unclipped.
        """
        if self._clip is not None:
            return self._clip

        return np.full((self.n_params, 2), [-np.inf, np.inf])

    @property
    def n_params(self):
        """Overrides base class to return the number of dimensions."""
        return len(self._names)


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
        loidx = xx <= lofrac
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
        loidx = xx <= lofrac
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
            yhi = self._edges[ii + 1]

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


def run_model(
    sam,
    hard,
    pta_dur=DEF_PTA_DUR,
    nfreqs=DEF_NUM_FBINS,
    nreals=DEF_NUM_REALS,
    nloudest=DEF_NUM_LOUDEST,
    gwb_flag=True,
    singles_flag=True,
    details_flag=False,
    params_flag=False,
    log=None,
):
    """Run the given SAM and hardening model to construct a binary population and GW signatures.

    Arguments
    ---------
    sam : :class:`holodeck.sams.sam.Semi_Analytic_Model` instance,
    hard : :class:`holodeck.hardening._Hardening` subclass instance,
    pta_dur : float, [seconds]
        Duration of PTA observations in seconds, used to determine Nyquist frequency basis at which
        GW signatures are calculated.
    nfreqs : int
        Number of Nyquist frequency bins at which to calculate GW signatures.
    nreals : int
        Number of 'realizations' (populations drawn from Poisson distributions) to construct.
    nloudest : int
        Number of loudest binaries to consider in each frequency bin.  These are the highest GW
        strain binaries in each frequency bin, for which the individual source strains are
        calculated.
    gwb_flag
    details_flag
    singles_flag
    params_flag
    log : ``logging.Logger`` instance

    Returns
    -------
    data : dict
        The population and GW data calculated from the simulation.  The dictionary elements are:

        * ``fobs_cents`` : Nyquist frequency bin centers, in units of [seconds].
        * ``fobs_edges`` : Nyquist frequency bin edgeds, in units of [seconds].

        * If ``details_flag == True``:

            * ``static_binary_density`` :
            * ``number`` :
            * ``redz_final`` :
            * ``gwb_params`` :
            * ``num_params`` :
            * ``gwb_mtot_redz_final`` :
            * ``num_mtot_redz_final`` :

        * If ``params_flag == True``:

            * ``sspar`` :
            * ``bgpar`` :

        * If ``singles_flag == True``:

            * ``hc_ss`` :
            * ``hc_bg`` :

        * If ``gwb_flag == True``:

            * ``gwb`` :

    """

    from holodeck.sams import sam_cyutils

    if not any([gwb_flag, details_flag, singles_flag, params_flag]):
        err = f"No flags set!  {gwb_flag=} {details_flag=} {singles_flag=} {params_flag=}"
        if log is not None:
            log.exception(err)
        raise RuntimeError(err)

    data = {}

    fobs_cents, fobs_edges = utils.pta_freqs(dur=pta_dur * YR, num=nfreqs)
    # convert from GW to orbital frequencies
    fobs_orb_cents = fobs_cents / 2.0
    fobs_orb_edges = fobs_edges / 2.0

    data["fobs_cents"] = fobs_cents
    data["fobs_edges"] = fobs_edges

    if not isinstance(hard, (holo.hardening.Fixed_Time_2PL_SAM, holo.hardening.Hard_GW)):
        err = f"`holo.hardening.Fixed_Time_2PL_SAM` must be used here!  Not {hard}!"
        if log is not None:
            log.exception(err)
        raise RuntimeError(err)

    redz_final, diff_num = sam_cyutils.dynamic_binary_number_at_fobs(fobs_orb_cents, sam, hard, cosmo)
    use_redz = redz_final
    edges = [sam.mtot, sam.mrat, sam.redz, fobs_orb_edges]
    number = sam_cyutils.integrate_differential_number_3dx1d(edges, diff_num)
    if details_flag:
        data["static_binary_density"] = sam.static_binary_density
        data["number"] = number
        data["redz_final"] = redz_final

        gwb_pars, num_pars, gwb_mtot_redz_final, num_mtot_redz_final = _calc_model_details(edges, redz_final, number)

        data["gwb_params"] = gwb_pars
        data["num_params"] = num_pars
        data["gwb_mtot_redz_final"] = gwb_mtot_redz_final
        data["num_mtot_redz_final"] = num_mtot_redz_final

    # calculate single sources and/or binary parameters
    if singles_flag or params_flag:
        nloudest = nloudest if singles_flag else 1

        vals = holo.single_sources.ss_gws_redz(
            edges,
            use_redz,
            number,
            realize=nreals,
            loudest=nloudest,
            params=params_flag,
        )
        if params_flag:
            hc_ss, hc_bg, sspar, bgpar = vals
            data["sspar"] = sspar
            data["bgpar"] = bgpar
        else:
            hc_ss, hc_bg = vals

        if singles_flag:
            data["hc_ss"] = hc_ss
            data["hc_bg"] = hc_bg

    if gwb_flag:
        gwb = holo.gravwaves._gws_from_number_grid_integrated_redz(edges, use_redz, number, nreals)
        data["gwb"] = gwb

    return data


def _calc_model_details(edges, redz_final, number):
    """Calculate derived properties from the given populations.

    Parameters
    ----------
    edges : (4,) list of 1darrays
        [mtot, mrat, redz, fobs_orb_edges] with shapes (M, Q, Z, F+1)
    redz_final : (M,Q,Z,F)
        Redshift final (redshift at the given frequencies).
    number : (M-1, Q-1, Z-1, F)
        Absolute number of binaries in the given bin (dimensionless).

    Returns
    -------
    gwb_pars
    num_pars
    gwb_mtot_redz_final
    num_mtot_redz_final

    """

    redz = edges[2]
    nmbins = len(edges[0]) - 1
    nzbins = len(redz) - 1
    nfreqs = len(edges[3]) - 1
    # (M-1, Q-1, Z-1, F) characteristic-strain squared for each bin
    hc2 = holo.gravwaves.char_strain_sq_from_bin_edges_redz(edges, redz_final)
    # strain-squared weighted number of binaries
    hc2_num = hc2 * number
    # (F,) total GWB in each frequency bin
    denom = np.sum(hc2_num, axis=(0, 1, 2))
    gwb_pars = []
    num_pars = []

    # Iterate over the parameters to calculate weighted averaged of [mtot, mrat, redz]
    for ii in range(3):
        # Get the indices of the dimensions that we will be marginalizing (summing) over
        # we'll also keep things in terms of redshift and frequency bins, so at most we marginalize
        # over 0-mtot and 1-mrat
        margins = [0, 1]
        # if we're targeting mtot or mrat, then don't marginalize over that parameter
        if ii in margins:
            del margins[ii]
        margins = tuple(margins)

        # Get straight-squared weighted values (numerator, of the average)
        numer = np.sum(hc2_num, axis=margins)
        # divide by denominator to get average
        tpar = numer / denom
        gwb_pars.append(tpar)

        # Get the total number of binaries
        tpar = np.sum(number, axis=margins)
        num_pars.append(tpar)

    # ---- calculate redz_final based distributions

    # get final-redshift at bin centers
    rz = redz_final.copy()
    for ii in range(3):
        rz = utils.midpoints(rz, axis=ii)

    gwb_mtot_redz_final = np.zeros((nmbins, nzbins, nfreqs))
    num_mtot_redz_final = np.zeros((nmbins, nzbins, nfreqs))
    gwb_rz = np.zeros((nzbins, nfreqs))
    num_rz = np.zeros((nzbins, nfreqs))
    for ii in range(nfreqs):
        rz_flat = rz[:, :, :, ii].flatten()
        # calculate GWB-weighted average final-redshift
        numer, *_ = sp.stats.binned_statistic(rz_flat, hc2_num[:, :, :, ii].flatten(), bins=redz, statistic="sum")
        tpar = numer / denom[ii]
        gwb_rz[:, ii] = tpar

        # calculate average final-redshift (number weighted)
        tpar, *_ = sp.stats.binned_statistic(rz_flat, number[:, :, :, ii].flatten(), bins=redz, statistic="sum")
        num_rz[:, ii] = tpar

        # Get values vs. mtot for redz-final
        for mm in range(nmbins):
            rz_flat = rz[mm, :, :, ii].flatten()
            numer, *_ = sp.stats.binned_statistic(rz_flat, hc2_num[mm, :, :, ii].flatten(), bins=redz, statistic="sum")
            tpar = numer / denom[ii]
            gwb_mtot_redz_final[mm, :, ii] = tpar

            tpar, *_ = sp.stats.binned_statistic(rz_flat, number[mm, :, :, ii].flatten(), bins=redz, statistic="sum")
            num_mtot_redz_final[mm, :, ii] = tpar

    gwb_pars.append(gwb_rz)
    num_pars.append(num_rz)

    return gwb_pars, num_pars, gwb_mtot_redz_final, num_mtot_redz_final


def load_pspace_from_path(path, space_class=None):
    """Load a ``_Param_Space`` subclass instance from a save file with the given path.

    This function tries to determine the correct class based on the save file name, then uses that
    class to load the save itself.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to directory containing save file.
        A single file matching `*.pspace.npz` is required in that directory.
        NOTE: the specific glob pattern is specified by `holodeck.librarian.PSPACE_FILE_SUFFIX` e.g. '.pspace.npz'
    space_class : `_Param_Space` subclass  or  `None`
        Class with which to call the `from_save()` method to load a new `_Param_Space` instance.
        If `None` is given, then the filename is used to try to determine the appropriate class.

    Returns
    -------
    space : `_Param_Space` subclass instance
        An instance of the `space_class` class.
    space_fname : pathlib.Path
        File that `space` was loaded from.

    """
    path = Path(path).absolute().resolve()
    log.debug(f"Loading pspace from path '{path}'")
    if not path.exists():
        err = f"Cannot load pspace from path that does not exist!  {path}"
        log.exception(err)
        raise RuntimeError(err)

    # If this is a directory, look for a pspace save file
    if path.is_dir():
        pattern = "*" + holo.librarian.PSPACE_FILE_SUFFIX
        space_fname = list(path.glob(pattern))
        if len(space_fname) != 1:
            err = f"found {len(space_fname)} matches to {pattern} in output {path}!"
            log.exception(err)
            raise FileNotFoundError(err)

        space_fname = space_fname[0]
        log.debug(f"Found space file '{space_fname}'")

    # if this is a file, assume that it's already the pspace save file
    elif path.is_file():
        space_fname = path

    else:
        raise

    if space_class is None:
        try:
            space_class = str(np.load(space_fname, allow_pickle=True)["class_name"])
            space_class = holo.librarian.param_spaces_dict[space_class]
        except Exception as err:
            log.error(f"Could not load `class_name` from save file '{space_fname}'.")
            log.error(str(err))

            # Based on the `space_fname`, try to find a matching PS (parameter-space) in `holodeck.param_spaces_dict`
            space_class = _get_space_class_from_space_fname(space_fname)

    log.debug(f"Constructing parameter space from {space_class} : {space_fname}")
    space = space_class.from_save(space_fname, log=log)
    return space, space_fname


def _get_space_class_from_space_fname(space_fname):
    # Based on the `space_fname`, try to find a matching PS (parameter-space) in `holodeck.param_spaces_dict`
    space_name = space_fname.name.split(".")[0]
    space_class = holo.librarian.param_spaces_dict[space_name]
    return space_class


def _get_sim_fname(path, pnum, library=True):
    if library:
        temp = holo.librarian.FNAME_LIBRARY_SIM_FILE
    else:
        temp = holo.librarian.FNAME_DOMAIN_SIM_FILE

    temp = temp.format(pnum=pnum)
    temp = path.joinpath(temp)
    return temp


def get_sam_lib_fname(path, gwb_only, library=True):
    # standard 'library'
    if library:
        fname = FNAME_LIBRARY_COMBINED_FILE
    # 'domain' of parameter space
    else:
        fname = FNAME_DOMAIN_COMBINED_FILE

    if gwb_only:
        fname += "_gwb-only"
    lib_path = path.joinpath(fname).with_suffix(".hdf5")
    return lib_path


def get_fits_path(library_path):
    """Get the name of the spectral fits file, given a library file path."""
    fits_path = library_path.with_stem(library_path.stem + "_fits")
    fits_path = fits_path.with_suffix(".npz")
    return fits_path


def log_mem_usage(log):

    try:
        import resource

        # results.ru_maxrss is KB on Linux, B on macos
        mem_max = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macos systems
        if sys.platform.lower().startswith("darwin"):
            mem_max = mem_max / 1024**3
        # linux systems
        else:
            mem_max = mem_max / 1024**2
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
