"""Utilities for Gaussian Processes."""
import time
import warnings
from functools import partial

import h5py
import jax
import jax.numpy as jnp
import jaxopt
import numpy as np
import scipy.signal as ssig
import tinygp
from holodeck import utils
from holodeck.constants import YR
from tinygp import kernels, transforms

# If you OOM frequently, try setting this below or lowering the fraction
# This is the fraction of the GPUs memory that will be preallocated for jax
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]="0.7"

jax.config.update('jax_enable_x64', True)
# This may end up being useful if we parallelize over mutliple GPUs
# It is MPI aware
#jax.distributed.initialize()
VERBOSE = True

FLOOR_STRAIN_SQUARED = 1e-40
FILTER_DEF_WINDOW_LENGTH = 7
FILTER_DEF_POLY_ORDER = 3



class GaussProc(object):
    """The gaussian process object.

    Attributes
    ----------
    x : array_like
        Input parameters for GP training
    y : array_like
        Input data from GP training
    yerr : array_like
        Error on `y`
    par_dict : dict
        Dictionary containing parameter names and their min and max values
    kernel : str, optional
        The type of kernel to use for the GP
    kernel_opts : dict, optional
        Dictionary of kwargs to pass to george.kernels when creating the kernels

    Methods
    -------
    lnprior
        Compute log prior
    lnlike
        Compute log likelihood
    lnprob
        Compute log posterior probability

    """

    def __init__(self,
                 x,
                 y,
                 yerr=None,
                 par_dict=None,
                 kernel="ExpSquaredKernel",
                 kernel_opts={}):

        self.x = x
        self.y = y
        self.yerr = yerr
        self.par_dict = par_dict

        # Validate kernel
        # Get kernels available as list[str]
        kernel_list = [kern.__name__ for kern in
                       kernels.stationary.Stationary.__subclasses__() + kernels.quasisep.Quasisep.__subclasses__()]
        # Lowercase them
        kernel_lcase = list(map(str.lower, kernel_list))
        try:
            self.kernel = kernel_list[kernel_lcase.index(kernel.lower())]
            self.kernel_class = getattr(kernels, self.kernel)
        except ValueError:
            print(f"Unexpected kernel '{kernel}'.")
            print("Acceptable values are:\n", *kernel_list, sep="\n")
            raise

        self.kernel_opts = kernel_opts

        # The number of GP parameters is one more than the number of spectra parameters.
        self.pmax = jnp.full(len(self.par_dict) + 1, 20.0)  # sampling ranges
        self.pmin = jnp.full(len(self.par_dict) + 1, -20.0)  # sampling ranges
        self.emcee_flatlnprob = None
        self.emcee_kernel_map = None

        # Instantiate empty attributes
        self.mean_spectra = None
        self.kernel_map = None
        self.chain = None

    def lnprior(self, p):

        if jnp.all(p <= self.pmax) and jnp.all(p >= self.pmin):
            logp = jnp.sum(jnp.log(1 / (self.pmax - self.pmin)))
        else:
            logp = -jnp.inf

        return logp


    @partial(jax.jit, static_argnums=(0,))
    def lnlike(self, p):

        gp = self.build_gp(p)
        return gp.log_probability(self.y)

    @partial(jax.jit, static_argnums=(0,))
    def neglnlike(self, p):

        return -self.lnlike(p)

    @partial(jax.jit, static_argnums=(0,))
    def lnprob(self, p):

        return self.lnprior(p) + self.lnlike(p)

    def build_gp(self, p):
        """Build a tinygp.

        This function exists to be put in other functions and then jax.jit'd.
        IT WILL BE PAINFULLY SLOW IF YOU DO NOT.
        """
        # Jax likes to be weird with dims sometimes
        p = p.flatten()

        # Get parameters in the right units
        a, tau = jnp.exp(p[0]), jnp.exp(p[1:])

        # Build the kernel
        kernel = a * transforms.Linear(tau, self.kernel_class(**self.kernel_opts))

        # Instantiate an tinygp GP
        gp = tinygp.gp.GaussianProcess(kernel, self.x, diag=self.yerr)
        return gp


def train_gp(spectra_file,
             nfreqs=30,
             test_frac=0.0,
             center_measure="median",
             kernel="ExpSquared",
             kernel_opts={},):
    """Train gaussian processes on the first `nfreqs` of the GWB in `spectra_file`.

    Parameters
    ----------
    spectra_file : str or pathlib.Path
        The spectral library
    nfreqs : int
        The number of frequencies to train on, starting with the lowest in the
        library
    test_frac : float
        Fraction of LHS points to reserve for testing. Reserves this fraction at the beginning of the samples.
    center_measure : str, optional
        The measure of center for the dataset that the GP will be trained on. Can be
        either "mean" or "median"
    kernel : str, optional
        The type of kernel to use for the GP
    kernel_opts : dict, optional
        The options to pass when constructing the kernel. Unpacks as **kwargs to george.kernels

    Returns
    -------
    Trained GPs

    Examples
    --------
    FIXME: Add docs.

    """
    spectra = h5py.File(spectra_file, "r")

    if VERBOSE:
        print(f"Loaded spectra from {spectra_file}")

    # Get smoothed GWB
    gp_freqs, xobs, yerr, yobs, yobs_mean = get_smoothed_gwb(spectra, nfreqs,
                                                             test_frac,
                                                             center_measure)

    pars = list(spectra.attrs["param_names"].astype(str))

    # xobs = get_parameter_values(spectra, test_frac)

    gp_tinygp, num_kpars = create_gp_kernels(gp_freqs, pars, xobs, yerr, yobs,
                                             kernel, kernel_opts)

    # Sample the posterior distribution of the kernel parameters
    # to find MAP value for each frequency.

    fit_kernel_params(gp_freqs, yobs_mean, gp_tinygp, num_kpars)

    return gp_tinygp


def get_smoothed_gwb(spectra,
                     nfreqs,
                     test_frac=0.0,
                     center_measure="median"):
    """Get the smoothed GWB from a number of realizations.

    Parameters
    ----------
    spectra : h5py._hl.files.File
        The variable containing the library in HDF5 format
    nfreqs : int
        The number of frequencies to train on, starting with the lowest in the
        library
    test_frac : float, optional
        The fraction of the data to reserve at the beginning as a test set
    center_measure : str, optional
        The measure of center for the dataset that the GP will be trained on. Can be
        either "mean" or "median"

    Returns
    -------
    gp_freqs : numpy.array
        The frequencies corresponding to the GWB data
    xobs : numpy.array
        A numpy array containing the parameters used to generate each GWB in `spectra`
    yerr : numpy.array
        The error on the GWB training data
    yobs : numpy.array
        The smoothed, zero-mean GWB training data
    yobs_mean : numpy.array
        The original smoothed mean of the GWB training data

    Examples
    --------
    FIXME: Add docs.

    """
    # Filter out NaN values which signify a failed sample point
    # shape: (samples, freqs, realizations)
    gwb_spectra = spectra['gwb'][:]
    xobs = spectra['sample_params'][:]
    bads = np.any(np.isnan(gwb_spectra), axis=(1, 2))
    if VERBOSE:
        print(f"Found {utils.frac_str(bads)} samples with NaN entries.  Removing them from library.")
    # when sample points fail, all parameters are set to zero.  Make sure this is consistent
    if not np.all(xobs[bads] == 0.0):
        raise RuntimeError("NaN elements of `gwb` did not correspond to zero elements of `sample_params`!")
    # Select valid spectra, and sample parameters
    gwb_spectra = gwb_spectra[~bads]
    xobs = xobs[~bads]
    # Make sure old/deprecated parameters are not in library
    if 'mmb_amp' in spectra.attrs['param_names']:
        raise RuntimeError("Parameter `mmb_amp` should not be here!  Needs to be log-spaced (`mmb_amp_log10`)!")

    # Cut out portion for test set later
    test_ind = int(gwb_spectra.shape[0] * test_frac)
    if VERBOSE:
        print(f"setting aside {test_frac} of samples ({test_ind}) for testing, and choosing {nfreqs} frequencies")

    gwb_spectra = gwb_spectra[test_ind:, :nfreqs, :]**2
    xobs = xobs[test_ind:, :]

    # Find all the zeros and set them to be h_c = 1e-20
    low_ind = (gwb_spectra < FLOOR_STRAIN_SQUARED)
    gwb_spectra[low_ind] = FLOOR_STRAIN_SQUARED

    # Find mean or median over realizations
    if center_measure.lower() == "median":
        center = np.log10(np.median(gwb_spectra, axis=-1))
    elif center_measure.lower() == "mean":
        center = np.log10(np.mean(gwb_spectra, axis=-1))
    else:
        raise ValueError(
            f"`center_measure` must be 'mean' or 'median', not '{center_measure}'"
        )

    # Smooth Mean Spectra
    filter_window = FILTER_DEF_WINDOW_LENGTH
    filter_poly_order = FILTER_DEF_POLY_ORDER
    if (filter_window is not None) and (nfreqs < filter_window):
        print(f"WARNING: {nfreqs=} < {filter_window=}, resetting default value")
        if nfreqs < 4:
            filter_window = None
            filter_poly_order = None
        else:
            filter_window = nfreqs // 2 + 1 if (nfreqs // 2) % 2 == 0 else nfreqs // 2
            filter_poly_order = filter_window // 2
        print(f"         {filter_window=} {filter_poly_order=}")

    if filter_window is not None:
        smooth_center = ssig.savgol_filter(center, filter_window, filter_poly_order)
    else:
        if VERBOSE:
            print("Not using any smoothing on center spectrum.")
        smooth_center = center

    # Find std
    err = np.std(np.log10(gwb_spectra), axis=-1)

    # The "y" data are the medians or means and errors for the spectra at each point in parameter space
    yobs = smooth_center.copy()  # mean.copy()
    yerr = err.copy()
    gp_freqs = spectra["fobs"][:nfreqs].copy()
    gp_freqs *= YR

    # Find mean in each frequency bin (remove it before analyzing with the GP)
    # This allows the GPs to oscillate around zero, where they are better behaved.
    yobs_mean = np.mean(yobs, axis=0)
    yobs -= yobs_mean[None, :]

    return gp_freqs, xobs, yerr, yobs, yobs_mean


def create_gp_kernels(gp_freqs, pars, xobs, yerr, yobs, kernel, kernel_opts):
    """Instantiate GP kernel for each frequency.

    Parameters
    ----------
    gp_freqs : numpy.array
        The frequencies corresponding to the GWB data
    pars : list
        Ordered list of parameters
    xobs : numpy.array
        The array of parameters to train on
    yerr : numpy.array
        The error on the GWB data
    yobs : numpy.array
        The smoothed, zero-mean GWB data
    kernel : str, optional
        The type of kernel to use for the GP

    Returns
    -------
    gp_tinygp : list[tinygp.gp.GaussianProcess]
        The created GP kernels
    nkpars : int
        Numer of kernel parameters

    Examples
    --------
    FIXME: Add docs.

    """
    # Instantiate a list of GP kernels and models [one for each frequency]
    gp_tinygp = []
    # Create the parameter dictionary for the gp objects
    par_dict = dict()
    for ind, par in enumerate(pars):
        par_dict[par] = {
            "min": np.min(xobs[:, ind]),
            "max": np.max(xobs[:, ind])
        }

    for freq_ind in range(len(gp_freqs)):
        gp_tinygp.append(
            GaussProc(xobs, yobs[:, freq_ind], yerr[:, freq_ind], par_dict,
                      kernel, kernel_opts))

    num_kpars = len(par_dict) + 1

    return gp_tinygp, num_kpars


def fit_kernel_params(gp_freqs,
                      yobs_mean,
                      gp_tinygp,
                      nkpars):
    """Fit the parameters of the GP kernels.

    Parameters
    ----------
    gp_freqs : numpy.array
        The frequencies corresponding to the GWB data
    yobs_mean : numpy.array
        The smoothed mean of the GWB data
    gp_tinygp : list[GaussProc]
        The GP model that has been read in from a .PKL file
    nkpars : int
        Number of kernel parameters
    nwalkers : int
        Number of emcee walkers to use
    nsamples : int
        Number of emcee samples
    burn_frac : float
        Burn-in fraction to discard from chains


    Examples
    --------
    FIXME: Add docs.

    """
    ndim = nkpars
    rng_key = jax.random.PRNGKey(34928)

    for freq_ind in range(len(gp_freqs)):
        # Paralellize emcee with nwalkers //2 or the maximum number of processors available, whichever is smaller
        # with Pool(min(nwalkers // 2, cpu_count())) as pool:
        gp = gp_tinygp[freq_ind]
        print("Starting minimization routine")
        t_start = time.time()
        rng_key, _ = jax.random.split(rng_key)
        p0 = 1.0e-4 * jax.random.normal(rng_key, (ndim,))

        solver = jaxopt.ScipyMinimize(fun=gp.neglnlike, method='l-bfgs-b')
        soln = solver.run(p0)
        print(
            f"Completed {freq_ind} out of {len(gp_freqs)-1} in {(time.time() - t_start) / 60.0:.2f} min\n"
        )

        # Populate the GP class with the details of the kernel
        # MAP values for each frequency.
        gp.emcee_flatchain = soln.params
        gp.emcee_flatlnprob = -1*soln.state.fun_val

        gp.emcee_kernel_map = soln.params
        # add-in mean yobs (freq) values
        gp.mean_spectra = yobs_mean[freq_ind]


def set_up_predictions(spectra, gp_tinygp):
    """Set up a list of GPs ready for predictions.

    Parameters
    ----------
    spectra : h5py._hl.files.File
        The variable containing the library in HDF5 format
    gp_tinygp : list[GaussProc]
        The GP model that has been read in from a .PKL file

    Returns
    -------
    gp_list : list[tinygp.gp.GaussianProcess]
        The configured GPs ready for predictions

    Examples
    --------
    FIXME: Add docs.


    """
    gp_list = []
    gp_freqs = spectra["fobs"][:len(gp_tinygp)].copy()

    # Check which attribute holds the kernel map. In older versions, we used
    # self.kernel_map. However, to be consistent we have switched to
    # self.emcee_kernel_map. The following lines are just for backwards
    # compatibility.

    if getattr(gp_tinygp[0], "kernel_map", None) is not None:
        kernel_map_attr = "kernel_map"
    elif getattr(gp_tinygp[0], "emcee_kernel_map", None) is not None:
        kernel_map_attr = "emcee_kernel_map"

    for ii, _ in enumerate(gp_freqs):
        gp_kparams = jnp.exp(getattr(gp_tinygp[ii], kernel_map_attr))

        gp_list.append(gp_tinygp[ii].build_gp(gp_kparams))

    return gp_list


@partial(jax.jit, static_argnums=(0,))
def predict(gp_tinygp, env_pars):
    """Make a prediction using tinygp.

    Parameters
    ----------
    gp_tinygp : GaussProc
        The GaussProc object
    env_pars : list[float]
        List of input parameters

    Examples
    --------
    FIXME: Add docs.

    """
    mu_pred, cov_pred = gp_tinygp.build_gp(gp_tinygp.emcee_kernel_map).predict(jnp.array(gp_tinygp.y), jnp.array(env_pars), return_cov=True)
    return mu_pred, cov_pred


def mean_par_dict(gp_tinygp):
    """Create a dictionary that is of the form parameter:mean(parameter_range).

    Parameters
    ----------
    gp_tinygp : list[GaussProc]
        The GP model that has been read in from a .PKL file

    Returns
    -------
    mean_pars : dict
        Dictionary with mean values

    Examples
    --------
    FIXME: Add docs.


    """
    mean_pars = {
        key:
        (gp_tinygp[0].par_dict[key]["max"] + gp_tinygp[0].par_dict[key]["min"]) / 2
        for key in gp_tinygp[0].par_dict.keys()
    }

    return mean_pars


def pars_linspace_dict(gp_tinygp, num_points=5):
    """Create a dictionary that is of the form parameter:linspace(min(parameter), max(parameter), num=`num_points`).

    Parameters
    ----------
    gp_tinygp : list[GaussProc]
        The GP model that has been read in from a .PKL file
    num_points : int
        Number of evenly-spaced samples in the linspace

    Returns
    -------
    pars_linspace : dict
        The dictionary of the form parameter:linspace

    Examples
    --------
    FIXME: Add docs.


    """
    pars_linspace = {
        key: np.linspace(
            gp_tinygp[0].par_dict[key]["min"],
            gp_tinygp[0].par_dict[key]["max"],
            num=num_points,
        )
        for key in gp_tinygp[0].par_dict.keys()
    }

    return pars_linspace


def hc_from_gp(gp_tinygp, env_pars, nfreqs=None):
    """Calculate the characteristic strain using a GP.

    Parameters
    ----------
    gp_tinygp : list[GaussProc]
        The GP model that has been read in from a .PKL file
    gp_list : list[tinygp.gp.GaussianProcess]
        The configured GPs ready for predictions
    env_pars : list
        List of ordered parameters for GP to use as input

    Returns
    -------
    hc : numpy.array
        The array of characteristic strains
    rho : numpy.array
        Array of predictive distribution means from GP, shifted by the original data's means.
    rho_pred : numpy.array
        Array of predictive distribution means from GP. It is import to remember
        that the training data was transformed to have zero mean.

    Examples
    --------
    FIXME: Add docs.


    """
    if nfreqs is None:
        nfreqs = len(gp_tinygp)

    elif nfreqs > len(gp_tinygp):
        raise ValueError(f"You requested {nfreqs} frequencies, but there are only {len(gp_tinygp)} GPs!")

    rho_pred = np.zeros((nfreqs, 2))
    for ii in range(nfreqs):
        mu_pred, cov_pred = predict(gp_tinygp[ii], [env_pars])
        if np.diag(cov_pred) < 0.0:
            rho_pred[ii, 0], rho_pred[ii, 1] = mu_pred, 1e-5 * mu_pred
        else:
            rho_pred[ii, 0], rho_pred[ii,
                                      1] = mu_pred, np.sqrt(np.diag(cov_pred))

    # transforming from zero-mean unit-variance variable to rho
    rho = (np.array([gp_tinygp[ii].mean_spectra
                     for ii in range(nfreqs)]) + rho_pred[:, 0])
    hc = np.sqrt(10**rho)
    return hc, rho, rho_pred


def sample_hc_from_gp(gp_tinygp, gp_list, env_pars, nsamples=100):
    """Calculate the characteristic strain using a GP.

    Parameters
    ----------
    gp_tinygp : list[GaussProc]
        The GP model that has been read in from a .PKL file
    gp_list : list[tinygp.gp.GaussianProcess]
        The configured GPs ready for predictions
    env_pars : list
        List of ordered parameter values for GP to use as input
    nsamples : int
        The number of samples to draw

    Returns
    -------
    hc : numpy.array
        The array of characteristic strains of shape (samples, freqs)

    Examples
    --------
    FIXME: Add docs.

    """
    # Warn if samples < 100
    if nsamples < 100:
        warnings.warn(
            "Variance recovery doesn't usually saturate unless nsamples >= 100. Consider a higher number of smamples!",
            category=UserWarning)

    # First, get iterable of arguments for the helper function
    # jax will not instatiate a random seed. You have to split the key or
    # it will make the same draw every time
    args = [(gp_list[i], gp_tinygp[i], env_pars, nsamples,
             jax.random.split(jax.random.PRNGKey(42))[1])
            for i in range(len(gp_list))]

    hc = []
    for arg in args:
        hc.append(_sample_hc_from_gp_helper(*arg))
    hc = jnp.array(hc).squeeze()

    # The multiprocessing routine returns hc in shape (freqs, samples), but it
    # makes more sense to have (samples, freqs). So, take the transpose
    return hc.T


@partial(jax.jit, static_argnums=(0, 1))
def _sample_hc_from_gp_helper(gp_at_freq,
                              gp_tinygp_at_freq,
                              env_pars,
                              nsamples,
                              key):
    """Helper function for `sample_hc_from_gp()`.

    This function returns samples of the GP predicted characteristic strain for
    a given frequency. It is not meant to be called directly, but instead is
    called by `sample_hc_from_gp()` which uses it to parallelize this process
    over the frequencies of interest.

    Parameters
    ----------
    gp_at_freq : GaussProc
        The read-in GaussProc object at a given frequency
    gp_tinygp_at_freq : tinygp.gp.GaussianProcess
        The configured george GP object at a given frequency
    env_pars : list
        List of ordered parameter values for GP to use as input
    nsamples : int
        The number of samples to draw

    Returns
    -------
    hc : numpy.array
        Characteristic strain array at given frequency of shape (nsamples)

    Examples
    --------
    FIXME: Add docs.


    """
    # Get the samples

    hc = jnp.zeros(nsamples)

    gp = gp_at_freq.condition(jnp.array(gp_tinygp_at_freq.y), jnp.array([env_pars])).gp

    # transforming from zero-mean unit-variance variable to rho
    rho_sample = gp.sample(key, shape=(nsamples, ))
    rho = jnp.array(gp_tinygp_at_freq.mean_spectra) + rho_sample

    hc = jnp.sqrt(10**rho)

    return hc
