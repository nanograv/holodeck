"""Utilities for Gaussian Processes."""
import sys
import time
import warnings
from functools import reduce
from multiprocessing import Pool, cpu_count

import emcee
import george
import george.kernels as kernels
import h5py
import numpy as np
import schwimmbad
from holodeck import utils
from holodeck.constants import YR

VERBOSE = True

FLOOR_STRAIN_SQUARED = 1e-40
FLOOR_ERR = 1.0


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
                 y_is_variance=False,
                 par_dict=None,
                 kernel="ExpSquaredKernel"):
        self.x = x
        self.y = y
        self.y_is_variance = y_is_variance
        self.yerr = yerr
        self.par_dict = par_dict
        self.par_list = list(par_dict.keys())

        # Validate kernel
        # Get kernels available as list[str]
        kernel_list = [cls.__name__ for cls in kernels.Kernel.__subclasses__()]
        # Lowercase them
        kernel_lcase = list(map(str.lower, kernel_list))
        try:
            self.kernel = {
                par: kernel_list[kernel_lcase.index(kernel[par].lower())]
                for par in kernel.keys()
            }
            self.kernel_class = {
                par: getattr(kernels, self.kernel[par])
                for par in self.kernel.keys()
            }
        # This will only utils.mpi_print the first kernel error, but subsequent runs will
        # catch the rest
        except ValueError as e:
            utils.mpi_print(f"Unexpected kernel given'{str(e)}'.")
            utils.mpi_print("Acceptable values are:\n", *kernel_list, sep="\n")
            raise

        # The number of GP parameters is equal to the number of spectra parameters
        # + the number of kernel-specific parameters (per spectra parameter) + one.
        self.uses_rational_quadratic = [par for par in self.kernel.keys()
                                        if self.kernel[par] == 'RationalQuadraticKernel']
        self.pmax = np.full(len(self.par_dict) + 1 + len(self.uses_rational_quadratic), 20.0)  # sampling ranges
        self.pmin = np.full(len(self.par_dict) + 1 + len(self.uses_rational_quadratic), -20.0)  # sampling ranges
        self.emcee_flatchain = None
        self.emcee_flatlnprob = None
        self.emcee_kernel_map = None

        # Instantiate empty attributes
        self.mean_spectra = None
        self.kernel_map = None
        self.chain = None

    def lnprior(self, p):

        if np.all(p <= self.pmax) and np.all(p >= self.pmin):
            logp = np.sum(np.log(1 / (self.pmax - self.pmin)))
        else:
            logp = -np.inf

        return logp

    def create_kernel(self, p):
        # Get parameters for kernel
        additional_pars = len(self.uses_rational_quadratic)
        if additional_pars == 0:
            a, tau = np.exp(p[0]), np.exp(p[1:])

            # Use list comprehension to generate list of kernels, one for each parameter
            kernel_list = [
                self.kernel_class[par](metric=tau[self.par_list.index(par)],
                                       ndim=len(self.par_list),
                                       axes=self.par_list.index(par))
                for par in self.par_list
            ]

        else:
            a, tau, log_alpha = np.exp(p[0]), np.exp(
                p[1:-additional_pars]), p[-additional_pars:]
            # Use list comprehension to generate list of kernels, one for each parameter
            kernel_list = [
                self.kernel_class[par](log_alpha=log_alpha[
                                       self.uses_rational_quadratic.index(par)],
                                       metric=tau[self.par_list.index(par)],
                                       ndim=len(self.par_list),
                                       axes=self.par_list.index(par))
                if self.kernel[par] == "RationalQuadraticKernel" else
                self.kernel_class[par](metric=tau[self.par_list.index(par)],
                                       ndim=len(self.par_list),
                                       axes=self.par_list.index(par))
                for par in self.par_list
            ]

        # Add scale parameter and take the product of the kernels
        kernel = a * reduce(lambda k1, k2: k1 * k2, kernel_list)

        return kernel

    def lnlike(self, p):

        try:
            gp = george.GP(self.create_kernel(p))
            gp.compute(self.x, self.yerr)

            # lnlike = gp.lnlikelihood(self.y, quiet=True)
            lnlike = gp.log_likelihood(self.y, quiet=True)
        except np.linalg.LinAlgError:
            lnlike = -np.inf

        return lnlike

    def lnprob(self, p):

        return self.lnprior(p) + self.lnlike(p)


def train_gp(spectra_file,
             nfreqs=30,
             nwalkers=36,
             nsamples=1500,
             burn_frac=0.25,
             test_frac=0.0,
             center_measure="median",
             y_is_variance=False,
             kernel="ExpSquaredKernel",
             mpi=True):
    """Train gaussian processes on the first `nfreqs` of the GWB in `spectra_file`.

    Parameters
    ----------
    spectra_file : str or pathlib.Path
        The spectral library
    nfreqs : int
        The number of frequencies to train on, starting with the lowest in the
        library
    nwalkers : int
        The number of MCMC walkers to use
    nsamples : int
        Number of emcee samples
    burn_frac : float
        Burn-in fraction to discard from chains
    test_frac : float
        Fraction of LHS points to reserve for testing. Reserves this fraction at the beginning of the samples.
    center_measure : str, optional
        The measure of center for the dataset that the GP will be trained on. Can be
        either "mean" or "median"
    kernel : str, optional
        The type of kernel to use for the GP
    kernel_opts : dict, optional
        The options to pass when constructing the kernel. Unpacks as **kwargs to george.kernels
    mpi : bool, optional
        Whether to use MPI or Python's multiprocessing module

    Returns
    -------
    Trained GPs

    Examples
    --------
    FIXME: Add docs.

    """
    spectra = h5py.File(spectra_file, "r")

    if VERBOSE:
        utils.mpi_print(f"Loaded spectra from {spectra_file}")

    # Get GWB
    gp_freqs, xobs, yerr, yobs, yobs_mean = get_gwb(spectra, nfreqs, test_frac,
                                                    center_measure)

    pars = list(spectra.attrs["param_names"].astype(str))

    # xobs = get_parameter_values(spectra, test_frac)

    # Check if we want the training data to be the center `yobs` or the variance `yerr`
    # In the `if` conditional, I've included the arguments as keyword arguments
    # to make this distinction clearer
    if y_is_variance:
        gp_george, num_kpars = create_gp_kernels(gp_freqs,
                                                 pars,
                                                 xobs,
                                                 yerr=(yerr/np.sqrt(2*spectra['gwb'].shape[-1] - 2)),
                                                 yobs=yerr,
                                                 y_is_variance=y_is_variance,
                                                 kernel=kernel)

    else:
        gp_george, num_kpars = create_gp_kernels(gp_freqs,
                                                 pars,
                                                 xobs,
                                                 yerr=(yerr/np.sqrt(spectra['gwb'].shape[-1])),
                                                 yobs=yobs,
                                                 y_is_variance=y_is_variance,
                                                 kernel=kernel)

    # Sample the posterior distribution of the kernel parameters
    # to find MAP value for each frequency.

    fit_kernel_params(gp_freqs, yobs_mean, gp_george, num_kpars, nwalkers,
                      nsamples, burn_frac, mpi)

    return gp_george


def get_gwb(spectra, nfreqs, test_frac=0.0, center_measure="median"):
    """Get the GWB from a number of realizations.

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
        The zero-mean GWB training data
    yobs_mean : numpy.array
        The original mean of the GWB training data

    Examples
    --------
    FIXME: Add docs.

    """
    # Filter out NaN values which signify a failed sample point
    # shape: (samples, freqs, realizations)
    gwb_spectra = spectra['gwb']
    xobs = spectra['sample_params']
    bads = np.any(np.isnan(gwb_spectra), axis=(1, 2))
    if VERBOSE:
        utils.mpi_print(f"Found {utils.frac_str(bads)} samples with NaN entries.  Removing them from library.")
    # when sample points fail, all parameters are set to zero.  Make sure this is consistent
    if not np.all(xobs[bads] == 0.0):
        raise RuntimeError(f"NaN elements of `gwb` did not correspond to zero elements of `sample_params`!")
    # Select valid spectra, and sample parameters
    gwb_spectra = gwb_spectra[~bads]
    xobs = xobs[~bads]
    # Make sure old/deprecated parameters are not in library
    if 'mmb_amp' in spectra.attrs['param_names']:
        raise RuntimeError("Parameter `mmb_amp` should not be here!  Needs to be log-spaced (`mmb_amp_log10`)!")

    # Cut out portion for test set later
    test_ind = int(gwb_spectra.shape[0] * test_frac)
    if VERBOSE:
        utils.mpi_print(f"setting aside {test_frac} of samples ({test_ind}) for testing, and choosing {nfreqs} frequencies")

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

    # Get realizations that are all low. We will later use this
    # boolean array to set a noise floor
    # I've done it this way in case only certain frequencies have
    # all ~0 realizations.
    low_real = np.all(low_ind, axis=-1)

    # Find std
    # Where low_real is True, return 1.0
    # else return the std along the realization dimension
    err = np.where(low_real, FLOOR_ERR, np.std(np.log10(gwb_spectra), axis=-1))

    # The "y" data are the medians or means and errors for the spectra at each point in parameter space
    yobs = center.copy()  # mean.copy()
    yerr = err.copy()
    gp_freqs = spectra["fobs"][:nfreqs].copy()
    gp_freqs *= YR

    # Find mean in each frequency bin (remove it before analyzing with the GP)
    # This allows the GPs to oscillate around zero, where they are better behaved.
    yobs_mean = np.mean(yobs, axis=0)
    yobs -= yobs_mean[None, :]

    return gp_freqs, xobs, yerr, yobs, yobs_mean


'''
def get_parameter_values(spectra, test_frac=0.0):
    """Get array of GWB parameters.

    Given list `pars` of ordered parameters, return an array of parameter
    values in that order corresponding to each GWB in `spectra`.

    Parameters
    ----------
    spectra : h5py._hl.files.File
        The variable containing the library in HDF5 format

    Returns
    -------
    xobs : numpy.array
        A numpy array containing the parameters used to generate each GWB in `spectra`

    Examples
    --------
    FIXME: Add docs.

    """
    # Cut out portion for test set later
    test_ind = int(spectra['gwb'].shape[0] * test_frac)

    pars = list(spectra.attrs["param_names"].astype(str))

    # The "x" data are the actual parameter values
    xobs = np.zeros((spectra["gwb"].shape[0] - test_ind, len(pars)))
    for ii in range(xobs.shape[0]):
        for k, par in enumerate(pars):
            # Make sure to account for test set offset
            xobs[ii, k] = spectra["sample_params"][ii + test_ind, k]

    # Put mmb_amp in logspace if it exists and isn't already
    if "mmb_amp" in pars and np.any(xobs[:, pars.index("mmb_amp")] > 100):
        xobs[:, pars.index("mmb_amp")] = np.log10(xobs[:,
                                                       pars.index("mmb_amp")])

    return xobs
'''


def create_gp_kernels(gp_freqs, pars, xobs, yerr, yobs, y_is_variance, kernel):
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
        The zero-mean GWB data
    y_is_variance: bool
        Whether or not this GP is trained on the variance of the data
    kernel : dict
        The dictionary mapping parameters to the kernels that will be used {par:kernel}

    Returns
    -------
    gp_george : list[george.gp.GP]
        The created GP kernels
    num_kpars : int
        Numer of kernel parameters

    Examples
    --------
    FIXME: Add docs.

    """
    # Instantiate a list of GP kernels and models [one for each frequency]
    gp_george = []
    # Create the parameter dictionary for the gp objects
    par_dict = dict()
    for ind, par in enumerate(pars):
        par_dict[par] = {
            "min": np.min(xobs[:, ind]),
            "max": np.max(xobs[:, ind])
        }

    for freq_ind in range(len(gp_freqs)):
        gp_george.append(
            GaussProc(xobs, yobs[:, freq_ind], yerr[:, freq_ind], y_is_variance, par_dict,
                      kernel))

    # get the length of one of the prior bound lists
    # this is the number of kernel parameters
    num_kpars = len(gp_george[0].pmax)

    return gp_george, num_kpars


def fit_kernel_params(gp_freqs, yobs_mean, gp_george, nkpars, nwalkers,
                      nsamples, burn_frac, mpi, sample_kwargs={}):
    """Fit the parameters of the GP kernels.

    Parameters
    ----------
    gp_freqs : numpy.array
        The frequencies corresponding to the GWB data
    yobs_mean : numpy.array
        The mean of the GWB data
    gp_george : list[GaussProc]
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
    sampler = [0.0] * len(gp_freqs)
    ndim = nkpars
    utils.mpi_print(f"{mpi=}")
    pool = schwimmbad.choose_pool(
        mpi=mpi)  # processes=min(nwalkers // 2, cpu_count()) )

    # Schwimmbad docs are not clear if this needs to be here if we are passing the pool to
    # EnsembleSampler, but I've added it just in case.
    if mpi and not pool.is_master():
        pool.wait()
        sys.exit(0)

    for freq_ind in range(len(gp_freqs)):
        # Paralellize emcee with nwalkers //2 or the maximum number of processors available, whichever is smaller
        # with Pool(min(nwalkers // 2, cpu_count())) as pool:
        t_start = time.time()

        # Set up the sampler.
        sampler[freq_ind] = emcee.EnsembleSampler(nwalkers,
                                                  ndim,
                                                  gp_george[freq_ind].lnprob,
                                                  pool=pool)

        # Initialize the walkers.
        p0 = [
            np.random.uniform(gp_george[freq_ind].pmin[0],
                              gp_george[freq_ind].pmax[0], ndim)
            for _ in range(nwalkers)
        ]

        utils.mpi_print(freq_ind, "Running burn-in")
        p0, lnp, _ = sampler[freq_ind].run_mcmc(p0, int(burn_frac * nsamples), **sample_kwargs)
        sampler[freq_ind].reset()

        utils.mpi_print(freq_ind, "Running second burn-in")
        p = p0[np.argmax(lnp)]
        p0 = [p + 1e-8 * np.random.randn(ndim) for _ in range(nwalkers)]
        p0, _, _ = sampler[freq_ind].run_mcmc(p0, int(burn_frac * nsamples))
        sampler[freq_ind].reset()

        utils.mpi_print(freq_ind, "Running production")
        p0, _, _ = sampler[freq_ind].run_mcmc(p0, int(nsamples))

        utils.mpi_print(
            f"Completed {freq_ind} out of {len(gp_freqs)-1} in {(time.time() - t_start) / 60.0:.2f} min\n"
        )

    # Close the pool
    pool.close()
    # Populate the GP class with the details of the kernel
    # MAP values for each frequency.
    for ii in range(len(gp_freqs)):
        gp_george[ii].emcee_flatchain = sampler[ii].flatchain
        gp_george[ii].emcee_flatlnprob = sampler[ii].flatlnprobability

        gp_george[ii].emcee_kernel_map = sampler[ii].flatchain[np.argmax(
            sampler[ii].flatlnprobability)]

        # add-in mean yobs (freq) values
        gp_george[ii].mean_spectra = yobs_mean[ii]


def set_up_predictions(spectra, gp_george):
    """Set up a list of GPs ready for predictions.

    Parameters
    ----------
    spectra : h5py._hl.files.File
        The variable containing the library in HDF5 format
    gp_george : list[GaussProc]
        The GP model that has been read in from a .PKL file

    Returns
    -------
    gp_list : list[george.gp.GP]
        The configured GPs ready for predictions

    Examples
    --------
    FIXME: Add docs.

    """
    gp_list = []
    # gp_freqs = spectra["fobs"][:len(gp_george)].copy()
    # for ii in range(len(gp_freqs)):

    num_freqs = len(gp_george)
    for ii in range(num_freqs):
        gp = gp_george[ii]

        # Try to use the kernel attribute. If it doesn't exist, default to ExpSquaredKernel
        gp_list.append(george.GP(gp.create_kernel(gp.emcee_kernel_map)))

        gp_list[ii].compute(gp_george[ii].x, gp_george[ii].yerr)

    return gp_list


def mean_par_dict(gp_george):
    """Create a dictionary that is of the form parameter:mean(parameter_range).

    Parameters
    ----------
    gp_george : list[GaussProc]
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
        (gp_george[0].par_dict[key]["max"] + gp_george[0].par_dict[key]["min"])
        / 2
        for key in gp_george[0].par_dict.keys()
    }

    return mean_pars


def pars_linspace_dict(gp_george, num_points=5):
    """Create a dictionary that is of the form parameter:linspace(min(parameter), max(parameter), num=`num_points`).

    Parameters
    ----------
    gp_george : list[GaussProc]
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
            gp_george[0].par_dict[key]["min"],
            gp_george[0].par_dict[key]["max"],
            num=num_points,
        )
        for key in gp_george[0].par_dict.keys()
    }

    return pars_linspace


def hc_from_gp(gp_george, gp_list, gp_george_variance, gp_list_variance,
               env_pars,include_gp_unc=True):
    """Calculate the characteristic strain using a GP.

    Parameters
    ----------
    gp_george : list[GaussProc]
        The GP model that has been read in from a .PKL file
    gp_list : list[george.gp.GP]
        The configured GPs ready for predictions
    gp_george_variance : list[GaussProc]
        The variance GP model that has been read in from a .PKL file
    gp_list_variance : list[george.gp.GP]
        The configured variance GPs ready for predictions
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
    rho_pred = np.zeros((len(gp_george), 2))
    for ii, freq in enumerate(gp_george):
        # Get mean and variance predictions
        # Add uncertainties in quadrature, return total
        mean_pred, mean_pred_unc = gp_list[ii].predict(gp_george[ii].y, [env_pars])
        std_pred, std_pred_unc = gp_list_variance[ii].predict(gp_george_variance[ii].y, [env_pars])

        if include_gp_unc:
            total_pred_unc = np.sqrt(std_pred**2 + std_pred_unc**2 + mean_pred_unc**2)
        else:
            total_pred_unc = std_pred

        rho_pred[ii, 0], rho_pred[ii, 1] = mean_pred, total_pred_unc

        # transforming from zero-mean unit-variance variable to rho
        rho = (np.array(
            [gp_george[ii].mean_spectra
             for ii in range(len(gp_list))]) + rho_pred[:, 0])
    hc = np.sqrt(10**rho)
    return hc, rho, rho_pred


def sample_hc_from_gp(gp_george, gp_list, env_pars, nsamples=100):
    """Calculate the characteristic strain using a GP.

    Parameters
    ----------
    gp_george : list[GaussProc]
        The GP model that has been read in from a .PKL file
    gp_list : list[george.gp.GP]
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

    # I parallelized this computation over the frequencies
    # First, get iterable of arguments for the helper function
    args = [(gp_list[i], gp_george[i], env_pars, nsamples)
            for i in range(len(gp_list))]

    # Now, start a pool and map the helper function onto `args`
    with Pool(cpu_count() - 1) as pool:
        hc = np.array(pool.starmap(_sample_hc_from_gp_helper, args))

    # The multiprocessing routine returns hc in shape (freqs, samples), but it
    # makes more sense to have (samples, freqs). So, take the transpose
    return hc.T


def _sample_hc_from_gp_helper(gp_at_freq, gp_george_at_freq, env_pars,
                              nsamples):
    """Helper function for `sample_hc_from_gp()`.

    This function returns samples of the GP predicted characteristic strain for
    a given frequency. It is not meant to be called directly, but instead is
    called by `sample_hc_from_gp()` which uses it to parallelize this process
    over the frequencies of interest.

    Parameters
    ----------
    gp_at_freq : GaussProc
        The read-in GaussProc object at a given frequency
    gp_george_at_freq : george.gp.GP
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
    # This conditional block is meant to check with chain attribute is populated.
    # I originally made a mistake and used self.chain, when I really should have used
    # self.emcee_flatchain. This has been updated in newer versions.
    if getattr(gp_george_at_freq, "chain", None) is not None:
        chain_var = "chain"
    elif getattr(gp_george_at_freq, "emcee_flatchain", None) is not None:
        chain_var = "emcee_flatchain"
    else:
        utils.mpi_print("Chains are not saved!")

    # Get the samples
    samples = getattr(gp_george_at_freq, chain_var)

    hc = np.zeros(nsamples)

    for samp_ind, sample in enumerate(samples[np.random.randint(
            len(samples), size=nsamples)]):
        gp_at_freq.set_parameter_vector(sample)

        # transforming from zero-mean unit-variance variable to rho
        rho_sample = gp_at_freq.sample_conditional(gp_george_at_freq.y,
                                                   [env_pars])
        rho = gp_george_at_freq.mean_spectra + rho_sample

        hc[samp_ind] = np.sqrt(10**rho)

    return hc
