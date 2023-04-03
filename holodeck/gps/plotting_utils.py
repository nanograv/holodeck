"""Plotting utilities for Gaussian Processes."""
import sys
from multiprocessing import cpu_count, Pool
from pathlib import Path

import holodeck as holo
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as ssig
from holodeck.constants import GYR, MSOL, PC, YR

from holodeck.gps import gp_utils as gu

from labellines import labelLine, labelLines

FLOOR_STRAIN_SQUARED = 1e-40


def sam_hard(env_pars, shape=40):
    """SAM used in the `hard04b` library.

    Parameters
    ----------
    env_pars : dict
        Dictionary in format parameter:value
    shape : int
        Shape parameter for the SAM

    Returns
    -------
    sam : holodeck.sam.Semi_Analytic_Model
        The configured SAM
    hard : holodeck.hardening.Fixed_Time
        The configured hardening-rate model

    Examples
    --------
    FIXME: Add docs.

    """
    time = (10.0**env_pars["hard_time"]) * GYR
    rchar = (10.0**env_pars["hard_rchar"]) * PC
    mmb_amp = (10.0**env_pars["mmb_amp"]) * MSOL

    gsmf = holo.sam.GSMF_Schechter(phi0=env_pars["gsmf_phi0"])
    gpf = holo.sam.GPF_Power_Law()
    gmt = holo.sam.GMT_Power_Law()
    mmbulge = holo.relations.MMBulge_KH2013(mamp=mmb_amp)

    sam = holo.sam.Semi_Analytic_Model(gsmf=gsmf,
                                       gpf=gpf,
                                       gmt=gmt,
                                       mmbulge=mmbulge,
                                       shape=shape)

    hard = holo.hardening.Fixed_Time.from_sam(
        sam,
        time,
        rchar=rchar,
        gamma_sc=env_pars["hard_gamma_inner"],
        gamma_df=env_pars["hard_gamma_outer"],
        exact=True,
        progress=False,
    )

    return sam, hard


def get_smooth_center(env_pars,
                      model,
                      fobs_edges,
                      nreal=50,
                      sam_shape=40,
                      center_measure="median"):
    """Calculate `nreal` GWB realizations and get the smoothed center (mean or median).

    Parameters
    ----------
    env_pars : dict
        Dictionary of the form parameter:value
    model : function
        The SAM to use, for example see `sam_hard()` in this file
    fobs_edges: numpy.array
        The array of frequency bin edges to use
    nreal : int
        The number of GWB realizations
    sam_shape : int, optional
        The shape of the SAM grid
    center_measure: str, optional
        The measure of center to use when returning a zero-center data. Can be
        either "mean" or "median"


    Returns
    -------
    numpy.array
        The smoothed mean of the GWB realizations

    Examples
    --------
    FIXME: Add docs.


    """
    sam, hard = model(env_pars, sam_shape)
    gwb = sam.gwb(fobs_edges, realize=nreal, hard=hard)

    # Take care of zeros
    low_ind = np.where(gwb < 1e-40)
    gwb[low_ind] = 1e-40

    # Find mean or median over realizations
    if center_measure.lower() == "median":
        center = np.log10(np.median(gwb, axis=-1))
    elif center_measure.lower() == "mean":
        center = np.log10(np.mean(gwb, axis=-1))
    else:
        raise ValueError(
            f"`center_measure` must be 'mean' or 'median', not '{center_measure}'"
        )
    #print(f"{center=}\n{center.shape=}")
    return ssig.savgol_filter(center, 7, 3)


def plot_individual_parameter(gp_george,
                              gp_list,
                              pars_const,
                              par_interest,
                              spectra,
                              num_points=5,
                              center_measure="median",
                              plot=True,
                              plot_dir=Path.cwd(),
                              find_sam_mean=True,
                              model=sam_hard,
                              sam_shape=40,
                              nreal=50,
                              color_map=plt.cm.Dark2,
                              return_results=False):
    """Plot GWBs while varying a single parameter.

    Parameters
    ----------
    gp_george : list[GaussProc]
        The GP model that has been read in from a .PKL file
    gp_list : list[george.gp.GP]
        The configured GPs ready for predictions
    pars_const : dict
        Dictionary of constant parameter values
    par_interest : str
        The parameter to vary
    spectra : h5py._hl.files.File
        The variable containing the library in HDF5 format
    num_points : int
        Number of evenly-spaced samples in the linspace(min(`par_interest`),
        max(`par_interest`))
    center_measure: str, optional
        The measure of center to use when returning a zero-center data. Can be
        either "mean" or "median"
    plot : bool, optional
        Make or supress plots
    plot_dir : str or Path, optional
        The directory to save plots in
    find_sam_mean : bool, optional
        Whether to calculate the GWB and find the smoothed mean
    model : function, optional
        The function which describes the SAM to use
    sam_shape : int, optional
        The shape of the SAM grid
    nreal : int, optional
        Number of GWB realizations
    color_map : matplotlib.pyplot.cm, optional
        The color map to use for the plots
    return_results: bool
        Whether to return the numerical results used for the plots

    Returns
    -------
    hc : numpy.array
        The array of characteristic strains
    rho : numpy.array
        Array of rho values
    rho_pred : numpy.array
        Array of rho_pred values
    smooth_mean : numpy.array
        Array of smoothed GWBs

    Examples
    --------
    FIXME: Add docs.

    """
    # Ensure Path object
    if type(plot_dir) == str:
        plot_dir = Path(plot_dir)

    if not plot_dir.is_dir():
        sys.exit(
            f"{plot_dir.absolute()} does not exist. Please create it first.")

    colors = color_map(np.linspace(0, 1, num=num_points))

    # Get frequencies used for GP training
    gp_freqs = spectra["fobs"][:len(gp_george)].copy()

    # Get linspace dict for parameters
    pars_linspace = gu.pars_linspace_dict(gp_george, num_points=num_points)

    hc = np.zeros((len(gp_freqs), num_points))
    rho = np.zeros((len(gp_freqs), num_points))
    rho_pred = np.zeros((len(gp_freqs), 2, num_points))
    smooth_center = np.zeros((len(gp_freqs), num_points))

    env_pars_list = []
    for i, par_varied in enumerate(pars_linspace[par_interest]):

        # create a dict where the parameter of interest takes a value from the `pars_linspace`,
        # and every other parameter takes its value from `par_const`.
        # This way, we iterate over the parameter of interest.
        env_pars = {
            par: (par_varied if par == par_interest else pars_const[par])
            for par in pars_const.keys()
        }

        env_pars_list.append(env_pars)

        # Get hc from GP
        hc[:, i], rho[:,
                      i], rho_pred[:, :,
                                   i] = gu.hc_from_gp(gp_george, gp_list,
                                                      list(env_pars.values()))

    # Get smoothed mean of GWB if using SAM
    if find_sam_mean:
        fobs_edges = spectra["fobs_edges"][:len(gp_freqs) + 1]
        args = [(env_pars_list[i], model, fobs_edges, nreal, sam_shape,
                 center_measure)
                for i, _ in enumerate(pars_linspace[par_interest])]

        with Pool(cpu_count() - 1) as pool:
            smooth_center = np.array(pool.starmap(get_smooth_center, args)).T

    # Make plot
    if plot:
        if find_sam_mean:
            # the smoothed mean
            for j in range(num_points):
                plt.loglog(
                    gp_freqs,
                    10**smooth_center[:, j],
                    color=colors[j],
                    lw=1,
                    linestyle="dashed",
                )

        for j in range(num_points):
            plt.semilogx(
                gp_freqs,
                hc[:, j],
                lw=1,
                label=f"{par_interest} = {pars_linspace[par_interest][j]:.2f}",
                c=colors[j],
                alpha=1,
            )

            plt.fill_between(
                gp_freqs,
                np.sqrt(10**(rho[:, j] + rho_pred[:, 1, j])),
                np.sqrt(10**(rho[:, j] - rho_pred[:, 1, j])),
                color=colors[j],
                alpha=0.25,
            )

        plt.xlabel("Observed GW Frequency [Hz]")
        plt.ylabel(r"$h_{c} (f)$")
        plt.yscale("log")
        plt.xlim(gp_freqs.min(), gp_freqs.max())
        plt.legend(loc=3)
        fname = plot_dir / f"param_varied_{par_interest}.png"
        plt.savefig(fname)
        print(f"Plot saved at {fname.absolute()}")

    if return_results:
        return hc, rho, rho_pred, smooth_center

def pub_plot_individual_parameter(gp_george,
                              gp_list,
                              pars_const,
                              par_interest,
                              spectra,
                              num_points=5,
                              center_measure="median",
                              plot=True,
                              plot_dir=Path.cwd(),
                              find_sam_mean=True,
                              model=sam_hard,
                              sam_shape=40,
                              nreal=50,
                              color_map=plt.cm.Dark2,
                              multiprocessing=False,
                              return_results=False):
    """Plot GWBs while varying a single parameter.

    Parameters
    ----------
    gp_george : list[GaussProc]
        The GP model that has been read in from a .PKL file
    gp_list : list[george.gp.GP]
        The configured GPs ready for predictions
    pars_const : dict
        Dictionary of constant parameter values
    par_interest : str
        The parameter to vary
    spectra : h5py._hl.files.File
        The variable containing the library in HDF5 format
    num_points : int
        Number of evenly-spaced samples in the linspace(min(`par_interest`),
        max(`par_interest`))
    center_measure: str, optional
        The measure of center to use when returning a zero-center data. Can be
        either "mean" or "median"
    plot : bool, optional
        Make or supress plots
    plot_dir : str or Path, optional
        The directory to save plots in
    find_sam_mean : bool, optional
        Whether to calculate the GWB and find the smoothed mean
    model : function, optional
        The function which describes the SAM to use
    sam_shape : int, optional
        The shape of the SAM grid
    nreal : int, optional
        Number of GWB realizations
    color_map : matplotlib.pyplot.cm, optional
        The color map to use for the plots
    return_results: bool
        Whether to return the numerical results used for the plots

    Returns
    -------
    hc : numpy.array
        The array of characteristic strains
    rho : numpy.array
        Array of rho values
    rho_pred : numpy.array
        Array of rho_pred values
    smooth_mean : numpy.array
        Array of smoothed GWBs

    Examples
    --------
    FIXME: Add docs.

    """
    # Ensure Path object
    if type(plot_dir) == str:
        plot_dir = Path(plot_dir)

    if not plot_dir.is_dir():
        sys.exit(
            f"{plot_dir.absolute()} does not exist. Please create it first.")

    colors = color_map(np.linspace(0, 1, num=num_points))

    # Get frequencies used for GP training
    gp_freqs = spectra["fobs"][:len(gp_george)].copy()

    # Get linspace dict for parameters
    pars_linspace = gu.pars_linspace_dict(gp_george, num_points=num_points)

    hc = np.zeros((len(gp_freqs), num_points))
    rho = np.zeros((len(gp_freqs), num_points))
    rho_pred = np.zeros((len(gp_freqs), 2, num_points))
    smooth_center = np.zeros((len(gp_freqs), num_points))

    env_pars_list = []
    for i, par_varied in enumerate(pars_linspace[par_interest]):

        # create a dict where the parameter of interest takes a value from the `pars_linspace`,
        # and every other parameter takes its value from `par_const`.
        # This way, we iterate over the parameter of interest.
        env_pars = {
            par: (par_varied if par == par_interest else pars_const[par])
            for par in pars_const.keys()
        }

        env_pars_list.append(env_pars)

        # Get hc from GP
        hc[:, i], rho[:,
                      i], rho_pred[:, :,
                                   i] = gu.hc_from_gp(gp_george, gp_list,
                                                      list(env_pars.values()))

    # Get smoothed mean of GWB if using SAM
    if find_sam_mean:
        fobs_edges = spectra["fobs_edges"][:len(gp_freqs) + 1]
        if multiprocessing:
            args = [(env_pars_list[i], model, fobs_edges, nreal, sam_shape,
                     center_measure)
                    for i, _ in enumerate(pars_linspace[par_interest])]

            with Pool(cpu_count() - 1) as pool:
                smooth_center = np.array(pool.starmap(get_smooth_center, args)).T
        else:
            smooth_center = []
            for i, _ in enumerate(pars_linspace[par_interest]):
                print(f"{i=} {env_pars_list[i]=} {model=} {fobs_edges=} {nreal=} {sam_shape=} {center_measure=}")
                smooth_center.append(get_smooth_center(env_pars_list[i], model, fobs_edges, nreal, sam_shape, center_measure))
            smooth_center = np.array(smooth_center)


    # Make plot
    if plot:
        if find_sam_mean:
            # the smoothed mean
            for j in range(num_points):
                plt.loglog(
                    gp_freqs * YR,
                    10**smooth_center[:, j],
                    color=colors[j],
                    lw=1,
                    linestyle="dashed",
                )

        for j in range(num_points):
            plt.semilogx(
                gp_freqs * YR,
                hc[:, j],
                lw=1,
                label=f"${pars_linspace[par_interest][j]:.2f}$",
                c=colors[j],
                alpha=1,
            )

            plt.fill_between(
                gp_freqs * YR,
                np.sqrt(10**(rho[:, j] + rho_pred[:, 1, j])),
                np.sqrt(10**(rho[:, j] - rho_pred[:, 1, j])),
                color=colors[j],
                alpha=0.25,
            )
        
        if par_interest == 'hard_time':
            xvals = np.array([7.0e-2, 7.0e-2, 1.25e-1, 2.5e-1, 2.5e-1])
            labelLines(plt.gca().get_lines(), xvals=xvals, zorder=2.5)
        else:
            labelLines(plt.gca().get_lines(), zorder=2.5)
        plt.xlabel("Observed GW Frequency [1/yr]")
        plt.ylabel(r"$h_{c} (f)$")
        plt.yscale("log")
        plt.xlim(3.0e-2, 3.0e0)
        plt.ylim(5.0e-17, 5e-14)
        plt.title(f"{par_interest}")
        #plt.legend(loc=3)
        fname = plot_dir / f"param_varied_{par_interest}.png"
        plt.savefig(fname)
        print(f"Plot saved at {fname.absolute()}")

    if return_results:
        return hc, rho, rho_pred, smooth_center


def plot_parameter_variances(
        gp_george,
        gp_list,
        pars_const,
        spectra,
        color_map=plt.cm.Dark2,
        alpha=0.65,
        plot_dir=Path.cwd(),
):
    """Plot the variance in the GWB with each parameter allowed to vary while the others are held constant.

    Parameters
    ----------
    gp_george : list[GaussProc]
        The GP model that has been read in from a .PKL file
    gp_list : list[george.gp.GP]
        The configured GPs ready for predictions
    pars_const : dict
        Dictionary of constant parameter values
    spectra : h5py._hl.files.File
        The variable containing the library in HDF5 format
    color_map : matplotlib.pyplot.cm, optional
        The color map to use for the plots
    alpha : float, optional
        The transparency value for the shaded regions
    plot_dir : str or Path, optional
        The directory to save plots in

    Examples
    --------
    FIXME: Add docs.


    """
    # Check if Path object
    if type(plot_dir) == str:
        plot_dir = Path(plot_dir)

    if not plot_dir.is_dir():
        sys.exit(
            f"{plot_dir.absolute()} does not exist. Please create it first.")

    gp_freqs = spectra["fobs"][:len(gp_george)].copy()

    colors = color_map(np.linspace(0, 1, num=len(pars_const)))

    for i, par_interest in enumerate(pars_const):
        result = plot_individual_parameter(gp_george,
                                           gp_list,
                                           pars_const,
                                           par_interest,
                                           spectra,
                                           plot=False,
                                           find_sam_mean=False,
                                           return_results=True)

        hc = result[0]

        plt.fill_between(
            gp_freqs,
            hc[:, 0],
            hc[:, -1],
            lw=0,
            label=f"{par_interest}",
            color=colors[i],
            alpha=alpha,
            zorder=((np.min(hc[0, :]) - np.max(hc[0, :])) * 1e16),
        )

    plt.xlabel("Observed GW Frequency [Hz]")
    plt.ylabel(r"$h_{c} (f)$")
    plt.yscale("log")
    plt.xscale("log")
    #plt.ylim(1e-16, 1e-13)
    plt.xlim(gp_freqs.min(), gp_freqs.max())
    plt.legend(loc=1)
    fname = plot_dir / "params_varied.png"
    plt.savefig(fname)
    print(f"Plot saved at {fname.absolute()}")


def plot_over_realizations(ind,
                           spectra,
                           gp_george,
                           gp_list,
                           center_measure="median",
                           plot_dir=Path.cwd(),
                           test_frac=0.0):
    """Plot the GP prediction over the GWB from `spectra`.

    Parameters
    ----------
    ind : int
        The index of parameter combinations to use. Max is
        `spectra['gwb'].shape[0]`
    spectra : h5py._hl.files.File
        The variable containing the library in HDF5 format
    gp_george : list[GaussProc]
        The GP model that has been read in from a .PKL file
    gp_list : list[george.gp.GP]
        The configured GPs ready for predictions
    center_measure: str, optional
        The measure of center to use when returning a zero-center data. Can be
        either "mean" or "median"
    plot_dir : str or Path, optional
        The directory to save plots in


    Examples
    --------
    FIXME: Add docs.

    """
    # Ensure Path object
    if type(plot_dir) == str:
        plot_dir = Path(plot_dir)

    if not plot_dir.is_dir():
        sys.exit(
            f"{plot_dir.absolute()} does not exist. Please create it first.")

    # Test frac is purposefully left out here so you can make plots using the
    # test set
    freqs, xobs, yerr, yobs, yobs_mean = gu.get_smoothed_gwb(
        spectra, len(gp_george), center_measure=center_measure)

    # Alert if in test set
    if ind < int(yobs.shape[0] * test_frac):
        print(f"Index {ind} is in the test set")

    smooth_center = yobs + yobs_mean

    # Take the test set offset into account for xobs
    env_param = xobs[ind - int(yobs.shape[0] * test_frac), :].copy()

    hc, rho, rho_pred = gu.hc_from_gp(gp_george, gp_list, env_param)

    # Convert to Hz
    freqs /= YR


    gwb_spectra = spectra['gwb']

    # Need to drop NaNs
    bads = np.any(np.isnan(gwb_spectra), axis=(1, 2))
    gwb_spectra = gwb_spectra[~bads]

    # Find all the zeros and set them to be h_c = 1e-20
    low_ind = (gwb_spectra < np.sqrt(FLOOR_STRAIN_SQUARED))
    gwb_spectra[low_ind] = np.sqrt(FLOOR_STRAIN_SQUARED)

    # the raw spectra
    # Let the last one be the one we apply a label to
    for ii in range(gwb_spectra.shape[-1] - 1):
        plt.loglog(freqs,
                   gwb_spectra[ind, :len(gp_george), ii],
                   color='C0',
                   alpha=0.2,
                   zorder=0)
    # Plot the last and add a label
    plt.loglog(freqs,
               gwb_spectra[ind, :len(gp_george), ii+1],
               color='C0',
               alpha=0.2,
               zorder=0,
               label='Original Spectra')

    # the smoothed mean
    plt.loglog(freqs,
               np.sqrt(10**smooth_center[ind][:len(gp_george)]),
               color='C1',
               label=f"Smoothed {center_measure}",
               lw=2)

    # the GP realization
    plt.semilogx(freqs, hc, color='C3', lw=2.5, label='GP')
    plt.fill_between(freqs,
                     np.sqrt(10**(rho + rho_pred[:, 1])),
                     np.sqrt(10**(rho - rho_pred[:, 1])),
                     color='C3',
                     alpha=0.5)

    plt.xlabel('Observed GW Frequency [Hz]')
    plt.ylabel(r'$h_{c} (f)$')

    plt.legend(loc=3)
    fname = plot_dir / "gp_overplotted.png"
    plt.savefig(fname)
    print(f"Plot saved at {fname.absolute()}")

    # Print the parameter values for this gwb
    pars = list(spectra.attrs["param_names"].astype(str))
    for i, par in enumerate(env_param):
        print(f"{pars[i]} = {env_param[i]:.2E}")

    return env_param
