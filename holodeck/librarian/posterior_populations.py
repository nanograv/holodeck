"""Script to generate Holodeck populations, drawing from the 15yr analysis constraints.

See `holodeck-pops.ipynb` for example usage of this data.

Usage (posterior_populations.py)
--------------------------------
See `python gen_holodeck_pop.py -h` for usage information.

Example::

    python gen_holodeck_pops.py -t 20 -f 30 -r 100 -l 10 -m
                                   |     |     |      |   |--> use maxmimum-likelihood values
                                   |     |     |      |------> 10 loudest binaries in each frequency bin
                                   |     |     |-------------> 100 realizations of populations
                                   |     |-------------------> 30 frequency bins
                                   |-------------------------> 20 years observing baseline = 1/(20yr) lowest frequency

To-Do (posterior_populations.py)
--------------------------------
* Improve handling of data path.
* Improve handling/specification of parameter space.
    * Allow changes to be passed in through API and or CL
    * Make each particular 15yr dataset specify its own parameter space (these need to match up anyway!)

"""

import argparse
from pathlib import Path
import numpy as np

import holodeck as holo
import holodeck.librarian
from holodeck.constants import YR

# Choose range of orbital periods of interest
TDUR = 16.0   # yr
NFREQS = 60
NREALS = 103
### NDRAWS = 101    # NOT IMPLEMENTED YET
NLOUDEST = 10

# Path to chains, fitting holodeck populations to data, giving parameter posteriors
# This is the `15yr_astro_data` currently stored on google drive:
# https://drive.google.com/drive/u/1/folders/1wFy_go_l8pznO9D-a2i2wFHe06xuOe5B
PATH_DATA = Path(
    "/Users/lzkelley/Programs/nanograv/15yr_astro_data/"
    "phenom/ceffyl_chains/astroprior_hdall/"
)

# Parameter space corresponding to the fit data
PSPACE = holo.librarian.param_spaces_classic.PS_Classic_Phenom_Uniform

# Path to save output data
PATH_OUTPUT = Path(holo._PATH_OUTPUT).resolve().joinpath("15yr_pops")


def main(args=None):
    """Top level function that does all the work.
    """

    # ---- Setup / Initialization

    # load arguments
    if args is None:
        args = setup_argparse()

    # load chains (i.e. parameter posterior distributions)
    # chains = load_chains(PATH_DATA)

    # select parameters for this population
    if args.maxlike:
        pkey = "ML"
        # pars = get_maxlike_pars_from_chains(chains)
        pars = get_maxlike_pars_from_chains()
    else:
        pkey = "draw"
        # pars = sample_pars_from_chains(chains)
        pars = sample_pars_from_chains()

    # construct output filename
    output = Path(args.output).resolve()
    output.mkdir(exist_ok=True)
    # Find a unique filename, in case other populations have already been created
    ml_warning = False
    for num in range(10000):
        fname = f"t{args.tdur:.1f}yr_nf{args.nfreqs}_nr{args.nreals}_nl{args.nloudest}_{pkey}_{num:04d}.npz"
        fname = output.joinpath(fname)
        if not fname.exists():
            break
        if (num > 0) and args.maxlike and (ml_warning is False):
            err = "Maximum likelihood population with these paramters already exists!  {fname}"
            holo.log.error(err)
            raise RuntimeError(err)

    else:
        raise RuntimeError(f"Could not find a filename that doesn't exist!  e.g. {fname}")

    # ---- Construct population and derived properties

    # Build populations with holodeck
    data, classes = load_population_for_pars(
        pars, pta_dur=args.tdur*YR, nfreqs=args.nfreqs, nreals=args.nreals, nloudest=args.nloudest,
    )

    # ---- Save to output file

    # Save data
    np.savez(
        fname,
        pspace_name=PSPACE.__name__,
        data_path=PATH_DATA.resolve(),
        **data
    )
    print(f"Saved size {holo.utils.get_file_size(fname)} : {fname}")

    return


def setup_argparse(*args, **kwargs):
    """Setup parameters/arguments.

    Note that this can be used to set parameters NOT from command-line usage,
    but in this case the `args` argument must be set to empty.  For example:

    This will load of all the default arguments (NOTE the empty string argument is typically needed):
        ``args = gen_holodeck_pops.setup_argparse("")``

    This will set the desired parameters, and otherwise load the defaults:
        ``args = gen_holodeck_pops.setup_argparse("", nloudest=12, nreals=6, maxlike=True)``

    """

    # Setup ArgumentParser and parameters / arguments

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-o', '--output', default=PATH_OUTPUT, dest='output', type=str,
        help="Directory to place output files (created if it doesn't already exist)."
    )
    parser.add_argument(
        '-t', '--dur', default=TDUR, dest='tdur', type=float,
        help='Observational duration in years.  Determines frequency bins.'
    )
    parser.add_argument(
        '-f', '--nfreqs', default=NFREQS, dest='nfreqs', type=int,
        help='Number of frequency bins.'
    )
    parser.add_argument(
        '-r', '--nreals', default=NREALS, dest='nreals', type=int,
        help='Number of realizations.'
    )
    parser.add_argument(
        '-l', '--nloudest', default=NLOUDEST, dest='nloudest', type=int,
        help='Number of loudest binaries in each frequency bin.'
    )
    parser.add_argument(
        '-m', '--maxlike', action='store_true', default=False, dest='maxlike',
        help='Use the maximum likelihood parameters, instead of drawing from the posteriors.'
    )

    # Set custom parameters passed through the `kwargs` argument (i.e. if this function is called from another module)
    args = parser.parse_args(*args)
    for kk, vv in kwargs.items():
        if not hasattr(args, kk):
            raise KeyError(f"Argparse {args} has no attribute {kk}!")
        setattr(args, kk, vv)

    return args


def load_population_for_pars(pars, pta_dur=TDUR, nfreqs=NFREQS, nreals=NREALS, nloudest=NLOUDEST):
    """Construct a holodeck population.

    Arguments
    ---------
    pars : dict
        Binary population parameters for the appropriate parameter space `PSPACE`.
        Typically the `pars` should be loaded using either the `sample_pars_from_chains` or the
        `get_maxlike_pars_from_chains` function.
    pta_dur : scalar [seconds]
        Duration of PTA observations, used to determine Fourier frequency bins.
        Bin centers are at frequencies ``f_i = (i+1) / pta_dur``
    nfreqs : int
        Number of frequency bins.
    nreals : int
        Number of realizations to construct.
    nloudest : int
        Number of loudest binaries to calculate, per frequency bin.

    Returns
    -------
    data : dict
        Binary population and derived properties.  Entries:

        * `number` : ndarray (M, Q, Z, F)
          Number of binaries in the Universe in each bin.
          The bins are total mass (M), mass ratio (Q), redshift (Z), and frequency (F).
        * `hc_ss` : ndarray (F, R, L)
          GW characteristic strain of the loudest L binaries in each frequency bin (F) and realization (R).
          The GW frequencies are assumed to be 2x the orbital frequencies (i.e. circular orbits).
        * `hc_bg` : ndarray (F, R)
          GW characteristic strain of all binaries besides the L loudest in each frequency bin,
          for frequency bins `F` and realizations `R`.
          The GW frequencies are assumed to be 2x the orbital frequencies (i.e. circular orbits).
        * `sspar` : ndarray (P, F, R, L)
          Binary parameters of the loudest `L` binaries in each frequency bin `F` for realizations `R`.
          The P=4 parameters included are {total mass [grams], mass ratio, initial redshift, final redshift},
          where initial redshift is at the time of galaxy merger, and final redshift is when reaching the frequency bin.
        * `mtot_edges` : ndarray (M+1,)
          The edges of the total-mass dimension of the SAM grid, in units of [grams].
          Note that there are `M+1` bin edges for `M` bins.
        * `mrat_edges` : ndarray (Q+1,)
          The edges of the mass-ratio dimension of the SAM grid.
          Note that there are `Q+1` bin edges for `Q` bins.
        * `redz_edges` : ndarray (Z+1,)
          The edges of the redshfit dimension of the SAM grid.
          Note that there are `Z+1` bin edges for `Z` bins.
        * `fobs_orb_edges` : ndarray (F+1,)
          The edges of the orbital-frequency dimension of the SAM grid.
          Note that there are `F+1` bin edges for `F` bins.

    """

    # Choose the appropriate Parameter Space (from 15yr astro analysis)
    # Load SAM and hardening model for desired parameters
    sam, hard = PSPACE.model_for_params(pars)

    fobs_orb_cents, fobs_orb_edges = holo.utils.pta_freqs(pta_dur, nfreqs)

    # calculate (differential) number of binaries
    redz_final, diff_num = holo.sams.sam_cyutils.dynamic_binary_number_at_fobs(
        fobs_orb_cents, sam, hard, holo.cosmo
    )
    # integrate to find total number of binaries in each bin
    edges = [sam.mtot, sam.mrat, sam.redz, fobs_orb_edges]
    number = holo.sams.sam_cyutils.integrate_differential_number_3dx1d(edges, diff_num)
    # print(f"Loaded {number.sum():.1e} binaries across frequency range")

    vals = holo.single_sources.ss_gws_redz(
        edges, redz_final, number,
        realize=nreals, loudest=nloudest, params=True,
    )

    # `sspar` parameters are (total mass, mass ratio, initial redshift, final redshift)
    hc_ss, hc_bg, sspar, bgpar = vals

    data = dict(
        number=number, hc_ss=hc_ss, hc_bg=hc_bg, sspar=sspar,
        mtot_edges=edges[0], mrat_edges=edges[1], redz_edges=edges[2], fobs_orb_edges=edges[3],
    )

    classes = dict(
        sam=sam,
        hard=hard,
    )

    return data, classes


def load_chains(path_data):
    """Load the MCMC chains from the given path.

    The path must contain the expected files resulting from fitting with `ceffyl`.

    Arguments
    ---------
    path_data : `str` or `pathlib.Path`
        Path to directory containing the `pars.txt` and `chain_1.0.txt` files.

    Returns
    -------
    data : dict
        The values at each step of the MCMC chains for each parameters.
        For example, the parameters may be::

            ['hard_time', 'gsmf_phi0', 'gsmf_mchar0_log10',
            'mmb_mamp_log10', 'mmb_scatter_dex', 'hard_gamma_inner']

        in which case each of these will be an entry in the dictionary, where the values are an
        array of the steps in each of these parameters.

    """
    path_data = Path(path_data)
    fname_pars = path_data.joinpath("pars.txt")
    fname_chains = path_data.joinpath("chain_1.0.txt")

    assert path_data.is_dir(), f"Path to chains '{path_data}' does not exist!"
    assert fname_chains.is_file(), f"Could not find chain file '{fname_chains}'!"
    assert fname_pars.is_file(), f"Could not find chain parameters file '{fname_pars}'!"

    # load the names of each parameter
    chain_pars = np.loadtxt(fname_pars, dtype=str)
    # load the chains themselves
    chains = np.loadtxt(fname_chains)
    # combine names and chains into dictionary
    data = {name: vals for name, vals in zip(chain_pars, chains.T)}
    return data


def sample_pars_from_chains(chains=None):
    """Sample randomly from the given chains (i.e. parameter posteriors).

    Arguments
    ---------
    chains : dict
        The MCMC parameter values for each of the parameters in this holodeck parameter-space.
        These chains should typically be loaded using the `load_chains` function.

    Returns
    -------
    pars : dict
        Randomly selected parameters drawn from the `chains`.
        This will be a single float value for each of the parameters in the holodeck parameter-space,
        for example::

            ['hard_time', 'gsmf_phi0', 'gsmf_mchar0_log10',
            'mmb_mamp_log10', 'mmb_scatter_dex', 'hard_gamma_inner'],

    """
    if chains is None:
        chains = load_chains(PATH_DATA)

    nlinks = list(chains.values())[0].size
    idx = np.random.choice(nlinks)
    pars = {key: value[idx] for key, value in chains.items()}
    return pars


def get_maxlike_pars_from_chains(chains=None):
    """Load the maximum-likelihood (ML) parameters from the given chains (i.e. parameter posteriors).

    KDEs from `kalepy` are used to construct the ML parameters.

    Arguments
    ---------
    chains : dict
        The MCMC parameter values for each of the parameters in this holodeck parameter-space.
        These chains should typically be loaded using the `load_chains` function.

    Returns
    -------
    pars : dict
        Maximum likelihood parameters drawn from the `chains`.
        This will be a single float value for each of the parameters in the holodeck parameter-space,
        for example::

            ['hard_time', 'gsmf_phi0', 'gsmf_mchar0_log10',
            'mmb_mamp_log10', 'mmb_scatter_dex', 'hard_gamma_inner']

    """
    import kalepy as kale
    if chains is None:
        chains = load_chains(PATH_DATA)

    # Get maximum likelihood parameters (estimate using KDE)
    mlpars = {}
    for name, vals in chains.items():
        extr = holo.utils.minmax(vals)
        xx, yy = kale.density(vals, reflect=extr)
        idx = np.argmax(yy)
        xmax = xx[idx]
        mlpars[name] = xmax

    return mlpars


# ---- Run the `main` function when this file is called as a script

if __name__ == "__main__":
    main()
