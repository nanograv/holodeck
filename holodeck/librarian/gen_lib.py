"""Library generation interface.

This file can be run from the command-line to generate holodeck libraries, and also provides some
API methods for quick/easy generation of simulations.  In general, these methods are designed to
run simulations for populations constructed from parameter-spaces (i.e.
:class:`~holodeck.librarian.lib_tools._Param_Space` subclasses).  This script is parallelized using
``mpi4py``, but can also be run in serial.

This script can be run by executing::

    python -m holodeck.librarian.gen_lib <ARGS>

Run ``python -m holodeck.librarian.gen_lib -h`` for usage information.

This script is also aliased to the console command, ``holodeck_lib_gen``.

See https://holodeck-gw.readthedocs.io/en/main/getting_started/libraries.html for more information
about generating holodeck libraries and using holodeck parameter-spaces in general.

For an example job-submission script using slurm, see the ``scripts/run_holodeck_lib_gen.sh`` file.

"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
import json
import shutil

import numpy as np

import holodeck as holo
from holodeck.constants import YR
import holodeck.librarian
import holodeck.librarian.combine
from holodeck.librarian import (
    lib_tools, ARGS_CONFIG_FNAME, PSPACE_DOMAIN_EXTREMA, DIRNAME_LIBRARY_SIMS, DIRNAME_DOMAIN_SIMS
)

#! DOPPLER
import holodeck.doppler
DOPPLER_FNAME = holo.get_holodeck_path("data", "doppler-sens-fit_2024-05-01.csv")
#! -------

#: maximum number of failed simulations before task terminates with error (`None`: no limit)
MAX_FAILURES = None

# FILES_COPY_TO_OUTPUT = [__file__, holo.librarian.__file__, holo.param_spaces.__file__]
FILES_COPY_TO_OUTPUT = []


comm = None


def main():   # noqa : ignore complexity warning
    """Parent method for generating libraries from the command-line.

    This function requires ``mpi4py`` for parallelization.

    This method does the following:

    (1) Loads arguments from the command-line (``args``, via :func:`_setup_argparse()`).

        (a) The ``output`` directory is created as needed, along with the ``sims/`` and ``logs/``
            subdirectories in which the simulation datafiles and log output files are saved.

    (2) Sets up a ``logging.Logger`` instance for each processor.
    (3) Constructs the parameter space specified by ``args.param_space``.  If this run is being
        resumed, then the param-space is loaded from an existing save file in the output directory.
    (4) Samples from the parameter space are allocated to all processors.
    (5) Each processor iterates over it's allocated parameters, calculates populations, saves them
        to files in the output directories.  This is handled in :func:`run_sam_at_pspace_num()`.
    (6) All of the individual simulation files are combined using
        :func:`holodeck.librarian.combine.sam_lib_combine()`.

    """

    # ---- load mpi4py module

    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
    except ModuleNotFoundError as err:
        comm = None
        holo.log.error(f"failed to load `mpi4py` in {__file__}: {err}")
        holo.log.error("`mpi4py` may not be included in the standard `requirements.txt` file.")
        holo.log.error("Check if you have `mpi4py` installed, and if not, please install it.")
        raise err

    # ---- setup arguments / settings, loggers, and outputs

    if comm.rank == 0:
        args = _setup_argparse()
    else:
        args = None

    # share `args` to all processes from rank=0
    args = comm.bcast(args, root=0)

    # setup log instance, separate for all processes
    log = _setup_log(comm, args)
    args.log = log

    if comm.rank == 0:

        # get parameter-space class
        space = _setup_param_space(args)

        # copy certain files to output directory
        copy_files = FILES_COPY_TO_OUTPUT
        if (not args.resume) and (copy_files is not None):
            for fname in copy_files:
                src_file = Path(fname)
                dst_file = args.output.joinpath("runtime_" + src_file.name)
                shutil.copyfile(src_file, dst_file)
                log.info(f"Copied {fname} to {dst_file}")

        # Load arguments/configuration from previous save
        if args.resume:
            args, config_fname = load_config_from_path(args.output, log)
            log.warning(f"Loaded configuration save from {config_fname}")
            args.resume = True
        # Save parameter space and args/configuration to output directory
        else:
            space_fname = space.save(args.output)
            log.info(f"saved parameter space {space} to {space_fname}")

            config_fname = _save_config(args)
            log.info(f"saved configuration to {config_fname}")

        # ---- Split simulations for all processes

        # Domain: Vary only one parameter at a time to explore the domain
        if args.domain:
            log.info("Constructing domain-exploration indices")
            # we will have `nsamples` in each of `nparameters`
            domain_shape = (space.nparameters, args.nsamples)
            indices = np.arange(np.product(domain_shape))
            # indices = np.random.permutation(indices)
            indices = np.array_split(indices, comm.size)
        # Standard Library: vary all parameters together
        else:
            log.info("Constructing library indices")
            indices = range(args.nsamples)
            indices = np.random.permutation(indices)
            indices = np.array_split(indices, comm.size)
            domain_shape = None

        num_per = [len(ii) for ii in indices]
        log.info(f"{args.nsamples=} cores={comm.size} || max sims per core = {np.max(num_per)}")

    else:
        space = None
        indices = None
        domain_shape = None

    # share parameter space across processes
    space = comm.bcast(space, root=0)
    domain_shape = comm.bcast(domain_shape, root=0)

    # If we've loaded a new `args`, then share to all processes from rank=0
    if args.resume:
        args = comm.bcast(args, root=0)
        args.log = log

    log.info(
        f"param_space={args.param_space}, parameters={space.nparameters}, samples={args.nsamples}\n"
        f"sam_shape={args.sam_shape}, nreals={args.nreals}\n"
        f"nfreqs={args.nfreqs}, pta_dur={args.pta_dur} [yr]\n"
    )

    # ---- distribute jobs to processors

    indices = comm.scatter(indices, root=0)
    iterator = holo.utils.tqdm(indices) if (comm.rank == 0) else np.atleast_1d(indices)

    comm.barrier()

    # ---- iterate over each processors' jobs

    beg = datetime.now()
    log.info(f"beginning tasks at {beg}")
    failures = 0
    num_done = 0
    for sim_num in iterator:
        log.info(f"{comm.rank=} {sim_num=}")

        # Domain: Vary only one parameter at a time to explore the domain
        if args.domain:
            param_num, samp_num = np.unravel_index(sim_num, domain_shape)
            # start out with all default parameters (signified by `None` values)
            norm_params = [None for ii in range(space.nparameters)]
            # determine the extrema for this parameter.  If parameter-distribution is bounded, use
            # full range.  If unbounded, use hardcoded values.
            yext = space._parameters[param_num].extrema   #: this is the extrema of the range
            xext = [0.0, 1.0]                             #: this is the extrema of the domain
            for ii in range(2):
                xext[ii] = xext[ii] if np.isfinite(yext[ii]) else PSPACE_DOMAIN_EXTREMA[ii]
                assert (0.0 <= xext[ii]) and (xext[ii] <= 1.0)
            # replace the parameter being varied with a fractional value based on the sample number
            norm_params[param_num] = xext[0] + samp_num * np.diff(xext)[0] / (domain_shape[1] - 1)
            params = space.normalized_params(norm_params)

        # Library: vary all parameters together
        else:
            params = space.param_dict(sim_num)
            msg = []
            for kk, vv in params.items():
                msg.append(f"{kk}={vv:.4e}")
            msg = ", ".join(msg)
            log.info(msg)

        rv, _sim_fname = run_sam_at_pspace_params(args, space, sim_num, params)

        if rv is False:
            failures += 1

        if (MAX_FAILURES is not None) and (failures > MAX_FAILURES):
            log.error("\n\n")
            err = f"Failed {failures} times on rank:{comm.rank}!"
            log.exception(err)
            raise RuntimeError(err)

        num_done += 1

    end = datetime.now()
    dur = (end - beg)
    log.info(f"\t{comm.rank} done at {str(end)} after {str(dur)} = {dur.total_seconds()}")

    # Make sure all processes are done so that all files are ready for merging
    comm.barrier()

    if (comm.rank == 0):
        log.info("Concatenating outputs into single file")
        holo.librarian.combine.sam_lib_combine(args.output, log, library=(not args.domain))
        log.info("Concatenation completed")

    return


def run_sam_at_pspace_params(args, space, pnum, params):
    """Run a given simulation (index number ``pnum``) in the ``space`` parameter-space.

    This function performs the following:

    (1) Constructs the appropriate filename for this simulation, and checks if it already exist.  If
        the file exists and the ``args.recreate`` option is False, the function returns
        ``True``, otherwise the function runs this simulation.
    (2) Calls ``space.model_for_params`` to generate the semi-analytic model and hardening
        instances; see the function
        :func:`holodeck.librarian.lib_tools._Param_Space.model_for_params()`.
    (3) Calculates populations and GW signatures from the SAM and hardening model using
        :func:`holodeck.librarian.lib_tools.run_model()`, and saves the results to an output file.
    (4) Optionally: some diagnostic plots are created in the :func:`make_plots()` function.

    Arguments
    ---------
    args : ``argparse.ArgumentParser`` instance
        Arguments from the ``gen_lib_sams.py`` script.
    space : :class:`holodeck.librarian.lib_tools._Param_space` instance
        Parameter space from which to construct populations.
    pnum : int
        Which parameter-sample from ``space`` should be run.
    params : dict
        Parameters for this particular simulation.  Dictionary of key-value pairs obtained from
        the parameter-space ``space`` and passed back to the parameter-space to produce a model.

    Returns
    -------
    rv : bool
        ``True`` if this simulation was successfully run, ``False`` otherwise.
        NOTE: an output file is created in either case.  See ``Notes`` [1] below.
    sim_fname : ``pathlib.Path`` instance
        Path of the simulation save file.

    Notes
    -----
    [1] When simulations are run, an output file is produced.  On caught failures, an output file is
      produced that contains a single key: 'fail'.  This designates the file as a failure.

    """
    log = args.log

    # ---- get output filename for this simulation, check if already exists

    library_flag = not args.domain
    sim_fname = lib_tools._get_sim_fname(args.output_sims, pnum, library=library_flag)

    beg = datetime.now()
    log.info(f"{pnum=} :: {sim_fname=} beginning at {beg}")

    if sim_fname.exists():
        log.info(f"File {sim_fname} already exists.  {args.recreate=}")
        temp = np.load(sim_fname)
        data_keys = list(temp.keys())

        if 'fail' in data_keys:
            log.info("Existing file was a failure, re-attempting...")
        # skip existing files unless we specifically want to recreate them
        elif not args.recreate:
            return True, sim_fname

    # ---- run Model

    try:
        log.debug("Selecting `sam` and `hard` instances")
        sam, hard = space.model_for_params(params)

        data = lib_tools.run_model(
            sam, hard,
            pta_dur=args.pta_dur, nfreqs=args.nfreqs, nreals=args.nreals, nloudest=args.nloudest,
            gwb_flag=args.gwb_flag, singles_flag=args.ss_flag, details_flag=False, params_flag=args.params_flag,
            log=log,
            #! DOPPLER
            doppler_args=args.doppler_args,
            #! -------
        )
        data['params'] = np.array([params[pn] for pn in space.param_names])
        data['param_names'] = space.param_names

        rv = True
    except Exception as err:
        log.exception(f"`run_model` FAILED on {pnum=}\n")
        log.exception(err)
        rv = False
        # failed simulations get an output file with a single key: 'fail'
        data = dict(fail=str(err))

    # ---- save data to file

    log.debug(f"Saving {pnum} to file | {args.gwb_flag=} {args.ss_flag=} {args.params_flag=}")
    log.debug(f"data has keys: {list(data.keys())}")
    np.savez(sim_fname, **data)
    log.info(f"Saved to {sim_fname}, size {holo.utils.get_file_size(sim_fname)} after {(datetime.now()-beg)}")

    # ---- make diagnostic plots

    if rv and args.plot:
        try:
            make_plots(args, data, sim_fname)
        except Exception as err:
            log.exception("Failed to make strain plot!")
            log.exception(err)

    return rv, sim_fname


def _setup_argparse(*args, **kwargs):
    """Setup the argument-parser for command-line usage.

    Arguments
    ---------
    *args : arguments
    **kwargs : keyword arguments

    """

    parser = argparse.ArgumentParser()
    parser.add_argument('param_space', type=str,
                        help="Parameter space class name, found in 'holodeck.param_spaces'.")

    parser.add_argument('output', metavar='output', type=str,
                        help='output path [created if doesnt exist]')

    # basic parameters
    parser.add_argument('-n', '--nsamples', action='store', dest='nsamples', type=int, default=1000,
                        help='number of parameter space samples')
    parser.add_argument('-r', '--nreals', action='store', dest='nreals', type=int,
                        help='number of realiz  ations', default=holo.librarian.DEF_NUM_REALS)
    parser.add_argument('-d', '--dur', action='store', dest='pta_dur', type=float,
                        help='PTA observing duration [yrs]', default=holo.librarian.DEF_PTA_DUR)
    parser.add_argument('-f', '--nfreqs', action='store', dest='nfreqs', type=int,
                        help='Number of frequency bins', default=holo.librarian.DEF_NUM_FBINS)
    parser.add_argument('-s', '--shape', action='store', dest='sam_shape', type=int,
                        help='Shape of SAM grid', default=None)
    parser.add_argument('-l', '--nloudest', action='store', dest='nloudest', type=int,
                        help='Number of loudest single sources', default=holo.librarian.DEF_NUM_LOUDEST)

    # what to run
    parser.add_argument('--gwb', dest="gwb_flag", default=True, action=argparse.BooleanOptionalAction,
                        help="calculate and store the 'gwb' per se")
    parser.add_argument('--ss', dest="ss_flag", default=True, action=argparse.BooleanOptionalAction,
                        help="calculate and store SS/CW sources and the BG separately")
    parser.add_argument('--params', dest="params_flag", default=True, action=argparse.BooleanOptionalAction,
                        help="calculate and store SS/BG binary parameters [NOTE: requires `--ss`]")
    parser.add_argument('--domain', default=False, action='store_true',
                        help="instead of generating a standard library, explore each parameter.")

    # how to run
    parser.add_argument('--resume', action='store_true', default=False,
                        help='resume production of a library by loading previous parameter-space from output directory')
    parser.add_argument('--recreate', action='store_true', default=False,
                        help='recreating existing simulation files')
    parser.add_argument('--plot', action='store_true', default=False,
                        help='produce plots for each simulation configuration')
    parser.add_argument('--seed', action='store', type=int, default=None,
                        help='Random seed to use')
    parser.add_argument('--TEST', action='store_true', default=False,
                        help='Run in test mode (NOTE: this resets other values)')

    # metavar='LEVEL', type=int, default=20,
    parser.add_argument('-v', '--verbose', metavar='LEVEL', type=int, nargs='?', const=20, default=30,
                        help='verbose output level (DEBUG=10, INFO=20, WARNING=30).')

    namespace = argparse.Namespace(**kwargs)
    args = parser.parse_args(*args, namespace=namespace)

    # ---- check / sanitize arguments

    output = Path(args.output).resolve()
    if not output.is_absolute:
        output = Path('.').resolve() / output
        output = output.resolve()

    if not args.gwb_flag and not args.ss_flag:
        raise RuntimeError("Either `--gwb` or `--ss` is required!")

    if args.params_flag and not args.ss_flag:
        raise RuntimeError("`--params` requires the `--ss` option!")

    if args.resume:
        # Need an existing output directory to resume from
        if not output.exists() or not output.is_dir():
            err = f"`--resume` is active but output path does not exist! '{output}'"
            raise FileNotFoundError(err)

        # Don't resume if we're recreating (i.e. erasing existing files)
        if args.recreate:
            err = "`resume` and `recreate` cannot both be set to True!"
            raise ValueError(err)

    # run in test mode
    if args.TEST:
        msg = "==== WARNING: running in test mode, other settings being overridden! ===="
        print("\n" + "=" * len(msg))
        print(msg)
        print("=" * len(msg) + "\n")

        global MAX_FAILURES
        MAX_FAILURES = 0
        args.nsamples = 10
        args.nreals = 3
        args.pta_dur = 10.0
        args.nfreqs = 5
        args.sam_shape = (11, 12, 13)
        args.nloudest = 2

        if not output.name.startswith("_"):
            output = output.with_name("_" + output.name)
            print(f"WARNING: changed output to '{output}'\n")

        if args.resume:
            raise RuntimeError("Cannot use `resume` in TEST mode!")


    # ---- Create output directories as needed

    output.mkdir(parents=True, exist_ok=True)
    holo.utils.mpi_print(f"output path: {output}")
    args.output = output

    if args.domain:
        sims_dirname = DIRNAME_DOMAIN_SIMS
    else:
        sims_dirname = DIRNAME_LIBRARY_SIMS

    output_sims = output.joinpath(sims_dirname)
    output_sims.mkdir(parents=True, exist_ok=True)
    args.output_sims = output_sims

    output_logs = output.joinpath("logs")
    output_logs.mkdir(parents=True, exist_ok=True)
    args.output_logs = output_logs

    if args.plot:
        output_plots = output.joinpath("figs")
        output_plots.mkdir(parents=True, exist_ok=True)
        args.output_plots = output_plots

    #! DOPPLER
    args.doppler_args = dict(
        expect    = 'optimistic',
        snr       = 8.0,
        tau_obs   = 10.0*YR,
        num_freqs = 200
    )

    doppler_data = np.loadtxt(DOPPLER_FNAME, delimiter=',')
    args.doppler_args['wlog_test']   = doppler_data[:,0]    #log10 of binary orbital frequency
    args.doppler_args['amplog_test'] = doppler_data[:,1]  #log10 of detectable amplitude (visual threshold)
    #! -------

    return args


def _setup_param_space(args):
    """Setup the parameter-space instance.

    For normal runs, identify the parameter-space class and construct a new class instance.
    For 'resume' runs, load a saved parameter-space instance.

    """
    log = args.log

    # ---- Determine and load the parameter-space class

    # `param_space` attribute must match the name of one of the classes in `holodeck.librarian`
    try:
        # if a namespace is specified for the parameter space, recursively follow it
        # i.e. this will work in two cases:
        # - `PS_Test` : if `PS_Test` is a class loaded in `librarian`
        # - `file_name.PS_Test` : as long as `file_name` is a module within `librarian`
        space_name = args.param_space.split(".")
        if len(space_name) > 1:
            space_class = holo.librarian
            for class_name in space_name:
                space_class = getattr(space_class, class_name)
        else:
            space_class = holo.librarian.param_spaces_dict[space_name[0]]

    except Exception as err:
        log.error(f"Failed to load parameter space '{args.param_space}' !")
        log.error("Make sure the class is defined, and imported into the `librarian` module.")
        log.error(err)
        raise err

    # ---- instantiate the parameter space class

    if args.resume:
        # Load pspace object from previous save
        log.info(f"{args.resume=} attempting to load pspace {space_class=} from {args.output=}")
        space, space_fname = holo.librarian.load_pspace_from_path(args.output, space_class=space_class, log=log)
        log.warning(f"Loaded param-space   save from {space_fname}")
    else:
        # we don't use standard samples when constructing a parameter-space 'domain'
        nsamples = None if args.domain else args.nsamples
        space = space_class(log, nsamples, args.sam_shape, args.seed)

    return space


def _save_config(args):
    import logging

    fname = args.output.joinpath(ARGS_CONFIG_FNAME)

    # Convert `args` parameters to serializable dictionary
    config = {}
    for kk, vv in args._get_kwargs():
        # convert `Path` instances to strings
        if isinstance(vv, Path):
            vv = str(vv)
        # cannot store `logging.Logger` instances (`log`)
        elif isinstance(vv, logging.Logger):
            continue
        #! DOPPLER
        elif kk.startswith('doppler'):
            continue
        #! -------

        config[kk] = vv

    # Add additional entries
    config['holodeck_version'] = holo.__version__
    config['holodeck_librarian_version'] = holo.librarian.__version__
    config['holodeck_git_hash'] = holo.utils.get_git_hash()
    config['created'] = str(datetime.now())

    with open(fname, 'w') as out:
        json.dump(config, out)

    args.log.warning(f"Saved to {fname} - {holo.utils.get_file_size(fname)}")

    return fname


def load_config_from_path(path, log):
    fname = Path(path).joinpath(ARGS_CONFIG_FNAME)

    with open(fname, 'r') as inp:
        config = json.load(inp)

    log.info("Loaded configuration from {fname}")

    pop_keys = [
        'holodeck_version', 'holodeck_librarian_version', 'holodeck_git_hash', 'created'
    ]
    for pk in pop_keys:
        val = config.pop(pk)
        log.info(f"\t{pk}={val}")

    pspace = config.pop('param_space')
    output = config.pop('output')

    args = _setup_argparse([pspace, output], **config)

    return args, fname


def _setup_log(comm, args):
    """Setup up the logging module logger for output messaging.

    Arguemnts
    ---------
    comm
    args

    Returns
    -------
    log : ``logging.Logger`` instance

    """
    beg = datetime.now()

    # ---- setup name of log file

    str_time = f"{beg.strftime('%Y%m%d-%H%M%S')}"
    # get the path to the directory containing the `holodeck` module
    # e.g.: "/Users/lzkelley/Programs/nanograv/holodeck"
    holo_parent = Path(holo.__file__).parent.parent
    # get the relative path from holodeck to this file
    # e.g.: "holodeck/librarian/gen_lib.py"
    log_name = Path(__file__).relative_to(holo_parent)
    # e.g.: "holodeck.librarian.gen_lib"
    log_name = ".".join(log_name.with_suffix("").parts)
    # e.g.: "holodeck.librarian.gen_lib__20230918-140722"
    log_name = f"{log_name}__{str_time}"
    # e.g.: "_holodeck.librarian.gen_lib__20230918-140722__r0003"
    if comm.rank > 0:
        log_name = f"_{log_name}__r{comm.rank:04d}"

    output = args.output_logs
    fname = f"{output.joinpath(log_name)}.log"

    # ---- setup logger

    log_lvl = args.verbose if comm.rank == 0 else holo.logger.DEBUG
    tostr = sys.stdout if comm.rank == 0 else False
    log = holo.logger.get_logger(name=log_name, level_stream=log_lvl, tofile=fname, tostr=tostr)
    log.info(f"Output path: {output}")
    log.info(f"        log: {fname}")
    log.info(args)
    return log


# ==============================================================================
# ====    Plotting Functions    ====
# ==============================================================================


def make_plots(args, data, sim_fname):
    """Generate diagnostic plots from the given simulation data and save to file.
    """
    import matplotlib.pyplot as plt
    log = args.log
    log.info("generating characteristic strain/psd plots")
    log.info("generating strain plots")
    plot_fname = args.output_plots.joinpath(sim_fname.name)
    hc_fname = str(plot_fname.with_suffix('')) + "_strain.png"

    if args.singles_flag:
        fobs_cents = data['fobs_cents']
        hc_bg = data['hc_bg']
        hc_ss = data['hc_ss']
        fig = holo.plot.plot_bg_ss(fobs_cents, bg=hc_bg, ss=hc_ss)
        fig.savefig(hc_fname, dpi=100)

    # log.info("generating PSD plots")
    # psd_fname = str(plot_fname.with_suffix('')) + "_psd.png"
    # fig = make_ss_plot(fobs_cents, hc_ss, hc_bg, fit_data)
    # fig.savefig(psd_fname, dpi=100)
    # log.info(f"Saved to {psd_fname}, size {holo.utils.get_file_size(psd_fname)}")

    if args.params_flag:
        log.info("generating pars plots")
        pars_fname = str(plot_fname.with_suffix('')) + "_pars.png"

        sspar = data['sspar']
        bgpar = data['bgpar']
        fig = make_pars_plot(fobs_cents, hc_ss, hc_bg, sspar, bgpar)
        fig.savefig(pars_fname, dpi=100)
        log.info(f"Saved to {pars_fname}, size {holo.utils.get_file_size(pars_fname)}")

    plt.close('all')
    return


def make_gwb_plot(fobs, gwb, fit_data):
    """Generate a GWB plot from the given data.

    """
    # fig = holo.plot.plot_gwb(fobs, gwb)
    psd = holo.utils.char_strain_to_psd(fobs[:, np.newaxis], gwb)
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


def make_ss_plot(fobs, hc_ss, hc_bg, fit_data):
    # fig = holo.plot.plot_gwb(fobs, gwb)
    psd_bg = holo.utils.char_strain_to_psd(fobs[:, np.newaxis], hc_bg)
    psd_ss = holo.utils.char_strain_to_psd(fobs[:, np.newaxis, np.newaxis], hc_ss)
    fig = holo.plot.plot_bg_ss(fobs, bg=psd_bg, ss=psd_ss, ylabel='GW Power Spectral Density')
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


def make_pars_plot(fobs, hc_ss, hc_bg, sspar, bgpar):
    """Plot total mass, mass ratio, initial d_c, final d_c

    """
    # fig = holo.plot.plot_gwb(fobs, gwb)
    fig = holo.plot.plot_pars(fobs, sspar, bgpar)

    return fig


if __name__ == "__main__":
    main()

    #! the below doesn't work for catching errors... maybe because of comm.barrier() calls?
    # try:
    #     main()
    # except Exception as err:
    #     print(f"Exception while running gen_lib.py::main() - '{err}'!")
    #     if comm is not None:
    #         comm.Abort()
    #     raise
    #! ----------
