"""Library generation interface.

This file can be run from the command-line to generate holodeck libraries, and also provides some
API methods for quick/easy generation of simulations.  In general, these methods are designed to
run simulations for populations constructed from parameter-spaces (i.e.
:class:`~holodeck.librarian.libraries._Param_Space` subclasses).

This script can be run by executing::

    python -m holodeck.librarian.gen_lib <ARGS>

Run ``python -m holodeck.librarian.gen_lib -h`` for usage information.

"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
import shutil

import numpy as np
# import scipy as sp
# import scipy.stats

import holodeck as holo
# from holodeck import cosmo
from holodeck.constants import YR
import holodeck.librarian
import holodeck.librarian.combine
from holodeck.librarian import (
    libraries,
    # DEF_NUM_FBINS, DEF_NUM_LOUDEST, DEF_NUM_REALS, DEF_PTA_DUR,
)
# from holodeck.sams import sam_cyutils

MAX_FAILURES = 5

# FILES_COPY_TO_OUTPUT = [__file__, holo.librarian.__file__, holo.param_spaces.__file__]
FILES_COPY_TO_OUTPUT = []


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

    Arguments
    ---------
    None

    Returns
    -------
    None

    """

    # ---- load mpi4py module

    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
    except ModuleNotFoundError as err:
        comm = None
        holo.log.error(f"failed to load `mpi4py` in {__file__}: {err}")
        holo.log.error("`mpi4py` may not be included in the standard `requirements.txt` file")
        holo.log.error("Check if you have `mpi4py` installed, and if not, please install it")
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
        copy_files = FILES_COPY_TO_OUTPUT
        # copy certain files to output directory
        if (not args.resume) and (copy_files is not None):
            for fname in copy_files:
                src_file = Path(fname)
                dst_file = args.output.joinpath("runtime_" + src_file.name)
                shutil.copyfile(src_file, dst_file)
                log.info(f"Copied {fname} to {dst_file}")

    # ---- get parameter-space class

    if comm.rank == 0:

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

        # instantiate the parameter space class
        if args.resume:
            # Load pspace object from previous save
            log.info(f"{args.resume=} attempting to load pspace {space_class=} from {args.output=}")
            space, space_fname = holo.librarian.load_pspace_from_dir(log, args.output, space_class)
            log.warning(f"resume={args.resume} :: Loaded param-space save from {space_fname}")
        else:
            space = space_class(log, args.nsamples, args.sam_shape, args.seed)
    else:
        space = None

    # share parameter space across processes
    space = comm.bcast(space, root=0)

    log.info(
        f"param_space={args.param_space}, samples={args.nsamples}, sam_shape={args.sam_shape}, nreals={args.nreals}\n"
        f"nfreqs={args.nfreqs}, pta_dur={args.pta_dur} [yr]\n"
    )
    comm.barrier()

    # ---- distribute jobs to processors

    # Split and distribute index numbers to all processes
    if comm.rank == 0:
        npars = args.nsamples
        indices = range(npars)
        indices = np.random.permutation(indices)
        indices = np.array_split(indices, comm.size)
        num_ind_per_proc = [len(ii) for ii in indices]
        log.info(f"{npars=} cores={comm.size} || max runs per core = {np.max(num_ind_per_proc)}")
    else:
        indices = None

    indices = comm.scatter(indices, root=0)
    iterator = holo.utils.tqdm(indices) if (comm.rank == 0) else np.atleast_1d(indices)

    # Save parameter space to output directory
    if (comm.rank == 0) and (not args.resume):
        space_fname = space.save(args.output)
        log.info(f"saved parameter space {space} to {space_fname}")

    comm.barrier()
    beg = datetime.now()
    log.info(f"beginning tasks at {beg}")
    failures = 0

    # ---- iterate over each processors' jobs

    for par_num in iterator:

        log.info(f"{comm.rank=} {par_num=}")
        pdict = space.param_dict(par_num)
        msg = []
        for kk, vv in pdict.items():
            msg.append(f"{kk}={vv:.4e}")
        msg = ", ".join(msg)
        log.info(msg)

        rv, _sim_fname = run_sam_at_pspace_num(args, space, par_num)
        if rv is False:
            failures += 1

        if failures > MAX_FAILURES:
            log.error("\n\n")
            err = f"Failed {failures} times on rank:{comm.rank}!"
            log.exception(err)
            raise RuntimeError(err)

    end = datetime.now()
    dur = (end - beg)
    log.info(f"\t{comm.rank} done at {str(end)} after {str(dur)} = {dur.total_seconds()}")

    # Make sure all processes are done so that all files are ready for merging
    comm.barrier()

    if (comm.rank == 0):
        log.info("Concatenating outputs into single file")
        holo.librarian.combine.sam_lib_combine(args.output, log)
        log.info("Concatenation completed")

    return


def run_sam_at_pspace_num(args, space, pnum):
    """Run a given simulation (index number ``pnum``) in the ``space`` parameter-space.

    This function performs the following:

    (1) Constructs the appropriate filename for this simulation, and checks if it already exist.  If
        the file exists and the ``args.recreate`` option is not specified, the function returns
        ``True``, otherwise the function runs this simulation.
    (2) Calls ``space.model_for_sample_number`` to generate the semi-analytic model and hardening
        instances; see the function
        :func:`holodeck.librarian.libraries._Param_Space.model_for_sample_number()`.
    (3) Calculates populations and GW signatures from the SAM and hardening model using
        :func:`holodeck.librarian.libraries.run_model()`, and saves the results to an output file.
    (4) Optionally: some diagnostic plots are created in the :func:`make_plots()` function.

    Arguments
    ---------
    args : ``argparse.ArgumentParser`` instance
        Arguments from the `gen_lib_sams.py` script.
        NOTE: this should be improved.
    space : :class:`holodeck.librarian.libraries._Param_space` instance
        Parameter space from which to construct populations.
    pnum : int
        Which parameter-sample from ``space`` should be run.

    Returns
    -------
    rv : bool
        ``True`` if this simulation was successfully run, ``False`` otherwise.
    sim_fname : ``pathlib.Path`` instance
        Path of the simulation save file.

    """
    log = args.log

    # ---- get output filename for this simulation, check if already exists

    sim_fname = libraries._get_sim_fname(args.output_sims, pnum)

    beg = datetime.now()
    log.info(f"{pnum=} :: {sim_fname=} beginning at {beg}")

    if sim_fname.exists():
        log.info(f"File {sim_fname} already exists.  {args.recreate=}")
        # skip existing files unless we specifically want to recreate them
        if not args.recreate:
            return True, sim_fname

    # ---- run Model

    try:
        log.debug("Selecting `sam` and `hard` instances")
        sam, hard = space.model_for_sample_number(pnum)

        data = libraries.run_model(
            sam, hard,
            pta_dur=args.pta_dur, nfreqs=args.nfreqs, nreals=args.nreals, nloudest=args.nloudest,
            gwb_flag=args.gwb_flag, singles_flag=args.ss_flag, details_flag=False, params_flag=args.params_flag,
            log=log,
        )

        rv = True
    except Exception as err:
        log.exception(f"`run_model` FAILED on {pnum=}\n")
        log.exception(err)
        rv = False
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

    # how do run
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

    # parser.add_argument('-v', '--verbose', action='store_true', default=False, dest='verbose',
    #                     help='verbose output [INFO]')

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
        if not output.exists() or not output.is_dir():
            raise FileNotFoundError(f"`--resume` is active but output path does not exist! '{output}'")

    #
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

    output_sims = output.joinpath("sims")
    output_sims.mkdir(parents=True, exist_ok=True)
    args.output_sims = output_sims

    output_logs = output.joinpath("logs")
    output_logs.mkdir(parents=True, exist_ok=True)
    args.output_logs = output_logs

    if args.plot:
        output_plots = output.joinpath("figs")
        output_plots.mkdir(parents=True, exist_ok=True)
        args.output_plots = output_plots

    return args


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

    # log_lvl = holo.logger.INFO if args.verbose else holo.logger.WARNING
    log_lvl = holo.logger.DEBUG
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
