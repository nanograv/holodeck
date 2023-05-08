"""Library Generation Script for Semi-Analytic Models.

Usage
-----

mpirun -n <NPROCS> python ./scripts/gen_lib_sams.py <PARAM_SPACE> <PATH> -n <SAMPS> -r <REALS> -f <FREQS> -s <SHAPE>

    <PARAM_SPACE> : name of the parameter space class (in holodeck.param_spaces) to run; must match exactly.
    <NPROCS> : number of processors to run on
    <PATH> : output directory to save data to
    <SAMPS> : number of parameter-space samples for latin hyper-cube
    <REALS> : number of realizations at each parameter-space location
    <FREQS> : number of frequencies (multiples of PTA observing baseline)
    <SHAPE> : SAM grid shape, as a single int value (applied to all dimensions)

Example:

    mpirun -n 8 python ./scripts/gen_lib_sams.py PS_Broad_Uniform_02B output/2022-12-05_01 -n 32 -r 10 -f 20 -s 80


To-Do
-----
* mark output directories as incomplete until all runs have been finished.
  Merged libraries from incomplete directories should also get some sort of flag!

"""

__version__ = '0.2.2'

import argparse
import os
import shutil
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
# import matplotlib.pyplot as plt
from mpi4py import MPI

import holodeck as holo
import holodeck.sam
import holodeck.logger
# from holodeck.constants import YR
from holodeck import log as _log     #: import the default holodeck log just so that we can silence it
from holodeck import utils
# silence default holodeck log
_log.setLevel(_log.WARNING)

# Default argparse parameters
DEF_NUM_REALS = 100
DEF_NUM_FBINS = 40
DEF_PTA_DUR = 16.03     # [yrs]

MAX_FAILURES = 5

comm = MPI.COMM_WORLD

FILES_COPY_TO_OUTPUT = [__file__, holo.librarian.__file__, holo.param_spaces.__file__]


def main():

    args, space, log = setup_basics()
    comm.barrier()

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

    if (comm.rank == 0) and (not args.resume):
        space_fname = space.save(args.output)
        log.info(f"saved parameter space {space} to {space_fname}")

    comm.barrier()
    beg = datetime.now()
    log.info(f"beginning tasks at {beg}")
    failures = 0

    for par_num in iterator:
        log.info(f"{comm.rank=} {par_num=}")
        pdict = space.param_dict(par_num)
        msg = "\n"
        for kk, vv in pdict.items():
            msg += f"{kk}={vv}\n"
        log.info(msg)

        rv = holo.librarian.run_sam_at_pspace_num(args, space, par_num)
        if rv is False:
            failures += 1

        if failures > MAX_FAILURES:
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
        holo.librarian.sam_lib_combine(args.output, log, path_sims=args.output_sims)
        log.info("Concatenating completed")

    return


def setup_basics():
    if comm.rank == 0:
        args = _setup_argparse()
    else:
        args = None

    # share `args` to all processes
    args = comm.bcast(args, root=0)

    # setup log instance, separate for all processes
    log = _setup_log(args)
    args.log = log

    if comm.rank == 0:
        # copy certain files to output directory
        if not args.resume:
            for fname in FILES_COPY_TO_OUTPUT:
                src_file = Path(fname)
                dst_file = args.output.joinpath("runtime_" + src_file.name)
                shutil.copyfile(src_file, dst_file)
                log.info(f"Copied {fname} to {dst_file}")

        # get parameter-space class
        try:
            # `param_space` attribute must match the name of one of the classes in `holo.param_spaces`
            space_class = getattr(holo.param_spaces, args.param_space)
        except Exception as err:
            log.exception(f"Failed to load '{args.param_space}' from holo.param_spaces!")
            log.exception(err)
            raise err

        # instantiate the parameter space class
        if args.resume:
            # Load pspace object from previous save
            space, space_fname = holo.librarian.load_pspace_from_dir(args.output, space_class)
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

    return args, space, log


def _setup_argparse(*args, **kwargs):
    assert comm.rank == 0

    parser = argparse.ArgumentParser()
    parser.add_argument('param_space', type=str,
                        help="Parameter space class name, found in 'holodeck.param_spaces'.")

    parser.add_argument('output', metavar='output', type=str,
                        help='output path [created if doesnt exist]')

    parser.add_argument('-n', '--nsamples', action='store', dest='nsamples', type=int, default=1000,
                        help='number of parameter space samples')
    parser.add_argument('-r', '--nreals', action='store', dest='nreals', type=int,
                        help='number of realiz  ations', default=DEF_NUM_REALS)
    parser.add_argument('-d', '--dur', action='store', dest='pta_dur', type=float,
                        help='PTA observing duration [yrs]', default=DEF_PTA_DUR)
    parser.add_argument('-f', '--nfreqs', action='store', dest='nfreqs', type=int,
                        help='Number of frequency bins', default=DEF_NUM_FBINS)
    parser.add_argument('-s', '--shape', action='store', dest='sam_shape', type=int,
                        help='Shape of SAM grid', default=None)

    parser.add_argument('--resume', action='store_true', default=False,
                        help='resume production of a library by loading previous parameter-space from output directory')
    parser.add_argument('--plot', action='store_true', default=False,
                        help='produce plots for each simulation configuration')
    parser.add_argument('--seed', action='store', type=int, default=None,
                        help='Random seed to use')
    # parser.add_argument('-t', '--test', action='store_true', default=False, dest='test',
    #                     help='Do not actually run, just output what parameters would have been done.')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, dest='verbose',
                        help='verbose output [INFO]')

    args = parser.parse_args(*args, **kwargs)

    output = Path(args.output).resolve()
    if not output.is_absolute:
        output = Path('.').resolve() / output
        output = output.resolve()

    if args.resume:
        if not output.exists() or not output.is_dir():
            raise FileNotFoundError(f"`--resume` is active but output path does not exist! '{output}'")
    elif output.exists():
        raise RuntimeError(f"Output {output} already exists!  Overwritting not currently supported!")

    # ---- Create output directories as needed

    output.mkdir(parents=True, exist_ok=True)
    utils.my_print(f"output path: {output}")
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


def _setup_log(args):
    beg = datetime.now()
    log_name = f"holodeck__gen_lib_sams_{beg.strftime('%Y%m%d-%H%M%S')}"
    if comm.rank > 0:
        log_name = f"_{log_name}_r{comm.rank}"

    output = args.output_logs
    fname = f"{output.joinpath(log_name)}.log"
    log_lvl = holo.logger.INFO if args.verbose else holo.logger.WARNING
    tostr = sys.stdout if comm.rank == 0 else False
    log = holo.logger.get_logger(name=log_name, level_stream=log_lvl, tofile=fname, tostr=tostr)
    log.info(f"Output path: {output}")
    log.info(f"        log: {fname}")
    log.info(args)
    return log


def mpiabort_excepthook(type, value, traceback):
    sys.__excepthook__(type, value, traceback)
    comm.Abort()
    return


if __name__ == "__main__":
    np.seterr(divide='ignore', invalid='ignore', over='ignore')
    warnings.filterwarnings("ignore", category=UserWarning)
    sys.excepthook = mpiabort_excepthook
    beg_time = datetime.now()
    beg_time = comm.bcast(beg_time, root=0)

    if comm.rank == 0:
        this_fname = os.path.abspath(__file__)
        head = f"holodeck :: {this_fname} : {str(beg_time)} - rank: {comm.rank}/{comm.size}"
        head = "\n" + head + "\n" + "=" * len(head) + "\n"
        utils.my_print(head)

    main()

    if comm.rank == 0:
        end = datetime.now()
        dur = end - beg_time
        tail = f"Done at {str(end)} after {str(dur)} = {dur.total_seconds()}"
        utils.my_print("\n" + "=" * len(tail) + "\n" + tail + "\n")

    sys.exit(0)
