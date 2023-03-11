"""Library Generation Script for Semi-Analytic Models.

Usage
-----

mpirun -n <NPROCS> python ./scripts/gen_lib_sams.py <PATH> -n <SAMPS> -r <REALS> -f <FREQS>

    <NPROCS> : number of processors to run on
    <PATH> : output directory to save data to
    <SAMPS> : number of parameter-space samples for latin hyper-cube
    <REALS> : number of realizations at each parameter-space location
    <FREQS> : number of frequencies (multiples of PTA observing baseline)

Example:

    mpirun -n 8 python ./scripts/gen_lib_sams.py output/2022-12-05_01 -n 32 -r 10 -f 20


To-Do
-----
* #! IMPORTANT: mark output directories as incomplete until all runs have been finished.  Merged libraries from incomplete directories should also get some sort of flag! !#
* Setting `_PARAM_NAMES` seems redundant to the arguments passed in super().__init__(), can these just be grabbed from kwargs?

"""

__version__ = '0.2.2'

import argparse
import os
import logging
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


# silence default holodeck log
_log.setLevel(_log.WARNING)

# Default argparse parameters
DEF_SAM_SHAPE = (61, 81, 101)
DEF_NUM_REALS = 100
DEF_NUM_FBINS = 40
DEF_PTA_DUR = 16.03     # [yrs]

DEF_ECCEN_NUM_STEPS = 123
DEF_ECCEN_NHARMS = 100

MAX_FAILURES = 5


# ---- setup argparse

def setup_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('param_space', type=str,
                        help="Parameter space class name, found in 'holodeck.param_spaces'.")

    parser.add_argument('output', metavar='output', type=str,
                        help='output path [created if doesnt exist]')

    parser.add_argument('-n', '--nsamples', action='store', dest='nsamples', type=int, default=25,
                        help='number of parameter space samples, must be square of prime')
    parser.add_argument('-r', '--nreals', action='store', dest='nreals', type=int,
                        help='number of realizations', default=DEF_NUM_REALS)
    parser.add_argument('-d', '--dur', action='store', dest='pta_dur', type=float,
                        help='PTA observing duration [yrs]', default=DEF_PTA_DUR)
    parser.add_argument('-f', '--nfreqs', action='store', dest='nfreqs', type=int,
                        help='Number of frequency bins', default=DEF_NUM_FBINS)
    parser.add_argument('-s', '--shape', action='store', dest='sam_shape', type=int,
                        help='Shape of SAM grid', default=DEF_SAM_SHAPE)

    parser.add_argument('--seed', action='store', type=int, default=None,
                        help='Random seed to use')
    parser.add_argument('-t', '--test', action='store_true', default=False, dest='test',
                        help='Do not actually run, just output what parameters would have been done.')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, dest='verbose',
                        help='verbose output [INFO]')

    args = parser.parse_args()  
    
    if args.test:
        args.verbose = True

    return args


comm = MPI.COMM_WORLD

args = setup_argparse() if comm.rank == 0 else None
args = comm.bcast(args, root=0)

# ---- setup outputs

BEG = datetime.now()
BEG = comm.bcast(BEG, root=0)

this_fname = os.path.abspath(__file__)
head = f"holodeck :: {this_fname} : {str(BEG)} - rank: {comm.rank}/{comm.size}"
head = "\n" + head + "\n" + "=" * len(head) + "\n"
if comm.rank == 0:
    print(head)

PATH_OUTPUT = Path(args.output).resolve()
if not PATH_OUTPUT.is_absolute:
    PATH_OUTPUT = Path('.').resolve() / PATH_OUTPUT
    PATH_OUTPUT = PATH_OUTPUT.resolve()

if comm.rank == 0:
    PATH_OUTPUT.mkdir(parents=True, exist_ok=True)

comm.barrier()

# ---- setup logger ----

log_name = f"holodeck__gen_lib_sams_{BEG.strftime('%Y%m%d-%H%M%S')}"
if comm.rank > 0:
    log_name = f"_{log_name}_r{comm.rank}"

fname = f"{PATH_OUTPUT.joinpath(log_name)}.log"
log_lvl = holo.logger.INFO if args.verbose else holo.logger.WARNING
tostr = sys.stdout if comm.rank == 0 else False
log = holo.logger.get_logger(name=log_name, level_stream=log_lvl, tofile=fname, tostr=tostr)
log.info(head)
log.info(f"Output path: {PATH_OUTPUT}")
log.info(f"        log: {fname}")
log.info(args)
args.log = log

if comm.rank == 0:
    copy_paths = [__file__, holo.librarian.__file__, holo.param_spaces.__file__]
    for fname in copy_paths:
        src_file = Path(fname)
        dst_file = PATH_OUTPUT.joinpath("runtime_" + src_file.name)
        # dst_file = dst_file.parent / ("runtime_" + dst_file.name)
        shutil.copyfile(src_file, dst_file)
        log.info(f"Copied {fname} to {dst_file}")

# ---- setup Parameter_Space instance

try:
    space = getattr(holo.param_spaces, args.param_space)
except Exception as err:
    log.exception(f"Failed to load '{args.param_space}' from holo.param_spaces!")
    log.exception(err)
    raise err

space = space(log, args.nsamples, args.sam_shape, args.seed) if comm.rank == 0 else None
space = comm.bcast(space, root=0)

log.info(
    f"param_space={args.param_space}, samples={args.nsamples}, sam_shape={args.sam_shape}, nreals={args.nreals}\n"
    f"nfreqs={args.nfreqs}, pta_dur={args.pta_dur} [yr]\n"
)

# ------------------------------------------------------------------------------
# ----    Methods
# ------------------------------------------------------------------------------


def main():
    failures = 0
    npars = args.nsamples

    comm.barrier()

    # Split and distribute index numbers to all processes
    if comm.rank == 0:
        indices = range(npars)
        indices = np.random.permutation(indices)
        indices = np.array_split(indices, comm.size)
        num_ind_per_proc = [len(ii) for ii in indices]
        log.info(f"{npars=} cores={comm.size} || max runs per core = {np.max(num_ind_per_proc)}")
    else:
        indices = None

    indices = comm.scatter(indices, root=0)

    iterator = holo.utils.tqdm(indices) if comm.rank == 0 else np.atleast_1d(indices)

    if args.test:
        log.info("Running in testing mode. Outputting parameters:")

    for par_num in iterator:
        log.info(f"{comm.rank=} {par_num=}")
        pdict = space.param_dict(par_num)
        msg = "\n"
        for kk, vv in pdict.items():
            msg += f"{kk}={vv}\n"
        log.info(msg)

        if args.test:
            continue

        try:
            rv = holo.librarian.run_sam_at_pspace_num(args, space, par_num, PATH_OUTPUT)
            if rv is False:
                failures += 1

        except Exception as err:
            failures += 1
            logging.warning(f"\n\nWARNING: error on rank:{comm.rank}, index:{par_num}")
            logging.warning(err)
            log.warning(f"\n\nWARNING: error on rank:{comm.rank}, index:{par_num}")
            log.warning(err)
            import traceback
            traceback.print_exc()
            print("\n\n")
            raise

        if failures > MAX_FAILURES:
            err = f"Failed {failures} times on rank:{comm.rank}!"
            log.exception(err)
            raise RuntimeError(err)

    end = datetime.now()
    log.info(f"\t{comm.rank} done at {str(end)} after {str(end-BEG)} = {(end-BEG).total_seconds()}")
    # Make sure all processes are done before exiting, so that all files are ready for merging
    comm.barrier()

    return


if __name__ == "__main__":
    np.seterr(divide='ignore', invalid='ignore', over='ignore')
    warnings.filterwarnings("ignore", category=UserWarning)

    main()

    if (comm.rank == 0) and (not args.test):
        log.info("Concatenating outputs into single file")
        holo.librarian.sam_lib_combine(PATH_OUTPUT, log)
        log.info("Concatenating completed")

    if comm.rank == 0:
        end = datetime.now()
        tail = f"Done at {str(end)} after {str(end-BEG)} = {(end-BEG).total_seconds()}"
        print("\n" + "=" * len(tail) + "\n" + tail + "\n")

    sys.exit(0)
