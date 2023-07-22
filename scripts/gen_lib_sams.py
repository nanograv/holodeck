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

__version__ = '0.3.0'

import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
# import matplotlib.pyplot as plt
from mpi4py import MPI

import holodeck as holo
import holodeck.sams.sam
import holodeck.logger
# from holodeck.constants import YR
from holodeck import log as _log     #: import the default holodeck log just so that we can silence it
from holodeck import utils
# silence default holodeck log
_log.setLevel(_log.WARNING)
# _log.setLevel(_log.DEBUG)


MAX_FAILURES = 5

comm = MPI.COMM_WORLD

FILES_COPY_TO_OUTPUT = [__file__, holo.librarian.__file__, holo.param_spaces.__file__]


def main():

    args, space, log = holo.librarian.setup_basics(comm, FILES_COPY_TO_OUTPUT)
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
        holo.librarian.sam_lib_combine(args.output, log)
        log.info("Concatenating completed")

    return


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
