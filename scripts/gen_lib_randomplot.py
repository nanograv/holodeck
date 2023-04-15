"""Library Generation Script for Semi-Analytic Models.

Usage
-----

mpirun -n <NPROCS> python ./scripts/gen_lib_test.py <PATH> -n <SAMPS> 

    <NPROCS> : number of processors to run on
    <PATH> : output directory to save data to
    <SAMPS> : number of parameter-space samples for latin hyper-cube
    
Example:

    mpirun -n 8 python ./scripts/gen_lib_test.py output/2022-12-05_01 -n 32 -r 10 -f 20 -p 1


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
import matplotlib.pyplot as plt
from mpi4py import MPI

import holodeck as holo
import holodeck.sam
import holodeck.logger
# from holodeck.constants import YR
from holodeck import log as _log     #: import the default holodeck log just so that we can silence it


# silence default holodeck log
_log.setLevel(_log.WARNING)


# ---- setup argparse

def setup_argparse():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('output', metavar='output', type=str,
                        help='output path [created if doesnt exist]')
    parser.add_argument('-n', '--nsamples', action='store', dest='nsamples', type=int, default=25,
                        help='number of parameter space samples, must be square of prime')
    
    args = parser.parse_args()  
    


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

log_name = f"holodeck__gen_lib_randomplot_{BEG.strftime('%Y%m%d-%H%M%S')}"
if comm.rank > 0:
    log_name = f"_{log_name}_r{comm.rank}"

fname = f"{PATH_OUTPUT.joinpath(log_name)}.log"
log_lvl = holo.logger.WARNING
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

    print('PATH_OUTPUT:', PATH_OUTPUT)

    for par_num in iterator:
        print('par_num:', par_num)
        data = np.random.poisson(par_num, size=10)

        # save plot
        fig, ax = plt.subplots()
        ax.hist(data, bins=10)
        ax.set_title('lam=%.2f' % par_num)
        ax.set_xlim(0,15)
        fig.savefig(str(PATH_OUTPUT)+'/'+str(par_num)+'_plot.png', dpi=300)
        # save text file
        np.savetxt(str(PATH_OUTPUT)+'/'+str(par_num)+'_text.txt', data)
        



    end = datetime.now()
    log.info(f"\t{comm.rank} done at {str(end)} after {str(end-BEG)} = {(end-BEG).total_seconds()}")
    # Make sure all processes are done before exiting, so that all files are ready for merging
    comm.barrier()

    return


if __name__ == "__main__":
    np.seterr(divide='ignore', invalid='ignore', over='ignore')
    warnings.filterwarnings("ignore", category=UserWarning)

    main()

    if comm.rank == 0:
        end = datetime.now()
        tail = f"Done at {str(end)} after {str(end-BEG)} = {(end-BEG).total_seconds()}"
        print("\n" + "=" * len(tail) + "\n" + tail + "\n")

    sys.exit(0)
