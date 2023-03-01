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
import psutil
import resource
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
from holodeck.constants import YR
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
    parser.add_argument('-l', '--lhs', action='store', choices=['scipy', 'pydoe'], default='scipy',
                        help='Latin Hyper Cube sampling implementation to use (scipy or pydoe)')
    parser.add_argument('--seed', action='store', type=int, default=None,
                        help='Random seed to use')
    parser.add_argument('-t', '--test', action='store_true', default=False, dest='test',
                        help='Do not actually run, just output what parameters would have been done.')
    parser.add_argument('-c', '--concatenate', action='store_true', default=False, dest='concatenateoutput',
                        help='Concatenate output into single hdf5 file.')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, dest='verbose',
                        help='verbose output [INFO]')

    args = parser.parse_args()

    if args.test:
        args.verbose = True

    return args


SPACE = holo.param_spaces.PS_Circ_01
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

if comm.rank == 0:
    src_file = Path(this_fname)
    dst_file = PATH_OUTPUT.joinpath(src_file.name)
    dst_file = dst_file.parent / ("runtime_" + dst_file.name)
    shutil.copyfile(src_file, dst_file)
    log.info(f"Copied {__file__} to {dst_file}")

# ---- setup Parameter_Space instance

log.warning(f"SPACE = {SPACE}")
if issubclass(SPACE, holo.librarian._LHS_Parameter_Space):
    space = SPACE(log, args.nsamples, args.sam_shape, args.lhs, args.seed) if comm.rank == 0 else None
else:
    space = SPACE(log, args.nsamples, args.sam_shape) if comm.rank == 0 else None
space = comm.bcast(space, root=0)

log.info(
    f"samples={args.nsamples}, sam_shape={args.sam_shape}, nreals={args.nreals}\n"
    f"nfreqs={args.nfreqs}, pta_dur={args.pta_dur} [yr]\n"
    # f"space.shape={space.shape}"
)

# ------------------------------------------------------------------------------
# ----    Methods
# ------------------------------------------------------------------------------


def main():
    bnum = 0
    failures = 0
    npars = args.nsamples

    bnum = _barrier(bnum)

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

    bnum = _barrier(bnum)
    iterator = holo.utils.tqdm(indices) if comm.rank == 0 else np.atleast_1d(indices)

    if args.test:
        log.info("Running in testing mode. Outputting parameters:")

    for ind in iterator:
        lhsparam = ind
        # log.info(f"{comm.rank=} {ind=} {space.param_dict_for_lhsnumber(lhsparam)}")
        log.info(f"{comm.rank=} {ind=}")
        pdict = space.param_dict_for_lhsnumber(lhsparam)
        msg = "\n"
        for kk, vv in pdict.items():
            msg += f"{kk}={vv}\n"
        log.info(msg)

        if args.test:
            continue

        try:
            rv = run_sam(lhsparam, PATH_OUTPUT)
            if rv is False:
                failures += 1

            if failures > MAX_FAILURES:
                err = f"Failed {failures} times on rank:{comm.rank}!"
                log.exception(err)
                raise RuntimeError(err)

        except Exception as err:
            logging.warning(f"\n\nWARNING: error on rank:{comm.rank}, index:{ind}")
            logging.warning(err)
            log.warning(f"\n\nWARNING: error on rank:{comm.rank}, index:{ind}")
            log.warning(err)
            import traceback
            traceback.print_exc()
            print("\n\n")
            raise

    end = datetime.now()
    log.info(f"\t{comm.rank} done at {str(end)} after {str(end-BEG)} = {(end-BEG).total_seconds()}")
    # Make sure all processes are done before exiting, so that all files are ready for merging
    bnum = _barrier(bnum)

    return


def run_sam(pnum, path_output):
    fname = f"lib_sams__p{pnum:06d}.npz"
    fname = Path(path_output, fname)
    log.debug(f"{pnum=} :: {fname=}")
    if os.path.exists(fname):
        log.warning(f"File {fname} already exists.")

    def log_mem():
        # results.ru_maxrss is KB on Linux, B on macos
        mem_max = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2)
        process = psutil.Process(os.getpid())
        mem_rss = process.memory_info().rss / 1024**3
        mem_vms = process.memory_info().vms / 1024**3
        log.info(f"Current memory usage: max={mem_max:.2f} GB, RSS={mem_rss:.2f} GB, VMS={mem_vms:.2f} GB")

    pta_dur = args.pta_dur * YR
    nfreqs = args.nfreqs
    hifr = nfreqs/pta_dur
    pta_cad = 1.0 / (2 * hifr)
    fobs_cents = holo.utils.nyquist_freqs(pta_dur, pta_cad)
    fobs_edges = holo.utils.nyquist_freqs_edges(pta_dur, pta_cad)
    log.info(f"Created {fobs_cents.size} frequency bins")
    log.info(f"\t[{fobs_cents[0]*YR}, {fobs_cents[-1]*YR}] [1/yr]")
    log.info(f"\t[{fobs_cents[0]*1e9}, {fobs_cents[-1]*1e9}] [nHz]")
    log_mem()
    assert nfreqs == fobs_cents.size

    try:
        log.debug("Selecting `sam` and `hard` instances")
        sam, hard = space.sam_for_lhsnumber(pnum)
        log_mem()
        log.debug(f"Calculating GWB for shape ({fobs_cents.size}, {args.nreals})")
        gwb = sam.gwb(fobs_edges, realize=args.nreals, hard=hard)
        log_mem()
        log.debug(f"{holo.utils.stats(gwb)=}")
        legend = space.param_dict_for_lhsnumber(pnum)
        log.debug(f"Saving {pnum} to file")
        data = dict(fobs=fobs_cents, fobs_edges=fobs_edges, gwb=gwb)
        rv = True
    except Exception as err:
        log.exception("\n\n")
        log.exception("="*100)
        log.exception(f"`run_sam` FAILED on {pnum=}\n")
        log.exception(err)
        log.exception("="*100)
        log.exception("\n\n")
        rv = False
        data = dict(fail=str(err))

    if rv:
        try:
            fits_data = holo.librarian.get_gwb_fits_data(fobs_cents, gwb)
        except Exception as err:
            log.exception("Failed to load gwb fits data!")
            log.exception(err)
            fits_data = {}

    else:
        fits_data = {}

    meta_data = dict(
        pnum=pnum, pdim=space.paramdimen, nsamples=args.nsamples,
        lhs_grid=space.sampleindxs, lhs_grid_idx=space.lhsnumber_to_index(pnum),
        params=space.params, names=space.names, version=__version__
    )

    np.savez(fname, **data, **meta_data, **fits_data, **legend)
    log.info(f"Saved to {fname}, size {holo.utils.get_file_size(fname)} after {(datetime.now()-BEG)} (start: {BEG})")

    if rv:
        try:
            fname = fname.with_suffix('.png')
            fig = make_gwb_plot(fobs_cents, gwb, fits_data)
            fig.savefig(fname, dpi=300)
            log.info(f"Saved to {fname}, size {holo.utils.get_file_size(fname)}")
            plt.close('all')
        except Exception as err:
            log.exception("Failed to make gwb plot!")
            log.exception(err)

    return rv


def make_gwb_plot(fobs, gwb, fits_data):
    fig = holo.plot.plot_gwb(fobs, gwb)
    ax = fig.axes[0]

    if len(fits_data):
        xx = fobs * YR
        yy = 1e-15 * np.power(xx, -2.0/3.0)
        ax.plot(xx, yy, 'r-', alpha=0.5, lw=1.0, label="$10^{-15} \cdot f_\\mathrm{yr}^{-2/3}$")

        fits = holo.librarian.get_gwb_fits_data(fobs, gwb)

        for ls, idx in zip([":", "--"], [1, -1]):
            med_lamp = fits['fit_med_lamp'][idx]
            med_plaw = fits['fit_med_plaw'][idx]
            yy = (10.0 ** med_lamp) * (xx ** med_plaw)
            label = fits['fit_nbins'][idx]
            label = 'all' if label in [0, None] else label
            ax.plot(xx, yy, color='k', ls=ls, alpha=0.5, lw=2.0, label=str(label) + " bins")

        label = fits['fit_label'].replace(" | ", "\n")
        fig.text(0.99, 0.99, label, fontsize=6, ha='right', va='top')

    return fig


def _barrier(bnum):
    log.debug(f"barrier {bnum}")
    comm.barrier()
    bnum += 1
    return bnum


if __name__ == "__main__":
    np.seterr(divide='ignore', invalid='ignore', over='ignore')
    warnings.filterwarnings("ignore", category=UserWarning)

    if not args.concatenateoutput:
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
