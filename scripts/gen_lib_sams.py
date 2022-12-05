"""

mpirun -n 14  python ./scripts/gen_lib_sams.py output/test_2022-06-27

To-Do
-----
* Copy `gen_lib_sams.py` to output directory when code is run!

* LHS (at least with pydoe) is not deterministic (i.e. not reproducible).  Find a way to make reproducible.
* Rely less on `argparse` and instead just hardcode the parameters.  Save runtime parameters to output npz/hdf5 files.
* Use LHS to choose parameters themselves, instead of grid-points.  Also remove usage of grid entirely.
    * Does this resolve irregularities between different LHS implementations?
* Use subclassing to cleanup `Parameter_Space` object.  e.g. implement LHS as subclass of generic Parameter_Space class.
* Should we extract the LHS code from `pydoe`, instead of relying on the `pydoe` package formally?

"""

__version__ = '0.1.2'

import argparse
import os
import logging
import shutil
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
from mpi4py import MPI

import holodeck as holo
import holodeck.sam
import holodeck.logger
from holodeck.constants import YR, MSOL, GYR
from holodeck import log as _log     #: import the default holodeck log just so that we can silence it

from scipy.stats import qmc
import pyDOE

# silence default holodeck log
_log.setLevel(_log.WARNING)

# Default argparse parameters
DEF_NUM_REALS = 100
DEF_NUM_FBINS = 50
DEF_PTA_DUR = 16.03     # [yrs]


class Parameter_Space:

    def __init__(
        self, nsamples,
        gsmf_phi0=[-3.28, -2.16, 5],
        times=[1e-2, 10.0, 7],   # [Gyr]
        gpf_qgamma=[-0.4, +0.4, 5],
        hard_gamma_inner=[-1.5, -0.5, 5],
        mmb_amp=[0.1e9, 1.0e9, 9],
        mmb_plaw=[0.8, 1.5, 11],
        sam_shape=50,
    ):
        self.sam_shape = sam_shape
        self.gsmf_phi0 = np.linspace(*gsmf_phi0)
        self.times = np.logspace(*np.log10(times[:2]), times[2])
        self.gpf_qgamma = np.linspace(*gpf_qgamma)
        self.hard_gamma_inner = np.linspace(*hard_gamma_inner)
        self.mmb_amp = np.linspace(*mmb_amp)
        self.mmb_plaw = np.linspace(*mmb_plaw)
        params = [
            self.gsmf_phi0,
            self.times,   # [Gyr]
            self.gpf_qgamma,
            self.hard_gamma_inner,
            self.mmb_amp,
            self.mmb_plaw
        ]
        self.names = [
            'gsmf_phi0',
            'times',
            'gpf_qgamma',
            'hard_gamma_inner',
            'mmb_amp',
            'mmb_plaw'
        ]

        self.paramdimen = len(params)
        maxints = [tmparr.size for tmparr in params]

        # do scipy LHS
        if False:
            LHS = qmc.LatinHypercube(d=self.paramdimen, centered=False, strength=1)
            # if strength = 2, then n must be equal to p**2, with p prime, and d <= p + 1
            sampleindxs = LHS.random(n=nsamples)

        # do pyDOE LHS
        else:
            sampleindxs = pyDOE.lhs(n=self.paramdimen, samples=nsamples, criterion='m')

        for i in range(self.paramdimen):
            sampleindxs[:, i] = np.floor(maxints[i] * sampleindxs[:, i])

        sampleindxs = sampleindxs.astype(int)
        log.debug(f"d={len(params)} samplelims={maxints} {nsamples=}")
        self.sampleindxs = sampleindxs

        self.params = params
        self.param_grid = np.meshgrid(*params, indexing='ij')
        self.shape = self.param_grid[0].shape
        self.size = np.product(self.shape)
        if self.size < nsamples:
            log.warning(f"There are only {self.size} gridpoints in parameter space but you are requesting {nsamples} samples of them. They will be over-sampled")
        self.param_grid = np.moveaxis(self.param_grid, 0, -1)

        pass

    def number_to_index(self, num):
        idx = np.unravel_index(num, self.shape)
        return idx

    def lhsnumber_to_index(self, lhsnum):
        idx = tuple(self.sampleindxs[lhsnum])
        return idx

    def index_to_number(self, idx):
        num = np.ravel_multi_index(idx, self.shape)
        return num

    def param_dict_for_number(self, num):
        idx = self.number_to_index(num)
        pars = self.param_grid[idx]
        rv = {nn: pp for nn, pp in zip(self.names, pars)}
        return rv

    def param_dict_for_lhsnumber(self, lhsnum):
        idx = self.lhsnumber_to_index(lhsnum)
        pars = self.param_grid[idx]
        rv = {nn: pp for nn, pp in zip(self.names, pars)}
        return rv

    def params_for_number(self, num):
        idx = self.number_to_index(num)
        pars = self.param_grid[idx]
        return pars

    def params_for_lhsnumber(self, lhsnum):
        idx = self.lhsnumber_to_index(lhsnum)
        pars = self.param_grid[idx]
        return pars

    # def sam_for_number(self, num):
    #     param_grid = self.params_for_number(num)

    #     time, mmb_amp, mmb_plaw = param_grid

    #     gsmf = holo.sam.GSMF_Schechter()
    #     gpf = holo.sam.GPF_Power_Law()
    #     gmt = holo.sam.GMT_Power_Law()
    #     mmbulge = holo.relations.MMBulge_KH2013(mamp=mmb_amp*MSOL, mplaw=mmb_plaw)

    #     sam = holo.sam.Semi_Analytic_Model(gsmf=gsmf, gpf=gpf, gmt=gmt, mmbulge=mmbulge, shape=SAM_SHAPE)
    #     hard = holo.evolution.Fixed_Time.from_sam(sam, time*GYR, exact=True, progress=False)
    #     return sam, hard

    def sam_for_lhsnumber(self, lhsnum):
        param_grid = self.params_for_lhsnumber(lhsnum)

        gsmf_phi0, time, gpf_qgamma, hard_gamma_inner, mmb_amp, mmb_plaw = param_grid

        gsmf = holo.sam.GSMF_Schechter(phi0=gsmf_phi0)
        gpf = holo.sam.GPF_Power_Law(qgamma=gpf_qgamma)
        gmt = holo.sam.GMT_Power_Law()
        mmbulge = holo.relations.MMBulge_KH2013(mamp=mmb_amp*MSOL, mplaw=mmb_plaw)

        sam = holo.sam.Semi_Analytic_Model(gsmf=gsmf, gpf=gpf, gmt=gmt, mmbulge=mmbulge, shape=self.sam_shape)
        hard = holo.evolution.Fixed_Time.from_sam(sam, time*GYR, gamma_sc=hard_gamma_inner, exact=True, progress=False)
        return sam, hard


comm = MPI.COMM_WORLD

# ---- Setup ArgParse

parser = argparse.ArgumentParser()
parser.add_argument('output', metavar='output', type=str,
                    help='output path [created if doesnt exist]')
parser.add_argument('-n', '--nsamples', action='store', dest='nsamples', type=int, default=25,
                    help='number of parameter space samples, must be square of prime')
parser.add_argument('-r', '--reals', action='store', dest='reals', type=int,
                    help='number of realizations', default=DEF_NUM_REALS)
parser.add_argument('-d', '--dur', action='store', dest='pta_dur', type=float,
                    help='PTA observing duration [yrs]', default=DEF_PTA_DUR)
parser.add_argument('-f', '--freqs', action='store', dest='freqs', type=int,
                    help='Number of frequency bins', default=DEF_NUM_FBINS)
parser.add_argument('-t', '--test', action='store_true', default=False, dest='test',
                    help='Do not actually run, just output what parameters would have been done.')
parser.add_argument('-c', '--concatenate', action='store_true', default=False, dest='concatenateoutput',
                    help='Concatenate output into single hdf5 file.')
parser.add_argument('-v', '--verbose', action='store_true', default=False, dest='verbose',
                    help='verbose output [INFO]')

args = parser.parse_args()

if args.test:
    args.verbose = True

# ---- Setup outputs

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
    shutil.copyfile(src_file, dst_file)
    log.info(f"Copied {__file__} to {dst_file}")

# ---- setup Parameter_Space instance

SPACE = Parameter_Space(args.nsamples) if comm.rank == 0 else None
SPACE = comm.bcast(SPACE, root=0)

# ------------------------------------------------------------------------------
# ----    Methods
# ------------------------------------------------------------------------------


def main():
    bnum = 0
    # npars = SPACE.size
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
        # Convert from 1D index into 2D (param, real) specification
        # param, real = np.unravel_index(ind, (npars, nreals))
        # log.info(f"rank:{comm.rank} index:{ind} => {param=} {real=}")
        lhsparam = ind

        log.info(f"{comm.rank=} {ind=} {SPACE.param_dict_for_lhsnumber(lhsparam)}")
        if args.test:
            continue

        try:
            run_sam(lhsparam, PATH_OUTPUT)
        except Exception as err:
            logging.warning(f"\n\nWARNING: error on rank:{comm.rank}, index:{ind}")
            logging.warning(err)
            log.warning(f"\n\nWARNING: error on rank:{comm.rank}, index:{ind}")
            log.warning(err)
            import traceback
            traceback.print_exc()
            print("\n\n")

    end = datetime.now()
    print(f"\t{comm.rank} done at {str(end)} after {str(end-BEG)} = {(end-BEG).total_seconds()}")
    log.info(f"\t{comm.rank} done at {str(end)} after {str(end-BEG)} = {(end-BEG).total_seconds()}")
    bnum = _barrier(bnum)

    return


def run_sam(pnum, path_output):

    fname = f"lib_sams__p{pnum:06d}.npz"
    fname = os.path.join(path_output, fname)
    log.debug(f"{pnum=} :: {fname=}")
    if os.path.exists(fname):
        log.warning(f"File {fname} already exists.")

    pta_dur = args.pta_dur * YR
    nfreqs = args.freqs
    hifr = nfreqs/pta_dur
    pta_cad = 1.0 / (2 * hifr)
    fobs_cents = holo.utils.nyquist_freqs(pta_dur, pta_cad)
    fobs_edges = holo.utils.nyquist_freqs_edges(pta_dur, pta_cad)
    log.info(f"Created {fobs_cents.size} frequency bins")
    log.info(f"\t[{fobs_cents[0]*YR}, {fobs_cents[-1]*YR}] [1/yr]")
    log.info(f"\t[{fobs_cents[0]*1e9}, {fobs_cents[-1]*1e9}] [nHz]")
    assert nfreqs == fobs_cents.size

    log.debug("Selecting `sam` and `hard` instances")
    sam, hard = SPACE.sam_for_lhsnumber(pnum)
    log.debug("Calculating GWB for shape ({fobs_cents.size}, {args.reals})")
    gwb = sam.gwb(fobs_edges, realize=args.reals, hard=hard)
    log.debug(f"{holo.utils.stats(gwb)=}")
    legend = SPACE.param_dict_for_lhsnumber(pnum)
    log.debug("Saving {pnum} to file")
    np.savez(fname, fobs=fobs_cents, fobs_edges=fobs_edges, gwb=gwb,
             pnum=pnum, pdim=SPACE.paramdimen, nsamples=args.nsamples,
             lhs_grid=SPACE.sampleindxs, lhs_grid_idx=SPACE.lhsnumber_to_index(pnum),
             params=SPACE.params, names=SPACE.names, version=__version__, **legend)

    log.info(f"Saved to {fname} after {(datetime.now()-BEG)} (start: {BEG})")

    return


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
