"""

mpirun -n 14  python ./scripts/gen_spec_lib_sams.py output/test_2022-06-27

"""

import argparse
import os
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
# import h5py
from mpi4py import MPI


import holodeck as holo
import holodeck.sam
import holodeck.logger
from holodeck.constants import YR, MSOL, GYR  # noqa

from scipy.stats import qmc
import pyDOE
import pickle

class Parameter_Space:

    def __init__(
        self,
        #gsmf_phi0=[-3.35, -2.23, 7],
        # gsmf_phi0=[-3.61, -1.93, 7],
        times=[1e-2, 10.0, 7],   # [Gyr]
        # gsmf_alpha0=[-1.56, -0.92, 5],
        # mmb_amp=[0.39e9, 0.61e9, 9], mmb_plaw=[1.01, 1.33, 11]
        mmb_amp=[0.1e9, 1.0e9, 9], mmb_plaw=[0.8, 1.5, 11],
        nsamples=25
    ):

        #self.gsmf_phi0 = np.linspace(*gsmf_phi0)
        self.times = np.logspace(*np.log10(times[:2]), times[2])
        # self.gsmf_alpha0 = np.linspace(*gsmf_alpha0)
        self.mmb_amp = np.linspace(*mmb_amp)
        self.mmb_plaw = np.linspace(*mmb_plaw)
        pars = [
            self.times,   # [Gyr]
            #self.gsmf_phi0,
            # self.gsmf_alpha0,
            self.mmb_amp,
            self.mmb_plaw
        ]
        self.names = [
            'times',
            #'gsmf_phi0',
            # 'gsmf_alpha0',
            'mmb_amp',
            'mmb_plaw'
        ]
        self.paramdimen = len(pars)
        maxints = [tmparr.size for tmparr in pars]
        if False: # do scipy LHS
            LHS = qmc.LatinHypercube(d=self.paramdimen, centered=False, strength=1)
            # if strength = 2, then n must be equal to p**2, with p prime, and d <= p + 1
            sampleindxs = LHS.random(n=nsamples)
        else: # do pyDOE LHS
            sampleindxs = pyDOE.lhs(n=self.paramdimen, samples=nsamples, criterion='m')
        for i in range(self.paramdimen):
            sampleindxs[:, i] = np.floor(maxints[i] * sampleindxs[:, i])
        sampleindxs = sampleindxs.astype(int)
        LOG.debug(f"d={len(pars)} samplelims={maxints} {nsamples=}")
        self.sampleindxs = sampleindxs
        self.params = np.meshgrid(*pars, indexing='ij')
        self.shape = self.params[0].shape
        self.size = np.product(self.shape)
        if self.size < nsamples:
            LOG.warning(f"There are only {self.size} gridpoints in parameter space but you are requesting {nsamples} samples of them. They will be over-sampled")
        self.params = np.moveaxis(self.params, 0, -1)

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
        pars = self.params[idx]
        rv = {nn: pp for nn, pp in zip(self.names, pars)}
        return rv
        
    def param_dict_for_lhsnumber(self, lhsnum):
        idx = self.lhsnumber_to_index(lhsnum)
        pars = self.params[idx]
        rv = {nn: pp for nn, pp in zip(self.names, pars)}
        return rv
        
    def params_for_number(self, num):
        idx = self.number_to_index(num)
        pars = self.params[idx]
        return pars

    def params_for_lhsnumber(self, lhsnum):
        idx = self.lhsnumber_to_index(lhsnum)
        pars = self.params[idx]
        return pars

    def sam_for_number(self, num):
        params = self.params_for_number(num)

        # gsmf_phi0, mmb_amp, mmb_plaw = params
        time, mmb_amp, mmb_plaw = params

        gsmf = holo.sam.GSMF_Schechter() #phi0=gsmf_phi0)
        gpf = holo.sam.GPF_Power_Law()
        gmt = holo.sam.GMT_Power_Law()
        mmbulge = holo.relations.MMBulge_KH2013(mamp=mmb_amp*MSOL, mplaw=mmb_plaw)

        sam = holo.sam.Semi_Analytic_Model(gsmf=gsmf, gpf=gpf, gmt=gmt, mmbulge=mmbulge)
        hard = holo.evolution.Fixed_Time.from_sam(sam, time*GYR, exact=True, progress=False)
        return sam, hard

    def sam_for_lhsnumber(self, lhsnum):
        params = self.params_for_lhsnumber(lhsnum)

        # gsmf_phi0, mmb_amp, mmb_plaw = params
        time, mmb_amp, mmb_plaw = params

        gsmf = holo.sam.GSMF_Schechter() #phi0=gsmf_phi0)
        gpf = holo.sam.GPF_Power_Law()
        gmt = holo.sam.GMT_Power_Law()
        mmbulge = holo.relations.MMBulge_KH2013(mamp=mmb_amp*MSOL, mplaw=mmb_plaw)

        sam = holo.sam.Semi_Analytic_Model(gsmf=gsmf, gpf=gpf, gmt=gmt, mmbulge=mmbulge)
        hard = holo.evolution.Fixed_Time.from_sam(sam, time*GYR, exact=True, progress=False)
        return sam, hard        
        

comm = MPI.COMM_WORLD

BEG = datetime.now()

# DEBUG = False

# ---- Fail on warnings
# # err = 'ignore'
# err = 'raise'
# np.seterr(divide=err, invalid=err, over=err)
# warn_err = 'error'
# # warnings.filterwarnings(warn_err, category=UserWarning)
# warnings.filterwarnings(warn_err)

# ---- Setup ArgParse

parser = argparse.ArgumentParser()
parser.add_argument('output', metavar='output', type=str,
                    help='output path [created if doesnt exist]')
parser.add_argument('-n', '--nsamples', action='store', dest='nsamples', type=int, help='number of parameter space samples, must be square of prime', default=25)
# parser.add_argument('-r', '--reals', action='store', dest='reals', type=int,
#                     help='number of realizations', default=10)
# parser.add_argument('-s', '--shape', action='store', dest='shape', type=int,
#                     help='shape of SAM grid', default=50)
# parser.add_argument('-t', '--threshold', action='store', dest='threshold', type=float,
#                     help='sample threshold', default=100.0)
# parser.add_argument('-d', '--dur', action='store', dest='dur', type=float,
#                     help='PTA observing duration [yrs]', default=20.0)
# parser.add_argument('-c', '--cad', action='store', dest='cad', type=float,
#                     help='PTA observing cadence [yrs]', default=0.1)
# parser.add_argument('-d', '--debug', action='store_true', default=False, dest='debug',
#                     help='run in DEBUG mode')
parser.add_argument('-t', '--test', action='store_true', default=False, dest='test',
                    help='Do not actually run, just output what parameters would have been done.')
parser.add_argument('-c', '--concatenate', action='store_true', default=False, dest='concatenateoutput', help='Concatenate output into single hdf5 file.')
parser.add_argument('-v', '--verbose', action='store_true', default=False, dest='verbose',
                    help='verbose output [INFO]')
# parser.add_argument('--version', action='version', version='%(prog)s 1.0')

args = parser.parse_args()
args.NUM_REALS = 100
args.PTA_DUR = 15.0 * YR
args.PTA_CAD = 0.1 * YR


BEG = comm.bcast(BEG, root=0)

this_fname = os.path.abspath(__file__)
head = f"holodeck :: {this_fname} : {str(BEG)} - rank: {comm.rank}/{comm.size}"
head = "\n" + head + "\n" + "=" * len(head) + "\n"
if comm.rank == 0:
    print(head)

log_name = f"holodeck__gen_lib_sams_{BEG.strftime('%Y%m%d-%H%M%S')}"
if comm.rank > 0:
    log_name = f"_{log_name}_r{comm.rank}"

PATH_OUTPUT = Path(args.output).resolve()
if not PATH_OUTPUT.is_absolute:
    PATH_OUTPUT = Path('.').resolve() / PATH_OUTPUT
    PATH_OUTPUT = PATH_OUTPUT.resolve()

if comm.rank == 0:
    PATH_OUTPUT.mkdir(parents=True, exist_ok=True)

comm.barrier()

# ---- Setup Logger ----

fname = f"{PATH_OUTPUT.joinpath(log_name)}.log"
log_lvl = holo.logger.INFO if args.verbose else holo.logger.WARNING
tostr = sys.stdout if comm.rank == 0 else False
# LOG = holo.logger.get_logger(name=log_name, level_stream=log_lvl, tofile=fname, tostr=tostr)
LOG = holo.logger.get_logger(name=log_name, level_stream=log_lvl, tofile=fname, tostr=tostr)
LOG.info(head)
LOG.info(f"Output path: {PATH_OUTPUT}")
LOG.info(f"        log: {fname}")

SPACE = Parameter_Space(nsamples=args.nsamples) if comm.rank == 0 else None
SPACE = comm.bcast(SPACE, root=0)

# ------------------------------------------------------------------------------
# ----    Methods
# ------------------------------------------------------------------------------


def main():
    bnum = 0
    LOG.info(f"{SPACE=}, {id(SPACE)=}")
    #npars = SPACE.size
    npars = args.nsamples
    nreals = args.NUM_REALS

    # # -- Load Parameters from Input File
    # params = None
    # if comm.rank == 0:
    #     input_file = os.path.abspath(input_file)
    #     if not os.path.isfile(input_file):
    #         raise ValueError(f"input_file '{input_file}' does not exist!")

    #     if not os.path.isdir(output_path):
    #         raise ValueError(f"output_path '{output_path}' does not exist!")

    #     params = _parse_params_file(input_file)

    #     # Copy input file to output directory
    #     fname_input_copy = os.path.join(output_path, "input_params.txt")
    #     # If file already exists, rename it to backup
    #     fname_backup = zio.modify_exists(fname_input_copy)
    #     if fname_input_copy != fname_backup:
    #         print(f"Moving previous parameters file '{fname_input_copy}' ==> '{fname_backup}'")
    #         shutil.move(fname_input_copy, fname_backup)
    #     print(f"Saving copy of input parameters file to '{fname_input_copy}'")
    #     shutil.copy2(input_file, fname_input_copy)

    # Distribute all parameters to all processes
    # params = comm.bcast(params, root=0)
    bnum = _barrier(bnum)

    # Split and distribute index numbers to all processes
    if comm.rank == 0:
        # indices = range(npars*nreals)
        indices = range(npars)
        indices = np.random.permutation(indices)
        indices = np.array_split(indices, comm.size)
        num_ind_per_proc = [len(ii) for ii in indices]
        # LOG.info(f"{npars=}, {nreals=}, total={npars*nreals} || ave runs per core = {np.mean(num_ind_per_proc)}")
        LOG.info(f"{npars=}, {nreals=} || avg runs per core = {np.mean(num_ind_per_proc)}")
    else:
        indices = None
    indices = comm.scatter(indices, root=0)

    bnum = _barrier(bnum)
    # prog_flag = (comm.rank == 0)
    iterator = holo.utils.tqdm(indices) if comm.rank == 0 else np.atleast_1d(indices)
    if args.test:
        LOG.info("Running in testing mode. Outputting parameters:")
    if comm.rank == 0:
        fname = f"parspaceobj.pickle"
        fname = os.path.join(PATH_OUTPUT, fname)
        if os.path.exists(fname):
            LOG.warning(f"File {fname} already exists.")

        with open(fname, 'wb') as fp:
            pickle.dump(SPACE, fp)
    for ind in iterator:
        # Convert from 1D index into 2D (param, real) specification
        # param, real = np.unravel_index(ind, (npars, nreals))
        # LOG.info(f"rank:{comm.rank} index:{ind} => {param=} {real=}")
        lhsparam = ind

        # # - Check if all output files already exist, if so skip
        # key = pipeline(progress=prog_flag, key_only=True, **pars)
        # if number_output:
        #     digits = int(np.floor(np.log10(999))) + 1
        #     key = f"{ind:0{digits:d}d}" + "__" + key

        # fname_plot_all, fname_plot_gwb = _save_plots_fnames(output_path, key)
        # fname_data = _save_data_fname(output_path, key)
        # fnames = [fname_plot_all, fname_plot_gwb, fname_data]
        # if np.all([os.path.exists(fn) and (os.path.getsize(fn) > 0) for fn in fnames]):
        #     print(f"\tkey: '{key}' already complete")
        #     continue
        if args.test:
            LOG.info(f"{comm.rank=} {ind=} {SPACE.param_dict_for_lhsnumber(lhsparam)}")
        else:
            try:
    	        run_sam(lhsparam, None, PATH_OUTPUT)
            except Exception as err:
                logging.warning(f"\n\nWARNING: error on rank:{comm.rank}, index:{ind}")
                logging.warning(err)
                LOG.warning(f"\n\nWARNING: error on rank:{comm.rank}, index:{ind}")
                LOG.warning(err)
                import traceback
                traceback.print_exc()
                print("\n\n")

    end = datetime.now()
    print(f"\t{comm.rank} done at {str(end)} after {str(end-BEG)} = {(end-BEG).total_seconds()}")
    LOG.info(f"\t{comm.rank} done at {str(end)} after {str(end-BEG)} = {(end-BEG).total_seconds()}")
    bnum = _barrier(bnum)
    if comm.rank == 0:
        end = datetime.now()
        tail = f"Done at {str(end)} after {str(end-BEG)} = {(end-BEG).total_seconds()}"
        print("\n" + "=" * len(tail) + "\n" + tail + "\n")

    return


def run_sam(pnum, real, path_output):

#     iterator = range(args.NUM_REALS)
#     if comm.rank == 0:
#         iterator = holo.utils.tqdm(iterator, leave=False)

    fname = f"lib_sams__p{pnum:06d}.npz"
    fname = os.path.join(path_output, fname)
    if os.path.exists(fname):
        LOG.warning(f"File {fname} already exists.")

    _fobs = holo.utils.nyquist_freqs(args.PTA_DUR, args.PTA_CAD)

    df = _fobs[0]/2
    fobs_edges = _fobs - df
    fobs_edges = np.concatenate([fobs_edges, [_fobs[-1] + df]])

    sam, hard = SPACE.sam_for_lhsnumber(pnum)
#         hard = holo.evolution.Hard_GW()
#         vals, weights, edges, dens, mass = holo.sam.sample_sam_with_hardening(sam, hard, fobs=fobs, sample_threshold=1e2, poisson_inside=True, poisson_outside=True)
#         gff, gwf, gwb = holo.gravwaves._gws_from_samples(vals, weights, fobs)
    gwbspec = sam.gwb(fobs_edges, realize=args.NUM_REALS, hard=hard)
    legend = SPACE.param_dict_for_lhsnumber(pnum)
    np.savez(fname, fobs=_fobs, fobs_edges=fobs_edges, gwbspec=gwbspec, pnum=pnum, nreals=args.NUM_REALS, fullparspace=SPACE, **legend)
    LOG.info(f"Saved to {fname} after {(datetime.now()-BEG)} (start: {BEG})")

    return

def concatenate_outputs():
    regex = "lib_sams__p*.npz"
    files = sorted(PATH_OUTPUT.glob(regex))
    num_files = len(files)
    print(PATH_OUTPUT, f"\n\texists={PATH_OUTPUT.exists()}", f"\n\tfound {num_files} files")

    all_exist = True
    for ii in range(num_files):
        temp = PATH_OUTPUT.joinpath(regex.replace('*', f"{ii:06d}"))
        exists = temp.exists()
        # print(f"{ii:4d}, {temp.name}, {exists=}")
        if not exists:
            all_exist = False
            break
    print(f"All files exist?  {all_exist}")     
    # Read in parameter space object
    picklefname = f"parspaceobj.pickle"
    picklefname = os.path.join(PATH_OUTPUT, picklefname)
    with open(picklefname, 'rb') as fp:
        space = pickle.load(fp)
    shape = space.shape
    # Check on one data file
    temp = files[0]
    data = np.load(temp)
    fobs = data['fobs']
    fobs_edges = data['fobs_edges']
    nreals = data['nreals'][()]
    nfreqs = fobs.size
    temp_gwb = data['gwbspec']
    
    assert np.ndim(temp_gwb) == 2
    assert temp_gwb.shape[0] == nfreqs
    assert temp_gwb.shape[1] == nreals
    
    gwb_shape = list(shape) + [nfreqs, nreals,]
    names = space.names + ['freqs', 'reals']
    gwb = np.zeros(gwb_shape)
    
    for ii, file in enumerate(files):
        temp = np.load(file)
        assert np.allclose(fobs, temp['fobs'])
        assert np.allclose(fobs_edges, temp['fobs_edges'])
        pars = [pp[()] for pp in [temp['times'], temp['mmb_amp'], temp['mmb_plaw']]]
        
        idx = space.lhsnumber_to_index(ii)
        space_pars = space.params[idx]
        assert np.allclose(pars, space_pars)
        
        gwb[idx] = temp['gwbspec'][:]
    print(holo.utils.stats(gwb))
    print(holo.utils.frac_str(gwb > 0.0))
    
    out_filename = PATH_OUTPUT.joinpath('sam_lib.hdf5')
    import h5py
    with h5py.File(out_filename, 'w') as h5:
        h5.create_dataset('params', data=space.params)
        h5.create_dataset('fobs', data=fobs)
        h5.create_dataset('fobs_edges', data=fobs_edges)
        h5.create_dataset('gwb', data=gwb)
        h5.create_dataset('names', data=names)

    print(f"Saved to {out_filename}, size: {holo.utils.get_file_size(out_filename)}")
    return

def _barrier(bnum):
    LOG.debug(f"barrier {bnum}")
    comm.barrier()
    bnum += 1
    return bnum


if __name__ == "__main__":
    np.seterr(divide='ignore', invalid='ignore', over='ignore')
    warnings.filterwarnings("ignore", category=UserWarning)
    if args.concatenateoutput == True:
        if comm.rank == 0:
            concatenate_outputs()
    else:
        main()
    sys.exit(0)
