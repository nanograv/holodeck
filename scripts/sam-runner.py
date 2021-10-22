"""
"""

import os
from datetime import datetime
# import glob
import logging
# import sys
import warnings

import numpy as np
# import h5py
from mpi4py import MPI

import holodeck as holo
import holodeck.sam
from holodeck.constants import YR, MSOL, GYR
from holodeck import log

comm = MPI.COMM_WORLD

PATH_OUTPUT = "./output"
BEG = datetime.now()

NUM_REALS = 30
NUM_PARAMS = 10


def run_sam(param, real, path_output):
    beg = datetime.now()

    fname = f"sam_output_p{param:03d}_r{real:03d}.npz"
    fname = os.path.join(path_output, fname)
    if os.path.exists(fname):
        log.warning(f"File {fname} already exists.")
        return

    fobs = holo.utils.nyquist_freqs(20.0*YR, 0.1*YR)

    mmbulge_mass_norm = np.logspace(7, 9, NUM_PARAMS) * MSOL
    mmbulge_mass_norm = mmbulge_mass_norm[param]
    log.info(f"{comm.rank=} {param=} {real=} {mmbulge_mass_norm=:.4e} [g] = {mmbulge_mass_norm/MSOL=:.4e}")

    gsmf = holo.sam.GSMF_Schechter()        # Galaxy Stellar-Mass Function (GSMF)
    gpf = holo.sam.GPF_Power_Law()          # Galaxy Pair Fraction         (GPF)
    gmt = holo.sam.GMT_Power_Law()          # Galaxy Merger Time           (GMT)
    mmbulge = holo.sam.MMBulge_Simple(mmbulge_mass_norm)     # M-MBulge Relation            (MMB)

    sam = holo.sam.Semi_Analytic_Model(gsmf=gsmf, gpf=gpf, gmt=gmt, mmbulge=mmbulge)
    gff, gwf, gwb = holo.sam.sampled_gws_from_sam(
        sam, fobs, sample_threshold=10.0,
        cut_below_mass=1e7*MSOL, limit_merger_time=4*GYR
    )

    np.savez(fname, fobs=fobs, gff=gff, gwb=gwb, gwf=gwf, mmbulge_norm=mmbulge_mass_norm)
    log.warning(f"Saved to {fname} after {(datetime.now()-beg)} (start: {BEG})")
    return


def main():

    bnum = 0

    # Print header (filename etc)
    beg = datetime.now()
    if comm.rank == 0:
        this_fname = os.path.abspath(__file__)
        head = f"{this_fname} : {str(beg)} - rank: {comm.rank}/{comm.size}"
        print("\n" + head + "\n" + "=" * len(head) + "\n")

    path_output = os.path.join(os.path.abspath(PATH_OUTPUT), '')
    if comm.rank == 0:
        if not os.path.exists(path_output):
            os.mkdir(path_output)

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
    bnum = _barrier(bnum, verbose=True)

    # Split and distribute index numbers to all processes
    if comm.rank == 0:
        indices = range(NUM_PARAMS*NUM_REALS)
        indices = np.random.permutation(indices)
        indices = np.array_split(indices, comm.size)
    else:
        indices = None
    indices = comm.scatter(indices, root=0)
    bnum = _barrier(bnum, verbose=True)
    # prog_flag = (comm.rank == 0)

    for ind in np.atleast_1d(indices):
        # pars = params[ind]
        # pars = ind
        # Convert from 1D index into 2D (param, real) specification
        param, real = np.unravel_index(ind, (NUM_PARAMS, NUM_REALS))

        print(f"rank:{comm.rank} index:{ind} => {param=} {real=}")

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

        try:
            run_sam(param, real, path_output)
        except Exception as err:
            logging.warning(f"\n\nWARNING: error on rank:{comm.rank}, index:{ind}")
            logging.warning(err)
            import traceback
            traceback.print_exc()
            print("\n\n")

    end = datetime.now()
    print(f"\t{comm.rank} done at {str(end)} after {str(end-beg)} = {(end-beg).total_seconds()}")
    bnum = _barrier(bnum, verbose=True)
    if comm.rank == 0:
        end = datetime.now()
        tail = f"Done at {str(end)} after {str(end-beg)} = {(end-beg).total_seconds()}"
        print("\n" + "=" * len(tail) + "\n" + tail + "\n")

    return


def _barrier(bnum, verbose=True):
    if comm.rank == 0:
        print(f"barrier {bnum}")
        bnum += 1

    comm.barrier()
    return bnum


if __name__ == "__main__":
    np.seterr(divide='ignore', invalid='ignore', over='ignore')
    warnings.filterwarnings("ignore", category=UserWarning)

    # args = sys.argv[1:]
    # if len(args) != 2:
    #     print("Incorrect usage!")
    #     comm.Abort(2)

    # output_path = sys.argv[3]

    # print(f"rank {comm.rank} : args = '{args}'")
    main()
