# """Library Generation Script for Semi-Analytic Models.

# Usage
# -----

# mpirun -n <NPROCS> python ./scripts/gen_lib_ss.py <PARAM_SPACE> <PATH> -n <SAMPS> -r <REALS> -f <FREQS> -s <SHAPE> -l <LOUDEST> -p <PARS>

#     <PARAM_SPACE> : name of the parameter space class (in holodeck.param_spaces) to run; must match exactly.
#     <NPROCS> : number of processors to run on
#     <PATH> : output directory to save data to
#     <SAMPS> : number of parameter-space samples for latin hyper-cube
#     <REALS> : number of realizations at each parameter-space location
#     <FREQS> : number of frequencies (multiples of PTA observing baseline)
#     <SHAPE> : SAM grid shape, as a single int value (applied to all dimensions)
#     <LOUDEST> : number of loudest single source to separate
#     <PARS> : Whether or not calculate sspar and bgpar, 1 for true, 0 for false

# Example:

#     mpirun -n 8 python ./scripts/gen_lib_ss.py PS_Broad_Uniform_02B output/2022-12-05_01 -n 32 -r 10 -f 20 -s 80 -l 5 -p 1


# To-Do
# -----
# * mark output directories as incomplete until all runs have been finished.
#   Merged libraries from incomplete directories should also get some sort of flag!

# """

# __version__ = '0.2.2'

# import argparse
# import os
# import shutil
# import sys
# import warnings
# from datetime import datetime
# from pathlib import Path

# import numpy as np
# # import matplotlib.pyplot as plt
# from mpi4py import MPI

# import holodeck as holo
# import holodeck.sam
# import holodeck.single_sources as ss
# import holodeck.logger
# # from holodeck.constants import YR
# from holodeck import log as _log     #: import the default holodeck log just so that we can silence it
# # silence default holodeck log
# _log.setLevel(_log.WARNING)

# # Default argparse parameters
# LIB_PATH = '/Users/emigardiner/GWs/holodeck/output/2023-05-12-mbp-ss16_n10_r10_f70_d12.5_l10_p0'
# DS_NUM = '2B'
# NPSRS = 60
# SIGMA = 5e-7

# NSKIES = 25
# THRESH = 0.5
# PLOT = True
# DEBUG = True
# nsamp = 10 # read this from hdf file

# MAX_FAILURES = 5

# comm = MPI.COMM_WORLD

# FILES_COPY_TO_OUTPUT = [__file__, holo.librarian.__file__, holo.param_spaces.__file__]


# def main():

#     args, space, log = setup_basics()
#     comm.barrier()
    

#     # Split and distribute index numbers to all processes
#     if comm.rank == 0:
#         npars = args.nsamples
#         indices = range(npars)
#         indices = np.random.permutation(indices)
#         indices = np.array_split(indices, comm.size)
#         num_ind_per_proc = [len(ii) for ii in indices]
#         log.info(f"{npars=} cores={comm.size} || max runs per core = {np.max(num_ind_per_proc)}")
#     else:
#         indices = None

#     indices = comm.scatter(indices, root=0)

#     iterator = holo.utils.tqdm(indices) if (comm.rank == 0) else np.atleast_1d(indices)

#     if comm.rank == 0:
#         space_fname = space.save(args.output)
#         log.info(f"saved parameter space {space} to {space_fname}")

#     comm.barrier()
#     beg = datetime.now()
#     log.info(f"beginning tasks at {beg}")
#     failures = 0

#     for par_num in iterator:
#         log.info(f"{comm.rank=} {par_num=}")
#         pdict = space.param_dict(par_num)
#         msg = "\n"
#         for kk, vv in pdict.items():
#             msg += f"{kk}={vv}\n"
#         log.info(msg)

#         rv = holo.librarian.run_ss_at_pspace_num(args, space, par_num)
#         if rv is False:
#             failures += 1

#         if failures > MAX_FAILURES:
#             err = f"Failed {failures} times on rank:{comm.rank}!"
#             log.exception(err)
#             raise RuntimeError(err)

#     end = datetime.now()
#     dur = (end - beg)
#     log.info(f"\t{comm.rank} done at {str(end)} after {str(dur)} = {dur.total_seconds()}")

#     # Make sure all processes are done so that all files are ready for merging
#     comm.barrier()

#     if (comm.rank == 0):
#         log.info("Concatenating outputs into single file")
#         holo.librarian.ss_lib_combine(args.output, log, bool(args.get_pars), path_sims=args.output_sims)
#         log.info("Concatenating completed")

#     return


# def setup_basics():
#     if comm.rank == 0:
#         args = _setup_argparse()
#     else:
#         args = None

#     # share `args` to all processes
#     args = comm.bcast(args, root=0)

#     # setup log instance, separate for all processes
#     log = _setup_log(args)
#     args.log = log

#     if comm.rank == 0:
#         # copy certain files to output directory
#         for fname in FILES_COPY_TO_OUTPUT:
#             src_file = Path(fname)
#             dst_file = args.output.joinpath("runtime_" + src_file.name)
#             shutil.copyfile(src_file, dst_file)
#             log.info(f"Copied {fname} to {dst_file}")

#         # setup parameter-space instance
#         try:
#             # `param_space` attribute must match the name of one of the classes in `holo.param_spaces`
#             space = getattr(holo.param_spaces, args.param_space)
#         except Exception as err:
#             log.exception(f"Failed to load '{args.param_space}' from holo.param_spaces!")
#             log.exception(err)
#             raise err

#         # instantiate the parameter space class
#         space = space(log, args.nsamples, args.sam_shape, args.seed)
#     else:
#         space = None

#     # share parameter space across processes
#     space = comm.bcast(space, root=0)

#     log.info(
#         f"param_space={args.param_space}, samples={args.nsamples}, sam_shape={args.sam_shape}, nreals={args.nreals}\n"
#         f"nfreqs={args.nfreqs}, pta_dur={args.pta_dur} [yr]\n"
#     )

#     return args, space, log


# def _setup_argparse(*args, **kwargs):
#     assert comm.rank == 0

#     parser = argparse.ArgumentParser()
#     parser.add_argument('ssdir', metavar='ssdir', type=str,
#                         help='ss output path [must already exist, contains ss data]')

#     parser.add_argument('-p', '--npsrs', action='store', dest='npsrs', type=int, default=40,
#                         help='number of pulsars', default=NPSRS)
#     parser.add_argument('--sigma', action='store', dest='sigma', type=int,
#                         help='sigma of pulsars', default=SIGMA)
#     parser.add_argument('--nskies', action='store', dest='nskies', type=int,
#                         help='Number of sky realizations', default=NSKIES)
                        
#     parser.add_argument('--plot', action='store_true', default=False,
#                         help='produce plots for each simulation configuration')

#     args = parser.parse_args(*args, **kwargs)

#     ssdir = Path(args.ssdir).resolve()
#     if not ssdir.is_absolute:
#         ssdir = Path('.').resolve() / ssdir
#         ssdir = ssdir.resolve()

#     assert ssdir.exists
#     my_print(f"ssdir path: {ssdir}")
#     # args.output = output

#     # output_sims = output.joinpath("sims")
#     # output_sims.mkdir(parents=True, exist_ok=True)
#     # args.output_sims = output_sims

#     # output_logs = output.joinpath("logs")
#     # output_logs.mkdir(parents=True, exist_ok=True)
#     # args.output_logs = output_logs

#     # if args.plot:
#     #     output_plots = output.joinpath("figs")
#     #     output_plots.mkdir(parents=True, exist_ok=True)
#     #     args.output_plots = output_plots

#     return args


# def _setup_log(args):
#     beg = datetime.now()
#     log_name = f"holodeck__gen_lib_ss_{beg.strftime('%Y%m%d-%H%M%S')}"
#     if comm.rank > 0:
#         log_name = f"_{log_name}_r{comm.rank}"

#     output = args.output_logs
#     fname = f"{output.joinpath(log_name)}.log"
#     log_lvl = holo.logger.INFO if args.verbose else holo.logger.WARNING
#     tostr = sys.stdout if comm.rank == 0 else False
#     log = holo.logger.get_logger(name=log_name, level_stream=log_lvl, tofile=fname, tostr=tostr)
#     log.info(f"Output path: {output}")
#     log.info(f"        log: {fname}")
#     log.info(args)
#     return log


# def my_print(*args, **kwargs):
#     return print(*args, flush=True, **kwargs)


# def mpiabort_excepthook(type, value, traceback):
#     sys.__excepthook__(type, value, traceback)
#     comm.Abort()
#     return


# if __name__ == "__main__":
#     np.seterr(divide='ignore', invalid='ignore', over='ignore')
#     warnings.filterwarnings("ignore", category=UserWarning)
#     sys.excepthook = mpiabort_excepthook
#     beg_time = datetime.now()
#     beg_time = comm.bcast(beg_time, root=0)

#     if comm.rank == 0:
#         this_fname = os.path.abspath(__file__)
#         head = f"holodeck :: {this_fname} : {str(beg_time)} - rank: {comm.rank}/{comm.size}"
#         head = "\n" + head + "\n" + "=" * len(head) + "\n"
#         my_print(head)

#     main()

#     if comm.rank == 0:
#         end = datetime.now()
#         dur = end - beg_time
#         tail = f"Done at {str(end)} after {str(dur)} = {dur.total_seconds()}"
#         my_print("\n" + "=" * len(tail) + "\n" + tail + "\n")

#     sys.exit(0)
