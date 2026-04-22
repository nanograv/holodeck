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
* LHS (at least with pydoe) is not deterministic (i.e. not reproducible).  Find a way to make reproducible.
* Use LHS to choose parameters themselves, instead of grid-points.  Also remove usage of grid entirely.
    * Does this resolve irregularities between different LHS implementations?
* Use subclassing to cleanup `Parameter_Space` object.  e.g. implement LHS as subclass of generic Parameter_Space class.
* BUG: `lhs_grid` and `lhs_grid_idx` are currently storing the same thing
* #! IMPORTANT: mark output directories as incomplete until all runs have been finished.  Merged libraries from incomplete directories should also get some sort of flag! !#

"""

__version__ = '0.2.0'

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
from holodeck.constants import YR, MSOL, GYR, PC
from holodeck import log as _log     #: import the default holodeck log just so that we can silence it

# from scipy.stats import qmc
# import pyDOE

# silence default holodeck log
_log.setLevel(_log.WARNING)

# Default argparse parameters
DEF_SAM_SHAPE = 50
DEF_NUM_REALS = 100
DEF_NUM_FBINS = 40
DEF_PTA_DUR = 16.03     # [yrs]

DEF_ECCEN_NUM_STEPS = 123
DEF_ECCEN_NHARMS = 100


class Parameter_Space_Mix01(holo.librarian._Parameter_Space):

    _PARAM_NAMES = [
        'gsmf_phi0',
        'time',
        'gpf_qgamma',
        'hard_gamma_inner',
        'mmb_amp',
        'mmb_plaw'
    ]

    def __init__(self, log, nsamples, sam_shape):
        super().__init__(
            log, nsamples, sam_shape,
            gsmf_phi0=[-3.28, -2.16, 5],
            time=[-2.0, +1.0, 7],   # [log10(Gyr)]
            gpf_qgamma=[-0.4, +0.4, 5],
            hard_gamma_inner=[-1.5, -0.5, 5],
            mmb_amp=[0.1e9, 1.0e9, 9],
            mmb_plaw=[0.8, 1.5, 11],
        )

    def sam_for_lhsnumber(self, lhsnum):
        param_grid = self.params_for_lhsnumber(lhsnum)

        gsmf_phi0, time, gpf_qgamma, hard_gamma_inner, mmb_amp, mmb_plaw = param_grid
        time = (10.0 ** time) * GYR
        mmb_amp = mmb_amp*MSOL

        gsmf = holo.sam.GSMF_Schechter(phi0=gsmf_phi0)
        gpf = holo.sam.GPF_Power_Law(qgamma=gpf_qgamma)
        gmt = holo.sam.GMT_Power_Law()
        mmbulge = holo.host_relations.MMBulge_KH2013(mamp=mmb_amp, mplaw=mmb_plaw)

        sam = holo.sam.Semi_Analytic_Model(gsmf=gsmf, gpf=gpf, gmt=gmt, mmbulge=mmbulge, shape=self.sam_shape)
        hard = holo.hardening.Fixed_Time.from_sam(sam, time, gamma_sc=hard_gamma_inner, exact=True, progress=False)
        return sam, hard


class Parameter_Space_Hard01(holo.librarian._Parameter_Space):

    _PARAM_NAMES = [
        'hard_time',
        'hard_gamma_inner',
        'hard_gamma_outer',
        'hard_rchar',
    ]

    def __init__(self, log, nsamples, sam_shape):
        super().__init__(
            log, nsamples, sam_shape,
            hard_time=[-1.0, +1.0, 7],   # [log10(Gyr)]
            hard_gamma_inner=[-1.5, -0.5, 7],
            hard_gamma_outer=[+1.0, +3.0, 7],
            hard_rchar=[1.0, 3.0, 5],
        )

    def sam_for_lhsnumber(self, lhsnum):
        param_grid = self.params_for_lhsnumber(lhsnum)

        time, gamma_inner, gamma_outer, rchar = param_grid
        time = (10.0 ** time) * GYR
        rchar = (10.0 ** rchar) * PC

        gsmf = holo.sam.GSMF_Schechter()
        gpf = holo.sam.GPF_Power_Law()
        gmt = holo.sam.GMT_Power_Law()
        mmbulge = holo.host_relations.MMBulge_KH2013()

        sam = holo.sam.Semi_Analytic_Model(
            gsmf=gsmf, gpf=gpf, gmt=gmt, mmbulge=mmbulge,
            shape=self.sam_shape
        )
        hard = holo.hardening.Fixed_Time.from_sam(
            sam, time, rchar=rchar, gamma_sc=gamma_inner, gamma_df=gamma_outer,
            exact=True, progress=False
        )
        return sam, hard


class Parameter_Space_Hard02_BAD(holo.librarian._Parameter_Space):

    _PARAM_NAMES = [
        'hard_time',
        'hard_gamma_inner',
        'hard_gamma_outer',
        'hard_rchar',
        'gsmf_phi0',
        'mmb_amp',
    ]

    def __init__(self, log, nsamples, sam_shape):
        super().__init__(
            log, nsamples, sam_shape,
            hard_time=[-1.0, +1.0, 7],   # [log10(Gyr)]
            hard_gamma_inner=[-1.5, -0.5, 7],
            hard_gamma_outer=[+1.0, +3.0, 7],
            hard_rchar=[1.0, 3.0, 5],
            gsmf_phi0=[-3.06, -2.5, 3],
            mmb_amp=[0.39e9, 0.61e9, 3],
        )

    def sam_for_lhsnumber(self, lhsnum):
        param_grid = self.params_for_lhsnumber(lhsnum)

        time, gamma_inner, gamma_outer, rchar, gsmf_phi0, mmb_amp = param_grid
        time = (10.0 ** time) * GYR
        rchar = (10.0 ** rchar) * PC
        mmb_amp = mmb_amp*MSOL

        gsmf = holo.sam.GSMF_Schechter(phi0=gsmf_phi0)
        gpf = holo.sam.GPF_Power_Law()
        gmt = holo.sam.GMT_Power_Law()
        mmbulge = holo.host_relations.MMBulge_KH2013(mamp=mmb_amp)

        sam = holo.sam.Semi_Analytic_Model(
            gsmf=gsmf, gpf=gpf, gmt=gmt, mmbulge=mmbulge,
            shape=self.sam_shape
        )
        hard = holo.hardening.Fixed_Time.from_sam(
            sam, time, rchar=rchar, gamma_sc=gamma_inner, gamma_df=gamma_outer,
            exact=True, progress=False
        )
        return sam, hard


class Parameter_Space_Hard03(holo.librarian._Parameter_Space):

    _PARAM_NAMES = [
        'hard_time',
        'hard_gamma_inner',
        'hard_gamma_outer',
        'hard_rchar',
        'gsmf_phi0',
        'mmb_amp',
    ]

    def __init__(self, log, nsamples, sam_shape):
        super().__init__(
            log, nsamples, sam_shape,
            hard_time=[-1.0, +1.0, 5],   # [log10(Gyr)]
            hard_gamma_inner=[-1.5, -0.5, 5],
            hard_gamma_outer=[+2.0, +3.0, 5],
            hard_rchar=[1.0, 3.0, 5],
            gsmf_phi0=[-3.06, -2.5, 3],
            mmb_amp=[0.39e9, 0.61e9, 3],
        )

    def sam_for_lhsnumber(self, lhsnum):
        param_grid = self.params_for_lhsnumber(lhsnum)

        time, gamma_inner, gamma_outer, rchar, gsmf_phi0, mmb_amp = param_grid
        time = (10.0 ** time) * GYR
        rchar = (10.0 ** rchar) * PC
        mmb_amp = mmb_amp*MSOL

        gsmf = holo.sam.GSMF_Schechter(phi0=gsmf_phi0)
        gpf = holo.sam.GPF_Power_Law()
        gmt = holo.sam.GMT_Power_Law()
        mmbulge = holo.host_relations.MMBulge_KH2013(mamp=mmb_amp)

        sam = holo.sam.Semi_Analytic_Model(
            gsmf=gsmf, gpf=gpf, gmt=gmt, mmbulge=mmbulge,
            shape=self.sam_shape
        )
        hard = holo.hardening.Fixed_Time.from_sam(
            sam, time, rchar=rchar, gamma_sc=gamma_inner, gamma_df=gamma_outer,
            exact=True, progress=False
        )
        return sam, hard


class Parameter_Space_Hard04(holo.librarian._Parameter_Space):

    _PARAM_NAMES = [
        'hard_time',
        'hard_gamma_inner',
        'hard_gamma_outer',
        'hard_rchar',
        'gsmf_phi0',
        'mmb_amp',
    ]

    def __init__(self, log, nsamples, sam_shape):
        super().__init__(
            log, nsamples, sam_shape,
            hard_time=[-1.0, +1.0, 5],   # [log10(Gyr)]
            hard_gamma_inner=[-1.5, -0.5, 5],
            hard_gamma_outer=[+2.0, +3.0, 5],
            hard_rchar=[1.0, 3.0, 5],
            gsmf_phi0=[-3.0, -2.0, 5],
            mmb_amp=[0.1e9, 1.0e9, 5],
        )

    def sam_for_lhsnumber(self, lhsnum):
        param_grid = self.params_for_lhsnumber(lhsnum)

        time, gamma_inner, gamma_outer, rchar, gsmf_phi0, mmb_amp = param_grid
        time = (10.0 ** time) * GYR
        rchar = (10.0 ** rchar) * PC
        mmb_amp = mmb_amp*MSOL

        gsmf = holo.sam.GSMF_Schechter(phi0=gsmf_phi0)
        gpf = holo.sam.GPF_Power_Law()
        gmt = holo.sam.GMT_Power_Law()
        mmbulge = holo.host_relations.MMBulge_KH2013(mamp=mmb_amp)

        sam = holo.sam.Semi_Analytic_Model(
            gsmf=gsmf, gpf=gpf, gmt=gmt, mmbulge=mmbulge,
            shape=self.sam_shape
        )
        hard = holo.hardening.Fixed_Time.from_sam(
            sam, time, rchar=rchar, gamma_sc=gamma_inner, gamma_df=gamma_outer,
            exact=True, progress=False
        )
        return sam, hard


class Parameter_Space_Hard04b(Parameter_Space_Hard04):

    def __init__(self, log, nsamples, sam_shape):
        grid_size = 100
        super(Parameter_Space_Hard04, self).__init__(
            log, nsamples, sam_shape,
            hard_time=[-1.0, +1.0, grid_size],   # [log10(Gyr)]
            hard_gamma_inner=[-1.5, -0.5, grid_size],
            hard_gamma_outer=[+2.0, +3.0, grid_size],
            hard_rchar=[1.0, 3.0, grid_size],
            gsmf_phi0=[-3.0, -2.0, grid_size],
            mmb_amp=[0.1e9, 1.0e9, grid_size],
        )


class Parameter_Space_Debug01(holo.librarian._Parameter_Space):

    _PARAM_NAMES = [
        'hard_time',
        'hard_rchar',
        'mmb_amp',
    ]

    def __init__(self, log, nsamples, sam_shape):
        super().__init__(
            log, nsamples, sam_shape,
            hard_time=[-1.0, +1.0, 3],   # [log10(Gyr)]
            hard_rchar=[1.0, 3.0, 3],
            mmb_amp=[0.1e9, 1.0e9, 100],
        )

    def sam_for_lhsnumber(self, lhsnum):
        param_grid = self.params_for_lhsnumber(lhsnum)

        time, rchar, mmb_amp = param_grid
        time = (10.0 ** time) * GYR
        rchar = (10.0 ** rchar) * PC
        mmb_amp = mmb_amp*MSOL

        gsmf_phi0 = -2.0
        gamma_inner = -1.0
        gamma_outer = +2.5
        gsmf = holo.sam.GSMF_Schechter(phi0=gsmf_phi0)
        gpf = holo.sam.GPF_Power_Law()
        gmt = holo.sam.GMT_Power_Law()
        mmbulge = holo.host_relations.MMBulge_KH2013(mamp=mmb_amp)

        sam = holo.sam.Semi_Analytic_Model(
            gsmf=gsmf, gpf=gpf, gmt=gmt, mmbulge=mmbulge,
            shape=self.sam_shape
        )
        hard = holo.hardening.Fixed_Time.from_sam(
            sam, time, rchar=rchar, gamma_sc=gamma_inner, gamma_df=gamma_outer,
            exact=True, progress=False
        )
        return sam, hard


class Parameter_Space_Debug01b(Parameter_Space_Debug01):

    _PARAM_NAMES = [
        'hard_time',
        'hard_rchar',
        'mmb_amp',
    ]

    def __init__(self, log, nsamples, sam_shape):
        super(Parameter_Space_Debug01).__init__(
            log, nsamples, sam_shape,
            hard_time=[-1.0, +1.0, 3],   # [log10(Gyr)]
            hard_rchar=[1.0, 3.0, 3],
            mmb_amp=[0.1e9, 1.0e9, 1000],
        )


class Parameter_Space_Simple01(holo.librarian._Parameter_Space):

    _PARAM_NAMES = [
        'hard_time',
        'hard_rchar',
        'gsmf_phi0',
        'mmb_amp',
    ]

    def __init__(self, log, nsamples, sam_shape):
        super().__init__(
            log, nsamples, sam_shape,
            hard_time=[-0.5, +1.0, 9],   # [log10(Gyr)]
            hard_rchar=[1.0, 2.5, 7],
            gsmf_phi0=[-3.0, -1.5, 7],
        )

    def sam_for_lhsnumber(self, lhsnum):
        param_grid = self.params_for_lhsnumber(lhsnum)

        time, rchar, gsmf_phi0 = param_grid
        time = (10.0 ** time) * GYR
        rchar = (10.0 ** rchar) * PC

        gsmf = holo.sam.GSMF_Schechter(phi0=gsmf_phi0)
        gpf = holo.sam.GPF_Power_Law()
        gmt = holo.sam.GMT_Power_Law()
        mmbulge = holo.host_relations.MMBulge_KH2013()

        sam = holo.sam.Semi_Analytic_Model(
            gsmf=gsmf, gpf=gpf, gmt=gmt, mmbulge=mmbulge,
            shape=self.sam_shape
        )
        hard = holo.hardening.Fixed_Time.from_sam(
            sam, time, rchar=rchar,
            exact=True, progress=False
        )
        return sam, hard


class LHS_Parameter_Space_Hard04(holo.librarian._LHS_Parameter_Space):

    _PARAM_NAMES = [
        'hard_time',
        'hard_gamma_inner',
        'hard_gamma_outer',
        'hard_rchar',
        'gsmf_phi0',
        'mmb_amp',
    ]

    def __init__(self, log, nsamples, sam_shape, lhs_sampler, seed):
        super().__init__(
            log, nsamples, sam_shape, lhs_sampler, seed,
            hard_time=[-1.0, +1.0],   # [log10(Gyr)]
            hard_gamma_inner=[-1.5, -0.5],
            hard_gamma_outer=[+2.0, +3.0],
            hard_rchar=[1.0, 3.0],
            gsmf_phi0=[-3.0, -2.0],
            mmb_amp=[0.1e9, 1.0e9],
        )

    def sam_for_lhsnumber(self, lhsnum):
        param_grid = self.params_for_lhsnumber(lhsnum)

        time, gamma_inner, gamma_outer, rchar, gsmf_phi0, mmb_amp = param_grid
        time = (10.0 ** time) * GYR
        rchar = (10.0 ** rchar) * PC
        mmb_amp = mmb_amp*MSOL

        gsmf = holo.sam.GSMF_Schechter(phi0=gsmf_phi0)
        gpf = holo.sam.GPF_Power_Law()
        gmt = holo.sam.GMT_Power_Law()
        mmbulge = holo.host_relations.MMBulge_KH2013(mamp=mmb_amp)

        sam = holo.sam.Semi_Analytic_Model(
            gsmf=gsmf, gpf=gpf, gmt=gmt, mmbulge=mmbulge,
            shape=self.sam_shape
        )
        hard = holo.evolution.Fixed_Time.from_sam(
            sam, time, rchar=rchar, gamma_sc=gamma_inner, gamma_df=gamma_outer,
            exact=True, progress=False
        )
        return sam, hard


class LHS_PSpace_Eccen_01(holo.librarian._LHS_Parameter_Space):

    _PARAM_NAMES = [
        'eccen_init',
        'gsmf_phi0',
        'gpf_zbeta',
        'mmb_amp',
    ]

    SEPA_INIT = 1.0 * PC

    def __init__(self, log, nsamples, sam_shape, lhs_sampler, seed):
        super().__init__(
            log, nsamples, sam_shape, lhs_sampler, seed,
            eccen_init=[0.0, +0.975],
            gsmf_phi0=[-3.0, -2.0],
            gpf_zbeta=[+0.0, +2.0],
            mmb_amp=[0.1e9, 1.0e9],
        )

    def sam_for_lhsnumber(self, lhsnum):
        param_grid = self.params_for_lhsnumber(lhsnum)

        eccen, gsmf_phi0, gpf_zbeta, mmb_amp = param_grid
        mmb_amp = mmb_amp*MSOL

        # favor higher values of eccentricity instead of uniformly distributed
        eccen = eccen ** (1.0/5.0)

        gsmf = holo.sam.GSMF_Schechter(phi0=gsmf_phi0)
        gpf = holo.sam.GPF_Power_Law(zbeta=gpf_zbeta)
        gmt = holo.sam.GMT_Power_Law()
        mmbulge = holo.host_relations.MMBulge_KH2013(mamp=mmb_amp)

        sam = holo.sam.Semi_Analytic_Model(
            gsmf=gsmf, gpf=gpf, gmt=gmt, mmbulge=mmbulge,
            shape=self.sam_shape
        )

        sepa_evo, eccen_evo = holo.sam.evolve_eccen_uniform_single(sam, eccen, self.SEPA_INIT, DEF_ECCEN_NUM_STEPS)

        return sam, sepa_evo, eccen_evo


class LHS_PSpace_Eccen_02(holo.librarian._LHS_Parameter_Space):

    _PARAM_NAMES = [
        'eccen_init',
        'gsmf_phi0',
        'gsmf_phiz',
        'gpf_malpha',
        'gpf_zbeta',
        'gpf_qgamma',

        'gmt_malpha',
        'gmt_zbeta',
        'gmt_qgamma',
        'mmb_amp',
        'mmb_plaw',
    ]

    SEPA_INIT = 1.0 * PC

    def __init__(self, log, nsamples, sam_shape, lhs_sampler, seed):
        super().__init__(
            log, nsamples, sam_shape, lhs_sampler, seed,
            eccen_init=[0.0, +1.0],
            gsmf_phi0=[-3.0, -2.0],
            gsmf_phiz=[-0.7, 0.0],
            gpf_malpha=[-0.5, 0.5],
            gpf_zbeta=[0.0, 2.0],
            gpf_qgamma=[-0.5, 0.5],
            gmt_malpha=[-0.5, +0.5],
            gmt_zbeta=[-3.0, +2.0],
            gmt_qgamma=[-0.5, +0.5],
            mmb_amp=[0.1e9, 1.0e9],
            mmb_plaw=[0.5, 1.5],
        )

    def sam_for_lhsnumber(self, lhsnum):
        param_grid = self.params_for_lhsnumber(lhsnum)

        eccen, gsmf_phi0, gsmf_phiz, \
            gpf_malpha, gpf_zbeta, gpf_qgamma, \
            gmt_malpha, gmt_zbeta, gmt_qgamma, \
            mmb_amp, mmb_plaw = param_grid
        mmb_amp = mmb_amp*MSOL

        # favor higher values of eccentricity instead of uniformly distributed
        eccen = eccen ** (1.0/5.0)

        gsmf = holo.sam.GSMF_Schechter(phi0=gsmf_phi0, phiz=gsmf_phiz)
        gpf = holo.sam.GPF_Power_Law(malpha=gpf_malpha, qgamma=gpf_qgamma, zbeta=gpf_zbeta)
        gmt = holo.sam.GMT_Power_Law(malpha=gmt_malpha, qgamma=gmt_qgamma, zbeta=gmt_zbeta)
        mmbulge = holo.host_relations.MMBulge_KH2013(mamp=mmb_amp, mplaw=mmb_plaw)

        sam = holo.sam.Semi_Analytic_Model(
            gsmf=gsmf, gpf=gpf, gmt=gmt, mmbulge=mmbulge,
            shape=self.sam_shape
        )

        sepa_evo, eccen_evo = holo.sam.evolve_eccen_uniform_single(sam, eccen, self.SEPA_INIT, DEF_ECCEN_NUM_STEPS)

        return sam, sepa_evo, eccen_evo


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


SPACE = LHS_PSpace_Eccen_02
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

        log.info(f"{comm.rank=} {ind=} {space.param_dict_for_lhsnumber(lhsparam)}")
        if args.test:
            continue

        try:
            # run_sam(lhsparam, PATH_OUTPUT)
            run_sam_eccen(lhsparam, PATH_OUTPUT)
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
    # print(f"\t{comm.rank} done at {str(end)} after {str(end-BEG)} = {(end-BEG).total_seconds()}")
    log.info(f"\t{comm.rank} done at {str(end)} after {str(end-BEG)} = {(end-BEG).total_seconds()}")
    bnum = _barrier(bnum)

    return


# def run_sam(pnum, path_output):
#     fname = f"lib_sams__p{pnum:06d}.npz"
#     fname = os.path.join(path_output, fname)
#     log.debug(f"{pnum=} :: {fname=}")
#     if os.path.exists(fname):
#         log.warning(f"File {fname} already exists.")

#     pta_dur = args.pta_dur * YR
#     nfreqs = args.nfreqs
#     hifr = nfreqs/pta_dur
#     pta_cad = 1.0 / (2 * hifr)
#     fobs_cents = holo.utils.nyquist_freqs(pta_dur, pta_cad)
#     fobs_edges = holo.utils.nyquist_freqs_edges(pta_dur, pta_cad)
#     log.info(f"Created {fobs_cents.size} frequency bins")
#     log.info(f"\t[{fobs_cents[0]*YR}, {fobs_cents[-1]*YR}] [1/yr]")
#     log.info(f"\t[{fobs_cents[0]*1e9}, {fobs_cents[-1]*1e9}] [nHz]")
#     assert nfreqs == fobs_cents.size

#     log.debug("Selecting `sam` and `hard` instances")
#     sam, hard = space.sam_for_lhsnumber(pnum)
#     log.debug(f"Calculating GWB for shape ({fobs_cents.size}, {args.nreals})")
#     gwb = sam.gwb(fobs_edges, realize=args.nreals, hard=hard)
#     log.debug(f"{holo.utils.stats(gwb)=}")
#     legend = space.param_dict_for_lhsnumber(pnum)
#     log.debug(f"Saving {pnum} to file")
#     np.savez(fname, fobs=fobs_cents, fobs_edges=fobs_edges, gwb=gwb,
#              pnum=pnum, pdim=space.paramdimen, nsamples=args.nsamples,
#              lhs_grid=space.sampleindxs, lhs_grid_idx=space.lhsnumber_to_index(pnum),
#              params=space.params, names=space.names, version=__version__, **legend)

#     log.info(f"Saved to {fname} after {(datetime.now()-BEG)} (start: {BEG})")
#     return


def run_sam_eccen(pnum, path_output):
    fname = f"lib_sams__p{pnum:06d}.npz"
    fname = os.path.join(path_output, fname)
    log.debug(f"{pnum=} :: {fname=}")
    if os.path.exists(fname):
        log.warning(f"File {fname} already exists.")

    pta_dur = args.pta_dur * YR
    nfreqs = args.nfreqs
    hifr = nfreqs/pta_dur
    pta_cad = 1.0 / (2 * hifr)
    fobs_cents = holo.utils.nyquist_freqs(pta_dur, pta_cad)
    fobs_edges = holo.utils.nyquist_freqs_edges(pta_dur, pta_cad)
    log.info(f"Created {fobs_cents.size} frequency bins")
    log.info(f"\t[{fobs_cents[0]*YR}, {fobs_cents[-1]*YR}] [1/yr]")
    log.info(f"\t[{fobs_cents[0]*1e9}, {fobs_cents[-1]*1e9}] [nHz]")
    assert nfreqs == fobs_cents.size

    log.debug("Selecting `sam` and `hard` instances")
    sam, sepa_evo, eccen_evo = space.sam_for_lhsnumber(pnum)
    log.debug(f"Calculating GWB for shape ({fobs_cents.size}, {args.nreals})")

    # gwb = sam.gwb(fobs_edges, realize=args.nreals, hard=hard)
    gwb = holo.gravwaves.sam_calc_gwb_single_eccen_discrete(
        fobs_cents, sam, sepa_evo, eccen_evo, nharms=DEF_ECCEN_NHARMS, nreals=DEF_NUM_REALS,
    )
    gwb = np.sqrt(np.sum(gwb, axis=1))

    log.debug(f"{holo.utils.stats(gwb)=}")
    legend = space.param_dict_for_lhsnumber(pnum)
    log.debug(f"Saving {pnum} to file")
    np.savez(fname, fobs=fobs_cents, fobs_edges=fobs_edges, gwb=gwb,
             pnum=pnum, pdim=space.paramdimen, nsamples=args.nsamples,
             lhs_grid=space.sampleindxs, lhs_grid_idx=space.lhsnumber_to_index(pnum),
             params=space.params, names=space.names, version=__version__, **legend)

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
