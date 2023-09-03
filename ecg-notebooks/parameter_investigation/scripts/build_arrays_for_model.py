import numpy as np
import holodeck as holo
import argparse
from holodeck import detstats
from datetime import datetime
from tqdm import tqdm
import os

# sample
DEF_SHAPE = None
DEF_NLOUDEST = 10
DEF_NREALS = 100
DEF_NFREQS = 40
DEF_NVARS = 21
DEF_CALVAR = None

# pta calibration
DEF_NSKIES = 100
DEF_RED_GAMMA = None
DEF_RED2WHITE = None

GAMMA_RHO_GRID_PATH = '/Users/emigardiner/GWs/holodeck/output/rho_gamma_grids' # modify for system
ANATOMY_PATH = '/Users/emigardiner/GWs/holodeck/output/anatomy_redz'


def _setup_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('target', action='store', type=str,
                        help="target parameter to vary")
    parser.add_argument('--favg', action='store_true', dest='favg', default=False,
                        help='whether or not to build favg arrays')
    
    # what to do
    parser.add_argument('--gw_only', action='store_true', dest='gw_only', default=False,
                        help='whether or not to use gw-only evolution')
      

    # retrieve data, params, detstats information
    parser.add_argument('-f', '--nfreqs', action='store', dest='nfreqs', type=int, default=DEF_NFREQS,
                        help='number of frequency bins')
    parser.add_argument('-r', '--nreals', action='store', dest='nreals', type=int, default=DEF_NREALS,
                        help='number of strain realizations')
    parser.add_argument('--shape', action='store', dest='shape', type=int, default=DEF_SHAPE,
                        help='sam shape')
    parser.add_argument('-l', '--nloudest', action='store', dest='nloudest', type=int, default=DEF_NLOUDEST,
                        help='number of loudest single sources')
    parser.add_argument('--bgl', '--bgl', action='store', dest='bgl', type=int, default=DEF_NLOUDEST,
                        help='number of loudest single sources subtracted from the background')
    parser.add_argument('-v', '--nvars', action='store', dest='nvars', type=int, default=DEF_NVARS,
                        help='number of variations on target param')
 
    
    # pta information
    parser.add_argument('-s', '--nskies', action='store', dest='nskies', type=int, default=DEF_NSKIES,
                        help='number of ss sky realizations')
    parser.add_argument('--cv', '--calvar', action='store', dest='calvar', type=int, default=DEF_CALVAR,
                        help='variation to use for calibration')
    parser.add_argument('--red_gamma', action='store', dest='red_gamma', type=float, default=DEF_RED_GAMMA,
                        help='Red noise gamma')
    parser.add_argument('--red2white', action='store', dest='red2white', type=float, default=DEF_RED2WHITE,
                        help='Red noise amplitude to white noise amplitude ratio.')

  
    # pta noise settings
    parser.add_argument('--ssn', action='store_true', dest='ss_noise', default=False, 
                        help='Whether or not to use single sources as a noise source in background calculations.') 
    parser.add_argument('--dsc', action='store_true', dest='dsc_flag', default=False, 
                        help='Whether or not to use DeterSensitivityCurve as single source noise.') 
    parser.add_argument('--gsc_clbrt', action='store_true', dest='gsc_flag', default=False, 
                        help='Whether or not to use gsc noise to calibrate the background and dsc noise for SS detstats.') 
    parser.add_argument('--divide', action='store_true', dest='divide_flag', default=False, 
                        help='Whether or not to divide sensitivity curves among the pulsars.') 
    parser.add_argument('--onepsr', action='store_true', dest='onepsr_flag', default=False, 
                        help='Whether or not to treat PTA with gsc/dsc noise as 1 psr.') 
    
    # rarely need changing
    
    args = parser.parse_args()
    return args



def main():
    start_time = datetime.now()
    print("-----------------------------------------")
    print(f"starting at {start_time}")
    print("-----------------------------------------")

    # set up args
    args = _setup_argparse()
    print(f"NREALS = {args.nreals}, NSKIES = {args.nskies}, target = {args.target}, NVARS={args.nvars}")
    print(f"CV={args.calvar}, NLOUDEST={args.nloudest}, BGL={args.bgl}, {args.gsc_flag=}, {args.dsc_flag=}")
    print(f"RED2WHITE={args.red2white}, RED_GAMMA={args.red_gamma}")

    if args.favg:
        print("---building favg array---")
        detstats.build_favg_arrays(
            target=args.target, nreals=args.nreals, nskies=args.nskies,
            gw_only=args.gw_only, red2white=args.red2white, red_gamma=args.red_gamma,
            nloudest=args.nloudest, bgl=args.bgl, cv=args.calvar, 
            gsc_flag=args.gsc_flag, dsc_flag=args.dsc_flag, divide_flag=args.divide_flag,
            )

    end_time = datetime.now()
    print("-----------------------------------------")
    print(f"ending at {end_time}")
    print(f"total time: {end_time - start_time}")
    print("-----------------------------------------")




if __name__ == "__main__":
    main()