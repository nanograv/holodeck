"""Run detection statistics on a library of semi-analytic-models.

Usage
-----

python ./scripts/detect_lib_ss.py <LIB_PATH> --grid_path <GRID_PATH> -p <NPSRS> --sigma <SIGMA> -s <NSKIES> 

    <LIB_PATH>  :  library directory that contains sam_lib.hdf5.
    <GRID_PATH> : directory containing gamma-rho interpolation grids. Will mkdir if it doesn't exist.
    <NPSRS>     : number of PTA pulsars to simulate, should be calibrated to data
    <SIGMA>     : white noise sigma of PTA pulsars, should be calibrated to data
    <NSKIES>    : number of sky realizations to generate for each single source strain realization

Example:

    python ./scripts/detect_lib_ss.py /Users/emigardiner/GWs/output/2023-06-22_uniform-09b_n500_r100_f40_l10 \
        --grid_path /Users/emigardiner/GWs/holodeck/output/rho_gamma_grids -p 45 --sigma 1e-6 -s 25
    

To-Do
-----
* mark output directories as incomplete until all runs have been finished.
  Merged libraries from incomplete directories should also get some sort of flag!

"""


import holodeck as holo
import holodeck.detstats as ds
from holodeck.constants import YR
import numpy as np
import argparse
from datetime import datetime


DEF_NFREQS = holo.librarian.DEF_NUM_FBINS
# DEF_PTA_DUR = holo.librarian.DEF_PTA_DUR 

DEF_NPSRS = 60
DEF_SIGMA = 1e-6

DEF_NSKIES = 25
DEF_THRESH = 0.5
DEF_SNR_CYTHON = True
DEF_SAVE_SSI = False
DEF_CLBRT = False
DEF_TOL = 0.01
DEF_MAXBADS = 5

GAMMA_RHO_GRID_PATH = '/Users/emigardiner/GWs/holodeck/output/rho_gamma_grids' # modify for system

def _setup_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('lib_path', action='store', type=str,
                        help="library path")
    parser.add_argument('--grid_path', action='store', dest ='grid_path', type=str, default=GAMMA_RHO_GRID_PATH,
                        help="gamma-rho interpolation grid path")
    
    parser.add_argument('-f', '--nfreqs', action='store', dest='nfreqs', type=int, default=DEF_NFREQS,
                        help='number of frequency bins')
    # parser.add_argument('-d', '--dur', action='store', dest='dur', type=int, default=DEF_PTA_DUR,
    #                     help='pta duration in yrs')

    parser.add_argument('-t', '--tol', action='store', dest='tol', type=float, default=DEF_TOL,
                         help='tolerance for BG DP calibration')
    parser.add_argument('-b', '--maxbads', action='store', dest='maxbads', type=int, default=DEF_MAXBADS,
                         help='number of bad sigmas to try before expanding the search range')
    
    parser.add_argument('-p', '--npsrs', action='store', dest='npsrs', type=int, default=DEF_NPSRS,
                        help='number of pulsars in pta')
    parser.add_argument('--sigma', action='store', dest='sigma', type=float, default=DEF_SIGMA,
                        help='sigma for white noise of pulsars, or starting sigma if using individual realization calibration')
    parser.add_argument('--sigmin', action='store', dest='sigmin', type=float, default=1e-10,
                        help='sigma minimum for calibration')
    parser.add_argument('--sigmax', action='store', dest='sigmax', type=float, default=1e-3,
                        help='sigma maximum for calibration')
    
    parser.add_argument('-s', '--nskies', action='store', dest='nskies', type=int, default=DEF_NSKIES,
                        help='number of ss sky realizations')
    parser.add_argument('--thresh', action='store', dest='thresh', type=float, default=DEF_THRESH,
                        help='threshold for detection fractions')
    
    parser.add_argument('--plot', action='store_true', default=False,
                        help='produce plots for each simulation configuration')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='print steps along the way')
    parser.add_argument('--snr_cython', action='store_true', default=DEF_SNR_CYTHON,
                        help='Use cython for ss snr calculations')
    parser.add_argument('--save_ssi', action='store_true', default=DEF_SAVE_SSI,
                        help="Save 'gamma_ssi', the detprob of each single source.")
    parser.add_argument('--clbrt', action='store_true', default=DEF_CLBRT,
                        help="Whether or not to calibrate the PTA for individual realizations.")
    
    args = parser.parse_args()
    return args

# def freq_data(args):
#     nfreqs = args.nfreqs
#     dur = args.dur * YR
#     hifr = nfreqs/dur
#     cad = 1.0 / (2 * hifr)
#     fobs_edges = holo.utils.nyquist_freqs_edges(dur, cad)
#     dfobs = np.diff(fobs_edges)
#     return dur, cad, dfobs

def main():

    start_time = datetime.now()
    print(f"starting at {start_time}")
    # setup command line arguments
    args = _setup_argparse()
    print('npsrs=%d, sigma=%e s, nskies=%d, thresh=%f' %
          (args.npsrs, args.sigma, args.nskies, args.thresh))
    
    # # calculate cad and dfobs from duration and nfreqs
    # dur, cad, dfobs = freq_data(args)
    # args = _setup_argparse()
    # print('dur=%f yr, cad=%f yr, nfreqs=%d' %
    #       (dur/YR, cad/YR, len(dfobs)))
    

    hdf_name = args.lib_path+'/sam_lib.hdf5'
    print('Hdf file:', hdf_name)
    if args.clbrt:
        output_dir = (args.lib_path+'/detstats/clbrt_psrs%d'
                    % (args.npsrs))
    else:
        output_dir = (args.lib_path+'/detstats/psrs%d_sigma%.2e'
                    % (args.npsrs, args.sigma))
    print('Output dir:', output_dir)

    if args.clbrt:
        ds.detect_lib_clbrt_pta(hdf_name, output_dir, args.npsrs, 
                                sigstart = args.sigma, sigmin=args.sigmin, sigmax=args.sigmax, tol=args.tol, maxbads=args.maxbads, 
                            nskies=args.nskies, thresh=args.thresh, plot=args.plot, debug=args.debug,
                            grid_path=args.grid_path, 
                            snr_cython=args.snr_cython, save_ssi=args.save_ssi)
    else: 
        ds.detect_lib(hdf_name, output_dir, args.npsrs, args.sigma, 
                            nskies=args.nskies, thresh=args.thresh, plot=args.plot, debug=args.debug,
                            grid_path=args.grid_path, 
                            snr_cython=args.snr_cython, save_ssi=args.save_ssi)
    end_time = datetime.now()
    print(f"Start time: {start_time}\nEnd time: {end_time}\nTotal time: {end_time-start_time}")

if __name__ == "__main__":
    main()