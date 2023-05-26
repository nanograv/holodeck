
import holodeck as holo
import holodeck.detstats as ds
from holodeck.constants import YR
import numpy as np
import argparse

DEF_NFREQS = holo.librarian.DEF_NUM_FBINS
DEF_PTA_DUR = holo.librarian.DEF_PTA_DUR 

DEF_NPSRS = 60
DEF_SIGMA = 1e-6

DEF_NSKIES = 25
DEF_THRESH = 0.5

GAMMA_RHO_GRID_PATH = '/Users/emigardiner/GWs/holodeck/output/rho_gamma_grids' # modify for system

def _setup_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('lib_path', action='store', type=str,
                        help="library path")
    parser.add_argument('--grid_path', action='store', dest ='grid_path', type=str, default=GAMMA_RHO_GRID_PATH,
                        help="gamma-rho interpolation grid path")
    
    parser.add_argument('-f', '--nfreqs', action='store', dest='nfreqs', type=int, default=DEF_NFREQS,
                        help='number of frequency bins')
    parser.add_argument('-d', '--dur', action='store', dest='dur', type=int, default=DEF_PTA_DUR,
                        help='pta duration in yrs')
    
    parser.add_argument('-p', '--npsrs', action='store', dest='npsrs', type=int, default=DEF_NPSRS,
                        help='number of pulsars in pta')
    parser.add_argument('--sigma', action='store', dest='sigma', type=float, default=DEF_SIGMA,
                        help='sigma for white noise of pulsars')
    
    parser.add_argument('-s', '--nskies', action='store', dest='nskies', type=int, default=DEF_NSKIES,
                        help='number of ss sky realizations')
    parser.add_argument('--thresh', action='store', dest='thresh', type=float, default=DEF_THRESH,
                        help='threshold for detection fractions')
    
    parser.add_argument('--plot', action='store_true', default=False,
                        help='produce plots for each simulation configuration')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='print steps along the way')
    
    args = parser.parse_args()
    return args

def freq_data(args):
    nfreqs = args.nfreqs
    dur = args.dur * YR
    hifr = nfreqs/dur
    cad = 1.0 / (2 * hifr)
    fobs_edges = holo.utils.nyquist_freqs_edges(dur, cad)
    dfobs = np.diff(fobs_edges)
    return dur, cad, dfobs

def main():
    # setup command line arguments
    args = _setup_argparse()
    print('npsrs=%d, sigma=%e s, nskies=%d, thresh=%f' %
          (args.npsrs, args.sigma, args.nskies, args.thresh))
    
    # calculate cad and dfobs from duration and nfreqs
    dur, cad, dfobs = freq_data(args)
    args = _setup_argparse()
    print('dur=%f yr, cad=%f yr, nfreqs=%d' %
          (dur/YR, cad/YR, len(dfobs)))
    

    hdf_name = args.lib_path+'/sam_lib.hdf5'
    print('Hdf file:', hdf_name)

    output_dir = (args.lib_path+'/detstats/psrs%d_sigma%.2e'
                % (args.npsrs, args.sigma))
    print('Output dir:', output_dir)

    ds.detect_lib(hdf_name, output_dir, args.npsrs, args.sigma, 
                        nskies=args.nskies, thresh=args.thresh, plot=args.plot, debug=args.debug,
                        dur=dur, cad=cad, dfobs=dfobs, grid_path=args.grid_path)
  

if __name__ == "__main__":
    main()