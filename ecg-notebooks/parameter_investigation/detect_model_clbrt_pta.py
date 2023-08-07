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

# pta calibration
DEF_NSKIES = 100
DEF_NPSRS = 40
DEF_RED_AMP = None
DEF_RED_GAMMA = None
DEF_RED2WHITE = None

DEF_TOL = 0.01
DEF_MAXBADS = 5
GAMMA_RHO_GRID_PATH = '/Users/emigardiner/GWs/holodeck/output/rho_gamma_grids' # modify for system
ANATOMY_PATH = '/Users/emigardiner/GWs/holodeck/output/anatomy_redz'

# settings to vary
DEF_CONSTRUCT = False
DEF_DETSTATS = False


def _setup_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('target', action='store', type=str,
                        help="target parameter to vary")
    # parser.add_argument('--grid_path', action='store', dest ='grid_path', type=str, default=GAMMA_RHO_GRID_PATH,
    #                     help="gamma-rho interpolation grid path")
    
    # sample models setup
    parser.add_argument('-f', '--nfreqs', action='store', dest='nfreqs', type=int, default=DEF_NFREQS,
                        help='number of frequency bins')
    parser.add_argument('-r', '--nreals', action='store', dest='nreals', type=int, default=DEF_NREALS,
                        help='number of strain realizations')
    parser.add_argument('-l', '--nloudest', action='store', dest='nloudest', type=int, default=DEF_NLOUDEST,
                        help='number of loudest single sources')
    parser.add_argument('-v', '--nvars', action='store', dest='nvars', type=int, default=DEF_NVARS,
                        help='number of variations on target param')
    parser.add_argument('--shape', action='store', dest='shape', type=int, default=DEF_SHAPE,
                        help='sam shape')
    # parser.add_argument('-d', '--dur', action='store', dest='dur', type=int, default=DEF_PTA_DUR,
    #                     help='pta duration in yrs')


    
    # pta setup
    parser.add_argument('-p', '--npsrs', action='store', dest='npsrs', type=int, default=DEF_NPSRS,
                        help='number of pulsars in pta')
    parser.add_argument('-s', '--nskies', action='store', dest='nskies', type=int, default=DEF_NSKIES,
                        help='number of ss sky realizations')
    parser.add_argument('--red_amp', action='store', dest='red_amp', type=float, default=DEF_RED_AMP,
                        help='Red noise amplitude')
    parser.add_argument('--red_gamma', action='store', dest='red_gamma', type=float, default=DEF_RED_GAMMA,
                        help='Red noise gamma')
    parser.add_argument('--red2white', action='store', dest='red2white', type=float, default=DEF_RED2WHITE,
                        help='Red noise amplitude to white noise amplitude ratio.')
    parser.add_argument('--ssn', action='store_true', dest='ss_noise', default=False, 
                        help='Whether or not to use single sources as a noise source in background calculations.') 
    parser.add_argument('--dsc', action='store_true', dest='dsc_flag', default=False, 
                        help='Whether or not to use single sources as a noise source in background calculations.') 
    parser.add_argument('--gsc-clbrt', action='store_true', dest='gsc_flag', default=False, 
                        help='Whether or not to use gsc noise to calibrate the background and dsc noise for SS detstats.') 
    
    # pta calibration settings
    parser.add_argument('--sigstart', action='store', dest='sigstart', type=float, default=1e-7,
                        help='starting sigma if for realization calibration')
    parser.add_argument('--sigmin', action='store', dest='sigmin', type=float, default=1e-10,
                        help='sigma minimum for calibration')
    parser.add_argument('--sigmax', action='store', dest='sigmax', type=float, default=1e-4,
                        help='sigma maximum for calibration')
    parser.add_argument('--thresh', action='store', dest='thresh', type=float, default=0.5,
                        help='threshold for detection fractions')
    parser.add_argument('-t', '--tol', action='store', dest='tol', type=float, default=DEF_TOL,
                         help='tolerance for BG DP calibration')
    parser.add_argument('-b', '--maxbads', action='store', dest='maxbads', type=int, default=DEF_MAXBADS,
                         help='number of bad sigmas to try before expanding the search range')
    
    # general settings
    parser.add_argument('--construct', action='store_true', default=DEF_CONSTRUCT,
                        help='construct data and detstats for each varying param')
    parser.add_argument('--detstats', action='store_true', default=DEF_DETSTATS,
                        help='construct detstats, using saved data')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='print steps along the way')
    
    # rarely need changing
    parser.add_argument('--snr_cython', action='store_true', default=True,
                        help='Use cython for ss snr calculations')
    parser.add_argument('--save_ssi', action='store_true', default=True,
                        help="Save 'gamma_ssi', the detprob of each single source.")
    parser.add_argument('--clbrt', action='store_true', default=True,
                        help="Whether or not to calibrate the PTA for individual realizations.")
    parser.add_argument('--grid_path', action='store', dest ='grid_path', type=str, default=GAMMA_RHO_GRID_PATH,
                        help="gamma-rho interpolation grid path")
    parser.add_argument('--anatomy_path', action='store', dest ='anatomy_path', type=str, default=ANATOMY_PATH,
                        help="path to load and save anatomy files")
    parser.add_argument('--load_file', action='store', dest ='load_file', type=str, default=None,
                        help="file to load sample data and params, excluding .npz suffice")
    parser.add_argument('--save_file', action='store', dest ='save_file', type=str, default=None,
                        help="file to save sample data, excluding .npz suffix")
    
    args = parser.parse_args()
    return args


# # construct a param_space instance, note that `nsamples` and `seed` don't matter here for how we'll use this
# pspace = holo.param_spaces.PS_Uniform_09B(holo.log, nsamples=1, sam_shape=SHAPE, seed=None)

def vary_parameter(
        target_param,    # the name of the parameter, has to exist in `param_names`
        params_list,  # the values we'll check
        nreals, nfreqs, nloudest,
        pspace,
        pars=None, debug=True
        ):

    # get the parameter names from this library-space
    param_names = pspace.param_names
    num_pars = len(pspace.param_names)
    if debug: print(f"{num_pars=} :: {param_names=}")

    # choose each parameter to be half-way across the range provided by the library
    if pars is None:
        pars = 0.5 * np.ones(num_pars) 
    str_pars = str(pars).replace(" ", "_").replace("[", "").replace("]", "")
    # Choose parameter to vary
    param_idx = param_names.index(target_param)

    data = []
    params = []
    for ii, par in enumerate(tqdm(params_list)):
        pars[param_idx] = par
        if debug: print(f"{ii=}, {pars=}")
        # _params = pspace.param_samples[0]*pars
        _params = pspace.normalized_params(pars)
        params.append(_params)
        # construct `sam` and `hard` instances based on these parameters
        sam, hard = pspace.model_for_params(_params, pspace.sam_shape)
        # run this model, retrieving binary parameters and the GWB
        _data = holo.librarian.run_model(sam, hard, nreals, nfreqs, nloudest=nloudest,
                                        gwb_flag=False, singles_flag=True, params_flag=True, details_flag=True)
        data.append(_data)

    return (data, params)



def main():
    start_time = datetime.now()
    print("-----------------------------------------")
    print(f"starting at {start_time}")
    print("-----------------------------------------")

    args = _setup_argparse()
    print("NREALS = %d, NSKIES = %d, NPSRS = %d, target = %s, NVARS=%d"
          % (args.nreals, args.nskies, args.npsrs, args.target, args.nvars))
    
    output_path = args.anatomy_path+f'/{args.target}_v{args.nvars}_r{args.nreals}_shape{str(args.shape)}'
    # check if output folder already exists, if not, make it.
    if os.path.exists(output_path) is False:
        os.makedirs(output_path)

    if args.load_file is None:
        load_data_from_file = output_path+'/data_params'
    else:
        load_data_from_file = args.load_file

    if args.save_file is None:
        save_data_to_file =  output_path+'/data_params'
    else:
        save_data_to_file = args.save_file

    save_dets_to_file = output_path+f'/detstats_s{args.nskies}'
    if args.ss_noise: save_dets_to_file = save_dets_to_file+'_ssn'

    if args.gsc_flag: save_dets_to_file = save_dets_to_file+'_gsc'
    elif args.dsc_flag: save_dets_to_file = save_dets_to_file+'_dsc' # only append 'dsc' if not gsc-calibrated

    if args.red2white is not None and args.red_gamma is not None:
        save_dets_to_file = save_dets_to_file+f'_r2w{args.red2white:.1f}_rg{args.red_gamma:.1f}'
    elif args.red_amp is not None and args.red_gamma is not None:
        save_dets_to_file = save_dets_to_file+f'_ra{args.red_amp:.1e}_rg{args.red_gamma:.1f}'
    else: 
        save_dets_to_file = save_dets_to_file+f'_white'

    if args.red2white is not None and args.red_amp is not None:
        print(f"{args.red2white=} and {args.red_amp} both provided. red_amp will be overriden by red2white ratio.")

    
    print(f"{load_data_from_file=}.npz")
    print(f"{save_data_to_file=}.npz")
    print(f"{save_dets_to_file=}.npz")

    if args.construct or args.detstats:
        if args.construct:
            params_list = np.linspace(0,1,args.nvars)
            data, params, = vary_parameter(
                target_param=args.target, params_list=params_list,
                nfreqs=args.nfreqs, nreals=args.nreals, nloudest=args.nloudest,
                pspace = holo.param_spaces.PS_Uniform_09B(holo.log, nsamples=1, sam_shape=args.shape, seed=None),)
            np.savez(save_data_to_file+'.npz', data=data, params=params) # save before calculating detstats, in case of crash
        else:
            file = np.load(load_data_from_file+'.npz', allow_pickle=True)
            print('loaded files:', file.files)
            data = file['data']
            params = file['params']
            file.close()

        fobs_cents = data[0]['fobs_cents']

        # get dsdat for each data/param
        dsdat = []
        for ii, _data in enumerate(tqdm(data)):
            if args.debug: print(f"on var {ii=} out of {args.nvars}")
            hc_bg = _data['hc_bg']
            hc_ss = _data['hc_ss']
            if args.gsc_flag:
                _dsdat = detstats.detect_pspace_model_clbrt_pta_gsc(
                    fobs_cents, hc_ss, hc_bg, args.npsrs, args.nskies, 
                    sigstart=args.sigstart, sigmin=args.sigmin, sigmax=args.sigmax, tol=args.tol, maxbads=args.maxbads,
                    thresh=args.thresh, debug=args.debug, ss_noise=args.ss_noise,
                )
            else:
                _dsdat = detstats.detect_pspace_model_clbrt_pta(
                    fobs_cents, hc_ss, hc_bg, args.npsrs, args.nskies, dsc_flag=args.dsc_flag,
                    sigstart=args.sigstart, sigmin=args.sigmin, sigmax=args.sigmax, tol=args.tol, maxbads=args.maxbads,
                    thresh=args.thresh, debug=args.debug, ss_noise=args.ss_noise,
                    red_amp=args.red_amp, red_gamma=args.red_gamma, red2white=args.red2white)
            dsdat.append(_dsdat)
        np.savez(save_dets_to_file+'.npz', dsdat=dsdat, red_amp=args.red_amp, red_gamma=args.red_gamma, npsrs=args.npsrs, red2white=args.red2white) # overwrite
    else:
        print(f"Neither {args.construct=} or {args.detstats} are true. Doing nothing.")

    end_time = datetime.now()
    print("-----------------------------------------")
    print(f"ending at {end_time}")
    print(f"total time: {end_time - start_time}")
    print("-----------------------------------------")

if __name__ == "__main__":
    main()

