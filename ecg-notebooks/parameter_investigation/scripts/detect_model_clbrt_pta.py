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
    
    parser.add_argument('--bgl', '--bg_nloudest', action='store', dest='bg_nloudest', type=int, default=DEF_NLOUDEST,
                        help='number of loudest single sources subtracted from the background')
    parser.add_argument('-v', '--nvars', action='store', dest='nvars', type=int, default=DEF_NVARS,
                        help='number of variations on target param')
    parser.add_argument('--shape', action='store', dest='shape', type=int, default=DEF_SHAPE,
                        help='sam shape')
    parser.add_argument('--gw_only', action='store_true', dest='gw_only', default=False,
                        help='whether or not to use gw-only evolution')

    # parameters
    parser.add_argument('--var_hard_time', action='store', dest='var_hard_time', type=int, default=None,
                        help='hardening time parameter variation')


    
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
    
    # pta noise settings
    parser.add_argument('--ssn', action='store_true', dest='ss_noise', default=False, 
                        help='Whether or not to use single sources as a noise source in background calculations.') 
    parser.add_argument('--dsc', action='store_true', dest='dsc_flag', default=False, 
                        help='Whether or not to use DeterSensitivityCurve as single source noise.') 
    parser.add_argument('--gsc-clbrt', action='store_true', dest='gsc_flag', default=False, 
                        help='Whether or not to use gsc noise to calibrate the background and dsc noise for SS detstats.') 
    parser.add_argument('--divide', action='store_true', dest='divide_flag', default=False, 
                        help='Whether or not to divide sensitivity curves among the pulsars.') 
    parser.add_argument('--onepsr', action='store_true', dest='onepsr_flag', default=False, 
                        help='Whether or not to treat PTA with gsc/dsc noise as 1 psr.') 
    parser.add_argument('--nexcl', '--nexcl_noise', action='store', dest='nexcl', type=int, default=0,
                        help='number of loudest single sources to exclude in hc_rest noise')
    
    # pta calibration settings
    parser.add_argument('--cv', '--calvar', action='store', dest='calvar', type=int, default=DEF_CALVAR,
                        help='variation to use for calibration')
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



def main():
    start_time = datetime.now()
    print("-----------------------------------------")
    print(f"starting at {start_time}")
    print("-----------------------------------------")

    # set up args
    args = _setup_argparse()
    print(f"NREALS = {args.nreals}, NSKIES = {args.nskies}, NPSRS = {args.npsrs}, target = {args.target}, NVARS={args.nvars}")
    print(f"CV={args.calvar}, NLOUDEST={args.nloudest}, BGL={args.bg_nloudest}, {args.gsc_flag=}, {args.dsc_flag=}, {args.gw_only=}")
    
    # get file names based on arguments
    load_data_from_file, save_data_to_file, save_dets_to_file = file_names(args)
    print(f"{load_data_from_file=}.npz")
    print(f"{save_data_to_file=}.npz")
    print(f"{save_dets_to_file=}.npz")


    # calculate model and/or detstats
    if args.construct or args.detstats:
        # calculate model
        if args.construct:
            data, params = construct_data(args)

            # save data file
            np.savez(save_data_to_file+'.npz', data=data, params=params) # save before calculating detstats, in case of crash

        # or just load model
        else:
            file = np.load(load_data_from_file+'.npz', allow_pickle=True)
            print('loaded files:', file.files)
            data = file['data']
            params = file['params']
            file.close()
            
        if args.detstats:
            # calculate detection statistics
            if args.calvar is not None:  #### 'FIXED-PTA' METHOD
                dsdat = fixed_pta_method(args, data)
            else:                        #### 'REALIZATION-CALIBRATED' METHOD
                dsdat = realization_calibrated_method(args, data)

            # Save detection statistics file
            np.savez(save_dets_to_file+'.npz', dsdat=dsdat, red_amp=args.red_amp, red_gamma=args.red_gamma, npsrs=args.npsrs, red2white=args.red2white) # overwrite

    else:
        print(f"Neither {args.construct=} or {args.detstats} are true. Doing nothing.")

    end_time = datetime.now()
    print("-----------------------------------------")
    print(f"ending at {end_time}")
    print(f"total time: {end_time - start_time}")
    print("-----------------------------------------")





def file_names(args):
    """ Set up output folder, data load/save file, and detstats save file."""
    # set up output folder
    if args.gw_only:
        anatomy_path = '/Users/emigardiner/GWs/holodeck/output/anatomy_7GW'
    else:
        anatomy_path = args.anatomy_path

    output_path = anatomy_path + f'/{args.target}_v{args.nvars}_r{args.nreals}_shape{str(args.shape)}'
    if args.var_hard_time is not None:
        output_path += f"_vtau{args.var_hard_time}"
    # check if output folder already exists, if not, make it.
    if os.path.exists(output_path) is False:
        os.makedirs(output_path)

    # set up load and save locations
    if args.load_file is None:
        load_data_from_file = output_path+'/data_params'
    else:
        load_data_from_file = args.load_file

    if args.save_file is None:
        save_data_to_file =  output_path+'/data_params'
    else:
        save_data_to_file = args.save_file

    # define detstats file name
    save_dets_to_file = output_path+f'/detstats_s{args.nskies}'
    if args.nloudest != DEF_NLOUDEST:                                           # if using nloudest that isn't the default 10
        save_dets_to_file += f"_l{args.nloudest}" 
        save_data_to_file += f"_l{args.nloudest}"
    if args.bg_nloudest != args.nloudest:
        save_dets_to_file += f"_bgl{args.bg_nloudest}" # only change nloudest subtracted from bg, not single sources loudest
    if args.calvar is not None: save_dets_to_file += f"_cv{args.calvar}"        # if using one variation to calibrate
    if args.ss_noise: save_dets_to_file += '_ssn'                               # if using single sources as noise

    if args.gsc_flag:                                                           # if using GSC as noise
        save_dets_to_file += '_gsc'
        if args.dsc_flag is False:                                          # if using GSC as noise
            save_dets_to_file += 'both'

        if args.onepsr_flag:
            save_dets_to_file = save_dets_to_file+'_onepsr'
            assert args.npsrs == 1, "To use '--onepsr' set -p 1"
            assert args.divide_flag is False, "only one of '--divide' and '--onepsr' should be True"
        if args.divide_flag:
            save_dets_to_file += '_divide'
        else:
            save_dets_to_file += '_nodiv'
    if args.dsc_flag: save_dets_to_file += '_dsc' # only append 'dsc' if not gsc-calibrated

    if args.nexcl > 0:
        save_dets_to_file += f'_nexcl{args.nexcl}'

    if args.npsrs != 40:
        save_dets_to_file += f'_p{args.npsrs}'


    if args.red2white is not None and args.red_gamma is not None:               # if using red noise with fixed red_gamma
        save_dets_to_file = save_dets_to_file+f'_r2w{args.red2white:.1e}_rg{args.red_gamma:.1f}'
    elif args.red_amp is not None and args.red_gamma is not None:               # if using fixed red noise 
        save_dets_to_file = save_dets_to_file+f'_ra{args.red_amp:.1e}_rg{args.red_gamma:.1f}'
    else: 
        save_dets_to_file = save_dets_to_file+f'_white'

    if args.red2white is not None and args.red_amp is not None:
        print(f"{args.red2white=} and {args.red_amp} both provided. red_amp will be overriden by red2white ratio.")

    
    return load_data_from_file, save_data_to_file, save_dets_to_file

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


def construct_data(args):

    params_list = np.linspace(0,1,args.nvars)

    if args.gw_only is True:
        print('using GW only')
        pspace = holo.param_spaces.PS_Uniform_07_GW(holo.log, nsamples=1, sam_shape=args.shape, seed=None)
        pars = None
    else:
        pspace = holo.param_spaces.PS_Uniform_09B(holo.log, nsamples=1, sam_shape=args.shape, seed=None)

        # set a hardening time other than the middle ones as args.var_hard_time
        if args.var_hard_time is not None:
            pars = 0.5 * np.ones(6) 
            pars[0] = params_list[args.var_hard_time]
        else:
            pars = None

    data, params, = vary_parameter(
        target_param=args.target, params_list=params_list,
        nfreqs=args.nfreqs, nreals=args.nreals, nloudest=args.nloudest, pars=pars,
        pspace = pspace,)
    return data, params


def resample_loudest(hc_ss, hc_bg, nloudest):
    if nloudest > hc_ss.shape[-1]: # check for valid nloudest
        err = f"{nloudest=} for detstats must be <= nloudest of hc data"
        raise ValueError(err)
    
    # recalculate new hc_bg and hc_ss
    new_hc_bg = np.sqrt(hc_bg**2 + np.sum(hc_ss[...,nloudest:-1]**2, axis=-1))
    new_hc_ss = hc_ss[...,0:nloudest]

    return new_hc_ss, new_hc_bg


def fixed_pta_method(args, data):
    """ 'FIXED-PTA' METHOD
    
    calibrate pta once if using calvar (calibration variation)
    """

    if args.red2white is not None or args.red_gamma is not None:
        err = "Error! Fixed_pta_method() is not set up for red noise."
        raise ValueError(err)

    fobs_cents = data[0]['fobs_cents']

    # get hc_ss and hc_bg from appropriate calibration variation
    hc_bg = data[args.calvar]['hc_bg']
    hc_ss = data[args.calvar]['hc_ss']
    hc_bg_noise = None
    print(f"{hc_ss.shape=}")
    if args.nloudest != hc_ss.shape[-1]:
        print(f"Resampling {args.nloudest=} loudest.")
        hc_ss, hc_bg = resample_loudest(hc_ss, hc_bg, args.nloudest)
    elif args.bg_nloudest != hc_ss.shape[-1]:
        print(f"Resampling {args.bg_nloudest} BG nloudest.")
        hc_bg_noise = hc_bg
        _, hc_bg = resample_loudest(hc_ss, hc_bg, args.bg_nloudest) # only change nloudest subtracted from bg, not single sources loudest

    # get median across realizations of calvar, for psr calibration
    hc_bg_med = np.median(hc_bg, axis=-1)
    hc_ss_med = np.median(hc_ss, axis=-2) # dummy hc_ss, not actually used 

    if args.gsc_flag:
        psrs = detstats.calibrate_one_pta_gsc(
            hc_bg_med, hc_ss_med, fobs_cents, args.npsrs, ret_sig=False,
            sigstart=args.sigstart, sigmin=args.sigmin, sigmax=args.sigmax, tol=args.tol, maxbads=args.maxbads,
            divide_flag=args.divide_flag,)
    else:
        psrs = detstats.calibrate_one_pta(
            hc_bg_med, hc_ss_med, fobs_cents, args.npsrs, ret_sig=False, 
            sigstart=args.sigstart, sigmin=args.sigmin, sigmax=args.sigmax, tol=args.tol, maxbads=args.maxbads,
            )

    # get dsdat for each data/param
    dsdat = []
    for ii, _data in enumerate(tqdm(data)):
        if args.debug: print(f"on var {ii=} out of {args.nvars}")
        hc_bg = _data['hc_bg']
        hc_ss = _data['hc_ss']

        # shift loudest as needed
        if args.nloudest != hc_ss.shape[-1]:
            hc_ss, hc_bg = resample_loudest(hc_ss, hc_bg, args.nloudest)
        elif args.bg_nloudest != hc_ss.shape[-1]:
            print(f"resampling {args.bg_nloudest=} loudest!")
            _, hc_bg = resample_loudest(hc_ss, hc_bg, args.bg_nloudest) # only change nloudest subtracted from bg, not single sources loudest

        _dsdat = detstats.detect_pspace_model_psrs(
                fobs_cents, hc_ss, hc_bg, psrs, args.nskies, hc_bg_noise=hc_bg_noise,
                thresh=args.thresh, debug=args.debug, nexcl_noise=args.nexcl )
        dsdat.append(_dsdat)
        # not updated to allow for dsc alone
        
    return dsdat

def realization_calibrated_method(args, data):
    """ 'REALIZATION-CALIBRATED' METHOD
    
    calibrate PTA realization by realization
    """
    fobs_cents = data[0]['fobs_cents']

    # get dsdat for each data/param
    dsdat = []
    for ii, _data in enumerate(tqdm(data)):
        if args.debug: print(f"on var {ii=} out of {args.nvars}")
        
        # get characteristic strains for the current variation
        hc_bg = _data['hc_bg']
        hc_ss = _data['hc_ss']
        hc_bg_noise=None

        # shift loudest as needed
        if args.nloudest != hc_ss.shape[-1]:
            hc_ss, hc_bg = resample_loudest(hc_ss, hc_bg, args.nloudest)
        elif args.bg_nloudest != hc_ss.shape[-1]:
            print(f"resampling {args.bg_nloudest=} loudest!")
            hc_bg_noise=hc_bg
            _, hc_bg = resample_loudest(hc_ss, hc_bg, args.bg_nloudest) # only change nloudest subtracted from bg, not single sources loudest
            # print(f"{np.shares_memory(hc_bg_noise, hc_bg)=}")

        if args.gsc_flag:
            _dsdat = detstats.detect_pspace_model_clbrt_pta_gsc(
                fobs_cents, hc_ss, hc_bg, args.npsrs, args.nskies, hc_bg_noise=hc_bg_noise,
                sigstart=args.sigstart, sigmin=args.sigmin, sigmax=args.sigmax, tol=args.tol, maxbads=args.maxbads,
                thresh=args.thresh, debug=args.debug, 
                ss_noise=args.ss_noise, divide_flag=args.divide_flag, dsc_flag=args.dsc_flag, nexcl_noise=args.nexcl
            )
        else:
            _dsdat = detstats.detect_pspace_model_clbrt_pta(
                fobs_cents, hc_ss, hc_bg, args.npsrs, args.nskies, hc_bg_noise=hc_bg_noise,
                sigstart=args.sigstart, sigmin=args.sigmin, sigmax=args.sigmax, tol=args.tol, maxbads=args.maxbads,
                thresh=args.thresh, debug=args.debug, ss_noise=args.ss_noise, dsc_flag=args.dsc_flag, nexcl_noise=args.nexcl,
                red_amp=args.red_amp, red_gamma=args.red_gamma, red2white=args.red2white)
        dsdat.append(_dsdat)

    return dsdat



if __name__ == "__main__":
    main()