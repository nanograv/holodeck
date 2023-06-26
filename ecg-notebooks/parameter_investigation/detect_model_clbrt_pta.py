import numpy as np
import holodeck as holo
import argparse
from holodeck import detstats
from datetime import datetime

# sample
SHAPE = None
DEF_NLOUDEST = 10
DEF_NREALS = 100
DEF_NFREQS = 40
DEF_NVARS = 21

# pta calibration
DEF_NSKIES = 100
DEF_NPSRS = 40

DEF_TOL = 0.01
DEF_MAXBADS = 5
GAMMA_RHO_GRID_PATH = '/Users/emigardiner/GWs/holodeck/output/rho_gamma_grids' # modify for system
ANATOMY_PATH = '/Users/emigardiner/GWs/holodeck/output/anatomy_09B'

# settings to vary
CONSTRUCT_ALL = False
CONSTRUCT_DETSTATS = False



def main():
    start_time = datetime.now()
    print("-----------------------------------------")
    print(f"starting at {start_time}")
    print("-----------------------------------------")

    load_data_from_file = args.anatomy_path+f'/{args.target}_{args.nvars}vars_clbrt_pta.npz' 
    save_data_to_file = args.anatomy_path+f'/{args.target}_{args.nvars}vars_clbrt_pta.npz' 

    args = _setup_argparse()
    print("NREALS = %d, NSKIES = %d, NPSRS = %d, target = %s, NVARS=%d"
          % (args.nreals, args.nskies, args.npsrs, args.target))
    if args.construct or args.detstats:
        if args.construct:
            params_list = np.linsapce(0,1,args.nvars)
            data, params, = vary_parameter(target_param=args.target, params_list=params_list,
                                        nfreqs=args.nfreqs, nreals=args.nreals, nloudest=args.nloudest)
        else:
            file = np.load(load_data_from_file, allow_pickle=True)
            print(file.files)
            data = file['data']
            params = file['params']
            file.close()
        dsdat = []
        for ii, _data in enumerate(data):
            if args.debug: print(f"on var {ii=} out of {args.nvars}")
    else:
        print(f"Neither {args.construct=} or {args.detstats} are true. Doing nothing.")





    end_time = datetime.now()
    print("-----------------------------------------")
    print(f"ending at {end_time}")
    print(f"total time: {end_time - start_time}")
    print("-----------------------------------------")

if __name__ == "__main__":
    main()







def _setup_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('target', action='store', type=str,
                        help="target parameter to vary")
    # parser.add_argument('--grid_path', action='store', dest ='grid_path', type=str, default=GAMMA_RHO_GRID_PATH,
    #                     help="gamma-rho interpolation grid path")
    
    # sample models setup
    parser.add_argument('-f', '--nfreqs', action='store', dest='nfreqs', type=int, default=DEF_NFREQS,
                        help='number of frequency bins')
    parser.add_argument('-f', '--nreals', action='store', dest='nreals', type=int, default=DEF_NREALS,
                        help='number of strain realizations')
    parser.add_argument('-l', '--nloudest', action='store', dest='nloudest', type=int, default=DEF_NLOUDEST,
                        help='number of loudest single sources')
    parser.add_argument('-v', '--nvars', actions='store', dest='nvars', type=int, default=DEF_NVARS,
                        help='number of variations on target param')
    # parser.add_argument('-d', '--dur', action='store', dest='dur', type=int, default=DEF_PTA_DUR,
    #                     help='pta duration in yrs')


    
    # pta setup
    parser.add_argument('-p', '--npsrs', action='store', dest='npsrs', type=int, default=DEF_NPSRS,
                        help='number of pulsars in pta')
    parser.add_argument('-s', '--nskies', action='store', dest='nskies', type=int, default=DEF_NSKIES,
                        help='number of ss sky realizations')
    
    # pta calibration settings
    parser.add_argument('--sigstart', action='store', dest='sigma', type=float, default=1e-7,
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
    parser.add_argument('--construct', action='store_true', default=CONSTRUCT_ALL,
                        help='construct data and detstats for each varying param')
    parser.add_argument('--detstats', action='store_true', default=CONSTRUCT_DETSTATS,
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
    
    args = parser.parse_args()
    return args
















# construct a param_space instance, note that `nsamples` and `seed` don't matter here for how we'll use this
pspace = holo.param_spaces.PS_Uniform_09B(holo.log, nsamples=1, sam_shape=SHAPE, seed=None)

def vary_parameter(
        target_param,    # the name of the parameter, has to exist in `param_names`
        params_list,  # the values we'll check
        nreals, nfreqs, nloudest,
        pspace = holo.param_spaces.PS_Uniform_09B(holo.log, nsamples=1, sam_shape=SHAPE, seed=None),
        pars=None, save_dir=None, 
        ):

    # get the parameter names from this library-space
    param_names = pspace.param_names
    num_pars = len(pspace.param_names)
    print(f"{num_pars=} :: {param_names=}")

    # choose each parameter to be half-way across the range provided by the library
    if pars is None:
        pars = 0.5 * np.ones(num_pars) 
    str_pars = str(pars).replace(" ", "_").replace("[", "").replace("]", "")
    # Choose parameter to vary
    param_idx = param_names.index(target_param)

    data = []
    params = []
    for ii, par in enumerate(params_list):
        pars[param_idx] = par
        print(f"{ii=}, {pars=}")
        # _params = pspace.param_samples[0]*pars
        _params = pspace.normalized_params(pars)
        params.append(_params)
        # construct `sam` and `hard` instances based on these parameters
        sam, hard = pspace.model_for_params(_params, pspace.sam_shape)
        if isinstance(hard, holo.hardening.Fixed_Time_2PL_SAM):
            hard_name = 'Fixed Time'
        elif isinstance(hard, holo.hardening.Hard_GW):
            hard_name = 'GW Only'
        # run this model, retrieving binary parameters and the GWB
        _data = holo.librarian.run_model(sam, hard, nreals, nfreqs, nloudest=nloudest,
                                        gwb_flag=False, singles_flag=True, params_flag=True, details_flag=True)
        data.append(_data)
    if save_dir is not None:
        str_shape = str(sam.shape).replace(", ", "_").replace("(", "").replace(")", "")
        filename = save_dir+'/%s_p%s_s%s.npz' % (target_param, str_pars, str_shape)
        np.savez(filename, data=data, params=params, hard_name=hard_name, shape=sam.shape, target_param=target_param )
        print('saved to %s' % filename)

    return (data, params)