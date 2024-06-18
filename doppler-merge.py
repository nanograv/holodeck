"""
"""

import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

import holodeck as holo
from holodeck import log, utils
import holodeck.librarian


def main(path):

    # ---- Load reference objects from path
    # -------------------------------------

    # path = (
    #     "/Users/lzkelley/Programs/nanograv/holodeck/output/"
    #     "doppler_astro-strong-all_n1080_r100_f40"
    # )
    path = Path(path).absolute().resolve()
    print(path)
    assert path.is_dir()

    space, space_fname = holo.librarian.lib_tools.load_pspace_from_path(path)
    args, args_fname = holo.librarian.gen_lib.load_config_from_path(path, log)
    def_params = space.default_params()
    sam, hard = space.model_for_params(def_params, sam_shape=args.sam_shape)

    # ---- Find simulation files and merge all data
    # ---------------------------------------------

    pattern = "library_sims/*.npz"
    files = list(path.glob(pattern))
    num_files = len(files)
    print(f"Found {num_files} matches to '{pattern}' in '{path}'")

    num_det_mqf = None
    num_all_mqf = None
    num_det_z = None
    num_all_z = None
    doppler_fobs_gw_cents = None
    fobs_gw_cents = None
    num_has_doppler = 0
    fits_det = []
    fits_undet = []
    for ii, fil in enumerate(tqdm(files)):
        data = np.load(fil, allow_pickle=True)
        keys = data.keys()
        if ii == 0:
            print(list(keys))

        if 'fail' in keys:
            # print(f"{ii=:04d} = failure | '{fil}'")
            continue
        fit = data['psd_fit']
        # print(fit)
        has_doppler = 'doppler_detect' in keys
        if not has_doppler:
            fits_undet.append(fit)
            continue

        fits_det.append(fit)
        doppler_fobs_gw_cents = data['doppler_fobs_gw_cents']
        fobs_gw_cents = data['fobs_cents']

        if num_det_mqf is None:
            mqf_shape = data['doppler_num_det'].shape
            num_det_mqf = np.zeros((num_files,) + mqf_shape)
            num_all_mqf = np.zeros_like(num_det_mqf)
            z_shape = data['doppler_num_det_redz'].shape
            num_det_z = np.zeros((num_files,) + z_shape)
            num_all_z = np.zeros_like(num_det_z)
            gwb_shape = data['gwb'].shape
            gwb_det = np.zeros((num_files,) + gwb_shape)

        num_det_mqf[num_has_doppler, ...] = data['doppler_num_det'][()]
        num_all_mqf[num_has_doppler, ...] = data['doppler_num_all'][()]
        num_det_z[num_has_doppler, ...] = data['doppler_num_det_redz'][()]
        num_all_z[num_has_doppler, ...] = data['doppler_num_all_redz'][()]
        gwb_det[num_has_doppler, ...] = data['gwb'][()]
        num_has_doppler += 1

    num_det_mqf = num_det_mqf[:num_has_doppler, ...]
    num_all_mqf = num_all_mqf[:num_has_doppler, ...]
    num_det_z = num_det_z[:num_has_doppler, ...]
    num_all_z = num_all_z[:num_has_doppler, ...]
    gwb_det = gwb_det[:num_has_doppler, ...]
    fits_undet = np.asarray(fits_undet)
    fits_det = np.asarray(fits_det)

    # ---- Save to File
    # -----------------

    doppler_fname = path.name + "_detected.npz"
    doppler_fname = path.joinpath(doppler_fname)

    """
    Values:
        # ---- Semi-Analytic Model
        mtot : (M+1,) total-mass bin edges [gram]
        mrat : (Q+1,) mass-ratio bin edges
        redz : (Z+1,) redshift (of galaxy merger)

        # ---- PTA calculations
        fobs_gw_cents : (Fp,) GW frequency-bin centers
        gwb : (S, Fp, R), GWB characteristic strain in PTA band
        fits : (S, 2) power-law fits to first 5 frequency bins of PTA GWB, 0=amplitude, 1=spectral-index

        # ---- Doppler calculations
        doppler_fobs_gw_cents : (Fd,) Doppler frequency-bin centers
        num_det_mqf : (S, M, Q, Fd) number of detected binaries in bins of mtot, mrat, freq (doppler GW frequency)
        num_det_z : (S, Z) number of detected binaries in bins of redz (of galaxy merger)

    Array shapes have the sizes:
        M : number of total-mass bins = 90
        Q : number of mass-ratio bins = 80
        Z : number of redshift bins = 100
        S : populations consistent with GWB observations = 140 (out of 1080 explored)
        Fp : PTA frequencies = 40
        R : number of realizations per population = 100
        Fd : Doppler frequencies = 200

    """

    np.savez(
        doppler_fname,

        # (M+1,) total-mass bin edges [gram]
        mtot=sam.mtot,
        # (Q+1,) mass-ratio bin edges
        mrat=sam.mrat,
        # (Z+1,) redshift (of galaxy merger)
        redz=sam.redz,

        # ---- PTA calculations
        # (Fp,) GW frequency-bin centers
        fobs_gw_cents=fobs_gw_cents,
        # (S, Fp, R), GWB characteristic strain in PTA band
        gwb=gwb_det,
        # (S, 2) power-law fits to first 5 frequency bins of PTA GWB, 0=amplitude, 1=spectral-index
        fits=fits_det,

        # ---- Doppler calculations
        # (Fd,) Doppler frequency-bin centers
        doppler_fobs_gw_cents=doppler_fobs_gw_cents,
        # (S, M, Q, Fd) number of detected binaries in bins of mtot, mrat, freq (doppler GW frequency)
        num_det_mqf=num_det_mqf,
        # (S, Z) number of detected binaries in bins of redz (of galaxy merger)
        num_det_z=num_det_z,

    )
    print(f"Saved to {doppler_fname}, {utils.get_file_size(doppler_fname)}")
    return


if __name__ == "__main__":

    paths = sys.argv[1:]
    for path in paths:
        main(path)

    print("\n\nDone.\n")