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

    first_pass = True

    dop_data = None
    # num_det_mqf = None
    # num_all_mqf = None
    # num_det_z = None
    # num_all_z = None

    doppler_fobs_gw_cents = None
    fobs_gw_cents = None
    num_has_doppler = 0
    fits_det = []
    fits_undet = []
    for ii, fil in enumerate(tqdm(files)):
        data = np.load(fil, allow_pickle=True)
        keys = data.keys()

        has_fail = 'fail' in keys
        if first_pass:
            print(f"{ii=:04d}: {has_fail=}")
        if has_fail:
            # print(f"{ii=:04d} = failure | '{fil}'")
            continue
        fit = data['psd_fit']
        has_doppler = np.any([kk.startswith("doppler_") for kk in keys])
        if first_pass:
            print(f"{ii=:04d}: {has_doppler=} :: {keys=}")
        if not has_doppler:
            fits_undet.append(fit)
            continue
        elif first_pass:
            first_pass = False

        fits_det.append(fit)
        doppler_fobs_gw_cents = data['doppler_fobs_gw_cents']
        fobs_gw_cents = data['fobs_cents']

        # ---- Determine Doppler settings
        doppler_key = "doppler_num_det_redz_"
        sel_keys = [kk for kk in keys if kk.startswith(doppler_key)]
        dop_pars = [kk.split(doppler_key)[-1].split("-SNR") for kk in sel_keys]
        # expect_list = ['optimistic', 'priority', 'base']
        # snr_list = [1.0, 3.0, 8.0]
        expect_list = []
        snr_list = []
        for dp in dop_pars:
            if dp[0] not in expect_list:
                expect_list.append(dp[0])
            if dp[1] not in snr_list:
                snr_list.append(dp[1])

        # ---- Initialize storage on the first (successful) pass

        def get_param_key(expect, snr):
            sk = f"{expect}-SNR{snr}"
            return sk

        if dop_data is None:
            for kk, vv in data.items():
                print(kk, np.shape(vv))

            gwb_shape = data['gwb'].shape
            gwb_det = np.zeros((num_files,) + gwb_shape)

            # vers_key = f"{expect_list[0]}_SNR{snr_list[0]}"
            vers_key = get_param_key(expect_list[0], snr_list[0])
            print(f"{vers_key=}")
            mqf_shape = data[f'doppler_num_det_{vers_key}'].shape
            z_shape = data[f'doppler_num_det_redz_{vers_key}'].shape

            mqf_shape = (num_files,) + mqf_shape
            z_shape = (num_files,) + z_shape
            store_shapes = [mqf_shape, z_shape]

            dop_data = {}
            dop_data['num_all_mqf'] = np.zeros(mqf_shape)
            dop_data['num_all_z'] = np.zeros(z_shape)
            for expect in expect_list:
                for snr in snr_list:
                    sk = get_param_key(expect, snr)
                    dop_data[f"num_det_mqf_{sk}"] = np.zeros(mqf_shape)
                    dop_data[f"num_det_z_{sk}"] = np.zeros(z_shape)

        # ---- Store data for this entry

        dop_data['num_all_mqf'][num_has_doppler, ...] = data[f"doppler_num_all"][()]
        dop_data['num_all_z'][num_has_doppler, ...] = data[f"doppler_num_all_redz"][()]

        for expect in expect_list:
            for snr in snr_list:
                sk = get_param_key(expect, snr)
                dop_data[f"num_det_mqf_{sk}"][num_has_doppler, ...] = data[f"doppler_num_det_{sk}"][()]
                dop_data[f"num_det_z_{sk}"][num_has_doppler, ...] = data[f"doppler_num_det_redz_{sk}"][()]

        gwb_det[num_has_doppler, ...] = data['gwb'][()]
        num_has_doppler += 1

    # ---- Trim arrays to used size
    dop_data['num_all_mqf'] = dop_data['num_all_mqf'][:num_has_doppler]
    dop_data['num_all_z'] = dop_data['num_all_z'][:num_has_doppler]

    for key, val in dop_data.items():
        print(f"{key=} {np.shape(val)=}")
        dop_data[key] = dop_data[key][:num_has_doppler, ...]

    for kk, vv in dop_data.items():
        print(f"{kk=:40s} :: {np.shape(vv)}")

    gwb_det = gwb_det[:num_has_doppler, ...]
    fits_undet = np.asarray(fits_undet)
    fits_det = np.asarray(fits_det)

    # ---- Save to File
    # -----------------

    # print("RETURNING BEFORE SAVING @*&(^9762134)")
    # return

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

        # # (S, M, Q, Fd) number of detected binaries in bins of mtot, mrat, freq (doppler GW frequency)
        # num_det_mqf=num_det_mqf,
        # # (S, Z) number of detected binaries in bins of redz (of galaxy merger)
        # num_det_z=num_det_z,
        **dop_data,
    )
    print(f"Saved to {doppler_fname}, {utils.get_file_size(doppler_fname)}")
    return


if __name__ == "__main__":

    paths = sys.argv[1:]
    for path in paths:
        main(path)

    print("\n\nDone.\n")