#!/usr/bin/env python3

import holodeck as holo
from holodeck import detstats, anisotropy
from holodeck.constants import YR
import numpy as np
import healpy as hp
import h5py
import os

LIB_PATH = '/Users/emigardiner/GWs/holodeck/output/2023-05-16-mbp-ss18_n100_r40_d15_f30_l1000_p0'
HC_REF15_10YR = 11.2*10**-15 

NSIDE = anisotropy.NSIDE
LMAX = 6
NBEST = 100

def main():

    # ---- read in file
    hdf_name = LIB_PATH+'/ss_lib.hdf5'
    print('Hdf file:', hdf_name)

    ss_file = h5py.File(hdf_name, 'r')
    print('Loaded file, with keys:', list(ss_file.keys()))
    hc_ss = ss_file['hc_ss'][...]
    hc_bg = ss_file['hc_bg'][...]
    fobs = ss_file['fobs'][:]
    # dfobs = ss_file['dfobs'][:]
    ss_file.close()

    shape = hc_ss.shape
    nsamps, nfreqs, nreals, nloudest = shape[0], shape[1], shape[2], shape[3]
    print('N,F,R,L =', nsamps, nfreqs, nreals, nloudest)


    # ---- rank samples
    nsort, fidx, hc_tt, hc_ref15 = detstats.rank_samples(hc_ss, hc_bg, fobs, fidx=1, hc_ref=HC_REF15_10YR, ret_all=True)
    
    print('Ranked samples by hc_ref = %.2e at fobs = %.2f/yr' % (hc_ref15, fobs[fidx]*YR))

    # --- calculate spherical harmonics

    npix = hp.nside2npix(NSIDE)
    Cl_best = np.zeros((NBEST, nfreqs, nreals, LMAX+1 ))
    moll_hc_best = np.zeros((NBEST, nfreqs, nreals, npix))
    for nn in range(NBEST):
        print('on nn=%d out of nbest=%d' % (nn,NBEST))
        moll_hc_best[nn,...], Cl_best[nn,...] = anisotropy.sph_harm_from_hc(
            hc_ss[nsort[nn]], hc_bg[nsort[nn]], nside=NSIDE, lmax=LMAX, )

    # --- save to npz file
    output_dir = LIB_PATH+'/anisotropy'
    # Assign output folder
    if (os.path.exists(output_dir) is False):
        print('Making output directory.')
        os.makedirs(output_dir)
    else:
        print('Writing to an existing directory.')

    output_name = output_dir+'/sph_harm_lmax%d_nside%d_nbest%d.npz' % (LMAX, NSIDE, NBEST))
    print('Saving npz file: ', output_name)
    np.savez(output_name,
             nsort=nsort, fidx=fidx, hc_tt=hc_tt, hc_ref15=hc_ref15, ss_shape=shape,
         moll_hc_best=moll_hc_best, Cl_best=Cl_best, nside=NSIDE, lmax=LMAX, fobs=fobs)

    
if __name__ == "__main__":
    main()