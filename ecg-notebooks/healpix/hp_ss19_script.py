#!/usr/bin/env python3

import healpy as hp
import h5py

LIB_PATH = '/Users/emigardiner/GWs/holodeck/output/2023-05-12-mbp-ss16_n10_r10_f70_d12.5_l10_p0'
hdf_name = LIB_PATH+'/ss_lib.hdf5'
print('Hdf file:', hdf_name)

# settings to change
DS_NUM = '2B'
NPSRS = 60
SIGMA = 5e-7

NSIDE = 24
LMAX = 8
PLOT = True
DEBUG = True

output_dir = (LIB_PATH+'/detstats/ds%s_psrs%d_sigma%.0e'
              % (DS_NUM, NPSRS, SIGMA))
print('Output dir:', output_dir)

vals = ds.detect_lib(hdf_name, output_dir, NPSRS, SIGMA, 
                     nskies=NSKIES, thresh=THRESH, plot=PLOT, debug=DEBUG)










# functions
def healpix_map(hc_ss, hc_bg, nside=NSIDE):
    """ Build mollview array of strains for a healpix map
    
    Parameters
    ----------
    hc_ss : (F,R,L) NDarray
        Characteristic strain of single sources.
    hc_bg : (F,R) NDarray
        Characteristic strain of the background.
    nside : integer
        number of sides for healpix map.

    Returns
    -------
    moll_hc : (NPIX,) 1Darray
        Array of strain at every pixel for a mollview healpix map.
    
    NOTE: Could speed up the for-loops, but it's ok for now.
    """

    npix = hp.nside2npix(nside)
    nfreqs = len(hc_ss)
    nreals = len(hc_ss[0])
    nloudest = len(hc_ss[0,0])

    # spread background evenly across pixels in moll_hc
    moll_hc = np.ones((nfreqs,nreals,npix)) * hc_bg[:,:,np.newaxis]/np.sqrt(npix) # (frequency, realization, pixel)

    # choose random pixels to place the single sources
    pix_ss = np.random.randint(0, npix-1, size=nfreqs*nreals*nloudest).reshape(nfreqs, nreals, nloudest)
    for ff in range(nfreqs):
        for rr in range(nreals):
            for ll in range(nloudest):
                moll_hc[ff,rr,pix_ss[ff,rr,ll]] = np.sqrt(moll_hc[ff,rr,pix_ss[ff,rr,ll]]**2
                                                          + hc_ss[ff,rr,ll]**2)
                
    return moll_hc

def sph_harm_from_map(moll_hc, lmax=LMAX):
    """ Calculate spherical harmonics from strains at every pixel of 
    a healpix mollview map.
    
    Parameters
    ----------
    moll_hc : (F,R,NPIX,) 1Darray
        Characteristic strain of each pixel of a healpix map.
    lmax : int
        Highest harmonic to calculate.

    Returns
    -------
    Cl : (F,R,lmax+1) NDarray
        Spherical harmonic coefficients 
        
    """
    nfreqs = len(moll_hc)
    nreals = len(moll_hc[0])

    Cl = np.zeros((nfreqs, nreals, lmax+1))
    for ff in range(nfreqs):
        for rr in range(nreals):
            Cl[ff,rr,:] = hp.anafast(moll_hc[ff,rr], lmax=lmax)

    return Cl

def sph_harm_from_hc(hc_ss, hc_bg, nside = NSIDE, lmax = LMAX):
    """ Calculate spherical harmonics and strain at every pixel
    of a healpix mollview map from single source and background char strains.

    Parameters
    ----------
    hc_ss : (F,R,L) NDarray
        Characteristic strain of single sources.
    hc_bg : (F,R) NDarray
        Characteristic strain of the background.
    nside : integer
        number of sides for healpix map.

    Returns
    -------
    moll_hc : (F,R,NPIX,) 2Darray
        Array of strain at every pixel for a mollview healpix map.
    Cl : (F,R,lmax+1) NDarray
        Spherical harmonic coefficients 
    
    """
    moll_hc = healpix_map(hc_ss, hc_bg, nside)
    Cl = sph_harm_from_map(moll_hc, lmax)

    return moll_hc, Cl