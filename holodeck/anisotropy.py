""" Module for predicting anisotropy with single source populations.

"""

import numpy as np
import matplotlib as plt
import matplotlib.cm as cm

import kalepy as kale
import healpy as hp

import holodeck as holo
from holodeck import utils, cosmo, log
from holodeck.constants import SPLC, NWTG, MPC

NSIDE = 32
NPIX = hp.nside2npix(NSIDE)
LMAX = 8

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



