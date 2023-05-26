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


def Cl_analytic_from_num(fobs_orb_edges, number, hs, realize = False):
    """ Calculate Cl using Eq. (17) of Sato-Polito & Kamionkowski
    Parameters
    ----------
    fobs_orb_edges : (F,) 1Darray
        Observed orbital frequency bin edges
    hs : (M,Q,Z,F) NDarray
        Strain amplitude of each M,q,z bin
    number : (M,Q,Z,F) NDarray
        Number of sources in each M,q,z, bin
    realize : boolean or integer
        How many realizations to Poisson sample.
    
    Returns
    -------
    C0 : (F,R) or (F,) NDarray
        C_0 
    Cl : (F,R) or (F,) NDarray
        C_l>0 for arbitrary l using shot noise approximation
    """

    df = np.diff(fobs_orb_edges)                 #: frequency bin widths
    fc = kale.utils.midpoints(fobs_orb_edges)    #: frequency-bin centers 

    # df = fobs_orb_widths[np.newaxis, np.newaxis, np.newaxis, :] # (M,Q,Z,F) NDarray
    # fc = fobs_orb_cents[np.newaxis, np.newaxis, np.newaxis, :]  # (M,Q,Z,F) NDarray


    # Poisson sample number in each bin
    if utils.isinteger(realize):
        number = np.random.poisson(number[...,np.newaxis], 
                                size = (number.shape + (realize,)))
        df = df[...,np.newaxis]
        fc = fc[...,np.newaxis]
        hs = hs[...,np.newaxis]
    elif realize is True:
        number = holo.gravwaves.poisson_as_needed(number)



    delta_term = (fc/(4*np.pi*df) * np.sum(number*hs**2, axis=(0,1,2)))**2

    Cl = (fc/(4*np.pi*df))**2 * np.sum(number*hs**4, axis=(0,1,2))

    C0 = Cl + delta_term

    return C0, Cl