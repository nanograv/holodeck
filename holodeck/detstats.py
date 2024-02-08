"""Detection Statistics module.

This module calculates detection statistics for single source and background strains
for Hasasia PTA's.

"""

import numpy as np
from scipy import special, integrate
from sympy import nsolve, Symbol
import h5py
import matplotlib.pyplot as plt
import os
from datetime import datetime
import warnings


import holodeck as holo
from holodeck import utils, cosmo, log, plot, anisotropy
from holodeck.constants import MPC, YR
from holodeck import cyutils 

import hasasia.sensitivity as hsen
import hasasia.sim as hsim
import hasasia as has

GAMMA_RHO_GRID_PATH = '/Users/emigardiner/GWs/holodeck/output/rho_gamma_grids'
HC_REF15_10YR = 11.2*10**-15 
DEF_THRESH=0.5



###################### Overlap Reduction Function ######################

def _gammaij_from_thetaij(theta_ij):
    """ Calculate gamma_ij for two pulsars of relative angle theta_ij.
    
    Parameters
    ----------
    theta_ij : scalar
        Relative angular position between the ith and jth pulsars.

    Returns
    -------
    gamma_ij : scalar
        [1 - cos(theta_ij)]/2

    """
    return (1-np.cos(theta_ij))/2


def _dirac_delta(i,j):
    """ Calculate the dirac delta function of i,j.
    Parameters
    ----------
    i : int
    j : int

    Returns
    -------
    dirac_ij : int
        Dirac delta function of i and j

    """
    if(i==j): return 1
    else: return 0

def _relative_angle(theta_i, phi_i, theta_j, phi_j):
    """ Calcualte relative angle between two pulsars i and j.

    Parameters
    ----------
    theta_i : scalar
        Polar (latitudinal) angular position in the sky of the ith pulsar.
    phi_i : scalar
        Azimuthal (longitudinal) angular position in the sky of the ith pulsar.
    theta_j : scalar
        Polar (latitudinal) angular position in the sky of the jth pulsar.
    phi_j : scalara
        Azimuthal (longitudinal) angular position in the sky of the jth pulsar.

    Returns
    -------
    theta_ij : scalar
        Relative angular position between the ith and jth pulsar.

    """

    theta_ij = np.arccos(np.cos(phi_i)*np.cos(phi_j)
                      + np.sin(phi_i)*np.sin(phi_j)*np.cos(theta_i - theta_j))

    return theta_ij

def _orf_ij(i, j, theta_ij):
    """ Calculate the overlap reduction function Gamma_i,j as a function of theta_i, theta_j, i, and j.

    Parameters
    ----------
    i : int
        index of the ith pulsar
    j : int
        index of the jth pulsar
    theta_ij : scalar
        Relative angular position between the ith and jth pulsars.

    Returns
    -------
    Gamma : scalar
        The overlap reduction function of the ith and jth pulsars.


    Follows Rosado et al. 2015 Eq. (24)
    """
    dirac_ij = _dirac_delta(i, j)
    gamma_ij = _gammaij_from_thetaij(theta_ij)

    Gamma = (3/2 * gamma_ij *np.log(gamma_ij)
            - 1/4 * gamma_ij
            + 1/2 + dirac_ij)
    if(np.isnan(Gamma) and i!=j):
        print('Gamma_%d,%d is nan, set to 0' % (i,j))
        return 0
    return Gamma


def _orf_pta(pulsars):
    """ Calculate the overlap reduction function matrix Gamma for a list of hasasia.Pulsar objects

    Paramters
    ---------
    pulsars : (P,) list of hasasia.Pulsar objects.

    Returns
    -------
    Gamma : (P,P) NDarray
        Overlap reduction function matrix for all pulsars i,j with j>i
        Only for j>1, 0 for j<=i

    """

    Gamma = np.zeros((len(pulsars), len(pulsars)))
    for ii in range(len(pulsars)):
        for jj in range(len(pulsars)):
            if (jj>ii): # 0 otherwise, allows sum over all
                # calculate angle between two vectors.
                theta_ij =  _relative_angle(pulsars[ii].theta, pulsars[ii].phi,
                                           pulsars[jj].theta, pulsars[jj].phi)
                # find ORF
                Gamma[ii,jj] = _orf_ij(ii, jj, theta_ij)

    return Gamma


######################## Noise Spectral Density ########################

def _white_noise(delta_t, sigma_i):
    """ Calculate the white noise for a given pulsar (or array of pulsars)
    2 * Delta_t sigma_i^2

    Parameters
    ----------
    delta_t : scalar
        Detection cadence, in seconds.
    sigma_i : arraylike
        Error/stdev/variance? for the ith pulsar, in seconds.

    Returns
    -------
    P_i : arraylike
        Noise spectral density for the ith pulsar, for bg detection.
        For single source detections, the noise spectral density S_i must also
        include red noise from all but the loudest single sources, S_h,rest.

    Follows Eq. (23) from Rosado et al. 2015.
    """
    P_i = 2 * delta_t * sigma_i**2
    return P_i



########################################################################
##################### Functions for the Background #####################
########################################################################

######################## Power Spectral Density ########################

def _power_spectral_density(hc_bg, freqs, reshape_freqs=True):
    """ Calculate the spectral density S_h(f_k) ~ S_h0(f_k) at the kth frequency

    Parameters
    ----------
    hc_bg : (F,R) NDarray of scalars
        Characteristic strain of the background at each frequency, for
        R realizations.
    freqs : (F,) 1Darray of scalars
        Frequency bin centers corresponding to each strain

    Returns
    -------
    S_h : (F,R) 1Darray of scalars
        Actual (S_h) or ~construction (S_h0) value of the background spectral density,
        in units of [freqs]^-3, for R realizations.

    Follows Eq. (25) of Rosado et al. 2015
    """
    if reshape_freqs:
        freqs = freqs[:,np.newaxis]

    S_h = hc_bg**2 / (12 * np.pi**2 * freqs**3)
    return S_h


######################## mu_1, sigma_0, sigma_1 ########################

def _sigma0_Bstatistic(noise, Gamma, Sh0_bg):
    """ Calculate sigma_1 for the background, by summing over all pulsars and frequencies.
    Assuming the B statistic, which maximizes S/N_B = mu_1/sigma_1

    Parameters
    ----------
    noise : (P,F,R) Ndarray of scalars
        Noise spectral density of each pulsar.
    Gamma : (P,P) 2Darray of scalars
        Overlap reduction function for j>i, 0 otherwise.
    Sh0_bg : (F,R) 1Darray of scalars
        Value of spectral density used to construct the statistic.

    Returns
    -------
    sigma_0B : (R,) 1Darray
        Standard deviation of the null PDF assuming the B-statistic.


    Follows Eq. (A17) from Rosado et al. 2015.
    """

    # Check that Gamma_{j<=i} =
    for ii in range(len(noise)):
        for jj in range(ii+1):
            assert Gamma[ii,jj] == 0, f'Gamma[{ii},{jj}] = {Gamma[ii,jj]}, but it should be 0!'

    # to get sum term in shape (P,P,F,R) we want:
    # Gamma in shape (P,P,1,1)
    # Sh0 and Sh in shape (1,1,F,R)
    # P_i in shape (P,1,F,R)
    # P_j in shape (1,P,F,R) 

    # Cast parameters to desired shapes
    Gamma = Gamma[:,:,np.newaxis,np.newaxis]
    Sh0_bg = Sh0_bg[np.newaxis,np.newaxis,:]
    noise_i = noise[:,np.newaxis,:,:]
    noise_j = noise[np.newaxis,:,:,:]

    # Calculate sigma_0B
    numer = (Gamma**2 * Sh0_bg**2
             * noise_i * noise_j)
    denom = ((noise_j + Sh0_bg)
              * (noise_j + Sh0_bg)
             + Gamma**2 * Sh0_bg**2)**2

    sum = np.sum(numer/denom, axis=(0,1,2))
    sigma_0B = np.sqrt(2*sum)
    return sigma_0B

def _sigma1_Bstatistic(noise, Gamma, Sh_bg, Sh0_bg):
    """ Calculate sigma_1 for the background, by summing over all pulsars and frequencies.
    Assuming the B statistic, which maximizes S/N_B = mu_1/sigma_1

    Parameters
    ----------
    noise : (P,F,R) 1darray of scalars
        Noise spectral density of each pulsar.
    Gamma : (P,P) 2Darray of scalars
        Overlap reduction function for j>i, 0 otherwise.
    Sh_bg : (F,R) 1Darray of scalars
        Spectral density in the background.
    Sh0_bg : (F,R) 1Darray of scalars
        Value of spectral density used to construct the statistic.
s
    Returns
    -------
    sigma_1B : (R,) 1Darray
        Standard deviation of the PDf with a GWB, assuming the B-statistic.


    Follows Eq. (A18) from Rosado et al. 2015.
    """

    # Check that Gamma_{j<=i} =
    for ii in range(len(noise)):
        for jj in range(ii+1):
            assert Gamma[ii,jj] == 0, f'Gamma[{ii},{jj}] = {Gamma[ii,jj]}, but it should be 0!'

    # to get sum term in shape (P,P,F,R) we want:
    # Gamma in shape (P,P,1,1)
    # Sh0 and Sh in shape (1,1,F,R)
    # P_i in shape (P,1,1,1)
    # P_j in shape (1,P,1,1)

    # Cast parameters to desired shapes
    Gamma = Gamma[:,:,np.newaxis,np.newaxis]
    Sh0_bg = Sh0_bg[np.newaxis,np.newaxis,:]
    Sh_bg = Sh_bg[np.newaxis,np.newaxis,:]
    noise_i = noise[:,np.newaxis,:,:]
    noise_j = noise[np.newaxis,:,:,:]


    # Calculate sigma_1B
    numer = (Gamma**2 * Sh0_bg**2
             * ((noise_i + Sh_bg) * (noise_j + Sh_bg)
                + Gamma**2 * Sh_bg**2))

    denom = ((noise_i + Sh0_bg)
              * (noise_j + Sh0_bg)
             + Gamma**2 * Sh0_bg**2)**2

    sum = np.sum(numer/denom, axis=(0,1,2))
    sigma_1B = np.sqrt(2*sum)
    return sigma_1B

def _mean1_Bstatistic(noise, Gamma, Sh_bg, Sh0_bg, debug=False):
    """ Calculate mu_1 for the background, by summing over all pulsars and frequencies.
    Assuming the B statistic, which maximizes S/N_B = mu_1/sigma_1

    Parameters
    ----------
    noise : (P,F,R) Ndarray of scalars
        Noise spectral density of each pulsar.
    Gamma : (P,P) 2Darray of scalars
        Overlap reduction function for j>i, 0 otherwise.
    Sh_bg : (F,R) 2Darray of scalars
        Spectral density in the background.
    Sh0_bg : (F,R) 2Darray of scalars
        Value of spectral density used to construct the statistic.

    Returns
    -------
    mu_1B : (R) 1Darray of scalars
        Expected value for the B statistic

    Follows Eq. (A16) from Rosado et al. 2015.
    """

    # Check that Gamma_{j<=i} =
    if debug:
        for ii in range(len(Gamma)):
            for jj in range(ii+1):
                assert Gamma[ii,jj] == 0, f'Gamma[{ii},{jj}] = {Gamma[ii,jj]}, but it should be 0!'

    # to get sum term in shape (P,P,F,R) for ii,jj,kk we want:
    # Gamma in shape (P,P,1,1)
    # Sh0 and Sh in shape (1,1,F,R)
    # P_i in shape (P,1,1,1)
    # P_j in shape (1,P,1,1)

    # Cast parameters to desired shapes
    Gamma = Gamma[:,:,np.newaxis,np.newaxis]
    Sh0_bg = Sh0_bg[np.newaxis,np.newaxis,:]
    Sh_bg = Sh_bg[np.newaxis,np.newaxis,:]
    noise_i = noise[:,np.newaxis,:,:]
    noise_j = noise[np.newaxis,:,:,:]


    # Calculate mu_1B
    numer = (Gamma **2 * Sh_bg * Sh0_bg)
    denom = ((noise_i + Sh0_bg) * (noise_j + Sh0_bg) + Gamma**2 * Sh0_bg**2)

    # Requires Gamma have all jj<=ii parts to zero
    sum = np.sum(numer/denom, axis=(0,1,2))
    mu_1B = 2*sum
    return mu_1B




######################## Detection Probability #########################

def _bg_detection_probability(sigma_0, sigma_1, mu_1, alpha_0=0.001):
    """ Calculate the background detection probability, gamma_bg.

    Parameters
    ----------
    sigma_0 : (R,) 1Darray
        Standard deviation of stochastic noise processes, for R realizations.
    sigma_1 : (R,) 1Darray
        Standard deviation of GWB PDF, for R realizations.
    mu_1 : (R,) 1Darray
        Mean of GWB PDF, for R realizations.
    alpha_0 : scalar
        False alarm probability max.

    Returns
    -------
    dp_bg : (R,) 1Darray
        Background detection probability, for R realizations.


    Follows Rosado et al. 2015 Eq. (15)
    """
    alpha_0 = np.array([alpha_0])
    temp = ((np.sqrt(2) * sigma_0 * special.erfcinv(2*alpha_0) - mu_1)
            /(np.sqrt(2) * sigma_1))
    dp_bg = .5 * special.erfc(temp)
    return dp_bg


def detect_bg(thetas, phis, sigmas, fobs, cad, hc_bg, alpha_0=0.001, ret = False):
    """ Calculate the background detection probability, and all intermediary steps.

    Parameters
    ----------
    thetas : (P,) 1Darray of scalars
        Angular position of each pulsar in radians.
    phis : (P,) 1Darray of scalars
        Angular position of each pulsar in radians.
    sigmas : (P,) 1Darray of scalars
        Sigma_i of each pulsar in seconds.
    fobs : (F,) 1Darray of scalars
        Frequency bin centers in hertz.
    cad : scalar
        Cadence of observations in seconds.
    hc_bg : (F,R)
        Characteristic strain of the background at each frequency,
        for R realizations.
    alpha_0 : scalar
        False alarm probability
    ret : Bool
        Whether to return all parameters or just dp_bg

    Returns
    -------
    dp_bg : (R,) 1Darray
        Background detection probability, for R realizations.
    Gamma : (P, P) 2D Array
        Overlap reduction function for j>i, 0 otherwise.
        Only returned if return = True.
    Sh_bg : (F,R) 1Darray
        Spectral density, for R realizations
        Only returned if return = True.
    noise : (P,) 1Darray
        Spectral noise density of each pulsar.
        Only returned if return = True.
    mu_1B : (R,) 1Darray
        Expected value for the B statistic.
        Only returned if return = True.
    sigma_0B : (R,) 1Darray
    sigma_1B : (R,) 1Darray

    TODO: Update or deprecate.
    """

    print("Detect_bg() is deprecated. Use detect_bg_pta() instead for red noise and ss noise.")
    
    # Overlap Reduction Function
    num = len(thetas) # number of pulsars, P
    Gamma = np.zeros((num, num)) # (P,P) 2Darray of scalars, Overlap reduction function between all puolsar
    for ii in range(num):
        for jj in range(num):
            if (jj>ii): # 0 otherwise, allows sum over all
                theta_ij =  _relative_angle(thetas[ii], phis[ii],
                                            thetas[jj], phis[jj])
                Gamma[ii,jj] = _orf_ij(ii, jj, theta_ij)

    # Spectral Density
    Sh_bg = _power_spectral_density(hc_bg, fobs) # spectral density of bg, using 0th realization
    Sh0_bg = Sh_bg # appropsimation used in Rosado et al. 2015
    # print('Sh_bg:', Sh_bg.shape)

    # Noise 
    noise = _white_noise(cad, sigmas)[:,np.newaxis, np.newaxis] # P, 1, 1
    # print('noise:', noise.shape)

    sigma_0B = _sigma0_Bstatistic(noise, Gamma, Sh0_bg)
    # print('sigma_0B:', sigma_0B.shape)

    sigma_1B = _sigma1_Bstatistic(noise, Gamma, Sh_bg, Sh0_bg)
    # print('sigma_1B:', sigma_1B.shape)

    mu_1B = _mean1_Bstatistic(noise, Gamma, Sh_bg, Sh0_bg)
    # print('mu_1B:', mu_1B.shape)

    dp_bg = _bg_detection_probability(sigma_0B, sigma_1B, mu_1B, alpha_0)
    # print('dp_bg', dp_bg.shape)

    if(ret):
        return dp_bg, Gamma, Sh_bg, noise, mu_1B, sigma_0B, sigma_1B
    else:
        return dp_bg





def detect_bg_pta(pulsars, fobs, hc_bg, hc_ss=None, custom_noise=None,
                  alpha_0=0.001, ret_snr = False,
                  red_amp=None, red_gamma=None, ss_noise=False):
    """ Calculate the background detection probability, and all the intermediary steps
    from a list of hasasia.Pulsar objects.

    Parameters
    ----------
    pulsars : (P,) list of hasasia.Pulsar objects
        A set of pulsars generated by hasasia.sim.sim_pta()
    fobs : (F,) 1Darray of scalars
        Frequency bin centers in hertz.
    hc_bg : (F,R)
        Characteristic strain of the background at each frequency,
        for R realizations.
    alpha_0 : scalar
        Falsa alarm probability
    return : Bool
        Whether or not to return intermediate variables.

    Returns
    -------
    dp_bg : (R,) 1Darray
        Background detection probability
    snr_bg : (R,) 1Darray
        Signal to noise ratio of the background, using the
        B statistic. 


    If a pulsar had differing toaerrs, the mean of that pulsar's
    toaerrs is used as the pulsar's sigma.
    TODO: implement red noise
    """

    cad = 1.0/(2*fobs[-1])

    # get pulsar properties
    thetas = np.zeros(len(pulsars))
    phis = np.zeros(len(pulsars))
    sigmas = np.zeros(len(pulsars))
    for ii in range(len(pulsars)):
        thetas[ii] = pulsars[ii].theta
        phis[ii] = pulsars[ii].phi
        sigmas[ii] = np.mean(pulsars[ii].toaerrs)


    Gamma = _orf_pta(pulsars)

    Sh_bg = _power_spectral_density(hc_bg[:], fobs)
    Sh0_bg = Sh_bg # note this refers to same object, not a copy

    # noise spectral density
    if custom_noise is not None:
        if custom_noise.shape != (len(pulsars), len(fobs), len(hc_bg[0])):
            err = f"{custom_noise.shape=}, must be shape (P,F,R)=({len(pulsars)}, {len(fobs)}, {len(hc_bg[0])})"
            raise ValueError(err)
        noise = custom_noise
    else:
        # calculate white noise
        noise = _white_noise(cad, sigmas)[:,np.newaxis] # P,1

        # add red noise
        if (red_amp is not None) and (red_gamma is not None):
            red_noise = _red_noise(red_amp, red_gamma, fobs)[np.newaxis,:] # (1,F,)
            noise = noise + red_noise # (P,F,)

        # add single source noise
        noise = noise[:,:,np.newaxis]
        if ss_noise:
            noise = noise + _Sh_ss_noise(hc_ss, fobs) # (P, F, R) 

    mu_1B = _mean1_Bstatistic(noise, Gamma, Sh_bg, Sh0_bg)

    sigma_0B = _sigma0_Bstatistic(noise, Gamma, Sh0_bg)

    sigma_1B = _sigma1_Bstatistic(noise, Gamma, Sh_bg, Sh0_bg)

    dp_bg = _bg_detection_probability(sigma_0B, sigma_1B, mu_1B, alpha_0)

    if(ret_snr):
        snr_bg = mu_1B/sigma_1B
        return dp_bg, snr_bg
    else:
        return dp_bg



######################## Signal-to-Noise Ratio ########################

def snr_bg_B(noise, Gamma, Sh_bg):
    """ Calculate S/N_B for the background, using P_i, Gamma, S_h and S_h0

    Parameters
    ----------
    noise : (P,) 1darray of scalars
        Noise spectral density of each pulsar, P_i.
    Gamma : (P,P) 2Darray of scalars
        Overlap reduction function for j>i, 0 otherwise.
    Sh_bg : (F,R) 2Darray of scalars
        Spectral density in the background.

    Returns
    -------
    snr_B : (R,) 1Darray of scalars
        Signal to noise ratio assuming the B statistic, mu_1B/sigma_1B, for each realization.


    Follows Eq. (A19) from Rosado et al. 2015. This should be equal to
    mu_1B/sigma_1B, and can be used as a sanity check.
    """


    # to get sum term in shape (P,P,F,R) we want:
    # Gamma in shape (P,P,1,1)
    # Sh in shape (1,1,F,R)
    # P_i in shape (P,1,1,1)
    # P_j in shape (1,P,1,1)

    P_i = noise[:,np.newaxis,np.newaxis,np.newaxis]
    P_j = noise[np.newaxis,:,np.newaxis,np.newaxis]
    Sh_bg = Sh_bg[np.newaxis,np.newaxis,:,:]
    Gamma = Gamma[:,:,np.newaxis,np.newaxis]

    numer = Gamma**2 * Sh_bg**2
    denom = (P_i*P_j + Sh_bg*(P_i+P_j) + Sh_bg**2 *(1 + Gamma**2))

    sum = np.sum(numer/denom, axis=(0,1,2))
    snr_B = np.sqrt(2*sum)
    return snr_B

def _Sh_hasasia_noise_bg(scGWB):
    """ Calculate the noise strain power spectral density,
        `Sh` for hasasia's SNR calculation

    Parameters
    ----------
    scGWB : hasasia.sensitivity.GWBSensitivityCurve object
        GWB sensitivity curve object.

    Returns
    -------
    Sh_h : (F,) 1Darray
        Sh as used in hasasia's SNR calculation, for each frequency.

    This function may not be working as we expect, since it does not produce SNR
    of noise to be 1.
    """
    freqs = scGWB.freqs
    H0 = scGWB._H_0.to('Hz').value
    Omega_gw = scGWB.Omega_gw
    Sh_h = 3*H0**2 / (2*np.pi**2) * Omega_gw / freqs**3
    return Sh_h

def snr_hasasia_noise_bg(scGWB):
    """ Calculate the effective noise signal to noise ratio with hasasia.

    Parameters
    ----------
    scGWB : hasasia.sensitivity.GWBSensitivityCurve object
        GWB sensitivity curve object.

    Returns
    -------
    snr_h : scalar
        Signal to noise ratio from hasasia.

    This function may not be working as we expect, since it does not produce SNR
    of noise to be 1.
    """
    Sh_h = _Sh_hasasia_noise_bg(scGWB)
    snr_h = scGWB.SNR(Sh_h)
    return snr_h
    

def _Sh_hasasia_modeled_bg(freqs, hc_bg):
    """ Calculate Sh for hsen.GWBSensitivityCurve.SNR(Sh) from a
    modeled GWB characteristic strain.

    Parameters
    ----------
    freqs : (F,) 1Darray
        Frequencies of char strain.
    hc_bg : (F,R) NDarray
        GWB characteristic strain for each frequency and realization.

    Returns
    -------
    Sh_h : (F,R) NDarray
        Sh as used in hasasia's SNR calculation, for each frequency.
    """

    Sh_h = hc_bg**2 / freqs[:,np.newaxis]
    return Sh_h

def snr_hasasia_modeled_bg(scGWB, hc_bg):
    """ Calculate the GWB signal to noise ratio with hasasia.

    Parameters
    ----------
    scGWB : hasasia.sensitivity.GWBSensitivityCurve object
        GWB sensitivity curve object.
    hc_bg : (F,R) NDarray
        Realistic characteristic strain of the background.

    Returns
    -------
    snr_h : (R,) 1Darray)
        Signal to noise ratio from hasasia, for each realization.
    """
    Sh_h = _Sh_hasasia_modeled_bg(scGWB.freqs, hc_bg)
    snr_h = np.zeros(len(hc_bg[0]))
    for rr in range(len(hc_bg[0])):
        snr_h[rr] = scGWB.SNR(Sh_h[:,rr])
    return snr_h
    



########################################################################
##################### Functions for Single Sources #####################
########################################################################

########################### Unitary Vectors  ###########################

def _m_unitary_vector(theta, phi, xi):
    """ Calculate the unitary vector m-hat for the antenna pattern functions
    for each of S sky realizations.

    Parameters
    ----------
    theta : (F,S,L) NDarray
        Spherical coordinate position of each single source.
    phi : (F,S,L) NDarray
        Spherical coordinate position of each single source.
    xi : (F,S,L) NDarray
        Inclination of binary? But thought that's what iota was?

    Returns
    -------
    m_hat : (3,F,S,L) NDarray
        Unitary vector m-hat with x, y, and z components at
        index 0, 1, and 2, respectively.

    """
    mhat_x = (np.sin(phi) * np.cos(xi)
              - np.sin(xi) * np.cos(phi) * np.cos(theta))
    mhat_y = -(np.cos(phi) * np.cos(xi)
               + np.sin(xi) * np.sin(phi) * np.cos(theta))
    mhat_z = (np.sin(xi) * np.sin(theta))

    m_hat = np.array([mhat_x, mhat_y, mhat_z])
    return m_hat

def _n_unitary_vector(theta, phi, xi):
    """ Calculate the unitary vector n-hat for the antenna pattern functions
    for each of S sky realizations.

    Paramters
    ---------
    theta : (F,S,L) NDarray
        Spherical coordinate position of each single source.
    phi : (F,S,L) NDarray
        Spherical coordinate position of each single source.
    xi : (F,S,L) 1Darray
        Inclination of binary? But thought that's what iota was?

    Returns
    -------
    n_hat : (3,F,R,L) NDarray
        Unitary vector n-hat.

    """

    nhat_x = (- np.sin(phi) * np.sin(xi)
              - np.cos(xi) * np.cos(phi) * np.cos(theta))
    nhat_y = (np.cos(phi) * np.sin(xi)
              - np.cos(xi) * np.sin(phi) * np.cos(theta))
    nhat_z = np.cos(xi) * np.sin(theta)

    n_hat = np.array([nhat_x, nhat_y, nhat_z])
    return n_hat

def _Omega_unitary_vector(theta, phi):
    """ Calculate the unitary vector n-hat for the antenna pattern functions
    for each of S sky realizations.

    Paramters
    ---------
    theta : (F,S,L) NDarray
        Spherical coordinate position of each single source.
    phi : (F,S,L) NDarray
        Spherical coordinate position of each single source.

    Returns
    -------
    Omega_hat : (3,F,R,L) NDarray
        Unitary vector Omega-hat.
    """

    Omegahat_x = - np.sin(theta) * np.cos(phi)
    Omegahat_y = - np.sin(theta) * np.sin(phi)
    Omegahat_z = - np.cos(theta)

    Omega_hat = np.array([Omegahat_x, Omegahat_y, Omegahat_z])
    return Omega_hat

def _pi_unitary_vector(phi_i, theta_i):
    """ Calculate the unitary vector p_i-hat for the ith pulsar.

    Parameters
    ----------
    phi : (P,) 1Darray
        Spherical coordinate position of each pulsar.
    theta : (P,) 1Darray
        Spherical coordinate position of each pulsar.
    Returns
    -------
    pi_hat : (3,P) vector
        pulsar unitary vector

    """

    pihat_x = np.sin(theta_i) * np.cos(phi_i)
    pihat_y = np.sin(theta_i) * np.sin(phi_i)
    pihat_z = np.cos(theta_i)

    pi_hat = np.array([pihat_x, pihat_y, pihat_z])
    return pi_hat


###################### Antenna Pattern Functions  ######################

def dotprod(vec1, vec2):
    """ Calculate the dot product for NDarrays of 3D vectors, with
     vector elements specified by the first index.

     Parameters
     ----------
     vec1 : (3,N1,N2,N3,...N) NDarray
     vec2 : (3,N1,N2,N3,...N) NDarray

     Returns
     -------
     dotted : (N1,N2,N3,...N) NDarray
        The dot product of the vectors specified by the first dimension,
        for every N1, N2, N3,...N.

    Example: find the dot product of 3D vectors for every P,F,R, using NDarrays
    of shape (3,P,F,R)
     """

    dotted = vec1[0,...]*vec2[0,...] + vec1[1,...]*vec2[1,...] + vec1[2,...]*vec2[2,...]
    return dotted


def _antenna_pattern_functions(m_hat, n_hat, Omega_hat, pi_hat):
    """ + antenna pattern function for the ith pulsar.

    Parameters
    ----------
    m_hat : (3,F,S,L) NDarray
        Single source m_hat unitary vector for each frequency and realization.
    n_hat : (3,F,S,L) NDarray
        Single source mnhat unitary vector for each frequency and realization.
    Omega_hat : (3,F,S,L) NDarray
        Single source Omega_hat unitary vector for each frequency and realization.
    pi_hat : (3,P) NDarray
        Pulsar term unitary vector for the ith pulsar.

    Returns
    -------
    F_iplus : (P,F,S,L) NDarray
        Plus antenna pattern function for each pulsar and binary of each realization.
    F_icross : (P,F,S,L) NDarray
        Cross antenna pattern function for each pulsar and binary of each realization.

    """
    mh = m_hat[:,np.newaxis,:,:]
    nh = n_hat[:,np.newaxis,:,:]
    Oh = Omega_hat[:,np.newaxis,:,:]
    ph = pi_hat[:,:,np.newaxis,np.newaxis,np.newaxis]
    denom = 1 + dotprod(Oh, ph)
    F_iplus = ((dotprod(mh, ph)**2 - dotprod(nh, ph)**2)
               / denom / 2)
    F_icross = dotprod(mh, ph) * dotprod(nh, ph) / denom

    return F_iplus, F_icross


######################## Noise Spectral Density ########################


def _Sh_rest_noise(hc_ss, hc_bg, freqs, nexcl=0):
    """ Calculate the noise spectral density contribution from all but the current single source.

    Parameters
    ----------
    hc_ss : (F,R,L) NDarray
        Characteristic strain from all loud single sources.
    hc_bg : (F,R) NDarray
        Characteristic strain from all but loudest source at each frequency.
    freqs : (F,) 1Darray
        Frequency bin centers.
    exclude_loudest : int
        Number of loudest single sources to exclude from hc_rest noise, in addition 
        to the source in question.

    Returns
    -------
    Sh_rest : (F,R,L) NDarray of scalars
        The noise in a single pulsar from other GW sources for detecting each single source.

    Follows Eq. (45) in Rosado et al. 2015.
    """

    if nexcl>0:
        Sh_rest = cyutils.Sh_rest(hc_ss, hc_bg, freqs, nexcl)
    else:
        hc2_louds = np.sum(hc_ss**2, axis=2) # (F,R)
        # subtract the single source from rest of loud sources and the background, for each single source
        hc2_rest = hc_bg[:,:,np.newaxis]**2 + hc2_louds[:,:,np.newaxis] - hc_ss**2 # (F,R,L)
        Sh_rest = hc2_rest / freqs[:,np.newaxis,np.newaxis]**3 /(12 * np.pi**2) # (F,R,L)
    
    return Sh_rest


def _Sh_ss_noise(hc_ss, freqs):
    """ Calculate the noise spectral density contribution from all but the first loudest 
    single sources.

    Parameters
    ----------
    hc_ss : (F,R,L) NDarray
        Characteristic strain from all loud single sources.
    hc_bg : (F,R) NDarray
        Characteristic strain from all but loudest source at each frequency.
    freqs : (F,) 1Darray
        Frequency bin centers.

    Returns
    -------
    Sh_ss : (P,F,R) NDarray of scalars
        The noise in a single pulsar from other GW sources for detecting each single source.

    Follows Eq. (45) in Rosado et al. 2015.
    """

    # sum of noise from all loudest single sources
    hc2_ss = np.sum(hc_ss[...,1:]**2, axis=2) # (F,R)
    Sh_ss = hc2_ss / freqs[:,np.newaxis]**3 /(12 * np.pi**2) # (F,R,)
    return Sh_ss


def _red_noise(red_amp, red_gamma, freqs, f_ref=1/YR):
    """ Calculate the red noise for a given pulsar (or array of pulsars)
    red_amp * f sigma_i^red_gamma

    Parameters
    ----------
    red_amp : scalar
        Amplitude of red noise.
    red_gamma : scalar
        Power-law index of red noise
    freqs : (F,) 1Darray of scalars
        Frequency bin centers.

    Returns
    -------
    P_red : (P,F) NDarray
        Red noise spectral density for the ith pulsar.

    Defined by Eq. (8) in Kelley et al. 2018
    ### what is f_ref

    """
    P_red = red_amp**2 / (12*np.pi**2) * (freqs/f_ref)**red_gamma * (f_ref)**-3
    return P_red



def _total_noise(delta_t, sigmas, hc_ss, hc_bg, freqs, red_amp=None, red_gamma=None,
                 nexcl=0):
    """ Calculate the noise spectral density of each pulsar, as it pertains to single
    source detections, i.e., including the background as a noise source.

    Parameters
    ----------
    delta_t : scalar
        Detection cadence, in seconds.
    sigmas : (P,) 1Darray
        Variance for the ith pulsar, in seconds
    hc_ss : (F,R,L) NDarray
        Characteristic strain from all loud single sources.
    hc_bg : (F,R) NDarray
        Characteristic strain from all but loudest source at each frequency.
    freqs : (F,) 1Darray
        Frequency bin centers.
    exclude_loudest : int
        Number of loudest single sources to exclude from hc_rest noise

    Returns
    -------
    noise : (P,F,R,L) NDarray of scalars
        The total noise in each pulsar for detecting each single source

    Follows Eq. (44) in Rosado et al. 2015.
    """

    noise = _white_noise(delta_t, sigmas) # (P,)
    Sh_rest = _Sh_rest_noise(hc_ss, hc_bg, freqs, nexcl) # (F,R,L,)
    noise = noise[:,np.newaxis,np.newaxis,np.newaxis] + Sh_rest[np.newaxis,:,:,:] # (P,F,R,L)
    if (red_amp is not None) and (red_gamma is not None):
        red_noise = _red_noise(red_amp, red_gamma, freqs) # (F,)
        noise = noise + red_noise[np.newaxis,:,np.newaxis,np.newaxis] # (P,F,R,L)
    return noise


def psrs_spectra_gwbnoise(psrs, fobs, nreals, npsrs, divide_flag=False):
    """ Get GWBSensitivityCurve noise and spectra for psrs
    
    Returns
    -------
    spectra : 
    noise_gsc : (P,F,R)
    """
    spectra = []
    for psr in psrs:
        sp = hsen.Spectrum(psr, freqs=fobs)
        sp.NcalInv
        spectra.append(sp)
    sc_bg = hsen.GWBSensitivityCurve(spectra).h_c
    noise_gsc = sc_bg**2 / (12 *np.pi**2 *fobs**3)
    noise_gsc = np.repeat(noise_gsc, npsrs*nreals).reshape(len(fobs), npsrs, nreals) # (F,P,R)
    noise_gsc = np.swapaxes(noise_gsc, 0, 1) # (P,F,R)

    if divide_flag: noise_gsc *= npsrs*(npsrs-1)

    return spectra, noise_gsc

def _dsc_noise(fobs, nreals, npsrs, nloudest, psrs=None, spectra=None, divide_flag=False):
    """ Get DeterSensitivityCurve noise using either psrs or spectra

    Returns
    -------
    noise_dsc : (P,F,R,L) NDarray
    """

    if spectra is None:
        assert psrs is not None, 'Must provide spectra or psrs'
        spectra = []
        for psr in psrs:
            sp = hsen.Spectrum(psr, freqs=fobs)
            sp.NcalInv
            spectra.append(sp)
    sc_ss = hsen.DeterSensitivityCurve(spectra).h_c
    noise_dsc = sc_ss**2 / (12 *np.pi**2 *fobs**3)
    noise_dsc = np.repeat(noise_dsc, npsrs*nreals*nloudest).reshape(len(fobs), npsrs, nreals, nloudest) # (F,P,R,L)
    noise_dsc = np.swapaxes(noise_dsc, 0, 1) # (P,F,R,L)

    if divide_flag: noise_dsc *= npsrs*(npsrs-1)
    return noise_dsc


################### GW polarization, phase, amplitude ###################

def _a_b_polarization(iotas):
    """ Polarization contribution variables a and b.

    Parameters
    ----------
    iotas : scalar or NDarray
        Typically will be (F,R,L) NDarray

    Returns
    -------
    a_pol : scalar or NDarray
        Same shape as iota
    b_pol : scalar or NDarray
        Same shape as iota

    """
    a_pol = 1 + np.cos(iotas) **2
    b_pol = -2 * np.cos(iotas)
    return a_pol, b_pol

def _gw_phase(dur, freqs, Phi_0):
    """ Calculate the detected gravitational wave phase at each frequency.

    Parameters
    ----------
    dur : scalar
        Duration (time elapsed from initial phase to detection)
    freqs : (F,) 1Darray
        Frequency of each single source.
    Phi_0 : (F,R,L) NDarray
        Initial GW phase of each binary.

    Returns
    -------
    Phi_T : (F,R,L) NDarray
        Detected GW phase of each single source.

    Follows Eq. (47) in Rosado et al. 2015
    """

    Phi_T = 2 * np.pi * freqs[:,np.newaxis,np.newaxis] * dur + Phi_0
    return Phi_T


def _amplitude(hc_ss, fobs, dfobs):
    """ Calculate the amplitude from the single source to use in DP calculations

    Parameters
    ----------
    hc_ss : (F,R,L) NDarray
        Characteristic strain of each single source at each realization.
    fobs : (F,) 1Darray
        Observer frame frequency. This can be orbital or gw frequency,
        it just has to match dfobs.
    dfobs_orb : (F,) 1Darray
        Observer frame frequency bin widths. This can be orbital or gw frequency,
        it just has to match dfobs.

    Returns
    -------
    Amp : (F,R,L)
        Dimensionless amplitude, A, of each single source at each frequency and realization.

    """

    Amp = hc_ss * np.sqrt(10) / 4 *np.sqrt(dfobs[:,np.newaxis,np.newaxis]/fobs[:,np.newaxis,np.newaxis])
    return Amp


####################### SS Signal to Noise Ratio  #######################

def _snr_ss(amp, F_iplus, F_icross, iotas, dur, Phi_0, S_i, freqs):
    """ Calculate the SNR for each pulsar wrt each single source detection,
    for S sky realizations and R strain realizations.

    Paramters
    ---------
    amp : (F,R,L) NDarray
        Dimensionless strain amplitude for loudest source at each frequency.
    F_iplus : (P,F,S,L) NDarray
        Antenna pattern function for each pulsar.
    F_icross : (P,F,S,L) NDarray
        Antenna pattern function for each pulsar.
    iotas : (F,S,L) NDarray
        Is this inclination? or what?
        Gives the wave polarizations a and b.
    dur : scalar
        Duration of observations.
    Phi_0 : (F,S,L) NDarray
        Initial GW Phase.
    S_i : (P,F,R,L) NDarray
        Total noise of each pulsar wrt detection of each single source, in s^3
    freqs : (F,) 1Darray
        Observed frequency bin centers.

    Returns
    -------
    snr_ss : (F,R,S,L) NDarray
        SNR from the whole PTA for each single source with each realized sky position (S)
        and realized strain (R).

    """


    snr_ss = cyutils.snr_ss(amp, F_iplus, F_icross, iotas, dur, Phi_0, S_i, freqs)
    return snr_ss

def _snr_ss_5dim(amp, F_iplus, F_icross, iotas, dur, Phi_0, S_i, freqs):
    """ Calculate the SNR for each pulsar wrt each single source detection,
    for S sky realizations and R strain realizations.

    Paramters
    ---------
    amp : (F,R,L) NDarray
        Dimensionless strain amplitude for loudest source at each frequency.
    F_iplus : (P,F,S,L) NDarray
        Antenna pattern function for each pulsar.
    F_icross : (P,F,S,L) NDarray
        Antenna pattern function for each pulsar.
    iotas : (F,S,L) NDarray
        Is this inclination? or what?
        Gives the wave polarizations a and b.
    dur : scalar
        Duration of observations.
    Phi_0 : (F,S,L) NDarray
        Initial GW Phase.
    S_i : (P,F,R,L) NDarray
        Total noise of each pulsar wrt detection of each single source, in s^3
    freqs : (F,) 1Darray

    Returns
    -------
    snr_ss : (F,R,S,L) NDarray
        SNR from the whole PTA for each single source with each realized sky position (S)
        and realized strain (R).

    """

    amp = amp[np.newaxis,:,:,np.newaxis,:]  # (F,R,L) to (P,F,R,S,L)

    a_pol, b_pol = _a_b_polarization(iotas) # (F,S,L)
    a_pol = a_pol[np.newaxis,:,np.newaxis,:,:] # (F,S,L) to (1,F,1,S,L)
    b_pol = b_pol[np.newaxis,:,np.newaxis,:,:] # (F,S,L) to (1,F,1,S,L)

    Phi_T = _gw_phase(dur, freqs, Phi_0) # (F,)
    Phi_T = Phi_T[np.newaxis,:,np.newaxis,:,:] # (F,S,L) to (1,F,1,S,L)

    Phi_0 = Phi_0[np.newaxis,:,np.newaxis,:,:] # (F,S,L) to (1,F,1,S,L)
    freqs = freqs[np.newaxis,:,np.newaxis,np.newaxis,np.newaxis] # (F,) to (1,F,1,1,1)

    S_i = S_i[:,:,:,np.newaxis,:] # (P,F,R,L) to (P,F,R,1,L)
    F_iplus = F_iplus[:,:,np.newaxis,:,:] # (P,F,S,L) to (P,F,1,S,L)
    F_icross = F_icross[:,:,np.newaxis,:,:] # (P,F,S,L) to (P,F,1,S,L)

    coef = amp**2 / (S_i * 8 * np.pi**3 * freqs**3) # [S_i] s^3 and [freqs^3] Hz^3 cancel

    term1 = a_pol**2 * F_iplus**2 * (Phi_T * (1 + 2 * np.sin(Phi_0)**2)
                                     + np.cos(Phi_T)*(-np.sin(Phi_T) + 4 * np.sin(Phi_0))
                                     - 4*np.sin(Phi_0))

    term2 = b_pol**2 * F_icross**2 * (Phi_T*(1+2*np.cos(Phi_0)**2)
                                      + np.sin(Phi_T)*(np.cos(Phi_T) - 4 * np.cos(Phi_0)))

    term3 = - (2*a_pol*b_pol*F_iplus*F_icross
               * (2*Phi_T*np.sin(Phi_0)*np.cos(Phi_0)
                  + np.sin(Phi_T)*(np.sin(Phi_T) - 2*np.sin(Phi_0)
                                   + 2*np.cos(Phi_T)*np.cos(Phi_0)
                                   - 2*np.cos(Phi_0))))

    snr2_pulsar_ss = coef*(term1 + term2 + term3) # (P,F,R,S,L)

    snr_ss = np.sqrt(np.sum(snr2_pulsar_ss, axis=0)) # (F,R,S,L), sum over the pulsars
    return snr_ss

######################### Detection Probability #########################

def _Fe_thresh(Num, alpha_0=0.001, guess=15):
    """ Calculate the threshold F_e statistic using sympy.nsolve

    Parameters
    ----------
    Num : int
        Number of single sources to detect.
    alpha_0 : scalar
        False alarm probability max.

    Returns
    -------
    Fe_bar : scalar
        Threshold Fe statistic
    """
    Fe_bar = Symbol('Fe_bar')
    func = 1 - (1 - (1 + Fe_bar)*np.e**(-Fe_bar))**Num - alpha_0
    Fe_bar = nsolve(func, Fe_bar, guess) # mod from
    return(Fe_bar)

def _I1_approx(xx):
    """ Modified Bessel Function of the first kind, first order, expansion for large values.
    """
    term1 = np.exp(xx)/np.sqrt(2*np.pi*xx)
    term2 = 1-3/8/xx * (1+5/2/8/xx * (1 + 21/3/8/xx))
    I_1 = term1*term2
    return I_1

def _integrand_approx(Fe, rho):
    """ Calculate an approximate integrand for the gamma_ssi integral
    using the large I_1 expansion approximation.

    """
    xx = rho*np.sqrt(2*Fe)
    termA = np.sqrt( Fe / np.pi / xx )/rho
    termB = np.exp(xx-Fe-rho**2/2)
    termC = 1-(3/8/xx*(1 + 5/2/8/xx * (1 + 21/3/8/xx)))
    return termA * termB * termC

def _integrand_gamma_ss_i(Fe, rho):
    """ Calculate the integrand of the gamma_ssi integral over Fe.

    """

    I_1 = special.i1(rho*np.sqrt(2*Fe))
    if np.isinf(I_1): # check if inf
        rv = _integrand_approx(Fe,rho)
    else:
        rv = (2*Fe)**(1/2) /rho * I_1 * np.exp(-Fe - rho**2 /2)
    return rv

def _gamma_of_rho(Fe_bar, rho, print_nans=False, max_peak = False):
    """ Calculate the detection probability for each single source in each realization.

    Parameters
    ----------
    rho : scalar
        SNR value for integral
    Fe_bar : scalar
        The threshold F_e statistic

    Returns
    -------
    gamma_ssi : (F,R,S,L) NDarray
        The detection probability for each single source, i, at each frequency and realization.

    TODO: Find a way to do this without the four embedded for-loops.
    """
    gamma_ssi = integrate.quad(_integrand_gamma_ss_i, Fe_bar, np.inf,
                               args=(rho))[0]
    return gamma_ssi

def _gamma_ssi(Fe_bar, rho, print_nans=False, max_peak = False):
    """ Calculate the detection probability for each single source in each realization.

    Parameters
    ----------
    rho : (F,R,S,L) NDarray
        Given by the total PTA signal to noise ratio, S/N_S, for each single source
    Fe_bar : scalar
        The threshold F_e statistic

    Returns
    -------
    gamma_ssi : (F,R,S,L) NDarray
        The detection probability for each single source, i, at each frequency and realization.

    TODO: Find a way to do this without the four embedded for-loops.
    """
    gamma_ssi = np.zeros((rho.shape))
    for ff in range(len(rho)):
        for rr in range(len(rho[0])):
            for ss in range(len(rho[0,0])):
                for ll in range(len(rho[0,0,0])):
                    gamma_ssi[ff,rr,ss,ll] = _gamma_of_rho(Fe_bar, rho[ff,rr,ss,ll])
                    if(np.isnan(gamma_ssi[ff,rr,ss,ll])):
                        if print_nans:
                            print(f'gamma_ssi[{ff},{rr},{ss},{ll}] is nan, setting to 0.')
                        gamma_ssi[ff,rr,ss,ll] = 0

    return gamma_ssi

def _gamma_above_peak(gamma):
    """ Set all gamma(rho>rho_peak) equal to gamma(rho_peak).

    """
    arg_peak = np.nanargmax(gamma)
    gamma[arg_peak:] = gamma[arg_peak]
    return gamma

def _gamma_above_one(gamma):
    """ Set all gamma values greater than one equal to one.

    """
    gamma[gamma>1.0] = 1.0
    return gamma

def _build_gamma_interp_grid(Num, grid_name):
    """ Build and save interpolation grid for rho to gamma, for the given Num.
    """
    rho_interp = np.geomspace(10**-3, 10**3, 10**3)
    Fe_bar = _Fe_thresh(Num)
    Fe_bar = np.float64(Fe_bar)
    gamma_interp = np.zeros_like(rho_interp)

    for rr in range(len(rho_interp)):
        gamma_interp[rr] = _gamma_of_rho(Fe_bar, rho_interp[rr])

    gamma_interp = _gamma_above_peak(gamma_interp)
    gamma_interp = _gamma_above_one(gamma_interp)

    np.savez(grid_name, rho_interp_grid=rho_interp, gamma_interp_grid=gamma_interp, Fe_bar=Fe_bar, Num=Num)

def _gamma_ssi_cython(rho, grid_path):
    """ Calculate the detection probability for each single source in each realization.

    Parameters
    ----------
    rho : (F,R,S,L) NDarray
        Given by the total PTA signal to noise ratio, S/N_S, for each single source
    Fe_bar : scalar
        The threshold F_e statistic

    Returns
    -------
    gamma_ssi : (F,R,S,L) NDarray
        The detection probability for each single source, i, at each frequency and realization.

    TODO: change grid save location to belong to some class or something?
    """

    Num = np.size(rho[:,0,0,:])

    grid_name = grid_path+'/rho_gamma_interp_grid_Num%d.npz' % (Num)

    # check if interpolation grid already exists, if not, build it
    if os.path.exists(grid_name) is False:
        # check if grid_path already exists, if not, makedir
        if (os.path.exists(grid_path) is False):
            os.makedirs(grid_path)
        _build_gamma_interp_grid(Num, grid_name)

    # read in data from saved grid
    grid_file = np.load(grid_name)
    rho_interp_grid = grid_file['rho_interp_grid']
    gamma_interp_grid = grid_file['gamma_interp_grid']
    grid_file.close()

    gamma_ssi = np.zeros_like(rho)
    # for ff in range(len(rho)):
    #     for rr in range(len(rho[0])):
    #         # interpolate for gamma in cython
    #         rho_flat = rho[ff,rr].flatten()
    #         rsort = np.argsort(rho_flat)
    #         gamma_flat = cyutils.gamma_of_rho_interp(rho_flat, rsort, rho_interp_grid, gamma_interp_grid)
    #         gamma_ssi[ff,rr] = gamma_flat.reshape(rho[ff,rr].shape)

    for rr in range(len(rho[0])):
        # interpolate for gamma in cython
        rho_flat = rho[:,rr].flatten()
        rsort = np.argsort(rho_flat)
        gamma_flat = cyutils.gamma_of_rho_interp(rho_flat, rsort, rho_interp_grid, gamma_interp_grid)
        gamma_ssi[:,rr] = gamma_flat.reshape(rho[:,rr].shape)


    return gamma_ssi



def _ss_detection_probability(gamma_ss_i):
    """ Calculate the probability of detecting any single source, given individual single
    source detection probabilities.


    Parameters
    ----------
    gamma_ss_i : (F,R,S,L) NDarray
        Detection probability of each single source, at each frequency and realization.

    Returns
    -------
    gamma_ss : (R,S) 2Darray
        Detection probability of any single source, for each R and S realization.
    """

    gamma_ss = 1 - np.product(1-gamma_ss_i, axis=(0,3))
    return gamma_ss


######################## Detection Probability #########################

def detect_ss(thetas, phis, sigmas, fobs, hc_ss, hc_bg,
              theta_ss, phi_ss=None, Phi0_ss=None, iota_ss=None, psi_ss=None,
              red_amp=None, red_gamma=None, alpha_0=0.001, ret_snr=False,):
    """ Calculate the single source detection probability, and all intermediary steps.

    Parameters
    ----------
    thetas : (P,) 1Darray of scalars
        Polar (latitudinal) angular position of each pulsar in radians.
    phis : (P,) 1Darray of scalars
        Azimuthal (longitudinal) angular position of each pulsar in radians.
    sigmas : (P,) 1Darray of scalars
        Sigma_i of each pulsar in seconds.
    fobs : (F,) 1Darray of scalars
        Observer frame gw frequency bin centers in Hz.
    dfobs : (F-1,) 1Darray of scalars
        Observer frame gw frequency bin widths in Hz.
    hc_ss : (F,R,L) NDarray of scalars
        Characteristic strain of the L loudest single sources at
        each frequency, for R realizations.
    hc_bg : (F,R)
        Characteristic strain of the background at each frequency,
        for R realizations.
    theta_ss : (F,S,L) NDarray
        Polar (latitudinal) angular position in the sky of each single source.
        Must be provided, to give the shape for sky realizations.
    phi_ss : (F,S,L) NDarray or None
        Azimuthal (longitudinal) angular position in the sky of each single source.
        If None, random values between 0 and 2pi will be assigned.
    Phi0_ss : (F,S,L) NDarray or None
        Initial GW phase.
        If None, random values between 0 and 2pi will be assigned.
    iota_ss : (F,S,L) NDarray or None
        Inclination of each single source with respect to the line of sight.
        If None, random values between 0 and pi will be assigned.
    psi_ss : (F,S,L) NDarray or None
        Polarization of each single source.
        If None, random values between 0 and pi will be assigned.
    red_amp : scalar or None
        Amplitude of pulsar red noise.
    red_gamma : scalar or None
        Power law index of pulsar red noise.
    alpha_0 : scalar
        False alarm probability
    ret_snr : Bool
        Whether or not to also return snr_ss.

    Returns
    -------
    gamma_ss : (R,S) NDarray
        Probability of detecting any single source, for each R and S realization.
    snr_ss : (F,R,S,L) NDarray
        SNR of each single source.

    """

    warnings.warn("'detect_ss()' is deprecated. Use 'detect_ss_pta()' instead.", DeprecationWarning)
    dur = 1.0/fobs[0]
    cad = 1.0/(2*fobs[-1])
    fobs_cents, fobs_edges = utils.pta_freqs(dur, num=len(fobs))
    dfobs = np.diff(fobs_edges)

    # Assign random single source sky params, if not provided.
    if phi_ss is None:
        phi_ss = np.random.uniform(0,2*np.pi, size=theta_ss.size).reshape(theta_ss.shape)
    if Phi0_ss is None:
        Phi0_ss = np.random.uniform(0,2*np.pi, size=theta_ss.size).reshape(theta_ss.shape)
    if iota_ss is None:
        iota_ss = np.random.uniform(0, np.pi, size = theta_ss.size).reshape(theta_ss.shape)
    if psi_ss is None:
        psi_ss = np.random.uniform(0, np.pi, size = theta_ss.size).reshape(theta_ss.shape)

    # unitary vectors
    m_hat = _m_unitary_vector(theta_ss, phi_ss, psi_ss) # (3,F,S,L)
    n_hat = _n_unitary_vector(theta_ss, phi_ss, psi_ss) # (3,F,S,L)
    Omega_hat = _Omega_unitary_vector(theta_ss, phi_ss) # (3,F,S,L)
    pi_hat = _pi_unitary_vector(phis, thetas) # (3,P)

    # antenna pattern functions
    F_iplus, F_icross = _antenna_pattern_functions(m_hat, n_hat, Omega_hat,
                                                   pi_hat) # (P,F,S,L)

    # noise spectral density
    S_i = _total_noise(cad, sigmas, hc_ss, hc_bg, fobs, red_amp, red_gamma)

    # amplitudw
    amp = _amplitude(hc_ss, fobs, dfobs) # (F,R,L)

    # SNR (includes a_pol, b_pol, and Phi_T calculations internally)
    snr_ss = _snr_ss(amp, F_iplus, F_icross, iota_ss, dur, Phi0_ss, S_i, fobs) # (F,R,S,L)

    Num = hc_ss[:,0,:].size # number of single sources in a single strain realization (F*L)
    Fe_bar = _Fe_thresh(Num, alpha_0=alpha_0) # scalar

    gamma_ssi = _gamma_ssi(Fe_bar, rho=snr_ss) # (F,R,S,L)
    gamma_ss = _ss_detection_probability(gamma_ssi) # (R,S)

    if ret_snr:
        return gamma_ss, snr_ss, gamma_ssi
    else:
        return gamma_ss


def detect_ss_pta(pulsars, fobs, hc_ss, hc_bg, 
                custom_noise=None, nexcl_noise=0,
              theta_ss=None, phi_ss=None, Phi0_ss=None, iota_ss=None, psi_ss=None, nskies=25, 
              Fe_bar = None, red_amp=None, red_gamma=None, alpha_0=0.001, Fe_bar_guess=15,
              ret_snr=False, print_nans=False, snr_cython=True, gamma_cython=True, grid_path=GAMMA_RHO_GRID_PATH):
    """ Calculate the single source detection probability, and all intermediary steps for
    R strain realizations and S sky realizations.

    Parameters
    ----------
    pulsars : (P,) list of hasasia.Pulsar objects
        A set of pulsars generated by hasasia.sim.sim_pta()
    fobs : (F,) 1Darray of scalars
        Observer frame gw frequency bin centers in Hz.
    hc_ss : (F,R,L) NDarray of scalars
        Characteristic strain of the L loudest single sources at
        each frequency, for R realizations.
    hc_bg : (F,R)
        Characteristic strain of the background at each frequency,
        for R realizations.
    custom_noise : (P,F,R,L) array or None
        Custom noise if not None, otherwise noise is calcualted normally.
    nexcl_noise : int
        Number of loudest single sources to exclude from hc_rest noise, in addition 
        to the source in question.
    theta_ss : (F,S,L) NDarray
        Polar (latitudinal) angular position in the sky of each single source.
        Must be provided, to give the shape for sky realizations.
    phi_ss : (F,S,L) NDarray or None
        Azimuthal (longitudinal) angular position in the sky of each single source.
        If None, random values between 0 and 2pi will be assigned.
    Phi0_ss : (F,S,L) NDarray or None
        Initial GW phase.
        If None, random values between 0 and 2pi will be assigned.
    iota_ss : (F,S,L) NDarray or None
        Inclination of each single source with respect to the line of sight.
        If None, random values between 0 and pi will be assigned.
    psi_ss : (F,S,L) NDarray or None
        Polarization of each single source.
        If None, random values between 0 and pi will be assigned.
    Fe_bar : scalar or None
        Threshold F-statistic
    red_amp : scalar or None
        Amplitude of pulsar red noise.
    red_gamma : scalar or None
        Power law index of pulsar red noise.
    alpha_0 : scalar
        False alarm probability
    ret_snr : Bool
        Whether or not to also return snr_ss.

    Returns
    -------
    gamma_ss : (R,S) NDarray
        Probability of detecting any single source, for each R and S realization.
    snr_ss : (F,R,S,L) NDarray
        SNR of each single source. Returned only if ret_snr is True.
    gamma_ssi : (F,R,S,L) NDarray
        DP of each single source. Returned only if ret_snr is True.

    """

    dur = 1.0/fobs[0]
    cad = 1.0/(2*fobs[-1])
    fobs_cents, fobs_edges = utils.pta_freqs(dur, num=len(fobs))
    dfobs = np.diff(fobs_edges)

    # Assign random single source sky params, if not provided.
    nfreqs, nreals, nloudest = [*hc_ss.shape]
    if theta_ss is None:
        theta_ss = np.random.uniform(0,np.pi, size=nfreqs*nskies*nloudest).reshape(nfreqs, nskies, nloudest)
    if phi_ss is None:
        phi_ss = np.random.uniform(0,2*np.pi, size=theta_ss.size).reshape(theta_ss.shape)
    if Phi0_ss is None:
        Phi0_ss = np.random.uniform(0,2*np.pi, size=theta_ss.size).reshape(theta_ss.shape)
    if iota_ss is None:
        iota_ss = np.random.uniform(0, np.pi, size = theta_ss.size).reshape(theta_ss.shape)
    if psi_ss is None:
        psi_ss = np.random.uniform(0, np.pi, size = theta_ss.size).reshape(theta_ss.shape)

    # unitary vectors
    m_hat = _m_unitary_vector(theta_ss, phi_ss, psi_ss) # (3,F,S,L)
    n_hat = _n_unitary_vector(theta_ss, phi_ss, psi_ss) # (3,F,S,L)
    Omega_hat = _Omega_unitary_vector(theta_ss, phi_ss) # (3,F,S,L)


    # check pulsar inputs
    # get pulsar properties
    thetas = np.zeros(len(pulsars))
    phis = np.zeros(len(pulsars))
    sigmas = np.zeros(len(pulsars))
    for ii in range(len(pulsars)):
        thetas[ii] = pulsars[ii].theta
        phis[ii] = pulsars[ii].phi
        sigmas[ii] = np.mean(pulsars[ii].toaerrs)

    pi_hat = _pi_unitary_vector(phis, thetas) # (3,P)

    # antenna pattern functions
    F_iplus, F_icross = _antenna_pattern_functions(m_hat, n_hat, Omega_hat,
                                                   pi_hat) # (P,F,S,L)

    # noise spectral density
    if custom_noise is not None:
        if custom_noise.shape != (len(pulsars), nfreqs, nreals, nloudest):
            err = f"{custom_noise.shape=}, must be shape (P,F,R,L)=({len(pulsars)}, {nfreqs}, {nreals}, {nloudest})"
            raise ValueError(err)
        S_i = custom_noise
    else:
        S_i = _total_noise(cad, sigmas, hc_ss, hc_bg, fobs, red_amp, red_gamma, nexcl=nexcl_noise)

    # amplitudw
    amp = _amplitude(hc_ss, fobs, dfobs) # (F,R,L)

    # SNR (includes a_pol, b_pol, and Phi_T calculations internally)
    if snr_cython:
        snr_ss = _snr_ss(amp, F_iplus, F_icross, iota_ss, dur, Phi0_ss, S_i, fobs) # (F,R,S,L)
    else:
        snr_ss = _snr_ss_5dim(amp, F_iplus, F_icross, iota_ss, dur, Phi0_ss, S_i, fobs) # (F,R,S,L)

    if gamma_cython:
        gamma_ssi = _gamma_ssi_cython(snr_ss, grid_path=grid_path) # (F,R,S,L)
    else:
        if (Fe_bar is None):
            Num = hc_ss[:,0,:].size # number of single sources in a single strain realization (F*L)
            Fe_bar = _Fe_thresh(Num, alpha_0=alpha_0, guess=Fe_bar_guess) # scalar

        gamma_ssi = _gamma_ssi(Fe_bar, rho=snr_ss, print_nans=print_nans) # (F,R,S,L)
    gamma_ss = _ss_detection_probability(gamma_ssi) # (R,S)

    if ret_snr:
        return gamma_ss, snr_ss, gamma_ssi
    else:
        return gamma_ss




########################################################################
######################### Running on Libraries #########################
########################################################################

def detect_lib(hdf_name, output_dir, npsrs, sigma, nskies, thresh=DEF_THRESH,
                plot=True, debug=False, grid_path=GAMMA_RHO_GRID_PATH, 
                snr_cython = True, save_ssi=False, ret_dict=False):
    """ Calculate detection statistics for an ss library output.

    Parameters
    ----------
    hdf_fname : String
        Name of hdf file, including path.
    output_dir : String
        Where to store outputs, including full path.
    npsrs : int
        Number of pulsars to place in pta.
    sigma : int or (P,) array
        Noise in each pulsar
    nskies : int
        Number of sky realizationt to create.
    thresh : float
        Threshold for detection in realization.
    plot : Bool
        Whether or not to make and save plots.
    debug : Bool
        Whether to print info along the way.
    grid_path : string
        Path to snr interpolation grid
    snr_cython : Bool
        Whether to use cython interpolation for ss snr calculation.
    save_ssi : Bool
        Whether to store gamma_ssi in npz arrays

    Returns
    -------
    dp_ss : (N,R,S) Ndarray
        Single source detection probability for each of
        - N parameter space samples
        - R strain realizations
        - S sky realizations
    dp_bg : (N,R) Ndarray
        Background detectin probability.
    snr_ss : (N,F,R,S,L)
        Signal to noise ratio for every single source in every
        realization at each of
        - F frequencies
        - L loudest at frequency
    snr_bg : (N,F,R)
        Signal to noise ratio of the background at each
        frequency of each realization.
    df_ss : (N,)
        Fraction of realizations with a single source detection, for each sample.
    df_bg : (N,) 1Darray
        Fraction of realizations with a background detection, for each sample.
    ev_ss : (N,R,) NDarray
        Expectation number of single source detections, averaged across realizations,
        for each sample.

    TODO: Update, no need to return ss_snr
    """

    # Read in hdf file
    ssfile = h5py.File(hdf_name, 'r')
    fobs = ssfile['fobs'][:]
    dur = 1.0/fobs[0]
    cad = 1.0/(2*fobs[-1])
    # if dfobs is None: dfobs = ssfile['dfobs'][:]
    # if dur is None: dur = ssfile['pta_dur'][0]
    # if cad is None: cad = ssfile['pta_cad'][0]
    hc_ss = ssfile['hc_ss'][...]
    hc_bg = ssfile['hc_bg'][...]
    shape = hc_ss.shape
    nsamps, nfreqs, nreals, nloudest = shape[0], shape[1], shape[2], shape[3]

    # Assign output folder
    import os
    if (os.path.exists(output_dir) is False):
        print('Making output directory.')
        os.makedirs(output_dir)
    else:
        print('Writing to an existing directory.')

    # build PTA
    if debug: print('Building pulsar timing array.')
    phis = np.random.uniform(0, 2*np.pi, size = npsrs)
    thetas = np.random.uniform(np.pi/2, np.pi/2, size = npsrs)
    # sigmas = np.ones_like(phis)*sigma
    if debug: print(f"{phis.shape=}, {thetas.shape=}, {dur=}, {cad=}, {sigma=}")
    psrs = hsim.sim_pta(timespan=dur/YR, cad=1/(cad/YR), sigma=sigma,
                    phi=phis, theta=thetas)

     # Build ss skies
    if debug: print('Building ss skies.')
    theta_ss, phi_ss, Phi0_ss, iota_ss, psi_ss = _build_skies(nfreqs, nskies, nloudest)

    # Calculate DPs, SNRs, and DFs
    if debug: print('Calculating SS and BG detection statistics.')
    dp_ss = np.zeros((nsamps, nreals, nskies)) # (N,R,S)
    dp_bg = np.zeros((nsamps, nreals)) # (N,R)
    snr_bg = np.zeros((nsamps, nfreqs, nreals))
    df_ss = np.zeros(nsamps)
    df_bg = np.zeros(nsamps)
    ev_ss = np.zeros((nsamps, nreals, nskies))
    if save_ssi: 
        snr_ss = np.zeros((nsamps, nfreqs, nreals, nskies, nloudest))
        gamma_ssi = np.zeros((nsamps, nfreqs, nreals, nskies, nloudest))

    # # one time calculations
    # Num = nfreqs * nloudest # number of single sources in a single strain realization (F*L)
    # Fe_bar = _Fe_thresh(Num) # scalar

    for nn in range(nsamps):
        if debug: print('on sample nn=%d out of N=%d' % (nn,nsamps))
        dp_bg[nn,:], snr_bg[nn,...] = detect_bg_pta(psrs, fobs, hc_bg[nn], ret_snr=True)
        vals_ss = detect_ss_pta(psrs, fobs, hc_ss[nn], hc_bg[nn], 
                                ret_snr=True, gamma_cython=True, snr_cython=snr_cython,
                                theta_ss=theta_ss, phi_ss=phi_ss, Phi0_ss=Phi0_ss,
                                iota_ss=iota_ss, psi_ss=psi_ss, grid_path=grid_path)
        dp_ss[nn,:,:]  = vals_ss[0]
        if save_ssi: 
            snr_ss[nn] = vals_ss[1]
            gamma_ssi[nn] = vals_ss[2]
        df_ss[nn], df_bg[nn] = detfrac_of_reals(dp_ss[nn], dp_bg[nn], thresh)
        ev_ss[nn] = expval_of_ss(vals_ss[2])


        if plot:
            fig = plot_sample_nn(fobs, hc_ss[nn], hc_bg[nn],
                         dp_ss[nn], dp_bg[nn],
                         df_ss[nn], df_bg[nn], nn=nn)
            plot_fname = (output_dir+'/p%06d_detprob.png' % nn) # need to make this directory
            fig.savefig(plot_fname, dpi=100)
            plt.close(fig)

    if debug: print('Saving npz files and allsamp plots.')
    fig1 = plot_detprob(dp_ss, dp_bg, nsamps)
    fig2 = plot_detfrac(df_ss, df_bg, nsamps, thresh)
    fig1.savefig(output_dir+'/allsamp_detprobs.png', dpi=300)
    fig2.savefig(output_dir+'/allsamp_detfracs.png', dpi=300)
    plt.close(fig1)
    plt.close(fig2)
    if save_ssi:
        np.savez(output_dir+'/detstats.npz', dp_ss=dp_ss, dp_bg=dp_bg, df_ss=df_ss, df_bg=df_bg,
              snr_ss=snr_ss, snr_bg=snr_bg, ev_ss = ev_ss, gamma_ssi=gamma_ssi)
    else:
        np.savez(output_dir+'/detstats.npz', dp_ss=dp_ss, dp_bg=dp_bg, df_ss=df_ss, df_bg=df_bg,
              snr_bg=snr_bg, ev_ss = ev_ss)
        
    # return dictionary 
    if ret_dict:
        data = {
            'dp_ss':dp_ss, 'dp_bg':dp_bg, 'df_ss':df_ss, 'df_bg':df_bg,
            'snr_bg':snr_bg, 'ev_ss':ev_ss
        }
        if save_ssi: 
            data.update({'gamma_ssi':gamma_ssi})
            data.update({'snr_ss':snr_ss})
        return data
    return 


def detect_lib_clbrt_pta(hdf_name, output_dir, npsrs, nskies, thresh=DEF_THRESH,
                         sigstart=1e-6, sigmin=1e-9, sigmax=1e-4, tol=0.01, maxbads=5,
                plot=True, debug=False, 
                save_ssi=False, ret_dict=False, ss_noise=False):
    """ Calculate detection statistics for an ss library output.

    Parameters
    ----------
    hdf_fname : String
        Name of hdf file, including path.
    output_dir : String
        Where to store outputs, including full path.
    npsrs : int
        Number of pulsars to place in pta.
    sigma : int or (P,) array
        Noise in each pulsar
    nskies : int
        Number of sky realizationt to create.
    thresh : float
        Threshold for detection in realization.
    plot : Bool
        Whether or not to make and save plots.
    debug : Bool
        Whether to print info along the way.
    grid_path : string
        Path to snr interpolation grid
    snr_cython : Bool
        Whether to use cython interpolation for ss snr calculation.
    save_ssi : Bool
        Whether to store gamma_ssi in npz arrays
    ss_noise : Bool
        Whether or not to use all but loudest SS as BG noise sources.

    Returns
    -------
    dp_ss : (N,R,S) Ndarray
        Single source detection probability for each of
        - N parameter space samples
        - R strain realizations
        - S sky realizations
    dp_bg : (N,R) Ndarray
        Background detectin probability.
    snr_ss : (N,F,R,S,L)
        Signal to noise ratio for every single source in every
        realization at each of
        - F frequencies
        - L loudest at frequency
    snr_bg : (N,F,R)
        Signal to noise ratio of the background at each
        frequency of each realization.
    df_ss : (N,)
        Fraction of realizations with a single source detection, for each sample.
    df_bg : (N,) 1Darray
        Fraction of realizations with a background detection, for each sample.
    ev_ss : (N,R,) NDarray
        Expectation number of single source detections, averaged across realizations,
        for each sample.

    # TODO: Add an option to calculate just for a particular selection of samples, 
    e.g. pass in nn_min and nn_max and use for nn in range(nn_min, nn_max) instead of 
    nn in range(neals). Then combine these into an hdf file separately.
    """

    # Read in hdf file
    ssfile = h5py.File(hdf_name, 'r')
    fobs = ssfile['fobs'][:]
    dur = 1.0/fobs[0]
    cad = 1.0/(2*fobs[-1])
    hc_ss = ssfile['hc_ss'][...]
    hc_bg = ssfile['hc_bg'][...]
    shape = hc_ss.shape
    nsamps, nfreqs, nreals, nloudest = shape[0], shape[1], shape[2], shape[3]

    # Assign output folder
    if (os.path.exists(output_dir) is False):
        print('Making output directory.')
        os.makedirs(output_dir)
    else:
        print('Writing to an existing directory.')

    # build PTA pulsar positions
    if debug: print('Placing pulsar.')
    phis = np.random.uniform(0, 2*np.pi, size = npsrs)
    thetas = np.random.uniform(np.pi/2, np.pi/2, size = npsrs)
    if debug: print(f"{phis.shape=}, {thetas.shape=}, {dur=}, {cad=}")
    # psrs = hsim.sim_pta(timespan=dur/YR, cad=1/(cad/YR), sigma=sigma,
    #                 phi=phis, theta=thetas)

     # Build ss skies
    if debug: print('Building ss skies.')
    theta_ss, phi_ss, Phi0_ss, iota_ss, psi_ss = _build_skies(nfreqs, nskies, nloudest)

    # Calculate DPs, SNRs, and DFs
    if debug: print('Calculating SS and BG detection statistics.')
    dp_ss = np.zeros((nsamps, nreals, nskies)) # (N,R,S)
    dp_bg = np.zeros((nsamps, nreals)) # (N,R)
    snr_bg = np.zeros((nsamps, nfreqs, nreals))
    df_ss = np.zeros(nsamps)
    df_bg = np.zeros(nsamps)
    ev_ss = np.zeros((nsamps, nreals, nskies))
    if save_ssi: 
        snr_ss = np.zeros((nsamps, nfreqs, nreals, nskies, nloudest))
        gamma_ssi = np.zeros((nsamps, nfreqs, nreals, nskies, nloudest))

    for nn in range(nsamps):
        if debug: 
            print('\non sample nn=%d out of N=%d' % (nn,nsamps))
            samp_dur = datetime.now()
            real_dur = datetime.now()

        # calibrate individual realization PTAs
        for rr in range(nreals):
            if rr==0:
                _sigstart, _sigmin, _sigmax = sigstart, sigmin, sigmax 
            if debug: 
                now = datetime.now()
                if (rr%10==0):
                    print(f"{nn=}, {rr=}, {now-real_dur} s per realization, {_sigmin=:.2e}, {_sigmax=:.2e}, {_sigstart=:.2e}")
                real_dur = now

            # use sigmin and sigmax from previous realization, 
            # unless it's the first realization of the sample
            psrs, _sigstart, _sigmin, _sigmax = calibrate_one_pta(hc_bg[nn,:,rr], hc_ss[nn,:,rr,:], fobs, npsrs, tol=tol, maxbads=maxbads,
                                     sigstart=_sigstart, sigmin=_sigmin, sigmax=_sigmax, debug=debug, ret_sig=True, ss_noise=ss_noise)
            _sigmin /= 2
            _sigmax *= 2
            
            # get background detstats
            _dp_bg, _snr_bg = detect_bg_pta(psrs, fobs, hc_bg[nn,:,rr:rr+1], ret_snr=True)
            dp_bg[nn,rr], snr_bg[nn,:,rr] = _dp_bg.squeeze(), _snr_bg.squeeze()

            # get single source detstats
            _dp_ss, _snr_ss, _gamma_ssi = detect_ss_pta(
                    psrs, fobs, hc_ss[nn,:,rr:rr+1], hc_bg[nn,:,rr:rr+1], 
                    nskies=nskies, ret_snr=True,
                    theta_ss=theta_ss, phi_ss=phi_ss, Phi0_ss=Phi0_ss, 
                    iota_ss=iota_ss, psi_ss=psi_ss)
            dp_ss[nn,rr,:] = _dp_ss.squeeze()
            if save_ssi:
                snr_ss[nn,:,rr] = _snr_ss.squeeze()
                gamma_ssi[nn,:,rr] = _gamma_ssi.squeeze() 
            ev_ss[nn,rr] = expval_of_ss(_gamma_ssi)
        if debug:
            now = datetime.now()
            print(f"Sample {nn} took {now-samp_dur} s")
            samp_dur = now

        df_ss[nn], df_bg[nn] = detfrac_of_reals(dp_ss[nn], dp_bg[nn], thresh)

        if save_ssi:
            np.savez(output_dir+f'/detstats_p{nn:06d}.npz', dp_ss=dp_ss, dp_bg=dp_bg, df_ss=df_ss, df_bg=df_bg,
                snr_ss=snr_ss, snr_bg=snr_bg, ev_ss = ev_ss, gamma_ssi=gamma_ssi)
        else:
            np.savez(output_dir+f'/detstats_p{nn:06d}.npz', dp_ss=dp_ss[nn], dp_bg=dp_bg[nn], df_ss=df_ss[nn], df_bg=df_bg[nn],
                snr_bg=snr_bg[nn], ev_ss = ev_ss[nn])
        
        if plot:
            fig = plot_sample_nn(fobs, hc_ss[nn], hc_bg[nn],
                         dp_ss[nn], dp_bg[nn],
                         df_ss[nn], df_bg[nn], nn=nn)
            plot_fname = (output_dir+'/p%06d_detprob.png' % nn) # need to make this directory
            fig.savefig(plot_fname, dpi=100)
            plt.close(fig)

    if debug: print('Saving npz files and allsamp plots.')
    fig1 = plot_detprob(dp_ss, dp_bg, nsamps)
    fig2 = plot_detfrac(df_ss, df_bg, nsamps, thresh)
    fig1.savefig(output_dir+'/allsamp_detprobs.png', dpi=300)
    fig2.savefig(output_dir+'/allsamp_detfracs.png', dpi=300)
    plt.close(fig1)
    plt.close(fig2)
    if save_ssi:
        np.savez(output_dir+'/detstats_lib.npz', dp_ss=dp_ss, dp_bg=dp_bg, df_ss=df_ss, df_bg=df_bg,
              snr_ss=snr_ss, snr_bg=snr_bg, ev_ss = ev_ss, gamma_ssi=gamma_ssi)
    else:
        np.savez(output_dir+'/detstats_lib.npz', dp_ss=dp_ss, dp_bg=dp_bg, df_ss=df_ss, df_bg=df_bg,
              snr_bg=snr_bg, ev_ss = ev_ss)
        
    # return dictionary 
    if ret_dict:
        data = {
            'dp_ss':dp_ss, 'dp_bg':dp_bg, 'df_ss':df_ss, 'df_bg':df_bg,
            'snr_bg':snr_bg, 'ev_ss':ev_ss
        }
        if save_ssi: 
            data.update({'gamma_ssi':gamma_ssi})
            data.update({'snr_ss':snr_ss})
        return data
    return 



def _build_pta(npsrs, sigma, dur, cad):
    # build PTA
    phis = np.random.uniform(0, 2*np.pi, size = npsrs)
    thetas = np.random.uniform(np.pi/2, np.pi/2, size = npsrs)
    # sigmas = np.ones_like(phis)*sigma
    psrs = hsim.sim_pta(timespan=dur/YR, cad=1/(cad/YR), sigma=sigma,
                    phi=phis, theta=thetas)
    return psrs

def _build_skies(nfreqs, nskies, nloudest):
    theta_ss = np.random.uniform(0, np.pi, size = nfreqs * nskies * nloudest).reshape(nfreqs, nskies, nloudest)
    phi_ss = np.random.uniform(0, 2*np.pi, size = theta_ss.size).reshape(theta_ss.shape)
    Phi0_ss = np.random.uniform(0,2*np.pi, size = theta_ss.size).reshape(theta_ss.shape)
    iota_ss = np.random.uniform(0,  np.pi, size = theta_ss.size).reshape(theta_ss.shape)
    psi_ss = np.random.uniform(0,   np.pi, size = theta_ss.size).reshape(theta_ss.shape)
    return theta_ss, phi_ss, Phi0_ss, iota_ss, psi_ss


def detfrac_of_reals(dp_ss, dp_bg, thresh=DEF_THRESH):
    """ Calculate the fraction of realizations with a detection.

    Parameters
    ----------
    dp_ss : (R,S) Ndarray
        Single source detection probability for each of
    dp_bg : (R) Ndarray
        Background detectin probability.
    thresh : float
        Fractional threshold for DP to claim a detection.
    
    Returns
    -------
    df_ss : float
        Fraction of realizations with dp_ss>threshold
    def_bg : float
        Fraction of realizations with dp_bg>threshold
    """ 

    df_ss = np.sum(dp_ss>thresh)/(dp_ss.size)
    df_bg = np.sum(dp_bg>thresh)/(dp_bg.size)
    return df_ss, df_bg
    


def expval_of_ss(gamma_ssi,):
    """ Calculate the expected number of single source detections, across all frequencies.

    Parameters
    ----------
    gamma_ssi : (F,R,S,L) NDarray
        Detection probability of each single source

    Returns
    -------
    ev_ss : (R,S)
        Expected (fractional) number of single source detection for each strain and sky realizations.
    
    """
    # print(f"{gamma_ssi.shape=}, {[*gamma_ssi.shape]}")
    # nfreqs, nreals, nskies, nloudest = [*gamma_ssi.shape]
    ev_ss = np.sum(gamma_ssi, axis=(0,3))
    return ev_ss
    # df_bg[nn] = np.sum(dp_bg[nn]>thresh)/(nreals)

def count_n_ss(gamma_ssi):
    """ Calculate the number of random single source detections.

    Parameters
    ----------
    gamma_ssi : (F,R,S,L) NDarray
        Detection probability of each single source

    Returns
    -------
    nn_ss : (R,S)
        Number of random single source detections for each strain and sky realization.
    
    """

    randoms = np.random.uniform(0,1, size=gamma_ssi.size).reshape(gamma_ssi.shape)
    nn_ss = np.sum((gamma_ssi>randoms), axis=(0,3))
    return nn_ss




############################# Plot Library #############################

def plot_sample_nn(fobs, hc_ss, hc_bg, dp_ss, dp_bg, df_ss, df_bg, nn):
    """ Plot strain and detection probability for a single sample.

    Parameters
    ----------
    fobs : (F,) 1Darray
        Observed GW frequencies.
    hc_ss : (F,R,L) NDarray
        Characteristic strain of loudest single sources, for one sample.
    hc_bg : (F,R) NDarray
        Characteristic strain of the background, for one sample.
    dp_ss : (R,S) 1Darray
        Single source detection probability of each strain and sky realization.
    dp_bg : (R,) 1Darray
        Background detection probability of each realization.
    df_ss : scalar
        Fraction of realizations with 'dp_ss' > 'thresh'.
    df_bg : scalar
        Fraction of realizations with 'dp_bg' > 'thresh'.

    Returns
    -------
    fig : figure object

    """
    shape = hc_ss.shape
    F, R, L = shape[0], shape[1], shape[2]
    S = dp_ss.shape[-1]
    fig, axs = plt.subplots(1, 2,figsize=(12,4))

    # Strains
    plot.draw_ss_and_gwb(axs[0], fobs*YR, hc_ss, hc_bg)
    axs[0].set_xlabel(plot.LABEL_GW_FREQUENCY_YR)
    axs[0].set_ylabel(plot.LABEL_CHARACTERISTIC_STRAIN)
    axs[0].set_title('Sample nn=%d (F=%d, R=%d, L=%d)' % (nn, F, R, L))
    axs[0].set_xscale('log')
    axs[0].set_yscale('log')

    for ss in range(S):
        axs[1].scatter(np.arange(R), dp_ss[:,ss], alpha=0.25)
    axs[1].scatter(np.arange(R), dp_bg, color='k',
                   label='BG, DF = %.2e' % df_bg,
                   marker='d')
    axs[1].errorbar(np.arange(R), np.mean(dp_ss[:,:], axis=1),
                     yerr = np.std(dp_ss[:,:], axis=1), color='orangered',
                      label = 'SS, sky-avg, DF = %.2e' % df_ss, linestyle='', capsize=3,
                       marker='o' )
    axs[1].set_xlabel('Realization (R)')
    axs[1].set_ylabel('SS DetProb')
    # axs[1].set_title('BG DF = ' %nn)
    fig.legend()
    fig.tight_layout()

    return fig

def plot_detprob(dp_ss_all, dp_bg_all, nsamps):
    """ Plot detection probability for many samples.

    Paramaters
    ----------
    dp_ss_all : (N,R, S) NDarray
        Single source detection probably of each strain and sky realization of each sample.
    dp_bg_all : (N,R) NDarray
        Background detection probability of each strain realization of each sample.

    Returns
    -------
    fig : figure object

    """
    fig, ax = plt.subplots(figsize=(6.5,4))
    ax.set_xlabel('Param Space Sample')
    ax.set_ylabel('Detection Probability, $\gamma$')
    ax.errorbar(np.arange(nsamps), np.mean(dp_bg_all, axis=1),
                yerr = np.std(dp_bg_all, axis=1), linestyle='',
                marker='d', capsize=5, color='cornflowerblue', alpha=0.5,
                label = r'$\langle \gamma_\mathrm{BG} \rangle$')
    ax.errorbar(np.arange(nsamps), np.mean(dp_ss_all, axis=(1,2)),
                yerr = np.std(dp_ss_all, axis=(1,2)), linestyle='',
                marker='o', capsize=5, color='orangered', alpha=0.5,
                label = r'$\langle \gamma_\mathrm{SS} \rangle$')
    ax.set_yscale('log')
    ax.set_title('Average DP across Realizations')

    ax.legend()
    fig.tight_layout()

    return fig


def plot_detfrac(df_ss, df_bg, nsamps, thresh):
    """ Plot detection fraction for many samples.

    Paramaters
    ----------
    df_all : (N,) NDarray
        Fraction of strain and sky realization with a 'dp_ss'>'thresh'.
    df_bg : (N,) NDarray
        Fraction of strain realization with a 'dp_bg'>'thresh'.


    Returns
    -------
    fig : figure object
    """
    fig, ax = plt.subplots(figsize=(6.5,4))
    ax.plot(np.arange(nsamps), df_bg, color='cornflowerblue', label='BG',
            marker='d', alpha=0.5)
    ax.plot(np.arange(nsamps), df_ss, color='orangered', label='SS',
            marker='o', alpha=0.5)
    ax.set_xlabel('Param Space Sample')
    ax.set_ylabel('Detection Fraction')
    ax.set_title('Fraction of Realizations with DP > %0.2f' % thresh)
    ax.legend()
    fig.tight_layout()
    return fig


############################# Rank Samples #############################


def amp_to_hc(amp_ref, fobs, dfobs):
    """ Calculate characteristic strain from strain amplitude (from 1/yr amplitude).

    """
    hc = amp_ref*np.sqrt(fobs/dfobs)
    return hc

def rank_samples(hc_ss, hc_bg, fobs, fidx=None, dfobs=None, amp_ref=None, hc_ref=HC_REF15_10YR, ret_all = False):
    """ Sort samples by those with f=1/yr char strains closest to some reference value.

    Parameters
    ----------
    hc_ss : (N,F,R,L) NDarray
        Characteristic strain of the loudest single sources.
    hc_bg : (N,F,R) NDarray
        Characteristic strain of the background.
    fobs : (F,)
        Observed GW frequency
    dfobs : (F,) or None
        Observed GW frequency bin widths.
        only needed if using amp_ref
    amp_ref : scalar or None
        Reference strain amplitude at f=0.1/yr
    hc_ref : scalar or None
        Reference characteristic strain at f=0.1/yr
        Only one of hc_ref and amp_ref should be provided.


    Returns
    -------
    nsort : (N,) 1Darray
        Indices of the param space samples sorted by proximity to the reference 1yr amplitude.
    fidx : integer
        Index of reference frequency.
    hc_ref : float
        Reference char strain extrapolated to fidx frequency.
    """

    # find frequency bin nearest to 1/10yr
    if fidx is None:
        fidx = (np.abs(fobs - 1/(10*YR))).argmin()


    # extrapolate hc_ref at freq closest to 1/10yr from 1/10yr ref
    hc_ref = hc_ref * (fobs[fidx]*YR/.1)**(-2/3)

    # select 1/yr median strains of samples
    hc_tt = np.sqrt(hc_bg[:,fidx,:]**2 + np.sum(hc_ss[:,fidx,:,:]**2, axis=-1)) # (N,R)
    hc_diff = np.abs(hc_tt - hc_ref) # (N,R)
    hc_diff = np.median(hc_diff, axis=-1) # median of differences (N,)

    # sort by closest
    nsort = np.argsort(hc_diff)

    if ret_all:
        return nsort, fidx, hc_ref
    return nsort

######################### Param Space Models ########################### 

def detect_pspace_model(fobs_cents, hc_ss, hc_bg, 
                        npsrs, sigma, nskies, thresh=DEF_THRESH, debug=False):
    
    nfreqs, nreals, nloudest = [*hc_ss.shape]
    dur = 1/fobs_cents[0]
    cad = 1.0 / (2 * fobs_cents[-1])
    # fobs_cents, fobs_edges = holo.utils.pta_freqs(dur)
    # dfobs = np.diff(fobs_edges)

    # build PTA
    if debug: print('Building pulsar timing array.')
    psrs = _build_pta(npsrs, sigma, dur, cad)
    # phis = np.random.uniform(0, 2*np.pi, size = npsrs)
    # thetas = np.random.uniform(np.pi/2, np.pi/2, size = npsrs)
    # # sigmas = np.ones_like(phis)*sigma
    # psrs = hsim.sim_pta(timespan=dur/YR, cad=1/(cad/YR), sigma=sigma,
    #                 phi=phis, theta=thetas)

    # Build ss skies
    if debug: print('Building ss skies.')
    theta_ss, phi_ss, Phi0_ss, iota_ss, psi_ss = _build_skies(nfreqs, nskies, nloudest)


    # Calculate DPs, SNRs, and DFs
    if debug: print('Calculating SS and BG detection statistics.')
    dp_bg, snr_bg = detect_bg_pta(psrs, fobs_cents, hc_bg, ret_snr=True)
    # print(f"{np.mean(dp_bg)=}")
    vals_ss = detect_ss_pta(
        psrs, fobs_cents, hc_ss, hc_bg, 
        gamma_cython=True, snr_cython=True, ret_snr=True, 
        theta_ss=theta_ss, phi_ss=phi_ss, Phi0_ss=Phi0_ss, iota_ss=iota_ss, psi_ss=psi_ss,
        )
    dp_ss, snr_ss, gamma_ssi = vals_ss[0], vals_ss[1], vals_ss[2]
    # print(f"{np.mean(dp_ss)=}")
    df_ss = np.sum(dp_ss>thresh)/(nreals*nskies)
    df_bg = np.sum(dp_bg>thresh)/(nreals)
    ev_ss = expval_of_ss(gamma_ssi)
    # print(f"{np.mean(ev_ss)=}")

    dsdata = {
        'dp_ss':dp_ss, 'snr_ss':snr_ss, 'gamma_ssi':gamma_ssi, 
        'dp_bg':dp_bg, 'snr_bg':snr_bg,
        'df_ss':df_ss, 'df_bg':df_bg, 'ev_ss':ev_ss,
    }

    return dsdata


def detect_pspace_model_psrs(fobs_cents, hc_ss, hc_bg, psrs, nskies, hc_bg_noise=None,
                        thresh=DEF_THRESH, debug=False, nexcl_noise=0):
    
    nfreqs, nreals, nloudest = [*hc_ss.shape]
    dur = 1/fobs_cents[0]
    cad = 1.0 / (2 * fobs_cents[-1])
    # fobs_cents, fobs_edges = holo.utils.pta_freqs(dur)
    # dfobs = np.diff(fobs_edges)

    if hc_bg_noise is None:
        hc_bg_noise=hc_bg
    elif debug:
        print("Using different hc_bg_noise for SS detstats.")


    # Build ss skies
    if debug: print('Building ss skies.')
    theta_ss, phi_ss, Phi0_ss, iota_ss, psi_ss = _build_skies(nfreqs, nskies, nloudest)


    # Calculate DPs, SNRs, and DFs
    if debug: print('Calculating SS and BG detection statistics.')
    dp_bg, snr_bg = detect_bg_pta(psrs, fobs_cents, hc_bg, ret_snr=True)
    # print(f"{np.mean(dp_bg)=}")
    vals_ss = detect_ss_pta(
        psrs, fobs_cents, hc_ss, hc_bg_noise, 
        gamma_cython=True, snr_cython=True, ret_snr=True, 
        theta_ss=theta_ss, phi_ss=phi_ss, Phi0_ss=Phi0_ss, iota_ss=iota_ss, psi_ss=psi_ss,
        nexcl_noise=nexcl_noise
        )
    dp_ss, snr_ss, gamma_ssi = vals_ss[0], vals_ss[1], vals_ss[2]
    # print(f"{np.mean(dp_ss)=}")
    df_ss = np.sum(dp_ss>thresh)/(nreals*nskies)
    df_bg = np.sum(dp_bg>thresh)/(nreals)
    ev_ss = expval_of_ss(gamma_ssi)
    # print(f"{np.mean(ev_ss)=}")

    dsdata = {
        'dp_ss':dp_ss, 'snr_ss':snr_ss, 'gamma_ssi':gamma_ssi, 
        'dp_bg':dp_bg, 'snr_bg':snr_bg,
        'df_ss':df_ss, 'df_bg':df_bg, 'ev_ss':ev_ss,
    }

    return dsdata

def detect_pspace_model_clbrt_pta(
        fobs_cents, hc_ss, hc_bg, npsrs, nskies, 
        hc_bg_noise=None,
        sigstart=1e-6, sigmin=1e-9, sigmax=1e-4, tol=0.01, maxbads=5,
        thresh=DEF_THRESH, debug=False, save_snr_ss=False, save_gamma_ssi=True,
        red_amp=None, red_gamma=None, red2white=None, ss_noise=False, dsc_flag=False, nexcl_noise=0): 
    """ Detect pspace model using individual sigma calibration for each realization
    
    Parameters
    ----------
    fobs_cents : 1Darray
        GW frequencies in Hz
    hc_ss : (F,R,L) NDarray
    hc_bg : (F,R) NDarray
    npsrs : int
    nskies : int
    hc_bg_noise " (F,R) NDarray or None
        the background to use as single source noise, when normal hc_bg has extra SS added to it.
    red2white : scalar or None
        Fixed ratio between red and white noise amplitude, if not None. 
        Otherwise, red noise stays fixed
    nexcl_noise : int
        Number of loudest single sources to exclude from hc_rest noise, in addition 
        to the source in question.
    """
    dur = 1.0/fobs_cents[0]
    cad = 1.0/(2*fobs_cents[-1])

    nfreqs, nreals, nloudest = [*hc_ss.shape]

    if hc_bg_noise is None:
        hc_bg_noise=hc_bg
    elif debug:
        print("Using different hc_bg_noise for SS detstats.")
        if np.any(hc_bg_noise>hc_bg):
            err = f"hc_bg_noise excluding SS is somehow larger than hc_bg with SS added in!!"
            raise ValueError(err)
        
    # form arrays for individual realization detstats
    # set all to nan, only to be replaced if successful pta is found
    dp_ss = np.ones((nreals, nskies)) * np.nan   
    dp_bg = np.ones(nreals) * np.nan
    snr_ss = np.ones((nfreqs, nreals, nskies, nloudest)) * np.nan
    snr_bg = np.ones((nreals)) * np.nan
    gamma_ssi = np.ones((nfreqs, nreals, nskies, nloudest)) * np.nan
    sigmas = np.ones((nreals)) * np.nan


    # for each realization, 
    # use sigmin and sigmax from previous realization, 
    # unless it's the first realization of the sample
    _sigstart, _sigmin, _sigmax = sigstart, sigmin, sigmax 
    if debug: 
        mod_start = datetime.now()
        real_dur = datetime.now()
    failed_psrs=0
    for rr in range(nreals):
        if debug: 
            now = datetime.now()
            if (rr%100==99):
                print(f"{rr=}, {(now-real_dur)/100} s per realization, {_sigmin=:.2e}, {_sigmax=:.2e}, {_sigstart=:.2e}")
                real_dur = now


        # get calibrated psrs 
        psrs, red_amp, _sigstart, _sigmin, _sigmax = calibrate_one_pta(hc_bg[:,rr], hc_ss[:,rr,:], fobs_cents, npsrs, tol=tol, maxbads=maxbads,
                                    sigstart=_sigstart, sigmin=_sigmin, sigmax=_sigmax, debug=debug, ret_sig=True,
                                    red_amp=red_amp, red_gamma=red_gamma, red2white=red2white, ss_noise=ss_noise)
        _sigmin /= 2
        _sigmax *= 2 + 2e-20 # >1e-20 to make sure it doesnt immediately fail the 0 check 
        sigmas[rr] = _sigstart

        if psrs is None:
            failed_psrs += 1
            continue # leave values as nan, if no successful PTA was found


        # use those psrs to calculate realization detstats
        _dp_bg, _snr_bg = detect_bg_pta(psrs, fobs_cents, hc_bg[:,rr:rr+1],  hc_ss[:,rr:rr+1,:], ret_snr=True, red_amp=red_amp, red_gamma=red_gamma)

        dp_bg[rr], snr_bg[rr] = _dp_bg.squeeze(), _snr_bg.squeeze()


        # calculate SS noise from DeterSensitivityCurve and S_h,rest
        if dsc_flag:
            spectra = []
            for psr in psrs:
                sp = hsen.Spectrum(psr, freqs=fobs_cents)
                sp.NcalInv
                spectra.append(sp)
            sc_hc = hsen.DeterSensitivityCurve(spectra).h_c
            noise_dsc = sc_hc**2 / (12 * np.pi**2 * fobs_cents**3)
            noise_dsc = _dsc_noise(fobs_cents, nreals, npsrs, nloudest, psrs, spectra) # (P,F,R,L)
            # np.repeat(noise_dsc, npsrs*1*nloudest).reshape(nfreqs, npsrs, 1, nloudest) # (F,P,R,L)
            # noise_dsc = np.swapaxes(noise_dsc, 0, 1)  # (P,F,R,L)
            noise_rest = _Sh_rest_noise(hc_ss[:,rr:rr+1,:], hc_bg[:,rr:rr+1], fobs_cents) # (F,R,L)
            noise_ss = noise_dsc + noise_rest[np.newaxis,:,:,:] # (P,F,R,L)
        else:
            noise_ss = None

        # calculate realization SS detstats
        _dp_ss, _snr_ss, _gamma_ssi = detect_ss_pta(
            psrs, fobs_cents, hc_ss[:,rr:rr+1], hc_bg_noise[:,rr:rr+1], custom_noise=noise_ss,
            nskies=nskies, ret_snr=True, red_amp=red_amp, red_gamma=red_gamma, nexcl_noise=nexcl_noise)

        dp_ss[rr] = _dp_ss.reshape(nskies) # from R=1,S to S
        snr_ss[:,rr] = _snr_ss.reshape(nfreqs, nskies, nloudest) # from F,R=1,S,L to F,S,L
        gamma_ssi[:,rr,:,:] = _gamma_ssi.reshape(nfreqs, nskies, nloudest) # from F,R=1,S,L to F,S,L

    ev_ss = expval_of_ss(gamma_ssi)
    df_ss, df_bg = detfrac_of_reals(dp_ss, dp_bg)
    _dsdat = {
        'dp_ss':dp_ss, 'snr_ss':snr_ss, 'gamma_ssi':gamma_ssi, 
        'dp_bg':dp_bg, 'snr_bg':snr_bg,
        'df_ss':df_ss, 'df_bg':df_bg, 'ev_ss':ev_ss, 
        'sigmas':sigmas,
        }
    if save_gamma_ssi:
        _dsdat.update(gamma_ssi=gamma_ssi)
    if save_snr_ss:
        _dsdat.update(snr_ss=snr_ss)
    if debug:
        print(f"Model took {datetime.now() - mod_start} s, {failed_psrs}/{nreals} realizations failed.")
    return _dsdat


def detect_pspace_model_clbrt_pta_gsc(
        fobs_cents, hc_ss, hc_bg, npsrs, nskies, hc_bg_noise=None,
        sigstart=1e-6, sigmin=1e-9, sigmax=1e-4, tol=0.01, maxbads=5,
        thresh=DEF_THRESH, debug=False, save_snr_ss=False, save_gamma_ssi=True,
        red_amp=None, red_gamma=None, red2white=None, ss_noise=False,
        divide_flag=False, dsc_flag=False, nexcl_noise=0): 
    """ Detect pspace model using individual sigma calibration for each realization 
    and sensitivity curve noise for both BG calibration and SS detstats.
    
    Parameters
    ----------
    fobs_cents : 1Darray
        GW frequencies in Hz
    hc_ss : (F,R,L) NDarray
    hc_bg : (F,R) NDarray
    npsrs : int
    nskies : int
    hc_bg_noise " (F,R) NDarray or None
        the background to use as single source noise, when normal hc_bg has extra SS added to it.
    red2white : scalar or None
        Fixed ratio between red and white noise amplitude, if not None. 
        Otherwise, red noise stays fixed
    nexcl_noise : int
        Number of loudest single sources to exclude from hc_rest noise, in addition 
        to the source in question.
    """
    dur = 1.0/fobs_cents[0]
    cad = 1.0/(2*fobs_cents[-1])
    if hc_bg_noise is None:
        hc_bg_noise=hc_bg

    nfreqs, nreals, nloudest = [*hc_ss.shape]
        
    # form arrays for individual realization detstats
    # set all to nan, only to be replaced if successful pta is found
    dp_ss = np.ones((nreals, nskies)) * np.nan   
    dp_bg = np.ones(nreals) * np.nan
    snr_ss = np.ones((nfreqs, nreals, nskies, nloudest)) * np.nan
    snr_bg = np.ones((nreals)) * np.nan
    gamma_ssi = np.ones((nfreqs, nreals, nskies, nloudest)) * np.nan


    # for each realization, 
    # use sigmin and sigmax from previous realization, 
    # unless it's the first realization of the sample
    _sigstart, _sigmin, _sigmax = sigstart, sigmin, sigmax 
    if debug: 
        mod_start = datetime.now()
        real_dur = datetime.now()
    failed_psrs=0
    for rr in range(nreals):
        if debug: 
            now = datetime.now()
            if (rr%100==99):
                print(f"{rr=}, {(now-real_dur)/100} s per realization, {_sigmin=:.2e}, {_sigmax=:.2e}, {_sigstart=:.2e}")
                real_dur = now


        # get calibrated psrs 
        psrs, red_amp, _sigstart, _sigmin, _sigmax, spectra, noise_gsc = calibrate_one_pta_gsc(
            hc_bg[:,rr], fobs_cents, npsrs, tol=tol, maxbads=maxbads,
            sigstart=_sigstart, sigmin=_sigmin, sigmax=_sigmax, debug=debug, ret_sig=True,
            red_amp=red_amp, red_gamma=red_gamma, red2white=red2white, ss_noise=ss_noise, divide_flag=divide_flag)
        _sigmin /= 2
        _sigmax *= 2 + 2e-20 # >1e-20 to make sure it doesnt immediately fail the 0 check 

        if psrs is None:
            failed_psrs += 1
            continue # leave values as nan, if no successful PTA was found


        # use those psrs to calculate realization detstats
        _dp_bg, _snr_bg = detect_bg_pta(psrs, fobs_cents, hc_bg[:,rr:rr+1],  hc_ss[:,rr:rr+1,:], custom_noise=noise_gsc,
                                        ret_snr=True, red_amp=red_amp, red_gamma=red_gamma)

        dp_bg[rr], snr_bg[rr] = _dp_bg.squeeze(), _snr_bg.squeeze()


        # calculate SS noise from DeterSensitivityCurve and S_h,rest
        if dsc_flag: # calculate the DSC noise separately
            noise_dsc = _dsc_noise(fobs_cents, spectra=spectra, nreals=1, npsrs=npsrs, nloudest=nloudest, divide_flag=divide_flag)
        else: # use the GSC as the DSC noise
            noise_dsc = np.repeat(noise_gsc, nloudest).reshape(npsrs, nfreqs, 1, nloudest) # (P,F,R,L)
        noise_rest = _Sh_rest_noise(hc_ss[:,rr:rr+1,:], hc_bg[:,rr:rr+1], fobs_cents) # (F,R,L)
        noise_ss = noise_dsc + noise_rest[np.newaxis,:,:,:] # (P,F,R,L)


        # calculate realizatoin SS detstats
        _dp_ss, _snr_ss, _gamma_ssi = detect_ss_pta(
            psrs, fobs_cents, hc_ss[:,rr:rr+1], hc_bg_noise[:,rr:rr+1], custom_noise=noise_ss,
            nskies=nskies, ret_snr=True, red_amp=red_amp, red_gamma=red_gamma, nexcl_noise=nexcl_noise)

        dp_ss[rr], snr_ss[:,rr], gamma_ssi[:,rr] = _dp_ss.squeeze(), _snr_ss.squeeze(), _gamma_ssi.squeeze()

    ev_ss = expval_of_ss(gamma_ssi)
    df_ss, df_bg = detfrac_of_reals(dp_ss, dp_bg)
    _dsdat = {
        'dp_ss':dp_ss, 'snr_ss':snr_ss, 'gamma_ssi':gamma_ssi, 
        'dp_bg':dp_bg, 'snr_bg':snr_bg,
        'df_ss':df_ss, 'df_bg':df_bg, 'ev_ss':ev_ss,
        }
    if save_gamma_ssi:
        _dsdat.update(gamma_ssi=gamma_ssi)
    if save_snr_ss:
        _dsdat.update(snr_ss=snr_ss)
    print(f"Model took {datetime.now() - mod_start} s, {failed_psrs}/{nreals} realizations failed.")
    return _dsdat


def detect_pspace_model_clbrt_ramp(fobs_cents, hc_ss, hc_bg, npsrs, nskies, sigma,
                        rampstart=1e-16, rampmin=1e-20, rampmax=1e-13, tol=0.01, maxbads=5,
                        thresh=DEF_THRESH, debug=False, save_snr_ss=False, save_gamma_ssi=True,
                        red_amp=None, red_gamma=None, ss_noise=False): 
    """ Detect pspace model using individual red noise amplitude calibration for each realization

    NOTE: Not supported, not updated for including single sources as noise for BG.
    
    """
    dur = 1.0/fobs_cents[0]
    cad = 1.0/(2*fobs_cents[-1])

    nfreqs, nreals, nloudest = [*hc_ss.shape]
        
    # form arrays for individual realization detstats
    # set all to nan, only to be replaced if successful pta is found
    dp_ss = np.ones((nreals, nskies)) * np.nan   
    dp_bg = np.ones(nreals) * np.nan
    snr_ss = np.ones((nfreqs, nreals, nskies, nloudest)) * np.nan
    snr_bg = np.ones((nreals)) * np.nan
    gamma_ssi = np.ones((nfreqs, nreals, nskies, nloudest)) * np.nan

    # get psrs 
    phis = np.random.uniform(0, 2*np.pi, size = npsrs)
    thetas = np.random.uniform(np.pi/2, np.pi/2, size = npsrs)
    psrs = hsim.sim_pta(timespan=dur/YR, cad=1/(cad/YR), sigma=sigma,
                    phi=phis, theta=thetas)

    # for each realization, 
    # use sigmin and sigmax from previous realization, 
    # unless it's the first realization of the sample
    _rampstart, _rampmin, _rampmax = rampstart, rampmin, rampmax 
    if debug: 
        mod_start = datetime.now()
        real_dur = datetime.now()
    failed_psrs=0
    for rr in range(nreals):
        if debug: 
            now = datetime.now()
            if (rr%10==0):
                print(f"{rr=}, {now-real_dur} s per realization, {_rampmin=:.2e}, {_rampmax=:.2e}, {_rampstart=:.2e}")
            real_dur = now

        # get calibrated psrs 
        ramp, _rampmin, _rampmax = calibrate_one_ramp(hc_bg[:,rr], fobs_cents, psrs,
                                    tol=tol, maxbads=maxbads,
                                    rampstart=_rampstart, rampmin=_rampmin, rampmax=_rampmax, debug=debug, 
                                    rgam=red_gamma, ss_noise=ss_noise)
        _rampstart = ramp
        _rampmin /= 2
        _rampmax *= 2 + 2e-50 # >1e-20 to make sure it doesnt immediately fail the 0 check 

        if ramp is None:
            failed_psrs += 1
            continue # leave values as nan, if no successful PTA was found
        # print(f"before calculation: {utils.stats(psrs[0].toaerrs)=}, \n{utils.stats(hc_bg[rr])=},\
        #         {utils.stats(fobs_cents)=}")
        # use those psrs to calculate realization detstats
        _dp_bg, _snr_bg = detect_bg_pta(psrs, fobs_cents, hc_bg[:,rr:rr+1], ret_snr=True, red_amp=ramp, red_gamma=red_gamma)
        # print(f"{utils.stats(psrs[0].toaerrs)=}, {utils.stats(hc_bg[rr])=},\
        #         {_dp_bg=},")
        # _dp_bg,  = detect_bg_pta(psrs, fobs_cents, hc_bg=hc_bg[:,rr:rr+1], red_amp=red_amp, red_gamma=red_gamma) #, ret_snr=True)
        # print(f"test2: {_dp_bg=}")
        dp_bg[rr], snr_bg[rr] = _dp_bg.squeeze(), _snr_bg.squeeze()
        _dp_ss, _snr_ss, _gamma_ssi = detect_ss_pta(
            psrs, fobs_cents, hc_ss[:,rr:rr+1], hc_bg[:,rr:rr+1], nskies=nskies, ret_snr=True, red_amp=ramp, red_gamma=red_gamma)
        # if debug: print(f"{_dp_ss.shape=}, {_snr_ss.shape=}, {_gamma_ssi.shape=}")
        dp_ss[rr], snr_ss[:,rr], gamma_ssi[:,rr] = _dp_ss.squeeze(), _snr_ss.squeeze(), _gamma_ssi.squeeze()

    ev_ss = expval_of_ss(gamma_ssi)
    df_ss, df_bg = detfrac_of_reals(dp_ss, dp_bg)
    _dsdat = {
        'dp_ss':dp_ss, 'snr_ss':snr_ss, 'gamma_ssi':gamma_ssi, 
        'dp_bg':dp_bg, 'snr_bg':snr_bg,
        'df_ss':df_ss, 'df_bg':df_bg, 'ev_ss':ev_ss,
        }
    if save_gamma_ssi:
        _dsdat.update(gamma_ssi=gamma_ssi)
    if save_snr_ss:
        _dsdat.update(snr_ss=snr_ss)
    print(f"Model took {datetime.now() - mod_start} s, {failed_psrs}/{nreals} realizations failed.")
    return _dsdat



def detect_pspace_model_clbrt_sigma(fobs_cents, hc_ss, hc_bg, 
                        npsrs, nskies, maxtrials=1): 
    """ Detect pspace model using individual PTA calibration for each realization
    
    """
    dur = 1.0/fobs_cents[0]
    cad = 1.0/(2*fobs_cents[-1])

    nfreqs, nreals, nloudest = [*hc_ss.shape]
    # get calibrated sigmas 
    sigmas, avg_dps, std_dps = calibrate_all_sigma(hc_bg, fobs_cents, npsrs, maxtrials=maxtrials)
        
    # form arrays for individual realization detstats
    dp_ss = np.zeros((nreals, nskies))     
    dp_bg = np.zeros(nreals)
    snr_ss = np.zeros((nfreqs, nreals, nskies, nloudest))
    snr_bg = np.zeros((nreals))
    gamma_ssi = np.zeros((nfreqs, nreals, nskies, nloudest))

    # for each realization, get individual detstats   
    for rr in range(nreals):
            # get psrs for the given calibrated sigma
            psrs = _build_pta(npsrs, sigmas[rr], dur, cad)
            # use those psrs to calculate realization detstats
            _dp_bg, _snr_bg = detect_bg_pta(psrs, fobs_cents, hc_bg[:,rr:rr+1], ret_snr=True)
            dp_bg[rr], snr_bg[rr] = _dp_bg.squeeze(), _snr_bg.squeeze()
            _dp_ss, _snr_ss, _gamma_ssi = detect_ss_pta(
                psrs, fobs_cents, hc_ss[:,rr:rr+1], hc_bg[:,rr:rr+1], ret_snr=True)
            dp_ss[rr], snr_ss[:,rr], gamma_ssi[:,rr] = _dp_ss.squeeze(), _snr_ss.squeeze(), _gamma_ssi.squeeze()
    ev_ss = expval_of_ss(gamma_ssi)
    df_ss, df_bg = detfrac_of_reals(dp_ss, dp_bg)
    _dsdat = {
        'dp_ss':dp_ss, 'snr_ss':snr_ss, 'gamma_ssi':gamma_ssi, 
        'dp_bg':dp_bg, 'snr_bg':snr_bg,
        'df_ss':df_ss, 'df_bg':df_bg, 'ev_ss':ev_ss,
        }
    return _dsdat


########################################################################
############################ Calibrate PTA ############################# 
########################################################################

def binary_sigma_calibration(hc_bg, fobs, npsrs, maxtrials=2, debug=False, 
                          sig_start = 1e-6, sig_min = 1e-10, sig_max = 1e-3, ):
    """ Calibrate the PTA to a 50% target DP for a given model, average over many realizations
    
    # BUG: This seems to get stuck on bad guesses, when requiring high max trials. 
    # TODO: Set up a check for bar guesses. 
    """
    dur = 1.0/fobs[0]
    cad = 1.0/(2*fobs[-1])
    sigma=sig_start

    avg_dp = 0
    trials = 1
    enough_trials = False
    while avg_dp<.495 or avg_dp>.505 or enough_trials==False: # must be within the desired target range using the max number of trials
        avg_dp, std_dp = _get_dpbg(hc_bg, npsrs=npsrs, sigma=sigma, trials=trials,
                          fobs=fobs, dur=dur, cad=cad)
        if debug: print(f"{avg_dp=}, {sigma=}, {sig_min=}, {sig_max=}, {trials=}")
        
        # if we're close, raise the number of trials
        if avg_dp>0.4 and avg_dp<0.5:
            if trials == maxtrials:
                enough_trials=True
            else:
                trials=maxtrials

        if avg_dp>=0.51:  # avg_dp too high, raise sigma
            sig_min = sigma
            sigma = np.mean([sig_min, sig_max])
        elif avg_dp<=0.49: # avg_dp too low, decrease sigma
            sig_max = sigma
            sigma = np.mean([sig_min, sig_max])     
    return sigma, avg_dp, std_dp

def _get_dpbg(hc_bg, npsrs, sigma, trials, fobs, dur, cad,):
    nreals = hc_bg.shape[-1]
    all_dp = np.zeros((trials, nreals))

    for ii in range(trials):
        # build PTA
        phis = np.random.uniform(0, 2*np.pi, size = npsrs)
        thetas = np.random.uniform(np.pi/2, np.pi/2, size = npsrs)
        psrs = hsim.sim_pta(timespan=dur/YR, cad=1/(cad/YR), sigma=sigma,
                        phi=phis, theta=thetas)
        # calculate bg detprob of each realizations for the given PTA
        all_dp[ii] = detect_bg_pta(psrs, fobs, hc_bg=hc_bg)

    avg_dp = np.mean(all_dp)
    std_dp = np.std(all_dp)
    return avg_dp, std_dp

def calibrate_all_sigma(hc_bg, fobs, npsrs, maxtrials, 
                         sig_start=1e-6, sig_min=1e-9, sig_max=1e-4, debug=False):
    """ Calibrate the PTA independently for each background realizations

    Parameters
    ----------
    hc_bg : (F,R) Ndarray
    fobs : (F,) 1Darray
    npsrs : integer
    maxtrials : integer

    Returns
    -------
    rsigmas : (R,) 1Darray
        Calibrated PTA white noise sigma for each realization.
    avg_dps : (R,) 1Darray
        Average background detprob across PTA realizations, calculated for the realization's PTA.
    std_dps : (R,) 1Darray
        Standard deviation among PTA realizations for the given bg realizations.

    """
    nreals = hc_bg.shape[-1]
    rsigmas = np.zeros(nreals)
    avg_dps = np.zeros(nreals)
    std_dps = np.zeros(nreals)
    for rr in range(nreals):
        hc_bg_rr = hc_bg[:,rr:rr+1]
        # print(hc_bg_rr.shape)
        rsigmas[rr], avg_dps[rr], std_dps[rr] = binary_sigma_calibration(
            hc_bg_rr, fobs, npsrs, maxtrials, debug=debug, 
            sig_start = sig_start, sig_min=sig_min, sig_max=sig_max, 
        )
    return rsigmas, avg_dps, std_dps


def _red_amp_from_white_noise(cad, sigma, red2white, fref=1/YR):
    red_amp = np.sqrt(12 * np.pi**2 * fref**3 * 
                      red2white * _white_noise(cad, sigma)) 
    return red_amp

def calibrate_one_pta(hc_bg, hc_ss, fobs, npsrs, 
                      sigstart=1e-6, sigmin=1e-9, sigmax=1e-4, debug=False, maxbads=20, tol=0.03,
                      phis=None, thetas=None, ret_sig = False, red_amp=None, red_gamma=None, red2white=None,
                      ss_noise=False):
    """ Calibrate the specific PTA for a given realization, and return that PTA

    Parameters
    ----------
    hc_bg : (F,) 1Darray
        The background characteristic strain for one realization.
    hc_ss : (F,L) NDarray
        The SS characteristic strains for one realization
    fobs : (F,) 1Darray
        Observed GW frequencies.
    npsrs : integer
        Number of pulsars.

    Returns 
    -------
    psrs : hasasia.sim.pta object
        Calibrated PTA.
    sigmin : float
        minimum of the final sigma range used, returned only if ret_sig=True
    sigmax : float, returned only if ret_sig=True
        maximum of the final sigma range used
    sigma : float
        final sigma, returned only if ret_sig=True

    """

    # get duration and cadence from fobs
    dur = 1.0/fobs[0]
    cad = 1.0/(2.0*fobs[-1])

    # randomize pulsar positions
    if phis is None: phis = np.random.uniform(0, 2*np.pi, size = npsrs)
    if thetas is None: thetas = np.random.uniform(np.pi/2, np.pi/2, size = npsrs)
    sigma = sigstart
    if red2white is not None:
        red_amp = _red_amp_from_white_noise(cad, sigma, red2white) 

    psrs = hsim.sim_pta(timespan=dur/YR, cad=1/(cad/YR), sigma=sigma,
                    phi=phis, theta=thetas)
    dp_bg = detect_bg_pta(psrs, fobs, hc_bg=hc_bg[:,np.newaxis], hc_ss=hc_ss[:,np.newaxis,:],
                            red_amp=red_amp, red_gamma=red_gamma, ss_noise=ss_noise)[0]

    nclose=0 # number of attempts close to 0.5, could be stuck close
    nfar=0 # number of attempts far from 0.5, could be stuck far

    # calibrate sigma
    while np.abs(dp_bg-0.50)>tol:
        sigma = np.mean([sigmin, sigmax]) # a weighted average would be better
        if red2white is not None:
            red_amp = _red_amp_from_white_noise(cad, sigma, red2white) 
        psrs = hsim.sim_pta(timespan=dur/YR, cad=1/(cad/YR), sigma=sigma,
                        phi=phis, theta=thetas)
        dp_bg = detect_bg_pta(psrs, fobs, hc_bg=hc_bg[:,np.newaxis], hc_ss=hc_ss[:,np.newaxis,:],
                                 red_amp=red_amp, red_gamma=red_gamma, ss_noise=ss_noise)[0]

        # if debug: print(f"{dp_bg=}")
        if (dp_bg < (0.5-tol)) or (dp_bg > (0.5+tol)):
            nfar +=1

        # check if we need to expand the range
        if (nfar>5*maxbads  # if we've had many bad guesses
            or (sigmin/sigmax > 0.99 # or our range is small and we are far from the goal
                and (dp_bg<0.4 or dp_bg>0.6))):
            
            # then we must expand the range
            if debug: print(f"STUCK! {nfar=}, {dp_bg=}, {sigmin=:e}, {sigmax=:e}")
            if dp_bg < 0.5-tol: # stuck way too low, allow much lower sigmin to raise DP
                sigmin = sigmin/3
            if dp_bg > 0.5+tol: # stuck way too high, allow much higher sigmax to lower DP
                sigmax = sigmax*3

            # reset count for far guesses
            nfar = 0

        # check how we should narrow our range
        if dp_bg<0.5-tol: # dp too low, lower sigma
            sigmax = sigma
        elif dp_bg>0.5+tol: # dp too high, raise sigma
            sigmin = sigma
        else:
            nclose += 1 # check how many attempts between 0.49 and 0.51 fail

        # check if we are stuck near the goal value with a bad range    
        if nclose>maxbads: # if many fail, we're stuck; expand sampling range
            if debug: print(f"{nclose=}, {dp_bg=}, {sigmin=:e}, {sigmax=:e}")
            sigmin = sigmin/3
            sigmax = sigmax*3
            nclose=0

        # check if goal DP is just impossible
        if sigmax<1e-20:
            psrs=None
            if debug: print(f"FAILED! DP_BG=0.5 impossible with {red_amp=}, {red_gamma=}")
            break
    # print(f"test1: {dp_bg=}")
    # print(f"test1: {sigma=}")
    # print(f"in calibration: {utils.stats(psrs[0].toaerrs)=}, \n{utils.stats(hc_bg)=},\
    #             {utils.stats(fobs)=}, {dp_bg=}")
    if ret_sig:
        return psrs, red_amp, sigma, sigmin, sigmax
    return psrs


def calibrate_one_pta_gsc(hc_bg, fobs, npsrs, 
                      sigstart=1e-6, sigmin=1e-9, sigmax=1e-4, debug=False, maxbads=20, tol=0.03,
                      phis=None, thetas=None, ret_sig = False, red_amp=None, red_gamma=None, red2white=None,
                      ss_noise=False, divide_flag=False,):
    """ Calibrate the specific PTA for a given realization, and return that PTA

    Parameters
    ----------
    hc_bg : (F,) 1Darray
        The background characteristic strain for one realization.
    hc_ss : (F,L) NDarray
        The SS characteristic strains for one realization
    fobs : (F,) 1Darray
        Observed GW frequencies.
    npsrs : integer
        Number of pulsars.
    divide_flag : Bool
        Whether or not to divide the GSC noise among the pulsars

    Returns 
    -------
    psrs : hasasia.sim.pta object
        Calibrated PTA.
    sigmin : float
        minimum of the final sigma range used, returned only if ret_sig=True
    sigmax : float, returned only if ret_sig=True
        maximum of the final sigma range used
    sigma : float
        final sigma, returned only if ret_sig=True

    """

    # get duration and cadence from fobs
    dur = 1.0/fobs[0]
    cad = 1.0/(2.0*fobs[-1])

    # randomize pulsar positions
    if phis is None: phis = np.random.uniform(0, 2*np.pi, size = npsrs)
    if thetas is None: thetas = np.random.uniform(np.pi/2, np.pi/2, size = npsrs)
    sigma = sigstart
    if red2white is not None:
        red_amp = _white_noise(cad, sigma) * red2white

    psrs = hsim.sim_pta(timespan=dur/YR, cad=1/(cad/YR), sigma=sigma,
                    phi=phis, theta=thetas)
    
    # get sensitivity curve
    spectra, noise_gsc = psrs_spectra_gwbnoise(psrs, fobs, nreals=1, npsrs=npsrs, divide_flag=divide_flag) 
    
    dp_bg = detect_bg_pta(psrs, fobs, hc_bg=hc_bg[:,np.newaxis], custom_noise=noise_gsc,
                            red_amp=red_amp, red_gamma=red_gamma, ss_noise=ss_noise)[0]

    nclose=0 # number of attempts close to 0.5, could be stuck close
    nfar=0 # number of attempts far from 0.5, could be stuck far

    # calibrate sigma
    while np.abs(dp_bg-0.50)>tol:
        sigma = np.mean([sigmin, sigmax]) # a weighted average would be better
        if red2white is not None:
            red_amp = sigma * red2white
        psrs = hsim.sim_pta(timespan=dur/YR, cad=1/(cad/YR), sigma=sigma,
                        phi=phis, theta=thetas)
        spectra, noise_gsc = psrs_spectra_gwbnoise(psrs, fobs, nreals=1, npsrs=npsrs) 
        dp_bg = detect_bg_pta(psrs, fobs, hc_bg=hc_bg[:,np.newaxis], custom_noise=noise_gsc,
                                 red_amp=red_amp, red_gamma=red_gamma, ss_noise=ss_noise)[0]

        # if debug: print(f"{dp_bg=}")
        if (dp_bg < (0.5-tol)) or (dp_bg > (0.5+tol)):
            nfar +=1

        # check if we need to expand the range
        if (nfar>5*maxbads  # if we've had many bad guesses
            or (sigmin/sigmax > 0.99 # or our range is small and we are far from the goal
                and (dp_bg<0.4 or dp_bg>0.6))):
            
            # then we must expand the range
            if debug: print(f"STUCK! {nfar=}, {dp_bg=}, {sigmin=:e}, {sigmax=:e}")
            if dp_bg < 0.5-tol: # stuck way too low, allow much lower sigmin to raise DP
                sigmin = sigmin/3
            if dp_bg > 0.5+tol: # stuck way too high, allow much higher sigmax to lower DP
                sigmax = sigmax*3

            # reset count for far guesses
            nfar = 0

        # check how we should narrow our range
        if dp_bg<0.5-tol: # dp too low, lower sigma
            sigmax = sigma
        elif dp_bg>0.5+tol: # dp too high, raise sigma
            sigmin = sigma
        else:
            nclose += 1 # check how many attempts between 0.49 and 0.51 fail

        # check if we are stuck near the goal value with a bad range    
        if nclose>maxbads: # if many fail, we're stuck; expand sampling range
            if debug: print(f"{nclose=}, {dp_bg=}, {sigmin=:e}, {sigmax=:e}")
            sigmin = sigmin/3
            sigmax = sigmax*3
            nclose=0

        # check if goal DP is just impossible
        if sigmax<1e-20:
            psrs=None
            if debug: print(f"FAILED! DP_BG=0.5 impossible with {red_amp=}, {red_gamma=}")
            break
    # print(f"test1: {dp_bg=}")
    # print(f"test1: {sigma=}")
    # print(f"in calibration: {utils.stats(psrs[0].toaerrs)=}, \n{utils.stats(hc_bg)=},\
    #             {utils.stats(fobs)=}, {dp_bg=}")
    if ret_sig:
        return psrs, red_amp, sigma, sigmin, sigmax, spectra, noise_gsc
    return psrs, red_amp


def calibrate_one_ramp(hc_bg, hc_ss, fobs, psrs,
                      rampstart=1e-6, rampmin=1e-9, rampmax=1e-4, debug=False, maxbads=20, tol=0.03,
                      phis=None, thetas=None, rgam=-1.5, ss_noise=False):
    """ Calibrate the red noise amplitude, for a given realization, and return that PTA

    Parameters
    ----------
    hc_bg : (F,) 1Darray
        The background characteristic strain for one realization.
    hc_ss : (F,L) NDarray
        The SS characteristic strains for one realization
    fobs : (F,) 1Darray
        Observed GW frequencies.
    psrs : hasasia.sim.pta object
        PTA w/ fixed white noise
    sigma : scalar
        White noise sigma

    Returns 
    -------
    redamp : float
        final redamp, returned only if ret_ramp = True
    redampmin : float
        minimum of the final sigma range used, returned only if ret_sig=True
    redampmax : float, returned only if ret_sig=True
        maximum of the final sigma range used

    """

    # get duration and cadence from fobs
    dur = 1.0/fobs[0]
    cad = 1.0/(2.0*fobs[-1])

    # randomize pulsar positions
    ramp = rampstart
    dp_bg = detect_bg_pta(psrs, fobs, hc_bg=hc_bg[:,np.newaxis], hc_ss=hc_ss[:,np.newaxis,:],
                red_amp=rgam, red_gamma=ramp, ss_noise=ss_noise)[0]

    nclose=0 # number of attempts close to 0.5, could be stuck close
    nfar=0 # number of attempts far from 0.5, could be stuck far

    # calibrate sigma
    while np.abs(dp_bg-0.50)>tol:
        ramp = np.mean([rampmin, rampmax]) # a weighted average would be better
        dp_bg = detect_bg_pta(psrs, fobs, hc_bg=hc_bg[:,np.newaxis], hc_ss=hc_ss[:,np.newaxis,:],
                    red_amp=ramp, red_gamma=rgam, ss_noise=ss_noise)[0]

        # if debug: print(f"{dp_bg=}")
        if (dp_bg < (0.5-tol)) or (dp_bg > (0.5+tol)):
            nfar +=1

        # check if we need to expand the range
        if (nfar>5*maxbads  # if we've had many bad guesses
            or (rampmin/rampmax > 0.99 # or our range is small and we are far from the goal
                and (dp_bg<0.4 or dp_bg>0.6))):
            
            # then we must expand the range
            if debug: print(f"STUCK! {nfar=}, {dp_bg=}, {rampmin=:e}, {rampmax=:e}")
            if dp_bg < 0.5-tol: # stuck way too low, allow much lower sigmin to raise DP
                rampmin = rampmin/3
            if dp_bg > 0.5+tol: # stuck way too high, allow much higher sigmax to lower DP
                rampmax = rampmax*3

            # reset count for far guesses
            nfar = 0

        # check how we should narrow our range
        if dp_bg<0.5-tol: # dp too low, lower sigma
            rampmax = ramp
        elif dp_bg>0.5+tol: # dp too high, raise sigma
            rampmin = ramp
        else:
            nclose += 1 # check how many attempts between 0.49 and 0.51 fail

        # check if we are stuck near the goal value with a bad range    
        if nclose>maxbads: # if many fail, we're stuck; expand sampling range
            if debug: print(f"{nclose=}, {dp_bg=}, {rampmin=:e}, {rampmax=:e}")
            rampmin = rampmin/3
            rampmax = rampmax*3
            nclose=0

        # check if goal DP is just impossible
        if rampmax<1e-50:
            ramp=None
            if debug: print(f"FAILED! DP_BG=0.5 impossible with sigma={np.mean(psrs[0].toaerrs)}, {rgam=}")
            break
    # print(f"test1: {dp_bg=}")
    # print(f"test1: {sigma=}")
    # print(f"in calibration: {utils.stats(psrs[0].toaerrs)=}, \n{utils.stats(hc_bg)=},\
    #             {utils.stats(fobs)=}, {dp_bg=}")
    return ramp, rampmin, rampmax



########################################################################
########################## Average Frequency ########################### 
########################################################################

def weighted_mean_variance(data, weights, debug=False,):
    """ Calculate the weighted average frequency and variance
    
    Parameters
    ----------
    data: NDarray
        Data
    weights: NDarray
        Weights

    Returns
    -------
    mean : float
        Weighted mean
    var2 : float
        Weighted variance (std^2)
    """
    mean = np.sum(weights * data) / np.sum(weights)
    if debug: print(f"{mean=}")
    var2 = np.sum(weights * (data - mean)**2) 
    nn = data.size
    var2 /= (nn-1)/nn * np.sum(weights)
    if debug: print(f"{var2=}")
    return mean, var2








########################################################################
###################### Functions Using V21 Models ######################
########################################################################

def get_data(
        target, dets=True,
        nvars=21, nreals=500, nskies=100, shape=None,  # keep as defaults
        nloudest = 10, bgl = 10, cv=None, ssn_flag=False,
        red_gamma = None, red2white=None, 
        gsc_flag=False,  dsc_flag=False, divide_flag=False, nexcl=0,
        gw_only=False, 
        var_hard_time=None, npsrs=40,
):
    if gw_only:
        path = '/Users/emigardiner/GWs/holodeck/output/anatomy_7GW'
    else:
        path = '/Users/emigardiner/GWs/holodeck/output/anatomy_redz'

    output_path = path+f'/{target}_v{nvars}_r{nreals}_shape{str(shape)}'

    if var_hard_time is not None:
        output_path+=f"_vtau{var_hard_time}"

    data_file = output_path +f'/data_params' 
    dets_file = output_path + f'/detstats_s{nskies}' 


    if nloudest != 10:                                           # if using nloudest that isn't the default 10
        dets_file += f"_l{nloudest}" 
        data_file += f"_l{nloudest}"
    if bgl != nloudest:
        dets_file += f"_bgl{bgl}" # only change nloudest subtracted from bg, not single sources loudest
    if cv is not None: 
        dets_file += f"_cv{cv}"        # if using one variation to calibrate
    if ssn_flag: 
        dets_file += '_ssn'                               # if using single sources as noise

    if gsc_flag:                                                           # if using GSC as noise
        dets_file += '_gsc'
        if dsc_flag is False:                                          # if using GSC as noise
            dets_file += 'both'
        if divide_flag:
            dets_file += '_divide'
        else:
            dets_file += '_nodiv'
    if dsc_flag: 
        dets_file += '_dsc'
    
    if nexcl>0:
        dets_file += f'_nexcl{nexcl}'

    if npsrs != 40:
        dets_file += f'_p{npsrs}'

    if red2white is not None and red_gamma is not None:               # if using red noise with fixed red_gamma
        dets_file += f'_r2w{red2white:.1e}_rg{red_gamma:.1f}'
    else: 
        dets_file += f'_white'

    dets_file += '.npz'
    data_file += '.npz'

    print(f'{data_file=}')
    print(f'{dets_file=}')

    if os.path.exists(data_file) is False:
        err = f"load data file '{data_file}' does not exist, you need to construct it."
        raise Exception(err)
    if os.path.exists(dets_file) is False and dets is True:
        err = f"load dets file '{dets_file}' does not exist, you need to construct it."
        raise Exception(err)
    file = np.load(data_file, allow_pickle=True)
    data = file['data']
    params = file['params']
    file.close()
    if dets is False:
        return data, params
    print(f"got data from {data_file}")
    file = np.load(dets_file, allow_pickle=True)
    print(f"got detstats from {dets_file}")
    # print(file.files)
    dsdat = file['dsdat']
    file.close()

    return data, params, dsdat

def append_filename(filename='', 
        gw_only=False, red_gamma = None, red2white=None, 
        nloudest = 10, bgl = 10, cv=None, 
        gsc_flag=False,  dsc_flag=False, divide_flag=False, nexcl=0,
        var_hard_time=None, 
        
        ):
    
    if cv is not None:
        filename += f"_cv{cv}"

    if nloudest != 10:
        filename += f"_l{nloudest}"        
    if bgl != nloudest:
        filename += f"_bgl{bgl}"
    
    if var_hard_time is not None:
        filename += f"_vtau{var_hard_time}"

    if red2white is not None and red_gamma is not None:
        filename += f"_r2w{red2white:.1e}_rg{red_gamma:.1f}"

    if gsc_flag: 
        filename = filename + '_gsc'
        if dsc_flag: filename += 'both'
        if divide_flag:
            filename += '_divide'
        else:
            filename += '_nodiv'
    elif dsc_flag: filename += '_dsc'

    if nexcl>0:
        filename += f'_nexcl{nexcl}'

    if gw_only:
        filename = filename+'_gw'
    

    return filename

def build_ratio_arrays(
        target, nreals=500, nskies=100,
        gw_only=False, red2white=None, red_gamma=None, 
        nloudest=10, bgl=1, 
        gsc_flag=False, dsc_flag=False, divide_flag=False, nexcl=0,
        var_hard_time=None, npsrs=40,
        figpath = '/Users/emigardiner/GWs/holodeck/output/anatomy_redz/figdata/ratio',):

    data, params, dsdat = get_data(target,
        gw_only=gw_only, red2white=red2white, red_gamma=red_gamma,
        nloudest=nloudest, bgl=bgl, nexcl=nexcl,
        gsc_flag=gsc_flag, dsc_flag=dsc_flag, divide_flag=divide_flag,
        var_hard_time=var_hard_time, npsrs=npsrs)
    xx=[]
    yy=[]
    for pp, par in enumerate(params):
        xx.append(params[pp][target])
        dp_bg = np.repeat(dsdat[pp]['dp_bg'], nskies).reshape(nreals, nskies)
        ev_ss = dsdat[pp]['ev_ss']
        yy.append(ev_ss/dp_bg)

    filename = figpath+f'/ratio_arrays_{target}'
    filename = append_filename(
        filename, 
        gw_only=gw_only, red_gamma=red_gamma, red2white=red2white, 
        nloudest=nloudest, bgl=bgl, cv=None, 
        gsc_flag=gsc_flag, dsc_flag=dsc_flag, divide_flag=divide_flag, nexcl=nexcl,
        var_hard_time=var_hard_time)
    filename += '.npz'  
    print(f'saving to {filename}')
    np.savez(filename, xx_params = xx, yy_ratio = yy,)

def get_ratio_arrays(
        target, nreals=500, nskies=100,
        gw_only=False, red2white=None, red_gamma=None, 
        nloudest=10, bgl=1, 
        gsc_flag=False, dsc_flag=False, divide_flag=False, nexcl=0,
        var_hard_time=None,
        figpath = '/Users/emigardiner/GWs/holodeck/output/anatomy_redz/figdata/ratio',):

    filename = figpath+f'/ratio_arrays_{target}'
    filename = append_filename(
        filename, 
        gw_only=gw_only, red_gamma=red_gamma, red2white=red2white, 
        nloudest=nloudest, bgl=bgl, cv=None, 
        gsc_flag=gsc_flag, dsc_flag=dsc_flag, divide_flag=divide_flag, nexcl=nexcl,
        var_hard_time=var_hard_time)
    filename += '.npz'  

    file = np.load(filename)
    xx_params = file['xx_params']
    yy_ratio = file['yy_ratio']
    file.close()
    return xx_params, yy_ratio

def build_noise_arrays(
        target, nreals=500, nskies=100,
        gw_only=False, red2white=None, red_gamma=None, 
        nloudest=10, bgl=1, save_temp=True,
        gsc_flag=False, dsc_flag=False, divide_flag=False, nexcl=0,
        var_hard_time=None, 
        figpath = '/Users/emigardiner/GWs/holodeck/output/anatomy_redz/figdata/noise',):

    data, params, dsdat = get_data(target,
        gw_only=gw_only, red2white=red2white, red_gamma=red_gamma,
        nloudest=nloudest, bgl=bgl, nexcl=nexcl,
        gsc_flag=gsc_flag, dsc_flag=dsc_flag, divide_flag=divide_flag,
        var_hard_time=var_hard_time)
    fobs_cents = data[0]['fobs_cents']
    nfreqs=len(fobs_cents)
    cad = 1.0/(2.0*fobs_cents[-1])

    sigmas = []
    hc_ss = []
    hc_bg = []
    dp_max = []
    dp_2nd = []
    count_cws_50 = [] # number of single sources with DP>0.5 in any realization
    count_cws_01 = [] # number of single sources with DP>0.5 in any realization
    for ii, dat in enumerate(data):
        sigmas.append(dsdat[ii]['sigmas']) # R,
        hc_ss.append(dat['hc_ss'])
        hc_bg.append(dat['hc_bg']) 

        dp_ssi = dsdat[ii]['gamma_ssi'] # F,R,S,L
        count_cws_01.append(np.sum(dp_ssi>0.01, axis=(0,3))) # from F,R,S,L to R,S
        count_cws_50.append(np.sum(dp_ssi>0.50, axis=(0,3)))

        dp_ssi = np.swapaxes(dp_ssi, 1,3).reshape(nfreqs*nloudest, nreals*nskies) # F*L, S*R
        argmax = np.argmax(dp_ssi, axis=0) # S*R
        reals = np.arange(nreals*nskies) # S*R
        _dp_max = dp_ssi[argmax, reals] # S*R

        dp_ssi[argmax, reals] = 0
        argmax = np.argmax(dp_ssi, axis=0) # S*R
        _dp_2nd = dp_ssi[argmax, reals] # S*R

        dp_max.append(_dp_max)
        dp_2nd.append(_dp_2nd)

    sigmas = np.array(sigmas) # V, R
    hc_ss = np.array(hc_ss) # (V,F,R,L)
    hc_bg = np.array(hc_bg) # (V,F,R)
    count_cws_50 = np.array(count_cws_50) # V,R,S
    count_cws_01 = np.array(count_cws_01) # V,R,S
    dp_max=np.array(dp_max) # V,R,S,
    dp_2nd=np.array(dp_2nd) # V,R,S,

    temp_name = '/Users/emigardiner/GWs/holodeck/output/temp'
    temp_name += f'/noise_arrays_{target}.npz'

    white_noise = _white_noise(cad, sigmas) # V,R, array
    if red2white is not None:
        red_amp = _red_amp_from_white_noise(cad, sigmas, red2white) # V,R, array
        red_noise = _red_noise(red_amp[np.newaxis,:,:], # (V,1,R,)
                               red_gamma, 
                               fobs_cents[np.newaxis,:,np.newaxis] # (1,F,1)
                               )
        np.savez(temp_name, white_noise=white_noise, red_noise=red_noise,
                 count_cws_50=count_cws_50, count_cws_01=count_cws_01,
                hc_ss=hc_ss, hc_bg=hc_bg)
        return white_noise, red_noise, count_cws_50, count_cws_01, hc_ss, hc_bg,  dp_max, dp_2nd
    np.savez(temp_name, white_noise=white_noise,
                count_cws_50=count_cws_50, count_cws_01=count_cws_01,
                hc_ss=hc_ss, hc_bg=hc_bg, dp_max=dp_max, dp_2nd=dp_2nd)
    return white_noise, count_cws_50, count_cws_01, hc_ss, hc_bg, dp_max, dp_2nd

def get_noise_arrays_temp(target, red=False):
    temp_name = '/Users/emigardiner/GWs/holodeck/output/temp'
    temp_name += f'/noise_arrays_{target}.npz'
    file = np.load(temp_name)
    white_noise = file['white_noise']
    count_cws_50 = file['count_cws_50']
    count_cws_01 = file['count_cws_01']
    hc_ss = file['hc_ss']
    hc_bg = file['hc_bg']
    dp_max = file['dp_max']
    dp_2nd = file['dp_2nd']
    if red:
        red_noise = file['red_noise']
        file.close()
        return white_noise, red_noise, count_cws_50, count_cws_01, hc_ss, hc_bg, dp_max, dp_2nd
    file.close()
    return white_noise, count_cws_50, count_cws_01, hc_ss, hc_bg, dp_max, dp_2nd



def build_anis_var_arrays(
        target, nvars=21, nreals=500, shape=None,
        gw_only=False, 
        nloudest=10,
        lmax=8, nside=8, 
        var_hard_time=None,
        figpath = '/Users/emigardiner/GWs/holodeck/output/anatomy_redz/figdata/anis_var',
      
):
    """ Calculate xx=params and yy=C1C0
    
    """
    data, params, = get_data(target, nvars=nvars, nreals=nreals, shape=shape,
        gw_only=gw_only, 
        nloudest=nloudest, dets=False,
        var_hard_time=var_hard_time)
    xx=[]
    yy=[]
    cl=[]
    for pp, par in enumerate(params):
        xx.append(params[pp][target])
        _, Cl = holo.anisotropy.sph_harm_from_hc(
            data[pp]['hc_ss'], data[pp]['hc_bg'], nside=nside, lmax=lmax
        )
        yy.append(Cl[...,1]/Cl[...,0])
        cl.append(Cl)

    filename = figpath+f'/anis_var_arrays_{target}'
    filename += f"_l{lmax}_ns{nside}"
    filename = append_filename(
        filename, 
        gw_only=gw_only, 
        nloudest=nloudest, bgl=nloudest,
        var_hard_time=var_hard_time)
    filename += '.npz'  

    np.savez(filename, xx_params=xx, yy_c1c0=yy, cl=cl)
    return xx, yy, cl

def get_anis_var_arrays(
        target, 
        gw_only=False, 
        nloudest=10, 
        lmax=8, nside=8, 
        var_hard_time=None,
        figpath = '/Users/emigardiner/GWs/holodeck/output/anatomy_redz/figdata/anis_var',
      
):
    """ Get xx=params and yy=C1C0
    
    """

    filename = figpath+f'/anis_var_arrays_{target}'
    filename += f"_l{lmax}_ns{nside}"
    filename = append_filename(
        filename, 
        gw_only=gw_only, 
        nloudest=nloudest, bgl=nloudest,
        var_hard_time=var_hard_time)
    filename += '.npz'  

    file = np.load(filename)
    xx_params = file['xx_params']
    yy_c1c0 = file['yy_c1c0']
    cl = file['cl']
    file.close()
    return xx_params, yy_c1c0, cl
    

def build_anis_freq_arrays(
        target, nvars=21, nreals=500, shape=None,
        gw_only=False, nloudest=10, 
        parvars = [0,10,20],
        lmax=8, nside=8,
        var_hard_time=None,

        ):

    if np.any(np.array(parvars)>nvars):
        parvars = np.arange(nvars)
        print(f'setting new parvars to {parvars}')
  
    data, params, = get_data(target, dets=False,
        nvars=nvars, nreals=nreals, shape=shape,  # keep as defaults
        gw_only=gw_only, nloudest=nloudest,
        var_hard_time=var_hard_time
        )


    yy_cl = [] # PV len array of (F,R,l=1) arrays
    params_cl = []
    xx_fobs = data[0]['fobs_cents']
    for var in parvars:
        _, Cl = holo.anisotropy.sph_harm_from_hc(
            data[var]['hc_ss'], data[var]['hc_bg'], nside=nside, lmax=lmax
        )
        yy_cl.append(Cl)
        params_cl.append(params[var])

    filename = f'/Users/emigardiner/GWs/holodeck/output/anatomy_redz/figdata/anis_freq/anis_freq_{target}'
    filename = append_filename(filename, nloudest=nloudest, bgl=nloudest,
                                gw_only=gw_only,
        var_hard_time=var_hard_time)
    filename += f"_pv{len(parvars)}"
    if nside != 8:
        filename += f"_ns{nside}"
    if lmax != 8:
        filename += f"_lmax{lmax}"
    filename += f".npz"
    np.savez(filename, yy_cl=yy_cl, xx_fobs=xx_fobs, params_cl=params_cl)

    return yy_cl, xx_fobs, params_cl

def get_anis_freq_arrays(
        target, nvars=21, nreals=500, nskies=100, shape=None,
        gw_only=False, 
        nloudest=10, 
        parvars = [0,10,20],
        nside=8, lmax=8, 
        var_hard_time=None,

        ):


    if np.any(np.array(parvars)>nvars):
        parvars = np.arange(nvars)

    filename = f'/Users/emigardiner/GWs/holodeck/output/anatomy_redz/figdata/anis_freq/anis_freq_{target}'
    filename = append_filename(filename, nloudest=nloudest, bgl=nloudest,
                               gw_only=gw_only,
        var_hard_time=var_hard_time)
    filename += f"_pv{len(parvars)}"
    if nside != 8:
        filename += f"_ns{nside}"
    if lmax != 8:
        filename += f"_lmax{lmax}"
    filename += f".npz"
    file = np.load(filename, allow_pickle=True)
    yy_cl=file['yy_cl']
    xx_fobs=file['xx_fobs']
    params_cl=file['params_cl']

    return yy_cl, xx_fobs, params_cl


def build_hcpar_arrays(
        target,nvars=21, nreals=500, shape=None,
        gw_only=False,
        nloudest=1, 
        parvars = [0,1,2,3,4], ss_zero=True,
        var_hard_time=None,
        ):
    """ Save and return hcpar arrays for plotting
    
    returns
    -----
    xx : 
    yy_ss : len(parvars) array of [F,R] NDarrays
        0th loudest [hc_ss, mass, distance] in dimensionless, M_sol, and Mpc units
    yy_bg : len(parvars) array of [F,R] NDarrays
        Background (all but nloudets) average [hc_ss, mass, distance] 
        in dimensionless, M_sol, and Mpc units
    labels : nparvars array of target values

    """

    if np.any(np.array(parvars)>nvars):
        parvars = np.arange(nvars)

    labels = []
    yy_ss = []
    yy_bg = []
    data, params, = get_data(target, dets=False,
        nvars=nvars, nreals=nreals, shape=shape,  # keep as defaults
        gw_only=gw_only, 
        nloudest=nloudest, 
        var_hard_time=var_hard_time)
    
    fobs_cents = data[0]['fobs_cents']
    xx = fobs_cents * YR

    for vv, var in enumerate(parvars):
        labels.append(f"{params[var][target]}")

        hc_ss = data[var]['hc_ss']
        hc_bg = data[var]['hc_bg']

        sspar = data[var]['sspar']
        bgpar = data[var]['bgpar']

        sspar = holo.single_sources.all_sspars(fobs_cents, sspar)
        bgpar = bgpar*holo.single_sources.par_units[:,np.newaxis,np.newaxis]
        sspar = sspar*holo.single_sources.par_units[:,np.newaxis,np.newaxis,np.newaxis]
        
        # parameters to plot
        if ss_zero:
            _yy_ss = [hc_ss[...,0], sspar[0,...,0], #sspar[1,...,0], # sspar[2,],  # strain, mass, mass ratio,
                    sspar[4,...,0]] # final comoving distance, single loudest only
        else:
            _yy_ss = [hc_ss[...], sspar[0,...], #sspar[1,...,0], # sspar[2,],  # strain, mass, mass ratio,
                    sspar[4,...]] # final comoving distance, single loudest only

        _yy_bg = [hc_bg, bgpar[0], #bgpar[1],  # strain, mass, mass ratio, initial redshift, final com distance
                bgpar[4],]
        yy_ss.append(_yy_ss)
        yy_bg.append(_yy_bg)

    filename = f'/Users/emigardiner/GWs/holodeck/output/anatomy_redz/figdata/hcpar/hcpar_{target}'
    filename = append_filename(filename, nloudest=nloudest, gw_only=gw_only, bgl=nloudest,
        var_hard_time=var_hard_time)
    filename += f"_pv{len(parvars)}"
    filename += f".npz"
    
    np.savez(filename, xx=xx, yy_ss=yy_ss, yy_bg=yy_bg, labels=labels)
    return np.array(xx), np.array(yy_ss), np.array(yy_bg), labels

def get_hcpar_arrays(
        target, 
        gw_only=False,
        nloudest=1, 
        parvars = [0,1,2,3,4],
        nvars=21,
        var_hard_time=None,
        ):
    """ Save and return hcpar arrays for plotting
    
    returns
    -------
    xx : [F,] 1Darray
    yy_ss : len(parvars) array of [F,R] NDarrays
        0th loudest [hc_ss, mass, distance] in dimensionless, M_sol, and Mpc units
    yy_bg : len(parvars) array of [F,R] NDarrays
        Background (all but nloudets) average [hc_ss, mass, distance] 
        in dimensionless, M_sol, and Mpc units
    labels : nparvars array of target values

    """


    if np.any(np.array(parvars)>nvars):
        parvars = np.arange(nvars)

    filename = f'/Users/emigardiner/GWs/holodeck/output/anatomy_redz/figdata/hcpar/hcpar_{target}'
    filename = append_filename(filename, nloudest=nloudest, gw_only=gw_only, bgl=nloudest,
        var_hard_time=var_hard_time)
    filename += f"_pv{len(parvars)}"
    filename += f".npz"

    file = np.load(filename, allow_pickle=True)
    xx = file['xx']
    yy_ss = file['yy_ss']
    yy_bg = file['yy_bg']
    labels = file['labels']

    return xx, yy_ss, yy_bg, labels

def build_favg_arrays(
        target, nreals=500, nskies=100,
        gw_only=False, red2white=None, red_gamma=None, 
        nloudest=10, bgl=10, cv=None,
        gsc_flag=False, dsc_flag=False, divide_flag=False, nexcl=0,
        var_hard_time=None,
        figpath = '/Users/emigardiner/GWs/holodeck/output/anatomy_redz/figdata/favg',):

    data, params, dsdat = get_data(target,
        gw_only=gw_only, red2white=red2white, red_gamma=red_gamma,
        nloudest=nloudest, bgl=bgl, nreals=nreals, nskies=nskies,
        gsc_flag=gsc_flag, dsc_flag=dsc_flag, divide_flag=divide_flag, nexcl=nexcl,
        var_hard_time=var_hard_time)

    xx = [] # param
    favg = [] # frequency means in log space
    stdv = [] # stdev in log space

    freqs = data[0]['fobs_cents']
    nfreqs = len(freqs)
    freqs = np.repeat(freqs, nreals*nskies*nloudest).reshape(
        nfreqs, nreals, nskies, nloudest)

    for pp, par in enumerate(params):
        xx.append(params[pp][target])
        dpssi = dsdat[pp]['gamma_ssi']
        logmean, logvar2 = weighted_mean_variance(np.log10(freqs), weights=dpssi)

        favg.append(logmean)
        stdv.append(np.sqrt(logvar2))

    filename = figpath+f'/favg_{target}'
    filename = append_filename(filename,
                gw_only=gw_only, red_gamma=red_gamma, red2white=red2white,
                nloudest=nloudest, bgl=bgl, cv=cv, 
                gsc_flag=gsc_flag, dsc_flag=dsc_flag, divide_flag=divide_flag, nexcl=nexcl,
        var_hard_time=var_hard_time)

    filename=filename+'.npz'
    np.savez(filename, xx = xx, yy_log = favg, sd_log=stdv)
    return xx, favg, stdv

def get_favg_arrays(
        target, 
        gw_only=False, red2white=None, red_gamma=None, 
        nloudest=10, bgl=10, cv=None,
        gsc_flag=False, dsc_flag=False, divide_flag=False, nexcl=0,
        var_hard_time=None,
        figpath = '/Users/emigardiner/GWs/holodeck/output/anatomy_redz/figdata/favg',):


    filename = figpath+f'/favg_{target}'
    filename = append_filename(filename,
                gw_only=gw_only, red_gamma=red_gamma, red2white=red2white,
                nloudest=nloudest, bgl=bgl, cv=cv, 
                gsc_flag=gsc_flag, dsc_flag=dsc_flag, divide_flag=divide_flag, nexcl=nexcl,
        var_hard_time=var_hard_time)

    filename=filename+'.npz'
    file = np.load(filename)
    xx = file['xx']
    yy_log = file['yy_log']
    sd_log = file['sd_log']
    file.close()

    return xx, yy_log, sd_log


def get_dp_arrays(
    target, nvars=21, nreals=500, nskies=100, shape=None, debug=False, 
    red=False, cv='midclbrt', gw_only=False, bgl=1,  
    ):

    path = '/Users/emigardiner/GWs/holodeck/output/anatomy_redz/figdata/dpboth'   
    filename = path+f'/dp_arrays_{target}_cv{cv}'
    if bgl != 10:
        filename += f"_bgl{bgl}"
    if gw_only:
        filename = filename+'_gw'
    filename = filename + '.npz'

    file = np.load(filename)
    if debug: print(f"{filename}\n{file.files}")
    xx = file['xx_params']
    yy_ss = file['yy_ss']
    ev_ss = file['ev_ss']
    yy_bg = file['yy_bg']
    file.close()
    return xx, yy_ss, ev_ss, yy_bg