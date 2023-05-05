"""Detection Statistics module.

This module calculates detection statistics for single source and background strains
for Hasasia PTA's.

"""

import numpy as np
from scipy import special, integrate
from sympy import nsolve, Symbol

import holodeck as holo
from holodeck import utils, cosmo, log
from holodeck.constants import MPC, YR

import hasasia.sensitivity as hsen
import hasasia.sim as hsim
import hasasia as has



###################### Overlap Reduction Function ######################

def _gammaij_from_thetaij(theta_ij):
    """ Calcualte gamma_ij for two pulsars of relative angle theta_ij.
    
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

def _power_spectral_density(hc_bg, freqs):
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

    S_h = hc_bg**2 / (12 * np.pi**2 * freqs[:,np.newaxis]**3)
    return S_h


######################## mu_1, sigma_0, sigma_1 ########################

def _sigma0_Bstatistic(noise, Gamma, Sh0_bg):
    """ Calculate sigma_1 for the background, by summing over all pulsars and frequencies.
    Assuming the B statistic, which maximizes S/N_B = mu_1/sigma_1
    
    Parameters
    ----------
    noise : (P,) 1darray of scalars
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
    # P_i in shape (P,1,1,1)
    # P_j in shape (1,P,1,1)

    numer = (Gamma[:,:,np.newaxis,np.newaxis]**2 * Sh0_bg[np.newaxis,np.newaxis,:]**2 
             * noise[:,np.newaxis,np.newaxis,np.newaxis] * noise[np.newaxis,:,np.newaxis,np.newaxis])
    denom = ((noise[:,np.newaxis,np.newaxis,np.newaxis] + Sh0_bg[np.newaxis, np.newaxis,:])
              * (noise[np.newaxis,:,np.newaxis,np.newaxis] + Sh0_bg[np.newaxis,np.newaxis,:])
             + Gamma[:,:,np.newaxis,np.newaxis]**2 * Sh0_bg[np.newaxis,np.newaxis,:]**2)**2
    
    sum = np.sum(numer/denom, axis=(0,1,2))
    sigma_0B = np.sqrt(2*sum)
    return sigma_0B

def _sigma1_Bstatistic(noise, Gamma, Sh_bg, Sh0_bg):
    """ Calculate sigma_1 for the background, by summing over all pulsars and frequencies.
    Assuming the B statistic, which maximizes S/N_B = mu_1/sigma_1
    
    Parameters
    ----------
    noise : (P,) 1darray of scalars
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

    numer = (Gamma[:,:,np.newaxis,np.newaxis]**2 * Sh0_bg[np.newaxis,np.newaxis,:]**2 
             * ((noise[:,np.newaxis,np.newaxis,np.newaxis] + Sh_bg[np.newaxis,np.newaxis,:])
                * (noise[np.newaxis,:,np.newaxis,np.newaxis] + Sh_bg[np.newaxis,np.newaxis,:])
                + Gamma[:,:,np.newaxis,np.newaxis]**2 * Sh_bg[np.newaxis,np.newaxis,:]**2))
             
    denom = ((noise[:,np.newaxis,np.newaxis,np.newaxis] + Sh0_bg[np.newaxis, np.newaxis,:])
              * (noise[np.newaxis,:,np.newaxis,np.newaxis] + Sh0_bg[np.newaxis,np.newaxis,:])
             + Gamma[:,:,np.newaxis,np.newaxis]**2 * Sh0_bg[np.newaxis,np.newaxis,:]**2)**2
    
    sum = np.sum(numer/denom, axis=(0,1,2))
    sigma_1B = np.sqrt(2*sum)
    return sigma_1B

def _mean1_Bstatistic(noise, Gamma, Sh_bg, Sh0_bg):
    """ Calculate mu_1 for the background, by summing over all pulsars and frequencies.
    Assuming the B statistic, which maximizes S/N_B = mu_1/sigma_1
    
    Parameters
    ----------
    noise : (P,) 1darray of scalars
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
    for ii in range(len(noise)):
        for jj in range(ii+1): 
            assert Gamma[ii,jj] == 0, f'Gamma[{ii},{jj}] = {Gamma[ii,jj]}, but it should be 0!'

    # to get sum term in shape (P,P,F,R) for ii,jj,kk we want:
    # Gamma in shape (P,P,1,1)
    # Sh0 and Sh in shape (1,1,F,R)
    # P_i in shape (P,1,1,1)
    # P_j in shape (1,P,1,1)

    numer = (Gamma[:,:,np.newaxis,np.newaxis] **2 
            * Sh_bg[np.newaxis,np.newaxis,:]
            * Sh0_bg[np.newaxis,np.newaxis,:])
    denom = ((noise[:,np.newaxis,np.newaxis,np.newaxis] + Sh0_bg[np.newaxis,np.newaxis,:])
               * (noise[np.newaxis,:,np.newaxis,np.newaxis] + Sh0_bg[np.newaxis,np.newaxis,:])
               + Gamma[:,:,np.newaxis,np.newaxis]**2 * Sh0_bg[np.newaxis, np.newaxis, :]**2)
    
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


def detect_bg(thetas, phis, sigmas, fobs, cad, hc_bg, alpha_0=0.001, ret_all = False):
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
    ret_all : Bool
        Whether to return all parameters or just dp_bg

    Returns
    -------
    dp_bg : (R,) 1Darray
        Background detection probability, for R realizations.
    Gamma : (P, P) 2D Array
        Overlap reduction function for j>i, 0 otherwise.
        Only returned if return_all = True.
    Sh_bg : (F,R) 1Darray
        Spectral density, for R realizations
        Only returned if return_all = True.
    noise : (P,) 1Darray
        Spectral noise density of each pulsar.
        Only returned if return_all = True.
    mu_1B : (R,) 1Darray
        Expected value for the B statistic.
        Only returned if return_all = True.
    sigma_0B : (R,) 1Darray
    sigma_1B : (R,) 1Darray

    """
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
    Sh0_bg = Sh_bg # approximation used in Rosado et al. 2015

    # Noise 
    noise = _white_noise(cad, sigmas) 

    sigma_0B = _sigma0_Bstatistic(noise, Gamma, Sh0_bg)

    sigma_1B = _sigma1_Bstatistic(noise, Gamma, Sh_bg, Sh0_bg)

    mu_1B = _mean1_Bstatistic(noise, Gamma, Sh_bg, Sh0_bg)

    dp_bg = _bg_detection_probability(sigma_0B, sigma_1B, mu_1B, alpha_0)

    if(ret_all):
        return dp_bg, Gamma, Sh_bg, noise, mu_1B, sigma_0B, sigma_1B
    else:
        return dp_bg
    




def detect_bg_pta(pulsars, spectra, cad, hc_bg, alpha_0=0.001, ret_all = False):
    """ Calculate the background detection probability, and all the intermediary steps
    from a list of hasasia.Pulsar objects.
    
    Parameters
    ----------
    pulsars : (P,) list of hasasia.Pulsar objects
        A set of pulsars generated by hasasia.sim.sim_pta()
    spectra : (P,) list of hasasia.Spectrum objects
        The spectrum for each pulsar.
    cad : scalar
        Cadence of observations in seconds.
    hc_bg : (F,R)
        Characteristic strain of the background at each frequency,
        for R realizations.
    alpha_0 : scalar
        Falsa alarm probability
    return_all : Bool
        Whether or not to return intermediate variables.

    Returns
    -------
    dp_bg : scalar
        Background detection probability
    Gamma : (P, P) 2D Array
        Overlap reduction function for j>i, 0 otherwise.
        Only returned if return_all = True.
    Sh_bg : (F,) 1Darray
        Spectral density
        Only returned if return_all = True.
    noise : (P,) 1Darray
        Spectral noise density of each pulsar.
        Only returned if return_all = True.
    mu_1B : scalar
        Expected value for the B statistic.
        Only returned if return_all = True.
    sigma_0B : scalar
    sigma_1B : scalar


    If a pulsar had differing toaerrs, the mean of that pulsar's 
    toaerrs is used as the pulsar's sigma.
    TODO: implement red noise
    """

    # check inputs
    assert len(pulsars) == len(spectra), f"'pulsars ({len(pulsars)}) does not match 'spectra' ({len(spectra)}) !"
    # get pulsar properties
    thetas = np.zeros(len(pulsars))
    phis = np.zeros(len(pulsars))
    sigmas = np.zeros(len(pulsars))
    for ii in range(len(pulsars)):
        thetas[ii] = pulsars[ii].theta
        phis[ii] = pulsars[ii].phi
        sigmas[ii] = np.mean(pulsars[ii].toaerrs)

    fobs = spectra[0].freqs

    Gamma = _orf_pta(pulsars)

    Sh_bg = _power_spectral_density(hc_bg[:], fobs)
    Sh0_bg = Sh_bg # note this refers to same object, not a copy

    noise = _white_noise(cad, sigmas)

    mu_1B = _mean1_Bstatistic(noise, Gamma, Sh_bg, Sh0_bg)

    sigma_0B = _sigma0_Bstatistic(noise, Gamma, Sh0_bg)

    sigma_1B = _sigma1_Bstatistic(noise, Gamma, Sh_bg, Sh0_bg)

    dp_bg = _bg_detection_probability(sigma_0B, sigma_1B, mu_1B, alpha_0)

    if(ret_all):
        return dp_bg, Gamma, Sh_bg, noise, mu_1B, sigma_0B, sigma_1B
    else:
        return dp_bg
    


######################## Signal-to-Noise Ratio ########################

def SNR_bg_B(noise, Gamma, Sh_bg):
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
    SNR_B : (R,) 1Darray of scalars
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
    SNR_B = np.sqrt(2*sum)
    return SNR_B

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

def SNR_hasasia_noise_bg(scGWB):
    """ Calculate the effective noise signal to noise ratio with hasasia.
    
    Parameters
    ----------
    scGWB : hasasia.sensitivity.GWBSensitivityCurve object
        GWB sensitivity curve object.
        
    Returns
    -------
    SNR_h : scalar
        Signal to noise ratio from hasasia.

    This function may not be working as we expect, since it does not produce SNR
    of noise to be 1.
    """
    Sh_h = _Sh_hasasia_noise_bg(scGWB)
    SNR_h = scGWB.SNR(Sh_h)
    return SNR_h
    

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

def SNR_hasasia_modeled_bg(scGWB, hc_bg):
    """ Calculate the GWB signal to noise ratio with hasasia.
    
    Parameters
    ----------
    scGWB : hasasia.sensitivity.GWBSensitivityCurve object
        GWB sensitivity curve object.
    hc_bg : (F,R) NDarray
        Realistic characteristic strain of the background.
        
    Returns
    -------
    SNR_h : (R,) 1Darray)
        Signal to noise ratio from hasasia, for each realization.
    """
    Sh_h = _Sh_hasasia_modeled_bg(scGWB.freqs, hc_bg)
    SNR_h = np.zeros(len(hc_bg[0]))
    for rr in range(len(hc_bg[0])):
        SNR_h[rr] = scGWB.SNR(Sh_h[:,rr])
    return SNR_h
    