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

GAMMA_RHO_GRID_PATH = '/Users/emigardiner/GWs/holodeck/output/rho_gamma_grids'
HC_REF15_10YR = 11.2*10**-15 


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

def _Sh_rest_noise(hc_ss, hc_bg, freqs):
    """ Calculate the noise spectral density contribution from all but the current single source.

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
    ss_noise : (F,R,L) NDarray of scalars
        The noise in a single pulsar from other GW sources for detecting each single source.

    Follows Eq. (45) in Rosado et al. 2015.
    TODO: modify this to allow for multiple loud sources.
    """
    hc2_louds = np.sum(hc_ss**2, axis=2) # (F,R) 
    # subtract the single source from rest of loud sources and the background, for each single source
    hc2_rest = hc_bg[:,:,np.newaxis]**2 + hc2_louds[:,:,np.newaxis] - hc_ss**2 # (F,R,L)
    Sh_rest = hc2_rest / freqs[:,np.newaxis,np.newaxis]**3 /(12 * np.pi**2) # (F,R,L)
    return Sh_rest

def _red_noise(A_red, gamma_red, freqs):
    """ Calculate the red noise for a given pulsar (or array of pulsars) 
    A_red * f sigma_i^gamma_red
    
    Parameters
    ----------
    A_red : scalar
        Amplitude of red noise.
    gamma_red : scalar
        Power-law index of red noise
    freqs : (F,) 1Darray of scalars
        Frequency bin centers.

    Returns
    -------
    P_red : (P,F) NDarray
        Red noise spectral density for the ith pulsar.

    """
    P_red = A_red * freqs**gamma_red
    return P_red
    


def _total_noise(delta_t, sigmas, hc_ss, hc_bg, freqs, A_red=None, gamma_red=None):
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
        
    Returns
    -------
    noise : (P,F,R,L) NDarray of scalars
        The total noise in each pulsar for detecting each single source

    Follows Eq. (44) in Rosado et al. 2015.
    """

    noise = _white_noise(delta_t, sigmas) # (P,)
    Sh_rest = _Sh_rest_noise(hc_ss, hc_bg, freqs) # (F,R,L,)
    noise = noise[:,np.newaxis,np.newaxis,np.newaxis] + Sh_rest[np.newaxis,:,:,:] # (P,F,R,L)
    if (A_red is not None) and (gamma_red is not None):
        red_noise = _red_noise(A_red, gamma_red) # (F,)
        noise = noise + red_noise[np.newaxis,:,np.newaxis,np.newaxis] # (P,F,R,L)
    return noise


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
    
    
    snr_ss = sam_cython.snr_ss(amp, F_iplus, F_icross, iotas, dur, Phi_0, S_i, freqs)
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
    # print('amp', amp.shape)

    a_pol, b_pol = _a_b_polarization(iotas) # (F,S,L)
    a_pol = a_pol[np.newaxis,:,np.newaxis,:,:] # (F,S,L) to (1,F,1,S,L)
    b_pol = b_pol[np.newaxis,:,np.newaxis,:,:] # (F,S,L) to (1,F,1,S,L)
    # print('a_pol', a_pol.shape)
    # print('b_pol', b_pol.shape)

    Phi_T = _gw_phase(dur, freqs, Phi_0) # (F,)
    # print('Phi_T', Phi_T.shape)
    Phi_T = Phi_T[np.newaxis,:,np.newaxis,:,:] # (F,S,L) to (1,F,1,S,L)
    # print('Phi_T', Phi_T.shape)

    Phi_0 = Phi_0[np.newaxis,:,np.newaxis,:,:] # (F,S,L) to (1,F,1,S,L)
    # print('Phi_0', Phi_0.shape)

    freqs = freqs[np.newaxis,:,np.newaxis,np.newaxis,np.newaxis] # (F,) to (1,F,1,1,1)
    # print('freqs', freqs.shape)

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
    #         gamma_flat = sam_cython.gamma_of_rho_interp(rho_flat, rsort, rho_interp_grid, gamma_interp_grid)
    #         gamma_ssi[ff,rr] = gamma_flat.reshape(rho[ff,rr].shape)

    for rr in range(len(rho[0])):
        # interpolate for gamma in cython
        rho_flat = rho[:,rr].flatten()
        rsort = np.argsort(rho_flat)
        gamma_flat = sam_cython.gamma_of_rho_interp(rho_flat, rsort, rho_interp_grid, gamma_interp_grid)
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

def detect_ss(thetas, phis, sigmas, cad, dur, fobs, dfobs, hc_ss, hc_bg, 
              theta_ss, phi_ss=None, Phi0_ss=None, iota_ss=None, psi_ss=None, 
              Amp_red=None, gamma_red=None, alpha_0=0.001, ret_snr=False,):
    """ Calculate the single source detection probability, and all intermediary steps.
    
    Parameters
    ----------
    thetas : (P,) 1Darray of scalars
        Polar (latitudinal) angular position of each pulsar in radians.
    phis : (P,) 1Darray of scalars
        Azimuthal (longitudinal) angular position of each pulsar in radians.
    sigmas : (P,) 1Darray of scalars
        Sigma_i of each pulsar in seconds.
    cad : scalar
        Cadence of observations in seconds.
    dur : scalar
        Duration of observations in seconds. 
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
    A_red : scalar or None
        Amplitude of pulsar red noise.
    gamma_red : scalar or None
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
    S_i = _total_noise(cad, sigmas, hc_ss, hc_bg, fobs, Amp_red, gamma_red)

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


def detect_ss_pta(pulsars, cad, dur, fobs, dfobs, hc_ss, hc_bg,
              theta_ss=None, phi_ss=None, Phi0_ss=None, iota_ss=None, psi_ss=None, 
              Fe_bar = None, Amp_red=None, gamma_red=None, alpha_0=0.001, Fe_bar_guess=15,
              ret_snr=False, print_nans=False, snr_cython=True, gamma_cython=True, grid_path=GAMMA_RHO_GRID_PATH):
    """ Calculate the single source detection probability, and all intermediary steps for
    R strain realizations and S sky realizations.
    
    Parameters
    ----------
    pulsars : (P,) list of hasasia.Pulsar objects
        A set of pulsars generated by hasasia.sim.sim_pta()
    cad : scalar
        Cadence of observations in seconds.
    dur : scalar
        Duration of observations in seconds. 
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
    Fe_bar : scalar or None
        Threshold F-statistic
    Amp_red : scalar or None
        Amplitude of pulsar red noise.
    gamma_red : scalar or None
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
    S_i = _total_noise(cad, sigmas, hc_ss, hc_bg, fobs, Amp_red, gamma_red)

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

def detect_lib(hdf_name, output_dir, npsrs, sigma, nskies, thresh=0.5,
                   dur=None, cad=None, dfobs=None, plot=True, debug=False,
                   grid_path=GAMMA_RHO_GRID_PATH, snr_cython = True):
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
        Fraction of realizations with a single source detection.
    df_bg : (N,) 1Darray
        Fraction of realizations with a background detection.
    
    TODO: Speed it up by doing the gamma_ssi integration in cython.

    """

    # Read in hdf file
    ssfile = h5py.File(hdf_name, 'r')
    fobs = ssfile['fobs'][:]
    if dfobs is None: dfobs = ssfile['dfobs'][:]
    if dur is None: dur = ssfile['pta_dur'][0]
    if cad is None: cad = ssfile['pta_cad'][0]
    hc_ss = ssfile['hc_ss'][...]
    hc_bg = ssfile['hc_bg'][...]
    shape = hc_ss.shape
    nsamp, nfreqs, nreals, nloudest = shape[0], shape[1], shape[2], shape[3]

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
    psrs = hsim.sim_pta(timespan=dur/YR, cad=1/(cad/YR), sigma=sigma,
                    phi=phis, theta=thetas)

     # Build ss skies
    if debug: print('Building ss skies.')
    theta_ss, phi_ss, Phi0_ss, iota_ss, psi_ss = _build_skies(nfreqs, nskies, nloudest)

    # Calculate DPs, SNRs, and DFs
    if debug: print('Calculating SS and BG detection statistics.')
    dp_ss = np.zeros((nsamp, nreals, nskies)) # (N,R,S)
    dp_bg = np.zeros((nsamp, nreals)) # (N,R)
    snr_ss = np.zeros((nsamp, nfreqs, nreals, nskies, nloudest))
    snr_bg = np.zeros((nsamp, nfreqs, nreals))
    df_ss = np.zeros(nsamp)
    df_bg = np.zeros(nsamp)
    gamma_ssi = np.zeros((nsamp, nfreqs, nreals, nskies, nloudest))

    # # one time calculations
    # Num = nfreqs * nloudest # number of single sources in a single strain realization (F*L)
    # Fe_bar = _Fe_thresh(Num) # scalar

    for nn in range(nsamp):
        if debug: print('on sample nn=%d out of N=%d' % (nn,nsamp))
        dp_bg[nn,:], snr_bg[nn,...] = detect_bg_pta(psrs, fobs, cad, hc_bg[nn], ret_snr=True)
        vals_ss = detect_ss_pta(psrs, cad, dur, fobs, dfobs,
                                                hc_ss[nn], hc_bg[nn], ret_snr=True, 
                                                gamma_cython=True, snr_cython=snr_cython,
                                                theta_ss=theta_ss, phi_ss=phi_ss, Phi0_ss=Phi0_ss,
                                                iota_ss=iota_ss, psi_ss=psi_ss, grid_path=grid_path)
        dp_ss[nn,:,:], snr_ss[nn,...], gamma_ssi[nn] = vals_ss[0], vals_ss[1], vals_ss[2]
        df_ss[nn] = np.sum(dp_ss[nn]>thresh)/(nreals*nskies)
        df_bg[nn] = np.sum(dp_bg[nn]>thresh)/(nreals)

        if plot:
            fig = plot_sample_nn(fobs, hc_ss[nn], hc_bg[nn],
                         dp_ss[nn], dp_bg[nn], 
                         df_ss[nn], df_bg[nn], nn=nn)  
            plot_fname = (output_dir+'/p%06d_detprob.png' % nn) # need to make this directory
            fig.savefig(plot_fname, dpi=100)
            plt.close(fig)
    
    if debug: print('Saving npz files and allsamp plots.')
    fig1 = plot_detprob(dp_ss, dp_bg, nsamp)
    fig2 = plot_detfrac(df_ss, df_bg, nsamp, thresh)
    fig1.savefig(output_dir+'/allsamp_detprobs.png', dpi=300)
    fig2.savefig(output_dir+'/allsamp_detfracs.png', dpi=300)
    plt.close(fig1)
    plt.close(fig2)
    np.savez(output_dir+'/detstats.npz', dp_ss=dp_ss, dp_bg=dp_bg, 
             df_ss=df_ss, df_bg=df_bg, snr_ss=snr_ss, snr_bg=snr_bg, gamma_ssi=gamma_ssi)

    return dp_ss, dp_bg, df_ss, df_bg, snr_ss, snr_bg
        

def _build_skies(nfreqs, nskies, nloudest):
    theta_ss = np.random.uniform(0, np.pi, size = nfreqs * nskies * nloudest).reshape(nfreqs, nskies, nloudest)
    phi_ss = np.random.uniform(0, 2*np.pi, size = theta_ss.size).reshape(theta_ss.shape)
    Phi0_ss = np.random.uniform(0,2*np.pi, size = theta_ss.size).reshape(theta_ss.shape)
    iota_ss = np.random.uniform(0,  np.pi, size = theta_ss.size).reshape(theta_ss.shape)
    psi_ss = np.random.uniform(0,   np.pi, size = theta_ss.size).reshape(theta_ss.shape)
    return theta_ss, phi_ss, Phi0_ss, iota_ss, psi_ss


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

def plot_detprob(dp_ss_all, dp_bg_all, nsamp):
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
    ax.errorbar(np.arange(nsamp), np.mean(dp_bg_all, axis=1), 
                yerr = np.std(dp_bg_all, axis=1), linestyle='', 
                marker='d', capsize=5, color='cornflowerblue', alpha=0.5,
                label = r'$\langle \gamma_\mathrm{BG} \rangle$')
    ax.errorbar(np.arange(nsamp), np.mean(dp_ss_all, axis=(1,2)),
                yerr = np.std(dp_ss_all, axis=(1,2)), linestyle='', 
                marker='o', capsize=5, color='orangered', alpha=0.5,
                label = r'$\langle \gamma_\mathrm{SS} \rangle$')
    ax.set_yscale('log')
    ax.set_title('Average DP across Realizations')

    ax.legend()
    fig.tight_layout()

    return fig


def plot_detfrac(df_ss, df_bg, nsamp, thresh):
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
    ax.plot(np.arange(nsamp), df_bg, color='cornflowerblue', label='BG',
            marker='d', alpha=0.5)
    ax.plot(np.arange(nsamp), df_ss, color='orangered', label='SS',
            marker='o', alpha=0.5)
    ax.set_xlabel('Param Space Sample')
    ax.set_ylabel('Detection Fraction')
    ax.set_title('Fraction of Realizations with DP > %0.2f' % thresh)
    ax.legend()
    fig.tight_layout()
    return fig


############################# Rank Samples ############################# 


def amp_to_hc(amp_ref, fobs, dfobs):
    """ Calculate characteristic strain from strain amplitude.
    
    """
    hc = amp_ref*np.sqrt(fobs/dfobs)
    return hc

def rank_samples(hc_ss, hc_bg, fobs, fidx=None, dfobs=None, amp_ref=None, hc_ref=None, ret_all = False):
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

    if (hc_ref is None):
        # find reference (e.g. 12.5 yr) char strain
        hc_ref = amp_to_hc(amp_ref, fobs[fidx], dfobs[fidx])
        

    # extrapolate hc_ref at freq closest to 1/10yr from 1/10yr ref
    hc_ref = hc_ref * (fobs[fidx]*YR/.1)**(-2/3)

    # select 1/yr median strains of samples
    hc_tt = np.sqrt(hc_bg[:,fidx,:]**2 + np.sum(hc_ss[:,fidx,:,:]**2, axis=-1)) # (N,R)
    hc_diff = np.abs(hc_tt - hc_ref) # (N,R)
    print('hc_diff', hc_diff.shape)
    hc_diff = np.median(hc_diff, axis=-1) # median of differences (N,)
    print('hc_diff', hc_diff.shape)

    # sort by closest
    nsort = np.argsort(hc_diff)

    if ret_all:
        return nsort, fidx, hc_ref
    return nsort

############################ Calibrate PTA ############################# 

