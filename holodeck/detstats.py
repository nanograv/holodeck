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

def _Sh_hasasia_generic_bg(scGWB):
    """ Calculate the signal strain power spectral density, 
        `Sh` for hasasia's SNR calculation
        
    Parameters
    ----------
    scGWB : hasasia.sensitivity.GWBSensitivityCurve object
        GWB sensitivity curve object.
        
    Returns
    -------
    Sh_h : (F,) 1Darray
        Sh as used in hasasia's SNR calculation, for each frequency.
    
    """
    freqs = scGWB.freqs
    H0 = scGWB._H_0.to('Hz').value 
    Omega_gw = scGWB.Omega_gw
    Sh_h = 3*H0**2 / (2*np.pi**2) * Omega_gw / freqs**3
    return Sh_h

def SNR_hasasia_generic_bg(scGWB):
    """ Calculate the GWB signal to noise ratio with hasasia.
    
    Parameters
    ----------
    scGWB : hasasia.sensitivity.GWBSensitivityCurve object
        GWB sensitivity curve object.
        
    Returns
    -------
    SNR_h : scalar
        Signal to noise ratio from hasasia.
    """
    Sh_h = _Sh_hasasia_generic_bg(scGWB)
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
    """ Calculate the unitary vector m-hat for the antenna pattern functions.
    
    Parameters
    ----------
    theta : (F,R,L) NDarray
        Spherical coordinate position of each single source.
    phi : (F,R,L) NDarray
        Spherical coordinate position of each single source.
    xi : (F,R,L) NDarray
        Inclination of binary? But thought that's what iota was?    
    
    Returns
    -------
    m_hat : (3,F,R,L) NDarray 
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
    """ Calculate the unitary vector n-hat for the antenna pattern functions.
    
    Paramters
    ---------
    theta : (F,R,L) NDarray
        Spherical coordinate position of each single source.
    phi : (F,R,L) NDarray
        Spherical coordinate position of each single source.
    xi : (F,R,L) 1Darray
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
    """ Calculate the unitary vector n-hat for the antenna pattern functions.
    
    Paramters
    ---------
    theta : (F,R,L) NDarray
        Spherical coordinate position of each single source.
    phi : (F,R,L) NDarray
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
    m_hat : (3,F,R,L) NDarray
        Single source m_hat unitary vector for each frequency and realization.
    n_hat : (3,F,R,L) NDarray
        Single source mnhat unitary vector for each frequency and realization.
    Omega_hat : (3,F,R,L) NDarray
        Single source Omega_hat unitary vector for each frequency and realization.
    pi_hat : (3,P) NDarray
        Pulsar term unitary vector for the ith pulsar.
        
    Returns
    -------
    F_iplus : (P,F,R,L) NDarray
        Plus antenna pattern function for each pulsar and binary of each realization.
    F_icross : (P,F,R,L) NDarray
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
    """

    Phi_T = 2 * np.pi * freqs[:,np.newaxis,np.newaxis] * dur + Phi_0
    return Phi_T

def _amplitude(hc_ss, f, df):
    """ Calculate the amplitude from the single source to use in DP calculations
    
    Parameters
    ----------
    hc_ss : (F,R,L) NDarray
        Characteristic strain of each single source at each realization.
    f : (F,) 1Darray
        Frequency
    df : (F,) 1Darray
        Frequency bin widths.

    Returns
    -------
    Amp : (F,R,L)
        Dimensionless amplitude, A, of each single source at each frequency and realization.
    
    """

    Amp = hc_ss * np.sqrt(5) / 4 / 2**(1/6) *np.sqrt(df[:,np.newaxis,np.newaxis]/f[:,np.newaxis,np.newaxis])
    return Amp


####################### SS Signal to Noise Ratio  #######################

def _SNR_ss(amp, F_iplus, F_icross, iotas, dur, Phi_0, S_i, freqs):
    """ Calculate the SNR for each pulsar wrt each single source detection.

    Paramters
    ---------
    amp : (F,R,L) NDarray 
        Dimensionless strain amplitude for loudest source at each frequency.
    F_iplus : (P,F,R,L) NDarray
        Antenna pattern function for each pulsar.
    F_icross : (P,F,R,L) NDarray
        Antenna pattern function for each pulsar.
    iotas : (F,R,L) NDarray
        Is this inclination? or what?
        Gives the wave polarizations a and b.
    dur : scalar
        Duration of observations.
    Phi_0 : (F,R,L) NDarray
        Initial GW Phase.
    S_i : (P,F,R,L) NDarray
        Total noise of each pulsar wrt detection of each single source, in s^3
    freqs : (F,) 1Darray 

    Returns
    -------
    SNR_ss : (F,R,L) NDarray
        SNR from the whole PTA for each single source.

    """
    
    amp = amp[np.newaxis,:,:,:]  # (F,R,L) to (P,F,R,L)
    # print('amp', amp.shape)

    a_pol, b_pol = _a_b_polarization(iotas)
    a_pol = a_pol[np.newaxis,:,:,:] # (F,R,L) to (P,F,R,L)
    b_pol = b_pol[np.newaxis,:,:,:] # (F,R,L) to (P,F,R,L)
    # print('a_pol', a_pol.shape)
    # print('b_pol', b_pol.shape)

    Phi_T = _gw_phase(dur, freqs, Phi_0) # (F,)
    # print('Phi_T', Phi_T.shape)
    Phi_T = Phi_T[np.newaxis,:] # (F,R,L) to (P,F,R,L)
    # print('Phi_T', Phi_T.shape)

    Phi_0 = Phi_0[np.newaxis,:,:,:] # (P,F,R,L)
    # print('Phi_0', Phi_0.shape)

    freqs = freqs[np.newaxis,:,np.newaxis,np.newaxis] # (F,) to (P,F,R,L)
    # print('freqs', freqs.shape)

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
    
    SNR2_pulsar_ss = coef*(term1 + term2 + term3) # (P,F,R,L)

    SNR_ss = np.sqrt(np.sum(SNR2_pulsar_ss, axis=0)) # (F,R,L), sum over the pulsars
    return SNR_ss


######################### Detection Probability #########################

def _Fe_thresh(Num, alpha_0=0.001):
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
    Fe_bar = nsolve(func, Fe_bar, 10)
    return(Fe_bar)

def _integrand_gamma_ss_i(Fe, rho):

    I_1 = special.i1(rho*np.sqrt(2*Fe))
    rv = (2*Fe)**(1/2) /rho * I_1 * np.exp(-Fe - rho**2 /2)
    return rv

def _gamma_ssi(Fe_bar, rho):
    """ Calculate the detection probability for each single source in each realization.
    
    Parameters
    ----------
    rho : (F,R,L) NDarray
        Given by the total PTA signal to noise ratio, S/N_S, for each single source
    Fe_bar : scalar
        The threshold F_e statistic

    Returns
    -------
    gamma_ssi : (F,R,L) NDarray
        The detection probability for each single source, i, at each frequency and realization.

    TODO: Find a way to do this without the embedded for-loops.
    """
    gamma_ssi = np.zeros((rho.shape))
    for ff in range(len(rho)):
        for rr in range(len(rho[0])):
            for ll in range(len(rho[0,0])):
                gamma_ssi[ff,rr,ll] = integrate.quad(_integrand_gamma_ss_i, Fe_bar, np.inf, args=(rho[ff,rr,ll]))[0]
                if(np.isnan(gamma_ssi[ff,rr,ll])):
                    print(f'gamma_ssi[{ff},{rr},{ll}] is nan, setting to 0.')
                    gamma_ssi[ff,rr,ll] = 0
   

    return gamma_ssi


def _gamma_ssi(Fe_bar, rho):
    """ Calculate the detection probability for each single source in each realization.
    
    Parameters
    ----------
    rho : (F,R,L) NDarray
        Given by the total PTA signal to noise ratio, S/N_S, for each single source
    Fe_bar : scalar
        The threshold F_e statistic

    Returns
    -------
    gamma_ssi : (F,R,L) NDarray
        The detection probability for each single source, i, at each frequency and realization.

    TODO: Find a way to do this without the embedded for loops!
    """
    gamma_ssi = np.zeros((rho.shape))
    for ff in range(len(rho)):
        for rr in range(len(rho[0])):
            for ll in range(len(rho[0,0])):
                gamma_ssi[ff,rr,ll] = integrate.quad(_integrand_gamma_ss_i, Fe_bar, np.inf, args=(rho[ff,rr,ll]))[0]
                if(np.isnan(gamma_ssi[ff,rr,ll])):
                    print(f'gamma_ssi[{ff},{rr},{ll}] is nan, setting to 0.')
                    gamma_ssi[ff,rr,ll] = 0
   

    return gamma_ssi

def _ss_detection_probability(gamma_ss_i):
    """ Calculate the probability of detecting any single source, given individual single 
    source detection probabilities.
    
    
    Parameters
    ----------
    gamma_ss_i : (F,R,L) NDarray
        Detection probability of each single source, at each frequency and realization.

    Returns
    -------
    gamma_ss : (R) 1Darray
        Detection probability of any single source, for each realization
    """
    
    gamma_ss = 1 - np.product(1-gamma_ss_i, axis=(0,2))
    return gamma_ss


######################## Detection Probability #########################

def detect_ss(thetas, phis, sigmas, cad, dur, 
              fobs, dfobs, hc_ss, hc_bg, alpha_0=0.001, ret_SNR=False,
              theta_ss=None, phi_ss=None, iota_ss=None, xi_ss=None, Phi0_ss=None,
              Amp_red=None, gamma_red=None):
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
        Frequency bin centers in hertz.
    dfobs : (F-1,) 1Darray of scalars
        Frequency bin widths in hertz.
    hc_ss : (F,R,L) NDarray of scalars
        Characteristic strain of the L loudest single sources at 
        each frequency, for R realizations.
    hc_bg : (F,R)
        Characteristic strain of the background at each frequency, 
        for R realizations.
    alpha_0 : scalar
        False alarm probability
    ret_SNR : Bool
        Whether or not to also return SNR_ss.
    theta_ss : (F,R,L) NDarray or None
        Polar (latitudinal) angular position in the sky of each single source.
        If None, random values between 0 and pi will be assigned.
    phi_ss : (F,R,L) NDarray or None
        Azimuthal (longitudinal) angular position in the sky of each single source.
        If None, random values between 0 and 2pi will be assigned.
    iota_ss : (F,R,L) NDarray or None
        Inclination of each single source with respect to the line of sight.
        If None, random values between 0 and pi will be assigned.
    xi_ss : (F,R,L) NDarray or None
        Polarization of each single source.
        If None, assigned same random values as iota_ss, because I think these
        might both be referring to the same thing.
    Phi0_ss : (F,R,L) NDarray or None
        Initial GW phase.
        If None, random values between 0 and 2pi will be assigned.
    A_red : scalar or None
        Amplitude of pulsar red noise.
    gamma_red : scalar or None
        Power law index of pulsar red noise.

    Returns
    -------
    gamma_ss : (R,) 1Darray
        Probability of detecting any single source, for each realization.
    SNR_ss : (F,R,L) NDarray
        SNR of each single source.

    """

    # Assign random single source sky positions, if not provided.
    if theta_ss is None:
        theta_ss = np.random.uniform(0, np.pi, size = hc_ss.size).reshape(hc_ss.shape)
    if phi_ss is None:
        phi_ss = np.random.uniform(0, 2*np.pi, size = hc_ss.size).reshape(hc_ss.shape)
    if iota_ss is None:
        iota_ss = np.random.uniform(0, np.pi, size = hc_ss.size).reshape(hc_ss.shape)
    if xi_ss is None:
        xi_ss = np.random.uniform(0, np.pi, size = hc_ss.size).reshape(hc_ss.shape)
    if Phi0_ss is None:
        Phi0_ss = np.random.uniform(0,2*np.pi, size=hc_ss).reshape(hc_ss.shape)

    # unitary vectors
    m_hat = _m_unitary_vector(theta_ss, phi_ss, xi_ss) # (3,F,R,L)
    n_hat = _n_unitary_vector(theta_ss, phi_ss, xi_ss) # (3,F,R,L)
    Omega_hat = _Omega_unitary_vector(theta_ss, phi_ss) # (3,F,R,L)
    pi_hat = _pi_unitary_vector(phis, thetas) # (3,P)

    # antenna pattern functions
    F_iplus, F_icross = _antenna_pattern_functions(m_hat, n_hat, Omega_hat, 
                                                   pi_hat) # (P,F,R,L)
    
    # noise spectral density
    S_i = _total_noise(cad, sigmas, hc_ss, hc_bg, fobs, Amp_red, gamma_red)

    # amplitudw
    amp = _amplitude(hc_ss, fobs, dfobs) # (F,R,L)

    # SNR (includes a_pol, b_pol, and Phi_T calculations internally)
    SNR_ss = _SNR_ss(amp, F_iplus, F_icross, iota_ss, dur, Phi0_ss, S_i, fobs) # (F,R,L)
    
    Num = hc_ss[:,0,:].size
    Fe_bar = _Fe_thresh(Num, alpha_0=alpha_0) # scalar

    gamma_ssi = _gamma_ssi(Fe_bar, rho=SNR_ss) # (F,R,L)
    gamma_ss = _ss_detection_probability(gamma_ssi) # (R,)

    if ret_SNR:
        return gamma_ss, SNR_ss
    else:
        return gamma_ss




def detect_ss_pta(pulsars, cad, dur, fobs,
              dfobs, hc_ss, hc_bg, alpha_0=0.001, ret_SNR=False,
              theta_ss=None, phi_ss=None, iota_ss=None, xi_ss=None, Phi0_ss=None,
              Amp_red=None, gamma_red=None):
    """ Calculate the single source detection probability, and all intermediary steps.
    
    Parameters
    ----------
    pulsars : (P,) list of hasasia.Pulsar objects
        A set of pulsars generated by hasasia.sim.sim_pta()
    cad : scalar
        Cadence of observations in seconds.
    dur : scalar
        Duration of observations in seconds. 
    fobs : (F,) 1Darray of scalars
        Frequency bin centers in Hz.
    dfobs : (F-1,) 1Darray of scalars
        Frequency bin widths in Hz.
    hc_ss : (F,R,L) NDarray of scalars
        Characteristic strain of the L loudest single sources at 
        each frequency, for R realizations.
    hc_bg : (F,R)
        Characteristic strain of the background at each frequency, 
        for R realizations.
    alpha_0 : scalar
        False alarm probability
    ret_SNR : Bool
        Whether or not to also return SNR_ss.
    theta_ss : (F,R,L) NDarray or None
        Polar (latitudinal) angular position in the sky of each single source.
        If None, random values between 0 and pi will be assigned.
    phi_ss : (F,R,L) NDarray or None
        Azimuthal (longitudinal) angular position in the sky of each single source.
        If None, random values between 0 and 2pi will be assigned.
    iota_ss : (F,R,L) NDarray or None
        Inclination of each single source with respect to the line of sight.
        If None, random values between 0 and pi will be assigned.
    xi_ss : (F,R,L) NDarray or None
        Polarization of each single source.
        If None, assigned same random values as iota_ss, because I think these
        might both be referring to the same thing.
    Phi0_ss : (F,R,L) NDarray or None
        Initial GW phase.
        If None, random values between 0 and 2pi will be assigned.
    A_red : scalar or None
        Amplitude of pulsar red noise.
    gamma_red : scalar or None
        Power law index of pulsar red noise.

    Returns
    -------
    gamma_ss : (R,) 1Darray
        Probability of detecting any single source, for each realization.
    SNR_ss : (F,R,L) NDarray
        SNR of each single source.

    """
    # Assign random single source sky positions, if not provided.
    if theta_ss is None:
        theta_ss = np.random.uniform(0, np.pi, size = hc_ss.size).reshape(hc_ss.shape)
    if phi_ss is None:
        phi_ss = np.random.uniform(0, 2*np.pi, size = hc_ss.size).reshape(hc_ss.shape)
    if iota_ss is None:
        iota_ss = np.random.uniform(0, np.pi, size = hc_ss.size).reshape(hc_ss.shape)
    if xi_ss is None:
        xi_ss = np.random.uniform(0, np.pi, size = hc_ss.size).reshape(hc_ss.shape)
    if Phi0_ss is None:
        Phi0_ss = np.random.uniform(0,2*np.pi, size=hc_ss.size).reshape(hc_ss.shape)

    # unitary vectors
    m_hat = _m_unitary_vector(theta_ss, phi_ss, xi_ss) # (3,F,R,L)
    n_hat = _n_unitary_vector(theta_ss, phi_ss, xi_ss) # (3,F,R,L)
    Omega_hat = _Omega_unitary_vector(theta_ss, phi_ss) # (3,F,R,L)


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
                                                   pi_hat) # (P,F,R,L)
    
    # noise spectral density
    S_i = _total_noise(cad, sigmas, hc_ss, hc_bg, fobs, Amp_red, gamma_red)

    # amplitudw
    amp = _amplitude(hc_ss, fobs, dfobs) # (F,R,L)

    # SNR (includes a_pol, b_pol, and Phi_T calculations internally)
    SNR_ss = _SNR_ss(amp, F_iplus, F_icross, iota_ss, dur, Phi0_ss, S_i, fobs) # (F,R,L)
    
    Num = hc_ss[:,0,:].size
    Fe_bar = _Fe_thresh(Num, alpha_0=alpha_0) # scalar

    gamma_ssi = _gamma_ssi(Fe_bar, rho=SNR_ss) # (F,R,L)
    gamma_ss = _ss_detection_probability(gamma_ssi) # (R,)

    if ret_SNR:
        return gamma_ss, SNR_ss
    else:
        return gamma_ss


def detect_ss_scDeter(pulsars, scDeter, hc_ss, alpha_0=0.001, ret_SNR=False,
              theta_ss=None, phi_ss=None, iota_ss=None, xi_ss=None, Phi0_ss=None):
    """ Calculate the single source detection probability, and all intermediary steps.
    
    Parameters
    ----------
    pulsars : (P,) list of hasasia.Pulsar objects
        A set of pulsars generated by hasasia.sim.sim_pta()
    scDeter : 
    
    hc_ss : (F,R,L) NDarray of scalars
        Characteristic strain of the L loudest single sources at 
        each frequency, for R realizations.
    alpha_0 : scalar
        False alarm probability
    ret_SNR : Bool
        Whether or not to also return SNR_ss.
    theta_ss : (F,R,L) NDarray or None
        Polar (latitudinal) angular position in the sky of each single source.
        If None, random values between 0 and pi will be assigned.
    phi_ss : (F,R,L) NDarray or None
        Azimuthal (longitudinal) angular position in the sky of each single source.
        If None, random values between 0 and 2pi will be assigned.
    iota_ss : (F,R,L) NDarray or None
        Inclination of each single source with respect to the line of sight.
        If None, random values between 0 and pi will be assigned.
    xi_ss : (F,R,L) NDarray or None
        ???
        If None, assigned same random values as iota_ss, because I think these
        might both be referring to the same thing.
    Phi0_ss : (F,R,L) NDarray or None
        Initial GW phase.
        If None, random values between 0 and 2pi will be assigned.

    Returns
    -------
    gamma_ss : (R,) 1Darray
        Probability of detecting any single source, for each realization.
    rho_h_ss: (F,R,L) NDarray
        SNR of each single source.

    """

    # get pulsar properties
    thetas = np.zeros(len(pulsars))
    phis = np.zeros(len(pulsars))
    sigmas = np.zeros(len(pulsars))
    for ii in range(len(pulsars)):
        thetas[ii] = pulsars[ii].theta
        phis[ii] = pulsars[ii].phi
        sigmas[ii] = np.mean(pulsars[ii].toaerrs)

    # Assign random single source sky positions, if not provided.
    if theta_ss is None:
        theta_ss = np.random.uniform(0, np.pi, size = hc_ss.size).reshape(hc_ss.shape)
    if phi_ss is None:
        phi_ss = np.random.uniform(0, 2*np.pi, size = hc_ss.size).reshape(hc_ss.shape)
    if iota_ss is None:
        iota_ss = np.random.uniform(0, np.pi, size = hc_ss.size).reshape(hc_ss.shape)
    if xi_ss is None:
        xi_ss = np.random.uniform(0, np.pi, size = hc_ss.size).reshape(hc_ss.shape)
    if Phi0_ss is None:
        Phi0_ss = np.random.uniform(0,2*np.pi, size=hc_ss.size).reshape(hc_ss.shape)

    # unitary vectors
    m_hat = _m_unitary_vector(theta_ss, phi_ss, xi_ss) # (3,F,R,L)
    n_hat = _n_unitary_vector(theta_ss, phi_ss, xi_ss) # (3,F,R,L)
    Omega_hat = _Omega_unitary_vector(theta_ss, phi_ss) # (3,F,R,L)
    pi_hat = _pi_unitary_vector(phis, thetas) # (3,P)

    # antenna pattern functions
    F_iplus, F_icross = _antenna_pattern_functions(m_hat, n_hat, Omega_hat, 
                                                   pi_hat) # (P,F,R,L)

    # rho_ss (corresponds to SNR)
    rho_h_ss = np.zeros(hc_ss.shape) # (F,R,L)
    for rr in range(len(hc_ss[0])):
        for ll in range(len(hc_ss[0,0])):
            rho_h_ss[:,rr,ll] =   scDeter.SNR(hc_ss[:,rr,ll]) 
    
    Num = hc_ss[:,0,:].size
    Fe_bar = _Fe_thresh(Num, alpha_0=alpha_0) # scalar

    gamma_ssi = _gamma_ssi(Fe_bar, rho=rho_h_ss) # (F,R,L)
    gamma_ss = _ss_detection_probability(gamma_ssi) # (R,)

    if ret_SNR:
        return gamma_ss, rho_h_ss,
    else:
        return gamma_ss

def detect_ss_scDeter_full(pulsars, scDeter, hc_ss, alpha_0=0.001, ret_SNR=False,
              theta_ss=None, phi_ss=None, iota_ss=None, xi_ss=None, Phi0_ss=None):
    """ Calculate the single source detection probability, and all intermediary steps.
    
    Parameters
    ----------
    pulsars : (P,) list of hasasia.Pulsar objects
        A set of pulsars generated by hasasia.sim.sim_pta()
    scDeter : 
       
    hc_ss : (F,R,L) NDarray of scalars
        Characteristic strain of the L loudest single sources at 
        each frequency, for R realizations.
    alpha_0 : scalar
        False alarm probability
    ret_SNR : Bool
        Whether or not to also return SNR_ss.
    theta_ss : (F,R,L) NDarray or None
        Polar (latitudinal) angular position in the sky of each single source.
        If None, random values between 0 and pi will be assigned.
    phi_ss : (F,R,L) NDarray or None
        Azimuthal (longitudinal) angular position in the sky of each single source.
        If None, random values between 0 and 2pi will be assigned.
    iota_ss : (F,R,L) NDarray or None
        Inclination of each single source with respect to the line of sight.
        If None, random values between 0 and pi will be assigned.
    xi_ss : (F,R,L) NDarray or None
        ???
        If None, assigned same random values as iota_ss, because I think these
        might both be referring to the same thing.
    Phi0_ss : (F,R,L) NDarray or None
        Initial GW phase.
        If None, random values between 0 and 2pi will be assigned.

    Returns
    -------
    gamma_ss : (R,) 1Darray
        Probability of detecting any single source, for each realization.
    rho_h_ss : (F,R,L) NDarray
        SNR of each single source.

    """

    # get pulsar properties
    thetas = np.zeros(len(pulsars))
    phis = np.zeros(len(pulsars))
    sigmas = np.zeros(len(pulsars))
    for ii in range(len(pulsars)):
        thetas[ii] = pulsars[ii].theta
        phis[ii] = pulsars[ii].phi
        sigmas[ii] = np.mean(pulsars[ii].toaerrs)

    # Assign random single source sky positions, if not provided.
    if theta_ss is None:
        theta_ss = np.random.uniform(0, np.pi, size = hc_ss.size).reshape(hc_ss.shape)
    if phi_ss is None:
        phi_ss = np.random.uniform(0, 2*np.pi, size = hc_ss.size).reshape(hc_ss.shape)
    if iota_ss is None:
        iota_ss = np.random.uniform(0, np.pi, size = hc_ss.size).reshape(hc_ss.shape)
    if xi_ss is None:
        xi_ss = np.random.uniform(0, np.pi, size = hc_ss.size).reshape(hc_ss.shape)
    if Phi0_ss is None:
        Phi0_ss = np.random.uniform(0,2*np.pi, size=hc_ss.size).reshape(hc_ss.shape)

    # rho_ss (corresponds to SNR)
    rho_h_ss = np.zeros(hc_ss.shape) # (F,R,L)
    for rr in range(len(hc_ss[0])):
        for ll in range(len(hc_ss[0,0])):
            rho_h_ss[:,rr,ll] =   scDeter.SNR(hc_ss[:,rr,ll], iota=iota_ss[:,rr,ll], psi=xi_ss[:,rr,ll]) 
            # this doesn't work
    
    Num = hc_ss[:,0,:].size
    Fe_bar = _Fe_thresh(Num, alpha_0=alpha_0) # scalar

    gamma_ssi = _gamma_ssi(Fe_bar, rho=rho_h_ss) # (F,R,L)
    gamma_ss = _ss_detection_probability(gamma_ssi) # (R,)

    if ret_SNR:
        return gamma_ss, rho_h_ss,
    else:
        return gamma_ss


def detect_ss_skymap(pulsars, skymap, hc_ss, alpha_0=0.001, ret_SNR=False,
              theta_ss=None, phi_ss=None, iota_ss=None, xi_ss=None, Phi0_ss=None,
              debug=True):
    """ Calculate the single source detection probability, and all intermediary steps.
    
    Parameters
    ----------
    pulsars : (P,) list of hasasia.Pulsar objects
        A set of pulsars generated by hasasia.sim.sim_pta()
    skymap : 
    hc_ss : (F,R,L) NDarray of scalars
        Characteristic strain of the L loudest single sources at 
        each frequency, for R realizations.
    alpha_0 : scalar
        False alarm probability
    ret_SNR : Bool
        Whether or not to also return SNR_ss.
    theta_ss : (F,R,L) NDarray or None
        Polar (latitudinal) angular position in the sky of each single source.
        If None, random values between 0 and pi will be assigned.
    phi_ss : (F,R,L) NDarray or None
        Azimuthal (longitudinal) angular position in the sky of each single source.
        If None, random values between 0 and 2pi will be assigned.
    iota_ss : (F,R,L) NDarray or None
        Inclination of each single source with respect to the line of sight.
        If None, not included.
    xi_ss : (F,R,L) NDarray or None
        Polarizationof each single source.
        If None, not included.
    Phi0_ss : (F,R,L) NDarray or None
        Initial GW phase.
        If None, random values between 0 and 2pi will be assigned.

    Returns
    -------
    gamma_ss : (R,) 1Darray
        Probability of detecting any single source, for each realization.
    rho_h_ss : (F,F,R,L) NDarray
        SNR of each single source.

    """

    # get pulsar properties
    thetas = np.zeros(len(pulsars))
    phis = np.zeros(len(pulsars))
    sigmas = np.zeros(len(pulsars))
    for ii in range(len(pulsars)):
        thetas[ii] = pulsars[ii].theta
        phis[ii] = pulsars[ii].phi
        sigmas[ii] = np.mean(pulsars[ii].toaerrs)

    # Assign random single source sky positions, if not provided.
    if theta_ss is None:
        theta_ss = np.random.uniform(0, np.pi, size = hc_ss.size).reshape(hc_ss.shape)
    if phi_ss is None:
        phi_ss = np.random.uniform(0, 2*np.pi, size = hc_ss.size).reshape(hc_ss.shape)
    # if iota_ss is None:
    #     iota_ss = np.random.uniform(0, np.pi, size = hc_ss.size).reshape(hc_ss.shape)
    # if xi_ss is None:
    #     xi_ss = np.random.uniform(0, np.pi, size = hc_ss.size).reshape(hc_ss.shape)
    if Phi0_ss is None:
        Phi0_ss = np.random.uniform(0,2*np.pi, size=hc_ss.size).reshape(hc_ss.shape)

    # rho_ss (corresponds to SNR)
    rho_h_ss = np.zeros((hc_ss.shape[0],hc_ss.shape[0],
                         hc_ss.shape[1], hc_ss.shape[2])) # (F,F,R,L)
    for rr in range(len(hc_ss[0])):
        for ll in range(len(hc_ss[0,0])):
            # print('hc_ss[:,rr,ll]', hc_ss[:,rr,ll].shape, hc_ss[:,rr,ll])
            if(iota_ss is not None): 
                iota=iota_ss[:,rr,ll] 
            else: 
                iota = None
            if(xi_ss is not None): 
                psi = xi_ss[:,rr,ll] # otherwise 
            else: 
                psi = None
            rho = skymap.SNR(hc_ss[:,rr,ll], iota=iota, psi=psi) 
            try:
                rho_h_ss[:,:,rr,ll]  = rho
                if (rr==0) and (ll==0):
                    if debug: print('rho', rho.shape) #, rho)
            except Exception:
                print('rho', rho.shape) #, rho)
                print("'rho_h_ss[:,:,rr,ll]  = rho' failed")
                raise
    
    Num = hc_ss[:,0,:].size
    Fe_bar = _Fe_thresh(Num, alpha_0=alpha_0) # scalar

    gamma_ssi = _gamma_ssi(Fe_bar, rho=rho_h_ss) # (F,R,L)
    gamma_ss = _ss_detection_probability(gamma_ssi) # (R,)

    if ret_SNR:
        return gamma_ss, rho_h_ss,
    else:
        return gamma_ss

def snr_skymap(pulsars, skymap, hc_ss, alpha_0=0.001, 
              theta_ss=None, phi_ss=None, iota_ss=None, xi_ss=None, Phi0_ss=None,
              debug=True):
    """ Calculate the single source detection probability, and all intermediary steps.
    
    Parameters
    ----------
    pulsars : (P,) list of hasasia.Pulsar objects
        A set of pulsars generated by hasasia.sim.sim_pta()
    skymap : 
    hc_ss : (F,R,L) NDarray of scalars
        Characteristic strain of the L loudest single sources at 
        each frequency, for R realizations.
    alpha_0 : scalar
        False alarm probability
    theta_ss : (F,R,L) NDarray or None
        Polar (latitudinal) angular position in the sky of each single source.
        If None, random values between 0 and pi will be assigned.
    phi_ss : (F,R,L) NDarray or None
        Azimuthal (longitudinal) angular position in the sky of each single source.
        If None, random values between 0 and 2pi will be assigned.
    iota_ss : (F,R,L) NDarray or None
        Inclination of each single source with respect to the line of sight.
        If None, not included.
    xi_ss : (F,R,L) NDarray or None
        Polarizationof each single source.
        If None, not included.
    Phi0_ss : (F,R,L) NDarray or None
        Initial GW phase.
        If None, random values between 0 and 2pi will be assigned.

    Returns
    -------
    rho_h_ss : (F,F,R,L) NDarray
        SNR of each single source.??

    """

    # get pulsar properties
    thetas = np.zeros(len(pulsars))
    phis = np.zeros(len(pulsars))
    sigmas = np.zeros(len(pulsars))
    for ii in range(len(pulsars)):
        thetas[ii] = pulsars[ii].theta
        phis[ii] = pulsars[ii].phi
        sigmas[ii] = np.mean(pulsars[ii].toaerrs)

    # Assign random single source sky positions, if not provided.
    if theta_ss is None:
        theta_ss = np.random.uniform(0, np.pi, size = hc_ss.size).reshape(hc_ss.shape)
    if phi_ss is None:
        phi_ss = np.random.uniform(0, 2*np.pi, size = hc_ss.size).reshape(hc_ss.shape)
    # if iota_ss is None:
    #     iota_ss = np.random.uniform(0, np.pi, size = hc_ss.size).reshape(hc_ss.shape)
    # if xi_ss is None:
    #     xi_ss = np.random.uniform(0, np.pi, size = hc_ss.size).reshape(hc_ss.shape)
    if Phi0_ss is None:
        Phi0_ss = np.random.uniform(0,2*np.pi, size=hc_ss.size).reshape(hc_ss.shape)

    # rho_ss (corresponds to SNR)
    rho_h_ss = np.zeros((hc_ss.shape[0],hc_ss.shape[0],
                         hc_ss.shape[1], hc_ss.shape[2])) # (F,F,R,L)
    for rr in range(len(hc_ss[0])):
        for ll in range(len(hc_ss[0,0])):
            if debug: print('hc_ss[:,rr,ll]', hc_ss[:,rr,ll].shape) #, hc_ss[:,rr,ll])
            if(iota_ss is not None): 
                iota=iota_ss[:,rr,ll] 
            else: 
                iota = None
            if(xi_ss is not None): 
                psi = xi_ss[:,rr,ll] # otherwise 
            else: 
                psi = None
            rho = skymap.SNR(hc_ss[:,rr,ll], iota=iota, psi=psi) 
            try:
                rho_h_ss[:,:,rr,ll]  = rho
                if (rr==0) and (ll==0):
                    print('rho', rho.shape) #, rho)
            except Exception:
                if debug: print('rho', rho.shape) #, rho)
                if debug: print("'rho_h_ss[:,:,rr,ll]  = rho' failed")
                raise
                # return rho_h_ss

    return rho_h_ss