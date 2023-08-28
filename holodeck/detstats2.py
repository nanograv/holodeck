"""Detection Statistics module.

This module calculates detection statistics for single source and background strains.
It is a cleaned up version of detstats2.

"""


import numpy as np
from scipy import special, integrate
from sympy import nsolve, Symbol
import h5py
import matplotlib.pyplot as plt
import os
from datetime import datetime
import warnings
import abc


import holodeck as holo
from holodeck import utils, cosmo, log, plot, sam_cython
from holodeck.constants import MPC, YR
from holodeck.sams import cyutils as sam_cyutils

import hasasia.sensitivity as hsen
import hasasia.sim as hsim
import hasasia as has

GAMMA_RHO_GRID_PATH = '/Users/emigardiner/GWs/holodeck/output/rho_gamma_grids'
HC_REF15_10YR = 11.2*10**-15 
DEF_THRESH=0.5

class DetStats:
    """ Base class for detection statistics of a given hc_ss and hc_bg model (any number of realizations).
    Includes basic methods to be inherited by different calibration classes.
    """

    def __init__(self, fobs, hc_ss, hc_bg, nskies=25, npsrs=40, debug=False,
                 # single source positions
                 theta_ss=None, phi_ss=None, Phi0_ss=None, iota_ss=None, psi_ss=None,
                 ):   

        self._hcss = hc_ss
        self._hcbg = hc_bg
        self._nskies = nskies
        self._npsrs = npsrs

        # Properties that need to be calculate when first used
        self._pulsars = None # Needs to be set
        self._orf = None
        self._S_h = None
        self._noise_ss = None # shape P,F,R,L
        self._noise_bg = None # shape P,F,R
        
        # # Build ss skies
        # if debug: print('Building ss skies.')
        # theta_ss, phi_ss, Phi0_ss, iota_ss, psi_ss = _build_skies(nfreqs, nskies, nloudest)



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


    ########################################################################
    ##################### Functions for the Background #####################
    ########################################################################

    ######################## Power Spectral Density ########################

    @property
    def _power_spectral_density(self):
        """ The spectral density S_h(f_k) ~ S_h0(f_k) at the kth frequency

        Returns
        -------
        S_h : (F,R) NDarray of scalars
            Actual (S_h) or ~construction (S_h0) value of the background spectral density,
            in units of [Hz]^-3, for R realizations.

        Follows Eq. (25) of Rosado et al. 2015
        """

        if self._S_h is None:
            self._S_h = self.hc_bg**2 / (12 * np.pi**2 * self.fobs[:,np.newaxis]**3)
        return self._S_h

    @property
    def _orf(self):
        if self._orf is None:
            self._orf = _orf_pta(self.pulsars)
        return self._orf

    ######################## mu_1, sigma_0, sigma_1 ########################

    def _sigma0_Bstatistic(self):
        """ Calculate sigma_1 for the background, by summing over all pulsars and frequencies.
        Assuming the B statistic, which maximizes S/N_B = mu_1/sigma_1.

        Uses
        ----
        self._noise_bg : (P,F,R) Ndarray of scalars
            Noise spectral density of each pulsar.
        self._orf : (P,P) 2Darray of scalars
            Overlap reduction function for j>i, 0 otherwise.
        Sh0_bg : (F,R) 1Darray of scalars
            Value of spectral density used to construct the statistic.

        Returns
        -------
        sigma_0B : (R,) 1Darray
            Standard deviation of the null PDF assuming the B-statistic.


        Follows Eq. (A17) from Rosado et al. 2015.
        """

        # # Check that Gamma_{j<=i}, when testing new methods.
        # for ii in range(self._npsrs):
        #     for jj in range(ii+1):
        #         assert Gamma[ii,jj] == 0, f'Gamma[{ii},{jj}] = {Gamma[ii,jj]}, but it should be 0!'

        # Cast parameters to desired shapes
        Gamma = self._orf[:,:,np.newaxis,np.newaxis] # (P,P,1,1)
        Sh0_bg = self._Sh0_bg[np.newaxis,np.newaxis,:,:] # (1,1,F,R)
        noise_i = self._noise_bg[:,np.newaxis,:,:] # (P,1,F,R)
        noise_j = self._noise_bg[np.newaxis,:,:,:] # (P,1,F,R)

        # Calculate sigma_0B
        numer = (Gamma**2 * Sh0_bg**2
                * noise_i * noise_j)
        denom = ((noise_j + Sh0_bg)
                * (noise_j + Sh0_bg)
                + Gamma**2 * Sh0_bg**2)**2

        sum = np.sum(numer/denom, axis=(0,1,2))
        sigma_0B = np.sqrt(2*sum)
        return sigma_0B

    def _sigma1_Bstatistic(self):
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
        Gamma = self._orf[:,:,np.newaxis,np.newaxis] # (P,P,1,1)
        Sh0_bg = self._Sh0_bg[np.newaxis,np.newaxis,:,:] # (1,1,F,R)
        Sh_bg = self.Sh_bg[np.newaxis,np.newaxis,:] # (1,1,F,R)
        noise_i = self._noise_bg[:,np.newaxis,:,:] # (P,1,F,R)
        noise_j = self._noise_bg[np.newaxis,:,:,:] # (P,1,F,R)



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



class Noise_Models:
    """ Class for calculating different pta noise models.
    
    """

    def white_noise(delta_t, sigma_i):
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



###############################################################
###################### Utility Functions ######################
###############################################################

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


