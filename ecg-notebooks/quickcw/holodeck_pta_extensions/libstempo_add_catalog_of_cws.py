from __future__ import division, print_function, absolute_import

import math, os

import numpy as N
import scipy.interpolate as interp
import scipy.constants as sc
import astropy as ap

from libstempo import eccUtils as eu
import libstempo.toasim as LT

import numba as nb
#make sure to use the right threading layer
from numba import config
config.THREADING_LAYER = 'omp'
#config.THREADING_LAYER = 'tbb'
print("Number of cores used for parallel running: ", config.NUMBA_NUM_THREADS)

from numba import jit,njit,prange

from holodeck import cosmo, utils, plot

import ephem

day = 24 * 3600
year = 365.25 * day
DMk = 4.15e3           # Units MHz^2 cm^3 pc sec

#from memory_profiler import profile

#@profile
def add_catalog_of_cws(psr, gwtheta_list, gwphi_list, mc_list, dist_list, fgw_list, phase0_list, psi_list, inc_list, pdist=1.0,
                       pphase=None, psrTerm=True, evolve=True, phase_approx=False, tref=0, chunk_size=10_000_000):
    # pulsar location
    if 'RAJ' and 'DECJ' in psr.pars():
        ptheta = N.pi/2 - psr['DECJ'].val
        pphi = psr['RAJ'].val
    elif 'ELONG' and 'ELAT' in psr.pars():
        fac = 180./N.pi
        coords = ephem.Equatorial(ephem.Ecliptic(str(psr['ELONG'].val*fac),
                                                 str(psr['ELAT'].val*fac)))

        ptheta = N.pi/2 - float(repr(coords.dec))
        pphi = float(repr(coords.ra))

    # use definition from Sesana et al 2010 and Ellis et al 2012
    phat = N.array([N.sin(ptheta)*N.cos(pphi), N.sin(ptheta)*N.sin(pphi),\
            N.cos(ptheta)])

    # get TOAs in seconds
    toas = psr.toas()*86400 - tref

    #numba doesn't work with float128, so we need to convert phat and toas to float64
    #this should not be a problem, since we do not need such high precision in pulsar sky location or toas
    #for toas, note that this is only used to calculate the GW waveform and not to form the actual residuals
    if mc_list.size>1_000:
        #print("parallel")
        N_chunk = int(N.ceil(mc_list.size/chunk_size))
        print(N_chunk)
        for jjj in range(N_chunk):
            print(str(jjj) + " / " + str(N_chunk))
            idxs = range(jjj*chunk_size, min((jjj+1)*chunk_size,mc_list.size) )
            print(idxs)
            res = loop_over_CWs_parallel(phat.astype('float64'), toas.astype('float64'),
                                         gwtheta_list[idxs], gwphi_list[idxs], mc_list[idxs], dist_list[idxs],
                                         fgw_list[idxs], phase0_list[idxs], psi_list[idxs], inc_list[idxs],
                                         pdist=pdist, pphase=pphase, psrTerm=psrTerm, evolve=evolve, phase_approx=phase_approx)
            #add current batch to TOAs
            psr.stoas[:] += res/86400
    else:
        res = loop_over_CWs(phat.astype('float64'), toas.astype('float64'), gwtheta_list, gwphi_list, mc_list, dist_list, fgw_list, phase0_list, psi_list, inc_list, pdist=pdist,
                            pphase=pphase, psrTerm=psrTerm, evolve=evolve, phase_approx=phase_approx)

        #End of loop over CW sources
        #Now add residual to TOAs
        psr.stoas[:] += res/86400

#@profile
@njit(fastmath=False, parallel=True)
def loop_over_CWs_parallel(phat, toas, gwtheta_list, gwphi_list, mc_list, dist_list, fgw_list, phase0_list, psi_list, inc_list, pdist=1.0,
                           pphase=None, psrTerm=True, evolve=True, phase_approx=False):
    # set up array for residuals
    res = N.zeros((len(mc_list),toas.size))

    for iii in prange(len(mc_list)):
    #for gwtheta, gwphi, mc, dist, fgw, phase0, psi, inc in zip(gwtheta_list, gwphi_list, mc_list, dist_list, fgw_list, phase0_list, psi_list, inc_list):
        #if iii%1_000==0: print(iii)
        gwtheta = gwtheta_list[iii]
        gwphi = gwphi_list[iii]
        mc = mc_list[iii]
        dist = dist_list[iii]
        fgw = fgw_list[iii]
        phase0 = phase0_list[iii]
        psi = psi_list[iii]
        inc = inc_list[iii]
        # convert units
        mc *= eu.SOLAR2S         # convert from solar masses to seconds
        dist *= eu.MPC2S    # convert from Mpc to seconds

        # define initial orbital frequency
        w0 = N.pi * fgw
        phase0 /= 2 # orbital phase
        w053 = w0**(-5/3)

        # define variable for later use
        cosgwtheta, cosgwphi = N.cos(gwtheta), N.cos(gwphi)
        singwtheta, singwphi = N.sin(gwtheta), N.sin(gwphi)
        sin2psi, cos2psi = N.sin(2*psi), N.cos(2*psi)
        incfac1, incfac2 = 0.5*(3+N.cos(2*inc)), 2*N.cos(inc)

        # unit vectors to GW source
        m = N.array([singwphi, -cosgwphi, 0.0])
        n = N.array([-cosgwtheta*cosgwphi, -cosgwtheta*singwphi, singwtheta])
        omhat = N.array([-singwtheta*cosgwphi, -singwtheta*singwphi, -cosgwtheta])

        # various factors invloving GW parameters
        fac1 = 256/5 * mc**(5/3) * w0**(8/3)
        fac2 = 1/32/mc**(5/3)
        fac3 = mc**(5/3)/dist

        # get antenna patterns
        fplus = 0.5 * (N.dot(m, phat)**2 - N.dot(n, phat)**2) / (1+N.dot(omhat, phat))
        fcross = (N.dot(m, phat)*N.dot(n, phat)) / (1 + N.dot(omhat, phat))
        cosMu = -N.dot(omhat, phat)

        # get values from pulsar object
        if pphase is not None:
            pd = pphase/(2*N.pi*fgw*(1-cosMu)) / eu.KPC2S
        else:
            pd = pdist

        # convert units
        pd *= eu.KPC2S   # convert from kpc to seconds

        # get pulsar time
        tp = toas-pd*(1-cosMu)

        # evolution
        if evolve:

            # calculate time dependent frequency at earth and pulsar
            omega = w0 * (1 - fac1 * toas)**(-3/8)
            omega_p = w0 * (1 - fac1 * tp)**(-3/8)

            # calculate time dependent phase
            phase = phase0 + fac2 * (w053 - omega**(-5/3))
            phase_p = phase0 + fac2 * (w053 - omega_p**(-5/3))

        # use approximation that frequency does not evlolve over observation time
        elif phase_approx:

            # frequencies
            omega = w0 * N.ones(toas.size) #make sure omega is always an array and never a float (numba typing stuff)
            omega_p = w0 * (1 + fac1 * pd*(1-cosMu))**(-3/8) * N.ones(toas.size) #make sure omega_p is always an array and never a float (numba typing stuff)

            # phases
            phase = phase0 + omega * toas
            phase_p = phase0 + fac2 * (w053 - omega_p**(-5/3)) + omega_p*toas

        # no evolution
        else:

            # monochromatic
            omega = w0 * N.ones(toas.size) #make sure omega is always an array and never a float (numba typing stuff)
            omega_p = omega

            # phases
            phase = phase0 + omega * toas
            phase_p = phase0 + omega * tp


        # define time dependent coefficients
        At = N.sin(2*phase) * incfac1
        Bt = N.cos(2*phase) * incfac2
        At_p = N.sin(2*phase_p) * incfac1
        Bt_p = N.cos(2*phase_p) * incfac2

        # now define time dependent amplitudes
        alpha = fac3 / omega**(1/3)
        alpha_p = fac3 / omega_p**(1/3)

        # define rplus and rcross
        rplus = alpha * (At*cos2psi + Bt*sin2psi)
        rcross = alpha * (-At*sin2psi + Bt*cos2psi)
        rplus_p = alpha_p * (At_p*cos2psi + Bt_p*sin2psi)
        rcross_p = alpha_p * (-At_p*sin2psi + Bt_p*cos2psi)

        # residuals
        if psrTerm:
            #make sure we add zeros rather than NaNs in the rare occasion when the binary already merged and produces negative frequencies
            rrr = fplus*(rplus_p-rplus)+fcross*(rcross_p-rcross)
            res[iii,:] = N.where(N.isnan(rrr), 0.0, rrr)
        else:
            rrr = -fplus*rplus - fcross*rcross
            res[iii,:] = N.where(N.isnan(rrr), 0.0, rrr)

    return N.sum(res, axis=0)


@njit(fastmath=False, parallel=False)
def loop_over_CWs(phat, toas, gwtheta_list, gwphi_list, mc_list, dist_list, fgw_list, phase0_list, psi_list, inc_list, pdist=1.0,
                  pphase=None, psrTerm=True, evolve=True, phase_approx=False):
    # set up array for residuals
    res = N.zeros(toas.size)

    for iii in range(len(mc_list)):
    #for gwtheta, gwphi, mc, dist, fgw, phase0, psi, inc in zip(gwtheta_list, gwphi_list, mc_list, dist_list, fgw_list, phase0_list, psi_list, inc_list):
        gwtheta = gwtheta_list[iii]
        gwphi = gwphi_list[iii]
        mc = mc_list[iii]
        dist = dist_list[iii]
        fgw = fgw_list[iii]
        phase0 = phase0_list[iii]
        psi = psi_list[iii]
        inc = inc_list[iii]
        # convert units
        mc *= eu.SOLAR2S         # convert from solar masses to seconds
        dist *= eu.MPC2S    # convert from Mpc to seconds

        # define initial orbital frequency
        w0 = N.pi * fgw
        phase0 /= 2 # orbital phase
        w053 = w0**(-5/3)

        # define variable for later use
        cosgwtheta, cosgwphi = N.cos(gwtheta), N.cos(gwphi)
        singwtheta, singwphi = N.sin(gwtheta), N.sin(gwphi)
        sin2psi, cos2psi = N.sin(2*psi), N.cos(2*psi)
        incfac1, incfac2 = 0.5*(3+N.cos(2*inc)), 2*N.cos(inc)

        # unit vectors to GW source
        m = N.array([singwphi, -cosgwphi, 0.0])
        n = N.array([-cosgwtheta*cosgwphi, -cosgwtheta*singwphi, singwtheta])
        omhat = N.array([-singwtheta*cosgwphi, -singwtheta*singwphi, -cosgwtheta])

        # various factors invloving GW parameters
        fac1 = 256/5 * mc**(5/3) * w0**(8/3)
        fac2 = 1/32/mc**(5/3)
        fac3 = mc**(5/3)/dist

        # get antenna patterns
        fplus = 0.5 * (N.dot(m, phat)**2 - N.dot(n, phat)**2) / (1+N.dot(omhat, phat))
        fcross = (N.dot(m, phat)*N.dot(n, phat)) / (1 + N.dot(omhat, phat))
        cosMu = -N.dot(omhat, phat)

        # get values from pulsar object
        if pphase is not None:
            pd = pphase/(2*N.pi*fgw*(1-cosMu)) / eu.KPC2S
        else:
            pd = pdist

        # convert units
        pd *= eu.KPC2S   # convert from kpc to seconds

        # get pulsar time
        tp = toas-pd*(1-cosMu)

        # evolution
        if evolve:

            # calculate time dependent frequency at earth and pulsar
            omega = w0 * (1 - fac1 * toas)**(-3/8)
            omega_p = w0 * (1 - fac1 * tp)**(-3/8)

            # calculate time dependent phase
            phase = phase0 + fac2 * (w053 - omega**(-5/3))
            phase_p = phase0 + fac2 * (w053 - omega_p**(-5/3))

        # use approximation that frequency does not evlolve over observation time
        elif phase_approx:

            # frequencies
            omega = w0 * N.ones(toas.size) #make sure omega is always an array and never a float (numba typing stuff)
            omega_p = w0 * (1 + fac1 * pd*(1-cosMu))**(-3/8) * N.ones(toas.size) #make sure omega_p is always an array and never a float (numba typing stuff)

            # phases
            phase = phase0 + omega * toas
            phase_p = phase0 + fac2 * (w053 - omega_p**(-5/3)) + omega_p*toas

        # no evolution
        else:

            # monochromatic
            omega = w0 * N.ones(toas.size) #make sure omega is always an array and never a float (numba typing stuff)
            omega_p = omega

            # phases
            phase = phase0 + omega * toas
            phase_p = phase0 + omega * tp


        # define time dependent coefficients
        At = N.sin(2*phase) * incfac1
        Bt = N.cos(2*phase) * incfac2
        At_p = N.sin(2*phase_p) * incfac1
        Bt_p = N.cos(2*phase_p) * incfac2

        # now define time dependent amplitudes
        alpha = fac3 / omega**(1/3)
        alpha_p = fac3 / omega_p**(1/3)

        # define rplus and rcross
        rplus = alpha * (At*cos2psi + Bt*sin2psi)
        rcross = alpha * (-At*sin2psi + Bt*cos2psi)
        rplus_p = alpha_p * (At_p*cos2psi + Bt_p*sin2psi)
        rcross_p = alpha_p * (-At_p*sin2psi + Bt_p*cos2psi)

        # residuals
        if psrTerm:
            #make sure we add zeros rather than NaNs in the rare occasion when the binary already merged and produces negative frequencies
            rrr = fplus*(rplus_p-rplus)+fcross*(rcross_p-rcross)
            res += N.where(N.isnan(rrr), 0.0, rrr)
        else:
            rrr = -fplus*rplus - fcross*rcross
            res += N.where(N.isnan(rrr), 0.0, rrr)

    return res

#@profile
def add_gwb_plus_outlier_cws(psrs, vals, weights, fobs, T_obs, outlier_per_bin=100, seed=1234):
    PC = ap.constants.pc.cgs.value
    MSOL = ap.constants.M_sun.cgs.value
    
    f_centers = []
    for iii in range(fobs.size-1):
        f_centers.append((fobs[iii+1]+fobs[iii])/2)
        
    f_centers = N.array(f_centers)
    
    mc = utils.chirp_mass(*utils.m1m2_from_mtmr(vals[0], vals[1]))
    rz = vals[2, :]
    frst = vals[3] * (1.0 + rz)
    #get comoving distance for h calculation
    dc = cosmo.z_to_dcom(rz)
    #and luminosity distance for outlier injections below
    dl = N.copy(dc) * (1.0 + rz) #cosmo.luminosity_distance(rz).cgs.value
    
    hs = utils.gw_strain_source(mc, dc, frst/2)
    #testing
    #speed_of_light = 299792458.0 #m/s
    #T_sun = 1.327124400e20 / speed_of_light**3
    #hs_mine = 8/N.sqrt(10) * (mc/MSOL*T_sun)**(5/3) * (N.pi*frst)**(2/3) / (dc/100.0) * speed_of_light
    #8/sqrt(10) * pi^2/3 * G^5/3 / c^4 * M_c^5/3* (2*f_orb_rest)^2/3 / dcom
    #print(hs[0], hs_mine[0])
    fo = vals[-1]

    #convert mc to observer frame since that's what we will need for outlier injection below
    mc = mc * (1.0 + rz) 
    
    freq_idxs = N.digitize(fo,fobs)
    
    free_spec = N.ones(fobs.shape[0]-1)*1e-100
    outlier_hs = N.zeros(free_spec.shape[0]*outlier_per_bin)
    outlier_fo = N.zeros(free_spec.shape[0]*outlier_per_bin)
    outlier_mc = N.zeros(free_spec.shape[0]*outlier_per_bin)
    outlier_dl = N.zeros(free_spec.shape[0]*outlier_per_bin)
    
    weighted_h_square = weights * hs**2 * fo * T_obs #apply weights and convert to characteristic strain
    for k in range(free_spec.shape[0]):
        bool_mask = (freq_idxs-1)==k
        weighted_h_square_bin = weighted_h_square[bool_mask]
        sort_idx = N.argsort(weighted_h_square_bin)[::-1]
        weighted_h_squared_bin_sorted = weighted_h_square_bin[sort_idx]
        fo_bin_sorted = fo[bool_mask][sort_idx] #Hz
        mc_bin_sorted = mc[bool_mask][sort_idx]/MSOL #solar mass
        dl_bin_sorted = dl[bool_mask][sort_idx]/PC/1e6 #Mpc
        
        if outlier_per_bin<weighted_h_squared_bin_sorted.shape[0]:
            outlier_limit = outlier_per_bin
        else:
            outlier_limit = weighted_h_squared_bin_sorted.shape[0]
        
        for j in range(outlier_limit):
            outlier_hs[outlier_per_bin*k+j] = weighted_h_squared_bin_sorted[j]
            outlier_fo[outlier_per_bin*k+j] = fo_bin_sorted[j]
            outlier_mc[outlier_per_bin*k+j] = mc_bin_sorted[j]
            outlier_dl[outlier_per_bin*k+j] = dl_bin_sorted[j]
        
        free_spec[k] += N.sum(weighted_h_squared_bin_sorted[outlier_per_bin:])
    
    FreeSpec = N.array([f_centers,N.sqrt(free_spec)]).T
    
    print(FreeSpec)

    howml = 10
    
    LT.createGWB(psrs, None, None, userSpec=FreeSpec, howml=howml, seed=seed)
    
    #for pulsar in psrs:
    #    #print("GWB")
    #    #print(pulsar.name)
    #    #print(pulsar.residuals())
    #    #pulsar.fit()
        
    #filter out empty entries in outliers
    outlier_hs = outlier_hs[N.where(outlier_hs>0)]
    outlier_fo = outlier_fo[N.where(outlier_fo>0)]
    outlier_mc = outlier_mc[N.where(outlier_mc>0)]
    outlier_dl = outlier_dl[N.where(outlier_dl>0)]
        
    N_CW = outlier_hs.shape[0]
    
    random_gwthetas = N.arccos(N.random.uniform(low=-1.0, high=1.0, size=N_CW))
    random_gwphis = N.random.uniform(low=0.0, high=2*N.pi, size=N_CW)
    random_phases = N.random.uniform(low=0.0, high=2*N.pi, size=N_CW)
    random_psis = N.random.uniform(low=0.0, high=N.pi, size=N_CW)
    random_incs = N.arccos(N.random.uniform(low=-1.0, high=1.0, size=N_CW))
    
    for pulsar in psrs:
        add_catalog_of_cws(pulsar,
                           gwtheta_list=random_gwthetas, gwphi_list=random_gwphis, 
                           mc_list=outlier_mc, dist_list=outlier_dl, fgw_list=outlier_fo,
                           phase0_list=random_phases, psi_list=random_psis, inc_list=random_incs,
                           pdist=1.0, pphase=None, psrTerm=True, evolve=True,
                           phase_approx=False, tref=53000*86400)
    
    #for pulsar in psrs:
    #    #print("GWB+CW")
    #    #print(pulsar.residuals())
    #    #pulsar.fit()
    
    return f_centers, free_spec, outlier_fo, outlier_hs, outlier_mc, outlier_dl, random_gwthetas, random_gwphis, random_phases, random_psis, random_incs 
