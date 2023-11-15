"""Single Source (ss) Gravitational Wave calculations module.

"""

import logging
import warnings
import numpy as np
# import astropy as ap
# import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm


import kalepy as kale # noqa
# import kalepy.plot

import holodeck as holo
import holodeck.cyutils
from holodeck import cosmo, utils, plot, gravwaves
from holodeck.constants import MSOL, PC, YR, MPC

# Silence annoying numpy errors
np.seterr(divide='ignore', invalid='ignore', over='ignore')
warnings.filterwarnings("ignore", category=UserWarning)

log = holo.log
log.setLevel(logging.INFO)

par_names = np.array(['mtot', 'mrat', 'redz_init', 'redz_final', 'dcom_final', 'sepa_final', 'angs_final'])
par_labels = np.array(['Total Mass $M$ ($M_\odot$)', 'Mass Ratio $q$', 'Initial Redshift $z_i$', 'Final Redshift $z_f$',
                   'Final Comoving Distance $d_c$ (Mpc)', 'Final Separation (pc)', 'Final Angular Separation (rad)'])
par_units = np.array([1/MSOL, 1, 1, 1, 1/MPC,  1/PC, 1])


###################################################
############## STRAIN CALCULATIONS ################
###################################################


def ss_gws_redz(edges, redz, number, realize, loudest = 1, params = False):

    """ Calculate strain from the loudest single sources and background.


    Parameters
    ----------
    edges : (4,) list of 1darrays
        A list containing the edges along each dimension.  The four dimensions correspond to
        total mass, mass ratio, redshift, and observer-frame orbital frequency.
        The length of each of the four arrays is M+1, Q+1, Z+1, F+1.
    redz : (M,Q,Z,F) NDarray
        redz_final for self-consistent hardening models (Fixed_Time).
        redz_prime for non self-consisten hardening models (Hard_GW).
    number : (M, Q, Z, F) ndarray of scalars
        The number of binaries in each bin of parameter space.  This is calculated by integrating
        `dnum` over each bin.
    realize : int
        Specification of how many discrete realizations to construct.
    loudest : int
        Number of loudest single sources to separate from background.


    Returns
    -------
    hc_ss : (F, R, L) NDarray of scalars
        The characteristic strain of the L loudest single sources at each frequency.
    hc_bg : (F, R) NDarray of scalars
        Characteristic strain of the GWB.
    sspar : (4, F, R, L) NDarray of scalars
        Astrophysical parametes (total mass, mass ratio, initial redshift, final redshift) of each
        loud single sources, for each frequency and realization.
        Returned only if params = True.
    bgpar : (7, F, R) NDarray of scalars
        Average effective binary astrophysical parameters (total mass, mass ratio, initial redshift,
        final redshift, final comoving distances, final separation, final angular separation) for background sources at each frequency and realization,
        Returned only if params = True.
    """

    # All other bin midpoints
    mt = kale.utils.midpoints(edges[0]) #: total mass
    mr = kale.utils.midpoints(edges[1]) #: mass ratio
    rz = kale.utils.midpoints(edges[2]) #: initial redshift


    # hsfdf = hsamp^2 * f/df # this is same as hc^2
    h2fdf = gravwaves.char_strain_sq_from_bin_edges_redz(edges, redz)

    # indices of bins sorted by h2fdf
    indices = np.argsort(-h2fdf[...,0].flatten()) # just sort for first frequency
    unraveled = np.array(np.unravel_index(indices, (len(mt),len(mr),len(rz))))
    msort = unraveled[0,:]
    qsort = unraveled[1,:]
    zsort = unraveled[2,:]

    if np.any(np.logical_and(redz<0, redz!=-1)):
                err = np.sum(np.logical_and(redz<0, redz!=-1))
                err = f"{err} redz < 0 and !=-1 found in redz, in ss_gws_redz()"
                raise ValueError(err)

    # For multiple realizations, using cython
    if(utils.isinteger(realize)):
        if(params == True):
            # hc2ss = char strain squared of each loud single source
            # hc2bg = char strain squared of the background
            # lspar = avg parameters of loudest sources
            # bgpar = avg parameters of background
            # ssidx = indices of loud single sources

            # redshifts are defined across 4D grid, shape (M, Q, Z, Fc)
            #    where M, Q, Z are edges and Fc is frequency centers
            # find midpoints of redshifts in M, Q, Z dimensions, to end up with (M-1, Q-1, Z-1, Fc)
            for dd in range(3):
                redz = np.moveaxis(redz, dd, 0)
                redz = kale.utils.midpoints(redz, axis=0)
                redz = np.moveaxis(redz, 0, dd)

            # if np.any(np.logical_and(redz<0, redz!=-1)):
            #     err = np.sum(np.logical_and(redz<0, redz!=-1))
            #     err = f"{err} redz < 0 and !=-1 found in redz, in ss_gws_redz() after kale.utils.midpoints"
            #     raise ValueError(err)

            dcom_final = +np.inf*np.ones_like(redz)

            sel = (redz > 0.0)
            redz[~sel] = -1.0
            redz[redz<0] = -1.0

            dcom_final[sel] = cosmo.comoving_distance(redz[sel]).cgs.value
            if np.any(dcom_final<0): print('dcom_final<0 found')
            if np.any(np.isnan(dcom_final)): print('nan dcom_final found')
            # redz[redz<0] = -1

            fobs_orb_edges = edges[-1]
            fobs_orb_cents = kale.utils.midpoints(fobs_orb_edges)
            frst_orb_cents = utils.frst_from_fobs(fobs_orb_cents[np.newaxis,np.newaxis,np.newaxis,:], redz) # (M,Q,Z,F,), final


            sepa = utils.kepler_sepa_from_freq(mt[:,np.newaxis,np.newaxis,np.newaxis], frst_orb_cents) # (M,Q,Z,F) in cm
            angs = utils.angs_from_sepa(sepa, dcom_final, redz) # (M,Q,Z,F) use sepa and dcom in cm



            hc2ss, hc2bg, sspar, bgpar = \
                holo.cyutils.loudest_hc_and_par_from_sorted_redz(
                    number, h2fdf, realize, loudest,
                    mt, mr, rz, redz, dcom_final, sepa, angs,
                    msort, qsort, zsort)
            hc_ss = np.sqrt(hc2ss) # calculate single source strain
            hc_bg = np.sqrt(hc2bg) # calculate background strain



            # check that all final redshifts are positive or -1
            if np.any(np.logical_and(sspar[3]<0, sspar[3]!=-1)):
                err = np.sum(np.logical_and(sspar[3]<0, sspar[3]!=-1))
                err = f"check 1: {err} out of {sspar[3].size} sspar[3] are negative and not -1 in sings.ss_gws_redz()"
                raise ValueError(err)


            # return
            return hc_ss, hc_bg, sspar, bgpar

        else:
            # use cython to get h_c^2 for ss and bg
            hc2ss, hc2bg = holo.cyutils.loudest_hc_from_sorted(number, h2fdf, realize, loudest,
                                                               msort, qsort, zsort)
            hc_ss = np.sqrt(hc2ss)
            hc_bg = np.sqrt(hc2bg)
            return hc_ss, hc_bg

    # OTHERWISE
    else:
        raise Exception("`realize` ({}) must be an integer!")


# version for running libraries
def ss_gws(edges, number, realize, loudest = 1, params = False):

    """ Calculate strain from the loudest single sources and background.


    Parameters
    ----------
    edges : (4,) list of 1darrays
        A list containing the edges along each dimension.  The four dimensions correspond to
        total mass, mass ratio, redshift, and observer-frame orbital frequency.
        The length of each of the four arrays is M+1, Q+1, Z+1, F+1.
    number : (M, Q, Z, F) ndarray of scalars
        The number of binaries in each bin of parameter space.  This is calculated by integrating
        `dnum` over each bin.
    realize : int,
        Specification of how many discrete realizations to construct.
    loudest : int
        Number of loudest single sources to separate from background.


    Returns
    -------
    hc_ss : (F, R, L) NDarray of scalars
        The characteristic strain of the L loudest single sources at each frequency.
    hc_bg : (F, R) NDarray of scalars
        Characteristic strain of the GWB.
    sspar : (3, F, R, L) NDarray of scalars
        Astrophysical parametes of each loud single sources,
        for each frequency and realization.
        Returned only if params = True.
    bgpar : (3, F, R) NDarray of scalars
        Average effective binary astrophysical parameters for background
        sources at each frequency and realization,
        Returned only if params = True.
    """

    # Frequency bin midpoints
    foo = edges[-1]                   #: should be observer-frame orbital-frequencies
    df = np.diff(foo)                 #: frequency bin widths
    fc = kale.utils.midpoints(foo)    #: use frequency-bin centers for strain (more accurate!)
    F = len(fc)                       #: number of frequencies

    # All other bin midpoints
    mt = kale.utils.midpoints(edges[0]) #: total mass
    mr = kale.utils.midpoints(edges[1]) #: mass ratio
    rz = kale.utils.midpoints(edges[2]) #: redshift


    # --- Chirp Masses --- in shape (M, Q)
    cmass = utils.chirp_mass_mtmr(mt[:,np.newaxis], mr[np.newaxis,:])

    # --- Comoving Distances --- in shape (Z)
    cdist = holo.cosmo.comoving_distance(rz).cgs.value

    # --- Rest Frame Frequencies --- in shape (Z, F)
    rfreq = holo.utils.frst_from_fobs(fc[np.newaxis,:], rz[:,np.newaxis])

    # --- Source Strain Amplitude --- in shape (M, Q, Z, F)
    hsamp = utils.gw_strain_source(cmass[:,:,np.newaxis,np.newaxis],
                                   cdist[np.newaxis,np.newaxis,:,np.newaxis],
                                   rfreq[np.newaxis,np.newaxis,:,:])
    # hsfdf = hsamp^2 * f/df
    h2fdf = hsamp**2 * (fc[np.newaxis, np.newaxis, np.newaxis,:]
                    /df[np.newaxis, np.newaxis, np.newaxis,:])
    # indices of bins sorted by h2fdf
    indices = np.argsort(-h2fdf[...,0].flatten()) # just sort for first frequency
    unraveled = np.array(np.unravel_index(indices, (len(mt),len(mr),len(rz))))
    msort = unraveled[0,:]
    qsort = unraveled[1,:]
    zsort = unraveled[2,:]

    # For multiple realizations, using cython
    if(utils.isinteger(realize)):
        if(params == True):
            # hc2ss = char strain squared of each loud single source
            # hc2bg = char strain squared of the background
            # lspar = avg parameters of loudest sources
            # bgpar = avg parameters of background
            # ssidx = indices of loud single sources
            hc2ss, hc2bg, lspar, bgpar, ssidx = \
                holo.cyutils.loudest_hc_and_par_from_sorted(number, h2fdf, realize, loudest,
                                                            mt, mr, rz, msort, qsort, zsort)
            hc_ss = np.sqrt(hc2ss) # calculate single source strain
            hc_bg = np.sqrt(hc2bg) # calculate background strain
            # calulate parameters of single sources
            sspar = np.array([mt[ssidx[0,...]], mr[ssidx[1,...]], rz[ssidx[2,...]]])
            return hc_ss, hc_bg, sspar, bgpar

        else:
            # use cython to get h_c^2 for ss and bg
            hc2ss, hc2bg = holo.cyutils.loudest_hc_from_sorted(number, h2fdf, realize, loudest,
                                                               msort, qsort, zsort)
            hc_ss = np.sqrt(hc2ss)
            hc_bg = np.sqrt(hc2bg)
            return hc_ss, hc_bg

    # OTHERWISE
    else:
        raise Exception("`realize` ({}) must be an integer!")





def loudest_by_cython(edges, number, realize, loudest, round = True, params = False):

    """ More efficient way to calculate strain from numbered
    grid integrated


    Parameters
    ----------
    edges : (4,) list of 1darrays
        A list containing the edges along each dimension.  The four dimensions correspond to
        total mass, mass ratio, redshift, and observer-frame orbital frequency.
        The length of each of the four arrays is M+1, Q+1, Z+1, F+1.
    number : (M, Q, Z, F) ndarray of scalars
        The number of binaries in each bin of parameter space.  This is calculated by integrating
        `dnum` over each bin.
    realize : int,
        Specification of how many discrete realizations to construct.
        TODO: Set up option for `bool` value, to get multiple sources without realizing. That makes no sense though.
    loudest : int
        Number of loudest single sources to separate from background.
    round : bool
        Specification of whether to discretize the sample if realize is False,
        by rounding number of binaries in each bin to integers.
        Does nothing if realize is True.


    Returns
    -------
    hc_ls : (F, R, L) NDarray of scalars
        The characteristic strain of the L loudest single sources at each frequency.
    hc_bg : (F, R) NDarray of scalars
        Characteristic strain of the GWB.
    lspar : (3, F, R) NDarray of scalars
        Average effective binary astrophysical parametes of the L loudest sources
        at each frequency, for each realization.
        Returned only if params = True.
    bgpar : (3, F, R) NDarray of scalars
        Average effective binary astrophysical parameters for background
        sources at each frequency and realization,
        Returned only if params = True.
    lsidx : (3, F, R, L) NDarray
        The M, q, and z indices of loudest single sources at each frequency of each realization.
        Example usage to get parameters of the nth loudest source: ss_masses = mt[lsidx[0,:,:,n]]

    """

    # Frequency bin midpoints
    foo = edges[-1]                   #: should be observer-frame orbital-frequencies
    df = np.diff(foo)                 #: frequency bin widths
    fc = kale.utils.midpoints(foo)    #: use frequency-bin centers for strain (more accurate!)
    F = len(fc)                       #: number of frequencies

    # All other bin midpoints
    mt = kale.utils.midpoints(edges[0]) #: total mass
    mr = kale.utils.midpoints(edges[1]) #: mass ratio
    rz = kale.utils.midpoints(edges[2]) #: redshift


    # --- Chirp Masses --- in shape (M, Q)
    cmass = utils.chirp_mass_mtmr(mt[:,np.newaxis], mr[np.newaxis,:])

    # --- Comoving Distances --- in shape (Z)
    cdist = holo.cosmo.comoving_distance(rz).cgs.value

    # --- Rest Frame Frequencies --- in shape (Z, F)
    rfreq = holo.utils.frst_from_fobs(fc[np.newaxis,:], rz[:,np.newaxis])

    # --- Source Strain Amplitude --- in shape (M, Q, Z, F)
    hsamp = utils.gw_strain_source(cmass[:,:,np.newaxis,np.newaxis],
                                   cdist[np.newaxis,np.newaxis,:,np.newaxis],
                                   rfreq[np.newaxis,np.newaxis,:,:])
    # hsfdf = hsamp^2 * f/df
    h2fdf = hsamp**2 * (fc[np.newaxis, np.newaxis, np.newaxis,:]
                    /df[np.newaxis, np.newaxis, np.newaxis,:])

    # indices of bins sorted by h2fdf
    indices = np.argsort(-h2fdf[...,0].flatten()) # just sort for first frequency
    unraveled = np.array(np.unravel_index(indices, (len(mt),len(mr),len(rz))))
    msort = unraveled[0,:]
    qsort = unraveled[1,:]
    zsort = unraveled[2,:]

    # For multiple realizations, using cython
    if(utils.isinteger(realize)):
        if(params == True):
            hc2ls, hc2bg, lspar, bgpar, lsidx = \
                holo.cyutils.loudest_hc_and_par_from_sorted(number, h2fdf, realize, loudest,
                                                            mt, mr, rz, msort, qsort, zsort)
            hc_ls = np.sqrt(hc2ls)
            hc_bg = np.sqrt(hc2bg)
            return hc_ls, hc_bg, lspar, bgpar, lsidx

        else:
            # use cython to get h_c^2 for ss and bg
            hc2ls, hc2bg = holo.cyutils.loudest_hc_from_sorted(number, h2fdf, realize, loudest,
                                                               msort, qsort, zsort)
            hc_ls = np.sqrt(hc2ls)
            hc_bg = np.sqrt(hc2bg)
            return hc_ls, hc_bg

    # OTHERWISE
    else:
        raise Exception("`realize` ({}) must be an integer!")



def ss_by_cdefs(edges, number, realize, round = True, params = False):

    """ More efficient way to calculate strain from numbered
    grid integrated


    Parameters
    ----------
    edges : (4,) list of 1darrays
        A list containing the edges along each dimension.  The four dimensions correspond to
        total mass, mass ratio, redshift, and observer-frame orbital frequency.
        The length of each of the four arrays is M+1, Q+1, Z+1, F+1.
    number : (M, Q, Z, F) ndarray of scalars
        The number of binaries in each bin of parameter space.  This is calculated by integrating
        `dnum` over each bin.
    realize : bool or int,
        Specification of how to construct one or more discrete realizations.
        If a `bool` value, then whether or not to construct a realization.
        If a `int` value, then how many discrete realizations to construct.
    round : bool
        Specification of whether to discretize the sample if realize is False,
        by rounding number of binaries in each bin to integers.
        Does nothing if realize is True.


    Returns
    -------
    hc_bg : (F, R) NDarray of scalars
        Characteristic strain of the GWB.
        just (F,) if realize = True or False.
    hc_ss : (F, R) NDarray of scalars
        The characteristic strain of the loudest single source at each frequency.
        just (F,) if realize = True or False.
    ssidx : (3, F, R) NDarray
        The indices of loudest single sources at each frequency of each realization
        in the format: [[M indices], [q indices], [z indices], [f indices], [r indices]]
        just (3,F,) if realize = True or False.
    hsamp : (M, Q, Z, F) NDarray
        Strain amplitude of a single source in every bin (regardless of if that bin
        actually has any sources.)
    bgpar : (3, F, R) NDarray of scalars
        Average effective binary astrophysical parameters for background
        sources at each frequency and realization, returned only if
        params = True.
    sspar : (3, F, R) NDarray of scalars
        Astrophysical parametes of single sources at each frequency
        for each realizations, returned only if params = True.

    """

    # Frequency bin midpoints
    foo = edges[-1]                   #: should be observer-frame orbital-frequencies
    df = np.diff(foo)                 #: frequency bin widths
    fc = kale.utils.midpoints(foo)    #: use frequency-bin centers for strain (more accurate!)
    F = len(fc)                       #: number of frequencies

    # All other bin midpoints
    mt = kale.utils.midpoints(edges[0]) #: total mass
    mr = kale.utils.midpoints(edges[1]) #: mass ratio
    rz = kale.utils.midpoints(edges[2]) #: redshift


    # --- Chirp Masses --- in shape (M, Q)
    cmass = utils.chirp_mass_mtmr(mt[:,np.newaxis], mr[np.newaxis,:])

    # --- Comoving Distances --- in shape (Z)
    cdist = holo.cosmo.comoving_distance(rz).cgs.value

    # --- Rest Frame Frequencies --- in shape (Z, F)
    rfreq = holo.utils.frst_from_fobs(fc[np.newaxis,:], rz[:,np.newaxis])

    # --- Source Strain Amplitude --- in shape (M, Q, Z, F)
    hsamp = utils.gw_strain_source(cmass[:,:,np.newaxis,np.newaxis],
                                   cdist[np.newaxis,np.newaxis,:,np.newaxis],
                                   rfreq[np.newaxis,np.newaxis,:,:])
    # hsfdf = hsamp^2 * f/df
    h2fdf = hsamp**2 * (fc[np.newaxis, np.newaxis, np.newaxis,:]
                    /df[np.newaxis, np.newaxis, np.newaxis,:])

    # For multiple realizations, using cython
    if(utils.isinteger(realize)):
        if(params == True):
            hc2ss, hc2bg, ssidx, bgpar, sspar = \
                holo.cyutils.ss_bg_hc_and_par(number, h2fdf, realize, mt, mr, rz)
            hc_ss = np.sqrt(hc2ss)
            hc_bg = np.sqrt(hc2bg)
            return hc_bg, hc_ss, ssidx, hsamp, bgpar, sspar

        else:
            # use cython to get h_c^2 for ss and bg
            hc2ss, hc2bg, ssidx = holo.cyutils.ss_bg_hc(number, h2fdf, realize)
            hc_ss = np.sqrt(hc2ss)
            hc_bg = np.sqrt(hc2bg)
            return hc_bg, hc_ss, ssidx, hsamp

    # OTHERWISE
    elif(realize==False):
        if (round == True):
            bgnum = np.copy(np.floor(number).astype(np.int64))
            assert (np.all(bgnum%1 == 0)), 'non integer numbers found with round=True'
            assert (np.all(bgnum >= 0)), 'negative numbers found with round=True'
        else:
            bgnum = np.copy(number)
            warnings.warn('Number grid used for single source calculation.')
    elif(realize == True):
        bgnum = np.random.poisson(number)
        assert (np.all(bgnum%1 ==0)), 'nonzero numbers found with realize=True'
    else:
        raise Exception("`realize` ({}) must be one of {{True, False, integer}}!"\
                            .format(realize))

    # --- Single Source Characteristic Strain --- in shape (F,)
    hc_ss = np.sqrt(np.amax(h2fdf, axis=(0,1,2)))

    # --- Indices of Loudest Bin --- in shape (3, F)
    # looks like [[m1,m2,..mF], [q1,q2,...qF], [z1,z2,...zF]]
    htemp = np.copy(hsamp) #htemp is s same as hsamp except 0 where bgnum=0
    htemp[(bgnum==0)] = 0
    shape = htemp.shape
    # print('shape', shape)
    newshape = (shape[0]*shape[1]*shape[2], shape[3])
    htemp = htemp.reshape(newshape) # change hsamp to shape (M*Q*Z, F)
    argmax = np.argmax(htemp, axis=0) # max at each frequency
    # print('argmax', argmax)
    ssidx = np.array(np.unravel_index(argmax, shape[:-1])) # unravel indices
    # print('ssidx', ssidx)

    # --- Background Characteristic Strain --- in shape (F,)
    hc_bg = np.sqrt(np.sum(bgnum*h2fdf, axis=(0,1,2)))
    return hc_bg, hc_ss, ssidx, hsamp


def ss_by_ndars(edges, number, realize, round = True):

    """ More efficient way to calculate strain from numbered
    grid integrated


    Parameters
    ----------
    edges : (4,) list of 1darrays
        A list containing the edges along each dimension.  The four dimensions correspond to
        total mass, mass ratio, redshift, and observer-frame orbital frequency.
        The length of each of the four arrays is M+1, Q+1, Z+1, F+1.
    number : (M, Q, Z, F) ndarray of scalars
        The number of binaries in each bin of parameter space.  This is calculated by integrating
        `dnum` over each bin.
    realize : bool or int,
        Specification of how to construct one or more discrete realizations.
        If a `bool` value, then whether or not to construct a realization.
        If a `int` value, then how many discrete realizations to construct.
    round : bool
        Specification of whether to discretize the sample if realize is False,
        by rounding number of binaries in each bin to integers.
        Does nothing if realize is True.
    ss : bool
        Whether or not to separate the loudest single source in each frequency bin.
    sum : bool
        Whether or not to sum the strain at a given frequency over all bins.
    print_test : bool
        Whether or not to print variable as they are calculated, for dev purposes.


    Returns
    -------
    hc_bg : (F, R) ndarray of scalars
        Characteristic strain of the GWB.
        If realize = True or False: R is 1
        If realize is an integer, R=realize
    hc_ss : (F, R) ndarray of scalars
        The characteristic strain of the loudest single source at each frequency.
    hsamp : (M, Q, Z, F, R) ndarray
        Strain amplitude of a single source in every bin (but equal to zero
        if the number in that bin is 0)
    ssidx : (5, F, R) ndarray
        The indices of loudest single sources at each frequency of each realization
        in the format: [[M indices], [q indices], [z indices], [f indices], [r indices]]
    hsmax : (F, R) array of scalars
        The maximum single source strain amplitude at each frequency.
    bgnum : (M, Q, Z, F, R) ndarray
        The number of binaries in each bin after the loudest single source
        at each frequency is subtracted out.


    In the unlikely scenario that there are two equal hsmaxes
    (at same OR dif frequencies), ssidx calculation will go wrong
    Could avoid this by using argwhere for each f_idx column separately.
    Or TODO implement some kind of check to see if any argwheres return multiple
    values for that hsmax and raises a warning/assertion error

    """

    # Frequency bin midpoints
    foo = edges[-1]                   #: should be observer-frame orbital-frequencies
    df = np.diff(foo)                 #: frequency bin widths
    fc = kale.utils.midpoints(foo)    #: use frequency-bin centers for strain (more accurate!)
    F = len(fc)                       #: number of frequencies

    # All other bin midpoints
    mt = kale.utils.midpoints(edges[0]) #: total mass
    mr = kale.utils.midpoints(edges[1]) #: mass ratio
    rz = kale.utils.midpoints(edges[2]) #: redshift


    # --- Chirp Masses ---
    # to get chirp mass in shape (M, Q) we need
    # mt in shape (M, 1)
    # mr in shape (1, Q)
    cmass = utils.chirp_mass_mtmr(mt[:,np.newaxis], mr[np.newaxis,:])

    # --- Comoving Distances ---
    # to get cdist in shape (Z) we need
    # rz in shape (Z)
    cdist = holo.cosmo.comoving_distance(rz).cgs.value

    # --- Rest Frame Frequencies ---
    # to get rest freqs in shape (Z, F) we need
    # rz in shape (Z, 1)
    # fc in shape (1, F)
    rfreq = holo.utils.frst_from_fobs(fc[np.newaxis,:], rz[:,np.newaxis])

    # --- Source Strain Amplitude ---
    # to get hs amplitude in shape (M, Q, Z, F) we need
    # cmass in shape (M, Q, 1, 1) from (M, Q)
    # cdist in shape (1, 1, Z, 1) from (Z)
    # rfreq in shape (1, 1, Z, F) from (Z, F)
    hsamp = utils.gw_strain_source(cmass[:,:,np.newaxis,np.newaxis],
                                   cdist[np.newaxis,np.newaxis,:,np.newaxis],
                                   rfreq[np.newaxis,np.newaxis,:,:])


    ########## HERE'S WHERE THINGS CHANGE FOR SS ###############

    # --------------- Single Sources ------------------
    ##### 0) Round and/or realize so numbers are all integers

    # --- Background Number ---
    # in shape (M, Q, Z, F, R)
    if(realize == False):
        R=1
        if (round == True):
            bgnum = np.copy(np.floor(number).astype(np.int64))[...,np.newaxis]
            assert (np.all(bgnum%1 == 0)), 'non integer numbers found with round=True'
            assert (np.all(bgnum >= 0)), 'negative numbers found with round=True'
        else:
            bgnum = np.copy(number)[...,np.newaxis]
    elif(realize == True):
        R=1
        bgnum = np.random.poisson(number[...,np.newaxis],
                                size = (number.shape + (R,)))
        assert (np.all(bgnum%1 ==0)), 'non integer numbers found with realize=True'
    elif(utils.isinteger(realize)):
        R=realize
        bgnum = np.random.poisson(number[...,np.newaxis],
                                size = (number.shape + (R,)))
    else:
        print("`realize` ({}) must be one of {{True, False, integer}}!"\
            .format(realize))

    #### 1) Identify the loudest (max hs) single source in a bin with N>0
    # NOTE this requires hsamp for every realization, might be slow/big
    # --- Bin Strain Amplitudes ---
    # to get hsamp in shape (M, Q, Z, F, R)
    hsamp = np.repeat(hsamp[...,np.newaxis], R, axis=4)
    hsamp[(bgnum==0)] = 0 #set hs=0 if number=0
    # hsamp[bgnum==0] = 0


    # --- Single Source Strain Amplitude At Each Frequency ---
    # to get max strain in shape (F) we need
    # hsamp in shape (M, Q, Z, F), search over first 3 axes
    hsmax = np.amax(hsamp, axis=(0,1,2)) #find max hs at each frequency


    #### 2) Record the indices and strain of that single source

    # --- Indices of Loudest Bin ---
    # Shape (5, F, R),
    # where ssidx[:,F,R] = m, q, z, f, r indices for loudest source at F, R
    shape = hsamp.shape
    newshape = (shape[0]*shape[1]*shape[2], shape[3], shape[4])
    hsamp = hsamp.reshape(newshape) # change hsamp to shape (M*Q*Z, F, R)
    argmax = np.argmax(hsamp, axis=0) # max at each frequency
    hsamp = hsamp.reshape(shape) # restore hsamp shape to (M, Q, Z, F, R)
    mqz = np.array(np.unravel_index(argmax, shape[:-2]))         # shape (3, F, R)
    # add frequency indices
    f_ids = np.arange(len(mqz[0])).astype(int) # shape (F)
    f_ids = np.repeat(f_ids, R).reshape(F, R)[np.newaxis,...]    # shape (1, F, R)
    ssidx = np.append(mqz, f_ids, axis=0)                        # shape (4, F, R)
    # add realization indices
    r_ids = np.arange(R).astype(int)                     # shape (R)
    r_ids = np.tile(r_ids, F).reshape(F,R)[np.newaxis,...]       # shape (1, F, R)
    ssidx = np.append(ssidx, r_ids, axis=0)                      # shape (5, F, R)



    #### 3) Subtract 1 from the number in that source's bin

    # --- Background Number ---
    # bgnum = subtract_from_number(bgnum, ssidx) # Find a better way to do this!
    if np.any( bgnum[(hsamp == hsmax)] <=0):
        raise Exception("bgnum <= found at hsmax")
    if np.any( hsamp[(hsamp == hsmax)] <=0):
        raise Exception("hsamp <=0 found at hsmax")
    if np.any(hsmax<=0):
        raise Exception("hsmax <=0 found")

    # print('bgnum stats:\n', holo.utils.stats(bgnum))
    # print('bgnum[hsamp==hsmax] stats:\n', holo.utils.stats(bgnum[(hsamp == hsmax)]))
    # print('ssidx\n', ssidx, ssidx.shape)
    bgnum[ssidx[0], ssidx[1], ssidx[2], ssidx[3], ssidx[4]] -= 1
    # print('\nafter subtraction')
    # print('bgnum stats:\n', holo.utils.stats(bgnum))
    # print('bgnum[hsamp==hsmax] stats:\n', holo.utils.stats(bgnum[(hsamp == hsmax)]))

    assert np.all(bgnum>=0), f"bgnum contains negative values at: {np.where(bgnum<0)}"


    #### 4) Calculate single source characteristic strain (hc)

    # --- Single Source Characteristic Strain ---
    # to get ss char strain in shape [F,R] need
    # fc in shape (F,1)
    # df in shape (F,1)
    # hsmax in shape (F,R)
    hc_ss = np.sqrt(hsmax**2 * (fc[:,np.newaxis]/df[:,np.newaxis]))



    #### 5) Calculate the background with the new number

    # --- Background Characteristic Strain Squared ---
    # to get characteristic strain in shape (M, Q, Z, F, R) we need
    # hsamp in shape (M, Q, Z, F, R)
    # fc in shape (1, 1, 1, F, 1)
    hchar = hsamp**2 * (fc[np.newaxis, np.newaxis, np.newaxis,:,np.newaxis]
                        /df[np.newaxis, np.newaxis, np.newaxis,:,np.newaxis])
    hchar *= bgnum



    # --- Background Characteristic Strain  ---
    # hc_bg in shape (F, R)
    hchar = np.sum(hchar, axis=(0, 1, 2)) # sum over all bins at a given frequency
    hc_bg = np.sqrt(hchar) # sqrt

    return hc_bg, hc_ss, hsamp, ssidx, hsmax, bgnum

def h2fdf(edges):

    """ More efficient way to calculate strain from numbered
    grid integrated


    Parameters
    ----------
    edges : (4,) list of 1darrays
        A list containing the edges along each dimension.  The four dimensions correspond to
        total mass, mass ratio, redshift, and observer-frame orbital frequency.
        The length of each of the four arrays is M+1, Q+1, Z+1, F+1.


    Returns
    -------
    h2fdf : (4,) ndarray of scalars
        strain amplitude squared x frequency x frequency bin width for a single
        source in each bing

    """

    # Frequency bin midpoints
    foo = edges[-1]                   #: should be observer-frame orbital-frequencies
    df = np.diff(foo)                 #: frequency bin widths
    fc = kale.utils.midpoints(foo)    #: use frequency-bin centers for strain (more accurate!)
    F = len(fc)                       #: number of frequencies

    # All other bin midpoints
    mt = kale.utils.midpoints(edges[0]) #: total mass
    mr = kale.utils.midpoints(edges[1]) #: mass ratio
    rz = kale.utils.midpoints(edges[2]) #: redshift


    # --- Chirp Masses ---
    # to get chirp mass in shape (M, Q) we need
    # mt in shape (M, 1)
    # mr in shape (1, Q)
    cmass = utils.chirp_mass_mtmr(mt[:,np.newaxis], mr[np.newaxis,:])

    # --- Comoving Distances ---
    # to get cdist in shape (Z) we need
    # rz in shape (Z)
    cdist = holo.cosmo.comoving_distance(rz).cgs.value

    # --- Rest Frame Frequencies ---
    # to get rest freqs in shape (Z, F) we need
    # rz in shape (Z, 1)
    # fc in shape (1, F)
    rfreq = holo.utils.frst_from_fobs(fc[np.newaxis,:], rz[:,np.newaxis])

    # --- Source Strain Amplitude ---
    # to get hs amplitude in shape (M, Q, Z, F) we need
    # cmass in shape (M, Q, 1, 1) from (M, Q)
    # cdist in shape (1, 1, Z, 1) from (Z)
    # rfreq in shape (1, 1, Z, F) from (Z, F)
    hsamp = utils.gw_strain_source(cmass[:,:,np.newaxis,np.newaxis],
                                   cdist[np.newaxis,np.newaxis,:,np.newaxis],
                                   rfreq[np.newaxis,np.newaxis,:,:])


    # --- Pass to Cython Function ---
    # hsfdf = hsamp^2 * f/df
    # hc_ss = max hsfdf at each freq
    # hc_bg = sum(hsfdf * num in bin)
    h2fdf = hsamp**2 * (fc[np.newaxis, np.newaxis, np.newaxis,:]
                        /df[np.newaxis, np.newaxis, np.newaxis,:])
    return h2fdf


def all_sspars(fobs_gw_cents, sspar):
    """ Calculate all single source parameters incl.
    ['mtot' 'mrat' 'redz_init' 'redz_final' 'dcom_final' 'sepa_final' 'angs_final']

    Parameters
    ----------
    fobs_gw_cents : (F,) 1Darray of scalars
        Observed gw frequency bin centers.
    sspar : (4, F,R,L) NDarray
        Single source parameters as calculated by ss_gws_redz().
        Includes mtot, mrat, redz_init, redz_final.

    Returns
    -------
    sspar_all : (7,F,R,L) NDarray
        All single source parameters, corresponding to those in bgpar as calculated by ss_gws_redz().
        Includes mtot, mrat, redz_init, redz_final, dcom_final, sepa_final (cm), and angs_final.
    """
    mtot = sspar[0,:,:] # (F,R,L) in g
    mrat = sspar[1,:,:] # (F,R,L) dimensionless
    redz_init = sspar[2,:,:]  # (F,R,L) dimensionless
    redz_final = sspar[3,:,:]  # (F,R,L) dimensionless
    dcom_final = holo.cosmo.comoving_distance(redz_final).to('cm').value # (F,R,L) in cm
    dcom_final[dcom_final<0] = np.nan
    fobs_orb_cents = fobs_gw_cents/2.0  # (F,)
    frst_orb_cents = utils.frst_from_fobs(fobs_orb_cents[:,np.newaxis,np.newaxis], redz_final) #  (F,R,L) in Hz
    sepa = utils.kepler_sepa_from_freq(mtot, frst_orb_cents) #  (F,R,L) in cm
    angs = utils.angs_from_sepa(sepa, dcom_final, redz_final)  # (F,R,L) in cm
    sspar_all = np.array([mtot, mrat, redz_init, redz_final, dcom_final, sepa, angs])
    return sspar_all


def parameters_from_indices(edges, ssidx):
    """
    Finds the mass, ratio, redshift of the loudest single sources at each
    frequency given their indices and edges.

    Parameters
    -----------
    edges : (4,) list of 1darrays
        A list containing the edges along each dimension.  The four dimensions correspond to
        total mass, mass ratio, redshift, and observer-frame orbital frequency.
        The length of each of the four arrays is M+1, Q+1, Z+1, F+1.
    ssidx : (3, F, R) or (3,F) ndarray
        The indices of loudest single sources at each frequency of each realization
        in the format: [[M indices], [q indices], [z indices], [f indices], [r indices]]


    Returns
    ----------
    m_arr : (F,) or (F,R) Ndarray of scalars
    q_arr : (F,) or (F,R) Ndarray of scalars
    z_arr : (F,) or (F,R) Ndarray of scalars
    f_arr : (F,) or (F,R) Ndarray of scalars

    """
    # Frequency bin midpoints
    foo = edges[-1]                   #: should be observer-frame orbital-frequencies
    fc = kale.utils.midpoints(foo)    #: use frequency-bin centers for strain (more accurate!)

    # All other bin midpoints
    mt = kale.utils.midpoints(edges[0]) #: total mass
    mr = kale.utils.midpoints(edges[1]) #: mass ratio
    rz = kale.utils.midpoints(edges[2]) #: redshift

    m_arr = mt[ssidx[0,...]]
    q_arr = mr[ssidx[1,...]]
    z_arr = rz[ssidx[2,...]]
    f_arr = np.reshape(np.repeat(fc, len(ssidx[0,0])), m_arr.shape)

    return m_arr, q_arr, z_arr, f_arr


def ss_by_loops(edges, number, realize=False, round=True,  print_test = False):

    """ Inefficient way to calculate strain from numbered
    grid integrated

    Parameters
    ----------
    edges : (4,) list of 1darrays
        A list containing the edges along each dimension.  The four dimensions correspond to
        total mass, mass ratio, redshift, and observer-frame orbital frequency.
        The length of each of the four arrays is M, Q, Z, F.
    number : (M, Q, Z, F) ndarray
        The number of binaries in each bin of parameter space.  This is calculated by integrating
        `dnum` over each bin.
    realize : bool or int,
        Specification of how to construct one or more discrete realizations.
        If a `bool` value, then whether or not to construct a realization.
        If a `int` value, then how many discrete realizations to construct.
    round : bool
        Specification of whether to discretize the sample if realize is False,
        by rounding number of binaries in each bin to integers.
    print_test : bool
        Whether or not to print variable as they are calculated, for dev purposes.


    Returns
    -------
    hc_bg : (F, R) ndarray
        Characteristic strain of the GWB.
        R=1 if realize is True or False,
        R=realize if realize is an integer
    hc_ss : (F, R) ndarray
        The characteristic strain of the loudest single source at each frequency,
        for each realization. (R=1 if realize is True or False, otherwise R=realize)
    sspar : (F, 3, R) ndarray
        The parameters (M, q, and z) of the loudest single source at each frequency.
    ssidx : (F, 3, R) ndarray
        The indices (m_idx, q_idx, and z_idx) of the parameters of the loudest single
        source's bin, at each frequency.
    maxhs : (F, R) ndarray
        The maximum single source strain amplitude at each frequency.
    bgnum : (M, Q, Z, F, R)
        The number of binaries in each bin after the loudest single source
        at each frequency is subtracted out.
        R=1 if realize is True or False,
        R=realize if realize is an integer

    """
    if(print_test):
        print('INPUTS: edges:', len(edges), '\n', edges,
        '\nINPUTS:number:', number.shape, '\n', number,'\n')

    # Frequency bin midpoints
    foo = edges[-1]                   #: should be observer-frame orbital-frequencies
    df = np.diff(foo)                 #: frequency bin widths
    fc = kale.utils.midpoints(foo)    #: use frequency-bin centers for strain (more accurate!)

    # All other bin midpoints
    mt = kale.utils.midpoints(edges[0]) #: total mass
    mr = kale.utils.midpoints(edges[1]) #: mass ratio
    rz = kale.utils.midpoints(edges[2]) #: redshift



    # new number array
    if(realize == False):
        reals = 1
        if(round == True):
            bgnum = np.copy(np.floor(number).astype(np.int64))[...,np.newaxis]
            if(print_test):
                print('noninteger bgnum values:', bgnum[(bgnum%1 !=0)])
        else:
            bgnum = np.copy(number)[...,np.newaxis]
    elif(realize == True):
        reals = 1
        bgnum = np.empty(number.shape + (reals,))
    elif(utils.isinteger(realize)):
        reals = realize
        bgnum = np.empty(number.shape + (reals,))
    else:
        raise Exception("`realize` ({}) must be one of {{True, False, integer}}!"\
                            .format(realize))

    # GW background characteristic strain
    hc_bg = np.empty_like(bgnum, dtype=float)

    # for single sources, make a grid with shape
    # (f, 3)
    # params of loudest bin with number>=1
    # shape (f,3,R) for 3 params
    sspar = np.empty((len(fc), 3, reals))
    # param indices of loudest bin with number>=1
    # shape (f,3,R) for 3 params
    ssidx = np.empty((len(fc), 3, reals))
    # max hs at each frequency
    maxhs = np.zeros((len(fc), reals))
    # (max)  single source characteristic strain at each frequency
    hc_ss = np.zeros((len(fc), reals))


    # --------------- Single Sources ------------------
    # 0) Round or realize so numbers are all integers
    # 1) Identify the loudest (max hs) single source in a bin with N>0
    # 2) Record the parameters, parameter indices, and strain
    #  of that single source
    # 3) Subtract 1 from the number in that source's bin,
    # 4) Calculate single source characteristic strain (hc)
    # 5) Calculate the background with the new number

    for m_idx in range(len(mt)):
        for q_idx in range(len(mr)):
            cmass = holo.utils.chirp_mass_mtmr(mt[m_idx], mr[q_idx])
            for z_idx in range(len(rz)):
                cdist = holo.cosmo.comoving_distance(rz[z_idx]).cgs.value

                # print M, q, z, M_c, d_c
                if(print_test):
                    print('BIN mt=%.2e, mr=%.2e, rz=%.2e' %
                        (mt[m_idx], mr[q_idx], rz[z_idx]))
                    print('\t m_c = %.2e, d_c = %.2e'
                        % (cmass, cdist))

                # check if loudest source in any bin
                for f_idx in range(len(fc)):
                    rfreq = holo.utils.frst_from_fobs(fc[f_idx], rz[z_idx])
                    # hs of a source in that bin
                    hs_mqzf = utils.gw_strain_source(cmass, cdist, rfreq)

                    for r_idx in range(reals):
                        # Poisson sample to get number in that bin if realizing
                        # (otherwise it was already set above by copying or rounding.)
                        if(utils.isinteger(realize) or realize==True):
                            bgnum[m_idx, q_idx, z_idx, f_idx, r_idx] \
                                = np.random.poisson(number[m_idx,q_idx,z_idx,f_idx])
                        # 1) IF LOUDEST
                        # check if loudest hs at that
                        # frequency and contains binaries

                        if(hs_mqzf>maxhs[f_idx, r_idx] and
                            bgnum[m_idx, q_idx, z_idx, f_idx, r_idx]>0):
                            if(bgnum[m_idx, q_idx, z_idx, f_idx, r_idx]<1):
                                print('number<1 used', bgnum[m_idx, q_idx, z_idx, f_idx, r_idx])  #DELETE
                            # 2) If so, RECORD:
                            # parameters M, q, z
                            sspar[f_idx,:,r_idx] = np.array([mt[m_idx], mr[q_idx],
                                                        rz[z_idx]])
                            # parameter indices
                            ssidx[f_idx,:,r_idx] = np.array([m_idx, q_idx, z_idx])
                            # new max strain
                            maxhs[f_idx,r_idx] = hs_mqzf


    # 3) SUBTRACT 1
    # from bin with loudest source at each frequency
    # can do this using the index of loudest, ssidx
    # recall ssidx has shape [3, F]
    # and = [(m_idx,q_idx,z_idx), fc],
    for f_idx in range(len(fc)):
        for r_idx in range(reals):
            bgnum[int(ssidx[f_idx,0,r_idx]), int(ssidx[f_idx,1,r_idx]), int(ssidx[f_idx,2,r_idx]),
                f_idx, r_idx] -= 1

            # 4) CALCULATE
            # single source characteristic strain
            hc_ss[f_idx, r_idx] = np.sqrt(maxhs[f_idx, r_idx]**2 * (fc[f_idx]/df[f_idx]))

    # CHECK no numbers should be <0
    if(np.any(bgnum<0)):
        error_index = np.where(bgnum<0)
        print('number<0 found at (M,q,z,f) =', error_index)


    # 5)
    # ----------------- Calculate Background Strains --------------------
    # then we can go back in and calculate characteristic strains
    # NOTE: could make this faster by saving rfreq and hs values from above
    # instead of recalculating
    for m_idx in range(len(mt)):
        for q_idx in range(len(mr)):
            cmass = holo.utils.chirp_mass_mtmr(mt[m_idx], mr[q_idx])
            for z_idx in range(len(rz)):
                cdist = holo.cosmo.comoving_distance(rz[z_idx]).cgs.value
                for f_idx in range(len(fc)):
                    rfreq = holo.utils.frst_from_fobs(fc[f_idx], rz[z_idx])
                    hs_mqzf = utils.gw_strain_source(cmass, cdist, rfreq)
                    hc_dlnf = hs_mqzf**2 * (fc[f_idx]/df[f_idx])
                    for r_idx in range(reals):
                        hc_bg[m_idx, q_idx, z_idx, f_idx, r_idx] = np.sqrt(hc_dlnf
                                        * bgnum[m_idx, q_idx, z_idx, f_idx, r_idx])

    # sum over all bins at a given frequency and realization
    hc_bg = np.sqrt(np.sum(hc_bg**2, axis=(0, 1, 2)))


    return hc_bg, hc_ss, sspar, ssidx, maxhs, bgnum


def gws_by_ndars(edges, number, realize, round = True, sum = True, print_test = False):

    """ More efficient way to calculate strain from numbered
    grid integrated

    Parameters
    ----------
    edges : (4,) list of 1darrays
        A list containing the edges along each dimension.  The four dimensions
        correspond to total mass, mass ratio, redshift, and observer-frame orbital
        frequency. The length of each of the four arrays is M, Q, Z, F.
    number : (M, Q, Z, F) ndarray
        The number of binaries in each bin of parameter space.  This is calculated
        by integrating `dnum` over each bin.
    realize : bool or int
        Specification of how to construct one or more discrete realizations.
        If a `bool` value, then whether or not to construct a realization.
        If an `int` value, then how many discrete realizations to construct.
    round : bool
        Specification of whether to discretize the sample if realize is False,
        by rounding number of binaries in each bin to integers. This has no impact
        if realize is true.
        NOTE: should add a warning if round and realize are both True
    sum : bool
        Whether or not to sum the strain at a given frequency over all bins.
    print_test : bool
        Whether or not to print variable as they are calculated, for dev purposes.


    Returns
    -------
    hchar : ndarray
        Characteristic strain of the GWB.
        The shape depends on whether realize is an integer or not
        realize = True or False, sum = False: shape is (M, Q, Z, F)
        realize = True or False, sum = True: shape is (F)
        realize = R, sum = False: shape is  (M, Q, Z, F, R)
        realize = R, sum = True: shape is  (F, R)

    """

    if(print_test):
        print('INPUTS: edges:', len(edges), # '\n', edges,
        '\nINPUTS:number:', number.shape, '\n', number,'\n')

    # Frequency bin midpoints
    foo = edges[-1]                   #: should be observer-frame orbital-frequencies
    df = np.diff(foo)                 #: frequency bin widths
    fc = kale.utils.midpoints(foo)    #: use frequency-bin centers for strain (more accurate!)

    # All other bin midpoints
    mt = kale.utils.midpoints(edges[0]) #: total mass
    mr = kale.utils.midpoints(edges[1]) #: mass ratio
    rz = kale.utils.midpoints(edges[2]) #: redshift


    # --- Chirp Masses ---
    # to get chirp mass in shape (M, Q) we need
    # mt in shape (M, 1)
    # mr in shape (1, Q)
    cmass = utils.chirp_mass_mtmr(mt[:,np.newaxis], mr[np.newaxis,:])
    if(print_test):
        print('cmass:', cmass.shape, '\n', cmass)

    # --- Comoving Distances ---
    # to get cdist in shape (Z) we need
    # rz in shape (Z)
    cdist = holo.cosmo.comoving_distance(rz).cgs.value
    if(print_test):
        print('cdist:', cdist.shape, '\n', cdist)

    # --- Rest Frame Frequencies ---
    # to get rest freqs in shape (Z, F) we need
    # rz in shape (Z, 1)
    # fc in shape (1, F)
    rfreq = holo.utils.frst_from_fobs(fc[np.newaxis,:], rz[:,np.newaxis])
    if(print_test):
        print('rfreq:', rfreq.shape, '\n', rfreq)

    # --- Source Strain Amplitude ---
    # to get hs amplitude in shape (M, Q, Z, F) we need
    # cmass in shape (M, Q, 1, 1) from (M, Q)
    # cdist in shape (1, 1, Z, 1) from (Z)
    # rfreq in shape (1, 1, Z, F) from (Z, F)
    hsamp = utils.gw_strain_source(cmass[:,:,np.newaxis,np.newaxis],
                                   cdist[np.newaxis,np.newaxis,:,np.newaxis],
                                   rfreq[np.newaxis,np.newaxis,:,:])
    if(print_test):
        print('hsamp', hsamp.shape, '\n', hsamp)

    # --- Characteristic Strain Squared ---
    # to get characteristic strain in shape (M, Q, Z, F) we need
    # hsamp in shape (M, Q, Z, F)
    # fc in shape (1, 1, 1, F)
    hchar = hsamp**2 * (fc[np.newaxis, np.newaxis, np.newaxis,:]
                        /df[np.newaxis, np.newaxis, np.newaxis,:])

    # Sample:
    if(realize == False):
        # without sampling, want strain in shape (M, Q, Z, F)
        if(round):
            # discretize by rounding number down to nearest integer
            hchar *= np.floor(number).astype(int)
        else:
            # keep non-integer values
            hchar *= number

    if(realize == True):
        # with a single sample, want strain in shape (M, Q, Z, F)
        hchar *= np.random.poisson(number)

    if(utils.isinteger(realize)):
        # with R realizations,
        # to get strain in shape (M, Q, Z, F, R) we need
        # hchar in shape(M, Q, Z, F, 1)
        # Poisson sample in shape (1, 1, 1, 1, R)
        npois = np.random.poisson(number[...,np.newaxis], size = (number.shape + (realize,)))
        if(print_test):
            print('npois', npois.shape)
        hchar = hchar[...,np.newaxis] * npois


    if(print_test):
        print('hchar', hchar.shape, '\n', hchar)


    if(sum):
        # sum over all bins at a given frequency and realization
        hchar = np.sum(hchar, axis=(0, 1, 2))
        if(print_test):
            print('hchar summed', hchar.shape, '\n', hchar)

    return np.sqrt(hchar)


def unrealized_ss_by_ndars(edges, number, realize, round = True, print_test = False):

    """ More efficient way to calculate strain from numbered
    grid integrated


    Parameters
    ----------
    edges : (4,) list of 1darrays
        A list containing the edges along each dimension.  The four dimensions correspond to
        total mass, mass ratio, redshift, and observer-frame orbital frequency.
        The length of each of the four arrays is M, Q, Z, F.
    number : (M, Q, Z, F) ndarray of scalars
        The number of binaries in each bin of parameter space.  This is calculated by integrating
        `dnum` over each bin.
    realize : bool or int,
        Specification of how to construct one or more discrete realizations.
        If a `bool` value, then whether or not to construct a realization.
        If a `int` value, then how many discrete realizations to construct.
    round : bool
        Specification of whether to discretize the sample if realize is False,
        by rounding number of binaries in each bin to integers.
        Does nothing if realize is True.
    ss : bool
        Whether or not to separate the loudest single source in each frequency bin.
    sum : bool
        Whether or not to sum the strain at a given frequency over all bins.
    print_test : bool
        Whether or not to print variable as they are calculated, for dev purposes.


    Returns
    -------
    hc_bg : ndarray
        Characteristic strain of the GWB.
        The shape depends on whether realize is an integer or not
        realize = True or False: shape is (F)
        realize = R: shape is  (F, R)
    hc_ss : (F,) array of scalars
        The characteristic strain of the loudest single source at each frequency.
    ssidx : (F, 4) ndarray
        The indices (m_idx, q_idx, z_idx, f_idx) of the parameters of the loudest single
        source's bin, at each frequency such that
        ssidx[i,0] = m_idx of the ith frequency
        ssidx[i,1] = q_idx of the ith frequency
        ssidx[i,2] = z_idx of the ith frequency
        ssidx[i,3] = f_idx of the ith frequency = i
    hsmax : (F) array of scalars
        The maximum single source strain amplitude at each frequency.
    bgnum : (M, Q, Z, F) ndarray
        The number of binaries in each bin after the loudest single source
        at each frequency is subtracted out.
    ssnew : (4, F) ndarray
        The indices of loudest single sources at each frequency in the
        format: [[M indices], [q indices], [z indices], [f indices]]



    Potential BUG: In the unlikely scenario that there are two equal hsmaxes
    (at same OR dif frequencies), ssidx calculation will go wrong
    Could avoid this by using argwhere for each f_idx column separately.
    Or TODO implement some kind of check to see if any argwheres return multiple
    values for that hsmax and raises a warning/assertion error


    TODO: Calculate sspar
    TODO: Implement realizations
    TODO: Implement not summing, or remove option
    """

    if(print_test):
        print('INPUTS: edges:', len(edges), # '\n', edges,
        '\nINPUTS:number:', number.shape, '\n', number,'\n')

    # Frequency bin midpoints
    foo = edges[-1]                   #: should be observer-frame orbital-frequencies
    df = np.diff(foo)                 #: frequency bin widths
    fc = kale.utils.midpoints(foo)    #: use frequency-bin centers for strain (more accurate!)

    # All other bin midpoints
    mt = kale.utils.midpoints(edges[0]) #: total mass
    mr = kale.utils.midpoints(edges[1]) #: mass ratio
    rz = kale.utils.midpoints(edges[2]) #: redshift


    # --- Chirp Masses ---
    # to get chirp mass in shape (M, Q) we need
    # mt in shape (M, 1)
    # mr in shape (1, Q)
    cmass = utils.chirp_mass_mtmr(mt[:,np.newaxis], mr[np.newaxis,:])
    if(print_test):
        print('cmass:', cmass.shape, '\n', cmass)

    # --- Comoving Distances ---
    # to get cdist in shape (Z) we need
    # rz in shape (Z)
    cdist = holo.cosmo.comoving_distance(rz).cgs.value
    if(print_test):
        print('cdist:', cdist.shape, '\n', cdist)

    # --- Rest Frame Frequencies ---
    # to get rest freqs in shape (Z, F) we need
    # rz in shape (Z, 1)
    # fc in shape (1, F)
    rfreq = holo.utils.frst_from_fobs(fc[np.newaxis,:], rz[:,np.newaxis])
    if(print_test):
        print('rfreq:', rfreq.shape, '\n', rfreq)

    # --- Source Strain Amplitude ---
    # to get hs amplitude in shape (M, Q, Z, F) we need
    # cmass in shape (M, Q, 1, 1) from (M, Q)
    # cdist in shape (1, 1, Z, 1) from (Z)
    # rfreq in shape (1, 1, Z, F) from (Z, F)
    hsamp = utils.gw_strain_source(cmass[:,:,np.newaxis,np.newaxis],
                                   cdist[np.newaxis,np.newaxis,:,np.newaxis],
                                   rfreq[np.newaxis,np.newaxis,:,:])
    if(print_test):
        print('hsamp', hsamp.shape, '\n', hsamp)


    ########## HERE'S WHERE THINGS CHANGE FOR SS ###############

    # --------------- Single Sources ------------------
    ##### 0) Round and/or realize so numbers are all integers
    if (round == True):
        bgnum = np.copy(np.floor(number).astype(np.int64))
        assert (np.all(bgnum%1 == 0)), 'non integer numbers found with round=True'
        assert (np.all(bgnum >= 0)), 'negative numbers found with round=True'
    else:
        bgnum = np.copy(number)
        warnings.warn('Number grid used for single source calculation.')

    if(realize == True):
        bgnum = np.random.poisson(number)
        assert (np.all(bgnum%1 ==0)), 'nonzero numbers found with realize=True'

    #### 1) Identify the loudest (max hs) single source in a bin with N>0
    hsamp[(bgnum==0)] = 0 #set hs=0 if number=0
    # hsamp[bgnum==0] = 0



    # --- Single Source Strain Amplitude At Each Frequency ---
    # to get max strain in shape (F) we need
    # hsamp in shape (M, Q, Z, F), search over first 3 axes
    hsmax = np.amax(hsamp, axis=(0,1,2)) #find max hs at each frequency

    #### 2) Record the indices and strain of that single source

    # --- Indices of Loudest Bin ---
    # Shape (F, 4), looks like
    # [[m_idx,q_idx,z_idx,0],
    #  [m_idx,q_idx,z_idx,1],
    #   ........
    #  [m_idx,q_idx,z_idx,F-2]]
    # no longer actually need this, but might be useful
    ssidx = np.argwhere(hsamp==hsmax)
    ssidx = ssidx[ssidx[:,-1].argsort()]

    # --- Indices of Loudest Bin, New Method ---
    # Shape (4, F), looks like
    # [[m1,m2,..mN], [q1,q2,...qN], [z1,z2,...zN], [0,1,...N-1]]
    # for N=F frequencies
    shape = hsamp.shape
    newshape = (shape[0]*shape[1]*shape[2], shape[3])
    hsamp = hsamp.reshape(newshape) # change hsamp to shape (M*Q*Z, F)
    argmax = np.argmax(hsamp, axis=0) # max at each frequency
    hsamp = hsamp.reshape(shape) # restore hsamp shape to (M, Q, Z, F)
    mqz = np.array(np.unravel_index(argmax, shape[:-1])) # unravel indices
    f_ids = np.linspace(0,len(mqz[0])-1,len(mqz[0])).astype(int) # frequency indices
    ssnew = np.append(mqz, f_ids).reshape(4,len(f_ids))



    ### 3) Subtract 1 from the number in that source's bin

    # --- Background Number ---
    # bgnum = subtract_from_number(bgnum, ssidx) # Find a better way to do this!
    if np.any( bgnum[(hsamp == hsmax)] <=0):
        raise Exception("bgnum <= found at hsmax")
    if np.any( hsamp[(hsamp == hsmax)] <=0):
        raise Exception("hsamp <=0 found at hsmax")
    if np.any(hsmax<=0):
        raise Exception("hsmax <=0 found")

    # print('bgnum stats:\n', holo.utils.stats(bgnum))
    # print('bgnum[hsamp==hsmax] stats:\n', holo.utils.stats(bgnum[(hsamp == hsmax)]))
    bgnum[(hsamp == hsmax)]-=1
    # NOTE keep an eye out for if hsmax is not found anywhere in hsasmp
    # could change to bgnum(np.where(hsamp==hsmax) & (bgnum >0))-=1
    # print('\nafter subtraction')
    # print('bgnum stats:\n', holo.utils.stats(bgnum))
    # print('bgnum[hsamp==hsmax] stats:\n', holo.utils.stats(bgnum[(hsamp == hsmax)]))

    assert np.all(bgnum>=0), f"bgnum contains negative values at: {np.where(bgnum<0)}"
    # if(np.any(bgnum<0)):   # alternate way to check for this error, and give index of neg number
    #         error_index = *np.where(bgnum<0)
    #         print('number<0 found at [M's], [q's], [z's], [f's]) =', error_index)



    ### 4) Calculate single source characteristic strain (hc)

    # --- Single Source Characteristic Strain ---
    # to get ss char strain in shape [F] need
    # fc in shape (F)
    # df in shape (F)
    hc_ss = np.sqrt(hsmax**2 * (fc/df))


    # --- Parameters of loudest source ---
    # NOTE: This would be useful to implement, or have separate function for

    ### 5) Calculate the background with the new number

    # --- Background Characteristic Strain Squared ---
    # to get characteristic strain in shape (M, Q, Z, F) we need
    # hsamp in shape (M, Q, Z, F)
    # fc in shape (1, 1, 1, F)
    hchar = hsamp**2 * (fc[np.newaxis, np.newaxis, np.newaxis,:]
                        /df[np.newaxis, np.newaxis, np.newaxis,:])
    if (realize==False):
        hchar *= bgnum
    else:
        raise Exception('realize not implemented yet')


    if(print_test):
        print('hchar', hchar.shape, '\n', hchar)


    # sum over all bins at a given frequency
    hchar = np.sum(hchar, axis=(0, 1, 2))
    if(print_test):
        print('hchar summed', hchar.shape, '\n', hchar)

    hc_bg = np.sqrt(hchar)

    return hc_bg, hc_ss, hsamp, ssidx, hsmax, bgnum, ssnew



###################################################
################### TESTING #######################
###################################################
# These don't yet work independently, just here to hang on to.
def max_test(hsmax, hsamp):
    # check hsmaxes are correct
    hsmax_hsamp_match = np.empty_like(hsmax)
    for f_idx in range(len(hsmax)):
        for r_idx in range(len(hsmax[0])):
            hsmax_hsamp_match[f_idx] = (np.max(hsamp[...,f_idx, r_idx]) == hsmax[f_idx, r_idx])
    assert np.all(hsmax_hsamp_match == True), "the max amplitudes in hsamp do not match those in hsmax"
    print('max_test passed')

def ssidx_test(hsmax, hsamp, ssidx, print_test):
    """
    Test ssidx in hsamp gives the same values as hsmax

    Parameters
    ----------
    hsmax : (F,) array of scalars
        Maximum strain amplitude of a single source at each frequency.
    hsamp : (M, Q, Z, F,) ndarray of scalar
        Strain amplitude of a source in each bin
    ssidx : (F, 4) ndarray


    """
    # check ssidx are correct and in frequency order
    for ff in range(len(hsmax)): #ith frequency
        for rr in range(len(hsmax[0])):
            m,q,z,f,r = ssidx[:,ff,rr]
            if(print_test):
                print('max is at m,q,z,f,r = %d, %d, %d, %d, %d and it = %.2e'
                  % (m, q, z, f, r, hsmax[ff,rr]))
            assert (hsamp[m,q,z,f,r] == hsmax[ff,rr]), f"The ssidx[{ff},{rr}] does not give the hsmax[{ff},{rr}]."
    print('ssidx test passes')


def ssnew_test(hsmax, hsamp, ssnew, print_test):
    """
    Test ssnew in hsamp gives the same values as hsmax

    Parameters
    ----------
    hsmax : (F,) array of scalars
        Maximum strain amplitude of a single source at each frequency.
    hsamp : (M, Q, Z, F,) ndarray of scalar
        Strain amplitude of a source in each bin
    ssnew : (4, F) ndarray


    """
    maxes = hsamp[ssnew[0], ssnew[1], ssnew[2], ssnew[3]]
    if(print_test):
        print('maxes by hsamp[ssnew[0], ssnew[1], ssnew[2], ssnew[3]] are:',
        maxes)

    assert np.all(maxes == hsmax), f"ssnew does not give correct hs maxes"
    print('ssnew test passes')



def number_test(num, bgnum, fobs, exname='', plot_test=False):
    '''
    Plots num - bgnum, where number is the ndarray of
    integer number of sources in each bin, i.e. after
    rounding or Poisson sampling

    Parameters
    ------------
    num : (M, Q, Z, F) array
        integer numbers in each bin, i.e. after rounding or
        Poisson sampling
    bgnum : (M, Q, Z, F) array
        number of background sources in each bin,
        after single source subtraction
    fobs : (F) array
        frequencies of each F, for ax titles
    exname : String
        name of example
    plot_test : Bool
        whether or not to print values a


    Returns
    -----------
    None

    '''
    if np.all(num%1 == 0) != True: warnings.warn("num contains at least one non-integer value")
    difs = num - bgnum
    assert len(difs[np.where(difs>0)]) == len(difs[0,0,0,:]), "More than one bin per frequency found with a single source subtracted."

    if(plot_test):
        fig, ax = plt.subplots(1,len(fobs), figsize = (10,3), sharey=True)
        fig.suptitle('integer number - numbg for each bin, '+ exname)
        ax[0].set_ylabel('number - number_bg')
        bins = np.arange(0, num[...,0].size, 1)
        bins = np.reshape(bins, num[...,0].shape)
        # print(bins.shape)
        # print(num[...,0].shape)
        for f in range(len(fobs)):
            ax[f].scatter(bins, (num[...,f] - bgnum[...,f]))
            ax[f].set_title('$f_\mathrm{obs}$ = %dnHz' % (fobs[f]*10**9))
            ax[f].set_xlabel('bin')
        fig.tight_layout()
    print('number test passed')



def compare_to_loops_test(edges, number, hc_bg, hc_ss, hsmax, ssidx, bgnum):
    hc_bg_loop, hc_ss_loop, sspar_loop, ssidx_loop, maxhs_loop, number_bg_loop \
      = ss_by_loops(edges, number, realize=False, round=True, print_test=False)

    # for i in range(len(ssidx)):
    #     assert np.all(ssidx[i, 0:3] == ssidx_loop[i,:]), \
    #         f"ssidx[{i}] by ndars does not match by loops"
    assert (np.all(bgnum == number_bg_loop)), "bgnum by ndars does not match by loops"
    assert (np.all(hc_ss == hc_ss_loop)), "hc_ss by ndars does not match by loops"
    assert (np.all(hsmax == maxhs_loop)), "hsmax by ndars does not match by loops"
    assert (np.all(hsmax == maxhs_loop)), "hsmax by ndars does not match by loops"
    if (np.all(np.isclose(hc_bg, hc_bg_loop, atol=1e-20, rtol=1e-20)) !=True):
        print("hc_bg by ndars does not match by loops, using atol=1e-20, rtol=1e-20")
    print('compare to loops test passed')

def quadratic_sum_test(hc_bg, hc_ss, hc_tt, print_test):
    test = (hc_bg**2 + hc_ss**2)
    error = (test-hc_tt**2)/hc_tt**2
    assert np.all(np.isclose(hc_tt, test, atol=2e-15, rtol=1e-15)), \
        "quadratic sum of hc_bg and hc_ss does not match hc_tt"
    if(print_test):
        print('percent error between (hc_bg^2+hc_ss^2) and hc_tt^2:', error)
        print('differences between np.sqrt((hc_bg^2+hc_ss^2)) and hc_tt:',
              np.sqrt(test) - hc_tt)
    print('quadratic sum test passed')



def run_ndars_tests(edges,number, fobs, exname='', print_test=False,
                     loop_comparison = True):
    '''
    Call tests for some edges, number
    Paramaters
    ----------
    edges : (4,) list of 1D arrays
        Mass, ratio, redshift, and frequency edges of bins
    number : (M, Q, Z, F) ndarray of scalars
        Number of binaries in each bin
    fobs : (F,) array of scalars
        Observed frequency bin centers
    exname : String
        Name of example (used for number plots)

    Returns
    ------
    hsamp
    hsmax
    ssidx
    bgnum
    '''
    hc_bg, hc_ss, hsamp, ssidx, hsmax, bgnum = ss_by_ndars(edges, number, realize=False, round=True)
    max_test(hsmax, hsamp)

    ssidx_test(hsmax, hsamp, ssidx, print_test)

    # ssnew_test(hsmax, hsamp, ssnew, print_test)

    # rounded = np.floor(number).astype(np.int64)
    # number_test(rounded, bgnum, fobs, exname, plot_test=print_test)

    if(loop_comparison): # optional because its faster without
        compare_to_loops_test(edges, number, hc_bg, hc_ss, hsmax, ssidx, bgnum)

    hc_tt = gws_by_ndars(edges, number, realize=False, round = True, sum=True)
    quadratic_sum_test(hc_bg, hc_ss, hc_tt, print_test)


    return hc_bg, hc_ss, hsamp, ssidx, hsmax, bgnum






###################################################
################## EXAMPLES #######################
###################################################

def example(dur, cad, mtot, mrat, redz, print_test):
    '''
    1) Choose the frequency bins at which to calculate the GWB, same as in semi-analytic-models.ipynb
    2) Build Semi-Analytic-Model with super simple parameters
    3) Get SAM edges and numbers as in sam.gwb()

    Parameters
    ----------oiuyuiuytd
    dur : scalar
        Duration of observation in secnods (multiply by YR)
    cad : scalar
        Cadence of observations in seconds (multiply by YR)
    mtot : (3,) list of scalars
        Min, max, and steps for total mass.
    mrat : (3,) list of scalars
        Min, max, and steps for mass ratio.
    redz : (3,) list of scalars
        Min, max, and steps for redshift.
    print_test :

    Returns
    -------
    edges : (4,) list of 1darrays
        A list containing the edges along each dimension.  The four dimensions correspond to
        total mass, mass ratio, redshift, and observer-frame orbital frequency.
        The length of each of the four arrays is M+1, Q+1, Z+1, F+1.
    number : (M, Q, Z, F) array
        The number of binaries in each bin of parameter space.  This is calculated by integrating
        `dnum` over each bin.
    fobs : (F) array
        observed frequency bin centers
    '''
    # 1) Choose the frequency bins at which to calculate the GWB, same as in semi-analytic-models.ipynb
    fobs = utils.nyquist_freqs(dur,cad)
    fobs_edges = utils.nyquist_freqs_edges(dur,cad)
    if(print_test):
        print(f"Number of frequency bins: {fobs.size-1}")
        print(f"  between [{fobs[0]*YR:.2f}, {fobs[-1]*YR:.2f}] 1/yr")
        print(f"          [{fobs[0]*1e9:.2f}, {fobs[-1]*1e9:.2f}] nHz")

    # 2) Build Semi-Analytic-Model with super simple parameters
    if(mtot==None or mrat==None or redz==None):
        print('using default mtot, mrat, and redz')
        sam = holo.sam.Semi_Analytic_Model(ZERO_GMT_STALLED_SYSTEMS=True,
                                           ZERO_DYNAMIC_STALLED_SYSTEMS=False)
    else:
        sam = holo.sam.Semi_Analytic_Model(mtot, mrat, redz,
                                           ZERO_GMT_STALLED_SYSTEMS=True,
                                           ZERO_DYNAMIC_STALLED_SYSTEMS=False)
    if(print_test):
        print('edges:', sam.edges)
    # get observed orbital frequency bin edges and centers
    # from observed GW frequency bin edges
    fobs_orb_edges = fobs_edges / 2.0 # f_orb = f_GW/2
    fobs_orb_cents = kale.utils.midpoints(fobs_edges) / 2.0

    # 3) Get SAM edges and numbers as in sam.gwb()
    # dynamic_binary_number

    # gets differential number of binaries per bin-vol per log freq interval
    edges, dnum = sam.dynamic_binary_number(holo.hardening.Hard_GW, fobs_orb=fobs_orb_cents, zero_stalled=False)
    edges[-1] = fobs_orb_edges

    # integrate (multiply by bin volume) within each bin
    number = utils._integrate_grid_differential_number(edges, dnum, freq=False)
    number = number * np.diff(np.log(fobs_edges))

    return edges, number, fobs


def example2(print_test = True, exname='Example 2'):
    '''
    Parameters
    ---------
    print_test : Bool
        Whether to print frequencies and edges


    Returns
    ---------
    edges : (M,Q,Z,F) array
    number : (M, Q, Z, F) array
    fobs : (F) array
        observed frequency bin centers
    '''

    dur = 5.0*YR/3.1557600
    cad = .5*YR/3.1455145557600

    mtot=(1.0e6*MSOL/1.988409870698051, 1.0e8*MSOL/1.988409870698051, 3)
    mrat=(1e-1, 1.0, 2)
    redz=(1e-3, 1.0, 4)

    edges, number, fobs = example(dur, cad, mtot, mrat, redz, print_test)
    return edges, number, fobs, exname

def example3(print_test = True, exname = 'Example 3'):
    '''
    Parameters
    ---------
    print_test : Bool
        Whether to print frequencies and edges


    Returns
    ---------
    edges : (M,Q,Z,F) array
    number : (M, Q, Z, F) array
    fobs : (F) array
        observed frequency bin centers
    '''
    dur = 5.0*YR/3.1557600
    cad = .5*YR/3.1557600


    mtot=(1.0e6*MSOL/1.988409870698051, 4.0e9*MSOL, 25)
    mrat=(1e-1, 1.0, 25)
    redz=(1e-3, 10.0, 25)

    edges, number, fobs = example(dur, cad, mtot, mrat, redz, print_test)
    return edges, number, fobs, exname


def example4(print_test = True, exname = 'Example 4'):
    '''
    Parameters
    ---------
    print_test : Bool
        Whether to print frequencies and edges


    Returns
    ---------
    edges : (M,Q,Z,F) array
    number : (M, Q, Z, F) array
    fobs : (F) array
        observed frequency bin centers
    '''
    dur = 5.0*YR/3.1557600
    cad = .2*YR/3.1557600

    mtot=(1.0e6*MSOL/1.988409870698051, (4.0e11*MSOL).astype(np.float64), 25)
    mrat=(1e-1, 1.0, 25)
    redz=(1e-3, 10.0, 25)

    edges, number, fobs = example(dur, cad, mtot, mrat, redz, print_test)
    return edges, number, fobs, exname


def example5(print_test = True, exname = 'Example 5'):
    '''
    Parameters
    ---------
    print_test : Bool
        Whether to print frequencies and edges


    Returns
    ---------
    edges : (M,Q,Z,F) array
    number : (M, Q, Z, F) array
    fobs : (F) array
        observed frequency bin centers
    '''
    dur = 10.0*YR
    cad = .2*YR

    # default mtot, mrat, redz

    edges, number, fobs = example(dur, cad, mtot=None, mrat=None, redz=None,
                                  print_test=print_test)
    return edges, number, fobs, exname




###################################################
################## PLOTTING #######################
###################################################
def plot_medians(ax, xx, BG=None, SS=None, LABEL='',
                 BG_COLOR='k', BG_ERRORS = False,
                 SS_COLOR='k', SS_ERRORS = True):
    """
    Parameters:
    ax
    xx
    BG
    SS
    REALS
    label
    COLOR
    """
    # plot median bg of samples
    if(BG is not None and BG_ERRORS == False):
        ax.plot(xx, np.median(BG, axis=1),
                color=BG_COLOR, label=('bg median' + LABEL),
                linewidth=3, linestyle='solid', alpha=0.8)
    elif(BG is not None and BG_ERRORS == True):
        ax.errorbar(xx, np.median(BG, axis=1), yerr=np.std(BG, axis=1),
                    color=BG_COLOR, label=('bg median'+LABEL),
                    fmt='-', linewidth=2, alpha=0.8, capsize=3)

    # plot median ss of samples
    if(SS is not None and SS_ERRORS == False):
        ax.scatter(xx, np.median(SS, axis=1),
                   color=SS_COLOR, label=('ss median'+LABEL),
                   marker ='*', s = 50, alpha=0.8)
    elif(SS is not None and SS_ERRORS == True):
        ax.errorbar(xx, np.median(SS, axis=1), yerr=np.std(SS, axis=1),
                    color=SS_COLOR, label=('ss median'+LABEL),
                    fmt='*', alpha=0.8, markersize=7, capsize=3)


def plot_BG(ax, xx, BG, LABEL, REALS=0, median=False, COLOR='b', rand=False):
    """
    Plot the background median, middle 50% and 98% confidence intervals,
    and optionally several realizations. All plotted in same color.

    Parameters
    ------------
    ax : pyplot ax object
    xx : (F,) 1darray of scalars
    BG : (F,R) Ndarray of scalars
    LABEL : String
    REALS : int
        How many bg realizations to plot
    median : bool
        Whether or not to plot the bg median (same color)
    COLOR : String
    rand : bool
        Whether to randomize which realizations are plotted (True)
        or plots the 0th to REALS-th realizations.
    """
    if(median==True): ax.plot(xx, np.median(BG, axis=1), color=COLOR, label=LABEL)
    # plot contours at 50% and 98% confidence intervals
    for pp in [50, 98]:
        percs = pp / 2
        percs = [50 - percs, 50 + percs]
        ax.fill_between(xx, *np.percentile(BG, percs, axis=-1), alpha=0.25, color=COLOR)
    # Plot `nsamp` random spectra

    if (REALS is not None):
        if(rand):
            idx = np.random.choice(BG.shape[1], REALS, replace=False)
            ax.plot(xx, BG[:, idx], lw=1.0, alpha=0.5, color=COLOR, linestyle = 'dotted')
        else:
            for rr in range(REALS):
                ax.plot(xx, BG[:, rr], lw=1.0, alpha=0.5, color=COLOR, linestyle = 'dotted')



def plot_samples(ax, xx, BG=None, SS=None, REALS=1, LABEL=''):
    """
    Plot the background and/or single sources for the first 'REALS'
    number of realizations, with each color corresponding to a difference
    realization.

    Parameters
    ----------
    ax : pyplot ax object
    xx : (F,) array of scalars
    BG : (F,R) ndarray or None
    SS : (F,R) ndarray or None
    REALS : int
    """
    colors = cm.rainbow(np.linspace(0,1,REALS))
    for rr in range(REALS):
        if rr==REALS-1:
            if(BG is not None):
                ax.plot(xx, BG[:,rr], lw=2.0, alpha=0.5, color=colors[rr], linestyle='solid',
                    label='background'+LABEL)
            if(SS is not None):
                ax.scatter(xx, SS[:,rr], color=colors[rr], marker='o', s=80,
                    edgecolor='k', alpha=0.5, label='single source'+LABEL)
        else:
            if(BG is not None):
                ax.plot(xx, BG[:,rr], lw=2.0, alpha=0.5, color=colors[rr], linestyle='solid')
            if(SS is not None):
                ax.scatter(xx, SS[:,rr], color=colors[rr], marker='o', s=80,
                    edgecolor='k', alpha=0.5)

def plot_std(ax, xx, BG, SS, COLOR='b', LABEL=''):
    """
    Plot the standard deviations of the bg and ss characteristic strains.

    Parameters
    --------
    ax : pyplot ax object
    xx : (F,) array of scalars
    BG : (F, R) Ndarray of scalars
    SS : (F, R) Ndarray of scalars
    COLOR : string
    LABEL : string

    Returns
    -------
    std_bg : (F,) array of scalars
    std_ss : (F,) array of scalars
    """
    std_bg = np.std(BG, axis=1)
    # med_bg = np.median(BG, axis=1)
    std_ss = np.std(SS, axis=1)
    # med_ss = np.median(SS, axis=1)

    ax.plot(xx, std_bg, lw=4.0, alpha=0.6, color=COLOR,
            label = 'bg stdev'+LABEL, marker='o', ms=10, linestyle = 'solid')
    ax.plot(xx, std_ss, lw=2.0, alpha=0.6, color=COLOR, markeredgecolor='k',
            label = 'ss stdev'+LABEL, marker='o', ms=10, linestyle='dotted')
    return std_bg, std_ss

def plot_IQR(ax, xx, BG=None, SS=None, COLOR='r', LABEL=''):
    """
    Plot the IQR of the bg and ss characteristic strains.

    Parameters
    ----------
    ax : pyplot ax object
    xx : (F,) array of scalars
    BG : (F, R) Ndarray of scalars
    SS : (F, R) Ndarray of scalars
    COLOR : string
    LABEL : string

    Returns
    -------
    """

    if (BG is not None):
        Q75_bg, Q25_bg= np.percentile(BG, [75 ,25], axis=1)
        IQR_bg = Q75_bg-Q25_bg
        ax.plot(xx, IQR_bg, lw=4.0, alpha=0.6, color=COLOR,
            label = 'bg stdev'+LABEL, marker='P', ms=10, linestyle = 'solid')
    if(SS is not None):
        Q75_ss, Q25_ss = np.percentile(SS, [75 ,25], axis=1)
        IQR_ss = Q75_ss-Q25_ss
        ax.plot(xx, IQR_ss, lw=2.0, alpha=0.6, color=COLOR, markeredgecolor='k',
                label = 'ss stdev'+LABEL, marker='P', ms=10, linestyle='dotted')



def plot_percentiles(ax, xx, BG=None, SS=None, LABEL='',
                     BG_COLOR='b', SS_COLOR='r',
                     BG_MARKER=None, SS_MARKER=None,
                     BG_LINESTYLE='solid', SS_LINESTYLE='solid'):
    """
    Plots 25th and 75th percentiles, and IQR region between
    (50% confidence interval).

    Parameters
    ---------
    ax : pyplot ax object
    xx : (F,) 1darray of scalars
    BG : (F, R) Ndarray of scalars or None
    SS : (F, R) Ndarray of scalars or None
    BG_COLOR : string
    SS_COLOR : string
    BG_MARKER : string
    SS_MARKER : string
    BG_LINESTYLE : string
    SS_LINESTYLE : string
    """

    if (BG is not None):
        Q75_bg, Q25_bg= np.percentile(BG, [75 ,25], axis=1)
        ax.plot(xx, Q25_bg, label = 'bg'+LABEL,
                lw=2.0, alpha=.8, color=BG_COLOR,
                marker = BG_MARKER, ms=5, linestyle = BG_LINESTYLE)
        ax.plot(xx, Q75_bg,
                lw=2.0, alpha=.8, color=BG_COLOR,
                marker = BG_MARKER, ms=5, linestyle = BG_LINESTYLE)
        ax.fill_between(xx, Q25_bg, Q75_bg, alpha=0.2, color=BG_COLOR)

    if(SS is not None):
        Q75_ss, Q25_ss = np.percentile(SS, [75 ,25], axis=1)
        ax.plot(xx, Q25_ss, label = 'ss'+LABEL,
                lw=2.0, alpha=.8, color=SS_COLOR,
                marker = SS_MARKER, ms=5, linestyle = SS_LINESTYLE)
        ax.plot(xx, Q75_ss,
                lw=2.0, alpha=.8, color=SS_COLOR,
                marker = SS_MARKER, ms=5, linestyle = SS_LINESTYLE)
        ax.fill_between(xx, Q25_ss, Q75_ss, alpha=0.2, color=SS_COLOR)


def plot_params(axs, xx, REALS=1, LABEL='', grid=None,
                BG_PARAMS=None, SS_PARAMS=None,
                BG_MEDIAN=True, SS_MEDIAN=True,
                BG_ERRORS=True, SS_ERRORS=True,
                BG_COLOR='k', SS_COLOR='mediumorchid',
                TITLES = np.array([['Total Mass $M/M_\odot$', 'Mass Ratio $q$'],
                                   ['Redshift $z$', 'Characteristic Strain $h_c$']]),
                XLABEL = 'Frequency $f_\mathrm{obs}$ (1/yr)',
                SHOW_LEGEND = True):
    """
    Plot mass, ratio, redshift, and strain in 4 separate subplots.

    Parameters:
    -----------
    axs : (2,2) array of pyplot ax object
    xx : (F,) 1d array of scalars
    params : (4,) 1Darray of (F,R,) NDarrays
    titles : (4,) array of strings
    xlabel : string
    legend : bool
        Whether or not to include a legend in each subplot

    """
    colors = cm.rainbow(np.linspace(0,1,REALS))
    for ii in range(len(axs)):
        for jj in range(len(axs)):
            axs[ii,jj].set_ylabel(TITLES[ii,jj])
    # if(BG is not None):
    #     ax.plot(xx, BG[:,rr], lw=2.0, alpha=0.5, color=colors[rr], linestyle='solid',
    #         label='background'+LABEL)
    # if(SS is not None):
    #     ax.scatter(xx, SS[:,rr], color=colors[rr], marker='o', s=80,
    #         edgecolor='k', alpha=0.5, label='single source'+LABEL)
            if(ii==0 or jj==0): # mass, ratio, or redshift
                # bin edges
                if(grid is not None):
                    for kk in range(len(grid[ii,jj])):
                        if(kk==0): edgelabel='edges'
                        else: edgelabel=None
                        axs[ii,jj].axhline(grid[ii,jj][kk], color='k', alpha=0.6, lw=0.15, label=edgelabel)

            if(BG_PARAMS is not None):
                for rr in range(REALS):
                    if(rr==0): bglabel = 'bg'+LABEL
                    else: bglabel = None
                    axs[ii,jj].plot(xx, BG_PARAMS[ii,jj,:,rr], label=bglabel,
                                    color=colors[rr], lw=2.0, alpha=0.5,
                                    linestyle='solid',)
            if (SS_PARAMS is not None):
                # single source realizations
                for rr in range(REALS):
                    if(rr==0): sslabel = 'ss'+LABEL
                    else: sslabel = None
                    axs[ii,jj].scatter(xx, SS_PARAMS[ii,jj,:,rr], label=sslabel,
                                        color=colors[rr], marker='o', s=80,
                                        alpha=0.5)

            # else: #strain
            #     if(BG_PARAMS is not None):

            #     if(SS_PARAMS is not None):
            #         for rr in range(REALS):
            #             axs[ii,jj].scatter(xx, SS_PARAMS[ii,jj,:,rr], color=colors[rr],
            #                             marker='o', s=80, alpha=0.5)
            # axs[ii,jj].errorbar(xx, np.mean(params[ii,jj], axis=1),
            #                 yerr = np.std(params[ii,jj], axis=1), label='mean',
            #                 fmt = 'o', color='darkmagenta', capsize=3, alpha=.8)

            # ax.errorbar(xx, np.median(BG, axis=1), yerr=np.std(BG, axis=1),
            #         color=BG_COLOR, label=('bg median'+LABEL),
            #         fmt='-', linewidth=2, alpha=0.8, capsize=3)

            # # plot median ss of samples
            # if(SS is not None and SS_ERRORS == False):
            #     ax.scatter(xx, np.median(SS, axis=1),
            #             color=SS_COLOR, label=('ss median'+LABEL),
            #             marker ='*', s = 50, alpha=0.8)
            # elif(SS is not None and SS_ERRORS == True):
            #     ax.errorbar(xx, np.median(SS, axis=1), yerr=np.std(SS, axis=1),
            #                 color=SS_COLOR, label=('ss median'+LABEL),
            #                 fmt='*', alpha=0.8, markersize=7, capsize=3)


            if(BG_MEDIAN == True):
                if(BG_ERRORS == True):
                    axs[ii,jj].errorbar(xx, np.median(BG_PARAMS[ii,jj], axis=1),
                                        yerr=np.std(BG_PARAMS[ii,jj], axis=1),
                                        color=BG_COLOR, label=('bg median' +LABEL),
                                        fmt='-', linewidth=2, alpha=0.8, capsize=3)
                else:
                    axs[ii,jj].plot(xx, np.median(BG_PARAMS[ii,jj], axis=1),
                                    color=BG_COLOR, label=('bg median'+LABEL),
                                    linewidth=2, alpha=0.8, linestyle='solid',)
            if(SS_MEDIAN == True):
                if(SS_ERRORS == True):
                    axs[ii,jj].errorbar(xx, np.median(SS_PARAMS[ii,jj], axis=1),
                                       yerr=np.std(SS_PARAMS[ii,jj], axis=1),
                                       color=SS_COLOR, label=('ss median'+LABEL),
                                       fmt='*', alpha=0.8, markersize=7, capsize=3,
                                       markeredgecolor='k', )
                else:
                    axs[ii,jj].scatter(xx, np.median(SS_PARAMS[ii,jj], axis=1),
                                       color=SS_COLOR, label=('ss median'+LABEL),
                                       marker='*', s=50, alpha=0.8,
                                       edgecolor='k', )
            axs[ii,jj].set_yscale('log')
            axs[ii,jj].set_xscale('log')



            # axs[ii,jj].fill_between(grid[ii,jj][0], grid[ii,jj][-1])

            if(ii==1): axs[ii,jj].set_xlabel(XLABEL)
            if(jj==1):
                axs[ii,jj].yaxis.set_label_position("right")
                axs[ii,jj].yaxis.tick_right()
            if(SHOW_LEGEND): axs[ii,jj].legend(loc='lower left')


###################################################
############ Utilities  #################
###################################################

def resample_loudest(hc_ss, hc_bg, nloudest):
    if nloudest > hc_ss.shape[-1]: # check for valid nloudest
        err = f"{nloudest=} for detstats must be <= nloudest of hc data"
        raise ValueError(err)

    # recalculate new hc_bg and hc_ss
    new_hc_bg = np.sqrt(hc_bg**2 + np.sum(hc_ss[...,nloudest:-1]**2, axis=-1))
    new_hc_ss = hc_ss[...,0:nloudest]

    return new_hc_ss, new_hc_bg




