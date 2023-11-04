""" Module for predicting anisotropy with single source populations.

"""

import numpy as np
import matplotlib as plt
import matplotlib.cm as cm

import kalepy as kale
import healpy as hp
import h5py

import holodeck as holo
from holodeck import utils, cosmo, log, detstats, plot
from holodeck.constants import YR

NSIDE = 32
NPIX = hp.nside2npix(NSIDE)
LMAX = 8
HC_REF15_10YR = 11.2*10**-15 


def healpix_map(hc_ss, hc_bg, nside=NSIDE, seed=None, ret_seed=False):
    """ Build mollview array of hc^2/dOmega for a healpix map
    
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
        Array of h_c^2/dOmega at every pixel for a mollview healpix map.
    
    NOTE: Could speed up the for-loops, but it's ok for now.
    """

    npix = hp.nside2npix(nside)
    area = hp.nside2pixarea(nside)
    nfreqs = len(hc_ss)
    nreals = len(hc_ss[0])
    nloudest = len(hc_ss[0,0])

    # set random seed
    if seed is None:
        seed = np.random.randint(99999)   # get a random number
    print(f"random seed: {seed}")                           # print it out so we can reuse it if desired
    np.random.seed(seed)   

    # spread background evenly across pixels in moll_hc
    moll_hc = np.ones((nfreqs,nreals,npix)) * hc_bg[:,:,np.newaxis]**2/(npix*area) # (frequency, realization, pixel)

    # choose random pixels to place the single sources
    pix_ss = np.random.randint(0, npix-1, size=nfreqs*nreals*nloudest).reshape(nfreqs, nreals, nloudest)
    for ff in range(nfreqs):
        for rr in range(nreals):
            for ll in range(nloudest):
                moll_hc[ff,rr,pix_ss[ff,rr,ll]] = (moll_hc[ff,rr,pix_ss[ff,rr,ll]] + hc_ss[ff,rr,ll]**2/area)
    if ret_seed:
        return moll_hc, seed           
    return moll_hc

def healpix_map_oldhc2(hc_ss, hc_bg, nside=NSIDE):
    """ Build mollview array of hc^2/dOmega for a healpix map
    
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
        Array of h_c^2 at every pixel for a mollview healpix map.
    
    NOTE: Could speed up the for-loops, but it's ok for now.
    """

    npix = hp.nside2npix(nside)
    area = hp.nside2pixarea(nside)
    nfreqs = len(hc_ss)
    nreals = len(hc_ss[0])
    nloudest = len(hc_ss[0,0])

    # spread background evenly across pixels in moll_hc
    moll_hc = np.ones((nfreqs,nreals,npix)) * hc_bg[:,:,np.newaxis]**2/(npix) # (frequency, realization, pixel)

    # choose random pixels to place the single sources
    pix_ss = np.random.randint(0, npix-1, size=nfreqs*nreals*nloudest).reshape(nfreqs, nreals, nloudest)
    for ff in range(nfreqs):
        for rr in range(nreals):
            for ll in range(nloudest):
                moll_hc[ff,rr,pix_ss[ff,rr,ll]] = (moll_hc[ff,rr,pix_ss[ff,rr,ll]] + hc_ss[ff,rr,ll]**2)
                
    return moll_hc

def healpix_map_oldhc(hc_ss, hc_bg, nside=NSIDE):
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


def healpix_hcsq_map(hc_ss, hc_bg, nside=NSIDE):
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
    moll_hc2 = np.ones((nfreqs,nreals,npix)) * hc_bg[:,:,np.newaxis]**2/(npix) # (frequency, realization, pixel)

    # choose random pixels to place the single sources
    pix_ss = np.random.randint(0, npix-1, size=nfreqs*nreals*nloudest).reshape(nfreqs, nreals, nloudest)
    for ff in range(nfreqs):
        for rr in range(nreals):
            for ll in range(nloudest):
                moll_hc2[ff,rr,pix_ss[ff,rr,ll]] = moll_hc2[ff,rr,pix_ss[ff,rr,ll]] + hc_ss[ff,rr,ll]**2
                
    return moll_hc2

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
        Array of hc^2/dOmega at every pixel for a mollview healpix map.
    Cl : (F,R,lmax+1) NDarray
        Spherical harmonic coefficients 
    
    """
    moll_hc = healpix_map(hc_ss, hc_bg, nside)
    Cl = sph_harm_from_map(moll_hc, lmax)

    return moll_hc, Cl


######################################################################
############# Plots
######################################################################

def plot_ClC0_medians(fobs, Cl_best, lmax, nshow):
    xx = fobs*YR
    fig, ax = holo.plot.figax(figsize=(8,5), xlabel=holo.plot.LABEL_GW_FREQUENCY_YR, ylabel='$C_{\ell>0}/C_0$')

    yy = Cl_best[:,:,:,1:]/Cl_best[:,:,:,0,np.newaxis] # (B,F,R,l)
    yy = np.median(yy, axis=-1) # (B,F,l) median over realizations

    colors = cm.gist_rainbow(np.linspace(0, 1, lmax))
    for ll in range(lmax):
        ax.plot(xx, np.median(yy[:,:,ll], axis=0), color=colors[ll], alpha=0.75, label='$l=%d$' % (ll+1))
        for pp in [50, 98]:
            percs = pp/2
            percs = [50-percs, 50+percs]
            ax.fill_between(xx, *np.percentile(yy[:,:,ll], percs, axis=0), alpha=0.1, color=colors[ll])
        
        for bb in range(0,nshow):
            ax.plot(xx, yy[bb,:,ll], color=colors[ll], linestyle=':', alpha=0.1,
                                 linewidth=1)         
        ax.legend(ncols=2)
    holo.plot._twin_hz(ax, nano=False)
    
    # ax.set_title('50%% and 98%% confidence intervals of the %d best samples \nusing realizations medians, lmax=%d'
    #             % (nbest, lmax))
    return fig


######################################################################
############# Libraries
######################################################################


def lib_anisotropy(lib_path, hc_ref_10yr=HC_REF15_10YR, nbest=100, nreals=50, lmax=LMAX, nside=NSIDE):

    # ---- read in file
    hdf_name = lib_path+'/sam_lib.hdf5'
    print('Hdf file:', hdf_name)

    ss_file = h5py.File(hdf_name, 'r')
    print('Loaded file, with keys:', list(ss_file.keys()))
    hc_ss = ss_file['hc_ss'][:,:,:nreals,:]
    hc_bg = ss_file['hc_bg'][:,:,:nreals]
    fobs = ss_file['fobs'][:]
    ss_file.close()

    shape = hc_ss.shape
    nsamps, nfreqs, nreals, nloudest = shape[0], shape[1], shape[2], shape[3]
    print('N,F,R,L =', nsamps, nfreqs, nreals, nloudest)


     # ---- rank samples
    nsort, fidx, hc_ref = detstats.rank_samples(hc_ss, hc_bg, fobs, fidx=1, hc_ref=hc_ref_10yr, ret_all=True)
    
    print('Ranked samples by hc_ref = %.2e at fobs = %.2f/yr' % (hc_ref, fobs[fidx]*YR))


    # ---- calculate spherical harmonics

    npix = hp.nside2npix(nside)
    Cl_best = np.zeros((nbest, nfreqs, nreals, lmax+1 ))
    moll_hc_best = np.zeros((nbest, nfreqs, nreals, npix))
    for nn in range(nbest):
        print('on nn=%d out of nbest=%d' % (nn,nbest))
        moll_hc_best[nn,...], Cl_best[nn,...] = sph_harm_from_hc(
            hc_ss[nsort[nn]], hc_bg[nsort[nn]], nside=nside, lmax=lmax, )
        

    # ---- save to npz file

    output_dir = lib_path+'/anisotropy'
    # Assign output folder
    import os
    if (os.path.exists(output_dir) is False):
        print('Making output directory.')
        os.makedirs(output_dir)
    else:
        print('Writing to an existing directory.')

    output_name = output_dir+'/sph_harm_hc2dOm_lmax%d_nside%d_nbest%d_nreals%d.npz' % (lmax, nside, nbest, nreals)
    print('Saving npz file: ', output_name)
    np.savez(output_name,
             nsort=nsort, fidx=fidx, hc_ref=hc_ref, ss_shape=shape,
         moll_hc_best=moll_hc_best, Cl_best=Cl_best, nside=nside, lmax=lmax, fobs=fobs)
    

    # ---- plot median Cl/C0
    
    print('Plotting Cl/C0 for median realizations')
    fig = plot_ClC0_medians(fobs, Cl_best, lmax, nshow=nbest)
    fig_name = output_dir+'/sph_harm_lmax%d_nside%d_nbest%d.png' % (lmax, nside, nbest)
    fig.savefig(fig_name, dpi=300)


def lib_anisotropy_split(lib_path, hc_ref_10yr=HC_REF15_10YR, nbest=100, nreals=50, lmax=LMAX, nside=NSIDE, split=2):

    # ---- read in file
    hdf_name = lib_path+'/sam_lib.hdf5'
    print('Hdf file:', hdf_name)

    ss_file = h5py.File(hdf_name, 'r')
    print('Loaded file, with keys:', list(ss_file.keys()))
    hc_ss = ss_file['hc_ss'][:,:,:nreals,:]
    hc_bg = ss_file['hc_bg'][:,:,:nreals]
    fobs = ss_file['fobs'][:]
    ss_file.close()

    shape = hc_ss.shape
    nsamps, nfreqs, nreals, nloudest = shape[0], shape[1], shape[2], shape[3]
    print('N,F,R,L =', nsamps, nfreqs, nreals, nloudest)


     # ---- rank samples
    nsort, fidx, hc_ref = detstats.rank_samples(hc_ss, hc_bg, fobs, fidx=1, hc_ref=hc_ref_10yr, ret_all=True)
    
    print('Ranked samples by hc_ref = %.2e at fobs = %.2f/yr' % (hc_ref, fobs[fidx]*YR))


    for ss in range(split):
        bestrange = (np.array([ss, (ss+1)])*(nbest)/split).astype(int)
        bestrange[1] = np.min([bestrange[1], nbest])
        print(f"{bestrange=}")
        # ---- calculate spherical harmonics

        npix = hp.nside2npix(nside)
        Cl_best = np.zeros((bestrange[1]-bestrange[0], nfreqs, nreals, lmax+1 ))
        moll_hc_best = np.zeros((bestrange[1]-bestrange[0], nfreqs, nreals, npix))
        for ii, nn in enumerate(range(bestrange[0], bestrange[1])):
            print('on nn=%d out of nbest=%d' % (nn,nbest))
            moll_hc_best[ii,...], Cl_best[ii,...] = sph_harm_from_hc(
                hc_ss[nsort[nn]], hc_bg[nsort[nn]], nside=nside, lmax=lmax, )
            

        # ---- save to npz file

        output_dir = lib_path+'/anisotropy'
        # Assign output folder
        import os
        if (os.path.exists(output_dir) is False):
            print('Making output directory.')
            os.makedirs(output_dir)
        else:
            print('Writing to an existing directory.')

        output_name =(output_dir+'/sph_harm_hc2dOm_lmax%d_ns%02d_r%d_b%02d-%-02d.npz' 
                      % (lmax, nside, nreals, bestrange[0], bestrange[1]-1))
        print('Saving npz file: ', output_name)
        np.savez(output_name,
                nsort=nsort, fidx=fidx, hc_ref=hc_ref, ss_shape=shape,
            moll_hc_best=moll_hc_best, Cl_best=Cl_best, nside=nside, lmax=lmax, fobs=fobs, split=split)
    

        # # ---- plot median Cl/C0
        
        # print('Plotting Cl/C0 for median realizations')
        # fig = plot_ClC0_medians(fobs, Cl_best, lmax, nshow=(bestrange[1]-bestrange[0]))
        # fig_name = (output_dir+'/sph_harm_hc2dOm_lmax%d_ns%02d_r%d_b%02d-%-02d.png' 
        #               % (lmax, nside, nreals, bestrange[0], bestrange[1]-1))
        # fig.savefig(fig_name, dpi=300)



######################################################################
############# Analytic/Sato-Polito
######################################################################

def Cl_analytic_from_num(fobs_orb_edges, number, hs, realize = False, floor = False):
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
    floor : boolean
        Whether or not to round numbers down to nearest integers, if not realizing
    
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
    elif floor is True: # assumes realize is False
        number = np.floor(number)


    delta_term = (fc/(4*np.pi*df) * np.sum(number*hs**2, axis=(0,1,2)))**2

    Cl = (fc/(4*np.pi*df))**2 * np.sum(number*hs**4, axis=(0,1,2))

    C0 = Cl + delta_term

    return C0, Cl


def strain_amp_at_bin_edges_redz(edges, redz=None):
    """ Calculate strain amplitude at bin edges, with final or initial redz.
    
    """
    assert len(edges) == 4
    assert np.all([np.ndim(ee) == 1 for ee in edges])

    foo = edges[-1]                   #: should be observer-frame orbital-frequencies
    # df = np.diff(foo)                 #: frequency bin widths
    fc = kale.utils.midpoints(foo)    #: use frequency-bin centers for strain (more accurate!)


    if redz is not None:
        dc = +np.inf * np.ones_like(redz)
        sel = (redz > 0.0)
        dc[sel] = holo.cosmo.comoving_distance(redz[sel]).cgs.value
    else: 
        redz = edges[2][np.newaxis,np.newaxis,:,np.newaxis]
        dc = holo.cosmo.comoving_distance(redz).cgs.value

    # ---- calculate GW strain ----
    mt = (edges[0])
    mr = (edges[1])
    mc = utils.chirp_mass_mtmr(mt[:, np.newaxis], mr[np.newaxis, :])
    mc = mc[:, :, np.newaxis, np.newaxis]
    
    # convert from observer-frame to rest-frame; still using frequency-bin centers
    fr = utils.frst_from_fobs(fc[np.newaxis, np.newaxis, np.newaxis, :], redz)

    hs_edges = utils.gw_strain_source(mc, dc, fr)
    return hs_edges


def strain_amp_at_bin_centers_redz(edges, redz=None):
    """ Calculate strain amplitude at bin centers, with final or initial redz.
    
    """
    assert len(edges) == 4
    assert np.all([np.ndim(ee) == 1 for ee in edges])

    foo = edges[-1]                   #: should be observer-frame orbital-frequencies
    df = np.diff(foo)                 #: frequency bin widths
    fc = kale.utils.midpoints(foo)    #: use frequency-bin centers for strain (more accurate!)

    # redshifts are defined across 4D grid, shape (M, Q, Z, Fc)
    #    where M, Q, Z are edges and Fc is frequency centers
    # find midpoints of redshifts in M, Q, Z dimensions, to end up with (M-1, Q-1, Z-1, Fc)
    if redz is not None:
        for dd in range(3):
            redz = np.moveaxis(redz, dd, 0)
            redz = kale.utils.midpoints(redz, axis=0)
            redz = np.moveaxis(redz, 0, dd)
        dc = +np.inf * np.ones_like(redz)
        sel = (redz > 0.0)
        dc[sel] = holo.cosmo.comoving_distance(redz[sel]).cgs.value
    else:
        redz = kale.utils.midpoints(edges[2])[np.newaxis,np.newaxis,:,np.newaxis]
        dc = holo.cosmo.comoving_distance(redz).cgs.value


    # ---- calculate GW strain ----
    mt = kale.utils.midpoints(edges[0])
    mr = kale.utils.midpoints(edges[1])
    mc = utils.chirp_mass_mtmr(mt[:, np.newaxis], mr[np.newaxis, :])
    mc = mc[:, :, np.newaxis, np.newaxis]
    
    # convert from observer-frame to rest-frame; still using frequency-bin centers
    fr = utils.frst_from_fobs(fc[np.newaxis, np.newaxis, np.newaxis, :], redz)

    hs = utils.gw_strain_source(mc, dc, fr)
    return hs


def Cl_analytic_from_dnum(edges, dnum, redz=None, realize=False):
    """ Calculate Cl using Eq. (17) of Sato-Polito & Kamionkowski
    Parameters
    ----------
    edges : (F,) 1Darray
        Observed orbital frequency bin edges
    dnum : (M,Q,Z,F) NDarray
        dN / [ dlog10M dq dz dlnf ]
    hs : (M,Q,Z,F) NDarray
        Strain amplitude of each M,q,z bin
    
    """
    fobs_orb_edges = edges[-1]
    fobs_gw_edges = fobs_orb_edges * 2.0

    df = np.diff(fobs_orb_edges)                 #: frequency bin widths
    fc = kale.utils.midpoints(fobs_orb_edges)    #: use frequency-bin centers for strain (more accurate!)


    if realize is False:
        hs_edges = strain_amp_at_bin_edges_redz(edges, redz)

        # ---- integrate from differential-number to number per bin
        # integrate over dlog10(M)
        numh2 = utils.trapz(dnum*hs_edges**2, np.log10(edges[0]), axis=0)
        # integrate over mass-ratio
        numh2 = utils.trapz(numh2, edges[1], axis=1)
        # integrate over redshift
        numh2 = utils.trapz(numh2, edges[2], axis=2)
        # times dln(f)
        numh2 = numh2 * np.diff(np.log(fobs_gw_edges)) 

        # integrate over dlog10(M)
        numh4 = utils.trapz(dnum*hs_edges**4, np.log10(edges[0]), axis=0)
        # integrate over mass-ratio
        numh4 = utils.trapz(numh4, edges[1], axis=1)
        # integrate over redshift
        numh4 = utils.trapz(numh4, edges[2], axis=2)
        # times dln(f)
        numh4 = numh4 * np.diff(np.log(fobs_gw_edges))  # how is this not a shape issue??

    elif utils.isinteger(realize):
        # add reals axis
        hs_cents = strain_amp_at_bin_centers_redz(edges, redz)[...,np.newaxis]
        df = df[:,np.newaxis] 
        fc = fc[:,np.newaxis] 

    
        number = holo.sam_cython.integrate_differential_number_3dx1d(edges, dnum)
        shape = number.shape + (realize,)
        number = holo.gravwaves.poisson_as_needed(number[...,np.newaxis] * np.ones(shape))

        # numh2 = number * hs_cents**2 * np.diff(np.log(fobs_gw_edges))[:,np.newaxis] 
        # numh4 = number * hs_cents**4 * np.diff(np.log(fobs_gw_edges))[:,np.newaxis] 
        numh2 = number * hs_cents**2 
        numh4 = number * hs_cents**4 
    else:
        err = "`realize` ({}) must be one of {{False, integer}}!".format(realize)
        raise ValueError(err)


    delta_term = (fc / (4*np.pi * df) * np.sum(numh2, axis=(0,1,2)))**2

    Cl = ((fc / (4*np.pi*df))**2 * np.sum(numh4, axis=(0,1,2)))

    C0 = Cl + delta_term

    return C0, Cl



######################################################################
############# Plotting Functions
######################################################################



def draw_analytic(ax, Cl, C0, fobs_gw_cents, color='tab:orange', label='Eq. 17 analytic', 
                  alpha=1, lw=2, ls='-.'):
    xx = fobs_gw_cents
    yy = Cl/C0 # (F,)
    hh, = ax.plot(xx, yy, color=color, lw=lw, label=label, linestyle=ls, alpha=alpha)
    return hh

def draw_reals(ax, Cl_many, C0_many, fobs_gw_cents,  color='tab:orange', label= 'Poisson number/bin realization',
                show_ci=False, show_reals=True, show_median=False, nshow=10, lw_median=2, ls_reals = ':'):
    xx = fobs_gw_cents
    yy = Cl_many/C0_many # (F,R)
    if show_median:
        ax.plot(xx, np.median(yy[:,:], axis=-1), color=color, lw=lw_median, alpha=0.75) #, label='median of samples, $l=%d$' % ll)     
    if show_ci:
        for pp in [50, 98]:
            percs = pp/2
            percs = [50-percs, 50+percs]
            ax.fill_between(xx, *np.percentile(yy[:,:], percs, axis=-1), color=color, alpha=0.15)
    if show_reals:
        rr = 0
        ax.plot(xx, yy[:,rr], color=color, alpha=0.15, linestyle=ls_reals, 
                label = label)
        for rr in range(1, np.min([nshow, len(Cl_many[0])])):
            ax.plot(xx, yy[:,rr], color=color, alpha=0.15, linestyle=ls_reals)

def draw_spk(ax, label='SP & K Rough Estimate'):
    spk_xx= np.array([3.5*10**-9, 1.25*10**-8, 1*10**-7]) /YR
    spk_yy= np.array([1*10**-5, 1*10**-3, 1*10**-1])
    ax.plot(spk_xx * YR, spk_yy, label=label, color='limegreen', ls='--')

def draw_bayes(ax, lmax, colors = ['k', 'b', 'r', 'g', 'c', 'm'], ms=8):
    xx_nihan = np.array([2.0, 4.0, 5.9, 7.9, 9.9]) *10**-9 # Hz
    
    ClC0_nihan = np.array([
    [0.20216773, 0.14690035, 0.09676646, 0.07453352, 0.05500382, 0.03177427],
    [0.21201336, 0.14884939, 0.10545698, 0.07734305, 0.05257189, 0.03090662],
    [0.20840993, 0.14836757, 0.09854803, 0.07205384, 0.05409881, 0.03305785],
    [0.19788951, 0.15765126, 0.09615489, 0.07475364, 0.0527356 , 0.03113331],
    [0.20182648, 0.14745265, 0.09681202, 0.0746824 , 0.05503161, 0.0317012 ]])
    for ll in range(lmax):
        ax.plot(xx_nihan, ClC0_nihan[:,ll], 
                    label = '$l=%d$' % (ll+1), 
                color=colors[ll], marker='o', ms=ms)
        
def draw_sim(ax, xx, Cl_best, lmax, nshow, show_ci=True, show_reals=True):

    yy = Cl_best[:,:,:,1:]/Cl_best[:,:,:,0,np.newaxis] # (B,F,R,l)
    yy = np.median(yy, axis=-1) # (B,F,l) median over realizations

    colors = ['k', 'b', 'r', 'g', 'c', 'm']
    for ll in range(lmax):
        ax.plot(xx, np.median(yy[:,:,ll], axis=0), color=colors[ll]) #, label='median of samples, $l=%d$' % ll)
        if show_ci:
            for pp in [50, 98]:
                percs = pp/2
                percs = [50-percs, 50+percs]
                ax.fill_between(xx, *np.percentile(yy[:,:,ll], percs, axis=0), alpha=0.1, color=colors[ll])
        if show_reals:
            for bb in range(0,nshow):
                # if ll==0 and bb==0:
                #     label = "individual best samples, median of realizations"
                # else: 
                label=None
                ax.plot(xx, yy[bb,:,ll], color=colors[ll], linestyle=':', alpha=0.1,
                                 linewidth=1, label=label)


def plot_ClC0_versions(fobs_gw_cents, spk=True, bayes=True, 
              sim=True, Cl_best_sim=None, lmax_sim=None,
              analytic=False, Cl_analytic=None, C0_analytic=None, label_analytic='analytic',
              anreals=False, Cl_anreals=None, C0_anreals=None, label_anreals=None, 
              xmax = 1/YR, leg_anchor=(0,-0.15), leg_cols=3, legend=False):
    fig, ax = plot.figax(xlabel=plot.LABEL_GW_FREQUENCY_HZ, ylabel='$C_{\ell>0}/C_0$')

    if analytic: draw_analytic(ax, Cl_analytic, C0_analytic, fobs_gw_cents, label=label_analytic)
    if anreals: draw_reals(ax, Cl_anreals, C0_anreals, fobs_gw_cents, label=label_anreals)
      
    if bayes: draw_bayes(ax, lmax=6)
    if spk: draw_spk(ax, label='S-P & K')
    if sim and (Cl_best_sim is not None) and (lmax_sim is not None): 
        draw_sim(ax, fobs_gw_cents, Cl_best_sim, lmax_sim, show_ci=True, show_reals=True, nshow=10)
    # ax.set_ylim(10**-6, 10**0)
    # plot._twin_yr(ax, nano=False)
    ax.set_xlim(fobs_gw_cents[0]- 10**(-10), xmax)

    if legend:
        fig.legend(bbox_to_anchor=leg_anchor, loc='upper left', bbox_transform = ax.transAxes, ncols=leg_cols)
    return fig