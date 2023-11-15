"""Gravitational Wave (GW) calculations module.

This module provides tools for calculating GW signals from MBH binaries.
Currently the components here are used with the 'discrete' / 'illustris' population of binaries,
and not the semi-analytic or observational population models.

"""

import numba
import numpy as np

import kalepy as kale

import holodeck as holo
from holodeck import utils, cosmo, log
from holodeck.constants import SPLC, NWTG, MPC


_CALC_MC_PARS = ['mass', 'sepa', 'dadt', 'scafa', 'eccen']


class Grav_Waves:

    def __init__(self, bin_evo, fobs_gw, nharms=103, nreals=100):
        self.fobs_gw = fobs_gw
        self.nharms = nharms
        self.nreals = nreals
        self._bin_evo = bin_evo
        return

    @property
    def freqs(self):
        return self.fobs_gw


class GW_Discrete(Grav_Waves):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._box_vol_cgs = self._bin_evo._sample_volume
        return

    def emit(self, eccen=None, stats=False, progress=True, nloudest=5):
        fobs_gw = self.fobs_gw
        nfreqs = fobs_gw.size
        nharms = self.nharms
        nreals = self.nreals
        bin_evo = self._bin_evo
        box_vol = self._box_vol_cgs

        if eccen is None:
            eccen = (bin_evo.eccen is not None)

        if eccen not in [True, False]:
            raise ValueError("`eccen` '{}' is invalid!".format(eccen))

        loudest = np.zeros((nfreqs, nloudest, nreals))
        fore = np.zeros((nfreqs, nreals))
        back = np.zeros((nfreqs, nreals))
        both = np.zeros((nfreqs, nreals))

        if eccen:
            harm_range = range(1, nharms+1)
        else:
            harm_range = [2]

        harms = np.zeros((nfreqs, nharms))

        freq_iter = enumerate(fobs_gw)
        freq_iter = utils.tqdm(freq_iter, total=len(fobs_gw), desc='GW frequencies') if progress else freq_iter
        for ii, fogw in freq_iter:
            lo = fobs_gw[0] if (ii == 0) else fobs_gw[ii-1]
            hi = fobs_gw[1] if (ii == 0) else fobs_gw[ii]
            dlnf = np.log(hi) - np.log(lo)
            _both, _fore, _back, _loud, _gwb_harms = _gws_harmonics_at_evo_fobs(
                fogw, dlnf, bin_evo, harm_range, nreals, box_vol, loudest=nloudest
            )
            loudest[ii, :] = _loud
            both[ii, :] = _both
            fore[ii, :] = _fore
            back[ii, :] = _back
            harms[ii, :] = _gwb_harms

        self.both = np.sqrt(both)
        self.fore = np.sqrt(fore)
        self.back = np.sqrt(back)
        self.strain = np.sqrt(back + fore)
        self.loudest = loudest
        self.harms = harms
        return


def _gws_harmonics_at_evo_fobs(fobs_gw, dlnf, evo, harm_range, nreals, box_vol, loudest=5):
    """Calculate GW signal at range of frequency harmonics for a single observer-frame GW frequency.

    Parameters
    ----------
    fobs_gw : float
        Observer-frame GW-frequency in units of [1/sec].  This is a single, float value.
    dlnf : float
        Log-width of observered-frequency bin, i.e. $\\Delta \\ln f$.  This is width of observed
        GW frequency bins.
    evo : `holodeck.evolution.Evolution`
        Initialized and evolved binary evolution instance, storing the binary evolution histories
        of each binary.
    harm_range : list[int]
        Harmonics of the orbital-frequency at which to calculate GW emission.  For circular orbits,
        only [2] is needed, as the GW frequency is twice the orbital frequency.  For eccentric
        orbital, GW emission is produced both at harmonic 1 and higher harmonics.  The higher the
        eccentricity the more GW energy is emitted at higher and higher harmonics.
    nreals : int
        Number of realizations to calculate in Poisson sampling.
    box_vol : float
        Volume of the simulation box that the binary population is derived from.  Units of [cm^3].
    loudest : int
        Number of 'loudest' (highest amplitude) strain values to calculate and return separately.

    Returns
    -------
    mc_ecc_both : (R,) ndarray,
        Combined (background + foreground) GW Strain at this frequency, for `R` realizations.
    mc_ecc_fore : (R,) ndarray,
        GW foreground strain (i.e. loudest single source) at this frequency, for `R` realizations.
    mc_ecc_back : (R,) ndarray,
        GW background strain (i.e. all sources except for the loudest) at this frequency, for `R`
        realizations.
    loud : (L, R) ndarray,
        Strains of the `L` loudest binaries (L=`loudest` input parameter) for each realization.
    gwb_harms : (H,)

    """

    # ---- Interpolate data to all harmonics of this frequency
    harm_range = np.asarray(harm_range)
    # (H,) observer-frame orbital-frequency for each harmonic
    fobs_orb = fobs_gw / harm_range
    # Each parameter will be (N, H) = (binaries, harmonics)
    data_harms = evo.at('fobs', fobs_orb, params=_CALC_MC_PARS)

    # Only examine binaries reaching the given locations before redshift zero (other redz=inifinite)
    # (N, H)
    redz = data_harms['scafa']
    redz = cosmo.a_to_z(redz)
    valid = (redz > 0.0)
    # There are 'V' valid == True elements of the (N, H) arrays, such that V <= N*H
    # anytime an (N, H) ndarray is sliced by the `valid` ndarray, it results in a (V,) ndarray

    # Broadcast harmonics numbers to correct shape, (N, H)
    harms_2d = np.ones_like(redz, dtype=int) * harm_range[np.newaxis, :]
    harms_1d = harms_2d[valid]

    # ---- Handle Eccentricities and eccentricity distribution function

    # `None`  or  ndarray shape (N, H)
    eccen = data_harms['eccen']
    # for circular binaries, we should only be consider the n=2 harmonic, and gne(n=2)=1.0
    if eccen is None:
        gne = 1
        assert np.all(harms_2d == 2)

    # If there are eccentricities, calculate the freq-dist-function
    else:
        # (V,) array [i.e. the `valid` slice of (N, H)]
        eccen = eccen[valid]
        gne = utils.gw_freq_dist_func(harms_1d, ee=eccen)

        # Handle (near-)zero eccentricities manually
        # when eccentricity is very low, set all harmonics to zero except for n=2

        # Select the elements corresponding to the n=2 (circular) harmonic, to use later
        # (N, H)
        sel_n2 = np.zeros_like(redz, dtype=bool)
        sel_n2[(harms_2d == 2)] = 1
        # (V,)
        sel_n2 = sel_n2[valid]

        # Select near-zero eccentricities and set the gne values manually
        sel_e0 = (eccen < 1e-12)
        gne[sel_e0] = 0.0
        gne[sel_n2 & sel_e0] = 1.0

    # ---- Calculate GWB

    frst_orb = utils.frst_from_fobs(fobs_orb, redz)
    # Select only the valid elements, also converts to 1D, i.e. (N, H) ==> (V,)
    redz = redz[valid]
    frst_orb = frst_orb[valid]
    # Calculate required parameters for valid binaries (V,)
    dcom = cosmo.z_to_dcom(redz)

    mchirp = data_harms['mass'][valid]
    mchirp = utils.chirp_mass(*mchirp.T)
    # Calculate strains from each source
    hs2 = utils.gw_strain_source(mchirp, dcom, frst_orb)**2

    dfdt, _ = utils.dfdt_from_dadt(data_harms['dadt'][valid], data_harms['sepa'][valid], frst_orb=frst_orb)
    _lambda_fact = utils.lambda_factor_dlnf(frst_orb, dfdt, redz, dcom=dcom) / box_vol
    num_binaries = _lambda_fact * dlnf

    shape = (num_binaries.size, nreals)
    num_pois = poisson_as_needed(num_binaries[:, np.newaxis] * np.ones(shape))

    # --- Calculate GW Signals
    temp = hs2 * gne * (2.0 / harms_1d)**2
    both = np.sum(temp[:, np.newaxis] * num_pois / dlnf, axis=0)

    # Calculate and return the expectation value hc^2 for each harmonic
    # (N, H)
    gwb_harms = np.zeros_like(harms_2d, dtype=float)
    gwb_harms[valid] = temp * num_binaries / dlnf
    # (N, H) ==> (H,)
    gwb_harms = np.sum(gwb_harms, axis=0)

    if np.any(num_pois > 0):
        # Find the L loudest binaries in each realizations
        loud = np.sort(temp[:, np.newaxis] * (num_pois > 0), axis=0)[::-1, :]
        fore = loud[0, :]
        loud = loud[:loudest, :]
    else:
        fore = np.zeros_like(both)
        loud = np.zeros((loudest, nreals))

    back = both - fore
    return both, fore, back, loud, gwb_harms


def _gws_from_samples(vals, weights, fobs_gw_edges):
    """Calculate GW signals at the given frequencies, from weighted samples of a binary population.

    Parameters
    ----------
    vals : (4, N) ndarray of scalar,
        Arrays of binary parameters.
        * vals[0] : mtot [grams]
        * vals[1] : mrat []
        * vals[2] : redz []
        * vals[3] : *observer*-frame *orbital*-frequency of binaries [1/sec]
    weights : (N,) array of scalar,
    fobs_gw_edges : (F,) array of scalar,
        Target observer-frame GW-frequencies to calculate GWs at.  Units of [1/sec].

    Returns
    -------
    gff : (F,) ndarry,
        Observer-frame GW-frequencies of the loudest binary in each bin [1/sec].
    gwf : (F,) ndarry,
        GW Foreground: the characteristic strain of the loudest binary in each frequency bin.
    gwb : (F,) ndarry,
        GW Background: the characteristic strain of the GWB in each frequency bin.
        Does not include the strain from the loudest binary in each bin (`gwf`).

    """
    # `fogw` is observer-frame GW-frequencies of binary samples
    hs, fogw = _strains_from_samples(vals)
    gff, gwf, gwb = gws_from_sampled_strains(fobs_gw_edges, fogw, hs, weights)
    return gff, gwf, gwb


def _strains_from_samples(vals):
    """From a sampled binary population, calculate the GW strains.

    Parameters
    ----------
    vals : (4,) array_like of array_like,
        Each element of `vals` is an array of binary parameters, the elements must be:
        * 0) total binary mass [grams]
        * 1) binary mass-ratio [],
        * 2) redshift at this frequency [],
        * 3) *observer*-frame *orbital*-frequency [1/sec].

    Returns
    -------
    hs : (N,) ndarray,
        Source strains (i.e. not characteristic strains) of each binary.
    fobs_gw : (N,) ndarray,
        Observer-frame GW-frequencies of each sampled binary.  [1/sec].

    """

    mc = utils.chirp_mass(*utils.m1m2_from_mtmr(vals[0], vals[1]))

    rz = vals[2]
    dc = cosmo.comoving_distance(rz).cgs.value

    fobs_orb = vals[3]
    frst_orb = utils.frst_from_fobs(fobs_orb, rz)
    hs = utils.gw_strain_source(mc, dc, frst_orb)

    fobs_gw = fobs_orb * 2.0
    return hs, fobs_gw


@numba.njit
def gws_from_sampled_strains(fobs_gw_edges, fo, hs, weights):
    """Calculate GW background/foreground from sampled GW strains.

    Parameters
    ----------
    fobs_gw_edges : (F,) array_like of scalar
        Observer-frame GW-frequency bin edges.
    fo : (S,) array_like of scalar
        Observer-frame GW-frequency of each binary sample.  Units of [1/sec]
    hs : (S,) array_like of scalar
        GW source strain (*not characteristic strain*) of each binary sample.
    weights : (S,) array_like of int
        Weighting factor for each binary.
        NOTE: the GW calculation is ill-defined if weights have fractional values
        (i.e. float values, instead of integral values; but the type itself doesn't matter)

    Returns
    -------
    gwf_freqs : (F,) ndarray of scalar
        Observer-frame GW frequency of foreground sources in each frequency bin.  Units of [1/sec].
    gwfore : (F,) ndarray of scalar
        Strain amplitude of foreground sources in each frequency bin.
    gwback : (F,) ndarray of scalar
        Strain amplitude of the background in each frequency bin.

    """

    # ---- Initialize
    num_samp = fo.size                 # number of binaries/samples
    num_freq = fobs_gw_edges.size - 1           # number of frequency bins (edges - 1)
    gwback = np.zeros(num_freq)        # store GWB characteristic strain
    gwfore = np.zeros(num_freq)        # store loudest binary characteristic strain, for each bin
    gwf_freqs = np.zeros(num_freq)     # store frequency of loudest binary, for each bin

    # ---- Sort input by frequency for faster iteration
    idx = np.argsort(fo)
    fo = fo[idx]
    hs = hs[idx]
    weights = weights[idx]

    # ---- Calculate GW background and foreground in each frequency bin
    ii = 0
    lo = fobs_gw_edges[0]
    for ff in range(num_freq):
        # upper-bound to this frequency bin
        hi = fobs_gw_edges[ff+1]
        # number of GW cycles (1/dlnf), for conversion to characteristic strain
        # dlnf = (np.log(hi) - np.log(lo))
        df = (hi - lo)
        # amplitude and frequency of the loudest source in this bin
        hmax = 0.0
        fmax = 0.0

        # iterate over all sources with frequencies below this bin's limit (right edge)
        while (ii < num_samp) and (fo[ii] < hi):
            # Store the amplitude and frequency of loudest source
            #    NOTE: loudest source could be a single-sample (weight==1) or from a weighted-bin (weight > 1)
            #          the max
            if (weights[ii] >= 1) and (hs[ii] > hmax):
                hmax = hs[ii]
                fmax = fo[ii]

            h2temp = weights[ii] * (hs[ii] ** 2) * fo[ii]
            gwback[ff] += h2temp

            # increment binary index
            ii += 1

        # subtract foreground source from background
        gwf_freqs[ff] = fmax
        gwback[ff] -= ((hmax**2) * fmax)
        # Convert to *characteristic* strain
        # gwback[ff] = gwback[ff] / dlnf      # hs^2 ==> hc^2  (squared, so dlnf^-1)
        # gwfore[ff] = hmax / np.sqrt(dlnf)   # hs ==> hc (not squared, so sqrt of 1/dlnf)
        gwback[ff] = gwback[ff] / df      # hs^2 ==> hc^2  (squared, so df^-1)
        gwfore[ff] = hmax * np.sqrt(fmax / df)   # hs ==> hc (not squared, so sqrt of 1/df)
        lo = hi

    gwback = np.sqrt(gwback)
    return gwf_freqs, gwfore, gwback


def sampled_gws_from_sam(sam, fobs_gw, hard=holo.hardening.Hard_GW, **kwargs):
    """Sample the given binary population between the target frequencies, and calculate GW signals.

    NOTE: the input `fobs` are interpretted as bin edges, and GW signals are calculate within the
    corresponding bins.

    Parameters
    ----------
    sam : `Semi_Analytic_Model` instance,
        Binary population to sample.
    fobs_gw : (F+1,) array_like,
        Target observer-frame GW-frequencies of interest in units of [1/sec]
    hard : `holodeck.evolution._Hardening` instance,
        Binary hardening model used to calculate binary residence time at each frequency.
    kwargs : dict,
        Additional keyword-arguments passed to `sample_sam_with_hardening()`

    Returns
    -------
    gff : (F,) ndarry,
        Observer-frame GW-frequencies of the loudest binary in each bin [1/sec].
    gwf : (F,) ndarry,
        GW Foreground: the characteristic strain of the loudest binary in each frequency bin.
    gwb : (F,) ndarry,
        GW Background: the characteristic strain of the GWB in each frequency bin.
        Does not include the strain from the loudest binary in each bin (`gwf`).

    """
    fobs_orb = fobs_gw / 2.0
    vals, weights, edges, dens, mass = holo.sam.sample_sam_with_hardening(sam, hard, fobs=fobs_orb, **kwargs)
    gff, gwf, gwb = _gws_from_samples(vals, weights, fobs_gw)
    return gff, gwf, gwb


def _gws_from_number_grid_integrated_redz(edges, redz, number, realize, sum=True):
    """

    Parameters
    ----------
    edges : (4,) list of 1darrays
        A list containing the edges along each dimension.  The four dimensions correspond to
        total mass, mass ratio, redshift, and observer-frame orbital frequency.
        The length of each of the four arrays is M, Q, Z, F.
    redz :
    number : (M-1, Q-1, Z-1, F-1) ndarray
        The number of binaries in each bin of parameter space.  This is calculated by integrating
        `dnum` over each bin.
    realize : bool or int,
        Specification of how to construct one or more discrete realizations.
        If a `bool` value, then whether or not to construct a realization.
        If an `int` value, then how many discrete realizations to construct.
    sum : bool,
        Whether or not to sum over axes {0, 1, 2}.

    Returns
    -------
    hc : ndarray
        Characteristic strain of the GWB.
        The shape depends on whether `sum` is true or false.
        sum = True:  shape is (F-1,)
        sum = False: shape is (M-1, Q-1, Z-1, F-1)

    """

    hc2 = char_strain_sq_from_bin_edges_redz(edges, redz)

    # Create a single realization
    if realize is True:
        hc2 = hc2 * poisson_as_needed(number)
        # Sum over M, Q, Z bins  ::  (M-1, Q-1, Z-1, F-1 [, R]) ==> (F-1, [, R])
        if sum:
            hc2 = np.sum(hc2, axis=(0, 1, 2))

    # Do not create a discrete realization, use the expectation values directly
    elif realize in [None, False]:
        hc2 = hc2 * number
        # Sum over M, Q, Z bins  ::  (M-1, Q-1, Z-1, F-1 [, R]) ==> (F-1, [, R])
        if sum:
            hc2 = np.sum(hc2, axis=(0, 1, 2))

    # Create multiple discrete realizations
    elif utils.isinteger(realize):
        if sum:
            import holodeck.cyutils   # noqa
            # This function reate
            hc2 = holo.cyutils.sam_poisson_gwb(number, hc2, realize)

        else:
            log.warning(f"`sum`={sum} :: this requires a large amount of memory!")
            shape = number.shape + (realize,)
            hc2 = hc2[..., np.newaxis] * poisson_as_needed(number[..., np.newaxis] * np.ones(shape))
            if holo.sam._DEBUG:
                log.info(f"number = {utils.stats(number)}")
                log.info(f"hc2 = {utils.stats(hc2)}")
                holo.sam._check_bads(edges + [np.arange(realize),], hc2, "hc2")

    else:
        err = "`realize` ({}) must be one of {{True, False, integer}}!".format(realize)
        log.error(err)
        raise ValueError(err)

    # convert from hc^2 to hc
    hc2 = np.sqrt(hc2)
    # this is for clarity, note that it does not duplicate the memory
    hc = hc2

    return hc


def _gws_from_number_grid_integrated(edges, number, realize, sum=True):
    """

    Parameters
    ----------
    edges : (4,) list of 1darrays
        A list containing the edges along each dimension.  The four dimensions correspond to
        total mass, mass ratio, redshift, and observer-frame orbital frequency.
        The length of each of the four arrays is M, Q, Z, F.
    number : (M-1, Q-1, Z-1, F-1) ndarray
        The number of binaries in each bin of parameter space.  This is calculated by integrating
        `dnum` over each bin.
    realize : bool or int,
        Specification of how to construct one or more discrete realizations.
        If a `bool` value, then whether or not to construct a realization.
        If an `int` value, then how many discrete realizations to construct.
    sum : bool,
        Whether or not to sum over axes {0, 1, 2}.

    Returns
    -------
    hc : ndarray
        Characteristic strain of the GWB.
        The shape depends on whether `sum` is true or false.
        sum = True:  shape is (F-1,)
        sum = False: shape is (M-1, Q-1, Z-1, F-1)

    """

    hc2 = char_strain_sq_from_bin_edges(edges)

    # Create a single realization
    if realize is True:
        hc2 = hc2 * poisson_as_needed(number)
        # Sum over M, Q, Z bins  ::  (M-1, Q-1, Z-1, F-1 [, R]) ==> (F-1, [, R])
        if sum:
            hc2 = np.sum(hc2, axis=(0, 1, 2))

    # Do not create a discrete realization, use the expectation values directly
    elif realize in [None, False]:
        hc2 = hc2 * number
        # Sum over M, Q, Z bins  ::  (M-1, Q-1, Z-1, F-1 [, R]) ==> (F-1, [, R])
        if sum:
            hc2 = np.sum(hc2, axis=(0, 1, 2))

    # Create multiple discrete realizations
    elif utils.isinteger(realize):
        if sum:
            import holodeck.cyutils   # noqa
            # This function reate
            hc2 = holo.cyutils.sam_poisson_gwb(number, hc2, realize)

        else:
            log.warning(f"`sum`={sum} :: this requires a large amount of memory!")
            shape = number.shape + (realize,)
            hc2 = hc2[..., np.newaxis] * poisson_as_needed(number[..., np.newaxis] * np.ones(shape))
            if holo.sam._DEBUG:
                log.info(f"number = {utils.stats(number)}")
                log.info(f"hc2 = {utils.stats(hc2)}")
                holo.sam._check_bads(edges + [np.arange(realize),], hc2, "hc2")

    else:
        err = "`realize` ({}) must be one of {{True, False, integer}}!".format(realize)
        log.error(err)
        raise ValueError(err)

    # convert from hc^2 to hc
    hc2 = np.sqrt(hc2)
    # this is for clarity, note that it does not duplicate the memory
    hc = hc2

    return hc


def gwb_ideal(fobs_gw, ndens, mtot, mrat, redz, dlog10, sum=True):

    const = ((4.0 * np.pi) / (3 * SPLC**2))
    mc = utils.chirp_mass_mtmr(mtot, mrat)
    mc = np.power(NWTG * mc, 5.0/3.0)
    rz = np.power(1 + redz, -1.0/3.0)
    fogw = np.power(np.pi * fobs_gw, -4.0/3.0)

    integ = ndens * mc * rz
    redz = redz * np.ones_like(integ)
    integ[redz <= 0.0] = 0.0

    arguments = [mtot, mrat, redz]
    if dlog10:
        arguments[0] = np.log10(arguments[0])

    for ax, xx in enumerate(arguments):
        integ = np.moveaxis(integ, ax, 0)
        xx = np.moveaxis(xx, ax, 0)

        # if integ is (X, A, B) and xx is (X, 1, 1), then this is fine
        try:
            integ = 0.5 * (integ[:-1] + integ[1:]) * np.diff(xx, axis=0)
        # BUT if integ is (X, A, B) and xx is (X, A+1, B+1), then need to average xx values down
        except ValueError:
            # average other dimensions as needed
            for jj in range(1, len(arguments)):
                sh = np.shape(xx)[jj]
                if (sh == 1) or (sh == np.shape(integ)[jj]):
                    continue

                xx = np.moveaxis(xx, jj, 0)
                xx = 0.5 * (xx[:-1] + xx[1:])
                xx = np.moveaxis(xx, 0, jj)

            # try integration step again
            integ = 0.5 * (integ[:-1] + integ[1:]) * np.diff(xx, axis=0)

        # return integ to the correct shape (axis order)
        integ = np.moveaxis(integ, 0, ax)

    gwb = const * fogw
    gwb = gwb * np.sum(integ) if sum else gwb * integ
    gwb = np.sqrt(gwb)
    return gwb


def poisson_as_needed(values, thresh=1e10):
    """Calculate Poisson distribution when values are below threshold, otherwise approximate with normal distribution.

    Parameters
    ----------
    values : ndarray
        Expectation values for poisson distribution.
    thresh : float
        Expectation value above which to use Normal distribution approximation.

    Returns
    -------
    output : ndarray
        (Approximately) Poisson distributed values.
        Same shape as input `values`.

    """
    # NOTE: do not use `int` type as it can cause overflow errors
    # output = np.zeros_like(values, dtype=int)
    output = np.zeros_like(values)
    idx = (values <= thresh)
    output[idx] = np.random.poisson(values[idx])
    tt = values[~idx]
    # output[~idx] = np.floor(np.random.normal(tt, np.sqrt(tt))).astype(int)
    output[~idx] = np.floor(np.random.normal(tt, np.sqrt(tt)))
    return output


def char_strain_sq_from_bin_edges_redz(edges, redz):
    assert len(edges) == 4
    assert np.all([np.ndim(ee) == 1 for ee in edges])

    foo = edges[-1]                   #: should be observer-frame orbital-frequencies
    df = np.diff(foo)                 #: frequency bin widths
    fc = kale.utils.midpoints(foo)    #: use frequency-bin centers for strain (more accurate!)

    # redshifts are defined across 4D grid, shape (M, Q, Z, Fc)
    #    where M, Q, Z are edges and Fc is frequency centers
    # find midpoints of redshifts in M, Q, Z dimensions, to end up with (M-1, Q-1, Z-1, Fc)
    for dd in range(3):
        redz = np.moveaxis(redz, dd, 0)
        redz = kale.utils.midpoints(redz, axis=0)
        redz = np.moveaxis(redz, 0, dd)

    # ---- calculate GW strain ----
    mt = kale.utils.midpoints(edges[0])
    mr = kale.utils.midpoints(edges[1])
    # rz = kale.utils.midpoints(edges[2])
    mc = utils.chirp_mass_mtmr(mt[:, np.newaxis], mr[np.newaxis, :])
    mc = mc[:, :, np.newaxis, np.newaxis]
    dc = +np.inf * np.ones_like(redz)
    sel = (redz > 0.0)
    dc[sel] = cosmo.comoving_distance(redz[sel]).cgs.value

    # convert from observer-frame to rest-frame; still using frequency-bin centers
    fr = utils.frst_from_fobs(fc[np.newaxis, np.newaxis, np.newaxis, :], redz)

    hs = utils.gw_strain_source(mc, dc, fr)
    hc2 = (hs ** 2) * (fc / df)
    return hc2

def strain_amp_from_bin_edges_redz(edges, redz):
    assert len(edges) == 4
    assert np.all([np.ndim(ee) == 1 for ee in edges])

    foo = edges[-1]                   #: should be observer-frame orbital-frequencies
    # df = np.diff(foo)                 #: frequency bin widths
    fc = kale.utils.midpoints(foo)    #: use frequency-bin centers for strain (more accurate!)

    # redshifts are defined across 4D grid, shape (M, Q, Z, Fc)
    #    where M, Q, Z are edges and Fc is frequency centers
    # find midpoints of redshifts in M, Q, Z dimensions, to end up with (M-1, Q-1, Z-1, Fc)
    for dd in range(3):
        redz = np.moveaxis(redz, dd, 0)
        redz = kale.utils.midpoints(redz, axis=0)
        redz = np.moveaxis(redz, 0, dd)

    # ---- calculate GW strain ----
    mt = kale.utils.midpoints(edges[0])
    mr = kale.utils.midpoints(edges[1])
    # rz = kale.utils.midpoints(edges[2])
    mc = utils.chirp_mass_mtmr(mt[:, np.newaxis], mr[np.newaxis, :])
    mc = mc[:, :, np.newaxis, np.newaxis]
    dc = +np.inf * np.ones_like(redz)
    sel = (redz > 0.0)
    dc[sel] = cosmo.comoving_distance(redz[sel]).cgs.value

    # convert from observer-frame to rest-frame; still using frequency-bin centers
    fr = utils.frst_from_fobs(fc[np.newaxis, np.newaxis, np.newaxis, :], redz)

    hs = utils.gw_strain_source(mc, dc, fr)
    return hs


def char_strain_sq_from_bin_edges(edges):
    assert len(edges) == 4
    assert np.all([np.ndim(ee) == 1 for ee in edges])

    foo = edges[-1]                   #: should be observer-frame orbital-frequencies
    df = np.diff(foo)                 #: frequency bin widths
    fc = kale.utils.midpoints(foo)    #: use frequency-bin centers for strain (more accurate!)

    # ---- calculate GW strain ----
    mt = kale.utils.midpoints(edges[0])
    mr = kale.utils.midpoints(edges[1])
    rz = kale.utils.midpoints(edges[2])
    mc = utils.chirp_mass_mtmr(mt[:, np.newaxis], mr[np.newaxis, :])
    mc = mc[:, :, np.newaxis, np.newaxis]
    dc = cosmo.comoving_distance(rz).cgs.value
    dc = dc[np.newaxis, np.newaxis, :, np.newaxis]

    # convert from observer-frame to rest-frame; still using frequency-bin centers
    fr = utils.frst_from_fobs(fc[np.newaxis, :], rz[:, np.newaxis])
    fr = fr[np.newaxis, np.newaxis, :, :]

    hs = utils.gw_strain_source(mc, dc, fr)
    hc2 = (hs ** 2) * (fc / df)
    return hc2


# ==============================================================================
# ====    SAM GW Functions    ====
# ==============================================================================


#! NOTE: THIS IS SLOW PYTHON IMPLEMENTATION FOR TESTING.  USE `holodeck.cytuls.sam_calc_gwb_single_eccen()` !#

def _python_sam_calc_gwb_single_eccen(gwfobs, sam, sepa_evo, eccen_evo, nharms=100):
    """

    Parameters
    ----------
    gwfobs : (F,) array_like
        Observer-frame frequencies at which to calculate GWB.
    sam : `Semi_Analytic_Model` instance
    forb_rst_evo : (M, E) array_like
        Rest-frame orbital frequencies of binaries, for each total-mass M and evolution step E.
    eccen_evo : (E,) array_like
        Eccentricities at each evolution step.  The same for all binaries, corresponding to fixed
        binary separations for all binaries.
    nharms : int
        Number of harmonics to use in calculating GWB.

    """

    # NOTE: need to check for coalescences and set to zero
    # NOTE: need to check for frequencies below starting separation and set to zero

    frst_orb_evo = utils.kepler_freq_from_sepa(sam.mtot[:, np.newaxis], sepa_evo[np.newaxis, :])

    assert np.ndim(gwfobs) == 1
    assert np.ndim(frst_orb_evo) == 2
    assert np.ndim(eccen_evo) == 1
    assert np.shape(frst_orb_evo) == (sam.mtot.size, eccen_evo.size)

    harm_nums = np.arange(1, nharms+1)
    two_over_nh_sq = (2.0 / harm_nums) ** 2

    # (M, Q, Z) units of [Mpc^-3]
    ndens = sam.static_binary_density

    # (F, H)
    gwfobs_harms = gwfobs[:, np.newaxis] / harm_nums[np.newaxis, :]

    # (Z,)
    dcom = cosmo.comoving_distance(sam.redz).to('Mpc').value

    # (Z, F, H)
    # gw_frst ==> frst_orb_harms
    # gw_frst = gwfobs_harms[np.newaxis, :, :] * (1.0 * sam.redz[:, np.newaxis, np.newaxis])

    # shape will be a tuple of (M, Q, Z, F, H)
    shape = sam.shape + np.shape(gwfobs_harms)
    # setup output arrays with shape (M, Q, Z, F, H)
    hc2 = np.zeros(shape)
    hs2 = np.zeros(shape)
    hsn2 = np.zeros(shape)
    tau_out = np.zeros(shape)
    ecc_out = np.zeros(shape)

    gwfr_check = np.zeros(shape[2:])

    # NOTE: should sort `gwfobs_harms` into an ascending 1D array to speed up processes

    for (aa, bb), gwfo in np.ndenumerate(gwfobs_harms):
        # iterate over mtot M
        for ii, mt in enumerate(sam.mtot):
            # (Q,) masses of each component for this total-mass, and all mass-ratios
            m1, m2 = utils.m1m2_from_mtmr(mt, sam.mrat)
            mchirp = utils.chirp_mass(m1, m2)

            # (E,) rest-frame orbital frequencies for this total-mass bin
            frst_evo = frst_orb_evo[ii]
            # iterate over redshifts Z
            for kk, zz in enumerate(sam.redz):
                # () scalar
                zterm = (1.0 + zz)
                dc = dcom[kk]   # this is still in units of [Mpc]
                dc_term = 4*np.pi*(SPLC/MPC) * (dc**2)
                # rest-frame frequency corresponding to target observer-frame frequency of GW observations
                gwfr = gwfo * zterm
                if ii > 0:
                    assert gwfr_check[kk, aa, bb] == gwfr
                else:
                    gwfr_check[kk, aa, bb] = gwfr
                sa = utils.kepler_sepa_from_freq(mt, gwfr)

                # interpolate to target (rest-frame) frequency
                # this is the same for all mass-ratios
                # () scalar
                ecc = np.interp(gwfr, frst_evo, eccen_evo, left=np.nan, right=np.nan)
                # ecc_2 = np.interp(sa, sepa[::-1], eccen_evo[::-1], left=np.nan, right=np.nan)

                # da/dt values are negative, get a positive rate
                tau = -utils.gw_hardening_rate_dadt(m1, m2, sa, ecc)
                # convert to timescale
                tau = sa / tau
                # print(f"{m1.shape")
                tau_out[ii, :, kk, aa, bb] = tau
                ecc_out[ii, :, kk, aa, bb] = ecc

                # Calculate the GW spectral strain at each harmonic
                #    see: [Amaro-seoane+2010 Eq.9]
                # ()
                temp = two_over_nh_sq[bb] * utils.gw_freq_dist_func(harm_nums[bb], ee=ecc, recursive=False)
                # (Q,)
                hs2[ii, :, kk, aa, bb] = utils.gw_strain_source(mchirp, dc*MPC, gwfr) ** 2
                hsn2[ii, :, kk, aa, bb] = temp * hs2[ii, :, kk, aa, bb]

                # (Q,)
                hc2[ii, :, kk, aa, bb] = ndens[ii, :, kk] * dc_term * zterm * tau * hsn2[ii, :, kk, aa, bb]

    # integrate
    gwb = hc2.copy()
    args = [np.log10(sam.mtot), sam.mrat, sam.redz]
    for ii, xx in enumerate(args):
        gwb = np.moveaxis(gwb, ii, 0)
        dx = np.diff(xx)
        gwb = dx * 0.5 * np.moveaxis(gwb[1:] + gwb[:-1], 0, -1)
        gwb = np.moveaxis(gwb, -1, ii)

    gwb = np.sum(gwb, axis=(0, 1, 2))

    # return gwfobs_harms, gwfr_check, gwb, hsn2, hs2, ecc_out, tau_out
    # return gwfobs_harms, gwb, ecc_out, tau_out
    return gwfobs_harms, gwb, ecc_out, tau_out


def sam_calc_gwb_single_eccen(gwfobs, sam, sepa_evo, eccen_evo, nharms=100):
    import holodeck.cyutils  # noqa

    ndens = sam.static_binary_density
    mt_l10 = np.log10(sam.mtot)
    mr = sam.mrat
    rz = sam.redz
    dc = cosmo.comoving_distance(sam.redz).to('Mpc').value
    gwb = holo.cyutils.sam_calc_gwb_single_eccen(ndens, mt_l10, mr, rz, dc, gwfobs, sepa_evo, eccen_evo, nharms)
    return np.asarray(gwb)


def sam_calc_gwb_single_eccen_discrete(gwfobs, sam, sepa_evo, eccen_evo, nharms=100, nreals=None):
    import holodeck.cyutils  # noqa

    ndens = sam.static_binary_density
    mt_l10 = np.log10(sam.mtot)
    mr = sam.mrat
    rz = sam.redz
    dc = cosmo.comoving_distance(sam.redz).to('Mpc').value
    if nreals is None:
        nreals = 1
        squeeze = True
    else:
        squeeze = False

    gwb = holo.cyutils.sam_calc_gwb_single_eccen_discrete(ndens, mt_l10, mr, rz, dc, gwfobs, sepa_evo, eccen_evo, nharms, nreals)

    if squeeze:
        gwb = gwb.squeeze()

    return np.asarray(gwb)


# ==============================================================================
# ====    Deprecated Functions    ====
# ==============================================================================


@utils.deprecated_fail(_gws_harmonics_at_evo_fobs)
def _calc_mc_at_fobs(*args, **kwargs):
    return


def _gws_from_number_grid_centroids(edges, dnum, number, realize):
    """Calculate GWs based on a grid of number-of-binaries.

    # ! BUG: THIS ASSUMES THAT FREQUENCIES ARE NYQUIST SAMPLED !
    # ! otherwise the conversion from hs to hc doesnt work !

    NOTE: `_gws_from_number_grid_integrated()` should be more accurate, but this method better
    matches GWB from sampled (`kale.sample_`) populations!!

    The input number of binaries is `N` s.t. $$N = (d^4 N / [dlog10(M) dq dz dlogf] ) * dlog10(M) dq dz dlogf$$
    The number `N` is evaluated on a 4d grid, specified by `edges`, i.e. $$N = N(M, q, z, f_r)$$
    NOTE: the provided `number` must also summed/integrated over dlogf.
    To calculate characteristic strain, this function divides again by the dlogf term.

    Parameters
    ----------
    edges : (4,) iterable of array_like,
        The edges of each dimension of the parameter space.
        The edges should be, in order: [mtot, mrat, redz, fobs],
        In units of [grams], [], [], [1/sec].
    dnum : (M, Q, Z, F) ndarray,
        Differential comoving number-density of binaries in each bin.
    number : (M, Q, Z, F) ndarray,
        Volumetric comoving number-density of binaries in each bin.
    realize : bool or int,
        Whether or not to calculate one or multiple realizations of the population.
        BUG: explain more.

    Returns
    -------
    hc : (M',Q',Z',F) ndarray,
        Total characteristic GW strain from each bin of parameter space.
        NOTE: to get total strain from all bins, must sum in quarature!
        e.g. ``gwb = np.sqrt(np.square(hc).sum())``

    """

    # # ---- find 'center-of-mass' of each bin (i.e. based on grid edges)
    # # (3, M', Q', Z')
    # # coms = self.grid
    # # ===> (3, M', Q', Z', 1)
    # coms = [cc[..., np.newaxis] for cc in grid]
    # # ===> (4, M', Q', Z', F)
    # coms = np.broadcast_arrays(*coms, fobs[np.newaxis, np.newaxis, np.newaxis, :])

    # # ---- find weighted bin centers
    # # get unweighted centers
    # cent = kale.utils.midpoints(dnum, log=False, axis=(0, 1, 2, 3))
    # # get weighted centers for each dimension
    # for ii, cc in enumerate(coms):
    #     coms[ii] = kale.utils.midpoints(dnum * cc, log=False, axis=(0, 1, 2, 3)) / cent
    # print(f"{kale.utils.jshape(edges)=}, {dnum.shape=}")
    coms = kale.utils.centroids(edges, dnum)

    # ---- calculate GW strain at bin centroids
    mc = utils.chirp_mass(*utils.m1m2_from_mtmr(coms[0], coms[1]))
    dc = cosmo.comoving_distance(coms[2]).cgs.value

    # ! -- 2022-08-19: `edges` should already be using *orbital*-frequency
    fr = utils.frst_from_fobs(coms[3], coms[2])
    # ! old:
    # convert from GW frequency to orbital frequency (divide by 2.0)
    # hs = utils.gw_strain_source(mc, dc, fr/2.0)
    # ! new:
    hs = utils.gw_strain_source(mc, dc, fr)
    # ! --

    # NOTE: for `dlogf` it doesnt matter if these are orbital- or GW- frequencies
    dlogf = np.diff(np.log(edges[-1]))
    dlogf = dlogf[np.newaxis, np.newaxis, np.newaxis, :]

    if realize is True:
        number = np.random.poisson(number)
    elif realize in [None, False]:
        pass
    elif utils.isinteger(realize):
        shape = number.shape + (realize,)
        number = np.random.poisson(number[..., np.newaxis], size=shape)
        hs = hs[..., np.newaxis]
        dlogf = dlogf[..., np.newaxis]
    else:
        err = "`realize` ({}) must be one of {{True, False, integer}}!".format(realize)
        raise ValueError(err)

    number = number / dlogf
    hs = np.nan_to_num(hs)
    hc = number * np.square(hs)

    # # (M',Q',Z',F) ==> (F,)
    # if integrate:
    #     hc = np.sqrt(np.sum(hc, axis=(0, 1, 2)))

    hc = np.sqrt(hc)

    return hc
