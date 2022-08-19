"""Gravitational Wave (GW) calculations module.

This module provides tools for calculating GW signals from MBH binaries.
Currently the components here are used with the 'discrete' / 'illustris' population of binaries,
and not the semi-analytic or observational population models.

"""

import numpy as np

from holodeck import utils, cosmo
from holodeck.constants import SPLC, MPC, MSOL


_CALC_MC_PARS = ['mass', 'sepa', 'dadt', 'scafa', 'eccen']


class Grav_Waves:

    def __init__(self, bin_evo, fobs_gw, nharms=30, nreals=100):
        self.fobs_gw = fobs_gw
        self.nharms = nharms
        self.nreals = nreals
        self._bin_evo = bin_evo
        return


class GW_Discrete(Grav_Waves):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._box_vol_cgs = self._bin_evo._sample_volume
        return

    def emit(self, eccen=None, stats=False, progress=True, nloudest=5):
        fobs_gw = self.fobs_gw
        nharms = self.nharms
        nreals = self.nreals
        bin_evo = self._bin_evo
        box_vol = self._box_vol_cgs

        if eccen is None:
            eccen = (bin_evo.eccen is not None)

        if eccen not in [True, False]:
            raise ValueError("`eccen` '{}' is invalid!".format(eccen))

        loudest = np.zeros((fobs_gw.size, nloudest, nreals))
        fore = np.zeros((fobs_gw.size, nreals))
        back = np.zeros((fobs_gw.size, nreals))
        both = np.zeros((fobs_gw.size, nreals))

        if eccen:
            harm_range = range(1, nharms+1)
        else:
            harm_range = [2]

        freq_iter = enumerate(fobs_gw)
        freq_iter = utils.tqdm(freq_iter, total=len(fobs_gw), desc='GW frequencies') if progress else freq_iter
        for ii, fogw in freq_iter:
            _both, _fore, _back, _loud = _calc_mc_at_fobs(
                fogw, harm_range, nreals, bin_evo, box_vol, loudest=nloudest
            )
            loudest[ii, :] = _loud
            both[ii, :] = _both
            fore[ii, :] = _fore
            back[ii, :] = _back

        self.both = np.sqrt(both)
        self.fore = np.sqrt(fore)
        self.back = np.sqrt(back)
        self.strain = np.sqrt(back + fore)
        self.loudest = loudest
        return


class GW_Continuous(Grav_Waves):

    def emit(self, eccen=None, stats=False, progress=True, nloudest=5):
        fobs_gw = self.fobs_gw
        bin_evo = self._bin_evo
        pop = bin_evo._pop
        weight = pop.weight
        dm = np.log10(pop._mtot[1]/MSOL) - np.log10(pop._mtot[0]/MSOL)
        dq = pop._mrat[1] - pop._mrat[0]
        dz = pop._redz[1] - pop._redz[0]

        # (N,) ==> (1, N)    for later conversion to (F, N)
        m1, m2 = [mm[np.newaxis, :] for mm in pop.mass.T]
        mchirp = utils.chirp_mass(m1, m2)

        H0 = cosmo.H0*1e5 / MPC   # convert from [km/s/Mpc] to [1/s]
        redz = cosmo.a_to_z(pop.scafa)                     # (N,)
        redz = np.clip(redz, 0.1, None)
        dlum = cosmo.luminosity_distance(redz).cgs.value
        dzdt = H0 * cosmo.efunc(redz) * np.square(1.0 + redz)  # (N,)

        # ==> shape (F,N)
        frest = fobs_gw[:, np.newaxis] / (1.0 + redz[np.newaxis, :])   # rest-frame GW frequency
        temp, _ = utils.gw_hardening_rate_dfdt(m1, m2, frest)
        dtr_dlnfr = frest / temp
        # Calculate source-strain for each source (h;  NOT characteristic strain)
        #     convert from rest-frame GW frequency to orbital frequency (divide by 2)
        strain = utils.gw_strain_source(mchirp, dlum[np.newaxis, :], frest/2.0)

        time_fac = dzdt * dtr_dlnfr

        # Convert to characteristic-strain (squared)
        strain = weight[np.newaxis, :] * time_fac * strain**2

        dvol = dm * dq * dz
        # Sum over all binaries, convert from hc^2 ==> hc
        strain = np.sqrt(np.sum(strain * dvol, axis=-1))
        self.strain = strain
        return


'''
def _calc_mc_at_fobs(fogw, _harms, nreals, bin_evo, box_vol, loudest=5):
    """
    """
    fo_orb = fo_gw / 2.0
    data_harms = bin_evo.at('fobs', fo_orb, params=_CALC_MC_PARS)

    redz = cosmo.a_to_z(data_harms['scafa'])
    valid = (redz > 0.0)
    redz = redz[valid]
    dcom = cosmo.z_to_dcom(redz)
    zp1 = redz + 1
    fr_orb = utils.frst_from_fobs(fo_orb, redz)
    mchirp = data_harms['mass'][valid]
    mchirp = utils.chirp_mass(*mchirp.T)
    hs2 = utils.gw_strain_source(mchirp, dcom, fr_orb)**2

    dfdt, _ = utils.dfdt_from_dadt(data_harms['dadt'][valid], data_harms['sepa'][valid], frst_orb=fr_orb)
    tfac = fr_orb / dfdt
    vfac = 4.0*np.pi*SPLC * zp1 * dcom**2 / box_vol

    num_frac = vfac * tfac
    num_pois = np.random.poisson(num_frac)
    both = np.sum(hs2 * num_pois) * np.ones(nreals)
    return both, np.zeros_like(both), np.zeros_like(both), np.zeros((loudest, nreals))
'''


def _calc_mc_at_fobs(fobs_gw, harm_range, nreals, bin_evo, box_vol, loudest=5):
    """Calculate GW signal at range of frequency harmonics for a single observer-frame GW frequency.

    Parameters
    ----------
    fobs_gw : float
        Observer-frame GW-frequency in units of [1/sec].  This is a single, float value.
    harm_range : list[int]
        Harmonics of the orbital-frequency at which to calculate GW emission.  For circular orbits,
        only [2] is needed, as the GW frequency is twice the orbital frequency.  For eccentric
        orbital, GW emission is produced both at harmonic 1 and higher harmonics.  The higher the
        eccentricity the more GW energy is emitted at higher and higher harmonics.
    nreals : int
        Number of realizations to calculate in Poisson sampling.
    bin_evo : `holodeck.evolution.Evolution`
        Initialized and evolved binary evolution instance, storing the binary evolution histories
        of each binary.
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

    """

    # ---- Interpolate data to all harmonics of this frequency
    harm_range = np.asarray(harm_range)
    # (H,) observer-frame orbital-frequency for each harmonic
    fobs_orb = fobs_gw / harm_range
    # Each parameter will be (N, H) = (binaries, harmonics)
    data_harms = bin_evo.at('fobs', fobs_orb, params=_CALC_MC_PARS)

    # Only examine binaries reaching the given locations before redshift zero (other redz=inifinite)
    redz = data_harms['scafa']
    redz = cosmo.a_to_z(redz)
    valid = (redz > 0.0)

    # Broadcast harmonics numbers to correct shape
    harms = np.ones_like(redz, dtype=int) * harm_range[np.newaxis, :]
    # Select only the valid elements, also converts to 1D, i.e. (N, H) ==> (V,)
    harms = harms[valid]
    redz = redz[valid]

    # If there are eccentricities, calculate the freq-dist-function
    eccen = data_harms['eccen']
    if eccen is None:
        gne = 1
    else:
        gne = utils.gw_freq_dist_func(harms, ee=eccen[valid])
        # Select the elements corresponding to the n=2 (circular) harmonic, to use later
        sel_n2 = np.zeros_like(redz, dtype=bool)
        sel_n2[(harms == 2)] = 1
        sel_n2 = sel_n2[valid]

        # BUG: FIX: NOTE: this fails for zero eccentricities (at times?)
        # This is a reasonable, perhaps temporary, fix: when eccentricity is very low, set all
        # harmonics to zero except for n=2
        sel_e0 = (eccen[valid] < 1e-12)
        gne[sel_e0] = 0.0
        gne[sel_n2 & sel_e0] = 1.0

    # Calculate required parameters for valid binaries (V,)
    dcom = cosmo.z_to_dcom(redz)
    frst_orb = utils.frst_from_fobs(fobs_gw, redz) / harms
    mchirp = data_harms['mass'][valid]
    mchirp = utils.chirp_mass(*mchirp.T)
    # Calculate strains from each source
    hs2 = utils.gw_strain_source(mchirp, dcom, frst_orb)**2

    dfdt, _ = utils.dfdt_from_dadt(data_harms['dadt'][valid], data_harms['sepa'][valid], frst_orb=frst_orb)
    lambda_fact = utils.lambda_factor_freq(frst_orb, dfdt, redz, dcom=None) / box_vol

    shape = (lambda_fact.size, nreals)
    num_pois = np.random.poisson(lambda_fact[:, np.newaxis], shape)

    # --- Calculate GW Signals
    temp = hs2 * gne * (2.0 / harms)**2
    mc_ecc_both = np.sum(temp[:, np.newaxis] * num_pois, axis=0)

    if np.count_nonzero(num_pois) > 0:
        # Find the L loudest binaries in each realizations
        loud = np.sort(temp[:, np.newaxis] * (num_pois > 0), axis=0)[::-1, :]
        mc_ecc_fore = loud[0, :]
        loud = loud[:loudest, :]
    else:
        mc_ecc_fore = np.zeros_like(mc_ecc_both)
        loud = np.zeros((loudest, nreals))

    mc_ecc_back = mc_ecc_both - mc_ecc_fore
    return mc_ecc_both, mc_ecc_fore, mc_ecc_back, loud
