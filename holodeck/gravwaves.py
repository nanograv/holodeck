"""
"""

import numpy as np

from holodeck import utils

_CALC_MC_PARS = ['mass', 'sepa', 'dadt', 'time', 'eccs']


class GWB:

    def __init__(self, bin_evo, freqs, box_vol_cgs, nharms=30, nreals=100, calculate=True):
        self.freqs = freqs
        self.nharms = nharms
        self.nreals = nreals
        self._box_vol_cgs = box_vol_cgs
        self._bin_evo = bin_evo

        if calculate:
            self.calculate(bin_evo)

        return

    def calculate(self, bin_evo, eccen=None, stats=False, progress=True, nloudest=5):
        freqs = self.freqs
        nharms = self.nharms
        nreals = self.nreals
        bin_evo = self._bin_evo
        box_vol = self._box_vol_cgs

        if eccen is None:
            eccen = (bin_evo.eccs is not None)

        if eccen not in [True, False]:
            raise ValueError("`eccen` '{}' is invalid!".format(eccen))

        eccen_fore = np.zeros((freqs.size, nreals))
        eccen_back = np.zeros((freqs.size, nreals))
        eccen_both = np.zeros((freqs.size, nreals))
        circ_fore = np.zeros((freqs.size, nreals))
        circ_back = np.zeros((freqs.size, nreals))
        circ_both = np.zeros((freqs.size, nreals))
        loudest = np.zeros((freqs.size, nloudest, nreals))
        sa_eccen = np.zeros_like(freqs)
        sa_circ = np.zeros_like(freqs)

        if eccen:
            harm_range = range(1, nharms+1)
        else:
            harm_range = [2]

        for ii, fobs in tqdm.tqdm(enumerate(freqs), total=len(freqs)):
            rv = _calc_mc_at_fobs(
                fobs, harm_range, nreals, bin_evo, box_vol,
                loudest=nloudest
            )
            mc_ecc, mc_circ, ret_sa_ecc, ret_sa_circ, loud = rv
            eccen_fore[ii, :] = mc_ecc[0]
            eccen_back[ii, :] = mc_ecc[1]
            eccen_both[ii, :] = mc_ecc[2]
            circ_fore[ii, :] = mc_circ[0]
            circ_back[ii, :] = mc_circ[1]
            circ_both[ii, :] = mc_circ[2]
            sa_eccen[ii] = ret_sa_ecc
            sa_circ[ii] = ret_sa_circ
            loudest[ii, :] = loud

        self.eccen_fore = np.sqrt(eccen_fore)
        self.eccen_back = np.sqrt(eccen_back)
        self.eccen_both = np.sqrt(eccen_both)

        self.circ_fore = np.sqrt(circ_fore)
        self.circ_back = np.sqrt(circ_back)
        self.circ_both = np.sqrt(circ_both)

        self.sa_eccen = np.sqrt(sa_eccen)
        self.sa_circ = np.sqrt(sa_circ)
        self.loudest = np.sqrt(loudest)

        return


def _calc_mc_at_fobs(fobs, harm_range, nreals, bin_evo, box_vol, loudest=5):
    """
    """

    # ---- Interpolate data to all harmonics of this frequency
    harm_range = np.asarray(harm_range)
    # Each parameter will be (N, H) = (binaries, harmonics)
    data_harms = bin_evo.at('fobs', fobs / harm_range, pars=_CALC_MC_PARS)

    # Only examine binaries reaching the given locations before redshift zero (other redz=inifinite)
    redz = data_harms['time']
    redz = utils.a_to_z(redz)
    valid = np.isfinite(redz) & (redz > 0.0)

    # Broadcast harmonics numbers to correct shape
    harms = np.ones_like(redz, dtype=int) * harm_range[np.newaxis, :]
    # Select the elements corresponding to the n=2 (circular) harmonic, to use later
    sel_n2 = np.zeros_like(redz, dtype=int)
    sel_n2[(harms == 2)] = 1
    # Select only the valid elements, also converts to 1D, i.e. (N, H) ==> (V,)
    sel_n2 = sel_n2[valid]
    harms = harms[valid]
    redz = redz[valid]
    # If there are eccentricities, calculate the freq-dist-function
    eccs = data_harms['eccs']
    if eccs is None:
        gne = 1
    else:
        gne = utils.gw_freq_dist_func(harms, ee=eccs[valid])
        # BUG: FIX: NOTE: this fails for zero eccentricities (at times?) fix manually!
        sel_e0 = (eccs[valid] == 0.0)
        gne[sel_e0] = 0.0
        gne[sel_n2 & sel_e0] = 1.0

    # Calculate required parameters for valid binaries (V,)
    dlum = cosmo.z_to_dlum(redz)
    zp1 = redz + 1
    frst_orb = fobs * zp1 / harms
    mchirp = data_harms['mass'][valid]
    mchirp = utils.chirp_mass(*mchirp.T)
    # NOTE: `dadt` is stored as positive values
    dfdt = utils.dfdt_from_dadt(
        -data_harms['dadt'][valid], data_harms['sepa'][valid], freq_orb=frst_orb)
    _tres = frst_orb / dfdt

    # Calculate strains from each source
    hs2 = utils.gw_strain_source(mchirp, dlum, frst_orb)**2
    # Calculate resampling factors
    vfac = 4.0*np.pi*SPLC * dlum**2 / box_vol   # * thub
    tfac = _tres  # / thub

    # Calculate weightings
    #    Sesana+08, Eq.10
    num_frac = vfac * tfac * zp1
    try:
        num_pois = np.random.poisson(num_frac, (nreals, num_frac.size)).T
    except:
        print(f"{dlum=}")
        print(f"{redz=}")
        print(f"{vfac=}")
        print(f"{tfac=}")
        print(f"{zp1=}")
        print(f"{num_frac=}")
        raise

    # --- Calculate GW Signals
    temp = hs2 * gne * (2.0 / harms)**2
    mc_ecc_both = np.sum(temp[:, np.newaxis] * num_pois, axis=0)
    mc_circ_both = np.sum(temp[:, np.newaxis] * num_pois * sel_n2[:, np.newaxis], axis=0)

    sa_ecc = np.sum(temp * num_frac, axis=0)
    sa_circ = np.sum(temp * num_frac * sel_n2, axis=0)

    if np.count_nonzero(num_pois) > 0:
        # Find the L loudest binaries in each realizations
        loud = np.sort(temp[:, np.newaxis] * (num_pois > 0), axis=0)[::-1, :]
        mc_ecc_fore = loud[0, :]
        loud = loud[:loudest, :]

        mc_circ_fore = np.max(temp[:, np.newaxis] * (num_pois > 0) * sel_n2[:, np.newaxis], axis=0)
    else:
        mc_ecc_fore = np.zeros_like(mc_ecc_both)
        mc_circ_fore = np.zeros_like(mc_circ_both)
        loud = np.zeros((loudest, nreals))

    mc_ecc_back = mc_ecc_both - mc_ecc_fore
    mc_circ_back = mc_circ_both - mc_circ_fore

    # Package and return
    mc_ecc = [mc_ecc_fore, mc_ecc_back, mc_ecc_both]
    mc_circ = [mc_circ_fore, mc_circ_back, mc_circ_both]

    return mc_ecc, mc_circ, sa_ecc, sa_circ, loud
