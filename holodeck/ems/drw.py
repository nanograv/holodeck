"""Damped Random Walks
"""

import numpy as np

import holodeck as holo
from holodeck.constants import MSOL
# from holodeck.ems import runnoe2012, bands_sdss


class MacLeod2010:
    """Damped Random Walk models from MacLeod+2010.

    MacLeod et al. 2010 - Modeling the Time Variability of SDSS Stripe 82 Quasars as a Damped Random Walk
    https://arxiv.org/abs/1004.0276
    https://ui.adsabs.harvard.edu/abs/2010ApJ...721.1014M/abstract
    """

    @classmethod
    def _fit_func(cls, pars, errs, lambda_rf, imag, mbh, randomize=False):
        if (randomize is not None) and (randomize is not False):
            shape = np.shape(mbh)
            if int(randomize) > 1:
                shape = shape + (int(randomize),)
                if not np.isscalar(imag):
                    imag = imag[..., np.newaxis]
                if not np.isscalar(mbh):
                    mbh = mbh[..., np.newaxis]

            shape = shape + (len(pars),)
            pars = np.random.normal(pars, errs, size=shape)
            pars = np.moveaxis(pars, -1, 0)

        aa, bb, cc, dd = pars

        # lf = aa + bb*np.log(lambda_rf/4000e-8) + cc*(imag + 23) + dd*np.log(mbh/(1e9*MSOL))
        # ff = np.exp(lf)
        rv = aa + bb*np.log10(lambda_rf/4000e-8) + cc*(imag + 23) + dd*np.log10(mbh/(1e9*MSOL))
        rv = 10**rv
        return rv

    @classmethod
    def sfinf(cls, imag, mbh, randomize=False):
        """`mbh` should be in grams (NOTE: I THINK!)
        """
        lambda_iband = 7690e-8
        pars = [-0.51, -0.479, 0.131, 0.18]
        errs = [0.02, 0.005, 0.008, 0.03]
        return cls._fit_func(pars, errs, lambda_iband, imag, mbh, randomize=randomize)

    @classmethod
    def tau(cls, imag, mbh, randomize=False):
        """`mbh` should be in grams (NOTE: I THINK!)
        `tau` is returned in units of days (NOTE: I THINK!)
        """
        lambda_iband = 7690e-8
        pars = [2.4, 0.17, 0.03, 0.21]
        errs = [0.2, 0.02, 0.04, 0.07]
        return cls._fit_func(pars, errs, lambda_iband, imag, mbh, randomize=randomize)


def drw_lightcurve(times, tau, mean_mag, sfinf, size=None):
    """Construct an AGN DRW lightcurve based on MacLeod+2010 model.

    Arguments
    ---------
    times : (N,) array_like of scalar
        Times at which to sample lightcurve
    tau : scalar,
        correlation timescale
    mean_mag : scalar,
        Mean absolute magnitude of AGN.
    sfinf : scalar,
        Structure-Function at Infinity
        (i.e. measure of variance of DRW, `sigma = sfinf / sqrt(2)`).

    Returns
    -------
    mags : (N,) ndarray of scalar
        Flux of continuum source in magnitudes.
    lums : (N,) ndarray of scalar
        Flux of continuum source in luminosity.

    """
    # sfinf = np.sqrt(2) * sigma_drw   # [MacLeod+2010] Eq. 4
    num = 1 if (size is None) else size

    mags = mean_mag * np.ones((num, times.size))
    dt = np.diff(times)

    if np.isscalar(tau):
        tau = np.ones(num) * tau
    if np.isscalar(mean_mag):
        mean_mag = np.ones(num) * mean_mag
    if np.isscalar(sfinf):
        sfinf = np.ones(num) * sfinf

    # [MacLeod+2010] Eq. 5
    exp = np.exp(-dt[np.newaxis, :] / tau[:, np.newaxis])
    rand = np.random.normal(size=exp.shape)
    var = 0.5 * np.square(sfinf[:, np.newaxis]) * (1 - exp**2)

    for i0, (ee, vv, rr) in enumerate(zip(exp.T, var.T, rand.T)):
        i1 = i0 + 1
        l0 = mags[:, i0]
        mean = ee * l0 + mean_mag * (1 - ee)

        # Transform the normal random variable to mean `mean` and variance `vv`
        temp = rr * np.sqrt(vv) + mean
        mags[:, i1] = temp

    lums = np.power(10.0, -0.4*mags)
    if size is None:
        mags = np.squeeze(mags)
        lums = np.squeeze(lums)
    else:
        mags = mags.T
        lums = lums.T

    return mags, lums


def drw_params(mass, fedd, eps=0.1, scatter=False):
    """DRW Parameters

    Returns
    -------
    imag
        Mean i-band absolute magnitude
    tau
        Correlation time of DRW
    sfi
        Structure-Function at Infinity

    """
    imag = holo.ems.runnoe2012.iband_from_mass_fedd(mass, fedd, eps=eps, magnitude=True).value
    taus = MacLeod2010.tau(imag, mass, randomize=scatter)
    sfis = MacLeod2010.sfinf(imag, mass, randomize=scatter)
    return imag, taus, sfis

