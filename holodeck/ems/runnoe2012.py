"""Functions and relations from Runnoe et al. 2012.

Runnoe+2012 [1201.5155] - Updating quasar bolometric luminosity corrections
https://ui.adsabs.harvard.edu/abs/2012MNRAS.422..478R/abstract

"""

import numpy as np
import astropy as ap
import astropy.units

import holodeck as holo
from holodeck import utils


__all__ = ["Runnoe2012"]

FRAC_ISO = 0.75


class Runnoe2012:

    _FITS = {
        '5100': {
            'alpha': (4.89, 1.66),
            'beta':  (0.91, 0.04),
            'wlen': 5100.0 * ap.units.angstrom,
        },
        '3000': {
            'alpha': (1.85, 1.27),
            'beta':  (0.98, 0.03),
            'wlen': 3000.0 * ap.units.angstrom,
        },
        '1450': {
            'alpha': (4.74, 1.00),
            'beta':  (0.91, 0.02),
            'wlen': 1450.0 * ap.units.angstrom,
        },
        '2to10': {
            'alpha': (25.14, 1.93),
            'beta':  (0.47, 0.043),
            'wlen': None,
        },
        '2to10rl': {
            'alpha': (23.04, 3.60),
            'beta':  (0.52, 0.080),
            'wlen': None,
        },
        '2to10rq': {
            'alpha': (33.06, 3.17),
            'beta':  (0.29, 0.072),
            'wlen': None,
        },
    }

    def __init__(self):
        self._names = self._FITS.keys()
        return

    @property
    def names(self):
        return self._names

    def _fit_params_for_band(self, band):
        if band not in self.names:
            raise KeyError(f"`band` ({band}) must be one of {self._options}!")

        vals = self._FITS[band]
        alpha = vals['alpha']
        beta = vals['beta']
        wlen = ap.units.Quantity(vals['wlen'], 'angstrom')
        return alpha, beta, wlen

    def lband_from_lbol(self, band, lbol, scatter=False, fiso=FRAC_ISO):
        """Convert from bolometric luminosity to luminosity in photometric band.

        Arguments
        ---------
        band : str
            Specification of which photometric band.  One of ``Runnoe2012.names``,
            {'5100', '3000', '1450', '2to10', '2to10rl', '2to10rq'}
        lbol : array_like, units of [erg/s]
            Bolometric luminosity.

        Returns
        -------
        lband : array_like,
            Luminosity in photometric band.

            - If `band` is one of the optical bands, return spectral luminosity,
              with units of [erg/s/Angstrom].
            - If `band` is one of the x-ray bands, return luminosity across the band,
              with units of [erg/s].

        """
        alpha, beta, wlen = self._fit_params_for_band(band)
        if not scatter:
            alpha = alpha[0]
            beta = beta[0]

        lbol = ap.units.Quantity(lbol, 'erg/s')
        lband = _lbol_to_lband__pow_law(lbol, alpha, beta, fiso=fiso)
        # if this is one of the x-ray bands, then wlen is None, and `lband` is the luminosity across
        # the band, in units of erg/s
        if wlen is None:
            # units = 'erg / s'
            pass
        # if this is one of the (near-)optical bands, then `wlen` should be wavelength in Angstroms
        # and `lband` starts out as lambda * F_lambda in units of [erg/s]
        # convert that to just F_lambda in units of [erg/s/Angstrom]
        else:
            lband = lband / wlen
            # units = 'erg / (s angstrom)'

        return lband

    def lbol_from_lband(self, band_name, lband, scatter=False, fiso=FRAC_ISO):
        """Convert from luminosity in photometric band to bolometric luminosity.

        Arguments
        ---------
        band_name : str
            Specification of which photometric band.  One of `Runnoe2012.names`:
            {'5100', '3000', '1450', '2to10', '2to10rl', '2to10rq'}
        lband : array_like,
            Luminosity in photometric band.

            - If `band_name` is one of the optical bands, `lband` must be spectral luminosity,
              with units of [erg/s/Angstrom].
            - If `band_name` is one of the x-ray bands, `lband` must be luminosity across the band,
              with units of [erg/s].

        Returns
        -------
        lbol : array_like, units of [erg/s]
            Bolometric luminosity.

        """

        alpha, beta, wlen = self._fit_params_for_band(band_name)
        if not scatter:
            alpha = alpha[0]
            beta = beta[0]

        # For the x-ray bands, `wlen` is `None`, and `lband` must have units of erg/s
        if wlen is None:
            pass
        # For the (near-)optical bands, `wlen` is the wavelength in Angstroms, and the relation
        # requires lambda * L_lambda, with units of erg/s.
        # Convert from `L_lambda` (units of [erg/s/Angstrom]) to lambda * L_lambda
        else:
            lband = lband * wlen

        units = 'erg / s'
        lband = ap.units.Quantity(lband, units)

        lbol = _lband_to_lbol__pow_law(lband, alpha, beta, fiso=0.75)
        lbol = ap.units.Quantity(lbol, units)
        return lbol

    def iband_from_mass_fedd(self, mass, fedd, eps=0.1, magnitude=True):
        lbol = utils.eddington_luminosity(mass, eps) * fedd
        lbol = ap.units.Quantity(lbol, 'erg/s')
        iband = self.lband_from_lbol('5100', lbol)
        if magnitude:
            iband = holo.ems.bands_sdss['i'].lum_to_abs_mag(iband, type='w')
        return iband


def _dist_pars(arg, num):
    """If `arg` is tuple (2,), draw `num` points from normal distribution with those parameters.
    """
    if isinstance(arg, tuple):
        if len(arg) != 2:
            raise ValueError("`arg` must be a tuple of (mean, std)!")
        arg = np.random.normal(*arg, size=num)

    return arg


def _lband_to_lbol__pow_law(lam_lum_lam, alpha, beta, fiso=1.0):
    """
    log(L_iso) = alpha + beta * log10(lambda * L_lambda)
    L_bol = fiso * L_iso
    """
    liso = alpha + beta*np.log10(lam_lum_lam.to('erg/s').value)
    lbol = fiso * (10**liso)
    lbol = ap.units.Quantity(lbol, 'erg/s')
    return lbol


def _lbol_to_lband__pow_law(lbol, alpha, beta, fiso=1.0):
    """Returns lambda*L_lambda

    log(L_iso) = alpha + beta * log10(lambda * L_lambda)
    L_iso = L_bol / fiso
    """
    liso_log = np.log10(lbol.to('erg/s').value/fiso)
    num = np.size(lbol)

    alpha = _dist_pars(alpha, num)
    beta = _dist_pars(beta, num)

    lband = np.power(10, (liso_log - alpha)/beta)
    lband = ap.units.Quantity(lband, 'erg/s')
    return lband

