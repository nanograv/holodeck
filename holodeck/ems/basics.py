"""Astronomical observations calculations.

http://svo2.cab.inta-csic.es/theory/fps/index.php?mode=browse&gname=SLOAN&asttype=
See also: https://ui.adsabs.harvard.edu/abs/2018ApJS..236...47W/abstract

- lots of reference fluxes: https://coolwiki.ipac.caltech.edu/index.php/Central_wavelengths_and_zero_points
- VEGA/Johnson/Bessell: http://web.ipac.caltech.edu/staff/fmasci/home/astro_refs/magsystems.pdf
- SDSS/AB/Fukugita: http://www.astronomy.ohio-state.edu/~martini/usefuldata.html

"""

import numpy as np
import astropy as ap

UNITS_WLEN = 'angstrom'
UNITS_FREQ = 'Hz'

UNITS_FLUX_WLEN = 'erg/(s cm2 angstrom)'
UNITS_FLUX_FREQ = 'erg/(s cm2 Hz)'

_units_erg_s_cm2_angstrom = ap.units.erg / ap.units.second / ap.units.cm**2 / ap.units.angstrom

# SDSS AB Magnitudes
# from http://svo2.cab.inta-csic.es/theory/fps/index.php?mode=browse&gname=SLOAN&asttype=
# Using "full transmission" filters, obtained on 2023-09-06
BANDS_SDSS = {
    "u": {
        "wlen": 3608.04 * ap.units.angstrom,
        "bandwidth_wlen": 540.97 * ap.units.angstrom,
        "vega": {
            "flux_ref_freq": 1582.54 * ap.units.jansky,
            "flux_ref_wlen": 3.75079e-9 * _units_erg_s_cm2_angstrom,
        },
        "AB": {
            "flux_ref_freq": 3631.00 * ap.units.jansky,
            "flux_ref_wlen": 8.60588e-9 * _units_erg_s_cm2_angstrom,
        },
    },
    "g": {
        "wlen": 4671.78 * ap.units.angstrom,
        "bandwidth_wlen": 1064.68 * ap.units.angstrom,
        "vega": {
            "flux_ref_freq": 4023.57 * ap.units.jansky,
            "flux_ref_wlen": 5.45476e-9 * _units_erg_s_cm2_angstrom,
        },
        "AB": {
            "flux_ref_freq": 3631.00 * ap.units.jansky,
            "flux_ref_wlen": 4.92255e-9 * _units_erg_s_cm2_angstrom,
        },
    },
    "r": {
        "wlen": 6141.12 * ap.units.angstrom,
        "bandwidth_wlen": 1055.51 * ap.units.angstrom,
        "vega": {
            "flux_ref_freq": 3177.38 * ap.units.jansky,
            "flux_ref_wlen": 2.49767e-9 * _units_erg_s_cm2_angstrom,
        },
        "AB": {
            "flux_ref_freq": 3631.00 * ap.units.jansky,
            "flux_ref_wlen": 2.85425e-9 * _units_erg_s_cm2_angstrom,
        },
    },
    "i": {
        "wlen": 7457.89 * ap.units.angstrom,
        "bandwidth_wlen": 1102.57 * ap.units.angstrom,
        "vega": {
            "flux_ref_freq": 2593.40 * ap.units.jansky,
            "flux_ref_wlen": 1.38589e-9 * _units_erg_s_cm2_angstrom,
        },
        "AB": {
            "flux_ref_freq": 3631.00 * ap.units.jansky,
            "flux_ref_wlen": 1.94038e-9 * _units_erg_s_cm2_angstrom,
        },
    },
    "z": {
        "wlen": 8922.78 * ap.units.angstrom,
        "bandwidth_wlen": 1164.01 * ap.units.angstrom,
        "vega": {
            "flux_ref_freq": 2238.99 * ap.units.jansky,
            "flux_ref_wlen": 8.38585e-10 * _units_erg_s_cm2_angstrom,
        },
        "AB": {
            "flux_ref_freq": 3631.00 * ap.units.jansky,
            "flux_ref_wlen": 1.35994e-9	 * _units_erg_s_cm2_angstrom,
        },
    },
}

BANDS_LSST = {
    "u": {
        "wlen": 3751.20 * ap.units.angstrom,
        "bandwidth_wlen": 473.19 * ap.units.angstrom,
        "AB": {
            "flux_ref_wlen": 8.03787e-9 * _units_erg_s_cm2_angstrom,
            "flux_ref_freq": 3631.00 * ap.units.jansky,
        },
    },
    "g": {
        "wlen": 4740.66 * ap.units.angstrom,
        "bandwidth_wlen": 1253.26 * ap.units.angstrom,
        "AB": {
            "flux_ref_wlen": 4.7597e-9 * _units_erg_s_cm2_angstrom,
            "flux_ref_freq": 3631.00 * ap.units.jansky,
        },
    },
    "r": {
        "wlen": 6172.34 * ap.units.angstrom,
        "bandwidth_wlen": 1206.92 * ap.units.angstrom,
        "AB": {
            "flux_ref_wlen": 2.8156e-9 * _units_erg_s_cm2_angstrom,
            "flux_ref_freq": 3631.00 * ap.units.jansky,
        },
    },
    "i": {
        "wlen": 7500.97 * ap.units.angstrom,
        "bandwidth_wlen": 1174.77 * ap.units.angstrom,
        "AB": {
            "flux_ref_wlen": 1.91864e-9 * _units_erg_s_cm2_angstrom,
            "flux_ref_freq": 3631.00 * ap.units.jansky,
        },
    },
    "z": {
        "wlen": 8678.90 * ap.units.angstrom,
        "bandwidth_wlen": 997.51 * ap.units.angstrom,
        "AB": {
            "flux_ref_wlen": 1.44312e-9 * _units_erg_s_cm2_angstrom,
            "flux_ref_freq": 3631.00 * ap.units.jansky,
        },
    },
    "y": {
        "wlen": 9711.82 	 * ap.units.angstrom,
        "bandwidth_wlen": 871.83 * ap.units.angstrom,
        "AB": {
            "flux_ref_wlen": 1.14978e-9 * _units_erg_s_cm2_angstrom,
            "flux_ref_freq": 3631.00 * ap.units.jansky,
        },
    },
}


def _get_wlen_freq(wlen, freq, error_if_neither):
    if error_if_neither and (wlen is None) and (freq is None):
        raise ValueError("neither 'wlen' or 'freq' is given!")

    if (wlen is None) and (freq is not None):
        freq = ap.units.Quantity(freq, UNITS_FREQ)
        wlen = Band.freq_to_wlen(freq)
    if (freq is None) and (wlen is not None):
        wlen = ap.units.Quantity(wlen, UNITS_WLEN)
        freq = Band.wlen_to_freq(wlen)

    return wlen, freq


def _get_flux_wlen_flux_freq(flux_wlen, flux_freq, freq, wlen):
    if (flux_wlen is None) and (flux_freq is None):
        raise ValueError("neither 'flux_wlen' or 'flux_freq' has been specified!")

    if (flux_wlen is None):
        flux_freq = ap.units.Quantity(flux_freq, UNITS_FLUX_FREQ)
        flux_wlen = Band.spectral_wlen(flux_freq, freq=freq)
    if (flux_freq is None):
        flux_wlen = ap.units.Quantity(flux_wlen, UNITS_FLUX_WLEN)
        flux_freq = Band.spectral_freq(flux_wlen, wlen=wlen)

    return flux_wlen, flux_freq


class Band:

    def __init__(self, name, wlen, freq, ref_flux_wlen, flux_ref_freq,
                 bandwidth_wlen=None, bandwidth_freq=None):
        wlen, freq = _get_wlen_freq(wlen, freq, error_if_neither=True)
        ref_flux_wlen, flux_ref_freq = _get_flux_wlen_flux_freq(ref_flux_wlen, flux_ref_freq, freq, wlen)
        bandwidth_wlen, bandwidth_freq = _get_wlen_freq(bandwidth_wlen, bandwidth_freq, error_if_neither=False)

        self.name = name
        self.wlen = wlen
        self.freq = freq
        self.flux_ref_wlen = ap.units.Quantity(ref_flux_wlen, UNITS_FLUX_WLEN)
        self.flux_ref_freq = ap.units.Quantity(flux_ref_freq, UNITS_FLUX_FREQ)
        self.bandwidth_freq = bandwidth_freq
        self.bandwidth_wlen = bandwidth_wlen
        return

    def __str__(self):
        rv = (
            f"{self.name} band:  wlen={self.wlen:.4e}, freq={self.freq:.4e}  |  "
            f"F_w={self.flux_ref_wlen:.4e}, F_f={self.flux_ref_freq:.4e}"
        )
        return rv

    @classmethod
    def wlen_to_freq(cls, wlen):
        return ap.units.Quantity(ap.constants.c / wlen, UNITS_FREQ)

    @classmethod
    def freq_to_wlen(cls, freq):
        return ap.units.Quantity(ap.constants.c / freq, UNITS_WLEN)

    @classmethod
    def spectral_freq(cls, spec_wlen, wlen):
        spec_freq = ap.units.Quantity(spec_wlen * wlen**2 / ap.constants.c, UNITS_FLUX_FREQ)
        return spec_freq

    @classmethod
    def spectral_wlen(cls, spec_freq, freq):
        spec_wlen = ap.units.Quantity(spec_freq * freq**2 / ap.constants.c, UNITS_FLUX_WLEN)
        return spec_wlen

    def mag_to_flux(self, mag, type):
        """Convert from broad-band filter magnitude to spectral flux.

        Returns
        -------
        flux : () scalar
            Flux in either [erg/s/cm^2/Hz] or [erg/s/cm^2/Angstrom] depending on `type`.

        """
        ref_flux = self._ref_flux_for_type(type)
        flux = ref_flux * np.power(10.0, mag/-2.5)
        return flux

    def flux_to_mag(self, flux, type, units=None):
        """Convert from spectral flux to broad-band filter magnitude.

        Arguments
        ---------
        flux : () scalar
            Flux in either [erg/s/cm^2/Hz] or [erg/s/cm^2/Angstrom] depending on `type`.
        type

        Returns
        -------
        mag

        """
        if units is not None:
            flux = ap.units.Quantity(flux, units)

        ref_flux = self._ref_flux_for_type(type)
        mag = flux / ref_flux

        try:
            mag = mag.to('')
        except ap.units.UnitConversionError as err:
            msg = (
                "Could not convert 'flux' to a spectral flux.  "
                f"Try using the `units` argument, e.g. ``units='erg/s/cm2/Hz'`` or ``units='erg/s/cm2/Angstrom'``.  "
                f"({err})"
            )
            raise ValueError(msg)

        mag = -2.5 * np.log10(mag)
        return mag

    def abs_mag_to_lum(self, abs_mag, type):
        """Convert from broad-band filter absolute-magnitude to spectral luminosity.
        """
        ref_flux = self._ref_flux_for_type(type)
        lum = 4.0 * np.pi * ref_flux * (10.0 * ap.units.parsec)**2 * np.power(10.0, -abs_mag/2.5)
        return lum

    def lum_to_abs_mag(self, lum, type, units=None):
        if units is not None:
            lum = ap.units.Quantity(lum, units)

        ref_flux = self._ref_flux_for_type(type)
        mag = lum / (ref_flux * 4.0 * np.pi * (10.0 * ap.units.parsec)**2)

        try:
            mag = mag.to('')
        except ap.units.UnitConversionError as err:
            msg = (
                "Could not convert 'lum' to a spectral luminosity.  "
                f"Try using the `units` argument, e.g. ``units='erg/s/Hz'`` or ``units='erg/s/Angstrom'``.  "
                f"({err})"
            )
            raise ValueError(msg)

        mag = -2.5 * np.log10(mag)
        return mag

    def _ref_flux_for_type(self, type):
        """Get the appropriate reference flux for the given 'type'.

        If `type` is '[f]requency'  : return `flux_ref_freq`
        If `type` is '[w]avelength' : return `flux_ref_wlen`

        Arguments
        ---------
        type : string,
            Specification of wavelength or frequency.

        Returns
        -------
        ref_flux : astropy.units.Quantity,
            Reference flux for this band, either F_nu or F_lambda based on the `type` argument.
            If `type` == '[f]requency',  then F_nu     is returned (e.g. in units of erg/s/cm^2/Hz)
            If `type` == '[w]avelength', then F_lambda is returned (e.g. in units of erg/s/cm^2/Angstrom)

        """
        if type.startswith('f'):
            ref_flux = self.flux_ref_freq
        elif type.startswith('w'):
            ref_flux = self.flux_ref_wlen
        else:
            raise ValueError(f"`type` ({type}) should be '[f]requency' or '[w]avelength'!")
        return ref_flux


class BANDS:

    def __init__(self, bands_dict, mag_type):
        bands = {}
        for name, values in bands_dict.items():

            wlen = values.get('wlen', None)
            freq = values.get('freq', None)
            if (wlen is None) and (freq is None):
                raise ValueError(f"Band {name} has neither 'wlen' or 'freq' specification!")
            if wlen is None:
                wlen = Band.freq_to_wlen(freq)
            if freq is None:
                freq = Band.wlen_to_freq(wlen)

            bandwidth_wlen = values.get('bandwidth_wlen', None)
            bandwidth_freq = values.get('bandwidth_freq', None)
            if (bandwidth_wlen is None) and (bandwidth_freq is not None):
                bandwidth_wlen = Band.freq_to_wlen(bandwidth_freq)
            if (bandwidth_freq is None) and (bandwidth_wlen is not None):
                bandwidth_freq = Band.wlen_to_freq(bandwidth_wlen)

            zero_points = values.get(mag_type, None)
            if zero_points is None:
                raise ValueError(f"Band '{name}' does not have specification for mag_type '{mag_type}'!")
            flux_ref_wlen = zero_points.get('flux_ref_wlen', None)
            flux_ref_freq = zero_points.get('flux_ref_freq', None)
            if (flux_ref_wlen is None) and (flux_ref_freq is None):
                raise ValueError(f"Band '{name}' '{mag_type}' has neither 'flux_ref_wlen' nor 'flux_ref_freq'!")
            if (flux_ref_wlen is None):
                flux_ref_wlen = Band.spectral_wlen(flux_ref_freq, freq=freq)
            if (flux_ref_freq is None):
                flux_ref_freq = Band.spectral_freq(flux_ref_wlen, wlen=wlen)

            band = Band(
                name, wlen, freq, flux_ref_wlen, flux_ref_freq,
                bandwidth_wlen=bandwidth_wlen, bandwidth_freq=bandwidth_freq
            )
            bands[name] = band

        if len(bands) == 0:
            raise ValueError("No bands provided!")

        self._bands = bands
        return

    def __getitem__(self, name):
        return self._bands[name]

    def __call__(self, name):
        return self._bands[name]

    @property
    def names(self):
        return self._bands.keys()


class Bands_SDSS(BANDS):
    """SDSS Generally uses AB magnitudes.
    """

    def __init__(self):
        super().__init__(BANDS_SDSS, "AB")
        return


class Bands_LSST(BANDS):
    """LSST Generally uses AB magnitudes.
    """

    def __init__(self):
        super().__init__(BANDS_LSST, "AB")
        return