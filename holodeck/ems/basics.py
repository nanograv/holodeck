"""Astronomical observations calculations.

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
        self.bandwidth_freq = bandwidth_wlen
        return

    def __str__(self):
        rv = (
            f"{self.name}:  w={self.wlen:.4e}, f={self.freq:.4e}  |  "
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

    def __init__(self, bands_dict):
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

            flux_ref_wlen = values.get('flux_ref_wlen', None)
            flux_ref_freq = values.get('flux_ref_freq', None)
            if (flux_ref_wlen is None) and (flux_ref_freq is None):
                raise ValueError(f"Band {name} has neither 'flux_ref_wlen' or 'flux_ref_freq' specification!")
            if (flux_ref_wlen is None):
                flux_ref_wlen = Band.spectral_wlen(flux_ref_freq, freq=freq)
            if (flux_ref_freq is None):
                flux_ref_freq = Band.spectral_freq(flux_ref_wlen, wlen=wlen)

            bandwidth_wlen = values.get('bandwidth_wlen', None)
            bandwidth_freq = values.get('bandwidth_freq', None)
            if (bandwidth_wlen is None) and (bandwidth_freq is not None):
                bandwidth_wlen = Band.freq_to_wlen(bandwidth_freq)
            if (bandwidth_freq is None) and (bandwidth_wlen is not None):
                bandwidth_freq = Band.wlen_to_freq(bandwidth_wlen)

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


class SDSS_Bands(BANDS):

    def __init__(self):
        super().__init__(BANDS_SDSS_AB_MAGS)
        return


units_erg_s_cm2_angstrom = ap.units.erg / ap.units.second / ap.units.cm**2 / ap.units.angstrom

# SDSS AB Magnitudes
BANDS_SDSS_AB_MAGS = {
    "u": {
        "wlen": 356 * ap.units.nm,
        "bandwidth_wlen": 46.3 * ap.units.nm,
        "flux_ref_freq": 3631 * ap.units.jansky,
        "flux_ref_wlen": 859.5e-11 * units_erg_s_cm2_angstrom,
    },
    "g": {
        "wlen": 483 * ap.units.nm,
        "bandwidth_wlen": 98.8 * ap.units.nm,
        "flux_ref_freq": 3631 * ap.units.jansky,
        "flux_ref_wlen": 466.9e-11 * units_erg_s_cm2_angstrom,
    },
    "r": {
        "wlen": 626 * ap.units.nm,
        "bandwidth_wlen": 95.5 * ap.units.nm,
        "flux_ref_freq": 3631 * ap.units.jansky,
        "flux_ref_wlen": 278.0e-11 * units_erg_s_cm2_angstrom,
    },
    "i": {
        "wlen": 767 * ap.units.nm,
        "bandwidth_wlen": 106.4 * ap.units.nm,
        "flux_ref_freq": 3631 * ap.units.jansky,
        "flux_ref_wlen": 185.2e-11 * units_erg_s_cm2_angstrom,
    },
    "z": {
        "wlen": 910 * ap.units.nm,
        "bandwidth_wlen": 124.8 * ap.units.nm,
        "flux_ref_freq": 3631 * ap.units.jansky,
        "flux_ref_wlen": 131.5e-11 * units_erg_s_cm2_angstrom,
    },
}


'''
# These wavelengths are in [cm]
BAND_EFF_LOC = {
    # Vega/Johnson/Bessell
    "U": {"l": 366e-7},
    "B": {"l": 438e-7},
    "V": {"l": 545e-7},
    "R": {"l": 641e-7},
    "I": {"l": 798e-7},
    # SDSS AB Magnitudes
    "u": {"l": 356e-7},
    "g": {"l": 483e-7},
    "r": {"l": 626e-7},
    "i": {"l": 767e-7},
    "z": {"l": 910e-7}
}
BAND_REF_FLUX = {
    # Vega/Johnson/Bessell
    "U": {"f": 1.790, "l": 417.5},
    "B": {"f": 4.063, "l": 632.0},
    "V": {"f": 2.636, "l": 363.1},
    "R": {"f": 3.064, "l": 217.7},
    "I": {"f": 2.416, "l": 112.6},
    # SDSS AB Magnitudes
    "u": {"f": 3.631, "l": 859.5},
    "g": {"f": 3.631, "l": 466.9},
    "r": {"f": 3.631, "l": 278.0},
    "i": {"f": 3.631, "l": 185.2},
    "z": {"f": 3.631, "l": 131.5}
}
BAND_ZERO_POINT = {
    # Vega/Johnson/Bessell
    "U": {"f": +0.770, "l": -0.152},
    "B": {"f": -0.120, "l": -0.602},
    "V": {"f": +0.000, "l": +0.000},
    "R": {"f": +0.186, "l": +0.555},
    "I": {"f": +0.444, "l": +1.271},
    # SDSS AB Magnitudes
    "u": {"f": 0.0, "l": 0.0},
    "g": {"f": 0.0, "l": 0.0},
    "r": {"f": 0.0, "l": 0.0},
    "i": {"f": 0.0, "l": 0.0},
    "z": {"f": 0.0, "l": 0.0}
}
UNITS = {
    "f": 1.0e-20,  # erg/s/Hz/cm^2
    "l": 1.0e-11   # erg/s/Angstrom/cm^2
}

# _band_name = ['u', 'b', 'v', 'r', 'i']
# _band_wlen = [365, 445, 551, 658, 806]   # nm
# _band_color = ['violet', 'blue', 'green', 'red', 'darkred']
# Band = namedtuple('band', ['name', 'freq', 'wlen', 'color'])
#
# BANDS = {nn: Band(nn, SPLC/(ll*1e-7), ll*1e-7, cc)
#          for nn, ll, cc in zip(_band_name, _band_wlen, _band_color)}


def _get_units_type(type):
    try:
        units = UNITS[type]
    except Exception:
        raise ValueError("Unrecognized `type` = '{}'".format(type))

    return units, type


def ABmag_to_flux(mag):
    """Convert from AB Magnitude to spectral-flux density.

    See: http://web.ipac.caltech.edu/staff/fmasci/home/astro_refs/magsystems.pdf

    Returns
    -------
    fnu : () scalar
        Spectral-flux density in units of [erg/s/cm^2/Hz]

    """
    fnu = np.power(10.0, (mag + 48.6)/-2.5)
    return fnu


def mag_to_flux_zero(mag, zero_jansky=None):
    if zero_jansky is None:
        raise

    zero_point = zero_jansky * JY
    flux = np.power(10.0, mag / -2.5) * zero_point
    return flux

'''