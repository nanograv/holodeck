"""
"""

import numpy as np
import pytest

# import holodeck as holo
from holodeck.ems import basics


def test_band_init_wlen():
    wlen = 123.0     # Angstrom
    freq = 2.437337e16  # Hz
    flux_wlen = "2.0 erg/(s cm^2 angstrom)"

    b1a = basics.Band('z', f"{wlen} angstrom", None, flux_wlen, None, "21.3 angstrom")
    b1b = basics.Band('z', f"{wlen} angstrom", None, flux_wlen, None, None)
    b2a = basics.Band('z', wlen, None, flux_wlen, None, 21.3)
    b2b = basics.Band('z', wlen, None, flux_wlen, None, None)

    for band in [b1a, b1b, b2a, b2b]:
        assert np.isclose(b1a.freq.to('Hz').value, freq, atol=0.0, rtol=1e-2)

    return


def test_band_init_freq():
    wlen = 123.0     # Angstrom
    freq = 2.437337e16  # Hz
    flux_freq = "2.0 erg/(s cm^2 Hz)"
    b1a = basics.Band('z', None, f"{freq} Hz", None, flux_freq, None, "21.3e15 Hz")
    b1b = basics.Band('z', None, f"{freq} Hz", None, flux_freq, None, None)
    b2a = basics.Band('z', None, freq, None, flux_freq, 21.3e15)
    b2b = basics.Band('z', None, freq, None, flux_freq, None)

    for band in [b1a, b1b, b2a, b2b]:
        assert np.isclose(b1a.wlen.to('Angstrom').value, wlen, atol=0.0, rtol=1e-3)

    return


def test_band_init_fail():
    wlen = 123.0     # Angstrom
    freq = 2.437337e16  # Hz
    flux_freq = "2.0 erg/(s cm^2 Hz)"
    bw_freq = "21.3e15 Hz"

    basics.Band('z', None, wlen, None, flux_freq, None, None)
    basics.Band('z', None, wlen, None, flux_freq, None, bw_freq)
    with pytest.raises(ValueError):
        basics.Band('z', None, None, None, flux_freq, None, bw_freq)

    basics.Band('z', freq, None, None, flux_freq, None, None)
    basics.Band('z', freq, None, None, flux_freq, None, bw_freq)
    with pytest.raises(ValueError):
        basics.Band('z', None, None, None, flux_freq, None, bw_freq)

    basics.Band('z', freq, None, None, flux_freq, None, bw_freq)
    with pytest.raises(ValueError):
        basics.Band('z', freq, None, None, None, None, bw_freq)

    return


def test_bands_sdss():
    bands = basics.Bands_SDSS()

    names = "ugriz"
    for name in names:
        assert name in bands.names
        b1 = bands(name)
        b2 = bands[name]
        assert b1 == b2

        # this is the reference flux for all bands, so by definition magnitude should be zero
        flux = "3631.0 Jansky"
        rmag = b1.flux_to_mag(flux, 'f')
        assert np.isclose(rmag, 0.0)
        check = b1.mag_to_flux(rmag, 'f')
        assert np.isclose(flux, check)

        rmag = b1.flux_to_mag("3631.0", 'f', units='jansky')
        assert np.isclose(rmag, 0.0)
        check = b1.mag_to_flux(rmag, 'f')
        assert np.isclose(flux, check)

        rmag = b1.flux_to_mag(3631.0, 'f', units='jansky')
        assert np.isclose(rmag, 0.0)
        check = b1.mag_to_flux(rmag, 'f')
        assert np.isclose(flux, check)

        # ---- Absolute Magnitude and Luminosity

        lum = b1.abs_mag_to_lum(0, 'f').to('erg/s/Hz')

        rmag = b1.lum_to_abs_mag(lum, 'f', units='erg/(s Hz)')
        assert np.isclose(rmag, 0.0, atol=1e-5)

        rmag = b1.lum_to_abs_mag(lum, 'f')
        assert np.isclose(rmag, 0.0, atol=1e-5)

        rmag = b1.lum_to_abs_mag(4.34447e20, 'f', units='erg/(s Hz)')
        assert np.isclose(rmag, 0.0, atol=1e-5)

        rmag = b1.lum_to_abs_mag("4.34447e20 erg/(s Hz)", 'f')
        assert np.isclose(rmag, 0.0, atol=1e-5)

    return



def test_bands_sdss__regression_2023_09_05():
    """Compare `flux_to_mag` calculation using current code, and previously computed data to check for regression.
    """

    bands = basics.Bands_SDSS()

    import json
    from pathlib import Path
    path = Path(__file__).resolve().parent
    FNAME = "./basics_sdss_bands_2023-09-05.json"
    data = json.load(open(path.joinpath(FNAME), 'r'))

    names = data.keys()
    for band in names:
        flux = data[band]['flux']
        mag_hz = data[band]['mag_hz']
        mag_angstrom = data[band]['mag_angstrom']
        units_hz = data[band]['units_hz']
        units_angstrom = data[band]['units_angstrom']

        b1 = bands[band].flux_to_mag(flux, 'f', units=units_hz).value
        b2 = bands[band].flux_to_mag(flux, 'w', units=units_angstrom).value

        assert np.allclose(b1, mag_hz, atol=0.0, rtol=1e-2)
        assert np.allclose(b2, mag_angstrom, atol=0.0, rtol=1e-2)

    return

