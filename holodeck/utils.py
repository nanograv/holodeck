"""

References:
- Peters-1964 : [Peters 1964](https://ui.adsabs.harvard.edu/abs/1964PhRv..136.1224P/abstract)
- EN07 : [Enoki & Nagashima 2007](https://ui.adsabs.harvard.edu/abs/2007PThPh.117..241E/abstract)
- Enoki+2004 : [Enoki et al. 2004](https://ui.adsabs.harvard.edu/abs/2004ApJ...615...19E/abstract)
- Sesana+2004 : [Sesana+2004](http://adsabs.harvard.edu/abs/2004ApJ...611..623S)

"""

import abc
import copy
import numbers
from typing import Optional, Tuple  # , Sequence,

import numpy as np
import numpy.typing as npt
import scipy as sp
import h5py

from holodeck import log
from holodeck.constants import NWTG, SCHW, SPLC, YR

# e.g. Sesana+2004 Eq.36
_GW_SRC_CONST = 8 * np.power(NWTG, 5/3) * np.power(np.pi, 2/3) / np.sqrt(10) / np.power(SPLC, 4)
_GW_DADT_SEP_CONST = - 64 * np.power(NWTG, 3) / 5 / np.power(SPLC, 5)
_GW_DEDT_ECC_CONST = - 304 * np.power(NWTG, 3) / 15 / np.power(SPLC, 5)
# EN07, Eq.2.2
_GW_LUM_CONST = (32.0 / 5.0) * np.power(NWTG, 7.0/3.0) * np.power(SPLC, -5.0)


class _Modifier(abc.ABC):

    def __call__(self, base):
        self.modify(base)
        return

    @abc.abstractmethod
    def modify(self, base):
        pass


# =================================================================================================
# ====    General Logistical    ====
# =================================================================================================


def error(msg, etype=ValueError):
    log.exception(msg, exc_info=True)
    raise etype(msg)


def load_hdf5(fname, keys=None):
    squeeze = False
    if (keys is not None) and np.isscalar(keys):
        keys = [keys]
        squeeze = True

    header = dict()
    data = dict()
    with h5py.File(fname, 'r') as h5:
        head_keys = h5.attrs.keys()
        for kk in head_keys:
            header[kk] = copy.copy(h5.attrs[kk])

        if keys is None:
            keys = h5.keys()

        for kk in keys:
            data[kk] = h5[kk][:]

    if squeeze:
        data = data[kk]

    return header, data


def python_environment():
    """Tries to determine the current python environment, one of: 'jupyter', 'ipython', 'terminal'.
    """
    try:
        # NOTE: `get_ipython` should not be explicitly imported from anything
        ipy_str = str(type(get_ipython())).lower()  # noqa
        # print("ipy_str = '{}'".format(ipy_str))
        if 'zmqshell' in ipy_str:
            return 'jupyter'
        if 'terminal' in ipy_str:
            return 'ipython'
    except:
        return 'terminal'


def tqdm(*args, **kwargs):
    if python_environment().lower().startswith('jupyter'):
        import tqdm.notebook
        tqdm_method = tqdm.notebook.tqdm
    else:
        import tqdm
        tqdm_method = tqdm.tqdm

    return tqdm_method(*args, **kwargs)


# =================================================================================================
# ====    Mathematical & Numerical    ====
# =================================================================================================


def broadcastable(*args):
    """Expand N, 1D arrays be able to be broadcasted into N, ND arrays.

    e.g. from arrays of len `3`,`4`,`2`, returns arrays with shapes: `3,1,1`, `1,4,1` and `1,1,2`.
    """
    ndim = len(args)
    assert np.all([1 == np.ndim(aa) for aa in args]), "Each array in `args` must be 1D!"

    cut_ref = [slice(None)] + [np.newaxis for ii in range(ndim-1)]
    cuts = [np.roll(cut_ref, ii).tolist() for ii in range(ndim)]
    outs = [aa[tuple(cc)] for aa, cc in zip(args, cuts)]
    return outs


def expand_broadcastable(*args):
    try:
        shape = np.shape(np.product(args, axis=0))
    except ValueError:
        shapes = [np.shape(aa) for aa in args]
        raise ValueError("Argument arrays are not broadcastable!  shapes={}".format(shapes))

    vals = [aa * np.ones(shape) for aa in args]
    return vals


def frac_str(vals, prec=2):
    """

    Arguments
    ---------
    vals : (N,) array of bool

    """
    num = np.count_nonzero(vals)
    den = vals.size
    frc = num / den
    rv = f"{num:.{prec}e}/{den:.{prec}e} = {frc:.{prec}e}"
    return rv


def interp(xnew, xold, yold, left=np.nan, right=np.nan, xlog=True, ylog=True):
    x1 = np.asarray(xnew)
    x0 = np.asarray(xold)
    y0 = np.asarray(yold)
    if xlog:
        x1 = np.log10(x1)
        x0 = np.log10(x0)
    if ylog:
        y0 = np.log10(y0)
        if (left is not None) and np.isfinite(left):
            left = np.log10(left)
        if (right is not None) and np.isfinite(right):
            right = np.log10(right)

    y1 = np.interp(x1, x0, y0, left=left, right=right)
    if ylog:
        y1 = np.power(10.0, y1)
    return y1


def isnumeric(val):
    try:
        float(str(val))
    except ValueError:
        return False

    return True


def isinteger(val):
    rv = isnumeric(val) and isinstance(val, numbers.Integral)
    return rv


def log_normal_base_10(mu, sigma, size=None, shift=0.0):
    _sigma = np.log(10**sigma)
    dist = np.random.lognormal(np.log(mu) + shift*np.log(10.0), _sigma, size)
    return dist


def minmax(vals, filter=False):
    if filter:
        vv = vals[np.isfinite(vals)]
    else:
        vv = vals
    extr = np.array([np.min(vv), np.max(vv)])
    return extr


def print_stats(stack=True, print_func=print, **kwargs):
    if stack:
        import traceback
        traceback.print_stack()
    for kk, vv in kwargs.items():
        print_func(f"{kk} = shape: {np.shape(vv)}, stats: {stats(vv)}")
    return


def nyquist_freqs(dur=15.0*YR, cad=0.1*YR, trim=None):
    """Calculate Nyquist frequencies for the given timing parameters.

    Arguments
    ---------
    dur : scalar, duration of observations
    cad : scalar, cadence of observations
    trim : (2,) or None,
        Specification of minimum and maximum frequencies outside of which to remove values.
        `None` can be used in place of either boundary, e.g. [0.1, None] would mean removing
        frequencies below `0.1` (and not trimming values above a certain limit).

    Returns
    -------
    freqs : array of scalar, Nyquist frequencies

    """
    fmin = 1.0 / dur
    fmax = 1.0 / cad
    # df = fmin / sample
    df = fmin
    freqs = np.arange(fmin, fmax + df/10.0, df)
    if trim is not None:
        if np.shape(trim) != (2,):
            raise ValueError("`trim` (shape: {}) must be (2,) of float!".format(np.shape(trim)))
        if trim[0] is not None:
            freqs = freqs[freqs > trim[0]]
        if trim[1] is not None:
            freqs = freqs[freqs < trim[1]]

    return freqs


def quantiles(values, percs=None, sigmas=None, weights=None, axis=None, values_sorted=False):
    """Compute weighted percentiles.

    NOTE: if `values` is a masked array, then only unmasked values are used!

    Arguments
    ---------
    values: (N,)
        input data
    percs: (M,) scalar [0.0, 1.0]
        Desired percentiles of the data.
    weights: (N,) or `None`
        Weighted for each input data point in `values`.
    values_sorted: bool
        If True, then input values are assumed to already be sorted.

    Returns
    -------
    percs : (M,) float
        Array of percentiles of the weighted input data.

    """
    if not isinstance(values, np.ma.MaskedArray):
        values = np.asarray(values)

    if (percs is None) == (sigmas is None):
        err = "either `percs` or `sigmas`, and not both, must be given!"
        log.error(err)
        raise ValueError(err)

    if percs is None:
        percs = sp.stats.norm.cdf(sigmas)

    if np.ndim(values) > 1:
        if axis is None:
            values = values.flatten()
    elif (axis is not None):
        raise ValueError("Cannot act along axis '{}' for 1D data!".format(axis))

    percs = np.array(percs)
    if weights is None:
        weights = np.ones_like(values)
    weights = np.array(weights)
    try:
        weights = np.ma.masked_array(weights, mask=values.mask)
    except AttributeError:
        pass

    assert np.all(percs >= 0.0) and np.all(percs <= 1.0), 'percentiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values, axis=axis)
        values = np.take_along_axis(values, sorter, axis=axis)
        weights = np.take_along_axis(weights, sorter, axis=axis)

    if axis is None:
        weighted_quantiles = np.cumsum(weights) - 0.5 * weights
        weighted_quantiles /= np.sum(weights)
        percs = np.interp(percs, weighted_quantiles, values)
        return percs

    weights = np.moveaxis(weights, axis, -1)
    values = np.moveaxis(values, axis, -1)

    weighted_quantiles = np.cumsum(weights, axis=-1) - 0.5 * weights
    weighted_quantiles /= np.sum(weights, axis=-1)[..., np.newaxis]
    percs = [np.interp(percs, weighted_quantiles[idx], values[idx])
             for idx in np.ndindex(values.shape[:-1])]
    percs = np.array(percs)
    return percs


def stats(vals, percs=None, prec=2):
    try:
        if len(vals) == 0:
            raise TypeError
    except TypeError:
        raise ValueError(f"`vals` (shape={np.shape(vals)}) is not iterable!")

    if percs is None:
        percs = [sp.stats.norm.cdf(1), 0.95, 1.0]
        percs = np.array(percs)
        percs = np.concatenate([1-percs[::-1], [0.5], percs])

    # stats = np.percentile(vals, percs*100)
    stats = quantiles(vals, percs)
    rv = ["{val:.{prec}e}".format(prec=prec, val=ss) for ss in stats]
    rv = ", ".join(rv)
    return rv


def trapz_loglog(
        yy: npt.ArrayLike,
        xx: npt.ArrayLike,
        bounds: Optional[Tuple[float, float]] = None,
        axis: int = -1,
        dlogx: Optional[float] = None,
        lntol: float = 1e-2,
        cumsum: bool = True,
) -> npt.ArrayLike:
    """Calculate integral, given `y = dA/dx` or `y = dA/dlogx` w/ trapezoid rule in log-log space.

    We are calculating the integral `A` given sets of values for `y` and `x`.
    To associate `yy` with `dA/dx` then `dlogx = None` [default], otherwise,
    to associate `yy` with `dA/dlogx` then `dlogx = True` for natural-logarithm, or `dlogx = b`
    for a logarithm of base `b`.

    For each interval (x[i+1], x[i]), calculate the integral assuming that y is of the form,
        `y = a * x^gamma`

    Arguments
    ---------
    yy : ndarray
    xx : (X,) array_like of scalar,
    bounds : (2,) array_like of scalar,
    axis : int,
    dlogx : scalar or None,
    lntol : scalar,

    Returns
    -------
    integ

    Notes
    -----
    *   When bounds are given that are not identical to input `xx` values, then interpolation must
        be performed.  This can be done on the resulting cumsum'd values, or on the input integrand
        values.  The cumsum values are *not necessarily a power-law* (for negative indices), and thus
        the interpolation is better performed on the input `yy` values.
    *   Interpolating the cumulative-integral works very badly, instead interpolate the x/y values
        initially to obtain the integral at the appropriate locations.

    """
    yy = np.asarray(yy)
    xx = np.asarray(xx)

    if bounds is not None:
        xextr = [xx.min(), xx.max()]
        if (len(bounds) != 2) or (bounds[0] < xextr[0]) or (xextr[1] < bounds[1]):
            err = f"Invalid `bounds` '{bounds}', xx extrema = '{xextr}'!"
            log.error(err)
            raise ValueError(err)

        newy = sp.interpolate.PchipInterpolator(np.log10(xx), np.log10(yy), extrapolate=False)
        newy = newy(bounds)

        ii = np.searchsorted(xx, bounds)
        xx = np.insert(xx, ii, bounds, axis=axis)
        yy = np.insert(yy, ii, newy, axis=axis)
        ii = np.array([ii[0], ii[1]+1])
        assert np.alltrue(xx[ii] == bounds), "FAILED!"

    # yy = np.ma.masked_values(yy, value=0.0, atol=0.0)

    if np.ndim(yy) != np.ndim(xx):
        if np.ndim(xx) != 1:
            raise ValueError("BAD SHAPES")
        cut = [slice(None)] + [np.newaxis for ii in range(np.ndim(yy)-1)]
        xx = xx[tuple(cut)]

    log_base = np.e
    if dlogx is not None:
        # If `dlogx` is True, then we're using log-base-e (i.e. natural-log)
        # Otherwise, set the log-base to the given value
        if dlogx is not True:
            log_base = dlogx

    # Numerically calculate the local power-law index
    delta_logx = np.diff(np.log(xx), axis=axis)
    gamma = np.diff(np.log(yy), axis=axis) / delta_logx
    xx = np.moveaxis(xx, axis, 0)
    yy = np.moveaxis(yy, axis, 0)
    aa = np.mean([xx[:-1] * yy[:-1], xx[1:] * yy[1:]], axis=0)
    aa = np.moveaxis(aa, 0, axis)
    xx = np.moveaxis(xx, 0, axis)
    yy = np.moveaxis(yy, 0, axis)
    # Integrate dA/dx   ::   A = (x1*y1 - x0*y0) / (gamma + 1)
    if ((dlogx is None) or (dlogx is False)):
        dz = np.diff(yy * xx, axis=axis)
        trapz = dz / (gamma + 1)
        # when the power-law is (near) '-1' then, `A = a * log(x1/x0)`
        idx = np.isclose(gamma, -1.0, atol=lntol, rtol=lntol)

    # Integrate dA/dlogx    ::    A = (y1 - y0) / gamma
    else:
        dy = np.diff(yy, axis=axis)
        trapz = dy / gamma
        # when the power-law is (near) '-1' then, `A = a * log(x1/x0)`
        idx = np.isclose(gamma, 0.0, atol=lntol, rtol=lntol)

    if np.any(idx):
        # if `xx.shape != yy.shape` then `delta_logx` should be shaped (N-1, 1, 1, 1...)
        # broadcast `delta_logx` to the same shape as `idx` in this case
        if np.shape(xx) != np.shape(yy):
            delta_logx = delta_logx * np.ones_like(aa)
        trapz[idx] = aa[idx] * delta_logx[idx]

    # integ = np.log(log_base) * np.cumsum(trapz, axis=axis)
    # integ = np.cumsum(trapz, axis=axis) / np.log(log_base)   # FIX: I think this is divided by base... 2021-10-05
    integ = trapz / np.log(log_base)
    if cumsum:
        integ = np.cumsum(integ, axis=axis)
    if bounds is not None:
        if not cumsum:
            log.warning("WARNING: bounds is not None, but cumsum is False!")
        integ = np.moveaxis(integ, axis, 0)
        lo, hi = integ[ii-1, ...]
        integ = hi - lo

    return integ


def trapz(yy: npt.ArrayLike, xx: npt.ArrayLike, axis: int = -1, cumsum: bool = True):
    """Perform a cumulative integration along the given axis.

    Arguments
    ---------
    yy : ArrayLike of scalar,
        Input to be integrated.
    xx : ArrayLike of scalar,
        The sample points corresponding to the `yy` values.
        This must be either be shaped as
        * the same number of dimensions as `yy`, with the same length along the `axis` dimension, or
        * 1D with length matching `yy[axis]`
    axis : int,
        The axis over which to integrate.

    Returns
    -------
    ct : ndarray of scalar,
        Cumulative trapezoid rule integration.

    """
    if np.ndim(xx) == 1:
        pass
    elif np.ndim(xx) == np.ndim(yy):
        xx = xx[axis]
    else:
        error(f"Bad shape for `xx` (xx.shape={np.shape(xx)}, yy.shape={np.shape(yy)})!")
    ct = np.moveaxis(yy, axis, 0)
    ct = 0.5 * (ct[1:] + ct[:-1])
    ct = np.moveaxis(ct, 0, -1)
    ct = ct * np.diff(xx)
    if cumsum:
        ct = np.cumsum(ct, axis=-1)
    ct = np.moveaxis(ct, -1, axis)
    return ct


def _parse_log_norm_pars(vals, size, default=None):
    """
    vals:
        ()   ==> (N,)
        (2,) ==> (N,) log_normal(vals)
        (N,) ==> (N,)

    """
    if (vals is None):
        if default is None:
            return None
        vals = default

    if np.isscalar(vals):
        vals = vals * np.ones(size)
    elif (isinstance(vals, tuple) or isinstance(vals, list)) and (len(vals) == 2):
        vals = log_normal_base_10(*vals, size=size)
    elif np.shape(vals) != (size,):
        err = "`vals` must be scalar, (2,) of scalar, or array (nbins={},) of scalar!".format(size)
        raise ValueError(err)

    return vals


# =================================================================================================
# ====    General Astronomy    ====
# =================================================================================================


def dfdt_from_dadt(dadt, sepa, mtot=None, freq_orb=None):
    if (mtot is None) and (freq_orb is None):
        error("Either `mtot` or `freq_orb` must be provided!")
    if freq_orb is None:
        freq_orb = kepler_freq_from_sepa(mtot, sepa)

    dfdt = - 1.5 * (freq_orb / sepa) * dadt
    return dfdt, freq_orb


def mtmr_from_m1m2(m1, m2=None):
    if m2 is not None:
        masses = np.stack([m1, m2], axis=-1)
    else:
        assert np.shape(m1)[-1] == 2, "If only `m1` is given, last dimension must be 2!"
        masses = np.asarray(m1)

    mtot = masses.sum(axis=-1)
    mrat = masses.min(axis=-1) / masses.max(axis=-1)
    return np.array([mtot, mrat])


def m1m2_from_mtmr(mt, mr):
    """Convert from total-mass and mass-ratio to individual masses.
    """
    mt = np.asarray(mt)
    mr = np.asarray(mr)
    m1 = mt/(1.0 + mr)
    m2 = mt - m1
    return np.array([m1, m2])


def frst_from_fobs(fobs, redz):
    """Calculate rest-frame frequency from observed frequency and redshift.
    """
    frst = fobs * (1.0 + redz)
    return frst


def fobs_from_frst(frst, redz):
    """Calculate observed frequency from rest-frame frequency and redshift.
    """
    fobs = frst / (1.0 + redz)
    return fobs


def kepler_freq_from_sepa(mass, sep):
    freq = (1.0/(2.0*np.pi))*np.sqrt(NWTG*mass)/np.power(sep, 1.5)
    return freq


def kepler_sepa_from_freq(mass, freq):
    mass = np.asarray(mass)
    freq = np.asarray(freq)
    sep = np.power(NWTG*mass/np.square(2.0*np.pi*freq), 1.0/3.0)
    return sep


def rad_isco(m1, m2, factor=3.0):
    """Inner-most Stable Circular Orbit, radius at which binaries 'merge'.
    """
    return factor * schwarzschild_radius(m1+m2)


def schwarzschild_radius(mass):
    rs = SCHW * mass
    return rs


# =================================================================================================
# ====    Gravitational Waves    ====
# =================================================================================================


def chirp_mass(m1, m2=None):
    # (N, 2)  ==>  (N,), (N,)
    if m2 is None:
        m1, m2 = np.moveaxis(m1, -1, 0)
    mc = np.power(m1 * m2, 3.0/5.0)/np.power(m1 + m2, 1.0/5.0)
    return mc


def gw_char_strain(hs, dur_obs, freq_orb_obs, freq_orb_rst, dfdt):
    """
    See, e.g., Sesana+2004, Eq.35

    Arguments
    ---------
    hs : array_like scalar
        Strain amplitude (e.g. `gw_strain()`, sky- and polarization- averaged)
    dur_obs : array_like scalar
        Duration of observations, in the observer frame

    """

    ncycles = freq_orb_rst**2 / dfdt
    ncycles = np.clip(ncycles, None, dur_obs * freq_orb_obs)
    hc = hs * np.sqrt(ncycles)
    return hc


def gw_dedt(m1, m2, sepa, eccen):
    """GW Eccentricity Evolution rate (de/dt).

    returned value is negative.

    See Peters 1964, Eq. 5.8
    http://adsabs.harvard.edu/abs/1964PhRv..136.1224P
    """
    cc = _GW_DEDT_ECC_CONST
    e2 = eccen**2
    dedt = cc * m1 * m2 * (m1 + m2) / np.power(sepa, 4)
    dedt *= (1.0 + e2*121.0/304.0) * eccen / np.power(1 - e2, 5.0/2.0)
    return dedt


def gw_dade(m1, m2, sepa, eccen):
    """GW Eccentricity Evolution rate (de/dt).

    returned value is positive (e and a go in same direction).

    See Peters 1964, Eq. 5.7
    http://adsabs.harvard.edu/abs/1964PhRv..136.1224P
    """
    e2 = eccen**2
    num = (1 + (73.0/24.0)*e2 + (37.0/96.0)*e2*e2)
    den = (1 - e2) * (1.0 + (121.0/304.0)*e2)
    dade = (12.0 / 19.0) * (sepa / eccen) * (num / den)
    return dade


def gw_freq_dist_func(nn, ee=0.0):
    """Frequency Distribution Function.

    See [Enoki & Nagashima 2007](astro-ph/0609377) Eq. 2.4.
    This function gives g(n,e)

    FIX: use recursion relation when possible,
        J_{n-1}(x) + J_{n+1}(x) = (2n/x) J_n(x)
    """
    import scipy as sp
    import scipy.special  # noqa

    # Calculate with non-zero eccentrictiy
    bessel = sp.special.jn
    ne = nn*ee
    n2 = np.square(nn)
    jn_m2 = bessel(nn-2, ne)
    jn_m1 = bessel(nn-1, ne)

    # Use recursion relation:
    jn = (2*(nn-1) / ne) * jn_m1 - jn_m2
    jn_p1 = (2*nn / ne) * jn - jn_m1
    jn_p2 = (2*(nn+1) / ne) * jn_p1 - jn

    aa = np.square(jn_m2 - 2.0*ee*jn_m1 + (2/nn)*jn + 2*ee*jn_p1 - jn_p2)
    bb = (1 - ee*ee)*np.square(jn_m2 - 2*ee*jn + jn_p2)
    cc = (4.0/(3.0*n2)) * np.square(jn)
    gg = (n2*n2/32) * (aa + bb + cc)
    return gg


def gw_hardening_rate_dadt(m1, m2, sepa, eccen=None):
    """GW Hardening rate in separation (da/dt).

    returned value is negative.

    See Peters 1964, Eq. 5.6
    http://adsabs.harvard.edu/abs/1964PhRv..136.1224P
    """
    cc = _GW_DADT_SEP_CONST
    dadt = cc * m1 * m2 * (m1 + m2) / np.power(sepa, 3)
    if eccen is not None:
        fe = _gw_ecc_func(eccen)
        dadt *= fe
    return dadt


def gw_hardening_rate_dfdt(m1, m2, freq, eccen=None):
    """GW Hardening rate in frequency (df/dt).
    """
    sepa = kepler_sepa_from_freq(m1+m2, freq)
    dfdt = gw_hardening_rate_dadt(m1, m2, sepa, eccen=eccen)
    dfdt = dfdt_from_dadt(dfdt, sepa, mtot=m1+m2)
    return dfdt


def gw_hardening_timescale_freq(mchirp, frst):
    """tau = f_r / (df_r / dt)

    e.g. [EN07] Eq.2.9

    Arguments
    ---------
    mchirp : scalar  or  array_like of scalar
        Chirp mass in [grams]
    frst : scalar  or  array_like of scalar
        Rest-frame orbital frequency

    Returns
    -------
    tau : float  or  array_like of float
        GW hardening timescale defined w.r.t. orbital frequency.

    """
    tau = (5.0 / 96.0) * np.power(NWTG*mchirp/SPLC**3, -5.0/3.0) * np.power(2*np.pi*frst, -8.0/3.0)
    return tau


def gw_lum_circ(mchirp, freq_orb_rest):
    """
    EN07: Eq. 2.2
    """
    lgw_circ = _GW_LUM_CONST * np.power(2.0*np.pi*freq_orb_rest*mchirp, 10.0/3.0)
    return lgw_circ


'''
def gw_strain_source(mchirp, dlum, freq_orb_rest):
    """GW Strain from a single source in a circular orbit.

    e.g. Sesana+2004 Eq.36
    e.g. EN07 Eq.17
    """
    #
    hs = _GW_SRC_CONST * mchirp * np.power(2*mchirp*freq_orb_rest, 2/3) / dlum
    return hs
'''


def gw_strain_source(mchirp, dcom, freq_orb_rest):
    """GW Strain from a single source in a circular orbit.

    e.g. Sesana+2004 Eq.36
    e.g. Enoki+2004 Eq.5
    """
    #
    hs = _GW_SRC_CONST * mchirp * np.power(2*mchirp*freq_orb_rest, 2/3) / dcom
    return hs


def sep_to_merge_in_time(m1, m2, time):
    """The initial separation required to merge within the given time.

    See: [Peters 1964].
    """
    GW_CONST = 64*np.power(NWTG, 3.0)/(5.0*np.power(SPLC, 5.0))
    a1 = rad_isco(m1, m2)
    return np.power(GW_CONST*m1*m2*(m1+m2)*time - np.power(a1, 4.0), 1./4.)


def time_to_merge_at_sep(m1, m2, sep):
    """The time required to merge starting from the given initial separation.

    See: [Peters 1964].
    """
    GW_CONST = 64*np.power(NWTG, 3.0)/(5.0*np.power(SPLC, 5.0))
    a1 = rad_isco(m1, m2)
    delta_sep = np.power(sep, 4.0) - np.power(a1, 4.0)
    return delta_sep/(GW_CONST*m1*m2*(m1+m2))


def _gw_ecc_func(eccen):
    """GW Hardening rate eccentricitiy dependence F(e).

    See Peters 1964, Eq. 5.6
    EN07: Eq. 2.3
    """
    e2 = eccen*eccen
    num = 1 + (73/24)*e2 + (37/96)*e2*e2
    den = np.power(1 - e2, 7/2)
    fe = num / den
    return fe
