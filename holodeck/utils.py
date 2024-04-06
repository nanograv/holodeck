"""Utility functions and tools.

References
----------
* [Peters1964]_ Peters 1964
* [Enoki2004]_ Enoki, Inoue, Nagashima, & Sugiyama 2004
* [Sesana2004]_ Sesana, Haardt, Madau, & Volonteri 2004
* [EN2007]_ Enoki & Nagashima 2007

"""

import abc
import copy
import functools
import inspect
import numbers
import os
import subprocess
import warnings
from pathlib import Path
from typing import Optional, Tuple, Union, List   #, Callable, TypeVar, Any  # , TypeAlias  # , Sequence,

# try:
#     from typing import ParamSpec
# except ImportError:
#     from typing_extensions import ParamSpec

import h5py
import numba
import numpy as np
import numpy.typing as npt
import scipy as sp
import scipy.stats    # noqa
import scipy.special  # noqa

from holodeck import log, cosmo
from holodeck.constants import NWTG, SCHW, SPLC, YR, GYR, EDDT

# [Sesana2004]_ Eq.36
_GW_SRC_CONST = 8 * np.power(NWTG, 5/3) * np.power(np.pi, 2/3) / np.sqrt(10) / np.power(SPLC, 4)
_GW_DADT_SEP_CONST = - 64 * np.power(NWTG, 3) / 5 / np.power(SPLC, 5)
_GW_DEDT_ECC_CONST = - 304 * np.power(NWTG, 3) / 15 / np.power(SPLC, 5)
# [EN2007]_, Eq.2.2
_GW_LUM_CONST = (32.0 / 5.0) * np.power(NWTG, 7.0/3.0) * np.power(SPLC, -5.0)

_AGE_UNIVERSE_GYR = cosmo.age(0.0).to('Gyr').value  # [Gyr]  ~ 13.78


class _Modifier(abc.ABC):
    """Base class for all types of post-processing modifiers.

    Notes
    -----
    * Must be subclassed for use.
    * ``__call__(base)`` ==> ``modify(base)``

    """

    def __call__(self, base):
        self.modify(base)
        return

    @abc.abstractmethod
    def modify(self, base: object):
        """Perform an in-place modification on the passed object instance.

        Parameters
        ----------
        base: object
            The object instance to be modified.

        """
        pass


# T = TypeVar('T')
# P = ParamSpec('P')
# WrappedFuncDeco: TypeAlias = Callable[[Callable[P, T]], Callable[P, T]]
# WrappedFuncDeco: 'TypeAlias' = Tuple[float, float]

# def copy_docstring(copy_func: Callable[..., Any]) -> WrappedFuncDeco[P, T]:
#     """Copies the doc string of the given function to the wrapped function.

#     see: https://stackoverflow.com/a/68901244/230468
#     """

#     def wrapped(func: Callable[P, T]) -> Callable[P, T]:
#         func.__doc__ = copy_func.__doc__
#         return func

#     return wrapped


# =================================================================================================
# ====    General Logistical    ====
# =================================================================================================


def deprecated_warn(msg, exc_info=True):
    """Decorator for functions that will be deprecated, add warning, but still execute function.
    """

    def decorator(func):
        nonlocal msg

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal msg
            old_name = func.__name__
            _frame = inspect.currentframe().f_back
            file_name = inspect.getfile(_frame.f_code)
            fline = _frame.f_lineno
            msg = f"{file_name}({fline}):{old_name} is deprecated!" + " | " + msg
            warnings.warn_explicit(msg, category=DeprecationWarning, filename=file_name, lineno=fline)
            log.warning(f"DEPRECATION: {msg}", exc_info=exc_info)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def deprecated_pass(new_func, msg="", exc_info=True):
    """Decorator for functions that have been deprecated, warn and pass arguments to new function.
    """

    def decorator(func):
        nonlocal msg

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal msg
            old_name = func.__name__
            try:
                new_name = new_func.__name__
            except AttributeError:
                new_name = str(new_func)
            _frame = inspect.currentframe().f_back
            file_name = inspect.getfile(_frame.f_code)
            fline = _frame.f_lineno
            msg = f"{file_name}({fline}):{old_name} ==> {new_name}" + (len(msg) > 0) * " | " + msg
            warnings.warn_explicit(msg, category=DeprecationWarning, filename=file_name, lineno=fline)
            log.warning(f"DEPRECATION: {msg}", exc_info=exc_info)
            return new_func(*args, **kwargs)

        return wrapper

    return decorator


def deprecated_fail(new_func, msg="", exc_info=True):
    """Decorator for functions that have been deprecated, warn and raise error.
    """

    def decorator(func):
        nonlocal msg

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal msg
            old_name = func.__name__
            try:
                new_name = new_func.__name__
            except AttributeError:
                new_name = str(new_func)
            _frame = inspect.currentframe().f_back
            file_name = inspect.getfile(_frame.f_code)
            fline = _frame.f_lineno
            msg = f"{file_name}({fline}):{old_name} ==> {new_name}" + (len(msg) > 0) * " | " + msg
            warnings.warn_explicit(msg, category=DeprecationWarning, filename=file_name, lineno=fline)
            log.exception(f"DEPRECATION: {msg}", exc_info=exc_info)
            raise RuntimeError

        return wrapper

    return decorator


def load_hdf5(fname, keys=None):
    """Load data and header information from HDF5 files into dictionaries.

    Parameters
    ----------
    fname : str
        Filename to load (must be an `hdf5` file).
    keys : None or (list of str)
        Specific keys to load from the top-level of the HDF5 file.
        `None`: load all top-level keys.

    Returns
    -------
    header : dict,
        All entries from `hdf5.File.attrs`, typically used for meta-data.
    data : dict,
        All top level datasets from the hdf5 file,
        specifically everything returned from `hdf5.File.keys()`.

    """
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


def mpi_print(*args, **kwargs):
    return print(*args, flush=True, **kwargs)


def python_environment():
    """Tries to determine the current python environment, one of: 'jupyter', 'ipython', 'terminal'.

    Returns
    -------
    str
        Description of the current python environment, one of ['jupter', 'ipython', 'terminal'].

    """
    try:
        # NOTE: `get_ipython` should *not* be explicitly imported from anything
        #       it will be defined if this is called from a jupyter or ipython environment
        ipy_str = str(type(get_ipython())).lower()  # noqa
        if 'zmqshell' in ipy_str:
            return 'jupyter'
        if 'terminal' in ipy_str:
            return 'ipython'
    except:   # noqa
        return 'terminal'

    raise RuntimeError(f"unexpected result from `get_ipython()`: '{ipy_str}'!")


def tqdm(*args, **kwargs):
    """Construct a progress bar appropriately based on the current environment (script vs. notebook)

    Parameters
    ----------
    *args, **kwargs : All arguments are passed directory to the `tqdm` constructor.

    Returns
    -------
    `tqdm.tqdm_gui`
        Decorated iterator that shows a progress bar.

    """
    if python_environment().lower().startswith('jupyter'):
        import tqdm.notebook
        tqdm_method = tqdm.notebook.tqdm
    else:
        import tqdm
        tqdm_method = tqdm.tqdm

    return tqdm_method(*args, **kwargs)


def get_file_size(fnames, precision=1):
    """Return a human-readable size of a file or set of files.

    Parameters
    ----------
    fnames : str or list
        Paths to target file(s)
    precisions : int,
        Sesired decimal precision of output

    Returns
    -------
    byte_str : str
        Human-readable size of file(s)

    """
    fnames = np.atleast_1d(fnames)

    byte_size = 0.0
    for fil in fnames:
        byte_size += os.path.getsize(fil)

    abbrevs = (
        (1 << 50, 'PiB'),
        (1 << 40, 'TiB'),
        (1 << 30, 'GiB'),
        (1 << 20, 'MiB'),
        (1 << 10, 'KiB'),
        (1, 'bytes')
    )

    for factor, suffix in abbrevs:
        if byte_size >= factor:
            break

    size = byte_size / factor
    byte_str = f"{size:.{precision:}f} {suffix}"
    return byte_str


def path_name_ending(path, ending):
    fname = Path(path)
    name_bare = fname.with_suffix("")
    fname = fname.parent.joinpath(str(name_bare) + ending).with_suffix(fname.suffix)
    return fname


def get_subclass_instance(value, default, superclass, allow_none=False):
    """Convert the given `value` into a subclass instance.

    `None` ==> instance from `default` class
    Class ==> instance from that class
    instance ==> check that this is an instance of a subclass of `superclass`, error if not

    Parameters
    ----------
    value : object,
        Object to convert into a class instance.
    default : class,
        Default class constructor to use if `value` is None.
    superclass : class,
        Super/parent class to compare against the class instance from `value` or `default`.
        If the class instance is not a subclass of `superclass`, a ValueError is raised.

    Returns
    -------
    value : object,
        Class instance that is a subclass of `superclass`.

    Raises
    ------
    ValueError : if the class instance is not a subclass of `superclass`.

    """
    import inspect

    # Set `value` to a default, if needed and it is given
    if (value is None) and (default is not None):
        value = default

    # if `value` is not set, and there is no default, and `None` is allowed... just return None
    if (value is None) and allow_none:
        return value

    # If `value` is a class (constructor), then construct an instance from it
    if inspect.isclass(value):
        value = value()

    # Raise an error if `value` is not a subclass of `superclass`
    if not isinstance(value, superclass):
        err = f"argument ({value}) must be an instance or subclass of `{superclass}`!"
        log.error(err)
        raise ValueError(err)

    return value


def get_git_hash(short=True) -> str:
    args = ['git', 'rev-parse', 'HEAD']
    if short:
        args.insert(2, "--short")
    return subprocess.check_output(args).decode('ascii').strip()


# =================================================================================================
# ====    Mathematical & Numerical    ====
# =================================================================================================


def roll_rows(arr, roll_num):
    """Roll each row (axis=0) of the given array by an amount specified.

    Parameters
    ----------
    arr : (R, D) ndarray
        Input data to be rolled.
    roll_num : (R,) ndarray of int
        Amount to roll each row.  Must match the number of rows (axis=0) in `arr`.

    Returns
    -------
    result : (R, D) ndarray
        Rolled version of the input data.

    Example
    -------
    >>> a = np.arange(12).reshape(3, 4); b = [1, -1, 2]; utils.roll_rows(a, b)
    array([[ 3,  0,  1,  2],
           [ 5,  6,  7,  4],
           [10, 11,  8,  9]])

    """
    roll = np.asarray(roll_num)
    assert np.ndim(arr) == 2 and np.ndim(roll) == 1
    nrows, ncols = arr.shape
    assert roll.size == nrows
    arr_roll = arr[:, [*range(ncols), *range(ncols-1)]].copy()
    strd_0, strd_1 = arr_roll.strides
    result = np.lib.stride_tricks.as_strided(arr_roll, (nrows, ncols, ncols), (strd_0, strd_1, strd_1))
    result = result[np.arange(nrows), (ncols - roll)%ncols]
    return result


def get_scatter_weights(uniform_cents, dist):
    """Get the weights (fractional mass) that should be transferred to each bin to introduce the given scatter.

    Parameters
    ----------
    uniform_cents : (N,) ndarray
        Uniformly spaced bin-centers specifying distances in the parameter of interest (e.g. mass).
    dist : `scipy.stats._distn_infrastructure.rv_continuous_frozen` instance
        Object providing a CDF function `cdf(x)` determining the weights for each bin.
        e.g. ``dist = sp.stats.norm(loc=0.0, scale=0.1)``

    Returns
    -------
    dm : (2*N - 1,) ndarray
        Array of weights for bins with the given distances.
        [-N+1, -N+2, ..., -2, -1, 0, +1, +2, ..., +N-2, +N-1]

    """
    num = uniform_cents.size
    # Get log-spacing between edges, this must be constant to work in this way!
    dx = np.diff(uniform_cents)
    # assert np.allclose(dx, dx[0]), "This method only works if `uniform_cents` are uniformly spaced!"
    if not np.allclose(dx, dx[0]):
        log.error(f"{dx[0]=} {dx=}")
        log.error(f"{uniform_cents=}")
        err = "`get_scatter_weights` only works if `uniform_cents` are uniformly spaced!"
        log.exception(err)
        raise ValueError(err)

    dx = dx[0]
    # The bin edges are at distance [dx/2, 1.5*dx, 2.5*dx, ...]
    dx = dx/2.0 + np.arange(num) * dx
    # Convert to both sides:  [..., -1.5*dx, -0.5dx, +0.5dx, +1.5dx, ...]
    dx = np.concatenate([-dx[::-1], dx])
    # Get the mass across each interval by differencing the CDF at each edge location
    dm = np.diff(dist.cdf(dx))
    return dm


def _scatter_with_weights(dens, weights, axis=0):
    # Perform the convolution
    dens = np.moveaxis(dens, axis, 0)
    dens_new = np.einsum("j...,jk...", dens, weights)
    dens_new = np.moveaxis(dens_new, 0, axis)
    dens = np.moveaxis(dens, 0, axis)
    return dens_new


def _get_rolled_weights(log_cents, dist):
    num = log_cents.size
    # Get the fractional weights that this bin should be redistributed to
    # (2*N - 1,)  giving the bins all the way to the left and the right
    # e.g. [-N+1, -N+2, ..., -2, -1, 0, +1, +2, ..., +N-2, +N-1]
    weights = get_scatter_weights(log_cents, dist)

    # Duplicate the weights into each row of an (N, N) matrix
    # e.g. [[-N+1, -N+2, ..., -2, -1, 0, +1, +2, ..., +N-2, +N-1]
    #       [-N+1, -N+2, ..., -2, -1, 0, +1, +2, ..., +N-2, +N-1]
    #       [-N+1, -N+2, ..., -2, -1, 0, +1, +2, ..., +N-2, +N-1]
    #        ...
    weights = weights[np.newaxis, :] * np.ones((num, weights.size))
    # Need to "roll" each row of the matrix such that the central bin is at number index=row
    #    rolls backward by default,
    roll = 1 - num + np.arange(num)
    # Roll each row
    # e.g. [[ 0, +1, +2, ..., +N-2, +N-1, -N+1, -N+2, ..., -2, -1]
    #       [-1,  0, +1, +2, ..., +N-2, +N-1, -N+1, -N+2, ..., -2]
    #       [-2, -1,  0, +1, +2, ..., +N-2, +N-1, -N+1, -N+2, ..., -3]
    #        ...
    weights = roll_rows(weights, roll)
    # Cutoff each row after N elements
    weights = weights[:, :num]
    return weights


def scatter_redistribute_densities(cents, dens, dist=None, scatter=None, axis=0):
    """Redistribute `dens` across the target axis to account for scatter/variance.

    Parameters
    ----------
    cents : (N,) ndarray
        Locations of bin centers in the parameter of interest.
    dist : `scipy.stats._distn_infrastructure.rv_continuous_frozen` instance
        Object providing a CDF function `cdf(x)` determining the weights for each bin.
        e.g. ``dist = sp.stats.norm(loc=0.0, scale=0.1)``
    dens : ndarray
        Input values to be redistributed.  Must match the size of `cents` along axis `axis`.

    Returns
    -------
    dens_new : ndarray
        Array with resitributed values.  Same shape as input `dens`.

    """
    if (dist is None) == (scatter is None):
        raise ValueError(f"One and only one of `dist` ({dist}) and `scatter` ({scatter}) must be provided!")

    if dist is None:
        dist = sp.stats.norm(loc=0.0, scale=scatter)

    log_cents = np.log10(cents)
    num = log_cents.size
    if np.shape(dens)[axis] != num:
        err = f"The size of `dens` ({np.shape(dens)}) along `axis` ({axis}) must match `cents` ({num})!!"
        log.exception(err)
        raise ValueError(err)

    weights = _get_rolled_weights(log_cents, dist)
    dens_new = _scatter_with_weights(dens, weights, axis=0)
    return dens_new


def eccen_func(cent: float, width: float, size: int) -> np.ndarray:
    """Draw random values between [0.0, 1.0] with a given center and width.

    This function is a bit contrived, but the `norm` defines the center-point of the distribution,
    and the `std` parameter determines the width of the distribution.  In all cases the resulting
    values are only between [0.0, 1.0].  This function is typically used to draw initial random
    eccentricities.

    Parameters
    ----------
    cent : float,
        Specification of the center-point of the distribution.  Range: positive numbers.
        Values `norm << 1` correspond to small eccentricities, while `norm >> 1` are large
        eccentricities, with the distribution symmetric around `norm=1.0` (and eccens of 0.5).
    width : float,
        Specification of the width of the distribution.  Specifically how near or far values tend
        to be from the given central value (`norm`).  Range: positive numbers.
        Note that the 'width' of the distribution depends on the `norm` value, in addition to `std`.
        Smaller values (typically `std << 1`) produce narrower distributions.
    size : int,
        Number of samples to draw.

    Returns
    -------
    eccen : ndarray,
        Values between [0.0, 1.0] with shape given by the `size` parameter.

    """
    assert np.shape(cent) == () and cent > 0.0
    assert np.shape(width) == () and width > 0.0
    eccen = log_normal_base_10(1.0/cent, width, size=size)
    eccen = 1.0 / (eccen + 1.0)
    return eccen


def frac_str(vals, prec=2):
    """Return a string with the fraction and decimal of non-zero elements of the given array.

    e.g. [0, 1, 2, 0, 0] ==> "2/5 = 4.0e-1"

    Parameters
    ----------
    vals : (N,) array_like,
        Input array to find non-zero elements of.
    prec : int
        Decimal precision in scientific notation string.

    Returns
    -------
    rv : str,
        Fraction string.

    """
    num = np.count_nonzero(vals)
    den = np.size(vals)
    frc = num / den
    rv = f"{num:.{prec}e}/{den:.{prec}e} = {frc:.{prec}e}"
    return rv


def interp(xnew, xold, yold, left=np.nan, right=np.nan, xlog=True, ylog=True):
    """Linear interpolation of the given arguments in log/lin-log/lin space.

    Parameters
    ----------
    xnew : npt.ArrayLike
        New locations (independent variable) to interpolate to.
    xold : npt.ArrayLike
        Old locations of independent variable.
    yold : npt.ArrayLike
        Old locations of dependent variable.
    left : float, optional
        Fill value for locations below the domain `xold`.
    right : float, optional
        Fill value for locations above the domain `xold`.
    xlog : bool, optional
        Linear interpolation in the log of x values.
    ylog : bool, optional
        Linear interpolation in the log of y values.

    Returns
    -------
    y1 : npt.ArrayLike
        Interpolated output values of the dependent variable.

    """
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


def isnumeric(val: object) -> bool:
    """Test if the input value can successfully be cast to a float.

    Parameters
    ----------
    val : object
        Value to test.

    Returns
    -------
    bool
        True if the input value can be cast to a float.

    """
    try:
        float(str(val))   # ! FIX: Why does this need to cast to `str` first?
    except ValueError:
        return False

    return True


def isinteger(val: object) -> bool:
    """Test if the input value is an integral (integer) number.

    Parameters
    ----------
    val : object
        Value to test.

    Returns
    -------
    bool
        True if the input value is an integer number.

    """
    rv = isnumeric(val) and isinstance(val, numbers.Integral)
    return rv


def log_normal_base_10(
    mu: float, sigma: float, size: Union[int, List[int]] = None, shift: float = 0.0,
) -> np.ndarray:
    """Draw from a log-normal distribution using base-10 standard-deviation.

    i.e. the `sigma` argument is in "dex", or powers of ten.

    Parameters
    ----------
    mu : float
        Mean value of the distribution.
    sigma : float
        Standard deviation in dex (i.e. powers of ten).
        `sigma=1.0` means a standard deviation of one order of magnitude around mu.
    size : Union[int, list[int]], optional
        Number of values to draw.  Either a single integer, or a tuple of integers describing a shape.
    shift : float, optional

    Returns
    -------
    dist : npt.ArrayLike
        Resulting distribution values.

    """
    _sigma = np.log(10.0 ** sigma)
    dist = np.random.lognormal(np.log(mu) + shift*np.log(10.0), _sigma, size)
    return dist


def midpoints(vals, axis=-1, log=False):
    mm = np.moveaxis(vals, axis, 0)
    if log:
        mm = np.log10(mm)
    mm = 0.5 * (mm[1:] + mm[:-1])
    if log:
        mm = 10.0 ** mm
    mm = np.moveaxis(mm, 0, axis)
    return mm


def midpoints_multiax(vals, axis, log=False):
    for aa in axis:
        vals = midpoints(vals, aa, log=log)
    return vals


def minmax(vals: npt.ArrayLike, filter: bool = False) -> np.ndarray:
    """Find the minimum and maximum values in the given array.

    Parameters
    ----------
    vals : npt.ArrayLike
        Input values in which to find extrema.
    filter : bool, optional
        Select only finite values from the input array.

    Returns
    -------
    extr : (2,) np.ndarray
        Minimum and maximum values.

    """
    if filter:
        vals = np.asarray(vals)
        vv = vals[np.isfinite(vals)]
    else:
        vv = vals
    extr = np.array([np.min(vv), np.max(vv)])
    return extr


def ndinterp(xx, xvals, yvals, xlog=False, ylog=False):
    """Interpolate 2D data to an array of points.

    `xvals` and `yvals` are (N, M) where the interpolation is done along the 1th (`M`)
    axis (i.e. interpolation is done independently for each `N` row.  Should be generalizeable to
    higher dim.

    Parameters
    ----------
    xx : (T,) or (N, T) ndarray
        Target x-values to interpolate to.
    xvals : (N, M) ndarray
        Evaluation points (x-values) of the functions to be interpolated.
        Interpolation is performed over the 1th (last) axis.
        NOTE: values *must* be monotonically increasing along axis=1 !
    yvals : (N, M) ndarray
        Function values (y-values) of the function to be interpolated.
        Interpolation is performed over the 1th (last) axis.

    Returns
    -------
    ynew : (N, T) ndarray
        Interpolated function values, for each of N functions and T evaluation points.

    """
    # assert np.ndim(xx) == 1
    assert np.ndim(xvals) == 2
    assert np.shape(xvals) == np.shape(yvals)

    xx = np.asarray(xx)
    xvals = np.asarray(xvals)
    yvals = np.asarray(yvals)

    if xlog:
        xx = np.log10(xx)
        xvals = np.log10(xvals)

    if ylog:
        yvals = np.log10(yvals)

    # --- Convert `xx` to be broadcastable with (N, T)
    # `xx` is shaped as (T,)  ==> (1, T)
    if np.ndim(xx) == 1:
        xx = xx[np.newaxis, :]
    # `xx` is shaped as (N, T)
    elif np.ndim(xx) == 2:
        assert np.shape(xx)[0] == np.shape(xvals)[0]
    else:
        err = f"`xx` ({np.shape(xx)}) must be shaped as (T,) or (N, T)!"
        log.exception(err)
        raise ValueError(err)

    # Convert to (N, T, M)
    #     `xx` is (T,)  `xvals` is (N, M) for N-binaries and M-steps
    select = (xx[:, :, np.newaxis] <= xvals[:, np.newaxis, :])

    # ---- Find the indices in `xvals` after and before each value of `xx`
    # Find the first indices in `xvals` AFTER `xx`
    # (N, T)
    aft = np.argmax(select, axis=-1)
    # zero values in `aft` mean that either (a) no xvals after the targets were found
    # of (b) that all xvals are after the targets.  In either case, we cannot interpolate!
    valid = (aft > 0)
    inval = ~valid
    # find the last indices when `xvals` is SMALLER than each value of `xx`
    bef = np.copy(aft)
    bef[valid] -= 1

    # (2, N, T)
    cut = [aft, bef]
    # (2, N, T)
    xvals = [np.take_along_axis(xvals, cc, axis=-1) for cc in cut]
    # Find how far to interpolate between values (in log-space)
    #     (N, T)
    frac = (xx - xvals[1]) / np.subtract(*xvals)

    # (2, N, T)
    data = [np.take_along_axis(yvals, cc, axis=-1) for cc in cut]
    # Interpolate by `frac` for each binary
    ynew = data[1] + (np.subtract(*data) * frac)
    # Set invalid binaries to nan
    ynew[inval, ...] = np.nan

    if ylog:
        ynew = 10.0 ** ynew

    return ynew


def pta_freqs(dur=16.03*YR, num=40, cad=None):
    """Get Fourier frequency bin specifications for the given parameters.

    Arguments
    ---------
    dur : float,
        Total observing duration, which determines the minimum sensitive frequency, ``1/dur``.
        Typically `dur` should be given in units of [sec], such that the returned frequencies are
        in units of [1/sec] = [Hz]
    num : int,
        Number of frequency bins.  If `cad` is not None, then the number of frequency bins is
        determined by `cad` and the `num` value is disregarded.
    cad : float or `None`,
        Cadence of observations, which determines the maximum sensitive frequency (i.e. the Nyquist
        frequency).  If `cad` is not given, then `num` frequency bins are constructed.

    Returns
    -------
    cents : (F,) ndarray
        Bin-center frequencies for `F` bins.  The frequency bin centers are at:
        ``F_i = (i + 1.5) / dur`` for i between 0 and `num-1`.
        The number of frequency bins, `F` is the argument `num`,
        or determined by `cad` if it is given.
    edges : (F+1,) ndarray
        Bin-edge frequencies for `F` bins, i.e. `F+1` bin edges.  The frequency bin edges are at:
        ``F_i = (i + 1) / dur`` for i between 0 and `num`.
        The number of frequency bins, `F` is the argument `num`,
        or determined by `cad` if it is given.

    """
    fmin = 1.0 / dur
    if cad is not None:
        num = dur / (2.0 * cad)
        num = int(np.floor(num))

    cents = np.arange(1, num+2) * fmin

    edges = cents - fmin / 2.0
    cents = cents[:-1]
    return cents, edges


def print_stats(stack=True, print_func=print, **kwargs):
    """Print out basic properties and statistics on the input key-value array_like values.

    Parameters
    ----------
    stack : bool,
        Whether or not to print a backtrace to stdout.
    print_func : callable,
        Function to use for returning/printing output.
    kwargs : dict,
        Key-value pairs where values are array_like for the shape/stats to be printed.

    """
    if stack:
        import traceback
        traceback.print_stack()
    for kk, vv in kwargs.items():
        print_func(f"{kk} = shape: {np.shape(vv)}, stats: {stats(vv)}")
    return


def quantile_filtered(values, percs, axis, func=np.isfinite):
    percs = np.asarray(percs)
    assert np.all((percs > 0.0) & (percs < 1.0))
    return np.apply_along_axis(lambda xx: np.percentile(np.asarray(xx)[func(xx)], percs*100), axis, values)


def quantiles(values, percs=None, sigmas=None, weights=None, axis=None,
              values_sorted=False, filter=None):
    """Compute weighted percentiles.

    NOTE: if `values` is a masked array, then only unmasked values are used!

    Parameters
    ----------
    values: (N,)
        input data
    percs: (M,) scalar
        Desired quantiles of the data.  Within range of [0.0, 1.0].
    weights: (N,) or `None`
        Weights for each input data point in `values`.
    axis: int or `None`,
        Axis over which to calculate quantiles.
    values_sorted: bool
        If True, then input values are assumed to already be sorted.
        Otherwise they are sorted before calculating quantiles (for efficiency).

    Returns
    -------
    percs : (M,) float
        Array of quantiles of the input data.

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
        ww = np.ones_like(values)
    else:
        ww = np.array(weights)

    try:
        ww = np.ma.masked_array(ww, mask=values.mask)  # type: ignore
    except AttributeError:
        pass

    assert np.all(percs >= 0.0) and np.all(percs <= 1.0), 'percentiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values, axis=axis)
        values = np.take_along_axis(values, sorter, axis=axis)
        ww = np.take_along_axis(ww, sorter, axis=axis)

    if axis is None:
        weighted_quantiles = np.cumsum(ww) - 0.5 * ww
        weighted_quantiles /= np.sum(ww)
        percs = np.interp(percs, weighted_quantiles, values)
        return percs

    ww = np.moveaxis(ww, axis, -1)
    values = np.moveaxis(values, axis, -1)

    weighted_quantiles = np.cumsum(ww, axis=-1) - 0.5 * ww
    weighted_quantiles /= np.sum(ww, axis=-1)[..., np.newaxis]
    percs = [np.interp(percs, weighted_quantiles[idx], values[idx])
             for idx in np.ndindex(values.shape[:-1])]
    percs = np.array(percs)
    return percs


def random_power(extr, pdf_index, size=1):
    """Draw from power-law PDF with the given extrema and index.

    FIX/BUG : negative `extr` values break `pdf_index=-1` !!

    Arguments
    ---------
    extr : array_like scalar
        The minimum and maximum value of this array are used as extrema.
    pdf_index : scalar
        The power-law index of the PDF distribution to be drawn from.  Any real number is valid,
        positive or negative.
        NOTE: the `numpy.random.power` function uses the power-law index of the CDF, i.e. `g+1`
    size : scalar
        The number of points to draw (cast to int).
    **kwags : dict pairs
        Additional arguments passed to `zcode.math_core.minmax` with `extr`.

    Returns
    -------
    rv : (N,) scalar
        Array of random variables with N=`size` (default, size=1).

    """

    extr = minmax(extr)
    if pdf_index == -1:
        rv = 10**np.random.uniform(*np.log10(extr), size=int(size))
    else:
        rr = np.random.random(size=int(size))
        gex = extr ** (pdf_index+1)
        rv = (gex[0] + (gex[1] - gex[0])*rr) ** (1./(pdf_index+1))

    return rv


def rk4_step(func, x0, y0, dx, args=None, check_nan=0, check_nan_max=5):
    """Perform a single 4th-order Runge-Kutta integration step.
    """
    if args is None:
        k1 = dx * func(x0, y0)
        k2 = dx * func(x0 + dx/2.0, y0 + k1/2.0)
        k3 = dx * func(x0 + dx/2.0, y0 + k2/2.0)
        k4 = dx * func(x0 + dx, y0 + k3)
    else:
        k1 = dx * func(x0, y0, *args)
        k2 = dx * func(x0 + dx/2.0, y0 + k1/2.0, *args)
        k3 = dx * func(x0 + dx/2.0, y0 + k2/2.0, *args)
        k4 = dx * func(x0 + dx, y0 + k3, *args)

    y1 = y0 + (1.0/6.0) * (k1 + 2*k2 + 2*k3 + k4)
    x1 = x0 + dx

    # Try recursively decreasing step-size until finite-value is reached
    if check_nan > 0 and not np.isfinite(y1):
        if check_nan > check_nan_max:
            err = "Failed to find finite step!  `check_nan` = {}!".format(check_nan)
            raise RuntimeError(err)
        # Note that `True+1 = 2`
        rk4_step(func, x0, y0, dx/2.0, check_nan=check_nan+1, check_nan_max=check_nan_max)

    return x1, y1


def stats(vals: npt.ArrayLike, percs: Optional[npt.ArrayLike] = None, prec: int = 2, weights=None) -> str:
    """Return a string giving quantiles of the given input data.

    Parameters
    ----------
    vals : npt.ArrayLike,
        Input values to get quantiles of.
    percs : npt.ArrayLike, optional
        Quantiles to calculate.
    prec : int, optional
        Precision in scientific notation of output.

    Returns
    -------
    rv : str
        Quantiles of input formatted as a string of scientific notation values.

    Raises
    ------
    TypeError: raised if input data is not iterable.

    """
    try:
        if len(vals) == 0:        # type: ignore
            raise TypeError
    except TypeError:
        raise TypeError(f"`vals` (shape={np.shape(vals)}) is not iterable!")

    if percs is None:
        percs = [sp.stats.norm.cdf(1), 0.95, 1.0]
        percs = np.array(percs)
        percs = np.concatenate([1-percs[::-1], [0.5], percs])

    # stats = np.percentile(vals, percs*100)
    stats = quantiles(vals, percs, weights=weights)
    _rv = ["{val:.{prec}e}".format(prec=prec, val=ss) for ss in stats]
    rv = ", ".join(_rv)
    return rv


def std(vals, weights):
    """Weighted standard deviation (stdev).

    See: https://www.itl.nist.gov/div898/software/dataplot/refman2/ch2/weightsd.pdf
    """
    mm = np.count_nonzero(weights)
    ave = np.sum(vals*weights) / np.sum(weights)
    num = np.sum(weights * (vals - ave)**2)
    den = np.sum(weights) * (mm - 1) / mm
    std = np.sqrt(num/den)
    return std


def trapz(yy: npt.ArrayLike, xx: npt.ArrayLike, axis: int = -1, cumsum: bool = True):
    """Perform a cumulative integration along the given axis.

    Parameters
    ----------
    yy : ArrayLike of scalar,
        Input to be integrated.
    xx : ArrayLike of scalar,
        The sample points corresponding to the `yy` values.
        This must either be shaped as
        * the same number of dimensions as `yy`, with the same length along the `axis` dimension, or
        * 1D with length matching `yy[axis]`
    axis : int,
        The axis over which to integrate.

    Returns
    -------
    ct : ndarray of scalar,
        Cumulative trapezoid rule integration.

    """
    xx = np.asarray(xx)
    if np.ndim(xx) == 1:
        pass
    elif np.ndim(xx) == np.ndim(yy):
        xx = xx[axis]
    else:
        err = f"Bad shape for `xx` (xx.shape={np.shape(xx)}, yy.shape={np.shape(yy)})!"
        log.error(err)
        raise ValueError(err)
    ct = np.moveaxis(yy, axis, 0)   # type: ignore
    ct = 0.5 * (ct[1:] + ct[:-1])
    ct = np.moveaxis(ct, 0, -1)
    ct = ct * np.diff(xx)
    if cumsum:
        ct = np.cumsum(ct, axis=-1)
    ct = np.moveaxis(ct, -1, axis)
    return ct


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

    Parameters
    ----------
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
        assert np.all(xx[ii] == bounds), "FAILED!"   # type: ignore

    if np.ndim(yy) != np.ndim(xx):
        if np.ndim(xx) != 1:
            raise ValueError("BAD SHAPES")
        # convert `xx` from shape (N,) to (1, ... N, ..., 1) where all
        # dimensions besides `axis` have length one
        cut = [np.newaxis for ii in range(np.ndim(yy))]
        cut[axis] = slice(None)
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


def _parse_log_norm_pars(vals, size, default=None):
    """Parse/Sanitize the parameters for a log-normal distribution.

    * ()   ==> (N,)
    * (2,) ==> (N,) log_normal(vals)
    * (N,) ==> (N,)

    BUG: this function should probably be deprecated / removed !

    Parameters
    ----------
    vals : object,
        Input can be a single value, (2,) array_like, of array_like of size `size`:
            * scalar : this value is broadcast to an ndarray of size `size`
            * (2,) array_like : these two arguments are passed to `log_normal_base_10` and `size` samples
              are drawn
            * (N,) array_like : if `N` matches `size`, these values are returned.

    Returns
    -------
    vals : ndarray
        Returned values.

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


def _parse_val_log10_val_pars(val, val_log10, val_units=1.0, name='value', only_one=True):
    """Given either a parameter value, or the log10 of the value, ensure that both are set.

    Parameters
    ----------
    val : array_like or None,
        The parameter value itself in the desired units (specified by `val_units`).
    val_log10 : array_like or None,
        The log10 of the parameter value in natural units.
    val_units : array_like,
        The conversion factor from natural units (used in `val_log10`) to the desired units (used in `val`).
    name : str,
        The name of the variable for use in error messages.
    only_one : bool,
        Whether one, and only one, of `val` and `val_log10` should be provided (i.e. not `None`).

    Returns
    -------
    val : array_like,
        The parameter value itself in desired units.
        e.g. mass in grams, s.t. mass = Msol * 10^{mass_log10}
    val_log10 : array_like,
        The log10 of the parameter value in natural units.
        e.g. log10 of mass in solar-masses, s.t. mass = Msol * 10^{mass_log10}

    """
    both_or_neither = ((val_log10 is not None) == (val is not None))
    if only_one and both_or_neither:
        err = f"One of {name} OR {name}_log10 must be provided!  {name}={val}, {name}_log10={val_log10}"
        log.exception(err)
        raise ValueError(err)

    if val is None:
        val = val_units * np.power(10.0, val_log10)

    if val_log10 is None:
        val_log10 = np.log10(val / val_units)

    return val, val_log10


def _integrate_grid_differential_number(edges, dnum, freq=False):
    """Integrate the differential number-density of binaries over the given grid (edges).

    NOTE: the `edges` provided MUST all be in linear space, mass is converted to ``log10(M)``
    and frequency is converted to ``ln(f)``.
    NOTE: the density `dnum` MUST correspond to `dn/ [dlog10(M) dq dz dln(f)]`

    Parameters
    ----------
    edges : (4,) iterable of ArrayLike
    dnum : ndarray
    freq : bool
        Whether or not to also integrate the frequency dimension.

    Returns
    -------
    number : ndarray
        Number of binaries in each bin of mass, mass-ratio, redshift, frequency.
        NOTE: if `freq=False`, then `number` corresponds to `dN/dln(f)`, the number of binaries
        per log-interval of frequency.

    """
    # ---- integrate from differential-number to number per bin
    # integrate over dlog10(M)
    number = trapz(dnum, np.log10(edges[0]), axis=0, cumsum=False)
    # integrate over mass-ratio
    number = trapz(number, edges[1], axis=1, cumsum=False)
    # integrate over redshift
    number = trapz(number, edges[2], axis=2, cumsum=False)
    # integrate over frequency (if desired)
    if freq:
        number = trapz(number, np.log(edges[3]), axis=3, cumsum=False)

    return number


# =================================================================================================
# ====    Fitting Functions    ====
# =================================================================================================


def _func_gaussian(xx, aa, mm, ss):
    yy = aa * np.exp(-(xx - mm)**2 / (2.0 * ss**2))
    return yy


def fit_gaussian(xx, yy, guess=None):
    """Fit a Gaussian/Normal distribution with the given initial guess of parameters.

    Arguments
    ---------
    xx : array, (N,)
    yy : array, (N,)
    guess : None or (3,) array of float
        Initial parameter values as starting point of fit.  The values correspond to:
        [amplitude, mean, stdev].
        If ``guess`` is None, then the maximum, mean, and stdev of the given values are used as a
        starting point.

    Return
    ------
    popt : (3,) array of float
        Best fit parameters: [amplitude, mean, stdev]
    pcov : (3, 3) array of float
        Covariance matrix of best fit parameters.

    """
    if guess is None:
        amp = np.max(yy)
        mean = np.sum(xx * yy) / np.sum(yy)
        stdev = std(xx, yy)
        guess = [amp, mean, stdev]

    popt, pcov = sp.optimize.curve_fit(_func_gaussian, xx, yy, p0=guess, maxfev=10000)
    return popt, pcov


def _func_line(xx, amp, slope):
    yy = amp + slope * xx
    return yy


def fit_powerlaw(xx, yy, init=[-15.0, -2.0/3.0]):
    """Fit the given data with a power-law.

    Returns
    -------
    log10_amp
    plaw

    """

    popt, pcov = sp.optimize.curve_fit(_func_line, np.log10(xx), np.log10(yy), p0=init, maxfev=10000)
    # log10_amp = popt[0]
    # gamma = popt[1]

    def fit_func(xx, log10_amp, gamma):
        yy = _func_line(np.log10(xx), log10_amp, gamma)
        yy = 10.0 ** yy
        return yy

    return popt, fit_func


def _func_powerlaw_psd(freqs, fref, amp, index):
    aa = (amp**2) / (12.0 * np.pi**2)
    yy = aa * np.power(freqs/fref, index) * np.power(fref, -3)
    return yy


def fit_powerlaw_psd(xx, yy, fref, init=[-15.0, -13.0/3.0]):
    def fit_func(xx, log10_amp, index):
        amp = 10.0 ** log10_amp
        yy = _func_powerlaw_psd(xx, fref, amp, index)
        return np.log10(yy)

    popt, pcov = sp.optimize.curve_fit(
        fit_func, xx, np.log10(yy),
        p0=init, maxfev=10000, full_output=False
    )

    def fit_func(xx, log10_amp, index):
        amp = 10.0 ** log10_amp
        yy = _func_powerlaw_psd(xx, fref, amp, index)
        return yy

    return popt, fit_func


def fit_powerlaw_fixed_index(xx, yy, index=-2.0/3.0, init=[-15.0]):
    """

    Returns
    -------
    log10_amp
    plaw

    """
    _func_fixed = lambda xx, amp: _func_line(xx, amp, index)
    popt, pcov = sp.optimize.curve_fit(_func_fixed, np.log10(xx), np.log10(yy), p0=init, maxfev=10000)
    log10_amp = popt[0]
    return log10_amp


'''
def _func_turnover_hc(freqs, fref, amp, gamma, fbreak, kappa):
    alpha = (3.0 + gamma) / 2.0
    bend = np.power(fbreak/freqs, kappa)
    yy = amp * np.power(freqs/fref, alpha) * np.power(1.0 + bend, -0.5)
    return yy


def _func_turnover_loglog_hc(xx, amp, gamma, fbreak, kappa):
    alpha = (3.0 + gamma) / 2.0
    uu = np.power(10.0, xx*kappa)
    bb = np.power(fbreak, kappa)
    yy = amp + alpha*xx - 0.5 * np.log10(1.0 + bb/uu)
    return yy


def fit_turnover_hc(xx, yy, init=[-16.0, -13/3, 0.3, 2.5]):
    """
    """
    popt, pcov = sp.optimize.curve_fit(
        _func_turnover_loglog_hc, np.log10(xx), np.log10(yy),
        p0=init, maxfev=10000, full_output=False
    )
    return popt
'''


def _func_turnover_psd(freqs, fref, amp, gamma, fbreak, kappa):
    bend = np.power(fbreak/freqs, kappa)
    bend = np.power(1.0 + bend, -1.0)
    aa = (amp**2) / (12 * np.pi**2)
    yy = aa * np.power(freqs/fref, gamma) * bend * np.power(fref, -3)
    return yy


def fit_turnover_psd(xx, yy, fref, init=[-16, -13/3, 0.3/YR, 2.5]):
    """

    Parameters
    ----------
    xx : (F,)
        Frequencies in units of reference-frequency (e.g. 1/yr)
    yy : (F,)
        GWB PSD

    """

    def fit_func(xx, log10_amp, *args):
        amp = 10.0 ** log10_amp
        yy = _func_turnover_psd(xx, fref, amp, *args)
        return np.log10(yy)

    popt, pcov = sp.optimize.curve_fit(
        fit_func, xx, np.log10(yy),
        p0=init, maxfev=10000, full_output=False
    )

    def fit_func(xx, log10_amp, *args):
        amp = 10.0 ** log10_amp
        yy = _func_turnover_psd(xx, fref, amp, *args)
        return yy

    return popt, fit_func


# =================================================================================================
# ====    General Astronomy    ====
# =================================================================================================


def dfdt_from_dadt(dadt, sepa, mtot=None, frst_orb=None):
    """Convert from hardening rate in separation to hardening rate in frequency.

    Parameters
    ----------
    dadt : array_like
        Hardening rate in terms of binary separation.
    sepa : array_like
        Binary separations.
    mtot : None or array_like
        Binary total-mass in units of [gram].
        Either `mtot` or `frst_orb` must be provided.
    frst_orb : None or array_like
        Binary rest-frame orbital-frequency in units of [1/sec].
        Either `mtot` or `frst_orb` must be provided.

    Returns
    -------
    dfdt :
        Hardening rate in terms of rest-frame frequency.  [1/sec^2]
        NOTE: Has the opposite sign as `dadt`.
    frst_orb :
        Orbital frequency, in the rest-frame.  [1/sec]

    """
    if (mtot is None) and (frst_orb is None):
        err = "Either `mtot` or `frst_orb` must be provided!"
        log.exception(err)
        raise ValueError(err)
    if frst_orb is None:
        frst_orb = kepler_freq_from_sepa(mtot, sepa)

    dfdt = - 1.5 * (frst_orb / sepa) * dadt
    return dfdt, frst_orb


def mtmr_from_m1m2(m1, m2=None):
    """Convert from primary and secondary masses into total-mass and mass-ratio.

    NOTE: it doesn't matter if `m1` or `m2` is the primary or secondary.

    Parameters
    ----------
    m1 : array_like,
        Mass.  If this is a single value, or a 1D array, it denotes the mass of one component of
        a binary.  It can also be shaped, (N,2) where the two elements are the two component masses.
    m2 : None or array_like,
        If array_like, it must match the shape of `m1`, and corresponds to the companion mass.

    Returns
    -------
    (2,N) ndarray
        Total mass and mass-ratio.  If the input values are floats, this is just shaped (2,).

    """
    if m2 is not None:
        masses = np.stack([m1, m2], axis=-1)
    else:
        assert np.shape(m1)[-1] == 2, "If only `m1` is given, last dimension must be 2!"
        masses = np.asarray(m1)

    mtot = masses.sum(axis=-1)
    mrat = masses.min(axis=-1) / masses.max(axis=-1)
    return np.array([mtot, mrat])


def m1m2_from_mtmr(mt: npt.ArrayLike, mr: npt.ArrayLike) -> npt.ArrayLike:
    """Convert from total-mass and mass-ratio to individual masses.

    Parameters
    ----------
    mt : array_like
        Total mass of the binary.
    mr : array_like
        Mass ratio of the binary.

    Returns
    -------
    (2,N) ndarray
        Primary and secondary masses respectively.
        0-primary (more massive component),
        1-secondary (less massive component)

    """
    mt = np.asarray(mt)
    mr = np.asarray(mr)
    m1 = mt/(1.0 + mr)
    m2 = mt - m1
    return np.array([m1, m2])


def frst_from_fobs(fobs, redz):
    """Calculate rest-frame frequency from observed frequency and redshift.

    Parameters
    ----------
    fobs : array_like
        Observer-frame frequencies.
    redz : array_like
        Redshifts.

    Returns
    -------
    fobs : array_like
        Rest-frame frequencies.

    """
    frst = fobs * (1.0 + redz)
    return frst


def fobs_from_frst(frst, redz):
    """Calculate observed frequency from rest-frame frequency and redshift.

    Parameters
    ----------
    frst : array_like
        Rest-frame frequencies.
    redz : array_like
        Redshifts.

    Returns
    -------
    fobs : array_like
        Observer-frame frequencies.

    """
    fobs = frst / (1.0 + redz)
    return fobs


def kepler_freq_from_sepa(mass, sepa):
    """Calculate binary orbital frequency using Kepler's law.

    Parameters
    ----------
    mass : array_like
        Binary total mass [grams].
    sepa : array_like
        Binary semi-major axis or separation [cm].

    Returns
    -------
    freq : array_like
        Binary orbital frequency [1/s].

    """
    freq = (1.0/(2.0*np.pi))*np.sqrt(NWTG*mass)/np.power(sepa, 1.5)
    return freq


def kepler_sepa_from_freq(mass, freq):
    """Calculate binary separation using Kepler's law.

    Parameters
    ----------
    mass : array_like
        Binary total mass [grams]
    freq : array_like
        Binary orbital frequency [1/s].

    Returns
    -------
    sepa : array_like
        Binary semi-major axis (i.e. separation) [cm].

    """
    mass = np.asarray(mass)
    freq = np.asarray(freq)
    sepa = np.power(NWTG*mass/np.square(2.0*np.pi*freq), 1.0/3.0)
    return sepa


def rad_isco(m1, m2=0.0, factor=3.0):
    """Inner-most Stable Circular Orbit, radius at which binaries 'merge'.

    ENH: allow single (total) mass argument.
    ENH: add function to calculate factor as a function of BH spin.

    Parameters
    ----------
    m1 : array_like,
        Mass of first (either) component of binary [grams].
    m2 : array_like,
        Mass of second (other) component of binary [grams].
    factor : float,
        Factor by which to multiple the Schwarzschild radius to define the ISCO.
        3.0 for a non-spinning black-hole.

    Returns
    -------
    rs : array_like,
        Radius of the inner-most stable circular orbit [cm].

    """
    return factor * schwarzschild_radius(m1+m2)


def frst_isco(m1, m2=0.0, **kwargs):
    """Get rest-frame orbital frequency of ISCO orbit.

    Arguments
    ---------
    m1 : array_like, units of [gram]
        Total mass, or mass of the primary.  Added together with `m2` to get total mass.
    m2 : array_like, units of [gram]  or  None
        Mass of secondary, or None if `m1` is already total mass.

    Returns
    -------
    fisco : array_like, units of [Hz]

    """
    risco = rad_isco(m1, m2, **kwargs)
    fisco = kepler_freq_from_sepa(m1+m2, risco)
    return fisco


def redz_after(time, redz=None, age=None):
    """Calculate the redshift after the given amount of time has passed.

    Parameters
    ----------
    time : array_like in units of [sec]
        Amount of time to pass.
    redz : None or array_like,
        Redshift of starting point after which `time` is added.
    age : None or array_like, in units of [sec]
        Age of the Universe at the starting point, after which `time` is added.

    Returns
    -------
    new_redz : array_like
        Redshift of the Universe after the given amount of time.

    """
    if (redz is None) == (age is None):
        raise ValueError("One of `redz` and `age` must be provided (and not both)!")

    if redz is not None:
        age = cosmo.age(redz).to('s').value
    new_age = age + time

    if np.isscalar(new_age):
        if new_age < _AGE_UNIVERSE_GYR * GYR:
            new_redz = cosmo.tage_to_z(new_age)
        else:
            new_redz = -1.0

    else:
        new_redz = -1.0 * np.ones_like(new_age)
        idx = (new_age < _AGE_UNIVERSE_GYR * GYR)
        new_redz[idx] = cosmo.tage_to_z(new_age[idx])

    return new_redz


def schwarzschild_radius(mass):
    """Return the Schwarschild radius [cm] for the given mass [grams].

    Parameters
    ----------
    m1 : array_like
        Mass [grams]

    Returns
    -------
    rs : array_like,
        Schwzrschild radius for this mass.

    """
    rs = SCHW * mass
    return rs


def velocity_orbital(mt, mr, per=None, sepa=None):
    sepa, per = _get_sepa_freq(mt, sepa, per)
    v2 = np.power(NWTG*mt/sepa, 1.0/2.0) / (1 + mr)
    # v2 = np.power(2*np.pi*NWTG*mt/per, 1.0/3.0) / (1 + mr)
    v1 = v2 * mr
    vels = np.moveaxis([v1, v2], 0, -1)
    return vels


def _get_sepa_freq(mt, sepa, freq):
    if (freq is None) and (sepa is None):
        raise ValueError("Either `freq` or `sepa` must be provided!")
    if (freq is not None) and (sepa is not None):
        raise ValueError("Only one of `freq` or `sepa` should be provided!")

    if freq is None:
        freq = kepler_freq_from_sepa(mt, sepa)

    if sepa is None:
        sepa = kepler_sepa_from_freq(mt, freq)

    return sepa, freq


def lambda_factor_dlnf(frst, dfdt, redz, dcom=None):
    """Account for the universe's differential space-time volume for a given hardening rate.

    For each binary, calculate the factor: $$\\Lambda \\equiv (dVc/dz) * (dz/dt) * [dt/dln(f)]$$,
    which has units of [Mpc^3].  When multiplied by a number-density [Mpc^-3], it gives the number
    of binaries in the Universe *per log-frequency interval*.  This value must still be multiplied
    by $\\Delta \\ln(f)$ to get a number of binaries across a frequency in.

    Parameters
    ----------
    frst : ArrayLike
        Binary frequency (typically rest-frame orbital frequency; but it just needs to match what's
        provided in the `dfdt` term.  Units of [1/sec].
    dfdt : ArrayLike
        Binary hardening rate in terms of frequency (typically rest-frame orbital frequency, but it
        just needs to match what's provided in `frst`).  Units of [1/sec^2].
    redz : ArrayLike
        Binary redshift.  Dimensionless.
    dcom : ArrayLike
        Comoving distance to binaries (for the corresponding redshift, `redz`).  Units of [cm].
        If not provided, calculated from given `redz`.

    Returns
    -------
    lambda_fact : ArrayLike
        The differential comoving volume of the universe per log interval of binary frequency.

    """
    zp1 = redz + 1
    if dcom is None:
        dcom = cosmo.z_to_dcom(redz)

    # Volume-factor
    # this is `(dVc/dz) * (dz/dt)`,  units of [Mpc^3/s]
    vfac = 4.0 * np.pi * SPLC * zp1 * (dcom**2)
    # Time-factor
    # this is `f / (df/dt) = dt/d ln(f)`,  units of [sec]
    tfac = frst / dfdt

    # Calculate weighting
    lambda_fact = vfac * tfac
    return lambda_fact


def angs_from_sepa(sepa, dcom, redz):
    """ Calculate angular separation

    Parameters
    ----------
    sepa : ArrayLike
        Binary separation, in cm
    dcom : ArrayLike
        Binary comoving distance, in cm
    redz : ArrayLike
        Binary redshift

    Returns
    -------
    angs : ArrayLike
        Angular separation

    """
    dang = dcom / (1.0 + redz)   # angular-diameter distance [cm]
    angs = sepa / dang           # angular-separation [radians]
    return angs


def eddington_accretion(mass, eps=0.1):
    """Eddington Accretion rate, $\\dot{M}_{Edd} = L_{Edd}/\\epsilon c^2$.

    Arguments
    ---------
    mass : array_like of scalar
        BH Mass.
    eps : array_like of scalar
        Efficiency parameter.

    Returns
    -------
    mdot : array_like of scalar
        Eddington accretion rate.

    """
    edd_lum = eddington_luminosity(mass)
    mdot = edd_lum / eps / np.square(SPLC)
    return mdot


def eddington_luminosity(mass):
    ledd = EDDT * mass
    return ledd


# =================================================================================================
# ====    Gravitational Waves    ====
# =================================================================================================


def chirp_mass(m1, m2=None):
    """Calculate the chirp-mass of a binary.

    Parameters
    ----------
    m1 : array_like,
        Mass [grams]
        This can either be the mass of the primary component, if scalar or 1D array_like,
        or the mass of both components, if 2D array_like, shaped (N, 2).
    m2 : None or array_like,
        Mass [grams] of the other component of the binary.  If given, the shape must be
        broadcastable against `m1`.

    Returns
    -------
    mc : array_like,
        Chirp mass [grams] of the binary.

    """
    m1, m2 = _array_args(m1, m2)
    # (N, 2)  ==>  (N,), (N,)
    if m2 is None:
        m1, m2 = np.moveaxis(m1, -1, 0)
    mc = np.power(m1 * m2, 3.0/5.0)/np.power(m1 + m2, 1.0/5.0)
    return mc


def chirp_mass_mtmr(mt, mr):
    """Calculate the chirp-mass of a binary.

    Parameters
    ----------
    mt : array_like,
        Total mass [grams].  This is ``M = m1+m2``.
    mr : array_like,
        Mass ratio.  ``q = m2/m1 <= 1``.
        This is defined as the secondary (smaller) divided by the primary (larger) mass.

    Returns
    -------
    mc : array_like,
        Chirp mass [grams] of the binary.

    """
    mt, mr = _array_args(mt, mr)
    mc = mt * np.power(mr, 3.0/5.0) / np.power(1 + mr, 6.0/5.0)
    return mc


def gw_char_strain_nyquist(dur_obs, hs, frst_orb, redz, dfdt_rst):
    """GW Characteristic Strain assuming frequency bins are Nyquist sampled.

    Nyquist assumption: the bin-width is equal to 1/T, for T the total observing duration.

    See, e.g., [Sesana2004]_, Eq.35, and surrounding text.
    NOTE: make sure this is the correct definition of "characteristic" strain for your application!

    # ! THIS FUNCTION MAY NOT BE CORRECT [LZK:2022-08-25] ! #

    Parameters
    ----------
    dur_obs : array_like,
        Duration of observations, in the observer frame, in units of [sec].
        Typically this is a single float value.
    hs : array_like,
        Strain amplitude of the source.  Dimensionless.
    frst_orb : array_like,
        Observer-frame orbital frequency, units of [1/sec].
    redz : array_like,
        Redshift of the binary.  Dimensionless.
    dfdt_rst : array_like,
        Rate of orbital-frequency evolution of the binary, in the rest-frame.  Units of [1/sec^2].

    Returns
    -------
    hc : array_like,
        Characteristic strain of the binary.

    """
    log.warning("`holodeck.utils.gw_char_strain_nyquist` may not be correct!", exc_info=True)

    fobs_gw = fobs_from_frst(frst_orb, redz) * 2.0
    # Calculate the time each binary spends within the band
    dfdt_obs = dfdt_rst * (1 + redz)**2
    bandwidth = (1.0 / dur_obs)   # I think this is right
    # bandwidth = fobs_gw           # I think this is wrong
    tband = bandwidth / dfdt_obs

    ncycles = fobs_gw * np.minimum(dur_obs, tband)
    hc = hs * np.sqrt(ncycles)
    return hc


def gw_dedt(m1, m2, sepa, eccen):
    """GW Eccentricity Evolution rate (de/dt) due to GW emission.

    NOTE: returned value is negative.

    See [Peters1964]_, Eq. 5.8

    Parameters
    ----------
    m1 : array_like,
        Mass of one component of the binary [grams].
    m2 : array_like,
        Mass of other component of the binary [grams].
    sepa : array_like,
        Binary semi-major axis (i.e. separation) [cm].
    eccen : array_like,
        Binary orbital eccentricity.

    Returns
    -------
    dedt : array_like
        Rate of eccentricity change of the binary.
        NOTE: returned value is negative or zero.

    """
    m1, m2, sepa, eccen = _array_args(m1, m2, sepa, eccen)
    cc = _GW_DEDT_ECC_CONST
    e2 = eccen**2
    dedt = cc * m1 * m2 * (m1 + m2) / np.power(sepa, 4)
    dedt *= (1.0 + e2*121.0/304.0) * eccen / np.power(1 - e2, 5.0/2.0)
    return dedt


def gw_dade(sepa, eccen):
    """Rate of semi-major axis evolution versus eccentricity, due to GW emission (da/de).

    NOTE: returned value is positive (e and a go in same direction).
    See [Peters1964]_, Eq. 5.7

    Parameters
    ----------
    sepa : array_like
        Binary semi-major axis (separation) [grams].
    eccen : array_like
        Binary eccentricity [grams].

    Returns
    -------
    dade : array_like
        Rate of change of semi-major axis versus eccentricity [cm].
        NOTE: returned value is positive.

    """
    sepa, eccen = _array_args(sepa, eccen)
    e2 = eccen**2
    num = (1 + (73.0/24.0)*e2 + (37.0/96.0)*e2*e2)
    den = (1 - e2) * (1.0 + (121.0/304.0)*e2)
    dade = (12.0 / 19.0) * (sepa / eccen) * (num / den)
    return dade


def gw_freq_dist_func(nn, ee=0.0, recursive=True):
    """GW frequency distribution function.

    See [EN2007]_ Eq. 2.4; this function gives g(n,e).

    NOTE: recursive relation fails for zero eccentricities!
    TODO: could choose to use non-recursive when zero eccentricities are found?

    TODO: replace `ee` variable with `eccen`

    Parameters
    ----------
    nn : int,
        Number of frequency harmonic to calculate.
    ee : array_like,
        Binary eccentricity.

    Returns
    -------
    gg : array_like,
        GW Frequency distribution function g(n,e).

    """

    # Calculate with non-zero eccentrictiy
    bessel = sp.special.jn
    ne = nn*ee
    n2 = np.square(nn)
    jn_m2 = bessel(nn-2, ne)
    jn_m1 = bessel(nn-1, ne)

    # Use recursion relation:
    if recursive:
        jn = (2*(nn-1) / ne) * jn_m1 - jn_m2
        jn_p1 = (2*nn / ne) * jn - jn_m1
        jn_p2 = (2*(nn+1) / ne) * jn_p1 - jn
    else:
        jn = bessel(nn, ne)
        jn_p1 = bessel(nn+1, ne)
        jn_p2 = bessel(nn+2, ne)

    aa = np.square(jn_m2 - 2.0*ee*jn_m1 + (2/nn)*jn + 2*ee*jn_p1 - jn_p2)
    bb = (1 - ee*ee)*np.square(jn_m2 - 2*ee*jn + jn_p2)
    cc = (4.0/(3.0*n2)) * np.square(jn)
    gg = (n2*n2/32) * (aa + bb + cc)
    return gg


def gw_hardening_rate_dadt(m1, m2, sepa, eccen=None):
    """GW Hardening rate in separation (da/dt).

    NOTE: returned value is negative.

    See [Peters1964]_, Eq. 5.6

    Parameters
    ----------
    m1 : array_like,
        Mass of one component of the binary [grams].
    m2 : array_like,
        Mass of other component of the binary [grams].
    sepa : array_like,
        Binary semi-major axis (i.e. separation) [cm].
    eccen : None or array_like,
        Binary orbital eccentricity.  Treated as zero if `None`.

    Returns
    -------
    dadt : array_like,
        Binary hardening rate [cm/s] due to GW emission.

    """
    m1, m2, sepa, eccen = _array_args(m1, m2, sepa, eccen)
    cc = _GW_DADT_SEP_CONST
    dadt = cc * m1 * m2 * (m1 + m2) / np.power(sepa, 3)
    if eccen is not None:
        fe = _gw_ecc_func(eccen)
        dadt *= fe
    return dadt


def gw_hardening_rate_dfdt(m1, m2, frst_orb, eccen=None):
    """GW Hardening rate in frequency (df/dt).

    Parameters
    ----------
    m1 : array_like
        Mass of one component of each binary [grams].
    m2 : array_like
        Mass of other component of each binary [grams].
    freq_orb : array_like
        Rest frame orbital frequency of each binary [1/s].
    eccen : array_like, optional
        Eccentricity of each binary.

    Returns
    -------
    dfdt : array_like,
        Hardening rate in terms of frequency for each binary [1/s^2].

    """
    m1, m2, frst_orb, eccen = _array_args(m1, m2, frst_orb, eccen)
    sepa = kepler_sepa_from_freq(m1+m2, frst_orb)
    dfdt = gw_hardening_rate_dadt(m1, m2, sepa, eccen=eccen)
    # dfdt, _ = dfdt_from_dadt(dfdt, sepa, mtot=m1+m2)
    dfdt, _ = dfdt_from_dadt(dfdt, sepa, frst_orb=frst_orb)
    return dfdt, frst_orb


def gw_hardening_timescale_freq(mchirp, frst):
    """GW Hardening timescale in terms of frequency (not separation).

    ``tau = f_r / (df_r / dt)``, e.g. [EN2007]_ Eq.2.9

    Parameters
    ----------
    mchirp : array_like,
        Chirp mass in [grams]
    frst : array_like,
        Rest-frame orbital frequency [1/s].

    Returns
    -------
    tau : array_like,
        GW hardening timescale defined w.r.t. orbital frequency [sec].

    """
    mchirp, frst = _array_args(mchirp, frst)
    tau = (5.0 / 96.0) * np.power(NWTG*mchirp/SPLC**3, -5.0/3.0) * np.power(2*np.pi*frst, -8.0/3.0)
    return tau


def gw_lum_circ(mchirp, freq_orb_rest):
    """Calculate the GW luminosity of a circular binary.

    [EN2007]_ Eq. 2.2

    Parameters
    ----------
    mchirp : array_like,
        Binary chirp mass [grams].
    freq_orb_rest : array_like,
        Rest-frame binary orbital frequency [1/s].

    Returns
    -------
    lgw_circ : array_like,
        GW Luminosity [erg/s].

    """
    mchirp, freq_orb_rest = _array_args(mchirp, freq_orb_rest)
    lgw_circ = _GW_LUM_CONST * np.power(2.0*np.pi*freq_orb_rest*mchirp, 10.0/3.0)
    return lgw_circ


def gw_strain_source(mchirp, dcom, freq_rest_orb):
    """GW Strain from a single source in a circular orbit.

    For reference, see:
    *   [Sesana2004]_ Eq.36 : they use `f_r` to denote rest-frame GW-frequency.
    *   [Enoki2004]_ Eq.5.

    Parameters
    ----------
    mchirp : array_like,
        Binary chirp mass [grams].
    dcom : array_like,
        Comoving distance to source [cm].
    freq_orb_rest : array_like,
        Rest-frame binary orbital frequency [1/s].

    Returns
    -------
    hs : array_like,
        GW Strain (*not* characteristic strain).

    """
    mchirp, dcom, freq_rest_orb = _array_args(mchirp, dcom, freq_rest_orb)
    # The factor of 2 below is to convert from orbital-frequency to GW-frequency
    hs = _GW_SRC_CONST * mchirp * np.power(2*mchirp*freq_rest_orb, 2/3) / dcom
    return hs


def sep_to_merge_in_time(m1, m2, time):
    """The initial separation required to merge within the given time.

    See: [Peters1964]_

    Parameters
    ----------
    m1 : array_like,
        Mass of one component of the binary [grams].
    m2 : array_like,
        Mass of other component of the binary [grams].
    time : array_like,
        The duration of time of interest [sec].

    Returns
    -------
    array_like
        Initial binary separation [cm].

    """
    m1, m2, time = _array_args(m1, m2, time)
    GW_CONST = 64*np.power(NWTG, 3.0)/(5.0*np.power(SPLC, 5.0))
    a1 = rad_isco(m1, m2)
    return np.power(GW_CONST*m1*m2*(m1+m2)*time - np.power(a1, 4.0), 1./4.)


def time_to_merge_at_sep(m1, m2, sepa):
    """The time required to merge starting from the given initial separation.

    See: [Peters1964]_.

    Parameters
    ----------
    m1 : array_like,
        Mass of one component of the binary [grams].
    m2 : array_like,
        Mass of other component of the binary [grams].
    sepa : array_like,
        Binary semi-major axis (i.e. separation) [cm].

    Returns
    -------
    array_like
        Duration of time for binary to coalesce [sec].

    """
    m1, m2, sepa = _array_args(m1, m2, sepa)
    GW_CONST = 64*np.power(NWTG, 3.0)/(5.0*np.power(SPLC, 5.0))
    a1 = rad_isco(m1, m2)
    delta_sep = np.power(sepa, 4.0) - np.power(a1, 4.0)
    return delta_sep/(GW_CONST*m1*m2*(m1+m2))


def gamma_psd_to_strain(gamma_psd):
    gamma_strain = (gamma_psd + 3.0) / 2.0
    return gamma_strain


def gamma_strain_to_psd(gamma_strain):
    gamma_psd = 2*gamma_strain - 3.0
    return gamma_psd


def gamma_strain_to_omega(gamma_strain):
    gamma_omega = (gamma_strain - 2.0) / 2.0
    return gamma_omega


def char_strain_to_psd(freqs, hc):
    """

    Arguments
    ---------
    freqs : array_like
        Frequencies of interest in [1/sec].
        Note: these should NOT be in units of reference frequency, but in units of [Hz] = [1/sec].
    hc : array_like
        Characteristic strain.

    Returns
    -------
    psd : array_like
        Power spectral density of gravitational waves.


    """
    psd = hc**2 / (12*np.pi**2)
    psd = psd * np.power(freqs, -3)
    # psd = psd * np.power(freqs/fref, -3) * np.power(fref, -3)
    return psd


def psd_to_char_strain(freqs, psd):
    hc = np.sqrt(psd * (12*np.pi**2 * freqs**3))
    return hc


def char_strain_to_rho(freqs, hc, tspan):
    psd = char_strain_to_psd(freqs, hc)
    rho = np.sqrt(psd/tspan)
    return rho


def rho_to_char_strain(freqs, rho, tspan):
    psd = tspan * rho**2
    hc = psd_to_char_strain(freqs, psd)
    return hc


def char_strain_to_strain_amp(hc, fc, df):
    """ Calculate the strain amplitude of single sources given
    their characteristic strains.

    Parameters
    ----------
    hc : array_like
        Characteristic strain of the single sources.
    fc : array_like
        Observed orbital frequency bin centers.
    df : array_like
        Observed orbital frequency bin widths.

    Returns
    -------
    hs : (F,R,L)
        Strain amplitude of the single sources.

    """
    hs = hc * np.sqrt(df/fc)
    return hs


@numba.njit
def _gw_ecc_func(eccen):
    """GW Hardening rate eccentricitiy dependence F(e).

    See [Peters1964]_ Eq. 5.6, or [EN2007]_ Eq. 2.3

    Parameters
    ----------
    eccen : array_like,
        Binary orbital eccentricity [].

    Returns
    -------
    fe : array_like
        Eccentricity-dependence term of GW emission [].

    """
    e2 = eccen*eccen
    num = 1 + (73/24)*e2 + (37/96)*e2*e2
    den = np.power(1 - e2, 7/2)
    fe = num / den
    return fe


def _array_args(*args):
    args = [np.asarray(aa) if aa is not None else None
            for aa in args]
    return args


@deprecated_fail(scatter_redistribute_densities)
def scatter_redistribute(cents, dist, dens, axis=0):
    pass


#! DEPRECATED
def nyquist_freqs(dur, cad):
    """DEPRECATED.  Use `holodeck.utils.pta_freqs` instead.
    """
    msg = ""
    old_name = "nyquist_freqs"
    new_name = "pta_freqs"
    _frame = inspect.currentframe().f_back
    file_name = inspect.getfile(_frame.f_code)
    fline = _frame.f_lineno
    msg = f"{file_name}({fline}):{old_name} ==> {new_name}" + (len(msg) > 0) * " | " + msg
    warnings.warn_explicit(msg, category=DeprecationWarning, filename=file_name, lineno=fline)
    log.warning(f"DEPRECATION: {msg}", exc_info=True)
    return pta_freqs(dur=dur, num=None, cad=cad)[0]


