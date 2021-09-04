"""Module for binary evolution from the time of formation/galaxy-merger until BH coalescence.

In `holodeck`, initial binary populations are typically defined near the time of galaxy-galaxy
merger, when two MBHs come together at roughly kiloparsec scales.  Environmental 'hardening'
mechanisms are required to dissipate orbital energy and angular momentum, allowing the binary
separation to shrink ('harden'). Typically *dynamical friction (DF)* is most important at large
scales ($\\sim \\mathrm{kpc}$).  Near where the pair of MBHs become a gravitationally-bound binary,
the DF approximations break down, and individual *stellar scattering* events (from stars in the
'loss cone' of parameter space) need to be considered.  In the presence of significant gas (i.e.
from accretion), a circumbinary accretion disk (CBD) may form, and gravitational
*circumbinary disk torques* from the gas distribution (typically spiral waves) may become important.
Finally, at the smaller separations, *GW emission* takes over.  The classic work describing the
overall process of binary evolution in its different stages is [BBR1980]_.  [Kelley2017a]_ goes
into significant detail on implementations and properties of each component of binary evolution
also.  Triple MBH interactions, and perturbations from other massive objects (e.g. molecular
clouds) can also be important.

The :mod:`holodeck.evolution` submodule provides tools for modeling the binary evolution from the
time of binary 'formation' (i.e. galaxy merger) until coalescence.  Models for binary evolutionary
can range tremendously in complexity.  In the simplest models, binaries are assumed to coalesce
instantaneously (in that the age of the universe is the same at formation and coalescence), and are
also assumed to evolve purely due to GW emission (in that the time spent in any range of orbital
frequencies can be calculated from the GW hardening timescale).  Note that these two assumptions
are contradictory, but commonly employed in the literature.  The ideal, self-consistent approach,
is to model binary evolution using self-consistently derived environments (e.g. gas and stellar
mass distributions), and applying the same time-evolution prescription to both the redshift
evolution of each binary, and also the GW calculation.  Note that GWs can only be calculated based
on some sort of model for binary evolution.  The model may be extremely simple, in which case it is
sometimes glanced over.

The core component of the evolution module is the :class:`Evolution` class.  This class combines a
population of initial binary parameters (i.e. from the :class:`holodeck.population.Pop_Illustris`
class), along with a specific binary hardening model (:class:`_Hardening` subclass), and performs
the numerical integration of each binary over time - from large separations to coalescence.  The
:class:`Evolution` class also acts to store the binary evolution histories ('trajectories' or
'tracks'), which are then used to calculate GW signals (e.g. the GWB).  To facilitate GW and
similar calculations, :class:`Evolution` also provides interpolation functionality along binary
trajectories.

To-Do
-----
*   General

    *   evolution modifiers should act at each step, instead of after all steps?  This would be
        a way to implement a changing accretion rate, for example; or set a max/min hardening rate.
    *   re-implement "magic" hardening models that coalesce in zero change-of-redshift or fixed
        amounts of time.

*   Dynamical_Friction_NFW

    *   Allow stellar-density profiles to also be specified (instead of using a hard-coded
        Dehnen profile)
    *   Generalize calculation of stellar characteristic radius.  Make self-consistent with
        stellar-profile, and user-specifiable.

*   Evolution

    *   `_sample_universe()` : sample in comoving-volume instead of redshift

*   Sesana_Scattering

    *   Allow stellar-density profile (or otherwise the binding-radius) to be user-specified
        and flexible.  Currently hard-coded to Dehnen profile estimate.

*   _SHM06

    *   Interpolants of hardening parameters return 2D arrays which we then take the diagonal
        of, but there should be a better way of doing this.

*   Fixed_Time

    *   Handle `rchar` better with respect to interpolation.  Currently not an interpolation
        variable, which restricts it's usage.
    *   This class should be separated into a generic `_Fixed_Time` class that can use any
        functional form, and then a 2-power-law functional form that requires a specified
        normalization.  When they're combined, it will produce the same effect.  Another good
        functional form to implement would be GW + log-uniform hardening time, the same as the
        current phenomenological model but with both power-laws set to 0.


References
----------
* [BBR1980]_ Begelman, Blandford & Rees 1980.
* [Chen2017]_ Chen, Sesana, & Del Pozzo 2017.
* [Kelley2017a]_ Kelley, Blecha & Hernquist 2017.
* [Quinlan1996]_ Quinlan 1996.
* [Sesana2006]_ Sesana, Haardt & Madau et al. 2006.
* [Sesana2010]_ Sesana 2010.

"""
from __future__ import annotations

# ! ===============================================================================================!
# ! -- UPDATE `_take_next_step` -- !
# ! -- write a function to set right-edge values, call it with estimate from left edge -- !
# ! -- then call it again with a revision to the estimate, using right-edge values -- !
# ! ===============================================================================================!

import abc
# import inspect
import json
import os
# from typing import Union, TypeVar  # , Callable, Iterator
import warnings

import numpy as np
import scipy as sp
import scipy.interpolate   # noqa

import kalepy as kale

import holodeck as holo
from holodeck import utils, cosmo, log, _PATH_DATA
from holodeck.constants import GYR, NWTG, PC, MSOL, YR
from holodeck import accretion

_MAX_ECCEN_ONE_MINUS = 1.0e-6
#: number of influence radii to set minimum radius for dens calculation
_MIN_DENS_RAD__INFL_RAD_MULT = 10.0
_DEF_TIME_DELAY = (5.0*GYR, 0.2)   #: default delay-time parameters, (mean, stdev)
_SCATTERING_DATA_FILENAME = "SHM06_scattering_experiments.json"

# Pop = TypeVar('Pop', bound=holo.population._Population_Discrete)  # Must be exactly str or bytes
# A = TypeVar('A', str, bytes)  # Must be exactly str or bytes
# S = TypeVar('S', bound=str)  # Can be any subtype of str
# Hard = TypeVar('Hard', bound=holo.evolution._Hardening)  # Can be any subtype of str
# Hard = TypeVar('Hard', holo.evolution._Hardening, list[holo.evolution._Hardening])  # Can be any subtype of str
# AliasType = Union[list[dict[tuple[int, str], set[int]]], tuple[str, list[str]]]
# Hard_list = Union[list[Hard], Hard]


# =================================================================================================
# ====    Evolution Class    ====
# =================================================================================================


class Evolution:
    """Base class to evolve discrete binary populations forward in time.

    NOTE: This class is primary built to be used with :class:`holodeck.population.Pop_Illustris`.

    The `Evolution` class is instantiated with a :class:`holodeck.population._Population_Discrete`
    instance, and a particular binary hardening model (subclass of :class:`_Hardening`).  It then
    numerically integrates each binary from their initial separation to coalescence, determining
    how much time that process takes, and thus the rate of redshift/time evolution.  NOTE: at
    initialization, a regular range of binary separations are chosen for each binary being evolved,
    and the integration calculates the time it takes for each step to complete.  This is unlike most
    types of dynamical integration in which there is a prescribed time-step, and the amount of
    distance (etc) traveled over that time is then calculated.

    **Initialization**: all attributes are set to empty arrays of the appropriate size.
    NOTE: the 0th step is *not* initialized at this time, it happens in :meth:`Evolution.evolve()`.

    **Evolution**: binary evolution is performed by running the :meth:`Evolution.evolve()` function.
    This function first calls :meth:`Evolution._init_step_zero()`, which sets the 0th step values,
    and then iterates over each subsequent step, calling :meth:`Evolution._take_next_step()`.  Once
    all steps are taken (integration is completed), then :meth:`Evolution._finalize()` is called,
    at which points any stored modifiers (:class:`utils._Modifier` subclasses, in the
    :attr:`Evolution._mods` attribute) are applied.

    NOTE: whenever `frequencies` are used (rest-frame or observer-frame), they refer to **orbital**
    frequencies, not GW frequencies.  For circular binaries, GW-freq = 2 * orb-freq.

    Additional Notes
    ----------------
    accmod: accretion model to use. current options: None (default), 'Basic', 'Proportional'

    """

    _EVO_PARS = ['mass', 'sepa', 'eccen', 'scafa', 'tlook', 'dadt', 'dedt']
    _LIN_INTERP_PARS = ['eccen', 'scafa', 'tlook', 'dadt', 'dedt']
    _SELF_CONSISTENT = None
    _STORE_FROM_POP = ['_sample_volume']

    def __init__(self, pop, hard, nsteps: int = 100, mods=None, debug: bool = False, accmod=None):
        """Initialize a new Evolution instance.

        Parameters
        ----------
        pop : `population._Population_Discrete` instance,
            Binary population with initial parameters of the binary from which to start evolution.
        hard : `_Hardening` instance, or list of
            Model for binary hardening used to evolve population's separation over time.
        nsteps : int,
            Number of steps between initial separations and coalescence for all binaries.
        mods : None, or list of `utils._Modifier` subclasses,
            NOTE: not fully implemented!
        debug : bool,
            Include verbose/debugging output information.

        """
        # --- Store basic parameters to instance
        self._pop = pop                       #: initial binary population instance
        self._debug = debug                   #: debug flag for performing extra diagnostics and output
        self._nsteps = nsteps                 #: number of integration steps for each binary
        self._mods = mods                     #: modifiers to be applied after evolution is completed
        self._accmod = accmod

        # Store hardening instances as a list
        if not np.iterable(hard):
            hard = [hard, ]
        self._hard = hard

        # Make sure types look right
        if not isinstance(pop, holo.population._Population_Discrete):
            err = f"`pop` is {pop}, must be subclass of `holo.population._Population_Discrete`!"
            log.exception(err)
            raise TypeError(err)

        for hh in self._hard:
            good = isinstance(hh, _Hardening) or issubclass(hh, _Hardening)
            if not good:
                err = f"hardening instance is {hh}, must be subclass of `{_Hardening}`!"
                log.exception(err)
                raise TypeError(err)

        # Store additional parameters
        for par in self._STORE_FROM_POP:
            setattr(self, par, getattr(pop, par))

        size = pop.size
        shape = (size, nsteps)
        self._shape = shape

        if pop.eccen is not None:
            eccen = np.zeros(shape)
            dedt = np.zeros(shape)
        else:
            eccen = None
            dedt = None

        # ---- Initialize empty arrays for tracking binary evolution
        self.scafa = np.zeros(shape)           #: scale-factor of the universe, set to 1.0 after z=0
        self.tlook = np.zeros(shape)           #: lookback time [sec], NOTE: negative after redshift zero
        self.sepa = np.zeros(shape)            #: semi-major axis (separation) [cm]
        self.mass = np.zeros(shape + (2,))     #: mass of BHs [g], 0-primary, 1-secondary
        self.mdot = np.zeros(shape + (2,))     #: accretion rate onto each component of binary [g/s]
        self.dadt = np.zeros(shape)            #: hardening rate in separation [cm/s]
        self.eccen = eccen                     #: eccentricity [], `None` if not being evolved
        self.dedt = dedt                       #: eccen evolution rate [1/s], `None` if not evolved

        # Derived and internal parameters
        self._freq_orb_rest = None
        self._evolved = False
        self._coal = None

        return

    # ==== API and Core Functions

    def evolve(self, progress=False):
        """Evolve binary population from initial separation until coalescence in discrete steps.

        Each binary has a fixed number of 'steps' from initial separation until coalescence.  The
        role of the `evolve()` method is to determine the amount of time each step takes, based on
        the 'hardening rate' (in separation and possible eccentricity i.e. $da/dt$ and $de/dt$).
        The hardening rate is calculated from the stored :class:`_Hardening` instances in the
        iterable :attr:`Evolution._hard` attribute.

        When :meth:`Evolution.evolve()` is called, the 0th step is initialized separately, using
        the :meth:`Evolution._init_step_zero()` method, and then each step is integrated by calling
        the :meth:`Evolution._take_next_step()` method.  Once all steps are completed, the
        :meth:`Evolution._finalize()` method is called, where any stored modifiers are applied.

        Parameters
        ----------
        progress : bool,
            Show progress-bar using `tqdm` package.

        """
        # ---- Initialize Integration Step Zero
        self._init_step_zero()

        # ---- Iterate through all integration steps
        size, nsteps = self.shape
        steps_list = range(1, nsteps)
        steps_list = utils.tqdm(steps_list, desc="evolving binaries") if progress else steps_list
        for step in steps_list:
            self._take_next_step(step)

        # ---- Finalize
        self._finalize()
        return

    def modify(self, mods=None):
        """Apply and modifiers after evolution has been completed.
        """
        self._check_evolved()

        if mods is None:
            mods = self._mods

        # Sanitize
        if mods is None:
            mods = []
        elif not isinstance(mods, list):
            mods = [mods]

        # Run Modifiers
        for mod in mods:
            mod(self)

        # Update derived quantites (following modifications)
        self._update_derived()
        return

    def at(self, xpar, targets, params=None, coal=False, lin_interp=None):
        """Interpolate evolution to the given target locations in either separation or frequency.

        The variable of interpolation is specified with the `xpar` argument.  The evolutionary
        tracks are interpolated in that variable, to the new target locations given by `targets`.
        We use 'x' to refer to the independent variable, and 'y' to refer to the dependent variable
        that is being interpolated.  Which values are interpolated are specified with the `params`
        argument.

        The behavior of this function is broken into three sub-functions, that are only used here:
        * :meth:`Evolution._at__inputs` : parse the input arguments.
        * :meth:`Evolution._at__index_frac` : find the indices in the evolutionary tracks bounding
          the target interpolation locations, and also the fractional distance to interpolate
          between them.
        * :meth:`Evolution._at__interpolate_array` : actually interpolate each parameter to a
          the target location.

        Parameters
        ----------
        xpar : str, in ['fobs', 'sepa']
            String specifying the variable of interpolation.
        targets : array_like,
            Locations to interpolate to.
            * if ``xpar == sepa`` : binary separation, units of [cm],
            * if ``xpar == fobs`` : binary orbital freq, observer-frame, units of [1/sec],
        params : None or (list of str)
            Names of the parameters that should be interpolated.
            If `None`, defaults to :attr:`Evolution._EVO_PARS` attribute.
        coal : bool,
            Only store evolution values for binaries coalescing before redshift zero.
            Interpolated values for other binaries are set to `np.nan`.
        lin_interp : None or bool,
            Interpolate parameters in linear space.
            * True : all parameters interpolated in lin-lin space.
            * False: all parameters interpolated in log-log space.
            * None : parameters are interpolated in log-log space, unless they're included in the
              :attr:`Evolution._LIN_INTERP_PARS` attribute.

        Returns
        -------
        vals : dict,
            Dictionary of arrays for each interpolated parameter.
            The returned shape is (N, T), where `T` is the number of target locations to interpolate
            to, and `N` is the total number of binaries.
            Each data array is filled with `np.nan` values if the targets are outside of its
            evolution track.  If ``coal=True``, then binaries that do *not* coalesce before redshift
            zero also have their data array values fillwed with `np.nan`.

        Notes
        -----
        * Out of bounds values are set to `np.nan`.
        * Interpolation is 1st-order in log-log space in general, but properties which are in the
          `_LIN_INTERP_PARS` array are interpolated at 1st-order in lin-lin space.  Parameters
          which can be negative should be interpolated in linear space.  Passing a boolean for the
          `lin_interp` parameter will override the behavior (see `Parameters`_ above).

        """
        # parse/sanitize input arguments
        xnew, xold, params, lin_interp_list, rev, squeeze = self._at__inputs(xpar, targets, params, lin_interp)

        # (N, M); scale-factors; make sure direction matches that of `xold`
        scafa = self.scafa[:, ::-1] if rev else self.scafa[...]

        # find indices between which to interpolate, and the fractional distance to go between them
        cut_idx, interp_frac, valid = self._at__index_frac(xnew, xold)

        # if we only want coalescing systems, set non-coalescing (stalling) systems to invalid
        if coal:
            valid = valid & self.coal[:, np.newaxis]

        # Valid binaries must be valid at both `bef` and `aft` indices
        # BUG: is this actually doing what it's supposed to be doing?
        for cc in cut_idx:
            valid = valid & np.isfinite(np.take_along_axis(scafa, cc, axis=-1))

        invalid = ~valid

        data = dict()
        # Interpolate each parameter to the given locations, store to `dict`
        for par in params:
            # Load the raw evolution data for this parameter, can be None or ndarray shaped (N, M) or (N, M, 2)
            yold = getattr(self, par)
            if yold is None:
                data[par] = None
                continue

            # Reverse data to match x-values, if needed
            if rev:
                yold = yold[..., ::-1]

            # interpolate
            lin_interp_flag = (par in lin_interp_list)
            ynew = self._at__interpolate_array(yold, cut_idx, interp_frac, lin_interp_flag)

            # fill 'invalid' (i.e. out of bounds, or non-coalescing binaries if ``coal==True``)
            ynew[invalid, ...] = np.nan
            # remove excess dimensions if a single target was requested (i.e. ``T=1``)
            if squeeze:
                ynew = ynew.squeeze()
            # store
            data[par] = ynew

        return data

    def _at__inputs(self, xpar, targets, params, lin_interp):
        """Parse/sanitize the inputs of the :meth:`Evolution.at` method.

        Parameters
        ----------
        xpar : str, in ['fobs', 'sepa']
            String specifying the variable of interpolation.
        targets : array_like,
            Locations to interpolate to.  One of:
            * if ``xpar == sepa``  : binary separation, units of [cm],
            * if ``xpar == fobs`` : binary orbital-frequency, observer-frame, units of [1/sec].
        params : None or list[str]
            Names of parameters that should be interpolated.
            If `None`, defaults to :attr:`Evolution._EVO_PARS` attribute.

        Returns
        -------
        xnew : np.ndarray
            (T,) Log10 of the target locations to interpolate to, i.e. ``log10(targets)``.
        xold : np.ndarray
            (N, M) Log10 of the x-values at which to evaluate the target interpolation points.
            Either ``log10(sepa)`` or ``log10(freq_orb_obs)``.
            NOTE: these values will be returned in *increasing* order.  If they have been reversed,
            then ``rev=True``.
        params : list[str]
            Names of parameters that should be interpolated.
        rev : bool
            Whether or not the `xold` array has been reversed.
        squeeze : bool
            Whether or not the `targets` were a single scalar value (i.e. ``T=1``, as opposed to an
            iterable).  If `targets` were a scalar, then the data returned from :meth:`Evolution.at`
            will be shaped as (N,) instead of (N,T); since in this case, T=1.

        """
        # Raise an error if this instance has not been evolved yet
        self._check_evolved()

        _allowed = ['sepa', 'fobs']
        if xpar not in _allowed:
            raise ValueError("`xpar` must be one of '{}'!".format(_allowed))

        if params is None:
            params = self._EVO_PARS
        if np.isscalar(params):
            params = [params]

        if lin_interp is None:
            lin_interp_list = self._LIN_INTERP_PARS
        elif isinstance(lin_interp, list):
            lin_interp_list = lin_interp
            lin_interp = None
            for ll in lin_interp_list:
                if ll not in params:
                    err = f"`lin_interp` value {ll} not in parameters list {params}!"
                    raise ValueError(err)
        elif lin_interp is True:
            lin_interp_list = params
        elif lin_interp is False:
            lin_interp_list = []
        else:
            err = f"`lin_interp` ({lin_interp}) must be `None`, boolean, or a list of parameter names!"
            raise ValueError(err)

        squeeze = False
        if np.isscalar(targets):
            targets = np.atleast_1d(targets)
            squeeze = True

        size, nsteps = self.shape

        # Observer-frame orbital frequency, units of [1/sec] = [Hz]
        if xpar == 'fobs':
            # frequency is already increasing (must be true for interplation later)
            xold = np.log10(self.freq_orb_obs)
            xnew = np.log10(targets)
            rev = False
        # Binary-Separation, units of [cm]
        elif xpar == 'sepa':
            # separation is decreasing, reverse to increasing (for interpolation)
            xold = np.log10(self.sepa)[:, ::-1]
            xnew = np.log10(targets)
            rev = True
        else:   # nocov
            # This should never be reached, we already checked `xpar` is valid above
            raise ValueError("Bad `xpar` {}!".format(xpar))

        # Make sure target values are within bounds
        textr = utils.minmax(xnew)
        xextr = utils.minmax(xold)
        if (textr[1] < xextr[0]) | (textr[0] > xextr[1]):
            err = "`targets` extrema ({}) outside `xvals` extema ({})!  Bad units?".format(
                (10.0**textr), (10.0**xextr))
            raise ValueError(err)

        return xnew, xold, params, lin_interp_list, rev, squeeze

    def _at__index_frac(self, xnew, xold):
        """Find indices bounding target locations, and the fractional distance to go between them.

        Parameters
        ----------
        xnew : np.ndarray
            Target locations to interplate to.  Shape (T,).
        xold : np.ndarray
            Values of the x-coordinate between which to interpolate.  Shape (N, M).
            These are the x-values of either `sepa` or `fobs` from the evolutionary tracks of each
            binary.

        Returns
        -------
        cut_idx : np.ndarray
            For each binary, the step-number indices between which to interpolate, for each target
            interpolation point.  shape (2, N, T); where the 0th dimension, the 0th value is the
            low/before index, and the 1th value is the high/after index.
            i.e. for binary 'i' and target 'j', we need to interpolate between indices given by
            [0, i, j] and [1, i, j].
        interp_frac : np.ndarray
            The fractional distance between the low value and the high value, to interpolate to.
            Shape (2, N, M).  For binary 'i' and target 'j', `interp_frac[i, j]` is how the
            fraction of the way, from index `cut_idx[0, i, j]` to `cut_idx[1, i, j]` to interpolate
            to, in the `data` array.
        valid : np.ndarray
            Array of boolean values, giving whether or not the target interpolation points are
            within the bounds of each binary's evolution track.  Shape (N, T).

        """
        # ---- For every binary, find the step index immediately following each target value
        # (N, T, M) | `xnew` is (T,) for T-targets,  `xold` is (N, M) for N-binaries and M-steps
        select = (xnew[np.newaxis, :, np.newaxis] <= xold[:, np.newaxis, :])
        # (N, T), xvalue index [0, M-1] following each target point (T,), for each binary (N,)
        aft = np.argmax(select, axis=-1)

        # ---- Determine which locations are 'valid' (i.e. within the evolutionary tracks)
        # zero values in `aft` mean no `xold` after the targets were found; these are 'invalid',
        # these will be converted to `np.nan` later
        valid = (aft > 0)

        # ---- get the x-value index immediately preceding each target point
        bef = np.copy(aft)
        bef[valid] -= 1
        # (2, N, T)
        cut_idx = np.array([aft, bef])

        # Get the x-values before and after the target locations  (2, N, T)
        xold_temp = [np.take_along_axis(xold, cc, axis=-1) for cc in cut_idx]
        # Find how far to interpolate between values (in log-space; `xold` was already log10'd
        #     (N, T)
        denom = np.subtract(*xold_temp)
        numer = xnew[np.newaxis, :] - xold_temp[1]
        interp_frac = np.zeros_like(numer)
        idx = (denom != 0.0)
        interp_frac[idx] = numer[idx] / denom[idx]

        return cut_idx, interp_frac, valid

    def _at__interpolate_array(self, yold, cut_idx, interp_frac, lin_interp_flag):
        """Interpolate a parameter to a fraction between integration steps.

        Parameters
        ----------
        yold : np.ndarray
            The data to be interpolated.  This is the raw evolution data, for each binary and
            each step.  Shaped either as (N, M) or (N, M, 2) if parameter is mass.
        cut_idx : np.ndarray
            For each binary, the step-number indices between which to interpolate, for each target
            interpolation point.  shape (2, N, T); where the 0th dimension, the 0th value is the
            low/before index, and the 1th value is the high/after index.
            i.e. for binary 'i' and target 'j', we need to interpolate between indices given by
            [0, i, j] and [1, i, j].
        interp_frac : np.ndarray
            The fractional distance between the low value and the high value, to interpolate to.
            Shape (2, N, M).  For binary 'i' and target 'j', `interp_frac[i, j]` is how the
            fraction of the way, from index `cut_idx[0, i, j]` to `cut_idx[1, i, j]` to interpolate
            to, in the `data` array.
        lin_interp_flag : bool,
            Whether data should be interpolated in lin-lin space (True), or log-log space (False).

        Returns
        -------
        ynew : np.ndarray
            The input `data` interpolated to the new target locations.
            Shape is (N, T) or (N, T, 2) for N-binaries, T-target points.  A third dimension is
            present if the input `data` was 3D.

        """

        reshape = False
        cut = cut_idx
        frac = interp_frac
        # Sometimes there is a third dimension for the 2 binaries (e.g. `mass`)
        #    which will have shape, (N, T, 2) --- calling this "double-data"
        if np.ndim(yold) != 2:
            if (np.ndim(yold) == 3) and (np.shape(yold)[-1] == 2):
                # Keep the interpolation axis last (N, T, 2) ==> (N, 2, T)
                yold = np.moveaxis(yold, -1, -2)
                # Expand other arrays appropriately
                cut = cut[:, :, np.newaxis]
                frac = frac[:, np.newaxis, :]
                reshape = True
            else:   # nocov
                raise ValueError("Unexpected shape of yold: {}!".format(np.shape(yold)))

        if not lin_interp_flag:
            yold = np.log10(yold)

        # (2, N, T) for scalar data or (2, N, 2, T) for "double-data"
        yold = [np.take_along_axis(yold, cc, axis=-1) for cc in cut]
        # Interpolate by `frac` for each binary   (N, T) or (N, 2, T) for "double-data"
        ynew = yold[1] + (np.subtract(*yold) * frac)
        # In the "double-data" case, move the doublet back to the last dimension
        #    (N, T) or (N, T, 2)
        if reshape:
            ynew = np.moveaxis(ynew, 1, -1)

        # fill return dictionary
        if not lin_interp_flag:
            ynew = 10.0 ** ynew

        return ynew

    @utils.deprecated_fail('Evolution.sample_universe')
    def sample_full_population(self, fobs, DOWN=None):
        return

    def sample_universe(self, fobs_orb_edges, down_sample=None):
        """Construct a full universe of binaries based on resampling this population.

        Parameters
        ----------
        fobs : array_like,
            Observer-frame *orbital*-frequencies at which to sample population. Units of [1/sec].
        down_sample : None or float,
            Factor by which to downsample the resulting population.
            For example, `10.0` will produce 10x fewer output binaries.

        Returns
        -------
        names : list[str], size (4,)
            Names of the returned data arrays in `samples`.
        samples : np.ndarray, shape (4, S)
            Sampled binary data.  For each binary samples S, 4 parameters are returned:
            ['mtot', 'mrat', 'redz', 'fobs'] (these are listed in the `names` returned value.)
            NOTE: `fobs` is *observer*-frame *orbital*-frequencies.

        To-Do
        -----
        * This should sample in volume instead of `redz`, see how it's done in sam module.

        """

        # these are `log10(values)` where values are in CGS units
        # names = ['mtot', 'mrat', 'redz', 'fobs']
        names, vals, weights = self._sample_universe__at_values_weights(fobs_orb_edges)

        samples = self._sample_universe__resample(fobs_orb_edges, vals, weights, down_sample)

        # Convert back to normal-space
        samples = np.asarray([10.0 ** ss for ss in samples])
        vals = np.asarray([10.0 ** vv for vv in vals])
        return names, samples, vals, weights

    def _sample_universe__at_values_weights(self, fobs_orb_edges):
        fobs_orb_cents = kale.utils.midpoints(fobs_orb_edges, log=False)
        dlnf = np.diff(np.log(fobs_orb_edges))

        # Interpolate binaries to given frequencies; these are the needed parameters
        PARAMS = ['mass', 'sepa', 'dadt', 'scafa']
        # each array within `data_fobs` is shaped (N, F) for N-binaries and F-frequencies (`fobs`)
        data_fobs = self.at('fobs', fobs_orb_cents, params=PARAMS)

        # Only examine binaries reaching the given locations before redshift zero (other redz=infinite)
        redz = cosmo.a_to_z(data_fobs['scafa'])
        valid = (redz > 0.0)
        log.debug(f"After interpolation, valid binary-targets: {utils.frac_str(valid)}")

        # Get rest-frame orbital-frequency [1/s]
        frst_orb_cents = utils.frst_from_fobs(fobs_orb_cents[np.newaxis, :], redz)
        # Comoving distance [cm]
        dcom = cosmo.z_to_dcom(redz)

        # `mass` has shape (Binaries, Frequencies, 2), units [gram]
        #    convert to (2, B, F), then separate into m1, m2 each with shape (B, F)
        m1, m2 = np.moveaxis(data_fobs['mass'], -1, 0)
        dfdt, _ = utils.dfdt_from_dadt(data_fobs['dadt'], data_fobs['sepa'], frst_orb=frst_orb_cents)

        _lambda_factor = utils.lambda_factor_dlnf(frst_orb_cents, dfdt, redz, dcom=dcom) / self._sample_volume
        num_binaries = _lambda_factor * dlnf[np.newaxis, :]

        # select only valid entries
        mt, mr = utils.mtmr_from_m1m2(m1[valid], m2[valid])
        # broadcast `fobs` to match the shape of binaries, then select valid entries
        fo = (fobs_orb_cents[np.newaxis, :] * np.ones_like(redz))[valid]
        redz = redz[valid]
        weights = num_binaries[valid]
        log.debug(f"Weights (lambda values) at targets: {utils.stats(weights)}")

        # Convert to log-space
        vals = [np.log10(mt), np.log10(mr), np.log10(redz), np.log10(fo)]
        names = ['mtot', 'mrat', 'redz', 'fobs']
        return names, vals, weights

    def _sample_universe__resample(self, fobs_orb_edges, vals, weights, down_sample):
        # down-sample weights to decrease the number of sample points
        prev_sum = weights.sum()
        log.info(f"Total weights (number of binaries in the universe): {prev_sum:.8e}")
        if down_sample is not None:
            weights = weights / down_sample
            next_sum = weights.sum()
            msg = f"downsampling artificially: down_sample={down_sample:g} :: total: {prev_sum:.4e}==>{next_sum:.4e}"
            log.warning(msg)

        # TODO/FIX: Consider sampling in comoving-volume instead of redz (like in sam.py)
        #           can also return dcom instead of redz for easier strain calculation
        nsamp = np.random.poisson(weights.sum())
        reflect = [None, [None, 0.0], None, np.log10([fobs_orb_edges[0], fobs_orb_edges[-1]])]
        samples = kale.resample(vals, size=nsamp, reflect=reflect, weights=weights, bw_rescale=0.5)
        samples = np.power(10.0, samples)
        num_samp = samples[0].size
        log.debug(f"Sampled {num_samp:.8e} binaries in the universe")
        return samples

    # ==== Internal Methods

    def _init_step_zero(self):
        """Set the initial conditions of the binaries at the 0th step.

        Transfers attributes from the stored :class:`holodeck.population._Population_Discrete`
        instance to the 0th index of the evolution arrays.  The attributes are [`sepa`, `scafa`,
        `mass`, and optionally `eccen`].  The hardening model is also used to calculate the 0th
        hardening rates `dadt` and `dedt`.  The initial lookback time, `tlook` is also set.

        """
        pop = self._pop
        size, nsteps = self.shape

        # ---- Initialize basic parameters

        # Initialize ALL separations ranging from initial to mutual-ISCO, for each binary
        rad_isco = utils.rad_isco(*pop.mass.T)
        # (2, N)
        sepa = np.log10([pop.sepa, rad_isco])
        # Get log-space range of separations for each of N ==> (N, S), for S steps
        sepa = np.apply_along_axis(lambda xx: np.logspace(*xx, nsteps), 0, sepa).T
        self.sepa[:, :] = sepa
        if (pop.eccen is not None):
            self.eccen[:, 0] = pop.eccen

        self.scafa[:, 0] = pop.scafa
        redz = cosmo.a_to_z(pop.scafa)
        tlook = cosmo.z_to_tlbk(redz)
        self.tlook[:, 0] = tlook
        # `pop.mass` has shape (N, 2), broadcast to (N, S, 2) for `S` steps
        #self.mass[:, :, :] = pop.mass[:, np.newaxis, :]
        self.mass[:, 0, :] = pop.mass
        #HERE INITIAL MASSES ARE COPIED FOR EVERY STEP
        self.mass[:, :, :] = self.mass[:, 0, np.newaxis, :]

        if self._debug:    # nocov
            for ii, hard in enumerate(self._hard):
                # Store individual hardening rates
                setattr(self, f"_dadt_{ii}", np.zeros_like(self.dadt))
                setattr(self, f"_dedt_{ii}", np.zeros_like(self.dadt))

        # ---- Initialize hardening rate at first step
        dadt_init, dedt_init = self._hardening_rate(step=0)

        self.dadt[:, 0] = dadt_init
        if (pop.eccen is not None):
            self.dedt[:, 0] = dedt_init

        return

    def _take_next_step(self, step):
        """Integrate the binary population forward (to smaller separations) by one step.

        For an integration step `s`, we are moving from index `s-1` to index `s`.  These correspond
        to the 'left' and 'right' edges of the step.  The evolutionary trajectory values have
        already been calculated on the left edges (during either the previous time step, or the
        initial time step).  Each subsequent integration step then proceeds as follows:

        (1) The hardening rate is calculated at the right edge of the step.
        (2) The time it takes to move from the left to right edge is calculated using a trapezoid
            rule in log-log space.
        (3) The right edge evolution values are stored and updated.

        Parameters
        ----------
        step : int
            The destination integration step number, i.e. `step=1` means integrate from 0 to 1.

        """
        # ---- Initialize
        size, nsteps = self.shape
        left = step - 1     # the previous time-step (already completed)
        right = step        # the next     time-step

        # get the separation $a$ on both edges
        sepa = self.sepa[:, (right, left)]   # sepa is decreasing, so switch left-right order

        # ! ====================================================================
        # ---- Hardening rates at the left-edge of the step
        # calculate
        dadt_l, dedt_l = self._hardening_rate(left, store_debug=False)
        da = np.diff(sepa, axis=-1)
        da = da[:, 0]
        dt = da / -dadt_l
        if np.any(dt < 0.0):    # nocov
            err = f"Negative time-steps found at step={step}!"
            log.exception(err)
            raise ValueError(err)

        if self.eccen is not None:
            de = dedt_l * dt
            ecc_r = self.eccen[:, left] + de
            ecc_r = np.clip(ecc_r, 0.0, 1.0 - _MAX_ECCEN_ONE_MINUS)
            self.eccen[:, right] = ecc_r

        # Update lookback time based on duration of this step
        tlook = self.tlook[:, left] - dt
        self.tlook[:, right] = tlook
        # update scale-factor for systems at z > 0.0 (i.e. a < 1.0 and tlook > 0.0)
        val = (tlook > 0.0)
        self.scafa[val, right] = cosmo.z_to_a(cosmo.tlbk_to_z(tlook[val]))
        # set systems after z = 0 to scale-factor of unity
        self.scafa[~val, right] = 1.0
        # ! ====================================================================

        # ---- Hardening rates at the right-edge of the step
        # calculate
        dadt_r, dedt_r = self._hardening_rate(right, store_debug=True)

        # store
        self.dadt[:, right] = dadt_r
        if self.eccen is not None:
            self.dedt[:, right] = dedt_r

        # ---- Calculate time between edges

        # get the $dt/da$ rate on both edges of the step
        dtda = 1.0 / - self.dadt[:, (left, right)]   # NOTE: `dadt` is negative, convert to positive
        # use trapezoid rule to find total time for this step
        dt = utils.trapz_loglog(dtda, sepa, axis=-1).squeeze()   # this should come out positive
        if np.any(dt < 0.0):    # nocov
            err = f"Negative time-steps found at step={step}!"
            log.exception(err)
            raise ValueError(err)

        # ---- Update right-edge values
        # NOTE/ENH: this would be a good place to make a function `_update_right_edge()` (or something like that),
        # that stores the updated right edge values, and also performs any additional updates, such as mass evolution

        # Update lookback time based on duration of this step
        tlook = self.tlook[:, left] - dt
        self.tlook[:, right] = tlook
        # update scale-factor for systems at z > 0.0 (i.e. a < 1.0 and tlook > 0.0)
        val = (tlook > 0.0)
        self.scafa[val, right] = cosmo.z_to_a(cosmo.tlbk_to_z(tlook[val]))
        # set systems after z = 0 to scale-factor of unity
        self.scafa[~val, right] = 1.0

        # update eccentricity if it's being evolved
        if self.eccen is not None:
            dedt = self.dedt[:, (left, right)]
            time = self.tlook[:, (right, left)]   # tlook is decreasing, so switch left-right order
            # decc = utils.trapz_loglog(dedt, time, axis=-1).squeeze()
            decc = utils.trapz(dedt, time, axis=-1).squeeze()
            ecc_r = self.eccen[:, left] + decc
            ecc_r = np.clip(ecc_r, 0.0, 1.0 - _MAX_ECCEN_ONE_MINUS)
            self.eccen[:, right] = ecc_r
            if self._debug:    # nocov
                bads = ~np.isfinite(decc)
                if np.any(bads):
                    utils.print_stats(print_func=log.error, dedt=dedt, time=time, decc=decc)
                    err = f"Non-finite changes in eccentricity found in step {step}!"
                    log.exception(err)
                    raise ValueError(err)

        return

    def _hardening_rate(self, step, store_debug=True):
        """Calculate the net hardening rate for the given integration step.

        The hardening rates (:class:`_Hardening` subclasses) stored in the :attr:`Evolution._hard`
        attribute are called in sequence, their :meth:`_Hardening.dadt_dedt` methods are called,
        and the $da/dt$ and $de/dt$ hardening rates are added together.
        NOTE: the da/dt and de/dt values are added together to get the net rate, this is an
        approximation.

        Parameters
        ----------
        step : int
            Current step number (the destination of the current step, i.e. step=1 is for integrating
            from 0 to 1.)

        Returns
        -------
        dadt : np.ndarray
            The hardening rate in separation, $da/dt$, in units of [cm/s].
            The shape is (N,) where N is the number of binaries.
        dedt : np.ndarray or None
            If eccentricity is not being evolved, this is `None`.  If eccentricity is being evolved,
            this is the hardening rate in eccentricity, $de/dt$, in units of [1/s].
            In this case, the shape is (N,) where N is the number of binaries.

        """
        dadt = np.zeros(self.shape[0])
        dedt = None if self.eccen is None else np.zeros_like(dadt)

        for ii, hard in enumerate(self._hard):
            _hard_dadt, _ecc = hard.dadt_dedt(self, step)
            dadt[:] += _hard_dadt
            if self._debug:    # nocov
                log.debug(f"{step} hard={hard} : dadt = {utils.stats(_hard_dadt)}")
                # Store individual hardening rates
                if store_debug:
                    getattr(self, f"_dadt_{ii}")[:, step] = _hard_dadt[...]
                # Raise error on invalid entries
                bads = ~np.isfinite(_hard_dadt) | (_hard_dadt > 0.0)
                if np.any(bads):
                    log.error(f"{step} hard={hard} : dadt = {utils.stats(_hard_dadt)}")
                    err = f"invalid `dadt` for hard={hard}  (bads: {utils.frac_str(bads)})!"
                    log.exception(err)
                    log.error(f"BAD dadt = {_hard_dadt[bads]}")
                    log.error(f"BAD sepa = {self.sepa[bads, step]}")
                    log.error(f"BAD mass = {self.sepa[bads, step]}")
                    raise ValueError(err)

            if (self.eccen is not None):
                if _ecc is None:
                    log.warning(f"`Evolution.eccen` is not None, but `dedt` is None!  {step} {hard}")
                    continue
                dedt[:] += _ecc
                if self._debug:    # nocov
                    log.debug(f"{step} hard={hard} : dedt = {utils.stats(_ecc)}")
                    # Raise error on invalid entries
                    if not np.all(np.isfinite(_ecc)):
                        err = f"invalid `dedt` for hard={hard}!"
                        log.exception(err)
                        raise ValueError(err)
                    # Store individual hardening rates
                    if store_debug:
                        getattr(self, f"_dedt_{ii}")[:, step] = _ecc[...]

            if self._accmod is not None:
                #Get total accretion rates
                acc = accretion.Accretion(self, step)
                if self._accmod == 'Basic':
                    self.mdot[:,step-1,:] = acc.basic_accretion()
                if self._accmod == 'Proportional':
                    self.mdot[:,step-1,:] = acc.proportional_accretion()
                if self._accmod == 'Primary':
                    self.mdot[:,step-1,:] = acc.primary_accretion()
                if self._accmod == 'Secondary':
                    self.mdot[:,step-1,:] = acc.secondary_accretion()
                if self._accmod == 'Duffell':
                    self.mdot[:,step-1,:] = acc.duffell_accretion()

                self.mass[:, step, 0] = self.mass[:, step-1, 0] + dt * self.mdot[:,step-1,0]
                self.mass[:, step, 1] = self.mass[:, step-1, 1] + dt * self.mdot[:,step-1,1]


        return dadt, dedt

    def _check(self):
        """Perform basic diagnostics on parameter validity after evolution.
        """
        _check_var_names = ['sepa', 'scafa', 'mass', 'tlook', 'dadt']
        _check_var_names_eccen = ['eccen', 'dedt']

        def check_vars(names):
            for cv in names:
                vals = getattr(self, cv)
                if np.any(~np.isfinite(vals)):    # pragma: no cover
                    err = "Found non-finite '{}' !".format(cv)
                    raise ValueError(err)

        check_vars(_check_var_names)

        if self.eccen is None:
            return

        check_vars(_check_var_names_eccen)

        return

    def _finalize(self):
        """Perform any actions after completing all of the integration steps.
        """
        # Set a flag to record that evolution has been completed
        self._evolved = True
        # Apply any modifiers
        self.modify()
        # Run diagnostics
        self._check()
        return

    def _update_derived(self):
        """Update any derived quantities after modifiers are applied.
        """
        pass

    # ==== Properties and generic functionality

    @property
    def shape(self):
        """The number of binaries and number of steps (N, S)."""
        return self._shape

    @property
    def size(self):
        """The number of binaries"""
        return self._shape[0]

    @property
    def steps(self):
        """The number of evolution steps"""
        return self._shape[1]

    @property
    def coal(self):
        """Indices of binaries that coalesce before redshift zero.
        """
        if self._coal is None:
            self._coal = (self.scafa[:, -1] < 1.0)
        return self._coal

    @property
    def tage(self):
        """Age of the universe [sec] for each binary-step.

        Derived from :attr:`Evolution.tlook`.

        Returns
        -------
        ta : np.ndarray,
            (B, S).  Age of the universe.

        """
        ta = cosmo.age(0.0).cgs.value - self.tlook
        return ta

    @property
    def mtmr(self):
        """Total-mass and mass-ratio.

        Returns
        -------
        mt : np.ndarray
            Total mass ($M = m_1 + m_2$) in [gram].
        mr : np.ndarray
            Mass ratio ($q = m_2/m_1 \leq 1.0$).

        """
        mass = np.moveaxis(self.mass, -1, 0)   # (N, M, 2) ==> (2, N, M)
        mt, mr = utils.mtmr_from_m1m2(*mass)
        return mt, mr

    @property
    def freq_orb_rest(self):
        """Rest-frame orbital frequency. [1/s]
        """
        if self._freq_orb_rest is None:
            self._check_evolved()
            mtot = self.mass.sum(axis=-1)
            self._freq_orb_rest = utils.kepler_freq_from_sepa(mtot, self.sepa)
        return self._freq_orb_rest

    @property
    def freq_orb_obs(self):
        """Observer-frame orbital frequency. [1/s]
        """
        redz = cosmo.a_to_z(self.scafa)
        fobs = self.freq_orb_rest / (1.0 + redz)
        return fobs

    def _check_evolved(self):
        """Raise an error if this instance has not yet been evolved.
        """
        if self._evolved is not True:
            raise RuntimeError("This instance has not been evolved yet!")
        return


# =================================================================================================
# ====    Hardening Classes    ====
# =================================================================================================


class _Hardening(abc.ABC):
    """Base class for binary-hardening models, providing the `dadt_dedt(evo, step, ...)` method.
    """

    @abc.abstractmethod
    def dadt_dedt(self, evo, step, *args, **kwargs):
        pass

    def dadt(self, *args, **kwargs):
        rv_dadt, _dedt = self.dadt_dedt(*args, **kwargs)
        return rv_dadt

    def dedt(self, *args, **kwargs):
        _dadt, rv_dedt = self.dadt_dedt(*args, **kwargs)
        return rv_dedt


class Hard_GW(_Hardening):
    """Gravitational-wave driven binary hardening.
    """

    @staticmethod
    def dadt_dedt(evo, step):
        """Calculate GW binary evolution (hardening rate) in semi-major-axis and eccentricity.

        Parameters
        ----------
        evo : `Evolution`
            Evolution instance providing the binary parameters for calculating hardening rates.
        step : int
            Evolution integration step index from which to load binary parameters.
            e.g. separations are loaded as ``evo.sepa[:, step]``.

        Returns
        -------
        dadt : np.ndarray
            Hardening rate in semi-major-axis, returns negative value, units [cm/s].
        dedt : np.ndarray
            Hardening rate in eccentricity, returns negative value, units [1/s].

        """
        m1, m2 = evo.mass[:, step, :].T    # (Binaries, Steps, 2) ==> (2, Binaries)
        sepa = evo.sepa[:, step]
        eccen = evo.eccen[:, step] if (evo.eccen is not None) else None
        dadt = utils.gw_hardening_rate_dadt(m1, m2, sepa, eccen=eccen)

        if eccen is None:
            dedt = None
        else:
            dedt = utils.gw_dedt(m1, m2, sepa, eccen)

        return dadt, dedt

    @staticmethod
    def dadt(mtot, mrat, sepa, eccen=None):
        """Calculate GW Hardening rate of semi-major-axis vs. time.

        Parameters
        ----------
        mtot : array_like
            Total mass of each binary system.  Units of [gram].
        mrat : array_like
            Mass ratio of each binary, defined as $q \equiv m_1/m_2 \leq 1.0$.
        sepa : array_like
            Binary semi-major axis (separation), in units of [cm].
        eccen : array_like or None
            Binary eccentricity, `None` is the same as zero eccentricity (circular orbit).

        Returns
        -------
        dadt : np.ndarray
            Hardening rate in semi-major-axis, result is negative, units [cm/s].

        """
        m1, m2 = utils.m1m2_from_mtmr(mtot, mrat)
        dadt = utils.gw_hardening_rate_dadt(m1, m2, sepa, eccen=eccen)
        return dadt

    @staticmethod
    def dedt(mtot, mrat, sepa, eccen=None):
        """Calculate GW Hardening rate of eccentricity vs. time.

        If `eccen` is `None`, zeros are returned.

        Parameters
        ----------
        mtot : array_like
            Total mass of each binary system.  Units of [gram].
        mrat : array_like
            Mass ratio of each binary, defined as $q \equiv m_1/m_2 \leq 1.0$.
        sepa : array_like
            Binary semi-major axis (separation), in units of [cm].
        eccen : array_like or None
            Binary eccentricity, `None` is the same as zero eccentricity (circular orbit).

        Returns
        -------
        dedt : np.ndarray
            Hardening rate in eccentricity, result is <= 0.0, units [1/s].
            Zero values if `eccen` is `None`.

        """
        if eccen is None:
            return np.zeros_like(mtot)
        m1, m2 = utils.m1m2_from_mtmr(mtot, mrat)
        dedt = utils.gw_dedt(m1, m2, sepa, eccen=eccen)
        return dedt


class Sesana_Scattering(_Hardening):
    """Binary-Hardening Rates calculated based on the Sesana stellar-scattering model.

    This module uses the stellar-scattering rate constants from the fits in [Sesana2006]_ using the
    `_SHM06` class.  Scattering is assumed to only be effective once the binary is bound.  An
    exponential cutoff is imposed at larger radii.

    """

    def __init__(self, gamma_dehnen=1.0, mmbulge=None, msigma=None):
        """Construct an `Stellar_Scattering` instance with the given MBH-Host relations.

        Parameters
        ----------
        gamma_dehnen : array_like
            Dehnen stellar-density profile inner power-law slope.
            Fiducial Dehnen inner density profile slope ``gamma=1.0`` is used in [Chen2017]_.
        mmbulge : None or `holodeck.relations._MMBulge_Relation`
            Mbh-Mbulge relation to calculate stellar mass for a given BH mass.
            If `None` a default relationship is used.
        msigma : None or `holodeck.relations._MSigma_Relation`
            Mbh-Sigma relation to calculate stellar velocity dispersion for a given BH mass.
            If `None` a default relationship is used.

        """
        self._mmbulge = holo.relations.get_mmbulge_relation(mmbulge)
        self._msigma = holo.relations.get_msigma_relation(msigma)
        self._gamma_dehnen = gamma_dehnen
        self._shm06 = _SHM06()
        return

    def dadt_dedt(self, evo, step):
        """Stellar scattering hardening rate.

        Parameters
        ----------
        evo : `Evolution`
            Evolution instance providing binary parameters at the given intergration step.
        step : int
            Integration step at which to calculate hardening rates.

        Returns
        -------
        dadt : array_like
            Binary hardening rates in units of [cm/s], defined to be negative.
        dedt : array_like
            Binary rate-of-change of eccentricity in units of [1/sec].

        """
        mass = evo.mass[:, step, :]
        sepa = evo.sepa[:, step]
        eccen = evo.eccen[:, step] if evo.eccen is not None else None
        dadt, dedt = self._dadt_dedt(mass, sepa, eccen)
        return dadt, dedt

    def _dadt_dedt(self, mass, sepa, eccen):
        """Stellar scattering hardening rate calculated from physical quantities.

        Parameters
        ----------
        mass : (N,2) array_like
            Masses of each MBH component (0-primary, 1-secondary) in units of [gram].
        sepa : (N,) array_like
            Binary separation in units of [cm].
        eccen : (N,) array_like or `None`
            Binary eccentricity.  `None` if eccentricity is not being evolved.

        Returns
        -------
        dadt : (N,) array-like of scalar
            Binary hardening rates in units of [cm/s], defined to be negative.
        dedt : (N,) array-like of scalar  or  `None`
            Binary rate-of-change of eccentricity in units of [1/sec].
            If eccentricity is not being evolved (i.e. `eccen==None`) then `None` is returned.

        """
        mass = np.atleast_2d(mass)
        sepa = np.atleast_1d(sepa)
        eccen = np.atleast_1d(eccen) if eccen is not None else None
        mtot, mrat = utils.mtmr_from_m1m2(mass)
        mbulge = self._mmbulge.mbulge_from_mbh(mtot, scatter=False)
        vdisp = self._msigma.vdisp_from_mbh(mtot, scatter=False)
        dens = _density_at_influence_radius_dehnen(mtot, mbulge, self._gamma_dehnen)

        rhard = _Quinlan1996.radius_hardening(mass[:, 1], vdisp)
        hh = self._shm06.H(mrat, sepa/rhard)
        dadt = _Quinlan1996.dadt(sepa, dens, vdisp, hh)

        rbnd = _radius_influence_dehnen(mtot, mbulge)
        atten = np.exp(-sepa / rbnd)
        dadt = dadt * atten

        if eccen is not None:
            kk = self._shm06.K(mrat, sepa/rhard, eccen)
            dedt = _Quinlan1996.dedt(sepa, dens, vdisp, hh, kk)
        else:
            dedt = None

        return dadt, dedt


class Dynamical_Friction_NFW(_Hardening):
    """Dynamical Friction (DF) hardening module assuming an NFW dark-matter density profile.

    This class calculates densities and orbital velocities based on a NFW profile with parameters based on those of
    each MBH binary.  The `holodeck.observations.NFW` class is used for profile calculations, and the halo parameters
    are calculated from Stellar-mass--Halo-mass relations (see 'arguments' below).  The 'effective-mass' of the
    inspiralling secondary is modeled as a power-law decreasing from the sum of secondary MBH and its stellar-bulge
    (calculated using the `mmbulge` - Mbh-Mbulge relation), down to just the bare secondary MBH after 10 dynamical
    times.  This is to model tidal-stripping of the secondary host galaxy.

    Attenuation of the DF hardening rate is typically also included, to account for the inefficiency of DF once the
    binary enters the hardened regime.  This is calculated using the prescription from [BBR1980]_.  The different
    characteristic radii, needed for the attenuation calculation, currently use a fixed Dehnen stellar-density profile
    as in [Chen2017]_, and a fixed scaling relationship to find the characteristic stellar-radius.

    Notes
    -----
    *   This module does not evolve eccentricity.
    *   The hardening rate (da/dt) is not allowed to be larger than the orbital/virial velocity of the halo
        (as a function of radius).

    """

    _TIDAL_STRIPPING_DYNAMICAL_TIMES = 10.0

    def __init__(self, mmbulge=None, msigma=None, smhm=None, coulomb=10.0, attenuate=True, rbound_from_density=True):
        """Create a hardening rate instance with the given parameters.

        Parameters
        ----------
        mmbulge : None or `holodeck.relations._MMBulge_Relation`
            Mbh-Mbulge relation to calculate stellar mass for a given BH mass.
            If `None` a default relationship is used.
        msigma : None or `holodeck.relations._MSigma_Relation`
            Mbh-Sigma relation to calculate stellar velocity dispersion for a given BH mass.
            If `None` a default relationship is used.
        smhm : class, instance or None
            Stellar-mass--halo-mass relation (_StellarMass_HaloMass subclass)
            If `None` the default is loaded.
        coulomb : scalar,
            coulomb-logarithm ("log(Lambda)"), typically in the range of 10-20.
            This parameter is formally the log of the ratio of maximum to minimum impact parameters.
        attenuate : bool,
            Whether the DF hardening rate should be 'attenuated' due to stellar-scattering effects at
            small radii.  If `True`, DF becomes significantly less effective for radii < R-hard and R-LC
        rbound_from_density : bool,
            Determines how the binding radius (of MBH pair) is calculated, which is used for attenuation.
            NOTE: this is only used if `attenuate==True`
            If True:  calculate R-bound using an assumed stellar density profile.
            If False: calculate R-bound using a velocity dispersion (constant in radius, from `gbh` instance).

        """
        self._mmbulge = holo.relations.get_mmbulge_relation(mmbulge)
        self._msigma = holo.relations.get_msigma_relation(msigma)
        self._smhm = holo.relations.get_stellar_mass_halo_mass_relation(smhm)
        self._coulomb = coulomb
        self._attenuate = attenuate
        self._rbound_from_density = rbound_from_density

        self._NFW = holo.relations.NFW
        self._time_dynamical = None
        return

    def dadt_dedt(self, evo, step, attenuate=None):
        """Calculate DF hardening rate given `Evolution` instance, and an integration `step`.

        Parameters
        ----------
        evo : `Evolution` instance
            The evolutionary tracks of the binary population, providing binary parameters.
        step : int,
            Integration step at which to calculate hardening rates.

        Returns
        -------
        dadt : (N,) np.ndarray of scalar,
            Binary hardening rates in units of [cm/s].
        dedt : (N,) np.ndarray or None
            Rate-of-change of eccentricity, which is not included in this calculation, it is zero.
            `None` is returned if the input `eccen` is None.

        """
        if attenuate is None:
            attenuate = self._attenuate

        mass = evo.mass[:, step, :]
        sepa = evo.sepa[:, step]
        eccen = evo.eccen[:, step] if evo.eccen is not None else None
        dt = evo.tlook[:, 0] - evo.tlook[:, step]   # positive time-duration since 'formation'
        # NOTE `scafa` is nan for systems "after" redshift zero (i.e. do not merge before redz=0)
        redz = np.zeros_like(sepa)
        val = (evo.scafa[:, step] > 0.0)
        redz[val] = cosmo.a_to_z(evo.scafa[val, step])

        dadt, dedt = self._dadt_dedt(mass, sepa, redz, dt, eccen, attenuate)

        return dadt, dedt

    def _dadt_dedt(self, mass, sepa, redz, dt, eccen, attenuate):
        """Calculate DF hardening rate given physical quantities.

        Parameters
        ----------
        mass : (N, 2) array_like
            Masses of both MBHs (0-primary, 1-secondary) in units of [grams].
        sepa : (N,) array_like
            Binary separation in [cm].
        redz : (N,) array_like
            Binary redshifts.
        dt : (N,) array_like
            Time step [sec], required for modeling tidal stripping of secondary galaxy.
        eccen : (N,) array_like or None
            Binary eccentricity.
        attenuate : bool
            Whether to include 'attenuation' as the radius approach the stellar-scattering regime.

        Returns
        -------
        dadt : (N,) np.ndarray
            Binary hardening rates in units of [cm/s].
        dedt : (N,) np.ndarray or None
            Rate-of-change of eccentricity, which is not included in this calculation, it is zero.
            `None` is returned if the input `eccen` is None.

        """
        assert np.shape(mass)[-1] == 2 and np.ndim(mass) <= 2
        mass = np.atleast_2d(mass)
        redz = np.atleast_1d(redz)

        # Get Host DM-Halo mass
        # assume galaxies are merged, and total stellar mass is given from Mstar-Mbh of total MBH mass
        mstar = self._mmbulge.mstar_from_mbh(mass.sum(axis=-1), scatter=False)
        mhalo = self._smhm.halo_mass(mstar, redz, clip=True)

        # ---- Get effective mass of inspiraling secondary
        m2 = mass[:, 1]
        mstar_sec = self._mmbulge.mstar_from_mbh(m2, scatter=False)
        # model tidal-stripping of secondary's bulge (see: [Kelley2017a]_ Eq.6)
        time_dyn = self._NFW.time_dynamical(sepa, mhalo, redz)
        tfrac = dt / (time_dyn * self._TIDAL_STRIPPING_DYNAMICAL_TIMES)
        power_index = np.clip(1.0 - tfrac, 0.0, 1.0)
        meff = m2 * np.power((m2 + mstar_sec)/m2, power_index)
        log.debug(f"DF tfrac = {utils.stats(tfrac)}")
        log.debug(f"DF meff/m2 = {utils.stats(meff/m2)} [Msol]")

        # ---- Get local density
        # set minimum radius to be a factor times influence-radius
        rinfl = _MIN_DENS_RAD__INFL_RAD_MULT * _radius_influence_dehnen(m2, mstar_sec)
        dens_rads = np.maximum(sepa, rinfl)
        dens = self._NFW.density(dens_rads, mhalo, redz)

        # ---- Get velocity of secondary MBH
        mt, mr = utils.mtmr_from_m1m2(mass)
        vhalo = self._NFW.velocity_circular(sepa, mhalo, redz)
        vorb = utils.velocity_orbital(mt, mr, sepa=sepa)[:, 1]  # secondary velocity
        velo = np.sqrt(vhalo**2 + vorb**2)

        # ---- Calculate hardening rate
        # dvdt is negative [cm/s]
        dvdt = self._dvdt(meff, dens, velo)
        # convert from deceleration to hardening-rate assuming virialized orbit (i.e. ``GM/a = v^2``)
        dadt = 2 * time_dyn * dvdt
        dedt = None if (eccen is None) else np.zeros_like(dadt)

        # ---- Apply 'attenuation' following [BBR1980]_ to account for stellar-scattering / loss-cone effects
        if attenuate:
            atten = self._attenuation_BBR1980(sepa, mass, mstar)
            dadt = dadt / atten

        # Hardening rate cannot be larger than orbital/virial velocity
        clip = (np.fabs(dadt) > velo)
        if np.any(clip):
            log.info(f"clipping {utils.frac_str(clip)} `dadt` values to vcirc")
            dadt[clip] = - velo[clip]

        return dadt, dedt

    def _dvdt(self, mass_sec_eff, dens, velo):
        """Chandrasekhar dynamical friction formalism providing a deceleration (dv/dt).

        Parameters
        ----------
        mass_sec_eff : (N,) array-like of scalar
            Effective mass (i.e. the mass that should be used in this equation) of the inspiraling
            secondary component in units of [gram].
        dens : (N,) array-like of scalar
            Effective density at the location of the inspiraling secondary in units of [g/cm^3].
        velo : (N,) array-like of scalar
            Effective velocity of the inspiraling secondary in units of [cm/s].

        Returns
        -------
        dvdt (N,) np.ndarray of scalar
            Deceleration rate of the secondary object in units of [cm/s^2].

        """
        dvdt = - 2*np.pi * mass_sec_eff * dens * self._coulomb * np.square(NWTG / velo)
        return dvdt

    def _attenuation_BBR1980(self, sepa, m1m2, mstar):
        """Calculate attentuation factor following [BBR1980]_ prescription.

        Characteristic radii are currently calculated using hard-coded Dehnen stellar-density profiles, and a fixed
        scaling-relationship between stellar-mass and stellar characteristic radius.

        The binding radius can be calculated either using the stellar density profile, or from a velocity dispersion,
        based on the `self._rbound_from_density` flag.  See the 'arguments' section of `docs::Dynamical_Friction_NFW`.

        The attenuation factor is defined as >= 1.0, with 1.0 meaning no attenuation.

        Parameters
        ----------
        sepa : (N,) array-like of scalar,
            Binary separations in units of [cm].
        m1m2 : (N, 2) array-like of scalar,
            Masses of each binary component (0-primary, 1-secondary).
        mstar : (N,) array-like of scalar,
            Mass of the stellar-bulge / stellar-core (ambiguous).

        Returns
        -------
        atten : (N,) np.ndarray of scalar
            Attenuation factor (defined as >= 1.0).

        """

        m1, m2 = m1m2.T
        mbh = m1 + m2

        # characteristic stellar radius in [cm]
        rstar = _radius_stellar_characteristic_dabringhausen_2008(mstar)
        # characteristic hardening-radius in [cm]
        rhard = _radius_hard_BBR1980_dehnen(mbh, mstar)
        # characteristic loss-cone-radius in [cm]
        rlc = _radius_loss_cone_BBR1980_dehnen(mbh, mstar)

        # Calculate R-bound based on stellar density profile (mass enclosed)
        if self._rbound_from_density:
            rbnd = _radius_influence_dehnen(mbh, mstar)
        # Calculate R-bound based on uniform velocity dispersion (MBH scaling relation)
        else:
            vdisp = self._msigma.vdisp_from_mbh(m1)   # use primary-bh's mass (index 0)
            rbnd = NWTG * mbh / vdisp**2

        # Number of stars in the stellar bulge/core
        nstar = mstar / (0.6 * MSOL)
        # --- Attenuation for separations less than the hardening radius
        # [BBR1980] Eq.3
        atten_hard = np.maximum((rhard/sepa) * np.log(nstar), np.square(mbh/mstar) * nstar)
        # use an exponential turn-on at larger radii
        cut = np.exp(-sepa/rhard)
        atten_hard *= cut

        # --- Attenuation for separations less than the loss-cone Radius
        # [BBR1980] Eq.2
        atten_lc = np.power(m2/m1, 1.75) * nstar * np.power(rbnd/rstar, 6.75) * (rlc / sepa)
        atten_lc = np.maximum(atten_lc, 1.0)
        # use an exponential turn-on at larger radii
        cut = np.exp(-sepa/rlc)
        atten_hard *= cut

        atten = np.maximum(atten_hard, atten_lc)
        # Make sure that attenuation is always >= 1.0 (i.e. this never _increases_ the hardening rate)
        atten = np.maximum(atten, 1.0)
        return atten


class Fixed_Time(_Hardening):
    """Provide a binary hardening rate such that the total lifetime matches a given value.

    This class uses a phenomenological functional form (defined in :meth:`Fixed_Time.function`) to
    model the hardening rate ($da/dt$) of binaries.  The functional form is,

    .. math::
        \\dot{a} = - A * (1.0 + x)^{-g_2 - 1} / x^{g_1 - 1},

    where :math:`x \\equiv a / r_\\mathrm{char}` is the binary separation scaled to a characteristic
    transition radius (:math:`r_\\mathrm{char}`) between two power-law indices $g_1$ and $g_2$.  There is
    also an overall normalization $A$ that is calculated to yield the desired binary lifetimes.

    NOTE/BUG: the actual binary lifetimes tend to be 1-5% shorter than the requested value.

    The normalization for each binary, to produce the desired lifetime, is calculated as follows:
    (1) A set of random test binary parameters are chosen.
    (2) The normalization constants are determined, using least-squares optimization, to yield the
        desired lifetime.
    (3) Interpolants are constructed to interpolate between the test binary parameters.
    (4) The interpolants are called on the provided binary parameters, to calculate the
        interpolated normalization constants to reach the desired lifetimes.

    Construction/Initialization: note that in addition to the standard :meth:`Fixed_Time.__init__`
    constructor, there are two additional constructors are provided:
    *   :meth:`Fixed_Time.from_pop` - accept a :class:`holodeck.population._Discrete_Population`,
    *   :meth:`Fixed_Time.from_sam` - accept a :class:`holodeck.sam.Semi_Analytic_Model`.

    #! Using a callable for `rchar` probably doesnt work - `_calculate_norm_interpolant` looks like
    #! it only accepts a scalar value.

    """

    # _INTERP_NUM_POINTS = 1e4             #: number of random data points used to construct interpolant
    _INTERP_NUM_POINTS = 1e4
    _INTERP_THRESH_PAD_FACTOR = 5.0      #: allowance for when to use chunking and when to process full array
    _TIME_TOTAL_RMIN = 1.0e-5 * PC       #: minimum radius [cm] used to calculate inspiral time
    _NORM_CHUNK_SIZE = 1e3

    def __init__(self, time, mtot, mrat, redz, sepa,
                 rchar=100.0*PC, gamma_sc=-1.0, gamma_df=+2.5, progress=True, exact=False):
        """Initialize `Fixed_Time` instance for the given binary properties and function parameters.

        Parameters
        ----------
        time : float,  callable  or  array_like
            Total merger time of binaries, units of [sec], specifiable in the following ways:
            *   float : uniform merger time for all binaries
            *   callable : function `time(mtot, mrat, redz)` which returns the total merger time
            *   array_like : (N,) matching the shape of `mtot` (etc) giving the merger time for
                each binary
        mtot : array_like
            Binary total-mass [gram].
        mrat : array_like
            Binary mass-ratio $q \equiv m_2 / m_1 \leq 1$.
        redz : array_like
            Binary Redshift.
            NOTE: this is only used as an argument to callable `rchar` and `time` values.
        sepa : array_like
            Binary semi-major axis (separation) [cm].
        rchar : scalar  or  callable
            Characteristic radius dividing two power-law regimes, in units of [cm]:
            *   scalar : uniform radius for all binaries
            *   callable : function `rchar(mtot, mrat, redz)` which returns the radius
        gamma_sc : scalar
            Power-law of hardening timescale in the stellar-scattering regime,
            (small separations: r < rchar), at times referred to internally as `g1`.
        gamma_df : scalar
            Power-law of hardening timescale in the dynamical-friction regime
            (large separations: r > rchar), at times referred to internally as `g1`.

        """
        self._progress = progress

        # ---- Initialize / Sanitize arguments

        # Ensure `time` is ndarray matching binary variables
        if np.isscalar(time):
            time = time * np.ones_like(mtot)
        elif callable(time):
            time = time(mtot, mrat, redz)
        elif np.shape(time) != np.shape(mtot):
            err = f"Shape of `time` ({np.shape(time)}) does not match `mtot` ({np.shape(mtot)})!"
            log.exception(err)
            raise ValueError(err)

        # `rchar` must be a function of only mtot, mrat; or otherwise a fixed value
        # This is because it is not being used as an interpolation variable, only an external parameter
        # FIX/BUG: either an ndarray could be allowed when interpolation is not needed (i.e. small numbers of systems)
        #      or `rchar` could be added as an explicit interpolation variable
        if callable(rchar):
            log.warning("!!It looks like you're using a callable `rchar`, this probably doesn't work!!")
            rchar = rchar(mtot, mrat, redz)
        elif not np.isscalar(rchar):
            err = "`rchar` must be a scalar or callable: (`rchar(mtot, mrat)`)!"
            log.exception(err)
            raise ValueError(err)

        # ---- Calculate normalization parameter
        mtot, mrat, time, sepa = np.broadcast_arrays(mtot, mrat, time, sepa)
        if mtot.ndim != 1:
            err = f"Error in input shapes (`mtot.shape={np.shape(mtot)})"
            log.exception(err)
            raise ValueError(err)

        # If there are lots of points, construct and use an interpolant
        lots_of_points = self._INTERP_THRESH_PAD_FACTOR * self._INTERP_NUM_POINTS
        log.debug(f"size={len(mtot)} vs. limit={lots_of_points}; `exact`={exact}")
        if (len(mtot) > lots_of_points) and (not exact):
            log.info("constructing hardening normalization interpolant")
            # both are callable as `interp(args)`, with `args` shaped (N, 4),
            # the 4 parameters are:      [log10(M/MSOL), log10(q), time/Gyr, log10(Rmax/PC)]
            # the interpolants return the log10 of the norm values
            interp, backup = self._calculate_norm_interpolant(rchar, gamma_sc, gamma_df)

            points = [np.log10(mtot/MSOL), np.log10(mrat), time/GYR, np.log10(sepa/PC)]
            points = np.array(points)
            norm = interp(points.T)
            bads = ~np.isfinite(norm)
            if np.any(bads):
                msg = f"Normal interpolant failed on {utils.frac_str(bads, 4)} points.  Using backup interpolant"
                log.info(msg)
                bp = points.T[bads]
                # If scipy throws an error on the shape here, see: https://github.com/scipy/scipy/issues/4123
                # or https://stackoverflow.com/a/26806707/230468
                norm[bads] = backup(bp)
                bads = ~np.isfinite(norm)
                if np.any(bads):
                    err = f"Backup interpolant failed on {utils.frac_str(bads, 4)} points!"
                    log.exception(err)
                    raise ValueError(err)

            norm = 10.0 ** norm

        # For small numbers of points, calculate the normalization directly
        else:
            log.info("calculating normalization exactly")
            norm = self._get_norm_chunk(time, mtot, mrat, rchar, gamma_sc, gamma_df, sepa, progress=progress)

        self._gamma_sc = gamma_sc
        self._gamma_df = gamma_df
        self._norm = norm
        self._rchar = rchar
        return

    # ====     Constructors    ====

    @classmethod
    def from_pop(cls, pop, time, **kwargs):
        """Initialize a `Fixed_Time` instance using a provided `_Discrete_Population` instance.

        Parameters
        ----------
        pop : `_Discrete_Population`
            Input population, from which to use masses, redshifts and separations.
        time : float,  callable  or  array_like
            Total merger time of binaries, units of [sec], specifiable in the following ways:
            *   float : uniform merger time for all binaries
            *   callable : function `time(mtot, mrat, redz)` which returns the total merger time
            *   array_like : (N,) matching the shape of `mtot` (etc) giving the merger time for
                each binary
        **kwargs : dict
            Additional keyword-argument pairs passed to the `Fixed_Time` initialization method.

        Returns
        -------
        `Fixed_Time`
            Instance configured for the given binary population.

        """
        return cls(time, *pop.mtmr, pop.redz, pop.sepa, **kwargs)

    @classmethod
    def from_sam(cls, sam, time, sepa_init=1e4*PC, **kwargs):
        """Initialize a `Fixed_Time` instance using a provided `Semi_Analytic_Model` instance.

        Parameters
        ----------
        sam : `holodeck.sam.Semi_Analytic_Model`
            Input population, from which to use masses, redshifts and separations.
        time : float,  callable  or  array_like
            Total merger time of binaries, units of [sec], specifiable in the following ways:
            *   float : uniform merger time for all binaries
            *   callable : function `time(mtot, mrat, redz)` which returns the total merger time
            *   array_like : (N,) matching the shape of `mtot` (etc) giving the merger time for
                each binary
        sepa_init : float  or  array_like
            Initial binary separation.  Units of [cm].
            *   float : initial separation applied to all binaries,
            *   array_like : initial separations for all binaries, shaped (N,) matching the number
                binaries.
        **kwargs : dict
            Additional keyword-argument pairs passed to the `Fixed_Time` initialization method.

        Returns
        -------
        `Fixed_Time`
            Instance configured for the given binary population.

        """
        mtot, mrat, redz = [gg.ravel() for gg in sam.grid]
        return cls(time, mtot, mrat, redz, sepa_init, **kwargs)

    # ====     Hardening Rate Methods    ====

    def dadt_dedt(self, evo, step):
        """Calculate hardening rate at the given integration `step`, for the given population.

        Parameters
        ----------
        evo : `Evolution` instance
            The evolutionary tracks of the binary population, providing binary parameters.
        step : int,
            Integration step at which to calculate hardening rates.

        Returns
        -------
        dadt : (N,) np.ndarray
            Binary hardening rates in units of [cm/s].
        dedt : (N,) np.ndarray or None
            Rate-of-change of eccentricity, which is not included in this calculation, it is zero.
            `None` is returned if the input `eccen` is None.

        """
        mass = evo.mass[:, step, :]
        sepa = evo.sepa[:, step]
        mt, mr = utils.mtmr_from_m1m2(mass)
        dadt, _dedt = self._dadt_dedt(mt, mr, sepa, self._norm, self._rchar, self._gamma_sc, self._gamma_df)
        dedt = None if evo.eccen is None else np.zeros_like(dadt)
        return dadt, dedt

    def dadt(self, mt, mr, sepa):
        dadt, _dedt = self._dadt_dedt(mt, mr, sepa, self._norm, self._rchar, self._gamma_sc, self._gamma_df)
        return dadt

    def dedt(self, mt, mr, sepa):
        _dadt, dedt = self._dadt_dedt(mt, mr, sepa, self._norm, self._rchar, self._gamma_sc, self._gamma_df)
        return dedt

    @classmethod
    def _dadt_dedt(cls, mtot, mrat, sepa, norm, rchar, g1, g2):
        """Calculate hardening rate for the given raw parameters.

        Parameters
        ----------
        mtot : array_like
            Binary total-mass [gram].
        mrat : array_like
            Binary mass-ratio $q \equiv m_2 / m_1 \leq 1$.
        redz : array_like
            Redshift.
        sepa : array_like
            Binary semi-major axis (separation) [cm].
        norm : array_like
            Hardening rate normalization, units of [cm/s].
        rchar : array_like
            Characteristic transition radius between the two power-law indices of the hardening
            rate model, units of [cm].
        g1 : scalar
            Power-law of hardening timescale in the stellar-scattering regime,
            (small separations: r < rchar).
        g2 : scalar
            Power-law of hardening timescale in the dynamical-friction regime,
            (large separations: r > rchar).

        Returns
        -------
        dadt : (N,) np.ndarray
            Binary hardening rates in units of [cm/s].
        dedt : (N,) np.ndarray or None
            Rate-of-change of eccentricity, which is not included in this calculation, it is zero.
            `None` is returned if the input `eccen` is None.

        """
        m1, m2 = utils.m1m2_from_mtmr(mtot, mrat)
        dadt_gw = utils.gw_hardening_rate_dadt(m1, m2, sepa)

        xx = sepa / rchar
        dadt = cls.function(norm, xx, g1, g2)
        dadt = dadt + dadt_gw

        dedt = None
        return dadt, dedt

    # ====     Internal Functions    ====

    @classmethod
    def function(cls, norm, xx, g1, g2):
        """Hardening rate given the parameters for this hardening model.

        The functional form is,
        .. math::
            \\dot{a} = - A * (1.0 + x)^{-g_2 - 1} / x^{g_1 - 1},

        Where $A$ is an overall normalization, and x \\equiv a / r_\\mathrm{char}$ is the binary
        separation scaled to a characteristic transition radius ($r_\\mathrm{char}$) between two
        power-law indices $g_1$ and $g_2$.

        Parameters
        ----------
        norm : array_like
            Hardening rate normalization, units of [cm/s].
        xx : array_like
            Dimensionless binary separation, the semi-major axis in units of the characteristic
            (i.e. transition) radius of the model `rchar`.
        g1 : scalar
            Power-law of hardening timescale in the stellar-scattering regime,
            (small separations: r < rchar).
        g2 : scalar
            Power-law of hardening timescale in the dynamical-friction regime,
            (large separations: r > rchar).

        """
        dadt = - norm * np.power(1.0 + xx, -g2-1) / np.power(xx, g1-1)
        return dadt

    @classmethod
    def _calculate_norm_interpolant(cls, rchar, gamma_sc, gamma_df):
        """Generate interpolants to map from binary parameters to hardening rate normalization.

        Interpolants are calculated as follows:
        (1) A set of random test binary parameters and lifetimes are chosen.
        (2) The normalizations to yield those binary lifetimes are calculated with least-squares
            optimization.
        (3) Interpolants are constructed to yield the normalization paramters for the given
            binary parameters and binary lifetime.

        Two interpolators are returned, a linear-interpolator that is the preferable one, and a
        backup nearest-interplator that is more robust and works at times when the linear
        interpolator fails.

        Parameters
        ----------
        rchar : scalar  or  array_like  #! Possible that only a scalar value is currently working!
            Characteristic radius separating the two power-law regimes, in units of [cm]:
            *   scalar : uniform radius for all binaries
            *   array_like : characteristic radius for each binary.
        gamma_sc : scalar
            Power-law of hardening timescale in the stellar-scattering regime,
            (small separations: r < rchar), at times referred to internally as `g1`.
        gamma_df : scalar
            Power-law of hardening timescale in the dynamical-friction regime
            (large separations: r > rchar), at times referred to internally as `g1`.

        Returns
        -------
        interp : callable
            Linear interpolator from (M, q, t, r) => A
            (total-mass, mass-ratio, lifetime, initial-radius) => hardening normalization
        backup : callable
            Nearest interpolator from (M, q, t, r) => A, to use as a backup when `interp` fails.
            (total-mass, mass-ratio, lifetime, initial-radius) => hardening normalization

        """

        def get_norm_for_random_points(num_points):
            num = int(num_points)

            # ---- Initialization
            # Define the range of parameters to be explored
            mt = [1e5, 1e11]   #: total mass [Msol]
            mr = [1e-5, 1.0]   #: mass ratio
            # td = [0.0, 20.0]   #: lifetime [Gyr]    LINEAR
            td = [1e-3, 20.0]   #: lifetime [Gyr]        LOG
            rm = [1e3, 1e5]    #: radius maximum (initial separation) [pc]

            # Choose random test binary parameters
            mt = 10.0 ** np.random.uniform(*np.log10(mt), num) * MSOL
            mr = 10.0 ** np.random.uniform(*np.log10(mr), num)
            td = np.random.uniform(*td, num+1)[1:] * GYR
            # td = 10.0 ** np.random.uniform(*np.log10(td), num) * GYR
            rm = 10.0 ** np.random.uniform(*np.log10(rm), num) * PC

            # ---- Get normalization for these parameters
            norm = cls._get_norm_chunk(td, mt, mr, rchar, gamma_sc, gamma_df, rm)

            points = [mt, mr, td, rm]

            return norm, points

        def convert_points_to_interp_vals(points):
            units = [MSOL, 1.0, GYR, PC]
            logs = [True, True, False, True]   #: which parameters to interpolate in log-space
            # logs = [True, True, True, True]   #: which parameters to interpolate in log-space
            vals = [pp/uu for pp, uu in zip(points, units)]
            vals = [np.log10(pp) if ll else pp for pp, ll in zip(vals, logs)]
            vals = np.array(vals).T
            return vals

        num_points = int(cls._INTERP_NUM_POINTS)
        norm, points = get_norm_for_random_points(num_points)
        vals = convert_points_to_interp_vals(points)

        # Make sure results are valid
        valid = np.isfinite(norm) & (norm > 0.0)
        if not np.all(valid):
            err = f"Invalid normalizations!  {utils.frac_str(valid, 4)}"
            log.exception(err)
            raise ValueError(err)

        # ---- Construct interpolants

        # construct both a linear (1th order) and nearest (0th order) interpolant
        interp = sp.interpolate.LinearNDInterpolator(vals, np.log10(norm))
        backup = sp.interpolate.NearestNDInterpolator(vals, np.log10(norm))

        '''
        check_norm, check_points = get_norm_for_random_points(100)
        check_vals = convert_points_to_interp_vals(check_points)
        interp_norm = 10.0 ** interp(check_vals)
        backup_norm = 10.0 ** backup(check_vals)
        error_interp = (interp_norm - check_norm) / check_norm
        error_backup = (backup_norm - check_norm) / check_norm

        print("\n")
        print(f"{utils.stats(check_norm)=}")
        print(f"{utils.stats(interp_norm)=}")
        print(f"{utils.stats(backup_norm)=}")
        print(f"{utils.stats(error_interp)=}")
        print(f"{utils.stats(error_backup)=}")
        '''

        return interp, backup

    @classmethod
    def _get_norm_chunk(cls, target_time, *args, progress=True, **kwargs):
        """Calculate normalizations in 'chunks' of the input arrays, to obtain the target lifetime.

        Calculates normalizations for groups of parameters of size `chunk` at a time.  Loops over
        these chunks until all inputs have been processed.  Calls :meth:`Fixed_Time._get_norm` to
        calculate the normalization for each chunk.

        Parameters
        ----------
        target_time : (N,) np.ndarray
            Target binary lifetimes, units of [sec].
        *args : list[np.ndarray]
            The parameters eventually passed to :meth:`Fixed_Time._time_total`, to get the total
            lifetime.  The normalization parameter is varied until the `_time_total` return value
            matches the target input lifetime.
        guess : float
            Initial value of the normalization parameter for the optimization routine to start on.
            Units of [cm/s].
        chunk : float
            Size of each 'chunk' of parameters to process at a time, cast to `int`.
        progress : bool
            Whether or not to show a `tqdm` progress bar while iterating over chunks.

        Returns
        -------
        norm : (N,) np.ndarray
            The normalizations required to produce the target lifetimes given by `target_time`.

        """
        if np.ndim(target_time) not in [0, 1]:
            raise

        chunk_size = int(cls._NORM_CHUNK_SIZE)
        size = np.size(target_time)
        # if number of array elements is less than (or comparable to) chunk size, to it all in one pass
        if size <= chunk_size * cls._INTERP_THRESH_PAD_FACTOR:
            return cls._get_norm(target_time, *args, **kwargs)

        # broadcast arrays to all be the same shape (some `floats` are passed in)
        args = [target_time, *args]
        target_time, *args = np.broadcast_arrays(*args)

        # iterate over each chunk, storing the normalization values
        num = int(np.ceil(size / chunk_size))
        norm = np.zeros_like(target_time)
        step_iter = range(num)
        step_iter = utils.tqdm(step_iter, desc='calculating hardening normalization') if progress else step_iter
        for ii in step_iter:
            lo = ii * chunk_size
            hi = np.minimum((ii + 1) * chunk_size, size)
            cut = slice(lo, hi)
            # calculate normalizations for this chunk
            norm[cut] = cls._get_norm(target_time[cut], *[aa[cut] for aa in args], **kwargs)

        return norm

    @classmethod
    def _get_norm(cls, target_time, *args, guess=1e7):
        """Calculate normalizations of the input arrays, to obtain the target binary lifetime.

        Uses deterministic least-squares optimization to find the best normalization values, using
        `scipy.optimize.newton`.

        Parameters
        ----------
        target_time : (N,) np.ndarray
            Target binary lifetimes, units of [sec].
        *args : list[np.ndarray]
            The parameters eventually passed to :meth:`Fixed_Time._time_total`, to get the total
            lifetime.  The normalization parameter is varied until the `_time_total` return value
            matches the target input lifetime.
        guess : float
            Initial value of the normalization parameter for the optimization routine to start on.
            Units of [cm/s].

        Returns
        -------
        norm : (N,) np.ndarray
            The normalizations required to produce the target lifetimes given by `target_time`.

        """

        # convenience wrapper function
        def integ(norm):
            return cls._time_total(norm, *args)

        # Assume linear scaling to refine the first guess
        g0 = guess * np.ones_like(target_time)
        test = integ(g0)
        g1 = g0 * (test / target_time)
        log.debug(f"Guess {guess:.4e} ==> {utils.stats(g1)}")

        # perform optimization
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            norm = sp.optimize.newton(lambda xx: integ(xx) - target_time, g1)

        return norm

    @classmethod
    def _time_total(cls, norm, mt, mr, rchar, gamma_sc, gamma_df, rmax, num=100):
        """For the given parameters, integrate the binary evolution to find total lifetime.

        Parameters
        ----------
        norm : float  or  array_like
            Hardening rate normalization, units of [cm/s].
        mtot : float  or  array_like
            Binary total-mass [gram].
        mrat : float  or  array_like
            Binary mass-ratio $q \equiv m_2 / m_1 \leq 1$.
        rchar : float  or  array_like
            Characteristic transition radius between the two power-law indices of the hardening
            rate model, units of [cm].
        gamma_sc : float  or  array_like
            Power-law of hardening timescale in the stellar-scattering regime,
            (small separations: r < rchar).
        gamma_df : float  or  array_like
            Power-law of hardening timescale in the dynamical-friction regime,
            (large separations: r > rchar).
        rmax : float  or  array_like
            Initial binary separation.  Units of [cm].
        num : int
            Number of steps in separation overwhich to integrate the binary evolution

        Returns
        -------
        tt : np.ndarray
            Total binary lifetime [sec].

        """

        # Convert input values to broadcastable np.ndarray's
        norm = np.atleast_1d(norm)
        args = [norm, mt, mr, rchar, gamma_sc, gamma_df, rmax, cls._TIME_TOTAL_RMIN]
        args = np.broadcast_arrays(*args)
        norm, mt, mr, rchar, gamma_sc, gamma_df, rmax, rmin = args

        # define separations (radii) for each binary's evolution
        rextr = np.log10([rmin, rmax]).T
        rads = np.linspace(0.0, 1.0, num)[np.newaxis, :]
        rads = rextr[:, 0, np.newaxis] + rads * np.diff(rextr, axis=1)
        # (N, R) for N-binaries and R-radii (`num`)
        rads = 10.0 ** rads

        # Make hardening parameters broadcastable
        args = [norm, mt, mr, rchar, gamma_sc, gamma_df]
        args = [aa[:, np.newaxis] for aa in args]
        norm, mt, mr, rchar, gamma_sc, gamma_df = args

        # Calculate hardening rates along full evolutionary history
        dadt, _ = cls._dadt_dedt(mt, mr, rads, norm, rchar, gamma_sc, gamma_df)

        # Integrate (inverse) hardening rates to calculate total lifetime
        tt = utils.trapz_loglog(- 1.0 / dadt, rads, axis=-1)
        tt = tt[:, -1]
        return tt


# =================================================================================================
# ====    Utility Classes and Functions    ====
# =================================================================================================


class _Quinlan1996:
    """Hardening rates from stellar scattering parametrized as in [Quinlan1996]_.

    Fits from scattering experiments must be provided as `hparam` and `kparam`.

    """

    @staticmethod
    def dadt(sepa, rho, sigma, hparam):
        """Binary hardening rate from stellar scattering.

        [Sesana2010]_ Eq.8

        Parameters
        ----------
        sepa : (N,) array-like of scalar
            Binary separation in units of [cm].
        rho : (N,) array-like of scalar
            Effective stellar-density at binary separation in units of [g/cm^3].
        sigma : (N,) array-like of scalar
            Stellar velocity-dispersion at binary separation in units of [cm/s].
        hparam : (N,) array-like of scalar
            Binary hardening efficiency parameter "H" (unitless).

        Returns
        -------
        rv : (N,) np.ndarray of scalar
            Binary hardening rate in units of [cm/s].

        """
        rv = - (sepa ** 2) * NWTG * rho * hparam / sigma
        return rv

    @staticmethod
    def dedt(sepa, rho, sigma, hparam, kparam):
        """Binary rate-of-change of eccentricity from stellar scattering.

        [Sesana2010]_ Eq.9

        Parameters
        ----------
        sepa : (N,) array-like of scalar
            Binary separation in units of [cm].
        rho : (N,) array-like of scalar
            Effective stellar-density at binary separation in units of [g/cm^3].
        sigma : (N,) array-like of scalar
            Stellar velocity-dispersion at binary separation in units of [cm/s].
        hparam : (N,) array-like of scalar
            Binary hardening efficiency parameter "H" (unitless).
        kparam : (N,) array-like of scalar
            Binary eccentricity-change efficiency parameter "K" (unitless).

        Returns
        -------
        rv : (N,) np.ndarray of scalar
            Change of eccentricity rate in units of [1/s].

        """
        rv = sepa * NWTG * rho * hparam * kparam / sigma
        return rv

    @staticmethod
    def radius_hardening(msec, sigma):
        """
        [Sesana2010]_ Eq. 10
        """
        rv = NWTG * msec / (4 * sigma**2)
        return rv


class _SHM06:
    """Fits to stellar-scattering hardening rates from [Sesana2006]_, based on the [Quinlan1996]_ formalism.

    Parameters describe the efficiency of hardening as a function of mass-ratio (`mrat`) and separation (`sepa`).

    """

    def __init__(self):
        self._bound_H = [0.0, 40.0]    # See [Sesana2006]_ Fig.3
        self._bound_K = [0.0, 0.4]     # See [Sesana2006]_ Fig.4

        # Get the data filename
        fname = os.path.join(_PATH_DATA, _SCATTERING_DATA_FILENAME)
        if not os.path.isfile(fname):
            err = f"file ({fname}) not does exist!"
            log.error(err)
            raise FileNotFoundError(err)

        # Load Data
        data = json.load(open(fname, 'r'))
        self._data = data['SHM06']
        # 'H' : Hardening Rate
        self._init_h()
        # 'K' : Eccentricity growth
        self._init_k()
        return

    def H(self, mrat, sepa_rhard):
        """Hardening rate efficiency parameter.

        Parameters
        ----------
        mrat : (N,) array-like of scalar
            Binary mass-ratio (q = M2/M1 <= 1.0).
        sepa_rhard : (N,) array-like of scalar
            Binary separation in *units of hardening radius (r_h)*.

        Returns
        -------
        hh : (N,) np.ndarray of scalar
            Hardening parameter.

        """
        xx = sepa_rhard / self._H_a0(mrat)
        hh = self._H_A(mrat) * np.power(1.0 + xx, self._H_g(mrat))
        hh = np.clip(hh, *self._bound_H)
        return hh

    def K(self, mrat, sepa_rhard, ecc):
        """Eccentricity hardening rate efficiency parameter.

        Parameters
        ----------
        mrat : (N,) array-like of scalar
            Binary mass-ratio (q = M2/M1 <= 1.0).
        sepa_rhard : (N,) array-like of scalar
            Binary separation in *units of hardening radius (r_h)*.
        ecc : (N,) array-like of scalar
            Binary eccentricity.

        Returns
        -------
        kk : (N,) np.ndarray of scalar
            Eccentricity change parameter.

        """
        use_a = (sepa_rhard / self._K_a0(mrat, ecc))
        A = self._K_A(mrat, ecc)
        g = self._K_g(mrat, ecc)
        B = self._K_B(mrat, ecc)

        # `interp2d` return a matrix of X x Y results... want diagonal of that
        # NOTE: FIX: this could be improved!!
        use_a = use_a.diagonal()
        A = A.diagonal()
        g = g.diagonal()
        B = B.diagonal()

        kk = A * np.power((1 + use_a), g) + B
        kk = np.clip(kk, *self._bound_K)
        return kk

    def _init_k(self):
        """Initialize and store the interpolants for calculating the K parameter.
        """
        data = self._data['K']
        # Get all of the mass ratios (ignore other keys)
        _kq_keys = list(data.keys())
        kq_keys = []
        for kq in _kq_keys:
            try:
                int(kq)
                kq_keys.append(kq)
            except (TypeError, ValueError):
                pass

        nq = len(kq_keys)
        if nq < 2:
            raise ValueError("Something is wrong... `kq_keys` = '{}'\ndata:\n{}".format(kq_keys, data))
        k_mass_ratios = 1.0/np.array(sorted([int(kq) for kq in kq_keys]))
        k_eccen = np.array(data[kq_keys[0]]['e'])
        ne = len(k_eccen)
        k_A = np.zeros((ne, nq))
        k_a0 = np.zeros((ne, nq))
        k_g = np.zeros((ne, nq))
        k_B = np.zeros((ne, nq))

        for ii, kq in enumerate(kq_keys):
            _dat = data[kq]
            k_A[:, ii] = _dat['A']
            k_a0[:, ii] = _dat['a0']
            k_g[:, ii] = _dat['g']
            k_B[:, ii] = _dat['B']

        self._K_A = sp.interpolate.interp2d(k_mass_ratios, k_eccen, k_A, kind='linear')
        self._K_a0 = sp.interpolate.interp2d(k_mass_ratios, k_eccen, k_a0, kind='linear')
        self._K_g = sp.interpolate.interp2d(k_mass_ratios, k_eccen, k_g, kind='linear')
        self._K_B = sp.interpolate.interp2d(k_mass_ratios, k_eccen, k_B, kind='linear')
        return

    def _init_h(self):
        """Initialize and store the interpolants for calculating the H parameter.
        """
        _dat = self._data['H']
        h_mass_ratios = 1.0/np.array(_dat['q'])
        h_A = np.array(_dat['A'])
        h_a0 = np.array(_dat['a0'])
        h_g = np.array(_dat['g'])

        self._H_A = sp.interpolate.interp1d(h_mass_ratios, h_A, kind='linear', fill_value='extrapolate')
        self._H_a0 = sp.interpolate.interp1d(h_mass_ratios, h_a0, kind='linear', fill_value='extrapolate')
        self._H_g = sp.interpolate.interp1d(h_mass_ratios, h_g, kind='linear', fill_value='extrapolate')
        return


def _radius_stellar_characteristic_dabringhausen_2008(mstar, gamma=1.0):
    """Characteristic stellar radius based on total stellar mass.

    [Chen2017]_ Eq.27 - from [Dabringhausen+2008]
    """
    rchar = 239 * PC * (np.power(2.0, 1.0/(3.0 - gamma)) - 1.0)
    rchar *= np.power(mstar / (1e9*MSOL), 0.596)
    return rchar


def _radius_influence_dehnen(mbh, mstar, gamma=1.0):
    """Characteristic radius of influence based on a Dehnen density profile.

    [Chen2017]_ Eq.25
    """
    rchar = _radius_stellar_characteristic_dabringhausen_2008(mstar, gamma)
    rinfl = np.power(2*mbh/mstar, 1.0/(gamma - 3.0))
    rinfl = rchar / (rinfl - 1.0)
    return rinfl


def _density_at_influence_radius_dehnen(mbh, mstar, gamma=1.0):
    """Density at the characteristic influence radius, based on a Dehnen density profile.
    [Chen2017]_ Eq.26
    """
    # [Chen2017] Eq.27 - from [Dabringhausen+2008]
    rchar = _radius_stellar_characteristic_dabringhausen_2008(mstar, gamma)
    dens = mstar * (3.0 - gamma) / np.power(rchar, 3.0) / (4.0 * np.pi)
    dens *= np.power(2*mbh / mstar, gamma / (gamma - 3.0))
    return dens


def _radius_hard_BBR1980_dehnen(mbh, mstar, gamma=1.0):
    """Characteristic 'hardened' radius from [BBR1980]_, assuming a Dehnen stellar density profile.

    [Kelley2017a]_ paragraph below Eq.8 - from [BBR1980]_
    """
    rbnd = _radius_influence_dehnen(mbh, mstar, gamma=gamma)
    rstar = _radius_stellar_characteristic_dabringhausen_2008(mstar, gamma)
    rhard = rstar * (rbnd/rstar) ** 3
    return rhard


def _radius_loss_cone_BBR1980_dehnen(mbh, mstar, gamma=1.0):
    """Characteristic 'loss-cone' radius from [BBR1980]_, assuming a Dehnen stellar density profile.

    [Kelley2017a]_ Eq.9 - from [BBR1980]_
    """
    mass_of_a_star = 0.6 * MSOL
    rbnd = _radius_influence_dehnen(mbh, mstar, gamma=gamma)
    rstar = _radius_stellar_characteristic_dabringhausen_2008(mstar, gamma)
    rlc = np.power(mass_of_a_star / mbh, 0.25) * np.power(rbnd/rstar, 2.25) * rstar
    return rlc
