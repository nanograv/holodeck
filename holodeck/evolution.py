"""
Holodeck - evolution submodule.

To-Do
-----
*   General
    -   [ ] evolution modifiers should act at each step, instead of after all steps?  This would be a way to implement a
        changing accretion rate, for example; or to set a max/min hardening rate.
    -   [ ] re-implement "magic" hardening models that coalesce in zero change-of-redshift or fixed amounts of time

*   Dynamical_Friction_NFW
    -   [ ] Allow stellar-density profiles to also be specified (instead of using a hard-coded Dehnan profile)
    -   [ ] Generalize calculation of stellar characteristic radius.  Make self-consistent with stellar-profile,
      and user-specifiable.

References
----------
*   [Quinlan1996] :: Quinlan 1996
    The dynamical evolution of massive black hole binaries I. Hardening in a fixed stellar background
    https://ui.adsabs.harvard.edu/abs/1996NewA....1...35Q/abstract
*   [SHM06] :: Sesana, Haardt & Madau et al. 2006
    Interaction of Massive Black Hole Binaries with Their Stellar Environment. I. Ejection of Hypervelocity Stars
    https://ui.adsabs.harvard.edu/abs/2006ApJ...651..392S/abstract
*   [Sesana10] :: Sesana 2010
    Self Consistent Model for the Evolution of Eccentric Massive Black Hole Binaries in Stellar Environments:
    Implications for Gravitational Wave Observations
    https://ui.adsabs.harvard.edu/abs/2010ApJ...719..851S/abstract
*   [Kelley17] : Kelley, Blecha & Hernquist 2017
    Massive black hole binary mergers in dynamical galactic environments
    https://ui.adsabs.harvard.edu/abs/2017MNRAS.464.3131K/abstract
*   [Chen17] : Chen, Sesana, & Del Pozzo 2017
    Efficient computation of the gravitational wave spectrum emitted by eccentric massive black hole binaries
    in stellar environments
    https://ui.adsabs.harvard.edu/abs/2017MNRAS.470.1738C/abstract

"""

import abc
import enum
import inspect
import json
import os

import numpy as np
import scipy as sp
import scipy.interpolate   # noqa

import holodeck
from holodeck import utils, cosmo, log, _PATH_DATA
from holodeck.constants import GYR, NWTG, PC, MSOL, YR

_DEF_TIME_DELAY = (5.0*GYR, 0.2)
_SCATTERING_DATA_FILENAME = "SHM06_scattering_experiments.json"


@enum.unique
class EVO(enum.Enum):
    CONT = 1
    END = -1


class Evolution:

    _EVO_PARS = ['mass', 'sepa', 'eccen', 'scafa', 'dadt', 'tlbk']
    _LIN_INTERP_PARS = ['eccen', 'scafa', 'tlbk']
    _SELF_CONSISTENT = None
    _STORE_FROM_POP = ['_sample_volume']

    def __init__(self, pop, hard, nsteps=100, mods=None, debug=False):
        self._pop = pop
        self._debug = debug
        self._nsteps = nsteps
        self._mods = mods

        if np.isscalar(hard):
            hard = [hard, ]
        self._hard = hard

        for par in self._STORE_FROM_POP:
            setattr(self, par, getattr(pop, par))

        size = pop.size
        shape = (size, nsteps)

        self._shape = shape
        self.scafa = np.zeros(shape)
        self.tlbk = np.zeros(shape)
        self.sepa = np.zeros(shape)
        self.mass = np.zeros(shape + (2,))
        self.mdot = np.zeros(shape + (2,))

        if pop.eccen is not None:
            self.eccen = np.zeros(shape)
            self.dedt = np.zeros(shape)
        else:
            self.eccen = None
            self.dedt = None

        # NOTE: these values should be stored as positive values
        self.dadt = np.zeros(shape)
        self._dadt_0 = None

        # Derived parameters
        self._freq_orb_rest = None
        self._evolved = False
        self._coal = None

        return

    @property
    def shape(self):
        return self._shape

    def evolve(self):
        # ---- Initialize Integration Step Zero
        self._init_step_zero()

        # ---- Iterate through all integration steps
        size, nsteps = self.shape
        steps_list = range(1, nsteps)
        for step in utils.tqdm(steps_list):
            rv = self._take_next_step(step)
            if rv is EVO.END:
                break
            elif rv not in EVO:
                raise ValueError("Recieved bad `rv` ({}) after step {}!".format(rv, step))

        # ---- Finalize
        self._finalize()
        return

    @property
    def coal(self):
        if self._coal is None:
            self._coal = (self.redz[:, -1] > 0.0)
        return self._coal

    def _init_step_zero(self):
        pop = self._pop
        _, nsteps = self.shape

        # Initialize ALL separations ranging from initial to mutual-ISCO, for each binary
        rad_isco = utils.rad_isco(*pop.mass.T)
        # (2, N)
        sepa = np.log10([pop.sepa, rad_isco])
        # Get log-space range of separations for each of N ==> (N, nsteps)
        sepa = np.apply_along_axis(lambda xx: np.logspace(*xx, nsteps), 0, sepa).T
        self.sepa[:, :] = sepa

        self.scafa[:, 0] = pop.scafa
        redz = cosmo.a_to_z(pop.scafa)
        tlbk = cosmo.z_to_tlbk(redz)
        self.tlbk[:, 0] = tlbk
        self.mass[:, 0, :] = pop.mass
        self.mass[:, :, :] = self.mass[:, 0, np.newaxis, :]

        if (pop.eccen is not None):
            self.eccen[:, 0] = pop.eccen

        return

    def _take_next_step(self, step):
        """Integrate the binary population forward (to smaller separations) by one step, to reach step `step`.
        """
        debug = self._debug
        size, nsteps = self.shape
        left = step - 1     # the previous time-step (already completed)
        right = step        # the next     time-step

        # initialize storage for hardening rates in separation and eccentricity (if enabled)
        dadt_1 = np.zeros(size)
        if self.eccen is not None:
            deccdt_1 = np.zeros_like(dadt_1)

        # addional record-keeping / diagnostics for debugging
        if debug and (self._dadt_0 is None):
            # initialzie variables to store da/dt hardening rates from each hardening component
            for ii in range(len(self._hard)):
                name = f"_dadt_{ii}"
                setattr(self, name, np.zeros_like(self.dadt))

        # Get hardening rates for each hardening model
        for ii, hard in enumerate(self._hard):
            _a1, _e1 = hard.dadt_dedt(self, left)
            if debug:
                log.debug(f"hard={hard} : dadt = {utils.stats(_a1)}")
                if not np.all(np.isfinite(_a1)):
                    err = f"non-finite `dadt` for hard={hard}!"
                    log.error(err)
                    raise ValueError(err)

                if np.any(_a1 > 0.0):
                    err = f"found positive hardening rates for hard={hard}!"
                    log.error(err)
                    raise ValueError(err)

                # Store individual hardening rates
                name = f"_dadt_{ii}"
                getattr(self, name)[:, left] = _a1[...]

            dadt_1[:] += _a1
            if self.eccen is not None:
                deccdt_1[:] += _e1

        # Calculate time between edges
        dadt = dadt_1
        self.dadt[:, left] = dadt
        # NOTE: `da` defined to be negative!  `dadt` is also negative
        da = self.sepa[:, right] - self.sepa[:, left]
        dt = da / dadt
        if np.any(dt < 0.0):
            err = f"Negative time-steps found at step={step}!"
            log.error(err)
            raise ValueError(err)

        # Update lookback time based on duration of this step
        tlbk = self.tlbk[:, left] - dt
        self.tlbk[:, right] = tlbk
        # update scale-factor for systems at z > 0.0 (i.e. a < 1.0 and tlbk > 0.0)
        val = (tlbk > 0.0)
        self.scafa[val, right] = cosmo.z_to_a(cosmo.tlbk_to_z(tlbk[val]))
        # set systems after z = 0 to scale-factor of unity
        self.scafa[~val, right] = 1.0
        if debug:
            log.debug(f"{step=:4d}")
            log.debug(
                f"\ta      = {utils.stats(self.sepa[:, left])}\n"
                f"\tda     = {utils.stats(da)}\n"
                f"\tdadt   = {utils.stats(dadt)}\n"
                f"\ta/dadt = {utils.stats(self.sepa[:, left]/dadt)}"
                f"\tdt     = {utils.stats(dt)}"
            )

        # update eccentricity if it's being evolved
        if self.eccen is not None:
            deccdt = deccdt_1
            decc = deccdt * dt
            self.eccen[:, right] = self.eccen[:, left] + decc

        return EVO.CONT

    def _check(self):
        _check_var_names = ['sepa', 'scafa', 'mass', 'tlbk', 'dadt']
        _check_var_names_eccen = ['eccen', 'dedt']

        def check_vars(names):
            for cv in names:
                vals = getattr(self, cv)
                if np.any(~np.isfinite(vals)):
                    err = "Found non-finite '{}' !".format(cv)
                    raise ValueError(err)

        check_vars(_check_var_names)

        if self.eccen is None:
            return

        check_vars(_check_var_names_eccen)

        return

    def _finalize(self):
        self._evolved = True

        self.modify()

        # Run diagnostics
        self._check()

        return

    def modify(self, mods=None):
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

    def _update_derived(self):
        pass

    @property
    def freq_orb_rest(self):
        if self._freq_orb_rest is None:
            self._check_evolved()
            mtot = self.mass.sum(axis=-1)
            self._freq_orb_rest = utils.kepler_freq_from_sep(mtot, self.sepa)
        return self._freq_orb_rest

    @property
    def freq_orb_obs(self):
        redz = cosmo.a_to_z(self.scafa)
        fobs = self.freq_orb_rest / (1.0 + redz)
        return fobs

    def _check_evolved(self):
        if self._evolved is not True:
            raise RuntimeError("This instance has not been evolved yet!")

        return

    def at(self, xpar, targets, pars=None, coal=False, lin_interp=None):
        """Interpolate evolution to the given observed, orbital frequency.

        Arguments
        ---------
        xpar : str, one of ['fobs', 'sepa']
            String specifying the variable to interpolate to.
        targets : array of scalar
            Locations to interpolate to.
            `sepa` : units of cm
            `fobs` : units of 1/s [Hz]

        Out of bounds values are set to `np.nan`.
        Interpolation is 1st-order in log-log space.

        """
        self._check_evolved()
        _allowed = ['sepa', 'fobs']
        if xpar not in _allowed:
            raise ValueError("`xpar` must be one of '{}'!".format(_allowed))

        if pars is None:
            pars = self._EVO_PARS
        if np.isscalar(pars):
            pars = [pars]
        squeeze = False
        if np.isscalar(targets):
            targets = np.atleast_1d(targets)
            squeeze = True

        size, nsteps = self.shape
        # Observed-Frequency, units of 1/yr
        if xpar == 'fobs':
            # frequency is already increasing
            _xvals = np.log10(self.freq_orb_obs)
            scafa = self.scafa[:, :]
            tt = np.log10(targets)
            rev = False
        # Binary-Separation, units of pc
        elif xpar == 'sepa':
            # separation is decreasing, reverse to increasing
            _xvals = np.log10(self.sepa)[:, ::-1]
            scafa = self.scafa[:, ::-1]
            tt = np.log10(targets)
            rev = True
        else:
            raise ValueError("Bad `xpar` {}!".format(xpar))

        # Find the evolution-steps immediately before and after the target frequency
        textr = utils.minmax(tt)
        xextr = utils.minmax(_xvals)
        if (textr[1] < xextr[0]) | (textr[0] > xextr[1]):
            err = "`targets` extrema ({}) ourside `xvals` extema ({})!  Bad units?".format(
                (10.0**textr), (10.0**xextr))
            raise ValueError(err)

        # Convert to (N, T, M)
        #     `tt` is (T,)  `xvals` is (N, M) for N-binaries and M-steps
        select = (tt[np.newaxis, :, np.newaxis] <= _xvals[:, np.newaxis, :])
        # Select only binaries that coalesce before redshift zero (a=1.0)
        if coal:
            select = select & (scafa[:, np.newaxis, :] > 0.0) & (scafa[:, np.newaxis, :] < 1.0)

        # (N, T), find the index of the xvalue following each target point (T,), for each binary (N,)
        aft = np.argmax(select, axis=-1)
        # zero values in `aft` mean no xvals after the targets were found
        valid = (aft > 0)
        bef = np.copy(aft)
        bef[valid] -= 1

        # (2, N, T)
        cut = np.array([aft, bef])
        # Valid binaries must be valid at both `bef` and `aft` indices
        for cc in cut:
            valid = valid & np.isfinite(np.take_along_axis(scafa, cc, axis=-1))

        inval = ~valid
        # Get the x-values before and after the target locations  (2, N, T)
        xvals = [np.take_along_axis(_xvals, cc, axis=-1) for cc in cut]
        # Find how far to interpolate between values (in log-space)
        #     (N, T)
        frac = (tt[np.newaxis, :] - xvals[1]) / np.subtract(*xvals)

        vals = dict()
        # Interpolate each target parameter
        for pp in pars:
            lin_interp_flag = (pp in self._LIN_INTERP_PARS) if lin_interp is None else lin_interp
            # Either (N, M) or (N, M, 2)
            _data = getattr(self, pp)
            if _data is None:
                vals[pp] = None
                continue

            reshape = False
            use_cut = cut
            use_frac = frac
            # Sometimes there is a third dimension for the 2 binaries (e.g. `mass`)
            #    which will have shape, (N, T, 2) --- calling this "double-data"
            if np.ndim(_data) != 2:
                if (np.ndim(_data) == 3) and (np.shape(_data)[-1] == 2):
                    # Keep the interpolation axis last (N, T, 2) ==> (N, 2, T)
                    _data = np.moveaxis(_data, -1, -2)
                    # Expand other arrays appropriately
                    use_cut = cut[:, :, np.newaxis]
                    use_frac = frac[:, np.newaxis, :]
                    reshape = True
                else:
                    raise ValueError("Unexpected shape of data: {}!".format(np.shape(_data)))

            if not lin_interp_flag:
                _data = np.log10(_data)

            if rev:
                _data = _data[..., ::-1]
            # (2, N, T) for scalar data or (2, N, 2, T) for "double-data"
            data = [np.take_along_axis(_data, cc, axis=-1) for cc in use_cut]
            # Interpolate by `frac` for each binary   (N, T) or (N, 2, T) for "double-data"
            new = data[1] + (np.subtract(*data) * use_frac)
            # In the "double-data" case, move the doublet back to the last dimension
            #    (N, T) or (N, T, 2)
            if reshape:
                new = np.moveaxis(new, 1, -1)
            # Set invalid binaries to nan
            new[inval, ...] = np.nan

            if np.any(~np.isfinite(new[valid, ...])):
                raise ValueError("Non-finite values after interpolation of '{}'".format(pp))

            # fill return dictionary
            if not lin_interp_flag:
                new = 10.0 ** new
            if squeeze:
                new = new.squeeze()
            vals[pp] = new

            if np.any(~np.isfinite(new[valid, ...])):
                raise ValueError("Non-finite values after exponentiation of '{}'".format(pp))

        return vals


'''
class Evo_Magic_Delay_Circ(_Evolution):

    _SELF_CONSISTENT = False

    def __init__(self, pop, *args, time_delay=None, **kwargs):
        # if pop.eccen is not None:
        #     raise ValueError("Cannot use {} on eccentric population!".format(self.__class__))
        super().__init__(pop, *args, **kwargs)
        self._time_delay = utils._parse_log_norm_pars(time_delay, self.shape[0], default=_DEF_TIME_DELAY)
        return

    def _init_step_zero(self):
        super()._init_step_zero()
        nbins, nsteps = self.shape
        mass = self.mass[:, 0, :]
        tlbk = self.tlbk[:, 0]
        sepa = self.sepa
        m1, m2 = mass.T

        # Time delay (N,)
        dtime = self._time_delay
        # Instantly step-forward this amount, discontinuously
        tlbk = np.clip(tlbk - dtime, 0.0, None)
        self.tlbk[:, 1:] = tlbk[:, np.newaxis]
        # calculate new scalefactors
        redz = cosmo.tlbk_to_z(tlbk)
        self.scafa[:, 1:] = cosmo.z_to_a(redz)[:, np.newaxis]

        # Masses don't evolve
        self.mass[:, :, :] = mass[:, np.newaxis, :]
        self.dadt[...] = - utils.gw_hardening_rate_dadt(m1[:, np.newaxis], m2[:, np.newaxis], sepa)
        return

    def _take_next_step(self, step):
        return EVO.END


class Evo_Magic_Delay_Eccen(_Evolution):

    _SELF_CONSISTENT = False

    def __init__(self, pop, *args, time_delay=None, **kwargs):
        super().__init__(pop, *args, **kwargs)
        self._time_delay = utils._parse_log_norm_pars(time_delay, self.shape[0], default=_DEF_TIME_DELAY)
        return

    def _init_step_zero(self):
        super()._init_step_zero()
        nbins, nsteps = self.shape
        mass = self.mass[:, 0, :]
        m1, m2 = mass.T
        tlbk = self.tlbk[:, 0]
        sepa = self.sepa[:, 0]
        eccen = self.eccen
        if eccen is not None:
            eccen = eccen[:, 0]

        # Time delay (N,)
        dtime = self._time_delay
        # Instantly step-forward this amount, discontinuously
        tlbk = np.clip(tlbk - dtime, 0.0, None)
        self.tlbk[:, 1:] = tlbk[:, np.newaxis]
        # calculate new scalefactors
        redz = cosmo.tlbk_to_z(tlbk)
        self.scafa[:, 1:] = cosmo.z_to_a(redz)[:, np.newaxis]

        # Masses don't evolve
        self.mass[:, :, :] = mass[:, np.newaxis, :]
        self.dadt[:, 0] = - utils.gw_hardening_rate_dadt(m1, m2, sepa, eccen=eccen)
        if eccen is not None:
            self.dedt[:, 0] = - utils.gw_dedt(m1, m2, sepa, eccen)
        return

    def _take_next_step(self, step):
        size, nsteps = self.shape

        # Get values at left-edge of step
        m1, m2 = self.mass[:, step-1, :].T
        a0 = self.sepa[:, step-1]
        a1 = self.sepa[:, step]
        # NOTE: `da` defined to be positive!
        da = a0 - a1
        e0 = self.eccen
        if e0 is not None:
            e0 = e0[:, step-1]

        if e0 is not None:
            dade = utils.gw_dade(m1, m2, a0, e0)
            de = da / dade
            e1 = e0 - de
        else:
            e1 = None

        dadt_1 = - utils.gw_hardening_rate_dadt(m1, m2, a1, eccen=e1)
        if e1 is not None:
            dedt_1 = - utils.gw_dedt(m1, m2, a1, e1)

        # Store
        self.dadt[:, step] = dadt_1
        if e1 is not None:
            self.dedt[:, step] = dedt_1
            self.eccen[:, step] = e1

        return EVO.CONT
'''


class _Hardening(abc.ABC):
    """Abstract class for binary-hardening models.  Subclasses must define `dadt_dedt(evo, step, ...)` method.
    """

    @abc.abstractmethod
    def dadt_dedt(self, evo, step, *args, **kwargs):
        pass


class Hard_GW(_Hardening):
    """Gravitational-wave driven binary hardening.
    """

    @staticmethod
    def dadt_dedt(evo, step):
        m1, m2 = evo.mass[:, step, :].T    # (Binaries, Steps, 2) ==> (2, Binaries)
        sepa = evo.sepa[:, step]
        eccen = evo.eccen[:, step] if (evo.eccen is not None) else None
        dadt = utils.gw_hardening_rate_dadt(m1, m2, sepa, eccen=eccen)

        if eccen is None:
            dedt = None
        else:
            dedt = utils.gw_dedt(m1, m2, evo.sepa, evo.eccen)

        return dadt, dedt


class Quinlan1996:
    """Hardening rates from stellar scattering parametrized as in [Quinlan1996].

    Fits from scattering experiments must be provided as `hparam` and `kparam`.

    """

    @staticmethod
    def dadt(sepa, rho, sigma, hparam):
        """
        [Sesana10] Eq.8
        """
        rv = - (sepa ** 2) * NWTG * rho * hparam / sigma
        return rv

    @staticmethod
    def dedt(sepa, rho, sigma, hparam, kparam):
        """
        [Sesana10] Eq.9
        """
        rv = sepa * NWTG * rho * hparam * kparam / sigma
        return rv

    @staticmethod
    def radius_hardening(msec, sigma):
        """
        [Sesana10] Eq. 10
        """
        rv = NWTG * msec / (4 * sigma**2)
        return rv


class SHM06:
    """Fits to stellar-scattering hardening rates from [SHM06], based on the [Quinlan1996] formalism.

    Parameters describe the efficiency of hardening as a function of mass-ratio (`mrat`) and separation (`sepa`).

    """

    def __init__(self):
        self._bound_H = [0.0, 40.0]    # See [SHM06] Fig.3
        self._bound_K = [0.0, 0.4]     # See [SHM06] Fig.4

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

    def H(self, mrat, sepa):
        """

        Arguments
        ---------
        sepa : binary separation in units of hardening radius (r_h)

        """
        xx = sepa / (PC * self._H_a0(mrat))
        hh = self._H_A(mrat) * np.power(1.0 + xx, self._H_g(mrat))
        hh = np.clip(hh, *self._bound_H)
        return hh

    def K(self, mrat, sepa, ecc):
        """

        Arguments
        ---------
        sepa : binary separation in units of hardening radius (r_h)

        """        # `interp2d` return a matrix of X x Y results... want diagonal of that
        use_a = (sepa/self._K_a0(mrat, ecc))
        A = self._K_A(mrat, ecc)
        g = self._K_g(mrat, ecc)
        B = self._K_B(mrat, ecc)

        use_a = use_a.diagonal()
        A = A.diagonal()
        g = g.diagonal()
        B = B.diagonal()

        kk = A * np.power((1 + use_a), g) + B
        return kk
        kk = np.clip(kk, *self._bound_K)
        return kk

    def _init_k(self):
        data = self._data['K']
        #    Get all of the mass ratios (ignore other keys)
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
        _dat = self._data['H']
        h_mass_ratios = 1.0/np.array(_dat['q'])
        h_A = np.array(_dat['A'])
        h_a0 = np.array(_dat['a0'])
        h_g = np.array(_dat['g'])

        self._H_A = sp.interpolate.interp1d(h_mass_ratios, h_A, kind='linear', fill_value='extrapolate')
        self._H_a0 = sp.interpolate.interp1d(h_mass_ratios, h_a0, kind='linear', fill_value='extrapolate')
        self._H_g = sp.interpolate.interp1d(h_mass_ratios, h_g, kind='linear', fill_value='extrapolate')
        return


class Sesana_Scattering(_Hardening):
    """

    Notes
    -----
    * Fiducial Dehnen inner density profile slope gamma=1.0 is used in [Chen17]

    """

    def __init__(self, gamma_dehnen=1.0, gbh=None):
        gbh = _get_galaxy_blackhole_relation(gbh)
        self._gbh = gbh
        self._gamma_dehnen = gamma_dehnen
        self._shm06 = SHM06()
        return

    def dadt_dedt(self, evo, step):
        mass = evo.mass[:, step, :]
        sepa = evo.sepa[:, step]
        eccen = evo.eccen[:, step] if evo.eccen is not None else None
        dadt, dedt = self._dadt_dedt(mass, sepa, eccen)
        return dadt, dedt

    def _dadt_dedt(self, mass, sepa, eccen):
        mtot, mrat = utils.mtmr_from_m1m2(mass)
        vdisp = self._gbh.vdisp_from_mbh(mtot)
        mbulge = self._gbh.mbulge_from_mbh(mtot)
        dens = _density_at_influence_radius_dehnen(mtot, mbulge, self._gamma_dehnen)
        hh = self._shm06.H(mrat, sepa)
        dadt = Quinlan1996.dadt(sepa, dens, vdisp, hh)

        rbnd = _radius_influence_dehnan(mtot, mbulge)
        atten = np.exp(-sepa / rbnd)
        dadt = dadt * atten

        if eccen is not None:
            kk = self._shm06.K(mrat, sepa, eccen)
            dedt = Quinlan1996.dedt(sepa, dens, vdisp, hh, kk)
        else:
            dedt = None

        return dadt, dedt


class Dynamical_Friction_NFW(_Hardening):
    """Dynamical Friction (DF) hardening module assuming an NFW dark-matter density profile.

    This class calculates densities and orbital velocities based on a NFW profile with parameters based on those of
    each MBH binary.  The `holodeck.observations.NFW` class is used for profile calculations, and the halo parameters
    are calculated from Stellar-mass--Halo-mass relations (see 'arguments' below).  The 'effective-mass' of the
    inspiralling secondary is modeled as a power-law decreasing from the sum of secondary MBH and its stellar-bulge
    (calculated using the `gbh` - Galaxy-Blackhole relation), down to just the bare secondary MBH after 10 dynamical
    times.  This is to model tidal-stripping of the secondary host galaxy.

    Attenuation of the DF hardening rate is typically also included, to account for the inefficiency of DF once the
    binary enters the hardened regime.  This is calculated using the prescription from [BBR1980].  The different
    characteristic radii, needed for the attenuation calculation, currently use a fixed Dehnan stellar-density profile
    as in [Chen17], and a fixed scaling relationship to find the characteristic stellar-radius.

    This module does not evolve eccentricity.

    Arguments
    ---------
    gbh : class, instance or None
        Galaxy-blackhole relation (_Galaxy_Blackhole_Relation subclass)
        If `None` the default is loaded.
    smhm : class, instance or None
        Stellar-mass--halo-mass relation (_StellarMass_HaloMass subclass)
        If `None` the default is loaded.
    coulomb : scalar,
        coulomb-logarithm ("log(Lambda)"), typically in the range of 10-20.
        This parameter is formally the log of the ratio of maximum to minimum impact parameters.
    attenuate : bool,
        Whether the DF hardening rate should be 'attenuated' due to stellar-scattering effects at
        small radii.  If `True`, DF becomes significantly less effective for radii < R_hard and R_LC
    rbound_from_density : bool,
        Determines how the binding radius (of MBH pair) is calculated, which is used for attenuation.
        NOTE: this is only used if `attenuate==True`
        If True:  calculate R_bound using an assumed stellar density profile.
        If False: calculate R_bound using a velocity dispersion (constant in radius, from `gbh` instance).

    Additional Notes
    ----------------
    *   The hardening rate (da/dt) is not allowed to be larger than the orbital/virial velocity of the halo
        (as a function of radius).

    """

    def __init__(self, gbh=None, smhm=None, coulomb=10.0, attenuate=True, rbound_from_density=True):
        gbh = _get_galaxy_blackhole_relation(gbh)
        smhm = _get_stellar_mass_halo_mass_relation(smhm)
        self._gbh = gbh
        self._smhm = smhm
        self._coulomb = coulomb
        self._attenuate = attenuate
        self._rbound_from_density = rbound_from_density

        self._NFW = holodeck.observations.NFW
        self._time_dynamical = None
        return

    def _dvdt(self, mass_sec_eff, dens, velo):
        """Chandrasekhar dynamical friction formalism providing a deceleration (dv/dt).

        Arguments
        ---------
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

    def dadt_dedt(self, evo, step):
        """Calculate DF hardening rate given `Evolution` instance, and an integration `step`.

        Arguments
        ---------
        evo : `Evolution` instance
        step : int,
            Integration step at which to calculate hardening rates.

        Returns
        -------
        dadt : (N,) np.ndarray of scalar,
            Binary hardening rates in units of [cm/s].
        dedt : (N,) np.ndarray of scalar, NOTE: always zero
            Rate-of-change of eccentricity, which is not included in this calculation (i.e. zero)

        """
        mass = evo.mass[:, step, :]
        sepa = evo.sepa[:, step]
        dt = evo.tlbk[:, 0] - evo.tlbk[:, step]   # positive time-duration since 'formation'
        # NOTE `scafa` is nan for systems "after" redshift zero (i.e. do not merge before redz=0)
        redz = np.zeros_like(sepa)
        val = (evo.scafa[:, step] > 0.0)
        redz[val] = cosmo.a_to_z(evo.scafa[val, step])

        dadt, dedt = self._dadt_dedt(mass, sepa, redz, dt)

        return dadt, dedt

    def _dadt_dedt(self, mass, sepa, redz, dt, attenuate=None):
        """Calculate DF hardening rate given physical quantities.

        Arguments
        ---------
        mass : (N, 2) array-like of scalar,
            Masses of both MBHs (0-primary, 1-secondary) in units of [grams].
        sepa : (N,) array-like of scalar,
            Binary separation in [cm].
        redz : (N,) array-like of scalar,
            Binary redshifts.

        Returns
        -------
        dadt : (N,) np.ndarray of scalar,
            Binary hardening rates in units of [cm/s].
        dedt : (N,) np.ndarray of scalar, NOTE: always zero
            Rate-of-change of eccentricity, which is not included in this calculation.

        """
        if attenuate is None:
            attenuate = self._attenuate

        # ---- Get Host DM-Halo mass
        # use "bulge-mass" as a proxy for total stellar mass
        mstar = self._gbh.mbulge_from_mbh(mass[:, 0])   # use primary-bh's mass (index 0)
        mhalo = self._smhm.halo_mass(mstar, redz, clip=True)

        # ---- Get effective mass of inspiraling secondary
        m2 = mass[:, 1]
        mstar_sec = self._gbh.mbulge_from_mbh(m2)
        if self._time_dynamical is None:
            self._time_dynamical = self._NFW.time_dynamical(sepa, mhalo, redz) * 10

        # model tidal-stripping of secondary's bulge (see: [Kelley17] Eq.6)
        pow = np.clip(1.0 - dt / self._time_dynamical, 0.0, 1.0)
        meff = m2 * np.power((m2 + mstar_sec)/m2, pow)

        dens = self._NFW.density(sepa, mhalo, redz)
        velo = self._NFW.velocity_circular(sepa, mhalo, redz)
        tdyn = self._NFW.time_dynamical(sepa, mhalo, redz)
        # Note: this should be returned as negative values
        dvdt = self._dvdt(meff, dens, velo)

        # convert from deceleration to hardening-rate assuming virialized orbit (i.e. ``GM/a = v^2``)
        dadt = 2 * tdyn * dvdt
        dedt = np.zeros_like(dadt)

        # Hardening rate cannot be larger than orbital/virial velocity
        clip = (np.fabs(dadt) > velo)
        if np.any(clip):
            log.info(f"clipping {utils.frac_str(clip)} `dadt` values to vcirc")
            prev = dadt[:]
            dadt[clip] = - velo[clip]
            log.debug(f"\t{utils.stats(prev*YR/PC)} ==> {utils.stats(dadt*YR/PC)}")
            del prev

        if attenuate:
            atten = self._attenuation_bbr80(sepa, mass, mstar)
            dadt = dadt / atten

        return dadt, dedt

    def _attenuation_bbr80(self, sepa, m1m2, mstar):
        """Calculate attentuation factor following [BBR1980] prescription.

        Characteristic radii are currently calculated using hard-coded Dehnan stellar-density profiles, and a fixed
        scaling-relationship between stellar-mass and stellar characteristic radius.

        The binding radius can be calculated either using the stellar density profile, or from a velocity dispersion,
        based on the `self._rbound_from_density` flag.  See the 'arguments' section of `docs::Dynamical_Friction_NFW`.

        The attenuation factor is defined as >= 1.0, with 1.0 meaning no attenuation.

        Arguments
        ---------
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
        rhard = _radius_hard_bbr80_dehnan(mbh, mstar)
        # characteristic loss-cone-radius in [cm]
        rlc = _radius_loss_cone_bbr80_dehnan(mbh, mstar)

        # Calculate R_bound based on stellar density profile (mass enclosed)
        if self._rbound_from_density:
            rbnd = _radius_influence_dehnan(mbh, mstar)
        # Calculate R_bound based on uniform velocity dispersion (MBH scaling relation)
        else:
            vdisp = self._gbh.vdisp_from_mbh(m1)   # use primary-bh's mass (index 0)
            rbnd = NWTG * mbh / vdisp**2

        # Number of stars in the stellar bulge/core
        nstar = mstar / (0.6 * MSOL)
        # --- Attenuation for separations less than the hardening radius
        # [BBR80] Eq.3
        atten_hard = np.maximum((rhard/sepa) * np.log(nstar), np.square(mbh/mstar) * nstar)
        # use an exponential turn-on at larger radii
        cut = np.exp(-sepa/rhard)
        atten_hard *= cut

        # --- Attenuation for separations less than the loss-cone Radius
        # [BBR80] Eq.2
        atten_lc = np.power(m2/m1, 1.75) * nstar * np.power(rbnd/rstar, 6.75) * (rlc / sepa)
        atten_lc = np.maximum(atten_lc, 1.0)
        # use an exponential turn-on at larger radii
        cut = np.exp(-sepa/rlc)
        atten_hard *= cut

        atten = np.maximum(atten_hard, atten_lc)
        # Make sure that attenuation is always >= 1.0 (i.e. this never _increases_ the hardening rate)
        atten = np.maximum(atten, 1.0)

        return atten


def _get_galaxy_blackhole_relation(gbh=None):
    if gbh is None:
        gbh = holodeck.observations.Kormendy_Ho_2013

    if inspect.isclass(gbh):
        gbh = gbh()
    elif not isinstance(gbh, holodeck.observations._Galaxy_Blackhole_Relation):
        err = "`gbh` must be an instance or subclass of `holodeck.observations._Galaxy_Blackhole_Relation`!"
        log.error(err)
        raise ValueError(err)

    return gbh


def _get_stellar_mass_halo_mass_relation(smhm=None):
    if smhm is None:
        smhm = holodeck.observations.Behroozi_2013

    if inspect.isclass(smhm):
        smhm = smhm()
    elif not isinstance(smhm, holodeck.observations._StellarMass_HaloMass):
        err = "`smhm` must be an instance or subclass of `holodeck.observations._StellarMass_HaloMass`!"
        log.error(err)
        raise ValueError(err)

    return smhm


def _radius_stellar_characteristic_dabringhausen_2008(mstar, gamma=1.0):
    # [Chen17] Eq.27 - from [Dabringhausen+2008]
    rchar = 239 * PC * (np.power(2.0, 1.0/(3.0 - gamma)) - 1.0)
    rchar *= np.power(mstar / (1e9*MSOL), 0.596)
    return rchar


def _radius_influence_dehnan(mbh, mstar, gamma=1.0):
    """
    [Chen17] Eq.25
    """
    rchar = _radius_stellar_characteristic_dabringhausen_2008(mstar, gamma)
    rinfl = np.power(2*mbh/mstar, 1.0/(gamma - 3.0))
    rinfl = rchar / (rinfl - 1.0)
    return rinfl


def _density_at_influence_radius_dehnen(mbh, mstar, gamma=1.0):
    """
    [Chen17] Eq.26
    """
    # [Chen17] Eq.27 - from [Dabringhausen+2008]
    rchar = _radius_stellar_characteristic_dabringhausen_2008(mstar, gamma)
    dens = mstar * (3.0 - gamma) / np.power(rchar, 3.0) / (4.0 * np.pi)
    dens *= np.power(2*mbh / mstar, gamma / (gamma - 3.0))
    return dens


def _radius_hard_bbr80_dehnan(mbh, mstar, gamma=1.0):
    """
    [Kelley17] paragraph below Eq.8 - from [BBR80]
    """
    rbnd = _radius_influence_dehnan(mbh, mstar, gamma=gamma)
    rstar = _radius_stellar_characteristic_dabringhausen_2008(mstar, gamma)
    rhard = rstar * (rbnd/rstar) ** 3
    return rhard


def _radius_loss_cone_bbr80_dehnan(mbh, mstar, gamma=1.0):
    """
    [Kelley17] Eq.9 - from [BBR80]
    """
    mass_of_a_star = 0.6 * MSOL
    rbnd = _radius_influence_dehnan(mbh, mstar, gamma=gamma)
    rstar = _radius_stellar_characteristic_dabringhausen_2008(mstar, gamma)
    rlc = np.power(mass_of_a_star / mbh, 0.25) * np.power(rbnd/rstar, 2.25) * rstar
    return rlc
