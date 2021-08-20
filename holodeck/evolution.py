"""
Holodeck - evolution submodule.

To-Do
-----
* [ ] evolution modifiers should act at each step, instead of after all steps?  This would be a way to implement a
    changing accretion rate, for example; or to set a max/min hardening rate.

References
----------
* [Quinlan96] :: Quinlan 1996
    The dynamical evolution of massive black hole binaries I. Hardening in a fixed stellar background
    https://ui.adsabs.harvard.edu/abs/1996NewA....1...35Q/abstract
* [SHM06] :: Sesana, Haardt & Madau et al. 2006
    Interaction of Massive Black Hole Binaries with Their Stellar Environment. I. Ejection of Hypervelocity Stars
    https://ui.adsabs.harvard.edu/abs/2006ApJ...651..392S/abstract
* [Sesana10] :: Sesana 2010
    Self Consistent Model for the Evolution of Eccentric Massive Black Hole Binaries in Stellar Environments:
    Implications for Gravitational Wave Observations
    https://ui.adsabs.harvard.edu/abs/2010ApJ...719..851S/abstract
* [Chen17] : Chen, Sesana, & Del Pozzo 2017
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
from holodeck.constants import GYR, NWTG, PC, MSOL

_DEF_TIME_DELAY = (5.0*GYR, 0.2)
_SCATTERING_DATA_FILENAME = "SHM06_scattering_experiments.json"


@enum.unique
class EVO(enum.Enum):
    CONT = 1
    END = -1


class _Evolution(abc.ABC):

    _EVO_PARS = ['mass', 'sepa', 'eccen', 'scafa', 'dadt', 'tlbk']
    _LIN_INTERP_PARS = ['eccen', 'scafa', 'tlbk']
    _SELF_CONSISTENT = None
    _STORE_FROM_POP = ['_sample_volume']

    def __init__(self, pop, hard, nsteps=100, mods=None, check=True):
        self._pop = pop
        if np.isscalar(hard):
            hard = [hard, ]
        self._hard = hard
        self._nsteps = nsteps
        self._mods = mods

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
        for step in steps_list:
            rv = self._take_next_step(step)
            if rv is EVO.END:
                self._evolved = True
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
        redz = utils.a_to_z(pop.scafa)
        tlbk = cosmo.z_to_tlbk(redz)
        self.tlbk[:, 0] = tlbk
        self.mass[:, 0, :] = pop.mass

        if (pop.eccen is not None):
            self.eccen[:, 0] = pop.eccen

        return

    def _take_next_step(self, step):
        size, nsteps = self.shape
        left = step - 1
        right = step

        dadt_1 = np.zeros(size)
        dadt_2 = np.zeros_like(dadt_1)
        if self.eccen is not None:
            deccdt_1 = np.zeros_like(dadt_1)
            deccdt_2 = np.zeros_like(dadt_2)

        # Get hardening rates and left (step-1) and right (step) edges
        for hard in self._hard:
            _a1, _e1 = hard.dadt_dedt(self, left)
            _a2, _e2 = hard.dadt_dedt(self, right)
            dadt_1[:] += _a1
            dadt_2[:] += _a2
            if self.eccen is not None:
                deccdt_1[:] += _e1
                deccdt_2[:] += _a2

        # Calculate time between edges
        dadt = 0.5 * np.log10([dadt_1, dadt_2]).sum(axis=0)
        dadt = np.power(10.0, dadt)
        # NOTE: `da` defined to be positive!
        da = self.sepa[:, left] - self.sepa[:, right]
        dt = da / dadt
        if np.any(dt < 0.0):
            err = f"Negative time-steps found at step={step}!"
            log.error(err)
            raise ValueError(err)
        self.tlbk[:, right] = self.tlbk[:, left] + dt
        self.scafa[:, right] = cosmo.tlbk_to_z(self.tlbk[:, right])

        if self.eccen is not None:
            deccdt = np.mean([deccdt_1, deccdt_2], axis=0)
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
        redz = utils.a_to_z(self.scafa)
        fobs = self.freq_orb_rest / (1.0 + redz)
        return fobs

    def _check_evolved(self):
        if self._evolved is not True:
            raise RuntimeError("This instance has not been evolved yet!")

        return

    def at(self, xpar, targets, pars=None, coal=False):
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
            lin_interp = (pp in self._LIN_INTERP_PARS)
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

            if not lin_interp:
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
            if not lin_interp:
                new = 10.0 ** new
            if squeeze:
                new = new.squeeze()
            vals[pp] = new

            if np.any(~np.isfinite(new[valid, ...])):
                raise ValueError("Non-finite values after exponentiation of '{}'".format(pp))

        return vals


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
        self.scafa[:, 1:] = utils.z_to_a(redz)[:, np.newaxis]

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
        self.scafa[:, 1:] = utils.z_to_a(redz)[:, np.newaxis]

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


class _Hardening(abc.ABC):

    @abc.abstractmethod
    def dadt_dedt(self, evo, step, *args, **kwargs):
        pass


class Hard_GW(_Hardening):

    @staticmethod
    def dadt_dedt(evo, step):
        m1, m2 = evo.mass[:, step, :].T    # (Binaries, Steps, 2) ==> (2, Binaries)
        dadt = utils.gw_hardening_rate_dadt(m1, m2, evo.sepa, eccen=evo.eccen)

        if evo.eccen is not None:
            dedt = utils.gw_dedt(m1, m2, evo.sepa, evo.eccen)

        return dadt, dedt


class Quinlan1996:

    @staticmethod
    def dadt(sepa, rho, sigma, hparam):
        """
        [Sesana10] Eq.8
        """
        rv = (sepa ** 2) * NWTG * rho * hparam / sigma
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
        xx = sepa / self._H_a0(mrat)
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
        k_mass_ratios = 1.0/np.array(_dat['q'])
        k_A = np.array(_dat['A'])
        k_a0 = np.array(_dat['a0'])
        k_g = np.array(_dat['g'])

        self._H_A = sp.interpolate.interp1d(k_mass_ratios, k_A, kind='linear', fill_value='extrapolate')
        self._H_a0 = sp.interpolate.interp1d(k_mass_ratios, k_a0, kind='linear', fill_value='extrapolate')
        self._H_g = sp.interpolate.interp1d(k_mass_ratios, k_g, kind='linear', fill_value='extrapolate')
        return


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
        mass = evo.mass[:, step]
        sepa = evo.sepa[:, step]
        eccen = evo.eccen[:, step] if evo.eccen is not None else None
        dadt, dedt = self._dadt_dedt(mass, sepa, eccen)
        return dadt, dedt

    def _dadt_dedt(self, mass, sepa, eccen):
        mtot, mrat = utils.mtmr_from_m1m2(*mass)
        vdisp = self._gbh.vdisp_from_mbh(mtot)
        mbulge = self._gbh.mbulge_from_mbh(mtot)
        dens = _density_at_influence_radius_dehnen(mtot, mbulge, self._gamma_dehnen)

        hh = self._shm06.H(mrat, sepa)
        dadt = Quinlan1996.dadt(sepa, dens, vdisp, hh)
        if eccen is not None:
            kk = self._shm06.K(mrat, sepa, eccen)
            dedt = Quinlan1996.dedt(sepa, dens, vdisp, hh, kk)
        else:
            dedt = None

        return dadt, dedt


class Dynamical_Friction_NFW(_Hardening):

    def __init__(self, gbh=None, smhm=None, coulomb=10.0):
        gbh = _get_galaxy_blackhole_relation(gbh)
        smhm = _get_stellar_mass_halo_mass_relation(smhm)
        self._gbh = gbh
        self._smhm = smhm
        self._coulomb = 10.0
        return

    def _dvdt(self, mass_sec_eff, dens, velo):
        dvdt = 2*np.pi * mass_sec_eff * dens * self._coulomb * np.square(NWTG / velo)
        return dvdt

    def dadt_dedt(self, evo, step):
        mass = evo.mass[:, step]
        sepa = evo.sepa[:, step]
        scal = evo.scafa[:, step]
        return

    def _dadt_dedt(self, mass_sec_eff, mhalo, sepa, redz):
        NFW = holodeck.observations.NFW
        dens = NFW.density(sepa, mhalo, redz)
        velo = NFW.velocity_circular(sepa, mhalo, redz)
        tdyn = NFW.time_dynamical(sepa, mhalo, redz)
        dvdt = self._dvdt(mass_sec_eff, dens, velo)
        dadt = 2 * tdyn * dvdt
        dedt = None
        return dadt, dedt


def _radius_stellar_characteristic_dabringhausen_2008(mstar, gamma=1.0):
    # [Chen17] Eq.27 - from [Dabringhausen+2008]
    rchar = 239 * PC * (np.power(2.0, 1.0/(3.0 - gamma)) - 1.0)
    rchar *= np.power(mstar / (1e9*MSOL), 0.596)
    return rchar


def _density_at_influence_radius_dehnen(mbh, mstar, gamma=1.0):
    """
    [Chen17] Eq.26
    """
    # [Chen17] Eq.27 - from [Dabringhausen+2008]
    rchar = _radius_stellar_characteristic_dabringhausen_2008(mstar, gamma)
    dens = mstar * (3.0 - gamma) / np.power(rchar, 3.0) / (4.0 * np.pi)
    dens *= np.power(2*mbh / mstar, gamma / (gamma - 3.0))
    return dens
