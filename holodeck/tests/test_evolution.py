"""Tests for the :mod:`holodeck.evolution` submodule.
"""

import numpy as np
import pytest

import holodeck as holo
import holodeck.population
import holodeck.evolution
from holodeck.constants import GYR, MSOL, KPC, YR, PC

TIME = 2.0 * GYR


def test_init_generic_evolution():
    """Check that instantiation of a generic evolution class succeeds and fails as expected.
    """

    SIZE = 10

    class Good_Pop(holo.population._Population_Discrete):

        def _init(self):
            self.mass = np.zeros((SIZE, 2))
            self.sepa = np.zeros(SIZE)
            self.scafa = np.zeros(SIZE)
            return

    class Good_Hard(holo.evolution._Hardening):

        def dadt_dedt(self, *args, **kwargs):
            return 0.0

    pop = Good_Pop()
    hard = Good_Hard()
    holo.evolution.Evolution(pop, hard)
    holo.evolution.Evolution(pop, [hard])

    with pytest.raises(TypeError):
        holo.evolution.Evolution(pop, pop)

    with pytest.raises(TypeError):
        holo.evolution.Evolution(hard, hard)

    with pytest.raises(TypeError):
        holo.evolution.Evolution(pop, None)

    with pytest.raises(TypeError):
        holo.evolution.Evolution(None, hard)

    return


@pytest.fixture(scope='session')
def evolution_illustris_fixed_time_circ():
    pop = holo.population.Pop_Illustris()
    fixed = holo.evolution.Fixed_Time.from_pop(pop, TIME)
    evo = holo.evolution.Evolution(pop, fixed, nsteps=30)
    evo.evolve()
    return evo


@pytest.fixture(scope='session')
def evolution_illustris_fixed_time_eccen():
    ecc = holo.population.PM_Eccentricity()
    pop = holo.population.Pop_Illustris(mods=ecc)
    fixed = holo.evolution.Fixed_Time.from_pop(pop, TIME)
    evo = holo.evolution.Evolution(pop, fixed, nsteps=30)
    evo.evolve()
    return evo


@pytest.fixture(scope='session')
def evo_def():
    ecc = holo.population.PM_Eccentricity()
    pop = holo.population.Pop_Illustris(mods=ecc)
    fixed = holo.evolution.Fixed_Time.from_pop(pop, TIME)
    evo = holo.evolution.Evolution(pop, fixed, nsteps=30)

    assert evo._evolved is False
    with pytest.raises(RuntimeError):
        evo._check_evolved()

    evo.evolve()
    assert evo._evolved is True
    evo._check_evolved()
    return evo


class Test_Illustris_Fixed:

    def _test_has_keys(self, evolution_illustris_fixed_time_circ):
        evo = evolution_illustris_fixed_time_circ
        # Make sure evolution attributes all exist
        keys = ['mass', 'sepa', 'eccen', 'scafa', 'tlook', 'dadt', 'dedt']
        for kk in keys:
            assert kk in evo._EVO_PARS, f"Missing attribute key '{kk}' in evolution instance!"

    def test_has_derived_keys(self, evolution_illustris_fixed_time_eccen):
        evo = evolution_illustris_fixed_time_eccen
        # Make sure evolution attributes all exist
        keys = ['mass', 'sepa', 'eccen', 'scafa', 'tlook', 'dadt', 'dedt']
        positive = ['mass', 'sepa', 'eccen', 'scafa', ]
        for kk in keys:
            vv = getattr(evo, kk)

            err = f"Attribute '{kk}' has non-finite values: {holo.utils.stats(vv)}"
            assert np.all(np.isfinite(vv)), err

            if kk in positive:
                err = f"Attribute '{kk}' has <= 0.0 values: {holo.utils.stats(vv)}"
                assert np.all(vv > 0.0), err

        return

    def test_has_keys_circ(self, evolution_illustris_fixed_time_circ):
        self._test_has_keys(evolution_illustris_fixed_time_circ)

    def test_has_keys_eccen(self, evolution_illustris_fixed_time_eccen):
        self._test_has_keys(evolution_illustris_fixed_time_eccen)

    def _test_evo_init(self, evo, eccen):
        # Make sure population attributes are initialized correctly
        keys = ['mass', 'sepa', 'scafa']
        if eccen:
            keys.append('eccen')

        for kk in keys:
            evo_val = getattr(evo, kk)
            pop_val = getattr(evo._pop, kk)

            # Compare evo 0th step values to initial population
            assert np.allclose(evo_val[:, 0], pop_val), f"population and evolution '{kk}' values do not match!"
            # make sure all values are non-zero and finite
            assert np.all(evo_val > 0.0) and np.all(np.isfinite(evo_val)), f"Found '{kk}' <= 0.0 values!"

        # Make sure eccentricity is not set or evolved
        if not eccen:
            assert evo.eccen is None
            assert evo.dedt is None
        return

    def test_evo_init_circ(self, evolution_illustris_fixed_time_circ):
        self._test_evo_init(evolution_illustris_fixed_time_circ, eccen=False)

    def test_evo_init_eccen(self, evolution_illustris_fixed_time_eccen):
        self._test_evo_init(evolution_illustris_fixed_time_eccen, eccen=True)

    def _test_evo_time(self, evo):
        # Make sure lifetimes are close to target
        time = evo.tlook
        dt = time[:, 0] - time[:, -1]
        ave = dt.mean()
        std = dt.std()
        assert np.isclose(ave, TIME, rtol=0.25), f"Mean dt differs significantly from input!  {ave/TIME:.4e}"
        assert (std/ave < 0.05), f"Significant variance in output dt values!  {std/ave:.4e}"

    def test_evo_time_circ(self, evolution_illustris_fixed_time_circ):
        self._test_evo_time(evolution_illustris_fixed_time_circ)

    def test_evo_time_eccen(self, evolution_illustris_fixed_time_eccen):
        self._test_evo_time(evolution_illustris_fixed_time_eccen)


class Test_Evolution_Advanced:

    _EVO_PARS = ['mass', 'sepa', 'eccen', 'scafa', 'tlook', 'dadt', 'dedt']
    _PARS_POSITIVE = ['mass', 'sepa', 'eccen', 'scafa']

    def test_at_failures(self, evo_def):
        evo = evo_def

        # Make sure single, and iterable values work
        xpar = 'fobs'
        evo.at(xpar, [1/YR, 2/YR])
        evo.at(xpar, [1/YR])
        evo.at(xpar, 1/YR)
        evo.at(xpar, 1/YR, params=['sepa', 'scafa'])
        evo.at(xpar, 1/YR, params=['mass'])
        evo.at(xpar, 1/YR, params='mass')

        xpar = 'sepa'
        evo.at(xpar, [1*PC, 2*PC])
        evo.at(xpar, [1*PC])
        evo.at(xpar, 1*PC)
        evo.at(xpar, 1*PC, params=['mass', 'scafa'])
        evo.at(xpar, 1*PC, params=['mass'])
        evo.at(xpar, 1*PC, params='mass')

        fobs = np.logspace(-2, 1, 4) / YR
        # sepa = np.logspace(-2, 2, 5) * PC

        # use invalid 'xpar' name ('mass')
        with pytest.raises(ValueError, match='`xpar` must be one of '):
            evo.at('mass', fobs)

        with pytest.raises(ValueError, match='`targets` extrema'):
            evo.at('fobs', 1e20/YR)

        with pytest.raises(ValueError, match='`targets` extrema'):
            evo.at('sepa', 1e6*PC)

        with pytest.raises(ValueError, match='`targets` extrema'):
            evo.at('sepa', 1e-10*PC)

        return

    def test_at_fobs_all(self, evo_def):
        evo = evo_def
        xpar = 'fobs'
        coal = False

        # Choose interpolation targets
        fobs = np.logspace(-3, 0, 9) / YR
        numx = fobs.size
        # For a moderate range of frequencies (after formation, before coalescence), then all values
        # should be finite (`nan`s are returned either before formation, or after coalescence)
        FINITE_FLAG = True
        print(f"{fobs*YR}")
        vals = evo.at(xpar, fobs, coal=coal)
        print(f"received `at` vals with keys: {vals.keys()}!")

        for par in self._EVO_PARS:
            # Make sure parameter is also included in `evolution` instance's list of parameters
            assert par in evo._EVO_PARS
            # Make sure parameter is actually returned
            assert par in vals, f"'{par}' missing from returned `at` dictionary!"

            # Check shape, should be (N, X) N-binaries, X-targets, except for `mass` which is (N, X, 2)
            vv = vals[par]
            assert vv.shape[0] == evo.size
            assert vv.shape[1] == numx
            assert np.ndim(vv) == 2 or np.shape(vv)[2] == 2

            if FINITE_FLAG:
                err = f"{par} found {holo.utils.frac_str(~np.isfinite(vv))} non-finite values!"
                assert np.all(np.isfinite(vv)), err

            # Make sure values are positive when they should be
            if par in self._PARS_POSITIVE:
                print(par, np.all(vv > 0.0), holo.utils.frac_str(vv > 0.0), np.any(vv <= 0.0), np.any(~np.isfinite(vv)))
                assert np.all(vv > 0.0), f"{par} values found to be non-positive ({holo.utils.frac_str(vv > 0.0)})!"

        for par in evo._EVO_PARS:
            assert par in vals, f"'{par}' missing from returned `at` dictionary!"

        # Choose interpolation targets
        # 0th element here should be within evolution range, while 1th element should be after coalesence -> `nan` vals
        fobs = np.array([0.1, 1e6]) / YR
        print(f"fobs = {fobs*YR} [1/yr]")
        vals = evo.at(xpar, fobs, coal=coal)

        for ii in range(2):
            msg = "all values should be "
            msg += "finite" if ii == 0 else "non-finite"
            for kk, vv in vals.items():
                vv = vv[:, ii]
                test = np.isfinite(vv) if ii == 0 else ~np.isfinite(vv)
                err = f"{kk} {msg}, good={holo.utils.frac_str(test)} | bad={holo.utils.frac_str(~test)}"
                print(err)
                assert np.all(test), err

    def test_at_sepa_coal(self, evo_def):
        evo = evo_def
        xpar = 'sepa'
        coal = True

        # Choose interpolation targets
        sepa = 1e1 * PC
        vals = evo.at(xpar, sepa, coal=coal)

        ncoal = np.count_nonzero(evo.coal)
        ntot = evo.size
        assert ncoal < ntot, f"This test requires that not all binaries are coalescing ({ncoal}/{ntot})!"

        vv = vals['sepa']

        assert vv.shape[0] == ntot
        assert np.count_nonzero(np.isfinite(vv)) == ncoal
        assert np.all(np.isfinite(vv) == evo.coal)

        # ---- make sure the right values are finite and non-finite
        sel = (evo.scafa[:, -1] < 1.0)

        def fstr(xx):
            return holo.utils.frac_str(xx, 8)

        # make sure returned `at` values are all finite for coalescing systems
        yesfin = np.isfinite(vv[sel])
        msg = f"{fstr(sel)} systems are coalescing, `at` samples should be finite: {fstr(yesfin)}"
        print(msg)
        assert np.all(yesfin), msg

        # make sure returned `at` values are all non-finite for non-coalescing systems
        notfin = ~np.isfinite(vv[~sel])
        msg = f"{fstr(~sel)} systems are stalling  , `at` samples should be non-finite: {fstr(notfin)}"
        print(msg)
        assert np.all(notfin), msg

        return

    def test_at_sepa_all(self, evo_def):
        evo = evo_def
        xpar = 'sepa'
        coal = False

        # Choose interpolation targets
        sepa = np.logspace(-1, 3, 8) * PC
        numx = sepa.size
        # For a moderate range of frequencies (after formation, before coalescence), then all values
        # should be finite (`nan`s are returned either before formation, or after coalescence)
        FINITE_FLAG = True
        print(f"sepa/PC = {sepa/PC}")
        vals = evo.at(xpar, sepa, coal=coal)
        print(f"received `at` vals with keys: {vals.keys()}!")

        for par in self._EVO_PARS:
            # Make sure parameter is also included in `evolution` instance's list of parameters
            assert par in evo._EVO_PARS
            # Make sure parameter is actually returned
            assert par in vals, f"'{par}' missing from returned `at` dictionary!"

            # Check shape, should be (N, X) N-binaries, X-targets, except for `mass` which is (N, X, 2)
            vv = vals[par]
            assert vv.shape[0] == evo.size
            assert vv.shape[1] == numx
            assert np.ndim(vv) == 2 or np.shape(vv)[2] == 2

            if FINITE_FLAG:
                err = f"{par} found {holo.utils.frac_str(~np.isfinite(vv))} non-finite values!"
                assert np.all(np.isfinite(vv)), err

            # Make sure values are positive when they should be
            if par in self._PARS_POSITIVE:
                print(par, np.all(vv > 0.0), holo.utils.frac_str(vv > 0.0), np.any(vv <= 0.0), np.any(~np.isfinite(vv)))
                assert np.all(vv > 0.0), f"{par} values found to be non-positive ({holo.utils.frac_str(vv > 0.0)})!"

        for par in evo._EVO_PARS:
            assert par in vals, f"'{par}' missing from returned `at` dictionary!"

        # Choose interpolation targets
        # 0th element here should be within evolution range, while 1th element should be after coalesence -> `nan` vals
        sepa = np.array([1e6, 0.1, 1e-8]) * PC
        print(f"sepa/PC = {sepa/PC}")
        vals = evo.at(xpar, sepa, coal=coal)

        for ii in range(2):
            msg = "all values should be "
            msg += "finite" if ii == 1 else "non-finite"
            for kk, vv in vals.items():
                vv = vv[:, ii]
                test = np.isfinite(vv) if ii == 1 else ~np.isfinite(vv)
                err = f"{kk} {msg}, good={holo.utils.frac_str(test)} | bad={holo.utils.frac_str(~test)}"
                print(err)
                assert np.all(test), err

        return


def mockup_modified():

    SIZE = 123

    class Pop(holo.population._Population_Discrete):

        def _init(self):
            self.mass = (10.0 ** np.random.uniform(6, 10, (SIZE, 2))) * MSOL
            self.sepa = (10.0 ** np.random.uniform(1, 3, SIZE)) * KPC
            self.scafa = np.random.uniform(0.25, 0.75, SIZE)
            return

    class Hard(holo.evolution._Hardening):

        def dadt_dedt(self, evo, step, *args, **kwargs):
            dadt = -(PC/YR) * np.ones(evo.size)
            dedt = None
            return dadt, dedt

    class Mod(holo.utils._Modifier):

        def modify(self, base):
            base.mass[...] = 0.0

    pop = Pop()
    hard = Hard()
    mod = Mod()
    evo = holo.evolution.Evolution(pop, hard, mods=mod)
    return evo


class Test_Modified:

    KEYS = ['sepa', 'mass', 'scafa']

    def test_uninit_without_evolving(self):
        evo = mockup_modified()

        for kk in self.KEYS:
            vv = getattr(evo, kk)
            assert np.all(vv[...] == 0.0)

        return

    def test_modified_after_evolved(self):
        evo = mockup_modified()
        print("before evolving, make sure all values are zero")
        for kk in self.KEYS:
            vv = getattr(evo, kk)
            assert np.all(vv[...] == 0.0)

        evo.evolve()
        print("after evolving, make sure non-modified values are non-zero")
        assert np.all(evo.sepa > 0.0)
        assert np.all(evo.scafa > 0.0)
        print("after evolving, make sure modified (sets mass to zero) took effect")
        assert np.all(evo.mass == 0.0)
        return
