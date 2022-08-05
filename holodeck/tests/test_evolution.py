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
    print("SETUP `evo` circ!")
    return evo


@pytest.fixture(scope='session')
def evolution_illustris_fixed_time_eccen():
    ecc = holo.population.PM_Eccentricity()
    pop = holo.population.Pop_Illustris(mods=ecc)
    fixed = holo.evolution.Fixed_Time.from_pop(pop, TIME)
    evo = holo.evolution.Evolution(pop, fixed, nsteps=30)
    evo.evolve()
    print("SETUP `evo` eccen!")
    return evo


class Test_Illustris_Fixed:

    def _test_has_keys(self, evolution_illustris_fixed_time_circ):
        evo = evolution_illustris_fixed_time_circ
        # Make sure evolution attributes all exist
        keys = ['mass', 'sepa', 'eccen', 'scafa', 'tlbk', 'dadt', 'dedt']
        for kk in keys:
            assert kk in evo._EVO_PARS, f"Missing attribute key '{kk}' in evolution instance!"

    def test_has_derived_keys(self, evolution_illustris_fixed_time_eccen):
        evo = evolution_illustris_fixed_time_eccen
        # Make sure evolution attributes all exist
        keys = ['mass', 'sepa', 'eccen', 'scafa', 'tlbk', 'dadt', 'dedt']
        for kk in keys:
            vv = getattr(evo, kk)
            err = f"Attribute '{kk}' has <= 0.0 or non-finite values: {holo.utils.stats(vv)}"
            assert np.all(vv > 0.0) and np.all(np.isfinite(vv)), err
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
        time = evo.tlbk
        dt = time[:, 0] - time[:, -1]
        ave = dt.mean()
        std = dt.std()
        assert np.isclose(ave, TIME, rtol=0.25), f"Mean dt differs significantly from input!  {ave/TIME:.4e}"
        assert (std/ave < 0.05), f"Significant variance in output dt values!  {std/ave:.4e}"

    def test_evo_time_circ(self, evolution_illustris_fixed_time_circ):
        self._test_evo_time(evolution_illustris_fixed_time_circ)

    def test_evo_time_eccen(self, evolution_illustris_fixed_time_eccen):
        self._test_evo_time(evolution_illustris_fixed_time_eccen)


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
