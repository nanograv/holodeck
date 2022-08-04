"""Tests for the :mod:`holodeck.evolution` submodule.
"""

import numpy as np
import pytest

import holodeck as holo
import holodeck.population
import holodeck.evolution


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