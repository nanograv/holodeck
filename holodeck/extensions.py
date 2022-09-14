"""Custom holodeck extensions not supported as part of the core package.
"""

import holodeck as holo
from holodeck.constants import MSOL, GYR


class Realizer:

    def __init__(self, fobs_orb_edges, resample=None, lifetime=2*GYR, **mmbulge_kwargs):
        pop = holo.population.Pop_Illustris()
        if resample is not None:
            mod_resamp = holo.population.PM_Resample(resample=resample)
            pop.modify(mod_resamp)

        # mmbulge = holo.relations.MMBulge_KH2013(mamp=7e8*MSOL)
        mmbulge = holo.relations.MMBulge_KH2013(**mmbulge_kwargs)
        mod_mm13 = holo.population.PM_Mass_Reset(mmbulge, scatter=True)
        mt, _ = holo.utils.mtmr_from_m1m2(pop.mass)
        print("mass bef = ", holo.utils.stats(mt/MSOL))
        pop.modify(mod_mm13)
        mt, _ = holo.utils.mtmr_from_m1m2(pop.mass)
        print("mass aft = ", holo.utils.stats(mt/MSOL))
        pop._sample_volume = pop._sample_volume / 2.0

        fixed = holo.evolution.Fixed_Time.from_pop(pop, lifetime)
        evo = holo.evolution.Evolution(pop, fixed)
        evo.evolve()

        self._pop = pop
        self._evo = evo

        names, vals, weights = evo._sample_universe__at_values_weights(fobs_orb_edges)
        self._fobs_orb_edges = fobs_orb_edges
        self._vals_names = names
        self._binary_vals = vals
        self._binary_weights = weights
        return

    def __call__(self, down_sample=None):
        evo = self._evo

        fobs_orb_edges = self._fobs_orb_edges
        names = self._vals_names
        vals = self._binary_vals
        weights = self._binary_weights
        samples = evo._sample_universe__resample(fobs_orb_edges, vals, weights, down_sample)
        return names, samples
