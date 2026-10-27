"""Tests for redz_bin behavior in ``holodeck.gravwaves``.
"""

import numpy as np
import pytest

from holodeck import gravwaves as grav
from holodeck import hardening, librarian, sams, utils
from holodeck.constants import GYR, YR


# Corner redshifts from nanograv/holodeck#165: one stalled corner in an otherwise normal cell
ISSUE_CELL = [-1.0, 0.008, 0.012, 0.025, 0.221, 0.241, 0.24, 0.261]


def _cell(vals):
    # a single (M, Q, Z) bin at one frequency, shape (2, 2, 2, 1)
    return np.reshape(np.asarray(vals, dtype=float), (2, 2, 2, 1))


class Test__redz_at_bin_centers:

    @pytest.mark.parametrize("stall, legacy", [(-1.0, np.mean(ISSUE_CELL)), (-np.inf, -np.inf)])
    def test_issue_cell(self, stall, legacy):
        # stalled corner as sam_cyutils (-1) or lib_tools.run_model (-inf) marks it;
        # dnum is nonzero there, so 'weighted' must zero its weight itself
        redz = _cell(ISSUE_CELL)
        redz[0, 0, 0, 0] = stall
        weighted = grav.redz_at_bin_centers(redz, method='weighted', dnum=np.ones_like(redz))
        assert weighted.shape == (1, 1, 1, 1)
        assert np.allclose(weighted, np.mean(ISSUE_CELL[1:]))   # 0.144
        assert np.allclose(grav.redz_at_bin_centers(redz, method='legacy'), legacy)   # 0.001 is the #165 bug
        assert np.all(grav.redz_at_bin_centers(redz, method='drop') == -np.inf)
        return

    def test_all_stalled(self):
        # no valid corner: 'weighted' keeps the -1 stall flag
        redz = _cell(-np.ones(8))
        test = grav.redz_at_bin_centers(redz, method='weighted', dnum=np.zeros_like(redz))
        assert np.all(test == -1.0)
        return

    def test_unequal_weights(self):
        # four corners at z = 0.1 with dnum = 1, four at z = 0.3 with dnum = 3:
        # (4*0.1*1 + 4*0.3*3) / (4*1 + 4*3) = 0.25
        redz = _cell([0.1] * 4 + [0.3] * 4)
        dnum = _cell([1.0] * 4 + [3.0] * 4)
        assert np.allclose(grav.redz_at_bin_centers(redz, method='weighted', dnum=dnum), 0.25)
        return

    def test_grid(self):
        # no stalled corners and equal dnum: 'weighted' must match the plain mean on a multi-bin grid
        rng = np.random.default_rng(165)
        redz = rng.uniform(0.01, 3.0, (5, 4, 6, 3))
        dnum = np.ones_like(redz)
        weighted = grav.redz_at_bin_centers(redz, method='weighted', dnum=dnum)
        assert weighted.shape == (4, 3, 5, 3)
        assert np.allclose(weighted, grav.redz_at_bin_centers(redz, method='legacy'))

        # inputs are left alone (lib_tools.run_model's -inf rule edits redz in place)
        redz[0, 0, 0, 0] = -1.0
        redz_input = redz.copy()
        for method in ('weighted', 'drop'):
            grav.redz_at_bin_centers(redz, method=method, dnum=dnum)
        assert np.array_equal(redz, redz_input)
        return

    def test_raises(self):
        redz = _cell(ISSUE_CELL)
        with pytest.raises(ValueError):
            grav.redz_at_bin_centers(redz, method='weighted')   # no dnum
        with pytest.raises(ValueError):
            grav.redz_at_bin_centers(redz, method='mean')       # unknown method
        with pytest.raises(ValueError):
            # integrated `number` instead of diff_num; on a single bin it broadcasts silently
            grav.redz_at_bin_centers(redz, method='weighted', dnum=np.ones((1, 1, 1, 1)))
        return


@pytest.fixture(scope="module")
def sam_hard():
    # small grid, with a hardening time long enough that some binaries stall
    sam = sams.Semi_Analytic_Model(shape=10)
    hard = hardening.Fixed_Time_2PL_SAM(sam, 3.0 * GYR)
    return sam, hard


class Test__redz_call_sites:
    # a call site that doesn't pass dnum raises under the default 'weighted' method

    def test_sam(self, sam_hard):
        sam, hard = sam_hard
        _, fobs_gw_edges = utils.pta_freqs(dur=16.0 * YR, num=3)
        gwb = sam.gwb_new(fobs_gw_edges, hard=hard, realize=2)
        assert np.all(np.isfinite(gwb))
        # params=True also exercises the second redshift average in ss_gws_redz
        hc_ss, hc_bg, sspar, bgpar = sam.gwb(fobs_gw_edges, hard=hard, realize=2, params=True)
        for vals in (hc_ss, hc_bg, sspar, bgpar):
            assert np.all(np.isfinite(vals))
        return

    def test_run_model(self, sam_hard):
        sam, hard = sam_hard
        data = librarian.run_model(
            sam, hard, nfreqs=3, nreals=2, nloudest=2,
            gwb_flag=True, singles_flag=True, details_flag=True, params_flag=True,
        )
        for key in ['gwb', 'hc_ss', 'hc_bg']:
            assert np.all(np.isfinite(data[key])), key
        return
