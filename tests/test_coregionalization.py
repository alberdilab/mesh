"""Gating recovery test for the M1+ coregionalization model.

The milestone criterion (see ``docs/roadmap.md``): simulate two co-existing
spatial scales and confirm the posterior **resolves both ranges** and **assigns
each feature to the right field**. We simulate four features over two
well-separated Matern fields (the small-range field carries features 0-1, the
large-range field carries features 2-3) and assert:

* each true range lies inside its 95% credible interval, and
* every feature's largest-magnitude loading points at the field it was
  generated from (sign-invariant assignment, since per-field sign is free).
"""

from __future__ import annotations

import arviz as az
import numpy as np
import numpyro
import pytest

from mesh.fit import coregion_counts_arrays, coregion_feature_order, fit_model
from mesh.model import coregionalized_negbinomial
from mesh.simulate import simulate_coregionalized
from mesh.summaries import summarize_loadings

numpyro.set_host_device_count(1)

TRUE_RANGES = (80.0, 280.0)
LOWER_FACTOR = 0.5
UPPER_FACTOR = 2.0


@pytest.mark.parametrize("seed", [0])
def test_recovery_coregionalized(seed):
    sim = simulate_coregionalized(
        n_samples=220, ranges=TRUE_RANGES, domain=800.0, seed=seed
    )
    arrays = coregion_counts_arrays(sim.table)
    # The non-centred GP + free per-field sign needs a well-adapted sampler;
    # 800 warmup at target_accept 0.95 converges the ranges (the sign flip
    # itself is a harmless discrete symmetry handled by abs_mean below).
    idata = fit_model(
        coregionalized_negbinomial,
        num_warmup=800,
        num_samples=800,
        num_chains=2,
        seed=seed,
        target_accept_prob=0.95,
        n_fields=len(TRUE_RANGES),
        **arrays,
    )

    # --- Both ranges resolved -------------------------------------------------
    # The ranges are sampled ordered, so range[..., 0] is the small field and
    # range[..., 1] the large one -- aligned with the ascending TRUE_RANGES.
    range_draws = np.asarray(idata.posterior["range"].values)  # (chain, draw, K)
    for k, true_range in enumerate(TRUE_RANGES):
        field_draws = range_draws[:, :, k]  # (chain, draw)
        flat = field_draws.reshape(-1)
        hdi_low, hdi_high = az.hdi(flat, prob=0.95)
        mean = float(np.mean(flat))
        r_hat = float(az.rhat(field_draws))
        assert hdi_low <= true_range <= hdi_high, (
            f"true range {true_range} outside 95% CI [{hdi_low:.1f}, {hdi_high:.1f}]"
        )
        assert LOWER_FACTOR * true_range <= mean <= UPPER_FACTOR * true_range, (
            f"posterior mean {mean:.1f} far from true range {true_range}"
        )
        assert r_hat < 1.1, f"poor convergence, r_hat={r_hat:.3f}"

    # --- Features assigned to the right field ---------------------------------
    feature_ids = coregion_feature_order(sim.table)
    loadings = summarize_loadings(idata, feature_ids=feature_ids)
    assigned = (
        loadings.drop_duplicates("feature").set_index("feature")["assigned_field"]
    )
    true_field = np.argmax(np.abs(sim.truth["loadings"]), axis=1)
    for j, feature in enumerate(feature_ids):
        assert assigned[feature] == true_field[j], (
            f"feature {feature} assigned to field {assigned[feature]}, "
            f"expected {true_field[j]}"
        )
