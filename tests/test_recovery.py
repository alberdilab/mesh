"""Simulation-based recovery tests for both M0/M1 observation models.

Each test simulates data with a *known* Matern range, fits the corresponding
model with a small-but-sufficient number of NUTS draws, and asserts that the
true range lies inside the 95% credible interval and that the posterior mean is
within a generous tolerance. This is the primary correctness guarantee for
models whose output cannot be eyeballed.
"""

from __future__ import annotations

import numpyro
import pytest

from mesh.fit import allele_arrays, counts_arrays, fit_model
from mesh.model import spatial_betabinomial, spatial_negbinomial
from mesh.simulate import simulate_allele, simulate_counts
from mesh.summaries import summarize_range

numpyro.set_host_device_count(1)

TRUE_RANGE = 200.0
# Recovery tolerance on the posterior mean, as a multiplicative factor of truth.
LOWER_FACTOR = 0.5
UPPER_FACTOR = 2.0


def _assert_recovered(summary, true_range):
    row = summary.iloc[0]
    # The true range must lie inside the 95% credible interval.
    assert row["hdi_low"] <= true_range <= row["hdi_high"], (
        f"true range {true_range} outside 95% CI "
        f"[{row['hdi_low']:.1f}, {row['hdi_high']:.1f}]"
    )
    # The posterior mean must be in the right ballpark.
    assert LOWER_FACTOR * true_range <= row["mean"] <= UPPER_FACTOR * true_range, (
        f"posterior mean {row['mean']:.1f} far from true range {true_range}"
    )
    # Sampler health.
    assert row["r_hat"] < 1.1, f"poor convergence, r_hat={row['r_hat']:.3f}"


@pytest.mark.parametrize("seed", [0])
def test_recovery_negbinomial(seed):
    sim = simulate_counts(n_samples=150, range_=TRUE_RANGE, eta=1.0, seed=seed)
    arrays = counts_arrays(sim.table)
    idata = fit_model(
        spatial_negbinomial,
        num_warmup=400,
        num_samples=400,
        num_chains=2,
        seed=seed,
        **arrays,
    )
    summary = summarize_range(idata)
    _assert_recovered(summary, sim.truth["range"])


@pytest.mark.parametrize("seed", [0])
def test_recovery_betabinomial(seed):
    sim = simulate_allele(n_samples=150, range_=TRUE_RANGE, eta=1.0, seed=seed)
    arrays = allele_arrays(sim.table)
    idata = fit_model(
        spatial_betabinomial,
        num_warmup=400,
        num_samples=400,
        num_chains=2,
        seed=seed,
        **arrays,
    )
    summary = summarize_range(idata)
    _assert_recovered(summary, sim.truth["range"])
