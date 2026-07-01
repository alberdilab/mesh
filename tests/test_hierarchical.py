"""Gating recovery test for hierarchical coregionalization -- genome level.

The milestone criterion (see ``docs/methods/coregionalization_hierarchy.md``):
treat the **genome as an entity** with its own spatial field, and recover a
**patch size per genome**. We simulate two genomes at well-separated ranges
(each carrying several genes that inherit its field identically) and assert:

* each genome's true range lies inside its 95% credible interval, and
* the chains converge (``r_hat < 1.1``).

Field identity is pinned by membership (each field loads on a disjoint set of
genes), so -- unlike the flat coregionalization model -- the ranges need no
ordering and there is no per-field sign symmetry to correct for.
"""

from __future__ import annotations

import arviz as az
import numpy as np
import numpyro
import pytest

from mesh.fit import fit_model, hierarchical_counts_arrays
from mesh.model import hierarchical_coregionalized_negbinomial
from mesh.simulate import simulate_hierarchical_genomes

numpyro.set_host_device_count(1)

TRUE_RANGES = (90.0, 300.0)
LOWER_FACTOR = 0.5
UPPER_FACTOR = 2.0


@pytest.mark.parametrize("seed", [0])
def test_recovery_hierarchical_genomes(seed):
    sim = simulate_hierarchical_genomes(
        n_samples=220,
        genome_ranges=TRUE_RANGES,
        genes_per_genome=(3, 3),
        domain=800.0,
        seed=seed,
    )
    arrays = hierarchical_counts_arrays(sim.table, sim.truth["genome"])

    idata = fit_model(
        hierarchical_coregionalized_negbinomial,
        num_warmup=800,
        num_samples=800,
        num_chains=2,
        seed=seed,
        target_accept_prob=0.95,
        **arrays,
    )

    # --- A patch size per genome, each recovered --------------------------------
    # range[..., k] is genome k, aligned with sorted genome_id order and thus
    # with the ascending TRUE_RANGES the simulator used.
    range_draws = np.asarray(idata.posterior["range"].values)  # (chain, draw, K)
    assert range_draws.shape[-1] == len(TRUE_RANGES)
    for k, true_range in enumerate(TRUE_RANGES):
        genome_draws = range_draws[:, :, k]  # (chain, draw)
        flat = genome_draws.reshape(-1)
        hdi_low, hdi_high = az.hdi(flat, prob=0.95)
        mean = float(np.mean(flat))
        r_hat = float(az.rhat(genome_draws))
        assert hdi_low <= true_range <= hdi_high, (
            f"genome {k}: true range {true_range} outside 95% CI "
            f"[{hdi_low:.1f}, {hdi_high:.1f}]"
        )
        assert LOWER_FACTOR * true_range <= mean <= UPPER_FACTOR * true_range, (
            f"genome {k}: posterior mean {mean:.1f} far from true {true_range}"
        )
        assert r_hat < 1.1, f"genome {k}: poor convergence, r_hat={r_hat:.3f}"
