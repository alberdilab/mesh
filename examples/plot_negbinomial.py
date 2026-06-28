"""Visualise the key inputs and outputs of a negative-binomial counts fit.

Simulates a counts table with a known patch size, fits ``spatial_negbinomial``
and draws a 2x2 panel:

1. the **input** -- observed counts over space;
2. the headline **output** -- the patch-size (range) posterior, with truth;
3. the **true vs inferred latent field**;
4. the Matern correlation decay implied by the range posterior.

Saves ``negbinomial_panel.png`` next to this script.

Run::

    python examples/plot_negbinomial.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from mesh.fit import counts_arrays, fit_model
from mesh.model import spatial_negbinomial
from mesh.plots import (
    plot_field,
    plot_matern_correlation,
    plot_range_posterior,
    plot_samples,
    posterior_field_mean,
)
from mesh.schema import validate_table
from mesh.simulate import simulate_counts


def main() -> None:
    true_range = 200.0
    sim = simulate_counts(n_samples=150, range_=true_range, eta=1.0, seed=0)
    validate_table(sim.table)
    arrays = counts_arrays(sim.table)

    idata = fit_model(
        spatial_negbinomial,
        num_warmup=500,
        num_samples=500,
        num_chains=2,
        seed=0,
        progress_bar=False,
        **arrays,
    )

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    plot_samples(sim.table, value="count", ax=axes[0, 0], title="Input: observed counts")
    plot_range_posterior(idata, truth=true_range, ax=axes[0, 1])
    plot_field(
        sim.coords, posterior_field_mean(idata),
        ax=axes[1, 0], title="Inferred latent field",
    )
    plot_matern_correlation(idata, ax=axes[1, 1])
    fig.tight_layout()

    out = Path(__file__).with_name("negbinomial_panel.png")
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
