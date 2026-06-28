"""Surface three architecture signals that go beyond patch size.

A single MESH fit reports more than the patch size (``range``). This example
draws the three views in :mod:`mesh.plots` that read the *other* axes of
spatial architecture:

1. **How strong** is the structure -- the field-amplitude (``eta``) posterior;
2. **How deterministic** is it -- the spatially organised share of variance;
3. **Does function share a scale with genotype** -- the abundance fit's patch
   size overlaid on the allele fit's territory.

To make (3) a real comparison it fits *two* models on two simulated features:
an abundance feature (negative-binomial) and a genotype/allele feature
(beta-binomial) with a deliberately smaller territory.

Saves ``architecture_panel.png`` next to this script.

Run::

    python examples/plot_architecture.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from mesh.fit import allele_arrays, counts_arrays, fit_model
from mesh.model import spatial_betabinomial, spatial_negbinomial
from mesh.plots import (
    plot_amplitude_posterior,
    plot_scale_comparison,
    plot_variance_partition,
)
from mesh.schema import validate_table
from mesh.simulate import simulate_allele, simulate_counts


def main() -> None:
    # A function (abundance) organised at a broad scale...
    function_range = 250.0
    counts_sim = simulate_counts(n_samples=150, range_=function_range, eta=1.0, seed=0)
    validate_table(counts_sim.table)
    idata_function = fit_model(
        spatial_negbinomial,
        num_warmup=500,
        num_samples=500,
        num_chains=2,
        seed=0,
        progress_bar=False,
        **counts_arrays(counts_sim.table),
    )

    # ...and a genotype (allele frequency) on a tighter territory.
    genotype_range = 120.0
    allele_sim = simulate_allele(n_samples=150, range_=genotype_range, eta=1.0, seed=1)
    validate_table(allele_sim.table, require_allele=True)
    idata_genotype = fit_model(
        spatial_betabinomial,
        num_warmup=500,
        num_samples=500,
        num_chains=2,
        seed=1,
        progress_bar=False,
        **allele_arrays(allele_sim.table),
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    plot_amplitude_posterior(
        idata_function, truth=1.0, ax=axes[0],
        title="1 · Structure strength (abundance)",
    )
    plot_variance_partition(
        idata_function, ax=axes[1],
        title="2 · Spatial determinism (abundance)",
    )
    plot_scale_comparison(
        [idata_function, idata_genotype],
        labels=["function (abundance)", "genotype (allele)"],
        ax=axes[2],
        title="3 · Function vs. genotype scale",
    )
    fig.tight_layout()

    out = Path(__file__).with_name("architecture_panel.png")
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
