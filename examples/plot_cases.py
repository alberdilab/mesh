"""Regenerate the figures embedded in the documentation case studies.

Each case study in ``docs/cases`` simulates a table with a known patch size,
fits the matching model, and embeds a figure. This script reproduces those
figures **with the exact parameters used on the pages**, so the images stay in
sync with the quoted numbers, and writes them into ``docs/_static/cases``.

Run from the repo root::

    python examples/plot_cases.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from mesh import (
    allele_arrays,
    counts_arrays,
    fit_model,
    plot_field,
    plot_matern_correlation,
    plot_range_posterior,
    plot_samples,
    plot_scale_comparison,
    posterior_field_mean,
    simulate_allele,
    simulate_counts,
    spatial_betabinomial,
    spatial_negbinomial,
    summarize_range,
    validate_table,
)

OUT = Path(__file__).resolve().parent.parent / "docs" / "_static" / "cases"


def functional_patch():
    """Case 1 — gene abundance, negative-binomial; 150 µm functional patch."""
    sim = simulate_counts(n_samples=200, range_=150.0, eta=1.0, domain=1000.0, seed=1)
    df = sim.table
    validate_table(df)
    idata = fit_model(
        spatial_negbinomial,
        num_warmup=500, num_samples=500, num_chains=2, seed=1,
        progress_bar=False, **counts_arrays(df),
    )
    print("Case 1 — functional patch:")
    print(summarize_range(idata).to_string(index=False))

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    plot_samples(df, value="count", ax=axes[0, 0], title="Input: gene counts")
    plot_range_posterior(idata, truth=150.0, ax=axes[0, 1])
    plot_field(
        sim.coords, posterior_field_mean(idata),
        ax=axes[1, 0], title="Inferred functional field",
    )
    plot_matern_correlation(idata, ax=axes[1, 1])
    fig.tight_layout()
    fig.savefig(OUT / "functional-patch.png", dpi=150)
    plt.close(fig)
    return idata


def strain_territory():
    """Case 2 — allele frequency, beta-binomial; 300 µm strain territory."""
    sim = simulate_allele(
        n_samples=200, range_=300.0, eta=1.2, domain=1500.0, coverage=40, seed=2
    )
    df = sim.table
    validate_table(df, require_allele=True)
    idata = fit_model(
        spatial_betabinomial,
        num_warmup=500, num_samples=500, num_chains=2, seed=2,
        progress_bar=False, **allele_arrays(df),
    )
    print("Case 2 — strain territory:")
    print(summarize_range(idata).to_string(index=False))

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    plot_samples(
        df, value="count", as_frequency=True, cmap="magma",
        ax=axes[0, 0], title="Input: alt-allele frequency",
    )
    plot_range_posterior(idata, truth=300.0, ax=axes[0, 1])
    plot_field(
        sim.coords, posterior_field_mean(idata),
        ax=axes[1, 0], title="Inferred frequency field",
    )
    plot_matern_correlation(idata, ax=axes[1, 1])
    fig.tight_layout()
    fig.savefig(OUT / "strain-territory.png", dpi=150)
    plt.close(fig)
    return idata


def function_vs_genotype(idata_function, idata_genotype):
    """Capstone — the two scales side by side (available today)."""
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    plot_scale_comparison(
        [idata_function, idata_genotype],
        labels=["function patch (Case 1)", "strain territory (Case 2)"],
        ax=ax,
        title="Function vs. genotype scale",
    )
    fig.tight_layout()
    fig.savefig(OUT / "function-vs-genotype.png", dpi=150)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    idata_function = functional_patch()
    idata_genotype = strain_territory()
    function_vs_genotype(idata_function, idata_genotype)
    print(f"Saved case figures to {OUT}")


if __name__ == "__main__":
    main()
