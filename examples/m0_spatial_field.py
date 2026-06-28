"""M0 spatial-field demo: recover a known patch size from allele frequencies.

This is the ``m0`` seed: a single Matern field drives logit allele frequencies,
observed coverage-aware via a beta-binomial. We simulate with a known range,
fit ``spatial_betabinomial`` and print the recovered patch size with a credible
interval.

Run::

    python examples/m0_spatial_field.py
"""

from __future__ import annotations

from mesh.fit import allele_arrays, fit_model
from mesh.model import spatial_betabinomial
from mesh.simulate import simulate_allele
from mesh.summaries import summarize_range


def main() -> None:
    true_range = 200.0
    sim = simulate_allele(n_samples=150, range_=true_range, eta=1.0, seed=0)
    arrays = allele_arrays(sim.table)

    idata = fit_model(
        spatial_betabinomial,
        num_warmup=500,
        num_samples=500,
        num_chains=2,
        seed=0,
        progress_bar=False,
        **arrays,
    )
    summary = summarize_range(idata)
    row = summary.iloc[0]

    print("M0 spatial field (beta-binomial allele frequencies)")
    print(f"  true range (patch size): {true_range:.1f} microns")
    print(
        f"  posterior mean range:    {row['mean']:.1f} microns "
        f"(95% CI {row['hdi_low']:.1f}-{row['hdi_high']:.1f})"
    )
    print(f"  r_hat: {row['r_hat']:.3f}  ess_bulk: {row['ess_bulk']:.0f}")


if __name__ == "__main__":
    main()
