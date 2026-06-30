"""Minimal counts-table entry point: recover a patch size from abundance counts.

Simulates a negative-binomial counts table over a Matern field with a known
range, validates it against the input schema, fits ``spatial_negbinomial`` and
prints the recovered patch size with a credible interval.

Run::

    python examples/run_negbinomial.py

Parallel chains
---------------
This script uses the default ``chain_method="auto"``: it runs chains on separate
CPU cores when enough devices are exposed, and otherwise falls back to a single
``vmap`` over chains (with a warning).

To run the chains on *separate* CPU cores, call ``mesh.enable_parallel_chains``
once before the first fit (after ``import mesh`` is fine -- the import no longer
initializes JAX)::

    import mesh
    mesh.enable_parallel_chains(2)   # before the first fit_model()
    ...
    idata = fit_model(..., num_chains=2, **arrays)   # auto -> parallel

If ``jax.local_device_count()`` still prints ``1`` afterwards, the backend was
already initialized earlier in the session -- ``enable_parallel_chains`` warns
in that case; restart the interpreter and call it before any other JAX op.
"""

from __future__ import annotations

from mesh.fit import counts_arrays, fit_model
from mesh.model import spatial_negbinomial
from mesh.schema import validate_table
from mesh.simulate import simulate_counts
from mesh.summaries import summarize_range


def main() -> None:
    true_range = 200.0
    sim = simulate_counts(n_samples=150, range_=true_range, eta=1.0, seed=0)

    # The table is the documented interface; validate before fitting.
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
    summary = summarize_range(idata)
    row = summary.iloc[0]

    print("Negative-binomial abundance counts")
    print(f"  true range (patch size): {true_range:.1f} microns")
    print(
        f"  posterior mean range:    {row['mean']:.1f} microns "
        f"(95% CI {row['hdi_low']:.1f}-{row['hdi_high']:.1f})"
    )
    print(f"  r_hat: {row['r_hat']:.3f}  ess_bulk: {row['ess_bulk']:.0f}")


if __name__ == "__main__":
    main()
