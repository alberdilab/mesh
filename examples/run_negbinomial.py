"""Minimal counts-table entry point: recover a patch size from abundance counts.

Simulates a negative-binomial counts table over a Matern field with a known
range, validates it against the input schema, fits ``spatial_negbinomial`` and
prints the recovered patch size with a credible interval.

Run::

    python examples/run_negbinomial.py

Parallel chains
---------------
This script uses the default ``chain_method="vectorized"``, which runs all
chains in one ``vmap`` on a single device -- the fastest option on a laptop CPU.

To run the chains on *separate* CPU cores instead, JAX must be told to expose
more than one host device, and that call has to happen **before JAX initializes
its backend** (i.e. before importing ``mesh``/``jax`` or running any JAX op)::

    import numpyro
    numpyro.set_host_device_count(2)   # FIRST, before the imports below

    from mesh.fit import counts_arrays, fit_model
    ...
    idata = fit_model(..., num_chains=2, chain_method="parallel", **arrays)

If ``jax.local_device_count()`` still prints ``1`` afterwards, the backend was
already initialized earlier in the session -- restart the interpreter and make
``set_host_device_count`` the very first line.
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
