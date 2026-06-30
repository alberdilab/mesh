# Inference

MESH does fully Bayesian inference with the No-U-Turn Sampler (NUTS) via
[NumPyro](https://num.pyro.ai/) on [JAX](https://jax.dev/). The runner is
{func}`mesh.fit_model`; it returns an [ArviZ](https://python.arviz.org/)
`InferenceData` so the full diagnostic ecosystem is available.

## Running a model

```python
from mesh import fit_model, spatial_negbinomial, counts_arrays, simulate_counts

arrays = counts_arrays(simulate_counts(n_samples=150, range_=200.0, seed=0).table)
idata = fit_model(
    spatial_negbinomial,
    num_warmup=500,
    num_samples=500,
    num_chains=2,
    target_accept_prob=0.9,
    seed=0,
    **arrays,
)
```

Key arguments:

- **`num_warmup` / `num_samples`** — NUTS adaptation and sampling lengths. The
  recovery tests use 400/400; 500–1000 each is a reasonable default for real
  fits.
- **`num_chains` / `chain_method`** — run ≥ 2 chains to obtain a meaningful
  $\hat R$. The default `chain_method="auto"` runs the chains on separate
  devices when enough are available, and otherwise falls back to a single `vmap`
  on one device (efficient for exact-GP models on CPU) with a warning. To run
  chains on separate cores, expose the devices first with
  `mesh.enable_parallel_chains(n)` (see
  [running chains in parallel](#running-chains-in-parallel)).
- **`target_accept_prob`** — raising it (e.g. 0.95) shrinks the step size and
  reduces divergences in awkward GP geometries, at some cost in speed.
- **`seed`** — PRNG seed for reproducibility.

## Running chains in parallel

The `chain_method` options differ only in *how* chains are mapped onto
hardware — the sampling is identical:

```{list-table}
:header-rows: 1
:widths: 18 22 60

* - Method
  - Devices used
  - When to use it
* - `"auto"` *(default)*
  - One per chain if available, else 1
  - Picks `"parallel"` when at least `num_chains` devices are exposed, otherwise
    falls back to `"vectorized"` with a warning. Needs no setup.
* - `"vectorized"`
  - 1 (batched `vmap`)
  - Single CPU or single GPU. Normally the **fastest** option there.
* - `"parallel"`
  - One per chain
  - You genuinely have several devices (multi-GPU node, HPC allocation, or CPU
    cores exposed as devices — see below).
* - `"sequential"`
  - 1 (one chain at a time)
  - Memory-constrained runs or debugging; slowest.
```

On a single CPU there is one device, so `"auto"` runs `"vectorized"` and warns
how to do better. To put chains on **separate cores**, expose more CPU devices
with `mesh.enable_parallel_chains(n)`. This must run **before JAX initializes its
backend** — JAX reads the device count only once, on first use — but `import
mesh` no longer initializes JAX, so calling it right after the import (and before
the first `fit_model`) is enough:

```python
import mesh
from mesh import fit_model, spatial_negbinomial, counts_arrays, simulate_counts

print(mesh.enable_parallel_chains(2))   # -> 2

arrays = counts_arrays(simulate_counts(n_samples=150, range_=200.0, seed=0).table)
idata = fit_model(
    spatial_negbinomial,
    num_warmup=500, num_samples=500,
    num_chains=2,                  # auto -> parallel on 2 separate CPU devices
    seed=0, **arrays,
)
```

:::{admonition} Common gotcha
:class: warning

If `enable_parallel_chains` returns `1` (or warns that the backend is already
initialized), JAX was started earlier in the session — e.g. a previous
`fit_model` run or any other JAX op. Restart the interpreter and call it before
the first fit. On Apple Silicon, also check `jax.devices()`: if it shows a
`METAL` device you have `jax-metal` installed, and the host-device flag only
multiplies **CPU** devices — set `os.environ["JAX_PLATFORMS"] = "cpu"` first.
:::

## Why exact GPs here

At MESH's scale (a few hundred to ~2k locations) the dense Matérn covariance and
its Cholesky factor are inexpensive, and exactness keeps the range estimate
faithful. MESH deliberately avoids SPDE, sparse, inducing-point and variational
approximations: they trade accuracy for a scalability the target datasets do not
require, and they can distort the very quantity — the range — that MESH reports.

The per-iteration cost is dominated by one $O(n^3)$ Cholesky of the $n \times n$
covariance. For $n$ in the hundreds this is milliseconds; the practical runtime
is set by the number of leapfrog steps and JAX compilation, not by linear
algebra.

## Diagnostics to check

Treat these as gates before trusting a patch-size estimate:

```{list-table}
:header-rows: 1
:widths: 22 78

* - Diagnostic
  - What to look for
* - **Divergences**
  - Should be zero (or a handful). Divergences flag a geometry the sampler could
    not explore; the non-centered parameterization is the first mitigation,
    raising `target_accept_prob` the second.
* - **$\hat R$ (R-hat)**
  - Close to 1.0 (e.g. < 1.01) for `range` and `eta`. Requires ≥ 2 chains.
    {func}`mesh.summarize_range` reports it per parameter.
* - **Effective sample size (ESS)**
  - Comfortably in the hundreds for the parameters you report. Low ESS means the
    interval is noisy — sample longer.
* - **Posterior predictive sense**
  - The recovered range should be consistent with the sampling resolution and
    domain extent (see [spatial scales](../biology/spatial-scales.md)).
```

Because `idata` is a standard `InferenceData`, you can use ArviZ directly:

```python
import arviz as az
az.summary(idata, var_names=["range", "eta"])
az.plot_trace(idata, var_names=["range"])
print("divergences:", int(idata.sample_stats["diverging"].sum()))
```

## Non-centered parameterization in practice

The field is sampled as standard-normal innovations `f_z` and transformed to the
deterministic site `f` (see [the model](model.md)). This decouples the field from
$\ell$ and $\eta$ and is what keeps NUTS efficient. If you adapt the models and
see divergences return, restoring or strengthening this reparameterization is the
first thing to check.

## Reproducibility notes

- Inference is seeded through `seed`; simulation generators take their own
  `seed`, so an end-to-end run (simulate → fit) is fully reproducible.
- The runner passes empty `dims`/`pred_dims` to ArviZ's NumPyro converter to skip
  its automatic dimension inference, which re-traces the model and is brittle
  across NumPyro/ArviZ versions. All latent sites here are 1-D vectors, so no
  dimension labels are lost.
```
