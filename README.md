# MESH — Metagenomic Ecology across Spatial Hierarchies

Spatial-scale-explicit, **coverage-aware** Bayesian inference for spatially
resolved shotgun metagenomics. MESH infers the **spatial scale** at which
microbial functions and genetic variants segregate across space.

Each latent field is a Gaussian process with a **Matérn covariance whose range
parameter is the patch size, in microns**. Molecular features (gene/contig
abundance now; SNP/strain frequencies later) load onto these fields. The
headline outputs are interpretable by construction: **patch sizes (ranges) with
credible intervals**, which features co-segregate, and (later) a variance
partition across scales.

Inference runs on **NumPyro / JAX** with **exact** Gaussian processes (dense
Matérn covariance + Cholesky) and a **non-centered** parameterization.

> MESH is **inference-only**. It contains no bioinformatics (no read mapping,
> assembly, binning, or variant calling). It consumes a validated, analysis-ready
> table and produces posterior summaries. The bioinformatics pipeline is a
> separate project; the two meet **only** through the schema in `mesh.schema`.

## Status — Milestone 0/1

Two observation models on synthetic data, each with a passing
simulation-based recovery test:

- `spatial_betabinomial` — coverage-aware allele frequencies (the `m0` seed).
- `spatial_negbinomial` — abundance counts with a depth offset (the **minimal
  counts-table entry point**).

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Input contract (`mesh.schema`)

A long-format table with columns:

| column      | meaning                                              |
|-------------|------------------------------------------------------|
| `feature_id`| gene/contig identifier (catalog **shared** across samples) |
| `sample_id` | microsample identifier                               |
| `x`, `y`    | coordinates, in microns                              |
| `count`     | observed count (NB model) or alt-allele count (BB model) |
| `depth`     | sequencing depth offset (NB) or site coverage (BB)   |
| `length`    | feature length (bp)                                  |
| `ref`,`alt` | *optional*, allele model only                        |

`validate_table` confirms coordinates exist, the catalog is shared (rejecting
per-sample catalogs), dtypes are correct, and there are no missing values —
failing loudly with actionable messages.

## Quickstart

```python
from mesh import simulate_counts, validate_table, counts_arrays
from mesh import fit_model, spatial_negbinomial, summarize_range

sim = simulate_counts(n_samples=150, range_=200.0, seed=0)  # known truth
validate_table(sim.table)
arrays = counts_arrays(sim.table)
idata = fit_model(spatial_negbinomial, num_warmup=500, num_samples=500, **arrays)
print(summarize_range(idata))   # patch size + 95% credible interval
```

### Plotting

`mesh.plots` visualises the key inputs and outputs — the input sampling map,
the headline patch-size posterior, the inferred latent field, and the Matérn
correlation decay implied by the range:

```python
from mesh import plot_samples, plot_range_posterior, plot_field
from mesh import plot_matern_correlation, posterior_field_mean

plot_samples(sim.table, value="count")        # input: observed signal over space
plot_range_posterior(idata, truth=200.0)       # output: patch size + HDI
plot_field(sim.coords, posterior_field_mean(idata))  # inferred latent field
plot_matern_correlation(idata)                 # what the range means
```

Runnable demos:

```bash
python examples/m0_spatial_field.py   # allele-frequency (beta-binomial)
python examples/run_negbinomial.py    # abundance counts (negative-binomial)
python examples/plot_negbinomial.py   # 2x2 panel of inputs & outputs -> PNG
```

## Documentation

Full documentation (biological background + technical implementation + API
reference) is built with Sphinx/MyST and is ReadTheDocs-ready:

```bash
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html   # or: cd docs && make html
open docs/_build/html/index.html
```

## Develop

```bash
ruff check .
pytest          # includes the two recovery tests
```

## Scope

This is Milestone 0/1. Not yet implemented (later milestones): multi-scale
fields, crossed organizational groupings, multi-host hierarchical GPs, and
multi-omics. No SPDE / sparse / inducing-point / variational-GP machinery.
