# Changelog

All notable changes to MESH are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `mesh.enable_parallel_chains(n)` — expose `n` host (CPU) devices so chains can
  run truly in parallel. Wraps `numpyro.set_host_device_count`; warns if called
  too late (after the JAX backend has initialised) when it can no longer take
  effect.

### Changed
- `fit_model` now defaults to `chain_method="auto"`: it runs chains in parallel
  when at least `num_chains` JAX devices are available, and otherwise falls back
  to `"vectorized"` (faster than `"sequential"` on a single device) with a
  warning explaining how to unlock real parallelism via
  `enable_parallel_chains`.

### Fixed
- `import mesh` no longer initialises the JAX backend. The default range prior
  used a module-level `jnp.log(...)` (a JAX op) that locked the host device
  count at import time; it now uses `math.log`, deferring backend init to the
  first fit so `enable_parallel_chains` can take effect afterwards.

## [0.1.0] — 2026-06-28

First public release. MESH is a spatial-scale-explicit, coverage-aware Bayesian
inference engine for spatially resolved shotgun metagenomics. It is
**inference-only**: it consumes a validated, analysis-ready table and produces
posterior summaries, meeting any upstream bioinformatics pipeline only through
the schema in `mesh.schema`.

### Added

#### Models (`mesh.model`)
- `spatial_betabinomial` — coverage-aware allele-frequency model (the `m0`
  seed), for alt-allele counts with site coverage.
- `spatial_negbinomial` — abundance-count model with a sequencing-depth offset
  (the minimal counts-table entry point).
- `gp_field` — shared latent Gaussian-process field with a **non-centered**
  parameterization for stable NUTS sampling.

#### Kernels (`mesh.kernels`)
- Matérn 3/2 covariance whose **range parameter is the patch size, in microns**.
- `pairwise_distances`, `matern32_kernel`, and `cholesky_factor` for exact
  (dense covariance + Cholesky) Gaussian processes.

#### Input contract (`mesh.schema`)
- `validate_table` enforces the long-format input contract: coordinates present,
  a catalog **shared** across samples (per-sample catalogs are rejected),
  correct dtypes, and no missing values — failing loudly with actionable
  messages.
- `SchemaError` for contract violations.

#### Inference (`mesh.fit`)
- `fit_model` — NUTS runner over either observation model.
- `counts_arrays` / `allele_arrays` — build model inputs from a validated table.
- `get_range_posterior` — extract the posterior draws of the range parameter.

#### Summaries (`mesh.summaries`)
- Posterior range/patch-size summaries with credible intervals.

#### Simulation (`mesh.simulate`)
- `simulate_counts` and `simulate_allele` — synthetic data generators with
  **known ground truth** for simulation-based recovery testing.
- `SimulatedData` container bundling the table, coordinates, and truth.

#### Plotting (`mesh.plots`)
- `plot_samples` — input sampling map of the observed signal over space.
- `plot_range_posterior` — headline patch-size posterior with HDI (optional
  truth marker).
- `plot_field` / `posterior_field_mean` — inferred latent field.
- `plot_matern_correlation` — Matérn correlation decay implied by the range.
- `plot_amplitude_posterior`, `plot_variance_partition`, `plot_scale_comparison`,
  and `plot_samples` helpers for inputs and outputs.

#### Tests
- Simulation-based recovery tests for both observation models.
- Schema validation tests.
- Plotting smoke tests.

#### Examples
- `examples/m0_spatial_field.py` — allele-frequency (beta-binomial) demo.
- `examples/run_negbinomial.py` — abundance-counts (negative-binomial) demo.
- `examples/plot_negbinomial.py` — 2×2 panel of inputs and outputs to PNG.
- `examples/plot_architecture.py` and `examples/plot_cases.py` — figure
  generators for the documentation.

#### Documentation
- Sphinx/MyST documentation toolchain with ReadTheDocs configuration.
- Biological background and technical methods pages.
- Autodoc-based API reference, including a plotting reference page.
- Two expandable biology-driven case studies (strain territory, functional
  patch) and a user guide.

### Notes
- Scope is Milestone 0/1. Not yet implemented (later milestones): multi-scale
  fields, crossed organizational groupings, multi-host hierarchical GPs, and
  multi-omics. No SPDE / sparse / inducing-point / variational-GP machinery.

[Unreleased]: https://github.com/alberdilab/mesh/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/alberdilab/mesh/releases/tag/v0.1.0
