# Changelog

All notable changes to MESH are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Hierarchical coregionalization — genome level** (first implementation phase
  of the design doc). The genome is modelled as an **entity**: each genome gets
  its own Matérn field with its own range (patch size) and a single positive
  amplitude, inherited identically by every gene it carries — not a hierarchical
  intercept.
  - `mesh.hierarchical_coregionalized_negbinomial` — structured coregionalization
    where the loadings come from `feature → genome` membership (`genome_index`),
    with per-genome ranges (no ordering needed — membership pins field identity)
    and `HalfNormal` amplitudes (no per-field sign symmetry).
  - `mesh.hierarchical_counts_arrays` / `mesh.hierarchical_genome_order` — build
    `coords/counts/log_offset/genome_index/n_genomes` from a counts table plus a
    genome annotation table.
  - `mesh.simulate_hierarchical_genomes` — known-truth generator (per-genome
    ranges + `feature → genome` table).
  - Gating test (`tests/test_hierarchical.py`): two genomes at well-separated
    ranges, each patch size recovered inside its 95% interval.
- **Design doc: hierarchical coregionalization** (`docs/methods/coregionalization_hierarchy.md`)
  — specifies, ahead of implementation, how the flat coregionalization loadings
  become a *structured* matrix built from biological membership: genomes (each an
  entity with its own spatial field, not a grouping level), gene families (residual
  cross-genome field) and traits (KEGG modules / MetaCyc pathways / BGCs, gated by
  genome-inferred completeness). Wired into the methods index and roadmap.

### Changed
- **Composable field core** — the architecture axes are no longer separate,
  mutually exclusive models. A shared field builder now lets number of scales,
  direction and boundary sharpness combine on one likelihood, so previously
  impossible crossings work:
  - `mesh.spatial_negbinomial` is the single composable entry point. It accepts
    `n_fields` (co-existing scales), `anisotropic` (per-axis ranges), and `nu`
    (smoothness), and detects single- vs multi-feature counts from the shape of
    `counts` (`(n,)` vs `(J, n)`). E.g. `spatial_negbinomial(..., anisotropic=True,
    n_fields=2)` is a **directional multi-scale** model, and
    `compare_smoothness(spatial_negbinomial, anisotropic=True, ...)` selects the
    smoothness of an **anisotropic** fit — neither expressible before.
  - `mesh.anisotropic_negbinomial` and `mesh.coregionalized_negbinomial` are now
    thin presets over `spatial_negbinomial` with **unchanged signatures and
    sample sites**, so existing code, tests and docs are unaffected.
  - `mesh.spatial_betabinomial` gains `anisotropic` (still single-feature).
  - New gating tests (`tests/test_compose.py`): a directional + two-scale fit
    still recovers both ranges without inventing a spurious direction on
    isotropic data, and smoothness selection over an anisotropic fit recovers
    both the rough kernel and the direction.

### Added
- **Direction (anisotropy)** — infer whether a feature organises along a host
  axis (proximal–distal gut, crypt–villus, depth into a biofilm) instead of
  assuming an isotropic patch. The field is an **axis-aligned** anisotropic
  Matérn GP: a separate patch size along each coordinate axis. Orient the
  sampling frame to the host axis (a free rotation is a later extension).
  - `mesh.anisotropic_matern_kernel(coords, lengthscales, nu=...)` and
    `mesh.anisotropic_scaled_distances` — a Matérn covariance with a **per-axis
    range** `(ell_x, ell_y)`; with equal lengthscales it is identical to the
    isotropic `mesh.matern_kernel`. The closed-form Matérn radial shape is now
    shared between the two kernels (same `nu` meaning).
  - `mesh.anisotropic_negbinomial` — the directional abundance model.
    Parameterised by an **overall** (geometric-mean) `range` and a **signed log
    anisotropy** `log_ratio = log(ell_x / ell_y)` with a `Normal(0, .)` prior
    centred at isotropy, so direction must be supported by the data. Exposes
    deterministic `lengthscales` and `anisotropy` (= `ell_x/ell_y`).
    `mesh.gp_field_anisotropic` is the shared non-centred field.
  - `mesh.simulate_anisotropic` / `mesh.draw_field_anisotropic` — known-truth
    generator for the model (default: elongated 3× along `x`), recording the
    per-axis `lengthscales`, the geometric-mean `range` and the `anisotropy`.
  - `mesh.summarize_anisotropy` — tidy per-axis patch sizes with the folded
    `anisotropy_ratio` (how directional) and `prob_x_longer` (which axis).
  - `mesh.plot_anisotropy` — overlays the `ell_x` and `ell_y` posteriors.
  - Gating recovery test (`tests/test_anisotropy.py`): a field elongated 3× along
    `x`, asserting both axis ranges land in their 95% intervals, the anisotropy
    is resolved, and its direction (`x` longer) is recovered; plus deterministic
    kernel checks (isotropic reduction and directional decorrelation).
- **Boundary sharpness (Matérn smoothness ν)** — infer whether patches have
  crisp edges or fade as gradients instead of fixing the smoothness at 3/2.
  Small ν = rough field = crisp boundaries (competitive exclusion or a
  physical/biofilm barrier); large ν = smooth field = gradients (diffusion-limited
  O₂/pH/nutrient gradients). Arbitrary ν needs the fractional-order Bessel `K_ν`
  (no autodiff-stable JAX form), so MESH uses the **closed-form family**
  {1/2, 3/2, 5/2} plus the squared-exponential ν→∞ limit and **selects among
  them by LOO model comparison** — one unambiguous field per fit, no discrete
  latent marginalised inside a chain.
  - `mesh.matern_kernel(coords, range, nu=...)` — parametric Matérn covariance
    over the closed-form `mesh.MATERN_NU` values (`0.5, 1.5, 2.5, math.inf`),
    keeping `ell` interpretable as the same **range** (patch size in microns)
    across every member. `mesh.matern32_kernel` is unchanged (now the ν=1.5
    path); `mesh.cholesky_factor` gains a `nu` argument.
  - `nu` argument on `mesh.gp_field`, `mesh.spatial_negbinomial` and
    `mesh.spatial_betabinomial` (default `1.5`), and on `mesh.simulate_counts` /
    `mesh.draw_field` so a field can be drawn at a chosen smoothness (recorded as
    `truth["nu"]`).
  - `mesh.compare_smoothness` — fit the same data under each fixed-ν kernel and
    rank them by PSIS-LOO (`arviz.compare`); returns the comparison table
    (best first, indexed by `mesh.nu_label`, with a `nu` column) and the
    per-ν fits. `fit_model` gains `log_likelihood=True` to store the pointwise
    log-likelihood LOO needs.
  - Gating test (`tests/test_smoothness.py`): the closed forms and the
    roughness-vs-ν ordering are checked deterministically (SE limit included),
    and a simulated **rough** field is recovered as the rough kernel decisively
    by LOO. Roughness is the reliably identifiable direction; confirming extra
    smoothness from noisy counts is intrinsically weaker.
- **Coregionalization (M1+ milestone seed)** — multiple features now share
  latent fields, so the inference can separate co-existing spatial scales and
  read out which features share a territory.
  - `mesh.coregionalized_negbinomial` — a linear model of coregionalization:
    `n_fields` unit-variance Matérn fields at **ordered** ranges, shared across
    features through a feature×field loadings matrix. Ordering the ranges pins
    field identity; per-field sign is left free.
  - `mesh.simulate_coregionalized` — known-truth generator for the model (a
    multi-feature counts table over two scales, default four-feature block
    structure).
  - `mesh.coregion_counts_arrays` / `mesh.coregion_feature_order` — pack a
    multi-feature table into `(J, n)` matrices and recover the feature row order.
  - `mesh.summarize_loadings` — tidy per-`(feature, field)` loadings with a
    sign-invariant `abs_mean` and the `assigned_field` for each feature.
  - Gating recovery test (`tests/test_coregionalization.py`): two co-existing
    scales, asserting both ranges land in their 95% intervals and every feature
    is assigned to the field it was generated from.
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
