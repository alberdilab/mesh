# Roadmap

MESH is built in milestones, each gated by a simulation-based recovery test. The
package today implements **Milestone 0/1**; later milestones extend the model
without changing its interpretable core (patch sizes with credible intervals).

Each milestone unlocks a **biological question about spatial architecture** —
not just *how big are the patches*, but *why is the community arranged this
way*. The list below is organised around those questions.

## Done — M0/M1: single-field, two likelihoods

- Matérn 3/2 exact-GP field with a non-centered parameterization.
- {func}`mesh.spatial_betabinomial` — coverage-aware allele frequencies (the
  `m0` seed).
- {func}`mesh.spatial_negbinomial` — abundance counts with a depth offset (the
  minimal counts-table entry point).
- Input {doc}`schema <methods/schema>` with loud validation.
- Recovery tests for both models (true range inside the 95% interval).

### Architecture signals already readable from one fit

A single fit answers more than patch size; these are surfaced now (see
{doc}`reading your results <guide/read-your-results>`):

- **How strong is the structure?** — the field amplitude `eta`
  ({func}`mesh.plot_amplitude_posterior`). Grain and intensity are independent
  axes.
- **How deterministic is it?** — the spatial share of variance
  ({func}`mesh.variance_partition`): structured signal vs. unstructured noise,
  a handle on deterministic vs. stochastic assembly.
- **Does a function share a scale with a genotype?** — overlay an abundance fit
  and an allele fit ({func}`mesh.plot_scale_comparison`): a function patch wider
  than any allele territory is shared across strains.

## Next — M1+: coregionalization (separating scales)

*Biological question: **which features share a territory, and do they cooperate
or compete there?*** The immediate next step is **shared loadings across multiple
features** so that features can co-segregate, and a model with **two Matérn
fields at different ranges** that the inference can **separate**. The sign and
structure of the shared loadings reads out ecology directly: features that
occupy the *same* patches suggest cross-feeding, syntrophy or a shared niche;
features that *anti-correlate* suggest competition or niche partitioning. The
gating test: simulate two co-existing scales and confirm the posterior resolves
both ranges and assigns features to the right field.

## Later milestones — more axes of architecture

Each item below is a distinct architecture axis, framed by the biological
question it answers. They are described in the design overview and are
intentionally **not** yet implemented. The first three stay inside the current
single-fit paradigm (same input table, same Matérn-GP machinery) and are the
cheapest to add.

- **Boundary sharpness** — *do patches have crisp edges or fade as gradients?*
  Crisp boundaries point to competitive exclusion or a physical/biofilm barrier;
  smooth gradients point to diffusion-limited gradients of O₂, pH or nutrients.
  Estimate (or model-compare) the Matérn smoothness ν instead of fixing it at
  3/2.
- **Direction (anisotropy)** — *does a feature organise along a host axis?*
  Proximal–distal gut, crypt–villus, or depth into a biofilm. A per-axis
  lengthscale (or a rotation) replaces the single isotropic `range`.
- **Stationarity** — *is the grain constant, or are there zones?* Finer near a
  surface, coarser in a lumen. A non-stationary range, or region-wise fits.
- **Multi-scale fields** — *is the architecture hierarchical?* A fine mosaic
  nested inside broad zones — a variance partition of each feature across several
  spatial scales (the clean version of the single-fit determinism index above).
- **Crossed organizational × spatial hierarchy** — *does function track the
  organism, or move independently of it?* Gene-centric grouping (function,
  operon/MGE, taxon of residence) crossed with spatial scale; a function whose
  patch crosses taxa is modular or horizontally transferred.
- **Multi-host hierarchical GP** — *is spatial architecture a reproducible host
  trait?* Partial pooling of patch sizes across hosts / individuals, and a test
  of whether perturbation or disease coarsens or fragments the architecture.
- **Multi-omics** — *do different molecular layers share the same architecture?*
  Additional layers (transcripts, metabolites) loading onto shared fields.
- **3D** — volumetric coordinates, for intact-tissue or thick-biofilm imaging.

## Explicit non-goals

To keep the method focused and honest, MESH will **not**:

- rebuild or wrap joint species distribution models (HMSC, gllvm, sjSDM);
- perform any bioinformatics (mapping, assembly, binning, variant calling);
- treat the **contig** as a modelling unit (contig identity is a future linkage
  covariate, not a level of the model);
- add SPDE / sparse / inducing-point / variational-GP machinery, or
  GPU-specific code, at the current data scale.

:::{admonition} Contributing a new model layer
:class: tip

Every new layer ships with (1) a generator of known truth in
{mod}`mesh.simulate` and (2) a recovery test that it recovers *its* parameters.
That is the contract that keeps an un-eyeballable method trustworthy — see
{doc}`methods/simulation`.
:::
