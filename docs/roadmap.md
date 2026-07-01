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

## Done — M1+: coregionalization (separating scales)

*Biological question: **which features share a territory, and do they cooperate
or compete there?*** Implemented as a linear model of coregionalization:
**shared loadings across multiple features** so that features can co-segregate,
over **several Matérn fields at different (ordered) ranges** that the inference
**separates**. The sign and structure of the shared loadings reads out ecology
directly: features that occupy the *same* patches suggest cross-feeding,
syntrophy or a shared niche; features that *anti-correlate* suggest competition
or niche partitioning.

- {func}`mesh.coregionalized_negbinomial` — `n_fields` unit-variance Matérn
  fields at ordered ranges, shared across features through a feature×field
  loadings matrix.
- {func}`mesh.simulate_coregionalized` — known-truth generator over two scales.
- {func}`mesh.summarize_loadings` — which feature sits on which field
  (sign-invariant assignment).
- Gating test (`tests/test_coregionalization.py`): two co-existing scales, both
  ranges recovered inside their 95% intervals and every feature assigned to the
  field it was generated from.

## Done — boundary sharpness: crisp edges vs. gradients

*Biological question: **do patches have crisp edges or fade as gradients?***
Crisp boundaries point to competitive exclusion or a physical/biofilm barrier;
smooth gradients point to diffusion-limited gradients of O₂, pH or nutrients.
The readout is the Matérn smoothness ν: small ν = rough field = crisp edges,
large ν = smooth field = gradients.

Arbitrary continuous ν needs the fractional-order modified Bessel function
`K_ν`, which has no autodiff-stable JAX implementation, so MESH uses the
**closed-form family** {1/2, 3/2, 5/2} plus the squared-exponential ν→∞ limit
and **selects among them by LOO model comparison** — each fit keeps one fixed ν,
so the latent field stays unambiguous (no discrete latent marginalised inside a
chain).

- {func}`mesh.matern_kernel` — parametric Matérn covariance over the closed-form
  {data}`mesh.MATERN_NU` smoothnesses (`0.5, 1.5, 2.5, math.inf`); the range
  keeps its meaning (patch size in microns) across every member.
- `nu` argument on {func}`mesh.spatial_negbinomial`, {func}`mesh.spatial_betabinomial`,
  {func}`mesh.gp_field` and {func}`mesh.simulate_counts` (the known-truth
  generator records `truth["nu"]`).
- {func}`mesh.compare_smoothness` — fit the same data under each fixed-ν kernel
  and rank them by PSIS-LOO ({func}`arviz.compare`); the winning ν is the
  boundary-sharpness readout.
- Gating test (`tests/test_smoothness.py`): the closed forms and the
  roughness-vs-ν ordering are checked deterministically (SE limit included), and
  a simulated **rough** field is recovered as the rough kernel decisively by LOO.
  Roughness is the reliably identifiable direction — confirming *extra* smoothness
  from noisy counts is intrinsically weaker (a rough kernel fits smooth data too).

## Done — direction (anisotropy): organising along a host axis

*Biological question: **does a feature organise along a host axis?*** The
proximal–distal gut, the crypt–villus axis, or depth into a biofilm all impose a
*direction* on spatial structure — patches elongated along one axis rather than
round. The readout is the **anisotropy ratio** `ell_x/ell_y` (with a credible
interval) and *which* axis is the long one.

MESH models this as an **axis-aligned** anisotropic Matérn field: a separate
patch size along each coordinate axis, staying inside the single-fit paradigm
(same input table, same Matérn-GP machinery). The axes are not rotated — orient
the sampling frame to the host axis; a free rotation is a later extension.

- {func}`mesh.anisotropic_matern_kernel` / {func}`mesh.anisotropic_scaled_distances`
  — a Matérn covariance with a **per-axis range** `(ell_x, ell_y)`; equal
  lengthscales reduce it exactly to the isotropic {func}`mesh.matern_kernel`.
- {func}`mesh.anisotropic_negbinomial` — the directional abundance model,
  parameterised by an overall (geometric-mean) `range` and a signed log
  anisotropy `log_ratio` with a prior centred at isotropy, so direction must be
  supported by the data ({func}`mesh.gp_field_anisotropic` is the shared field).
- {func}`mesh.simulate_anisotropic` — known-truth generator (default elongated
  3× along `x`); {func}`mesh.summarize_anisotropy` reports the per-axis patch
  sizes, the folded anisotropy ratio and which axis is longer;
  {func}`mesh.plot_anisotropy` overlays the two per-axis posteriors.
- Gating test (`tests/test_anisotropy.py`): both axis ranges recovered inside
  their 95% intervals, the anisotropy resolved and its direction recovered.

## Later milestones — more axes of architecture

Each item below is a distinct architecture axis, framed by the biological
question it answers. They are described in the design overview and are
intentionally **not** yet implemented. The first one below stays inside the
current single-fit paradigm (same input table, same Matérn-GP machinery) and is
the cheapest to add.

- **Stationarity** — *is the grain constant, or are there zones?* Finer near a
  surface, coarser in a lumen. A non-stationary range, or region-wise fits.
- **Multi-scale fields** — *is the architecture hierarchical?* A fine mosaic
  nested inside broad zones — a variance partition of each feature across several
  spatial scales (the clean version of the single-fit determinism index above).
- **Hierarchical coregionalization (genomes × gene families × traits)** — *does
  function track the organism, or move independently of it, and is a multi-gene
  capacity realised in one genome or assembled across the community?* Turns the
  flat coregionalization loadings into a **structured** matrix built from
  biological membership: each **genome** carries its own spatial field (organism
  as an entity, not a grouping level), **gene families** get a residual
  cross-genome field, and **traits** (KEGG modules, MetaCyc pathways, BGCs) — sets
  of co-functioning gene families, gated by genome-inferred completeness — get
  their own fields. Each level reports a patch size, a coregionalization matrix
  and a within-genome-vs-distributed variance partition. Fully specified in
  {doc}`methods/coregionalization_hierarchy`; needs annotation/membership tables
  added to the {doc}`schema <methods/schema>`.
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
