# Roadmap

MESH is built in milestones, each gated by a simulation-based recovery test. The
package today implements **Milestone 0/1**; later milestones extend the model
without changing its interpretable core (patch sizes with credible intervals).

## Done — M0/M1: single-field, two likelihoods

- Matérn 3/2 exact-GP field with a non-centered parameterization.
- {func}`mesh.spatial_betabinomial` — coverage-aware allele frequencies (the
  `m0` seed).
- {func}`mesh.spatial_negbinomial` — abundance counts with a depth offset (the
  minimal counts-table entry point).
- Input {doc}`schema <methods/schema>` with loud validation.
- Recovery tests for both models (true range inside the 95% interval).

## Next — M1+: coregionalization (separating scales)

The immediate next step is **shared loadings across multiple features** so that
features can co-segregate, and a model with **two Matérn fields at different
ranges** that the inference can **separate**. The gating test: simulate two
co-existing scales and confirm the posterior resolves both ranges and assigns
features to the right field.

## Later milestones

These are described in the design overview and are intentionally **not** yet
implemented:

- **Multi-scale fields** — a variance partition of each feature across several
  spatial scales.
- **Crossed organizational × spatial hierarchy** — gene-centric grouping
  (function, operon/MGE, taxon of residence) crossed with spatial scale.
- **Multi-host hierarchical GP** — partial pooling of patch sizes across hosts /
  individuals.
- **Multi-omics** — additional molecular layers loading onto shared fields.
- **3D** — volumetric coordinates.

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
