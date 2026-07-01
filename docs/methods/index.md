# Methods & implementation

This section documents how MESH turns a table of coverage-aware observations into
a posterior over patch sizes. It is written to be read alongside the source: each
page links to the relevant functions in the [API reference](../api/index.md).

## The pipeline in one diagram

```{mermaid}
flowchart TD
    T["Long-format table<br/>(mesh.schema)"] -->|validate_table| V[Validated table]
    V -->|counts_arrays / allele_arrays| A["Model arrays<br/>coords, counts/alleles, offsets"]
    A --> M{"Observation model"}
    M -->|abundance| NB["spatial_negbinomial"]
    M -->|alleles| BB["spatial_betabinomial"]
    NB --> G
    BB --> G["Shared latent field<br/>f = η·(L z), Matérn 3/2"]
    G -->|fit_model → NUTS| ID[ArviZ InferenceData]
    ID -->|summarize_range| S["Patch size + credible interval"]
```

## Design commitments

The implementation makes a few deliberate choices, each motivated by keeping the
output interpretable and the inference honest at the scale of data MESH targets:

:::{list-table}
:header-rows: 1
:widths: 35 65

* - Choice
  - Why
* - **Exact** Gaussian processes (dense Matérn covariance + Cholesky)
  - Datasets are a few hundred to ~2k patches, so exact GPs are cheap and avoid
    approximations (SPDE, inducing points, variational GPs) that would blur the
    range estimate.
* - **Non-centered** parameterization, $f = \eta\,(L z)$
  - Keeps the NUTS geometry well-behaved; it is the first mitigation when the GP
    posterior is awkward.
* - **Coverage-aware** likelihoods (negative-binomial, beta-binomial)
  - Down-weights low-evidence observations instead of treating them as confident.
* - **Single shared field machinery** across models
  - The abundance and allele models differ only in their observation layer; the
    spatial core is identical and tested once.
* - **Simulation-first** development
  - Every model ships with a generator of known truth and a recovery test — the
    primary correctness guarantee for outputs that cannot be eyeballed.
:::

## Scope of the current milestone

The single-feature models (M0/M1) fit **one** Matérn field with **one** range.
The **coregionalization** model (M1+) extends this to several features over
**multiple ordered fields**, separating co-existing scales
({func}`mesh.coregionalized_negbinomial`). The next step turns that *flat*
loadings matrix into a **structured** one that follows the biological
organisation of the features — genomes, gene families and traits (modules /
pathways / BGCs); it is specified in {doc}`coregionalization_hierarchy` ahead of
implementation. Still ahead beyond that: multi-host hierarchical GPs and
multi-omics — later milestones (see the [roadmap](../roadmap.md)), intentionally
**not** present yet. There is no SPDE/sparse/variational-GP machinery and no
GPU-specific code.

```{toctree}
:hidden:

model
coregionalization_hierarchy
inference
schema
simulation
```
