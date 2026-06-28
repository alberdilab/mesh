# Biological background

## The question MESH answers

Microbial life is spatially organised. In a gut, a soil aggregate, a biofilm or
a root surface, the same taxon and the same gene are abundant *here* and rare a
few hundred microns *there*. That structure is not noise — it encodes the
processes that generate and maintain microbial diversity: dispersal limitation,
local competition and cross-feeding, host-imposed gradients (oxygen, pH, mucus,
immune pressure) and the physical architecture of the habitat.

The central quantity is the **scale of segregation**: over what distance does a
function or a genotype stay correlated before it decorrelates into a different
neighbourhood? MESH calls that distance the **patch size** and estimates it,
with uncertainty, directly from spatially-resolved shotgun metagenomics.

## Why patch size, specifically

A patch size is a single, interpretable number in physical units (microns). It
is comparable across studies, across features and — eventually — across hosts. It
turns a vague notion ("this community is patchy") into a measurable parameter
with a credible interval. Two genes with the same patch size plausibly respond
to the same structuring process; two genes at very different scales do not.

Compared with the usual summaries of spatial metagenomics:

```{list-table}
:header-rows: 1
:widths: 30 70

* - Approach
  - What it gives you
* - Alpha/beta diversity vs. distance
  - Whether communities differ with distance, but not the *scale* or its
    uncertainty, and not per-feature.
* - Variograms / Moran's I per feature
  - A descriptive range, but no coverage-awareness, no sharing of information
    across features, and no propagated uncertainty.
* - Joint species distribution models (HMSC, gllvm, sjSDM)
  - Co-occurrence and environmental response, but they are **not**
    spatial-scale-explicit and do not reach sub-species resolution.
* - **MESH**
  - A **scale-explicit**, **coverage-aware** posterior over patch size(s), with
    credible intervals, shared latent structure across features, and a path to
    SNP/strain resolution.
```

:::{admonition} MESH is not another joint species distribution model
:class: note

MESH does not rebuild or wrap HMSC, gllvm or sjSDM. Its distinctive contribution
is **spatial-scale-explicit, coverage-aware** inference that reaches down to
**sub-species (SNP/strain)** resolution. The modelling unit is the *feature*
(gene/contig as a count or allele substrate), not the species.
:::

## How the data are generated (conceptually)

The motivating design is dense spatial microsampling of a tissue or substrate —
for example, on the order of **~200 microsamples per intestinal cross-cut at
~50 µm resolution**. Each microsample is shotgun-sequenced and quantified
against a **shared feature catalog**, yielding, per feature and per location:

- a **count** (reads assigned to a gene/contig), with a per-sample sequencing
  **depth** and a feature **length** that set the expected count; and/or
- an **allele** observation (alternate vs. total reads at a variant site), where
  the **coverage** sets how much that site can tell us.

MESH works in **2D now** and is designed to extend to **3D** later.

Read on:

- [Spatial scales and patch size](spatial-scales.md) — what the Matérn range
  means biologically and how to read it.
- [Sampling design and the data](sampling-design.md) — what MESH expects from the
  upstream pipeline and why coverage matters.
```{toctree}
:hidden:

spatial-scales
sampling-design
```
