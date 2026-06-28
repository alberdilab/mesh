# Sampling design and the data

MESH is **inference-only**. It begins where the bioinformatics ends: with a
validated, analysis-ready table. This page describes what that table represents
biologically and what a good sampling design looks like. The precise column
contract and its validation are in [the input schema](../methods/schema.md).

## What a row is

The input is a **long-format** table. Each row is one **feature** observed in one
**sample** (microsample) at a known location:

```{list-table}
:header-rows: 1
:widths: 18 82

* - Column
  - Meaning
* - `feature_id`
  - Gene/contig identifier (or variant site for the allele model). The catalog
    must be **shared** — the same set of `feature_id`s in every sample.
* - `sample_id`
  - Microsample identifier.
* - `x`, `y`
  - Spatial coordinates of the microsample, in **microns**.
* - `count`
  - Reads assigned to the feature (abundance model), or alternate-allele reads
    (allele model).
* - `depth`
  - Per-sample sequencing **depth** (abundance model) or per-site **coverage**
    (allele model).
* - `length`
  - Feature length in base pairs.
* - `ref`, `alt`
  - *Optional*, allele model only: reference and alternate read counts.
```

## Why a shared catalog is mandatory

MESH places a latent spatial field over the **same** feature measured at many
locations. That only makes sense if a feature means the same thing everywhere, so
the feature catalog must be identical across all samples. Per-sample catalogs —
where each sample reports its own ad-hoc set of features — break the comparison
and are **rejected by validation** with an explicit error. Upstream, this means
quantifying every microsample against one common reference (gene catalog, contig
set, or variant panel).

:::{admonition} The contig is *not* a modelling unit
:class: warning

Features are genes/contigs used as a **count (or allele) substrate**. Contig
identity is a *future* linkage/residence covariate — which gene lives on which
contig — not a level of the model. MESH does not model contigs as entities.
:::

## Why coverage matters: coverage-aware inference

Sequencing gives you *evidence*, not *truth*, and the amount of evidence varies
enormously between locations and features. MESH builds this directly into the
likelihood:

- **Abundance (counts).** The expected count scales with the per-sample
  sequencing **depth** and the feature **length**. A larger library or a longer
  gene yields more reads at the *same* underlying abundance, so MESH carries
  `log(depth) + log(length)` as a fixed **offset**. Counts are modelled with a
  **negative binomial**, which absorbs the extra-Poisson variability typical of
  metagenomic counts.

- **Alleles (frequencies).** At a variant site, an allele frequency estimated
  from 4 reads is far less certain than one from 400. MESH models alternate
  counts with a **beta-binomial** given the site **coverage**, so low-coverage
  sites contribute little and are not mistaken for confident extremes. This is
  what *coverage-aware* means in practice.

The consequence for design: **record depth/coverage and feature length
faithfully**, and prefer broad, even spatial coverage of locations over deep
sequencing of a few. The patch size is identified by how correlation decays *with
distance*, so the spatial arrangement of samples is the part of the design that
most affects what MESH can resolve.

## Design rules of thumb

- **Resolution vs. range.** To resolve a patch size $\ell$, sample spacing should
  be comfortably **smaller** than $\ell$, and the overall sampled field should be
  several times **larger** than $\ell$. Roughly: spacing $\ll \ell \ll$ domain.
- **Number of locations.** A few hundred microsamples per cross-section (the
  motivating design uses ~200) is enough for exact-GP inference and to identify a
  range with a usable credible interval.
- **2D now, 3D later.** The current models work in 2D; the framework is designed
  to extend to 3D coordinates.

## What MESH deliberately does *not* do

MESH performs **no** read mapping, assembly, binning, or variant calling. Those
belong to a separate bioinformatics project. The two meet **only** through the
input table, which keeps the statistical method auditable and independent of any
particular upstream pipeline.
