# MESH

**Metagenomic Ecology across Spatial Hierarchies**

MESH infers the **spatial scale** at which microbial functions and genetic
variants segregate across space, from spatially-resolved shotgun metagenomics.

Microbial communities are not spatially uniform. Functions and genotypes form
patches, and the *size* of those patches is itself a biological signal: it
reflects dispersal, competition, host structure and the physical architecture of
the habitat. MESH estimates that patch size directly, with uncertainty, and tells
you which molecular features share a scale.

The engine is a **hierarchical Bayesian spatial latent-factor model**. Each
latent field is a Gaussian process with a **Matérn covariance whose range
parameter is the patch size, in microns**. Molecular features — gene/contig
abundance now, SNP/strain frequencies later — load onto these fields. The
headline outputs are interpretable by construction:

- **patch sizes (ranges) with credible intervals**,
- **which features co-segregate**, and
- (in later milestones) a **variance partition across scales**.

Inference runs on [NumPyro](https://num.pyro.ai/) / [JAX](https://jax.dev/)
using **exact** Gaussian processes.

:::{admonition} MESH is inference-only
:class: important

MESH contains **no bioinformatics** — no read mapping, assembly, binning, or
variant calling. It consumes a validated, analysis-ready table and produces
posterior summaries. The bioinformatics pipeline is a separate project; the two
meet **only** through the documented [input schema](methods/schema.md).
:::

## Where to start

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item-card} {octicon}`rocket` Getting started
:link: getting-started/installation
:link-type: doc

Install MESH and recover a known patch size from simulated data in a few lines.
:::

:::{grid-item-card} {octicon}`mortar-board` Step-by-step guide
:link: guide/index
:link-type: doc

New to this? A teaching path from data to results that explains *why* at every
step — no statistics or spatial-ecology background assumed.
:::

:::{grid-item-card} {octicon}`beaker` The biology
:link: biology/index
:link-type: doc

What "spatial scale of segregation" means, and why patch size is the quantity
worth estimating.
:::

:::{grid-item-card} {octicon}`cpu` The method
:link: methods/index
:link-type: doc

The Matérn Gaussian-process model, coverage-aware likelihoods, and how
inference is done.
:::

:::{grid-item-card} {octicon}`beaker` Case studies
:link: cases/index
:link-type: doc

Two worked, biology-driven cases — a functional patch and a strain's territory —
that grow as MESH gains capabilities.
:::

:::{grid-item-card} {octicon}`code` API reference
:link: api/index
:link-type: doc

Module-by-module reference generated from the source docstrings.
:::

::::

```{toctree}
:hidden:
:caption: Getting started

getting-started/installation
getting-started/quickstart
```

```{toctree}
:hidden:
:caption: Step-by-step guide

guide/index
```

```{toctree}
:hidden:
:caption: Biological background

biology/index
```

```{toctree}
:hidden:
:caption: Methods & implementation

methods/index
```

```{toctree}
:hidden:
:caption: Case studies

cases/index
```

```{toctree}
:hidden:
:caption: Reference

api/index
roadmap
glossary
```
