# Case studies

These worked cases show MESH end-to-end on biologically motivated questions:
the system, the scientific question, the table MESH consumes, the analysis, and
how to read the answer. They are **living examples** — each one is written to
*grow* as MESH gains capabilities, so the same biological story will gain new
analyses at later [milestones](../roadmap.md) (coregionalization, multi-scale
partitions, multi-host pooling, 3D).

:::{admonition} These cases use simulated data
:class: note

MESH is [inference-only](../biology/sampling-design.md): it consumes a validated
table from an upstream pipeline. To keep the cases self-contained and
reproducible, each one **simulates** a table with a *known* patch size using
{mod}`mesh.simulate`, then recovers it. With real data you would replace the
`simulate_*` call with `pd.read_parquet(...)` and {func}`mesh.validate_table`;
everything downstream is identical.

The figures embedded below are produced by the exact code shown on each page;
regenerate them with `python examples/plot_cases.py` (writes into
`docs/_static/cases/`).
:::

## The two cases

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item-card} {octicon}`beaker` A functional patch in the gut wall
:link: functional-patch
:link-type: doc

*How big is a metabolic-function patch?* Gene **abundance** across an intestinal
cross-section, modelled with the negative-binomial likelihood.
:::

:::{grid-item-card} {octicon}`git-branch` A strain's territory
:link: strain-territory
:link-type: doc

*How far does a genotype stay dominant before it is replaced?* **Allele
frequencies** at a variant site, modelled coverage-aware with the beta-binomial
likelihood — sub-species resolution.
:::

::::

## How to read a case

Every case follows the same five beats, and ends with a **"How this case grows"**
section tied to the roadmap:

1. **The system** — the biology and the sampling design.
2. **The question** — what we want to know, phrased as a spatial-scale question.
3. **The data** — what the table looks like.
4. **The analysis** — the MESH calls, runnable as written.
5. **The answer** — the patch size, its credible interval, and what it means.

```{toctree}
:hidden:

functional-patch
strain-territory
```
