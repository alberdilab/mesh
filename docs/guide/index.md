# Step-by-step guide

This guide is the **teaching path** through MESH. It assumes you can run Python,
but it does **not** assume you are a statistician, a Bayesian, or a spatial
ecologist. Every modelling choice is explained in plain language, with the
statistics introduced only when you need it and always tied back to a question
you actually have about your data.

If you just want the shortest possible working example, the
[Quickstart](../getting-started/quickstart.md) is six lines. This guide is the
opposite: it walks slowly, explains *why* at each step, and is meant to be read
in order the first time you use MESH.

## What MESH does, in one paragraph

You sampled a microbial community at many points in space and measured the same
genes or variants at each point. Some genes are common *here* and rare a little
way *there*. MESH answers one question: **over what distance does a feature stay
similar before it changes into a different neighbourhood?** That distance is the
**patch size**, measured in microns. MESH estimates it with an honest measure of
uncertainty (a credible interval), so you can say "this function is organised at
roughly 200 µm" and know how sure you are.

:::{admonition} The one piece of jargon worth learning now
:class: tip

The patch size is the **range** of a *Gaussian process*. You do not need to
understand Gaussian processes to use MESH. Think of the range as a dial: small
range = fine-grained mosaic, large range = broad zones. Everything MESH reports
is a statement about that dial. See [spatial scales](../biology/spatial-scales.md)
for the intuition and the [glossary](../glossary.md) for the terms.
:::

## The four steps

```{mermaid}
flowchart LR
    A["1 · Prepare your data<br/>one tidy table"] --> B["2 · Choose your model<br/>counts or alleles?"]
    B --> C["3 · Run the fit<br/>let the sampler work"]
    C --> D["4 · Read your results<br/>number + interval + plots"]
```

1. **[Prepare your data](prepare-your-data.md)** — get your measurements into the
   one table shape MESH expects, and let the validator catch mistakes early.
2. **[Choose your model](choose-your-model.md)** — decide whether you are
   modelling *abundances* or *allele frequencies*, and understand *why* each gets
   a different likelihood (this is the "why negative binomial?" page).
3. **[Run the fit](run-your-first-fit.md)** — turn the table into a fitted model,
   with every sampler setting explained.
4. **[Read your results](read-your-results.md)** — interpret the patch-size
   number, its credible interval, the diagnostic checks, and the four standard
   plots — and learn when *not* to trust an estimate.

## Which model do I need? (decide this first)

Almost everything downstream follows from one question: **what did you measure at
each location?**

```{mermaid}
flowchart TD
    Q{"What is the number<br/>at each location?"}
    Q -->|"reads assigned to a gene/contig<br/>(how MUCH of it is there)"| NB["Abundance model<br/><b>spatial_negbinomial</b>"]
    Q -->|"alt-allele reads out of total<br/>(which VERSION is there)"| BB["Allele model<br/><b>spatial_betabinomial</b>"]
```

```{list-table}
:header-rows: 1
:widths: 30 35 35

* - You measured…
  - …so the question is
  - …and the model is
* - **Gene/contig abundance** (read counts per feature)
  - *How much* of this function lives where?
  - {func}`mesh.spatial_negbinomial`
* - **Allele frequency** (alt reads out of coverage at a variant site)
  - *Which genetic variant* dominates where?
  - {func}`mesh.spatial_betabinomial`
```

If you have both, run MESH twice — once per data type. The spatial engine is
identical; only the final "how the count was generated" layer differs. The next
two pages explain exactly why those two layers are the right ones.

```{toctree}
:hidden:

prepare-your-data
choose-your-model
run-your-first-fit
read-your-results
```
