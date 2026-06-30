# Case 3 — Co-segregating functions on shared territories

> **In one line.** Several metabolic functions are measured across the same gut
> wall. Do they segregate at *one* scale or *several*, and *which functions share
> a territory*? MESH fits them **jointly** and reads out both the co-existing
> patch sizes and which function sits on which.

## The system

We extend [Case 1](functional-patch.md) from one function to a small **panel of
gene families** quantified across the same intestinal cross-section — the same
~200 microsamples at ~50 µm resolution, the same shared gene catalog, now with a
**count per (family, microsample)**.

Biology rarely organises everything at a single scale. Some functions track a
**fine micro-architecture** — a cross-feeding guild packed into a shared
microhabitat, tens of microns across. Others follow the **broad
epithelium-to-lumen gradient** — hundreds of microns. If two functions crest and
trough together over space, they are candidates for a **shared niche or
metabolic partnership**; if they tile complementary territory, they may
**compete or partition** it.

## The question

> **How many spatial scales are present, and which functions share each one?**
> One range hides this. Fitting the families jointly lets MESH resolve *two*
> patch sizes at once and assign each function to the scale it actually
> segregates at — turning "are these functions related?" into a measurement.

## The data MESH sees

The same long-format table as Case 1, now with **several `feature_id`s** sharing
the samples — the [schema](../methods/schema.md) already requires a catalog
shared across samples, which is exactly this. Here we simulate four families over
**two** true scales: a fine **80 µm** field carrying two families and a broad
**280 µm** field carrying the other two.

```python
from mesh import simulate_coregionalized, validate_table

# Stand-in for an upstream multi-feature counts table. Truth: two co-existing
# scales (80 µm and 280 µm); families feat0/feat1 load on the fine field,
# feat2/feat3 on the broad field.
sim = simulate_coregionalized(
    n_samples=220,            # ~220 microsamples, shared across families
    ranges=(80.0, 280.0),     # the two true patch sizes (µm) — unknown in practice
    domain=800.0,             # 0.8 mm sampled field
    seed=0,
)
df = sim.table                # in practice: df = pd.read_parquet("microsamples.parquet")
validate_table(df)            # the shared-catalog contract is the multi-feature contract
```

## The analysis

The coregionalization model fits all families at once over `n_fields` shared
Matérn fields. It needs a little more warmup than a single-field fit — the shared
fields couple the features — so we raise the warmup and `target_accept_prob`
(the same settings the recovery test uses).

```python
from mesh import (
    coregion_counts_arrays, coregion_feature_order,
    coregionalized_negbinomial, fit_model, summarize_loadings,
)

arrays = coregion_counts_arrays(df)          # coords, counts (J×n), log-offset (J×n)
idata = fit_model(
    coregionalized_negbinomial,
    num_warmup=800, num_samples=800, num_chains=2,
    target_accept_prob=0.95, n_fields=2, seed=0, **arrays,
)

feature_ids = coregion_feature_order(df)     # loadings rows follow this order
print(summarize_loadings(idata, feature_ids=feature_ids))
```

The figure shows the fit at a glance — an input family, the loadings (the
answer), and the two reconstructed fields at their different scales:

```python
import matplotlib.pyplot as plt
from mesh import plot_samples, plot_loadings, plot_field, posterior_field_mean

fig, axes = plt.subplots(2, 2, figsize=(11, 9))
plot_samples(df[df["feature_id"] == feature_ids[0]], value="count", ax=axes[0, 0])
plot_loadings(idata, feature_ids=feature_ids, ax=axes[0, 1])   # which feature → which field
plot_field(sim.coords, posterior_field_mean(idata, var_name="field0"), ax=axes[1, 0])
plot_field(sim.coords, posterior_field_mean(idata, var_name="field1"), ax=axes[1, 1])
fig.tight_layout()
```

## The answer

```{figure} ../_static/cases/co-segregation.png
:alt: Four-panel coregionalization fit — input counts, loadings heatmap, two fields.
:width: 100%

The joint fit. **Top-left:** input counts for one family over the 0.8 mm field.
**Top-right:** the loading magnitudes — each family's assigned field is outlined;
two families sit on the fine field, two on the broad field. **Bottom:** the two
reconstructed latent fields, the left one fine-grained (≈80 µm), the right one
broad (≈280 µm).
```

```text
field 0 (fine):   range ≈  74 µm   (95% CrI  60–91 µm)    truth  80
field 1 (broad):  range ≈ 286 µm   (95% CrI 195–404 µm)   truth 280
```

```text
feature  field  abs_mean  assigned_field
  feat0      0      1.50            0
  feat0      1      0.36            0
  feat1      0      1.25            0
  feat1      1      0.34            0
  feat2      0      0.09            1
  feat2      1      1.54            1
  feat3      0      0.07            1
  feat3      1      1.23            1
```

Reading it:

- **Two scales were recovered, not averaged into one.** The ordered ranges
  separate a fine field near the true 80 µm and a broad field near the true
  280 µm, each with its own credible interval (the chains converged,
  $\hat R \approx 1.0$ on the ranges).
- **Each family is assigned to the right field.** In the loadings table the
  largest `abs_mean` per family points at its generating field, so `feat0`/`feat1`
  read as fine-scale and `feat2`/`feat3` as broad — exactly the co-segregating
  groups the data were built with.
- Families sharing a field **share a territory**: biologically, the fine-field
  pair are candidates for a co-localised guild, the broad-field pair for
  functions riding the gross epithelium-to-lumen gradient.

:::{admonition} Sign is free; magnitude carries the assignment
:class: note

The per-field sign is not identified, so read assignment from the loading
**magnitudes** (`abs_mean` / the heatmap), not the signed values. The *relative*
sign of two families **within one draw** still distinguishes co- from
anti-segregation — see [reading the loadings](../guide/read-your-results.md#fitting-several-features-jointly-reading-the-loadings)
and {func}`mesh.summarize_loadings`.
:::

## How this case grows with MESH

Joint fitting over shared fields is the [M1+ milestone](../roadmap.md). The same
panel will support more as MESH expands:

- [ ] **More than two scales** — raise `n_fields`; the ordered ranges extend to a
      hierarchy of co-existing grains.
- [ ] **A clean multi-scale variance partition** — attribute *one* family's
      variance across the fine and broad fields, the principled version of the
      single-fit determinism index in [Case 1](functional-patch.md).
- [ ] **Organizational × spatial structure** — group families by function and by
      taxon-of-residence and ask whether scale is set by the function or by who
      carries it.
- [ ] **Allele features** — the same model fits linked **variants**, finding
      those that travel together as a strain/haplotype
      ([Case 2](strain-territory.md)).

Each capability arrives with a recovery test (see
[simulation & recovery](../methods/simulation.md)); this page will be extended as
they land.
```

This case study was generated by `examples/plot_cases.py` (the
`co_segregation()` function), which fits the model with the parameters above and
writes `docs/_static/cases/co-segregation.png`.
