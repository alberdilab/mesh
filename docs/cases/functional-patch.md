# Case 1 — A functional patch in the gut wall

> **In one line.** Over what distance is a metabolic *function* spatially
> coherent in the gut wall — fine patches or broad zones? MESH turns that into a
> patch size in microns, with a credible interval.

## The system

Consider a cross-section through the intestinal wall, sampled densely: on the
order of **~200 microsamples at ~50 µm resolution**, tiling the tissue from the
epithelial surface through the mucus layer into the lumen. Each microsample is
shotgun-sequenced and quantified against a **shared gene catalog**.

We focus on one function — say a **glycan-foraging gene family** (mucin/host-
glycan degradation). Its carriers exploit host-derived sugars near the mucus, so
biology predicts the function is *not* uniform: it should concentrate where its
substrate is, forming patches. The open question is the **size** of those
patches.

## The question

> **At what spatial scale does this function segregate?**
> If patches are tens of microns, the function tracks micro-architecture (crypts,
> mucus microstructure). If they are hundreds of microns, it reflects the broad
> epithelium-to-lumen gradient. The number discriminates between these
> mechanisms.

## The data MESH sees

A long-format table, one row per (feature, microsample), with a **count**, a
per-sample sequencing **depth**, and the gene **length** — the ingredients of the
coverage-aware abundance model (see [the schema](../methods/schema.md)). Counts
span a wide dynamic range (here ~1–900 reads), which the depth/length offset and
the negative-binomial likelihood handle directly.

```python
from mesh import simulate_counts, validate_table

# Stand-in for an upstream counts table; truth: a 150 µm functional patch.
sim = simulate_counts(
    n_samples=200,     # ~200 microsamples
    range_=150.0,      # true patch size (µm) — unknown in a real study
    eta=1.0,           # strength of spatial variation
    domain=1000.0,     # 1 mm sampled field
    seed=1,
)
df = sim.table         # in practice: df = pd.read_parquet("microsamples.parquet")
validate_table(df)     # enforce the input contract before modelling
```

## The analysis

```python
from mesh import counts_arrays, fit_model, spatial_negbinomial, summarize_range

arrays = counts_arrays(df)                 # coords, counts, log(depth)+log(length)
idata = fit_model(
    spatial_negbinomial,
    num_warmup=500, num_samples=500, num_chains=2,
    seed=1, **arrays,
)
print(summarize_range(idata))
```

The four standard views make the fit interpretable at a glance — the input, the
headline posterior, the reconstructed field, and what the range *means*:

```python
import matplotlib.pyplot as plt
from mesh import (
    plot_samples, plot_range_posterior, plot_field,
    plot_matern_correlation, posterior_field_mean,
)

fig, axes = plt.subplots(2, 2, figsize=(11, 9))
plot_samples(df, value="count", ax=axes[0, 0], title="Input: gene counts")
plot_range_posterior(idata, truth=150.0, ax=axes[0, 1])
plot_field(
    sim.coords, posterior_field_mean(idata),
    ax=axes[1, 0], title="Inferred functional field",
)
plot_matern_correlation(idata, ax=axes[1, 1])
fig.tight_layout()
```

## The answer

```{figure} ../_static/cases/functional-patch.png
:alt: Four-panel figure of the functional-patch fit.
:width: 100%

The fit at a glance. **Top-left:** the input counts over the 1 mm field.
**Top-right:** the patch-size posterior — the 95% interval brackets the
simulated truth (150 µm). **Bottom-left:** MESH's reconstruction of the hidden
functional field; it should look like a smoothed version of the input.
**Bottom-right:** the spatial correlation the range implies, with range
uncertainty as a band.
```


```text
 parameter   mean  median    sd  hdi_low  hdi_high  hdi_prob  r_hat  ess_bulk
     range  175.3   168.8  35.0    121.2     248.4      0.95  1.001     206
```

**Patch size ≈ 175 µm (95% CI 121–248 µm).** Reading it:

- The interval comfortably brackets the simulated truth (150 µm), and the chains
  converged ($\hat R \approx 1.00$).
- The scale is several times the ~50 µm sampling resolution and a fraction of the
  1 mm field — so it is **well resolved**: the design can see it, and it is not an
  artefact of the domain edge.
- Biologically, ~150–200 µm points to organisation at the scale of the broad
  epithelium-to-lumen architecture rather than the finest micro-features — the
  function forms zones, not pinpoints.

Because the likelihood is coverage-aware, low-depth microsamples are
automatically down-weighted: the patch size is driven by where the *evidence*
is, not by shallow samples masquerading as zeros.

## How this case grows with MESH

This single-function, single-scale analysis is Milestone 0/1. As MESH expands,
the *same gut-wall dataset* will support richer questions:

- [x] **Co-segregating functions** ([coregionalization](co-segregation.md)) —
      *available now*: fit several gene families jointly and ask *which functions
      share a patch size*, pointing to a common structuring process or a shared
      guild. Worked through in [Case 3](co-segregation.md).
- [ ] **Multiple scales at once** — partition a function's spatial variance
      across a fine and a coarse scale (micro-architecture *and* the gross
      gradient) instead of reporting one range.
- [ ] **Organizational × spatial structure** — group genes by function and by
      taxon-of-residence and ask whether scale is set by the function or by who
      carries it.
- [ ] **Across hosts** — pool patch-size estimates over individuals to separate a
      conserved tissue-level scale from host-to-host variation.
- [ ] **3D** — extend from a cross-section to a volume.

Each capability will arrive with a recovery test (see
[simulation & recovery](../methods/simulation.md)); this page will be extended
with the corresponding analysis as they land.
