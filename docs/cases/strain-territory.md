# Case 2 — A strain's territory

> **In one line.** How far does a sub-species genotype stay locally dominant
> before another takes over? MESH estimates the spatial scale of an *allele
> frequency* — coverage-aware — turning "strain territory" into microns with a
> credible interval.

## The system

Within a single species, co-existing **strains** differ at variant sites. Pick
one diagnostic site: at each microsample we observe how many reads carry the
**alternate** allele out of the total **coverage** at that position. Strains are
not perfectly mixed — competition, priority effects and local dispersal carve the
habitat into territories where one genotype prevails.

Crucially, the evidence is uneven. At this resolution a site might be covered by
**~20× in one microsample and ~80× in another** (here coverage spans ≈ 21–80×).
An allele frequency from 21 reads is far less certain than one from 80, and a
naïve analysis that treats both as equally trustworthy will invent structure that
is not there.

## The question

> **At what spatial scale do sub-species genotypes segregate?**
> A small range means tightly interleaved strains (territories of tens of
> microns); a large range means each genotype dominates a broad region. This is
> the reach that distinguishes MESH from species-level methods: it works
> **below** the species, on allele frequencies, **accounting for coverage**.

## The data MESH sees

The same long-format table, now with the optional **`ref`/`alt`** columns. For
the allele model, `count` is the alternate-allele count and `depth` is the site
coverage — the number of trials in the beta-binomial (see
[the schema](../methods/schema.md)).

```python
from mesh import simulate_allele, validate_table

# Stand-in for an upstream variant table; truth: a 300 µm strain territory.
sim = simulate_allele(
    n_samples=200,     # ~200 microsamples
    range_=300.0,      # true territory size (µm) — unknown in a real study
    eta=1.2,           # strength of frequency variation (logit scale)
    domain=1500.0,     # 1.5 mm sampled field
    coverage=40,       # nominal coverage; actual spans ≈ 21–80×
    seed=2,
)
df = sim.table         # in practice: df = pd.read_parquet("variant_sites.parquet")
validate_table(df, require_allele=True)    # allele contract: needs ref/alt
```

## The analysis

```python
from mesh import allele_arrays, fit_model, spatial_betabinomial, summarize_range

arrays = allele_arrays(df)                 # coords, alt_count, total_count
idata = fit_model(
    spatial_betabinomial,
    num_warmup=500, num_samples=500, num_chains=2,
    seed=2, **arrays,
)
print(summarize_range(idata))
```

The same four views as [Case 1](functional-patch.md), but the input map now
shows the **alt-allele frequency** (`count / depth`) rather than a raw count:

```python
import matplotlib.pyplot as plt
from mesh import (
    plot_samples, plot_range_posterior, plot_field,
    plot_matern_correlation, posterior_field_mean,
)

fig, axes = plt.subplots(2, 2, figsize=(11, 9))
plot_samples(
    df, value="count", as_frequency=True, cmap="magma",
    ax=axes[0, 0], title="Input: alt-allele frequency",
)
plot_range_posterior(idata, truth=300.0, ax=axes[0, 1])
plot_field(
    sim.coords, posterior_field_mean(idata),
    ax=axes[1, 0], title="Inferred frequency field",
)
plot_matern_correlation(idata, ax=axes[1, 1])
fig.tight_layout()
```

## The answer

```{figure} ../_static/cases/strain-territory.png
:alt: Four-panel figure of the strain-territory fit.
:width: 100%

The fit at a glance. **Top-left:** the observed alt-allele frequency over the
1.5 mm field. **Top-right:** the territory-size posterior — wider than Case 1's,
the honest cost of variable coverage. **Bottom-left:** the reconstructed logit
frequency field. **Bottom-right:** the implied spatial correlation decay.
```


```text
 parameter   mean  median    sd  hdi_low  hdi_high  hdi_prob  r_hat  ess_bulk
     range  310.7   303.5  62.9    202.7     432.7      0.95  1.002     271
```

**Strain territory ≈ 311 µm (95% CI 203–433 µm).** Reading it:

- The interval brackets the simulated truth (300 µm) and the chains converged.
- The interval is wider than in [Case 1](functional-patch.md) — both because the
  territory is larger relative to the 1.5 mm field (less room for many
  decorrelations) and because allele frequencies under variable coverage carry
  less information per site than abundances. **That uncertainty is the honest
  result**, not a nuisance to be hidden.
- Coverage-awareness is doing real work: low-coverage microsamples contribute
  proportionally less, so the territory size reflects confident allele calls
  rather than thin, noisy ones.

## Function meets genotype — today

Even without a joint model, you can put the two cases together. With both fits in
hand, {func}`mesh.plot_scale_comparison` overlays the function's patch
([Case 1](functional-patch.md)) and the strain's territory on a single axis, so
you compare **credible intervals, not point estimates** (to infer co-segregation
*jointly* instead, see [coregionalization](co-segregation.md)):

```python
from mesh import plot_scale_comparison

plot_scale_comparison(
    [idata_function, idata_genotype],          # the two fits from each case
    labels=["function patch (Case 1)", "strain territory (Case 2)"],
)
```

```{figure} ../_static/cases/function-vs-genotype.png
:alt: Two overlaid range posteriors — function patch versus strain territory.
:width: 80%
:align: center

The function organises at a *finer* scale (≈ 175 µm) than the strain territory
(≈ 311 µm), though the intervals overlap. A function patch that is wider than any
single strain's territory would imply the function is shared across genotypes;
here the genotype's reach is the broader of the two. (Different simulated
systems — illustrative of the comparison, not a claim that these two features
co-occur.)
```

## How this case grows with MESH

Today this resolves **one** variant site at **one** scale. The same variant
dataset will support more as MESH expands:

- [x] **Linked variants** ([coregionalization](co-segregation.md)) — *available
      now*: fit many features jointly on shared fields and find those that share a
      territory, i.e. that travel together as a **strain/haplotype**. Demonstrated
      for functions in [Case 3](co-segregation.md); the same model applies to
      allele features.
- [ ] **Residence covariate** — annotate each variant by the contig / mobile
      element it sits on and ask whether territory size is set by the genomic
      context (a future *linkage* covariate, not a modelling unit).
- [x] **Function meets genotype (scale)** — *available now*: compare a function's
      patch and a genotype's territory by overlaying their ranges (above). To go
      further, [coregionalization](co-segregation.md) puts features on **shared
      fields** so co-segregation is inferred jointly rather than compared after
      the fact — see [Case 3](co-segregation.md).
- [ ] **Across hosts** — compare strain-territory sizes between individuals.
- [ ] **3D** — territories as volumes.

As in [Case 1](functional-patch.md), each new capability ships with a recovery
test, and this page will gain the matching analysis when it lands.
