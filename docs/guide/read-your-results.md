# Step 4 · Read your results

A fit gives you three things: a **number** (the patch size), an **interval**
(how sure MESH is), and **diagnostics** (whether to trust either). This page
reads all three in plain language, then shows the four standard plots and what
each one is telling you. The golden rule first:

:::{admonition} Check the diagnostics *before* you read the number
:class: important

A patch size from a fit that didn't converge is not a conservative estimate —
it is **meaningless**. Always clear the diagnostic gates below before quoting a
result. It takes ten seconds and saves you from confidently reporting noise.
:::

## The number and the interval

```python
from mesh import summarize_range
print(summarize_range(idata))
```

```text
  parameter   mean  median    sd  hdi_low  hdi_high  hdi_prob  r_hat  ess_bulk
0     range  209.8   206.1  40.2    134.4     289.2      0.95  1.004     223.0
```

- **`mean` / `median`** — the best single guess of the patch size, in microns.
- **`hdi_low` / `hdi_high`** — the 95% **credible interval**: MESH believes
  there is a 95% chance the true patch size lies in here. This, not the point
  estimate, is the result you report: *"≈ 210 µm (95% CrI 134–289 µm)."*
- **`sd`** — how spread out the estimate is.
- **`r_hat`, `ess_bulk`** — the trust gauges, explained next.

:::{admonition} Why the interval *is* the answer
:class: note

A point estimate alone invites over-interpretation. Two features at 210 µm and
250 µm look different until you notice their intervals overlap heavily — at which
point claiming they "differ in scale" is unsupported. MESH reports uncertainty so
you don't overclaim. **Compare intervals, not point estimates.**
:::

## The three diagnostic gates

Treat these as a checklist. If any fails, fix the fit (below) — don't report the
number.

```{list-table}
:header-rows: 1
:widths: 22 30 48

* - Gate
  - Pass looks like
  - What a fail means
* - **Divergences**
  - 0 (a handful is tolerable)
  - The sampler hit terrain it couldn't explore; the geometry is suspect.
    Raise `target_accept_prob` toward 0.95.
* - **$\hat R$ (`r_hat`)**
  - ≈ 1.00 (and < 1.01)
  - Your independent chains **disagree** — they haven't settled on one answer.
    Needs ≥ 2 chains to even compute. Sample longer / more warmup.
* - **ESS (`ess_bulk`)**
  - comfortably in the hundreds
  - The draws are too autocorrelated to pin the interval precisely; the interval
    is noisy. Increase `num_samples`.
```

Read them straight off `summarize_range`, or with ArviZ for the full picture:

```python
import arviz as az
print("divergences:", int(idata.sample_stats["diverging"].sum()))
az.summary(idata, var_names=["range", "eta"])
az.plot_trace(idata, var_names=["range"])   # chains should overlap like fuzzy caterpillars
```

If a gate fails, the fixes in order are: **(1)** run ≥ 2 chains and more
warmup/samples; **(2)** raise `target_accept_prob`; **(3)** set a `range_prior`
that matches your physical scale ([choosing a model](choose-your-model.md#what-you-dont-have-to-set)).
Full reasoning in [inference · diagnostics](../methods/inference.md#diagnostics-to-check).

## The four plots — and what to look for

`mesh.plots` draws the four views that make a fit interpretable. Each function
takes an optional `ax` and returns it, so they compose into one figure
(see {doc}`../api/plots`).

```python
import matplotlib.pyplot as plt
from mesh import (
    plot_samples, plot_range_posterior, plot_field,
    plot_matern_correlation, posterior_field_mean,
)

fig, axes = plt.subplots(2, 2, figsize=(11, 9))
plot_samples(sim.table, value="count", ax=axes[0, 0])          # the input
plot_range_posterior(idata, truth=200.0, ax=axes[0, 1])         # the headline
plot_field(sim.coords, posterior_field_mean(idata), ax=axes[1, 0])  # the inferred surface
plot_matern_correlation(idata, ax=axes[1, 1])                   # what the range means
fig.tight_layout()
```

```{list-table}
:header-rows: 1
:widths: 30 70

* - Plot
  - What it tells you
* - **`plot_samples`** — input map
  - Your raw signal over space. *Before* trusting any estimate, look here: do you
    *see* patches by eye? MESH should agree with what your eyes find.
* - **`plot_range_posterior`** — the headline
  - The full posterior over patch size, with the 95% interval shaded. A narrow,
    single peak = a well-identified scale. Pass `truth=` on simulated data to
    check the true value lands under the peak.
* - **`plot_field`** — inferred surface
  - MESH's reconstruction of the hidden enrichment surface. Compare it to
    `plot_samples`: it should look like a smoothed version of the input. On
    simulations, compare to the true field (`sim.field`) side by side.
* - **`plot_matern_correlation`** — meaning of the range
  - Translates the number into a curve: how similarity falls off with distance.
    The shaded band carries the range uncertainty. This is the picture to show
    someone who asks "what does 210 µm actually mean?"
```

A ready-made 2×2 panel script is in the repo:

```bash
python examples/plot_negbinomial.py   # writes examples/negbinomial_panel.png
```

## Beyond patch size: three more questions this fit already answers

Patch size is the headline, but it is only one axis of spatial architecture. The
*same* posterior carries three further signals — no re-fit needed. If your
interest is *why* a community is arranged the way it is, these are where the
biology lives.

```{list-table}
:header-rows: 1
:widths: 24 30 46

* - Question
  - Quantity
  - How to read it
* - **How *strong* is the structure?**
  - field amplitude `eta`
  - Grain and intensity are independent: a feature can sit at 200 µm with sharp
    contrast between patches, or at 200 µm so faint it is effectively well-mixed.
    `eta` is the field's SD on the latent scale; near zero = "no real structure
    at this scale, whatever the range says." See `plot_amplitude_posterior`.
* - **How *deterministic* is it?**
  - spatial fraction of variance
  - The share of latent variance the spatial field explains, versus the model's
    unstructured overdispersion (the "nugget"). Near 1 = the arrangement is
    strongly spatially determined; near 0 = mostly noise below your sampling
    resolution. A direct handle on deterministic vs. stochastic assembly. See
    `variance_partition` / `plot_variance_partition`.
* - **Does *function* share a scale with *genotype*?**
  - two fits, compared
  - Run the abundance model (a function's patch) and the allele model (a strain's
    territory) on the same region and overlay their ranges. A function patch
    *larger* than any single allele territory means the function is shared across
    strains; coinciding scales mean it is strain-private. See
    `plot_scale_comparison`.
```

```python
from mesh import (
    summarize_parameters, variance_partition,
    plot_amplitude_posterior, plot_variance_partition, plot_scale_comparison,
)

# 1 · how strong — the amplitude posterior (eta is included by default here)
print(summarize_parameters(idata))            # rows for range AND eta
plot_amplitude_posterior(idata)

# 2 · how deterministic — the spatial share of variance
print(variance_partition(idata))              # spatial_fraction is the headline row
plot_variance_partition(idata)

# 3 · function vs genotype — overlay an abundance fit and an allele fit
plot_scale_comparison(
    [idata_function, idata_genotype],
    labels=["function (abundance)", "genotype (allele)"],
)
```

:::{admonition} What the variance split is, and isn't
:class: note

The spatial fraction mixes a latent-field variance with an observation-level
overdispersion (computed exactly from the gamma/beta mixture), so read it as a
well-defined **index of spatial determinism**, not a partition of one observable
quantity. A clean variance partition *across several spatial scales* is a later
milestone — see the [roadmap](../roadmap.md).
:::

A ready-made panel for these three views is in the repo:

```bash
python examples/plot_architecture.py   # writes examples/architecture_panel.png
```

## Fitting several features jointly — reading the loadings

The three questions above come from a **single-feature** fit. When you fit
several features **together** with the coregionalization model
({func}`mesh.coregionalized_negbinomial`), the output gains two things: a
**range per shared field** (the co-existing scales) and a **loadings matrix**
that says *which feature sits on which field*. This is the direct read-out of
**co-segregation** — features on the same field share a territory.

```python
from mesh import (
    coregion_counts_arrays, coregion_feature_order,
    coregionalized_negbinomial, fit_model,
    summarize_loadings, plot_loadings,
)

arrays = coregion_counts_arrays(df)          # multi-feature table -> (J, n) matrices
idata = fit_model(
    coregionalized_negbinomial,
    num_warmup=800, num_samples=800, num_chains=2,
    target_accept_prob=0.95, n_fields=2, **arrays,
)

feature_ids = coregion_feature_order(df)     # loadings rows follow this order
print(summarize_loadings(idata, feature_ids=feature_ids))
plot_loadings(idata, feature_ids=feature_ids)   # heatmap, assigned field outlined
```

Reading it:

- **`abs_mean`** — the magnitude of a feature's loading on each field. The largest
  per row is the feature's **`assigned_field`**: the scale it segregates at.
  Features sharing a field share a territory.
- **`mean` (signed)** — within a single field, the *relative* sign of two features
  distinguishes **co-segregation** (same sign — they crest and trough together)
  from **anti-segregation** (opposite sign — one fills where the other empties).
- The **per-field `range`** posteriors are the co-existing scales; read each one
  exactly like the single-field range above.

:::{admonition} Why magnitudes, not signed loadings, for assignment
:class: note

A coregionalization fit does not identify the **sign** of a field (flipping a
field and its loading column together changes nothing observable), so a signed
loading can average toward zero across chains. Assignment therefore uses the
sign-invariant magnitude `abs_mean`; the *relative* sign **within one draw** is
still meaningful, which is what the co- vs. anti-segregation reading relies on.
See {func}`mesh.summarize_loadings`.
:::

## Two sanity checks specific to spatial scale

Even a clean fit can be quietly mis-scaled. Two quick checks
([spatial scales](../biology/spatial-scales.md)):

1. **Range vs. sample spacing.** A patch size that isn't comfortably *larger*
   than the distance between microsamples is at the edge of what your design can
   resolve. Treat it cautiously — you may simply lack the resolution to see it.
2. **Range vs. domain size.** A patch size approaching the size of the whole
   sampled area means "structured at the scale of the entire sample." The data
   can't tell that apart from an even longer range, so expect a wide,
   right-skewed interval — and don't over-read the upper edge.

## Reporting, in one sentence

> Feature X segregates at a patch size of **≈ 210 µm (95% CrI 134–289 µm)**;
> $\hat R = 1.00$, no divergences, ESS > 200.

That single line carries the estimate, the uncertainty, *and* the evidence that
the fit is trustworthy — everything a reader needs.

## Where to go next

- A second feature? Re-run from [step 3](run-your-first-fit.md); compare the two
  **intervals**.
- Want the biology framing of these numbers? See the worked
  [case studies](../cases/index.md).
- Need a parameter or function detail? The [API reference](../api/index.md) is
  generated from the source.
