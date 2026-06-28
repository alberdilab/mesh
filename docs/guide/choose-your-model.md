# Step 2 · Choose your model (and understand why)

MESH has two models. They share an identical spatial core and differ only in the
**last step** — how the number at each location was generated. This page explains
that last step in plain language, because choosing it correctly (and knowing
*why*) is what separates a trustworthy patch size from a misleading one.

Every model here is built from two parts:

1. **A spatial field** — the same for both models. It captures "nearby places are
   similar; far-apart places drift" and hides the patch size inside it.
2. **An observation model** — how a hidden field value becomes the integer you
   actually counted. This is where abundances and alleles part ways.

We take the shared part first, then the two endings.

## The shared part: the spatial field

Picture an invisible surface laid over your sampling area. Where the surface is
high, the feature is enriched; where it is low, the feature is depleted. The
surface is smooth: it can't jump from high to low between two touching points.
*How quickly* it is allowed to change is exactly the **patch size**. MESH's whole
job is to infer that surface and, with it, the patch size.

That surface is a **Gaussian process** with a **Matérn covariance**. You don't
need the math, but two design choices are worth understanding because they affect
your results.

### Why Matérn (and not the "smoother" alternative)

A covariance function encodes a belief about *how wiggly* the surface is. The
common textbook choice (the squared-exponential / RBF kernel) assumes the surface
is **infinitely smooth** — gradients that never have a sharp turn. Biology is not
that polite: a mucus layer ends, an oxygen gradient steepens, a colony has an
edge. MESH uses the **Matérn 3/2** kernel, which produces surfaces that are
continuous and gently curved but **allowed to have realistic edges**.

:::{admonition} Why this is not just an aesthetic choice
:class: note

An over-smooth kernel systematically *over-estimates* how far correlation
reaches, because it refuses to believe in sharp transitions — so it would inflate
your patch size. Matérn 3/2 is the deliberate "realistic, not unrealistically
smooth" middle ground. More detail in [the model](../methods/model.md).
:::

### Why the "non-centered" trick (you'll see it in diagnostics)

The field is written as `field = eta × (L z)` — standard-normal numbers `z` run
through a transform `L`, scaled by an amplitude `eta`. This is a numerical
reformulation, not a different model: it makes the sampler's job dramatically
easier and is the first thing to check if you ever see *divergences*
([inference](../methods/inference.md)). You won't touch it directly; just know
that "non-centered" in a warning is normal MESH machinery.

## Ending A — abundance counts → negative binomial

**Use this when** your number is *reads assigned to a gene/contig*: how **much**
of a function is present. Model: {func}`mesh.spatial_negbinomial`.

### Why a count model at all (not just a percentage)

You might be tempted to convert counts to relative abundances and model those.
Don't. A count of `3` and a count of `300` carry different *confidence*, and that
information lives in the raw count together with the sequencing depth. Modelling
the integer counts directly lets MESH weigh each observation by how much evidence
it really has.

### Why the depth-and-length offset

The same gene, equally abundant, produces **more reads** in a deeply sequenced
sample, and a **longer gene** produces more reads than a short one. Neither has
anything to do with spatial patchiness — they are bookkeeping. So MESH adds a
**fixed offset**, `log(depth) + log(length)`, to the expected count. It is *not
estimated*; it is known bookkeeping that puts every observation on a common
footing (this is the same idea as RPKM/TPM normalisation, done inside the model
instead of beforehand).

$$
\log(\text{expected count}) = \text{baseline} + \underbrace{\log(\text{depth}) + \log(\text{length})}_{\text{fixed offset}} + \text{field}_i
$$

Read it left to right: a baseline rate, plus the bookkeeping correction, plus the
spatial signal we actually care about.

### Why *negative* binomial and not Poisson

The textbook model for counts is the **Poisson**. Poisson makes one strong
promise: the variance equals the mean. Metagenomic counts break that promise
badly — replicate samples vary far more than Poisson allows (biological
patchiness, PCR, uneven extraction). This extra scatter is called
**overdispersion**.

The **negative binomial** is the Poisson's relaxed cousin: same idea, but with a
spare knob, the concentration $\phi$, that lets variance exceed the mean:

$$
\text{variance} = \mu + \frac{\mu^2}{\phi}
$$

- Large $\phi$ → the extra term vanishes → it behaves like a Poisson.
- Small $\phi$ → lots of extra scatter → heavy overdispersion.

```{list-table}
:header-rows: 1
:widths: 30 35 35

* - 
  - Poisson
  - Negative binomial (what MESH uses)
* - Variance
  - exactly the mean
  - mean **plus** an overdispersion term
* - Handles noisy metagenomic counts?
  - no — too rigid
  - **yes**
* - Risk if you pick wrong
  - over-confident intervals; false patches
  - —
```

:::{admonition} Why this protects your answer
:class: important

If you forced a Poisson onto overdispersed counts, the model would mistake
ordinary noise for real spatial structure and hand you a **falsely narrow**
credible interval — over-confident, sometimes inventing a patch that isn't there.
The negative binomial absorbs that noise so the patch size reflects genuine
spatial signal. This is what *coverage-aware* means for abundances.
:::

## Ending B — allele frequencies → beta-binomial

**Use this when** your number is *alt-allele reads out of total coverage* at a
variant site: which **version** of a gene dominates. Model:
{func}`mesh.spatial_betabinomial`.

### Why coverage is the whole story here

An allele frequency of "½" means something completely different when it comes
from **2 reads** versus **400 reads**. The first is a coin-flip's worth of
evidence; the second is solid. If you collapsed both to the number `0.5` and
modelled that, MESH would treat a guess and a near-certainty as equal — and a
single 1-out-of-1 site would scream "100% this variant!" with no humility.

So MESH never models the frequency directly. It models the **alt count out of the
coverage**, which keeps the evidence attached to the estimate. Low-coverage sites
automatically count for little; high-coverage sites anchor the field. *That* is
coverage-aware inference.

### Why *beta*-binomial and not plain binomial

The natural model for "alt reads out of total" is the **binomial** (like counting
heads in coin tosses). But real sequencing has extra noise beyond clean coin
tosses — mapping bias, local strain mixtures, amplification quirks — so allele
counts are again **overdispersed**.

The **beta-binomial** is the binomial with the same spare knob idea: the success
probability is itself allowed to wobble (via a Beta distribution) before the
counts are drawn. A precision parameter $\kappa$ tunes it:

- Large $\kappa$ → almost no wobble → behaves like a plain binomial.
- Small $\kappa$ → more wobble → heavier overdispersion.

The parallel with the count case is exact, and deliberate:

```{list-table}
:header-rows: 1
:widths: 34 33 33

* - Your data
  - Clean textbook model
  - MESH uses (overdispersed version)
* - Abundance counts (how much)
  - Poisson
  - **Negative binomial** (knob: $\phi$)
* - Allele counts (which version)
  - Binomial
  - **Beta-binomial** (knob: $\kappa$)
```

In both endings the move is the same: take the clean model, add one parameter
that lets real data be noisier than the clean model assumes, and condition on how
much evidence each observation carries.

## What you *don't* have to set

The priors — the model's mild starting assumptions about each parameter — have
sensible defaults, so a first fit needs none of them. The **one** worth revisiting
for a real study is the **range prior**, because it lives on your physical scale.
The default is broad and centred below 200 µm so it doesn't bias the answer, but
you can bracket your system's plausible patch sizes:

```python
from mesh import spatial_negbinomial, fit_model
# range_prior is (loc, scale) of a LogNormal in *log-microns*
idata = fit_model(spatial_negbinomial, range_prior=(5.3, 0.7), **arrays)  # ~200 µm-centred
```

The full prior table and the reasoning behind every default is in
[the model · priors](../methods/model.md#priors). When in doubt, leave them and
let the data dominate.

## Decide and move on

- Counts of *how much* a feature is present → **negative binomial**
  ({func}`mesh.spatial_negbinomial`).
- Counts of *which variant* is present → **beta-binomial**
  ({func}`mesh.spatial_betabinomial`).
- Both? Run MESH once per data type.

Next: [run your first fit](run-your-first-fit.md), with every sampler setting
explained.
