# The model

MESH represents the spatial structure of a feature as a **Gaussian-process latent
field** and connects that field to observed reads through a **coverage-aware
likelihood**. This page states the generative model for both observation types
and explains each prior. The implementation is in {mod}`mesh.model` and
{mod}`mesh.kernels`.

## The shared latent field

Let $\mathbf{s}_i \in \mathbb{R}^2$ be the coordinates (microns) of microsample
$i$, for $i = 1, \dots, n$. A single latent field $f$ is a zero-mean Gaussian
process with a Matérn 3/2 covariance:

$$
f \sim \mathcal{GP}\!\big(0,\; \eta^2\, \rho_\ell(\cdot, \cdot)\big),
\qquad
\rho_\ell(r) = \left(1 + \frac{\sqrt{3}\,r}{\ell}\right)
               \exp\!\left(-\frac{\sqrt{3}\,r}{\ell}\right),
$$

where $r = \lVert \mathbf{s}_i - \mathbf{s}_j \rVert$ is the Euclidean distance,
$\ell$ is the **range** (the patch size, in microns) and $\eta$ is the field
standard deviation. The covariance is computed by {func}`mesh.matern32_kernel`.

### Why Matérn 3/2

The Matérn family interpolates between very rough and very smooth processes via a
smoothness parameter $\nu$. MESH fixes $\nu = 3/2$: the resulting fields are
continuous and once-differentiable — smooth enough to represent biological
gradients, but **not** infinitely smooth like the squared-exponential kernel,
whose unrealistic smoothness can bias range estimates and stiffen the posterior
geometry.

### Non-centered parameterization

Sampling $f$ directly couples it tightly to $\ell$ and $\eta$ and produces a
posterior that NUTS struggles with. MESH instead uses the **non-centered** form

$$
f = \eta \, (L\,\mathbf{z}), \qquad
\mathbf{z} \sim \mathcal{N}(\mathbf{0}, I_n), \qquad
L L^\top = \rho_\ell,
$$

where $L$ is the Cholesky factor of the **unit-variance** correlation matrix
(plus a small diagonal jitter for numerical stability). The standard-normal
innovations $\mathbf{z}$ are sampled, and the field is a deterministic transform.
This is implemented once in {func}`mesh.gp_field` and reused by both observation
models.

## Observation model 1 — abundance counts (negative-binomial)

For feature counts $y_i$ with per-sample sequencing depth $d_i$ and feature
length $L_{\text{feat}}$, the expected count follows the standard library-size /
length offset:

$$
\log \mu_i = \beta_0 + \underbrace{\log d_i + \log L_{\text{feat}}}_{\text{offset}}
             + f_i,
\qquad
y_i \sim \text{NegBinomial2}(\mu_i, \phi).
$$

- $\beta_0$ is an intercept (a log baseline rate per read·bp).
- The **offset** is fixed, not estimated — it encodes that more sequencing or a
  longer gene yields more reads at the same underlying abundance.
- $\phi$ is the negative-binomial concentration; the variance is
  $\mu_i + \mu_i^2/\phi$, so smaller $\phi$ means more overdispersion. As
  $\phi \to \infty$ the model approaches Poisson.

This is implemented by {func}`mesh.spatial_negbinomial` and is the **minimal
counts-table entry point**.

## Observation model 2 — allele frequencies (beta-binomial)

For a variant site with alternate-allele count $a_i$ out of coverage $c_i$, the
field structures the logit allele frequency:

$$
\operatorname{logit} p_i = \beta_0 + f_i,
\qquad
a_i \sim \text{BetaBinomial}(p_i\,\kappa,\; (1 - p_i)\,\kappa,\; c_i).
$$

- $p_i$ is the latent allele frequency at location $i$.
- $\kappa$ is a precision: larger $\kappa$ approaches a plain binomial, smaller
  $\kappa$ adds overdispersion.
- Because the likelihood is conditioned on coverage $c_i$, **low-coverage sites
  contribute proportionally less information** — this is the coverage-aware
  behaviour.

This is implemented by {func}`mesh.spatial_betabinomial` (the `m0` seed model).

## Direction — anisotropic fields

The models above use an **isotropic** field: one `range` in every direction, so
patches are round. The **anisotropic** model gives the field a **separate patch
size along each axis**, so it can read out *direction* — whether a feature
organises along a host axis (the proximal–distal gut, the crypt–villus axis, or
depth into a biofilm). The field is **axis-aligned**: elongation is assumed to
lie along `x`/`y`, so orient the sampling frame to the host axis (a free rotation
is a [later milestone](../roadmap.md)).

The covariance replaces the single isotropic distance with a **per-axis-scaled**
lag (implemented by {func}`mesh.anisotropic_matern_kernel`):

$$
\rho_\nu(u_{ij}), \qquad
u_{ij} = \sqrt{\left(\frac{x_i - x_j}{\ell_x}\right)^2
             + \left(\frac{y_i - y_j}{\ell_y}\right)^2},
$$

so the patch size is $\ell_x$ along `x` and $\ell_y$ along `y`. With
$\ell_x = \ell_y$ this is exactly the isotropic {func}`mesh.matern_kernel`.

### Overall size × signed anisotropy

Rather than sampling $\ell_x$ and $\ell_y$ directly, MESH parameterizes them by
an **overall** patch size and a **signed anisotropy**, which are orthogonal and
each identified from the data:

$$
\ell_x = \ell\, e^{+\rho/2}, \qquad \ell_y = \ell\, e^{-\rho/2},
$$

so $\ell = \sqrt{\ell_x \ell_y}$ is the geometric-mean `range` — the *same*
quantity the isotropic model reports — and $e^{\rho} = \ell_x/\ell_y$ is the
anisotropy. The prior on $\rho = \log(\ell_x/\ell_y)$ is
$\mathcal{N}(0, \sigma_\rho)$, **centred at isotropy** ($\rho = 0$), so a
directional reading has to be supported by the data rather than assumed; the two
coordinate axes are physically distinct, so no ordering is needed to identify
which is which.

This is implemented by {func}`mesh.anisotropic_negbinomial` (with the shared
non-centred field {func}`mesh.gp_field_anisotropic`). Read the result with
{func}`mesh.summarize_anisotropy` — the per-axis patch sizes, the folded
**anisotropy ratio** $\max(\ell_x,\ell_y)/\min(\ell_x,\ell_y)$ (how directional)
and `prob_x_longer` (which axis) — or {func}`mesh.plot_anisotropy`. As with the
other layers it ships with a known-truth generator
({func}`mesh.simulate_anisotropic`) and a recovery test.

## Multiple features on shared fields — coregionalization

The two models above fit **one feature** and recover **one** patch size. The
**coregionalization** model fits **several features at once** over **multiple**
shared latent fields, so the inference can *separate co-existing spatial scales*
and read out *which features share a territory*. This is the
[M1+ milestone](../roadmap.md) and the entry point for co-segregation questions
(worked through in [the co-segregation case study](../cases/co-segregation.md)).

It is a **linear model of coregionalization**. There are $K$ unit-variance
Matérn fields $f_1, \dots, f_K$, each with its own range $\ell_k$. A
feature-by-field **loadings** matrix $W \in \mathbb{R}^{J \times K}$ mixes them
into the $J$ features; for feature $j$ at location $i$ the abundance model
becomes

$$
\log \mu_{ji} = \beta_j + \underbrace{\log d_i + \log L_{\text{feat}}}_{\text{offset}}
                + \sum_{k=1}^{K} W_{jk}\, f_k(\mathbf{s}_i),
\qquad
y_{ji} \sim \text{NegBinomial2}(\mu_{ji}, \phi).
$$

The fields are unit-variance, so the **loadings carry the amplitude**: each
feature's spatial signal is split across scales by its row of $W$. The structure
of $W$ reads out ecology directly — features that load on the *same* field occupy
the *same* patches (a shared niche, cross-feeding, syntrophy, or co-residence on
one mobile element); features that load with *opposite* sign anti-segregate
(competition, niche partitioning).

### Identifiability — ordered ranges, free sign

Two symmetries would otherwise make the posterior unidentifiable, and MESH
handles each:

- **Which field is which.** Swapping two fields (and the matching columns of $W$)
  leaves the likelihood unchanged. MESH samples the ranges **ordered**,
  $\ell_1 < \ell_2 < \dots < \ell_K$, which pins each field to a *specific* scale
  so a feature can be assigned to one.
- **The sign of a field.** Flipping $f_k$ and column $k$ of $W$ together is also
  invariant. This sign is left **free** — it does not affect the ranges or
  *which* features share a field. Read assignment from the loading **magnitudes**
  $|W|$ ({func}`mesh.summarize_loadings`, {func}`mesh.plot_loadings`); the
  *relative* sign within a single posterior draw still carries the co- vs.
  anti-segregation distinction.

A third issue is not a symmetry but a **runaway mode**: a range drifting past the
sampling extent makes the Matérn correlation matrix all-ones, so the field
flattens to a constant the per-feature intercepts absorb, and its range floats
off to the prior edge — collapsing the feature assignment. Each range is
therefore softly **bounded at the spatial extent** of the samples (a patch larger
than your sampling area is unidentifiable anyway), which removes that basin and
makes recovery seed-robust without distorting the prior below the extent.

This is implemented by {func}`mesh.coregionalized_negbinomial`. Build its array
inputs from a **multi-feature** table with {func}`mesh.coregion_counts_arrays`,
and recover the feature row order with {func}`mesh.coregion_feature_order`. As
with the single-field models, it ships with a known-truth generator
({func}`mesh.simulate_coregionalized`) and a recovery test that confirms it
resolves both ranges and assigns every feature to the right field (see
[simulation & recovery](simulation.md)).

## Composing the axes

The models above read as four things, but under the hood they are **one field
core plus a likelihood**. The latent spatial contribution — the part that
differs between them — is built once, from three orthogonal knobs:

| knob | meaning | values |
|---|---|---|
| `n_fields` | co-existing scales | `1` (single field) … `K` (coregionalization) |
| `anisotropic` | direction | isotropic · per-axis ranges |
| `nu` | boundary sharpness | fixed per fit, LOO-selected |

crossed with one or many features (single vs. multi-feature `counts`). Because
these are independent, {func}`mesh.spatial_negbinomial` exposes all of them and
they **combine freely** — including crossings no single earlier model could
express, such as a *directional multi-scale* fit
(`spatial_negbinomial(..., anisotropic=True, n_fields=2)`) or smoothness
selection over an *anisotropic* fit
(`compare_smoothness(spatial_negbinomial, anisotropic=True, ...)`).

{func}`mesh.anisotropic_negbinomial` and {func}`mesh.coregionalized_negbinomial`
remain as named presets (thin wrappers with the same parameters), so the earlier
examples keep working unchanged. One deliberate seam: the single-field regime
keeps a **positive amplitude** `eta` (its sign is fixable, so the posterior stays
unimodal), while the multi-field regime uses **signed loadings** whose magnitude
carries assignment — the identifiability structure each regime needs, not an
accident of history.

## Priors

The priors are **weakly informative** and documented in the function docstrings.
The most important is the **range** prior, which sits on the micron scale.

```{list-table}
:header-rows: 1
:widths: 22 28 50

* - Parameter
  - Default prior
  - Notes
* - range $\ell$
  - $\text{LogNormal}(\log 150,\, 1.0)$
  - In log-microns; broad (≈ 20–1100 µm central 95%). **Set it to your sampling
    domain.** Positive by construction.
* - field SD $\eta$
  - $\text{HalfNormal}(1)$
  - Scale of spatial variation on the link scale.
* - intercept $\beta_0$ (NB)
  - $\mathcal{N}(0, 25)$
  - Broad: the RPKM-style offset pushes the intercept to large-magnitude
    (negative) log-rates.
* - intercept $\beta_0$ (BB)
  - $\mathcal{N}(0, 3)$
  - On the logit scale.
* - concentration $\phi$ (NB)
  - $\text{Gamma}(2, 0.1)$
  - Overdispersion of counts.
* - precision $\kappa$ (BB)
  - $\text{HalfNormal}(100)$
  - Overdispersion of allele frequencies.
* - log anisotropy $\rho$ (aniso)
  - $\mathcal{N}(0, 1)$
  - $\rho = \log(\ell_x/\ell_y)$; centred at isotropy so direction is
    data-driven. One prior SD is an axis ratio of $e \approx 2.7$.
* - loadings $W$ (coregion)
  - $\mathcal{N}(0, 1)$
  - Feature-by-field loadings; carry the per-field amplitude (the shared fields
    are unit-variance). The $K$ ranges are sampled **ordered**.
```

:::{admonition} Choosing the range prior
:class: tip

The default range prior is centred below 200 µm so it does not sit on any
particular truth, but it is broad enough to be **data-dominated**. For a real
study, pass a `range_prior=(loc, scale)` (LogNormal parameters in log-microns)
that brackets the plausible patch sizes for your system and resolution.
:::

## From model to patch size

Given data arrays, {func}`mesh.fit_model` runs NUTS and returns an ArviZ
`InferenceData`; {func}`mesh.summarize_range` reduces the `range` posterior to a
point estimate and a credible interval. How that inference is configured and
diagnosed is covered in [inference](inference.md).
