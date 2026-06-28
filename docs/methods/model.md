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
