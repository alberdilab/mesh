# Spatial scales and patch size

## The Matérn range *is* the patch size

MESH models each latent biological field as a Gaussian process with a **Matérn
covariance**. The covariance between two locations depends only on the distance
between them, and it is controlled by a single interpretable parameter — the
**range** (also called the lengthscale), denoted $\ell$. The range is reported in
the same physical units as the coordinates: **microns**.

Intuitively:

- At distances **much smaller than $\ell$**, two locations are strongly
  correlated — they belong to the same patch.
- At distances **comparable to $\ell$**, correlation has substantially decayed —
  you are crossing into a neighbouring patch.
- At distances **much larger than $\ell$**, locations are effectively
  independent.

So $\ell$ is a direct, physical readout of **how big a patch is** for a given
feature. That is the headline output of MESH, and it comes with a credible
interval rather than a point estimate.

For the Matérn 3/2 kernel used in the M0/M1 models, the correlation as a function
of distance $r$ is

$$
\rho(r) = \left(1 + \frac{\sqrt{3}\,r}{\ell}\right)
          \exp\!\left(-\frac{\sqrt{3}\,r}{\ell}\right).
$$

This is smooth enough to be realistic for biological gradients but rougher (less
unrealistically smooth) than a squared-exponential kernel — a deliberate choice
discussed in [the model](../methods/model.md).

```{mermaid}
flowchart LR
    subgraph small["r ≪ ℓ"]
      direction TB
      s1((•)) --- s2((•))
    end
    subgraph mid["r ≈ ℓ"]
      direction TB
      m1((•)) -. weak .- m2((•))
    end
    subgraph big["r ≫ ℓ"]
      direction TB
      b1((•)) b2((•))
    end
    small -->|"strongly correlated<br/>(same patch)"| mid -->|"decorrelating<br/>(patch edge)"| big
```

## Reading a patch-size estimate

A MESH result for one feature looks like a posterior over $\ell$:

> patch size ≈ **210 µm** (95% credible interval **134–289 µm**)

Some practical guidance for interpretation:

- **Compare to your sampling resolution.** A range that is not comfortably larger
  than the spacing between microsamples is at the edge of what the design can
  resolve; treat such estimates cautiously.
- **Compare to your domain extent.** A range approaching the size of the sampled
  field means "structured at the scale of the whole sample" — the data cannot
  distinguish that from an even longer range, so expect a wide, right-skewed
  credible interval.
- **Width matters.** The credible interval is the result. Two features with
  overlapping intervals should not be claimed to differ in scale.

## Co-segregation: features that share a scale

Because features load onto shared latent fields, MESH can identify groups of
features whose abundance (or allele frequency) varies *together* across space and
at the *same* scale. Biologically, co-segregating features are candidates for a
common structuring process — co-localised members of a guild, genes on the same
mobile element, or functions responding to the same micro-gradient.

:::{admonition} Scope note
:class: note

Co-segregation across **multiple features on shared fields** —
*coregionalization* — is implemented: {func}`mesh.coregionalized_negbinomial`
fits several features over multiple ordered Matérn fields, separating co-existing
scales and reporting which feature loads on which field (worked through in the
[co-segregation case study](../cases/co-segregation.md)). The single-field models
remain the entry point for one feature at a time. A clean variance *partition* of
one feature across several scales is still a later milestone — see the
[roadmap](../roadmap.md).
:::

## From genes to genotypes

The same machinery applies below the species level. Where a feature is an
**allele** at a variant site, the field structures the **allele frequency**, and
the patch size describes how far a genetic variant stays locally dominant before
giving way to another. This sub-species, **coverage-aware** view is the
distinctive reach of MESH: low-coverage sites are automatically down-weighted
rather than treated as confident zeros or ones (see
[sampling design](sampling-design.md)).
