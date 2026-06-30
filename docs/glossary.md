# Glossary

```{glossary}
patch size
  The physical distance over which a feature stays spatially correlated before
  decorrelating into a neighbouring patch. In MESH it is the Matérn **range**,
  reported in microns, and it is the headline output.

range
lengthscale
  The parameter $\ell$ of the Matérn covariance controlling how fast correlation
  decays with distance. Synonymous with the {term}`patch size`.

Matérn covariance
  A family of stationary covariance functions parameterised by a smoothness $\nu$
  and a range $\ell$. MESH uses $\nu = 3/2$: continuous, once-differentiable
  fields that are realistic without being unrealistically smooth.

Gaussian process (GP)
  A distribution over functions such that any finite set of locations has a
  multivariate normal distribution. MESH uses **exact** GPs (dense covariance +
  Cholesky), not approximations.

non-centered parameterization
  Writing the field as $f = \eta\,(L z)$ with $z \sim \mathcal{N}(0, I)$ and $L$
  the Cholesky factor of the correlation matrix, rather than sampling $f$
  directly. Improves the geometry NUTS must explore.

feature
  The modelling unit: a gene/contig (as a count or allele substrate) or a variant
  site. **Not** a species, and **not** a contig-as-entity.

shared catalog
  The requirement that every sample report the same set of `feature_id`s.
  Per-sample catalogs are rejected by validation.

coverage-aware
  Likelihoods that condition on the amount of evidence per observation — site
  coverage (beta-binomial) or sequencing depth and feature length (negative
  binomial) — so low-evidence observations are down-weighted rather than trusted.

depth
  Per-sample sequencing depth (abundance model) used as a fixed offset, or
  per-site coverage (allele model) used as the number of trials.

offset
  The fixed term $\log(\text{depth}) + \log(\text{length})$ added to the log mean
  of the negative-binomial model so that expected counts scale with library size
  and feature length.

negative binomial (NB2)
  The count likelihood for abundances, with mean $\mu$ and concentration $\phi$;
  variance $\mu + \mu^2/\phi$. Captures overdispersion beyond Poisson.

beta-binomial
  The allele-count likelihood, a binomial whose probability is itself
  beta-distributed; adds overdispersion and is conditioned on coverage.

coregionalization
  Letting multiple features load onto shared latent fields so they can
  co-segregate. Implemented by {func}`mesh.coregionalized_negbinomial`, which
  fits several features over multiple ordered Matérn fields and reports which
  feature loads on which field (see the {doc}`co-segregation case study
  <cases/co-segregation>`).

InferenceData
  The [ArviZ](https://python.arviz.org/) container returned by
  {func}`mesh.fit_model`, holding posterior draws and sampler diagnostics.

R-hat
  The Gelman–Rubin convergence diagnostic; values near 1.0 (with ≥ 2 chains)
  indicate chains have mixed.
```
