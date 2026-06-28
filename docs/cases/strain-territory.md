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

## The answer

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

## How this case grows with MESH

Today this resolves **one** variant site at **one** scale. The same variant
dataset will support more as MESH expands:

- [ ] **Linked variants** ([coregionalization](../roadmap.md)) — fit many sites
      jointly and find variants that share a territory, i.e. that travel together
      as a **strain/haplotype**.
- [ ] **Residence covariate** — annotate each variant by the contig / mobile
      element it sits on and ask whether territory size is set by the genomic
      context (a future *linkage* covariate, not a modelling unit).
- [ ] **Function meets genotype** — analyse the abundance of a function
      ([Case 1](functional-patch.md)) and the genotype carrying it on shared
      fields, asking whether they segregate at the same scale.
- [ ] **Across hosts** — compare strain-territory sizes between individuals.
- [ ] **3D** — territories as volumes.

As in [Case 1](functional-patch.md), each new capability ships with a recovery
test, and this page will gain the matching analysis when it lands.
