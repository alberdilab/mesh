# Simulation & recovery

MESH is developed **simulation-first**. The output — a patch size — cannot be
checked by eye, so the primary correctness guarantee is that each model
**recovers its own parameters** from data generated with a known truth. Every
model ships with a matching generator and a recovery test; no model layer is
added without one.

## Generators of known truth

{mod}`mesh.simulate` provides two generators, each drawing one Matérn field with
a known range and returning a long-format table plus a `truth` dictionary:

```{list-table}
:header-rows: 1
:widths: 28 72

* - Function
  - Produces
* - {func}`mesh.simulate_allele`
  - Beta-binomial allele frequencies over a Matérn field (the `m0` variant),
    with per-site coverage and `ref`/`alt` columns.
* - {func}`mesh.simulate_counts`
  - Negative-binomial abundance counts over a Matérn field, with a per-sample
    depth offset and feature length.
```

Both return a {class}`mesh.SimulatedData` with `.table`, `.truth`, `.coords` and
`.field`, so the generated table is exactly what the [schema](schema.md) expects.

```python
from mesh import simulate_counts

sim = simulate_counts(n_samples=150, range_=200.0, eta=1.0, seed=0)
sim.truth["range"]      # 200.0  — the value the model must recover
sim.table.head()
```

## The recovery test pattern

A recovery test simulates with a known range, fits with a small-but-sufficient
number of draws, and asserts that:

1. the **true range lies inside the 95% credible interval**, and
2. the **posterior mean** is within a generous multiplicative tolerance, and
3. the sampler **converged** ($\hat R < 1.1$).

```python
def test_recovery_negbinomial():
    sim = simulate_counts(n_samples=150, range_=200.0, eta=1.0, seed=0)
    arrays = counts_arrays(sim.table)
    idata = fit_model(spatial_negbinomial, num_warmup=400, num_samples=400,
                      num_chains=2, seed=0, **arrays)
    row = summarize_range(idata).iloc[0]
    assert row["hdi_low"] <= sim.truth["range"] <= row["hdi_high"]
```

The repository's [`tests/test_recovery.py`](https://github.com/anttonalberdi/mesh/blob/main/tests/test_recovery.py)
applies this to **both** models.

## Recovered numbers (true range = 200 µm)

Representative output of the recovery runs:

```{list-table}
:header-rows: 1
:widths: 34 22 28 16

* - Model
  - Posterior mean
  - 95% credible interval
  - $\hat R$
* - Beta-binomial (allele)
  - 228.8 µm
  - 143.0 – 330.0 µm
  - 1.005
* - Negative-binomial (counts)
  - 209.8 µm
  - 134.4 – 289.2 µm
  - 1.004
```

Both intervals bracket the truth and both means are well within tolerance.

## Why this is the right guarantee

For models whose result is a number with no external ground truth, a passing
recovery test is the strongest routine evidence that the implementation is
correct: the priors, the likelihood, the parameterization, the offset handling
and the summary code must all be right for the truth to land inside the interval
across seeds. New model layers (e.g. coregionalization in the
[roadmap](../roadmap.md)) will each arrive with a generator and a test that they
recover *their* parameters — including, for multi-scale models, a test that the
**two scales are separated**.
```
