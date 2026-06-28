# Quickstart

This walkthrough simulates a counts table with a **known** patch size, validates
it against the input contract, fits the negative-binomial model, and reads back
the recovered patch size with a credible interval.

## Recover a patch size from abundance counts

```python
from mesh import (
    simulate_counts,
    validate_table,
    counts_arrays,
    fit_model,
    spatial_negbinomial,
    summarize_range,
)

# 1. Simulate a counts table over a Matérn field with a known range (200 µm).
sim = simulate_counts(n_samples=150, range_=200.0, eta=1.0, seed=0)

# 2. The long-format table is the documented interface. Validate before fitting.
validate_table(sim.table)

# 3. Turn the validated table into model arrays (coords, counts, log offset).
arrays = counts_arrays(sim.table)

# 4. Fit with NUTS and summarise the patch-size posterior.
idata = fit_model(
    spatial_negbinomial,
    num_warmup=500,
    num_samples=500,
    num_chains=2,
    seed=0,
    **arrays,
)
print(summarize_range(idata))
```

You should see a one-row table whose 95% credible interval brackets the true
range of 200 µm, for example:

```text
  parameter    mean  median      sd  hdi_low  hdi_high  hdi_prob   r_hat  ess_bulk
0     range  209.8   206.1   40.2    134.4     289.2      0.95   1.004     223.0
```

The same workflow with the allele-frequency model uses
{func}`mesh.simulate_allele`, {func}`mesh.allele_arrays` and
{func}`mesh.spatial_betabinomial`.

## What just happened

```{mermaid}
flowchart LR
    A[Long-format table<br/>feature_id, sample_id, x, y,<br/>count, depth, length] --> B[validate_table]
    B --> C[counts_arrays / allele_arrays]
    C --> D[spatial_negbinomial /<br/>spatial_betabinomial]
    D --> E[fit_model → NUTS → InferenceData]
    E --> F[summarize_range<br/>patch size + credible interval]
```

- The **table** is the only interface between upstream bioinformatics and MESH
  ([input schema](../methods/schema.md)).
- The **model** places a single Matérn Gaussian-process field over the sample
  coordinates; its range *is* the patch size ([the model](../methods/model.md)).
- **Inference** uses a non-centered parameterization and NUTS
  ([inference](../methods/inference.md)).

## Runnable examples

The repository ships two end-to-end scripts:

```bash
python examples/m0_spatial_field.py   # allele frequencies (beta-binomial)
python examples/run_negbinomial.py    # abundance counts (negative-binomial)
```

Each prints the true range alongside the recovered posterior mean and 95%
credible interval.
