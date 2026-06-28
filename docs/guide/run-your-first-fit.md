# Step 3 · Run your first fit

With a validated table and a chosen model, fitting is three calls: build the
arrays, run the sampler, read the summary. This page runs the whole thing on
**simulated data with a known answer**, so you can see the machine recover a
patch size you already know — the best way to build trust before you point it at
real data.

## Learn on simulated data first

MESH ships a generator that builds a table with a patch size *you choose*. If the
fit can't recover a known 200 µm, something is wrong with the setup, not with
your biology. This is also exactly how MESH tests itself
([simulation & recovery](../methods/simulation.md)).

```python
from mesh import (
    simulate_counts, validate_table, counts_arrays,
    fit_model, spatial_negbinomial, summarize_range,
)

# 1. Make a counts table whose TRUE patch size is 200 µm.
sim = simulate_counts(n_samples=150, range_=200.0, seed=0)

# 2. Validate (always — it's cheap insurance).
validate_table(sim.table)

# 3. Turn the table into the arrays the model consumes:
#    coordinates, counts, and the fixed log(depth)+log(length) offset.
arrays = counts_arrays(sim.table)

# 4. Fit, then summarise the patch-size posterior.
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

You should get a one-row table whose 95% interval brackets the true 200 µm:

```text
  parameter   mean  median    sd  hdi_low  hdi_high  hdi_prob  r_hat  ess_bulk
0     range  209.8   206.1  40.2    134.4     289.2      0.95  1.004     223.0
```

The true value (200) sits comfortably inside `[134, 289]`. The machine works;
now you understand what each knob did.

## What `counts_arrays` did for you

`counts_arrays` (or `allele_arrays` for the allele model) is the bridge from the
human-friendly table to the model's numeric inputs. It pulls out the coordinates,
the counts, and — crucially — **builds the `log(depth) + log(length)` offset for
you** so you never assemble it by hand. It validates by default; pass
`validate=False` only if you already validated.

## Every sampler setting, in plain words

MESH does **Bayesian inference** with a sampler called **NUTS**. Instead of
returning a single best-guess patch size, it draws thousands of plausible values
in proportion to how well each explains your data. That cloud of draws *is* the
answer; the credible interval is just its middle 95%.

```{list-table}
:header-rows: 1
:widths: 26 74

* - Setting
  - What it does, and how to choose it
* - `num_warmup`
  - Throwaway tuning steps where NUTS learns the shape of the problem before it
    starts collecting. **500** is a good default; raise it if diagnostics complain.
* - `num_samples`
  - The draws you actually keep per chain. More draws = smoother, less noisy
    interval. 500–1000 is reasonable for a real fit.
* - `num_chains`
  - How many independent runs to start from different places. **Use ≥ 2.** You
    need at least two to compute $\hat R$, the check that the runs agree (below).
* - `chain_method`
  - How the chains share your hardware. The default `"vectorized"` runs them all
    at once on one device and is the fastest choice on a laptop CPU — leave it
    alone unless you have several cores/GPUs to spread chains across, in which
    case see [running chains in parallel](../methods/inference.md#running-chains-in-parallel).
* - `seed`
  - Random seed. Same seed + same data = identical result, so your analysis is
    reproducible.
* - `target_accept_prob`
  - How carefully the sampler steps (default 0.9). If you see *divergences*,
    raise it toward 0.95 — smaller, more cautious steps, a bit slower.
* - `progress_bar`
  - Set `True` to watch it run. Off by default.
```

:::{admonition} Why ≥ 2 chains is non-negotiable
:class: important

A single chain can look perfectly healthy while being stuck in the wrong place.
Two or more chains started independently let MESH check they **landed in the same
answer** ($\hat R \approx 1.0$). One chain gives you no such safety net. The tiny
extra runtime is the cheapest insurance in the whole workflow.
:::

## How long will this take?

At MESH's scale — a few hundred to ~2000 locations — each step is a small matrix
operation that takes milliseconds, and MESH uses **exact** Gaussian processes (no
approximations that would blur the very patch size you're estimating). The wall
time is dominated by the number of steps and by JAX compiling the model on the
first run, not by the data size. A few hundred microsamples fits comfortably on a
laptop CPU; no GPU needed
([why exact GPs](../methods/inference.md#why-exact-gps-here)).

## The allele model is the same shape

Swap three names and the workflow is identical:

```python
from mesh import simulate_allele, allele_arrays, spatial_betabinomial

sim = simulate_allele(n_samples=150, range_=200.0, seed=0)
arrays = allele_arrays(validate_table(sim.table, require_allele=True))
idata = fit_model(spatial_betabinomial, num_warmup=500, num_samples=500,
                  num_chains=2, seed=0, **arrays)
```

## On your real data

Once the simulated round-trip makes sense, point the same code at your validated
table — skip `simulate_*`, feed your own `df` to `counts_arrays`/`allele_arrays`:

```python
arrays = counts_arrays(validate_table(my_df))
idata = fit_model(spatial_negbinomial, num_warmup=800, num_samples=800,
                  num_chains=4, seed=0, **arrays)
```

Then **do not read the number yet** — first check the fit is trustworthy. That's
the next page.

Next: [read your results, and know when to trust them](read-your-results.md).
