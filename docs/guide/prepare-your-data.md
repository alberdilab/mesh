# Step 1 · Prepare your data

MESH reads **one table**. Get the table right and everything else is
mechanical. This page explains what each column means, *why* MESH needs it, and
the handful of mistakes the validator will stop you from making.

## The shape: one row per feature per location

MESH expects a **long-format** (tidy) table. "Long" means: instead of a grid of
features × samples, you have **one row for every (feature, location) pair**. If
you measured 50 genes at 200 microsamples, that is 50 × 200 = 10,000 rows.

| `feature_id` | `sample_id` | `x` | `y` | `count` | `depth` | `length` |
|---|---|---|---|---|---|---|
| geneA | s001 | 12.0 | 5.0 | 31 | 1.2e6 | 900 |
| geneB | s001 | 12.0 | 5.0 | 4  | 1.2e6 | 1500 |
| geneA | s002 | 60.0 | 5.0 | 27 | 9.0e5 | 900 |
| … | … | … | … | … | … | … |

:::{admonition} Why "long" and not a spreadsheet grid?
:class: note

A grid hides the coordinates and the per-sample sequencing effort inside the
column headers, where a model can't see them. The long format puts every fact
MESH needs — *where* the sample is, *how deep* it was sequenced, *how long* the
feature is — onto the same row as the measurement. One row is one self-contained
observation.
:::

## What each column means, and why it's there

```{list-table}
:header-rows: 1
:widths: 16 40 44

* - Column
  - What it is
  - Why MESH needs it
* - `feature_id`
  - The gene, contig, or variant site you measured.
  - This is the **thing whose patch size you want**. The same `feature_id` must
    mean the same thing in every sample (a *shared catalog* — see below).
* - `sample_id`
  - The microsample the row came from.
  - Lets MESH group the features that share a location.
* - `x`, `y`
  - The location of the microsample, **in microns**.
  - This is the spatial part. The patch size is read off *how the signal changes
    with distance*, so without real coordinates there is nothing to estimate.
* - `count`
  - The measured number: reads on a gene (abundance) **or** alt-allele reads
    (allele model).
  - The signal itself.
* - `depth`
  - How much sequencing that sample/site got: library **depth** (abundance) or
    site **coverage** (alleles).
  - Tells MESH *how much to trust* each number. A count of 5 from a shallow
    sample and a count of 5 from a deep one mean very different things.
* - `length`
  - The feature length in base pairs (abundance model).
  - A longer gene collects more reads at the same true abundance; `length` lets
    MESH correct for that. (Ignored by the allele model.)
* - `ref`, `alt`
  - *Optional, allele model only:* reference and alternate read counts.
  - The substrate for the allele-frequency model.
```

Units matter: **coordinates must be in microns**, because the headline output —
the patch size — is reported in the same units. If your coordinates are in
millimetres, convert them first, or your patch sizes will be off by 1000×.

## The one rule people trip on: a shared catalog

Every sample must report the **same set of `feature_id`s**. If sample `s001`
lists `geneA, geneB, geneC` then *every* sample must list `geneA, geneB, geneC`
— even where the count is `0`.

:::{admonition} Why this rule exists
:class: important

MESH lays a spatial field over *one feature measured at many places*. That only
makes sense if "geneA" is the same gene everywhere. If each sample invented its
own feature list (a *per-sample catalog*), then a missing row would be ambiguous:
was the gene absent, or just never looked for? MESH refuses to guess. Upstream,
this means quantifying every microsample against **one common reference**
(a shared gene catalog, contig set, or variant panel) and writing explicit zeros.
:::

The validator rejects per-sample catalogs with a message naming the first
offending sample, so you find out immediately rather than getting a silently
wrong answer.

## Load and validate — before you fit anything

```python
import pandas as pd
from mesh import validate_table, SchemaError

df = pd.read_parquet("microsamples.parquet")   # or read_csv, etc.

try:
    validate_table(df)                          # abundance/counts table
    # validate_table(df, require_allele=True)   # allele table: also needs ref/alt
except SchemaError as err:
    print("Input rejected:", err)
```

`validate_table` checks the things that quietly ruin a fit: missing coordinates,
non-numeric values, nulls, a broken shared catalog, duplicate rows, negative
counts, non-positive depth. It **returns the table unchanged on success**, so you
can chain it:

```python
from mesh import counts_arrays
arrays = counts_arrays(validate_table(df))      # validate, then build model inputs
```

:::{admonition} Treat a validation error as a friend
:class: tip

Every `SchemaError` is written so an upstream engineer can fix the table from the
message alone — for example, *"Feature catalog is not shared across samples …
First offending sample: 's0017' (missing ['geneB']; extra [])."* It is far
cheaper to fix the table now than to discover a subtle bias after a long fit.
:::

A worked, design-level view of all this — how many microsamples, what spacing,
why coverage matters — is in [sampling design](../biology/sampling-design.md).
The exact column contract and every validation rule are in
[the input schema](../methods/schema.md).

## Checklist before moving on

- [ ] One row per (feature, location); coordinates are in **microns**.
- [ ] Every sample lists the **same** `feature_id`s (zeros written explicitly).
- [ ] `depth` is recorded faithfully per sample/site; `length` per feature.
- [ ] `validate_table(df)` runs without raising.

Next: [decide which model fits your data, and why](choose-your-model.md).
