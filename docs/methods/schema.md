# The input schema

The schema is the **only** interface between upstream bioinformatics and MESH.
It is a long-format table plus a validator that fails loudly with actionable
messages. The implementation is {mod}`mesh.schema`; the biological rationale is
in [sampling design](../biology/sampling-design.md).

## Required and optional columns

```{list-table}
:header-rows: 1
:widths: 16 14 70

* - Column
  - Required?
  - Description
* - `feature_id`
  - yes
  - Gene/contig (or variant-site) identifier. The catalog must be **shared**
    across all samples.
* - `sample_id`
  - yes
  - Microsample identifier.
* - `x`, `y`
  - yes
  - Coordinates in microns; must be present and finite.
* - `count`
  - yes
  - Reads for the feature (NB model) or alternate-allele count (BB model).
    Non-negative integer.
* - `depth`
  - yes
  - Sequencing depth (NB) or site coverage (BB). Strictly positive.
* - `length`
  - yes
  - Feature length in bp. Non-negative integer.
* - `ref`, `alt`
  - allele model
  - Reference / alternate read counts. Required when `require_allele=True`.
```

## What validation guarantees

{func}`mesh.validate_table` enforces, and raises {class}`mesh.SchemaError` on
violation:

- **Columns present** — all required columns exist (missing coordinates get a
  dedicated message).
- **Coordinates** — `x` and `y` are numeric, non-missing and finite.
- **Dtypes** — numeric columns are numeric; integer-like columns hold whole
  numbers.
- **Shared catalog** — every sample reports the *same* set of `feature_id`s.
  Per-sample catalogs are **rejected**. Duplicate `(sample_id, feature_id)` rows
  are rejected.
- **Missingness** — no nulls in any required column.
- **Ranges** — counts/lengths are non-negative, depth is strictly positive; for
  the allele model `alt ≤ depth` and `count == alt`.

## Usage

```python
import pandas as pd
from mesh import validate_table, SchemaError

df = pd.read_parquet("microsamples.parquet")

try:
    validate_table(df)                 # abundance/counts table
    # validate_table(df, require_allele=True)   # allele table with ref/alt
except SchemaError as err:
    print("Input rejected:", err)
```

Validation returns the table unchanged on success, so it composes:

```python
arrays = counts_arrays(validate_table(df))
```

(`counts_arrays` and `allele_arrays` also validate by default; pass
`validate=False` to skip when you have already validated.)

## Example messages

The validator is designed so an upstream engineer can fix the table from the
message alone:

```text
SchemaError: Table is missing spatial coordinates: ['x', 'y'].
MESH requires per-sample 'x' and 'y' (microns).

SchemaError: Feature catalog is not shared across samples (per-sample catalogs
are not allowed). MESH requires the same feature_id set in every sample.
First offending sample: 's0017' (missing ['geneB']; extra []).
```

## Why a single shared table

Keeping the interface to one validated table makes the statistical method
**auditable** and **independent** of any particular upstream pipeline. MESH never
calls bioinformatics tools; conversely, the pipeline never needs to know how MESH
models the data. The schema is the contract that lets both sides evolve
separately.
