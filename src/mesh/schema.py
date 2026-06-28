"""Input contract between the bioinformatics pipeline and MESH.

MESH is inference-only. It consumes a **validated, analysis-ready** long-format
table and never performs any bioinformatics. The table and this validator are
the *only* interface between the two projects.

Required columns
----------------
``feature_id, sample_id, x, y, count, depth, length``

Optional columns (allele model)
-------------------------------
``ref, alt``

Validation guarantees
---------------------
* coordinates (``x``, ``y``) exist and are finite,
* the feature catalog is **shared** -- the same ``feature_id`` set in every
  sample (per-sample catalogs are rejected),
* dtypes are numeric where required and counts are non-negative integers,
* no missing values in required columns.

All failures raise :class:`SchemaError` with an actionable message.
"""

from __future__ import annotations

import pandas as pd
from pandas.api import types as ptypes

__all__ = ["SchemaError", "REQUIRED_COLUMNS", "ALLELE_COLUMNS", "validate_table"]

REQUIRED_COLUMNS: list[str] = [
    "feature_id",
    "sample_id",
    "x",
    "y",
    "count",
    "depth",
    "length",
]
ALLELE_COLUMNS: list[str] = ["ref", "alt"]

_COORD_COLUMNS = ["x", "y"]
_NUMERIC_COLUMNS = ["x", "y", "count", "depth", "length"]
_NONNEGATIVE_INT_COLUMNS = ["count", "length"]


class SchemaError(ValueError):
    """Raised when an input table violates the MESH interface contract."""


def validate_table(df: pd.DataFrame, *, require_allele: bool = False) -> pd.DataFrame:
    """Validate a long-format input table against the MESH contract.

    Parameters
    ----------
    df : pandas.DataFrame
        Candidate input table.
    require_allele : bool, optional
        If ``True``, also require the ``ref``/``alt`` columns (allele model).

    Returns
    -------
    pandas.DataFrame
        The validated table (unchanged) for convenient chaining.

    Raises
    ------
    SchemaError
        If any part of the contract is violated. The message names the offending
        columns/samples so the upstream pipeline can be fixed.
    """
    if not isinstance(df, pd.DataFrame):
        raise SchemaError(f"Expected a pandas DataFrame, got {type(df).__name__}.")

    required = list(REQUIRED_COLUMNS)
    if require_allele:
        required = required + ALLELE_COLUMNS

    _check_columns_present(df, required)
    _check_coordinates(df)
    _check_numeric_dtypes(df, require_allele=require_allele)
    _check_missingness(df, required)
    _check_nonnegative(df, require_allele=require_allele)
    _check_shared_catalog(df)
    if require_allele:
        _check_allele_consistency(df)

    return df


def _check_columns_present(df: pd.DataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        if set(_COORD_COLUMNS) & set(missing):
            raise SchemaError(
                "Table is missing spatial coordinates: "
                f"{sorted(set(_COORD_COLUMNS) & set(missing))}. "
                "MESH requires per-sample 'x' and 'y' (microns)."
            )
        raise SchemaError(
            f"Table is missing required columns: {missing}. "
            f"Required columns are {required}."
        )


def _check_coordinates(df: pd.DataFrame) -> None:
    for col in _COORD_COLUMNS:
        if not ptypes.is_numeric_dtype(df[col]):
            raise SchemaError(
                f"Coordinate column '{col}' must be numeric, got dtype "
                f"'{df[col].dtype}'."
            )
        if df[col].isna().any():
            n = int(df[col].isna().sum())
            raise SchemaError(
                f"Coordinate column '{col}' has {n} missing value(s); "
                "every sample must have finite coordinates."
            )
        if not _is_finite(df[col]):
            raise SchemaError(
                f"Coordinate column '{col}' contains non-finite values "
                "(inf/-inf); coordinates must be finite microns."
            )


def _check_numeric_dtypes(df: pd.DataFrame, *, require_allele: bool) -> None:
    cols = list(_NUMERIC_COLUMNS)
    if require_allele:
        cols = cols + ALLELE_COLUMNS
    bad = {c: str(df[c].dtype) for c in cols if not ptypes.is_numeric_dtype(df[c])}
    if bad:
        raise SchemaError(f"Columns must be numeric but are not: {bad}.")


def _check_missingness(df: pd.DataFrame, required: list[str]) -> None:
    counts = {c: int(df[c].isna().sum()) for c in required if df[c].isna().any()}
    if counts:
        raise SchemaError(
            f"Required columns contain missing values: {counts}. "
            "Missing values are not allowed in the input contract."
        )


def _check_nonnegative(df: pd.DataFrame, *, require_allele: bool) -> None:
    cols = list(_NONNEGATIVE_INT_COLUMNS)
    if require_allele:
        cols = cols + ALLELE_COLUMNS
    for col in cols:
        if (df[col] < 0).any():
            raise SchemaError(f"Column '{col}' must be non-negative.")
        if not _is_integer_valued(df[col]):
            raise SchemaError(
                f"Column '{col}' must be integer-valued (whole numbers)."
            )
    if (df["depth"] <= 0).any():
        raise SchemaError("Column 'depth' must be strictly positive.")


def _check_shared_catalog(df: pd.DataFrame) -> None:
    by_sample = df.groupby("sample_id")["feature_id"]
    catalogs = by_sample.agg(lambda s: frozenset(s))
    # Duplicate (sample_id, feature_id) pairs are also a contract violation.
    if df.duplicated(["sample_id", "feature_id"]).any():
        raise SchemaError(
            "Duplicate (sample_id, feature_id) rows found; each feature must "
            "appear exactly once per sample."
        )
    reference = catalogs.iloc[0]
    mismatched = [s for s, cat in catalogs.items() if cat != reference]
    if mismatched:
        example = mismatched[0]
        missing = sorted(reference - catalogs[example])
        extra = sorted(catalogs[example] - reference)
        raise SchemaError(
            "Feature catalog is not shared across samples (per-sample catalogs "
            "are not allowed). MESH requires the same feature_id set in every "
            f"sample. First offending sample: '{example}' "
            f"(missing {missing[:5]}; extra {extra[:5]})."
        )


def _check_allele_consistency(df: pd.DataFrame) -> None:
    if (df["alt"] > df["depth"]).any():
        raise SchemaError("Allele 'alt' count cannot exceed 'depth' (coverage).")
    if (df["count"] != df["alt"]).any():
        raise SchemaError(
            "For the allele model, 'count' must equal the 'alt' allele count."
        )


def _is_finite(series: pd.Series) -> bool:
    import numpy as np

    arr = series.to_numpy()
    if not np.issubdtype(arr.dtype, np.number):
        return True
    return bool(np.isfinite(arr).all())


def _is_integer_valued(series: pd.Series) -> bool:
    if ptypes.is_integer_dtype(series):
        return True
    import numpy as np

    arr = series.to_numpy()
    return bool(np.all(np.equal(np.mod(arr, 1), 0)))
