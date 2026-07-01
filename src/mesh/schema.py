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

__all__ = [
    "SchemaError",
    "REQUIRED_COLUMNS",
    "ALLELE_COLUMNS",
    "GENOME_COLUMNS",
    "FAMILY_COLUMNS",
    "TRAIT_COLUMNS",
    "COMPLETENESS_COLUMNS",
    "validate_table",
    "validate_annotations",
]

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

# --- Hierarchical-coregionalization annotation tables (optional). ---
# See docs/methods/coregionalization_hierarchy.md. Membership is many-to-many
# (a gene belongs to one genome and one family but contributes to many traits),
# so it cannot live in columns of the counts table -- it is carried in these
# side tables, keyed on feature_id (or genome_id/trait_id for completeness).
GENOME_COLUMNS: list[str] = ["feature_id", "genome_id"]  # 1:1
FAMILY_COLUMNS: list[str] = ["feature_id", "family_id"]  # n:1
TRAIT_COLUMNS: list[str] = ["feature_id", "trait_id"]  # n:m
COMPLETENESS_COLUMNS: list[str] = ["genome_id", "trait_id", "completeness"]

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


def validate_annotations(
    df: pd.DataFrame,
    *,
    genome: pd.DataFrame | None = None,
    family: pd.DataFrame | None = None,
    trait: pd.DataFrame | None = None,
    completeness: pd.DataFrame | None = None,
    validate: bool = True,
) -> None:
    """Validate the optional hierarchical-coregionalization annotation tables.

    These side tables map the shared feature catalog onto its biological
    organisation (see :doc:`../methods/coregionalization_hierarchy`). All are
    optional; passing ``None`` skips that table. Every ``feature_id`` referenced
    must exist in the counts catalog of ``df``.

    Parameters
    ----------
    df : pandas.DataFrame
        The counts table whose shared feature catalog the annotations key into.
    genome : pandas.DataFrame, optional
        ``feature_id, genome_id`` -- each feature carried by exactly one genome
        (1:1). Every catalog feature must appear exactly once.
    family : pandas.DataFrame, optional
        ``feature_id, family_id`` -- each feature in one gene family (n:1).
    trait : pandas.DataFrame, optional
        ``feature_id, trait_id`` -- feature-to-trait membership (n:m); a feature
        may appear in many traits, but each ``(feature_id, trait_id)`` pair once.
    completeness : pandas.DataFrame, optional
        ``genome_id, trait_id, completeness`` -- genome-inferred trait
        completeness in ``[0, 1]``, one row per ``(genome_id, trait_id)``. When
        ``genome`` and ``trait`` are both given, every genome that carries a
        member gene of a trait must have a completeness row for that trait.

    Raises
    ------
    SchemaError
        On any contract violation, with an actionable message.
    """
    if validate:
        validate_table(df)
    catalog = frozenset(df["feature_id"].unique())

    if genome is not None:
        _check_columns_present(genome, GENOME_COLUMNS)
        _check_feature_refs(genome, catalog, "genome")
        _check_unique_feature(genome, "genome")
        _check_covers_catalog(genome, catalog, "genome")
    if family is not None:
        _check_columns_present(family, FAMILY_COLUMNS)
        _check_feature_refs(family, catalog, "family")
        _check_unique_feature(family, "family")
    if trait is not None:
        _check_columns_present(trait, TRAIT_COLUMNS)
        _check_feature_refs(trait, catalog, "trait")
        if trait.duplicated(TRAIT_COLUMNS).any():
            raise SchemaError(
                "Duplicate (feature_id, trait_id) rows in the trait table; each "
                "feature-to-trait membership must appear exactly once."
            )
    if completeness is not None:
        _check_columns_present(completeness, COMPLETENESS_COLUMNS)
        _check_completeness(completeness, genome=genome, trait=trait)


def _check_feature_refs(
    ann: pd.DataFrame, catalog: frozenset[str], name: str
) -> None:
    unknown = sorted(set(ann["feature_id"].unique()) - catalog)
    if unknown:
        raise SchemaError(
            f"The {name} annotation references feature_id(s) absent from the "
            f"counts catalog: {unknown[:5]}. Annotations must key into the "
            "shared feature catalog."
        )


def _check_unique_feature(ann: pd.DataFrame, name: str) -> None:
    if ann["feature_id"].duplicated().any():
        dup = sorted(ann.loc[ann["feature_id"].duplicated(), "feature_id"].unique())
        raise SchemaError(
            f"The {name} annotation maps feature_id(s) more than once: "
            f"{dup[:5]}. Each feature belongs to exactly one {name}."
        )


def _check_covers_catalog(
    ann: pd.DataFrame, catalog: frozenset[str], name: str
) -> None:
    missing = sorted(catalog - set(ann["feature_id"].unique()))
    if missing:
        raise SchemaError(
            f"The {name} annotation is missing {len(missing)} catalog "
            f"feature(s), e.g. {missing[:5]}. Every feature must be assigned a "
            f"{name}."
        )


def _check_completeness(
    completeness: pd.DataFrame,
    *,
    genome: pd.DataFrame | None,
    trait: pd.DataFrame | None,
) -> None:
    if not ptypes.is_numeric_dtype(completeness["completeness"]):
        raise SchemaError(
            "Column 'completeness' must be numeric, got dtype "
            f"'{completeness['completeness'].dtype}'."
        )
    vals = completeness["completeness"]
    if vals.isna().any() or not _is_finite(vals):
        raise SchemaError("Column 'completeness' has missing/non-finite values.")
    if (vals < 0).any() or (vals > 1).any():
        raise SchemaError("Column 'completeness' must lie in [0, 1].")
    if completeness.duplicated(["genome_id", "trait_id"]).any():
        raise SchemaError(
            "Duplicate (genome_id, trait_id) rows in the completeness table; "
            "each genome-trait completeness must appear exactly once."
        )
    # Referential integrity: every genome carrying a member gene of a trait needs
    # a completeness entry for that trait.
    if genome is not None and trait is not None:
        carried = trait.merge(genome, on="feature_id", how="inner")[
            ["genome_id", "trait_id"]
        ].drop_duplicates()
        have = completeness.set_index(["genome_id", "trait_id"]).index
        missing = carried[~carried.set_index(["genome_id", "trait_id"]).index.isin(have)]
        if not missing.empty:
            first = missing.iloc[0]
            raise SchemaError(
                "The completeness table is missing genome-trait pair(s) that the "
                "incidence tables require, e.g. genome "
                f"'{first['genome_id']}' x trait '{first['trait_id']}' "
                f"({len(missing)} missing). Every genome carrying a member gene "
                "of a trait needs a completeness entry."
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
