"""Tests for the input-contract validator (:mod:`mesh.schema`)."""

from __future__ import annotations

import pandas as pd
import pytest

from mesh.schema import SchemaError, validate_table
from mesh.simulate import simulate_counts


def _valid_two_feature_table() -> pd.DataFrame:
    """Two samples x two features, shared catalog -- a valid table."""
    rows = []
    for sample, (x, y) in zip(["s0", "s1"], [(0.0, 0.0), (10.0, 5.0)], strict=True):
        for feat in ["f0", "f1"]:
            rows.append(
                {
                    "feature_id": feat,
                    "sample_id": sample,
                    "x": x,
                    "y": y,
                    "count": 3,
                    "depth": 1000.0,
                    "length": 500,
                }
            )
    return pd.DataFrame(rows)


def test_valid_table_passes():
    df = _valid_two_feature_table()
    # Returns the table unchanged; should not raise.
    assert validate_table(df) is df


def test_simulated_counts_table_passes():
    sim = simulate_counts(n_samples=20, seed=1)
    validate_table(sim.table)


def test_rejects_non_shared_catalog():
    df = _valid_two_feature_table()
    # Drop one feature from a single sample -> per-sample catalog.
    df = df.drop(df[(df.sample_id == "s1") & (df.feature_id == "f1")].index)
    with pytest.raises(SchemaError, match="not shared"):
        validate_table(df)


def test_rejects_missing_coordinates():
    df = _valid_two_feature_table().drop(columns=["x", "y"])
    with pytest.raises(SchemaError, match="coordinates"):
        validate_table(df)


def test_rejects_missing_required_column():
    df = _valid_two_feature_table().drop(columns=["count"])
    with pytest.raises(SchemaError, match="missing required columns"):
        validate_table(df)


def test_rejects_coordinate_nans():
    df = _valid_two_feature_table()
    df.loc[0, "x"] = float("nan")
    with pytest.raises(SchemaError, match="missing value"):
        validate_table(df)


def test_rejects_duplicate_rows():
    df = _valid_two_feature_table()
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    with pytest.raises(SchemaError, match="[Dd]uplicate"):
        validate_table(df)
