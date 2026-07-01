"""Tests for the input-contract validator (:mod:`mesh.schema`)."""

from __future__ import annotations

import pandas as pd
import pytest

from mesh.schema import SchemaError, validate_annotations, validate_table
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


# --- Annotation tables (hierarchical coregionalization) --------------------


def _valid_annotations():
    """Genome/family/trait/completeness tables for the two-feature catalog."""
    genome = pd.DataFrame(
        {"feature_id": ["f0", "f1"], "genome_id": ["g0", "g1"]}
    )
    family = pd.DataFrame(
        {"feature_id": ["f0", "f1"], "family_id": ["K1", "K1"]}
    )
    # f0 is in two traits; f1 in one.
    trait = pd.DataFrame(
        {
            "feature_id": ["f0", "f0", "f1"],
            "trait_id": ["M1", "M2", "M1"],
        }
    )
    completeness = pd.DataFrame(
        {
            "genome_id": ["g0", "g0", "g1"],
            "trait_id": ["M1", "M2", "M1"],
            "completeness": [1.0, 0.5, 0.75],
        }
    )
    return genome, family, trait, completeness


def test_valid_annotations_pass():
    df = _valid_two_feature_table()
    genome, family, trait, completeness = _valid_annotations()
    # Should not raise.
    validate_annotations(
        df, genome=genome, family=family, trait=trait, completeness=completeness
    )


def test_annotations_all_none_pass():
    validate_annotations(_valid_two_feature_table())


def test_rejects_unknown_feature_in_annotation():
    df = _valid_two_feature_table()
    genome = pd.DataFrame(
        {"feature_id": ["f0", "fZ"], "genome_id": ["g0", "g1"]}
    )
    with pytest.raises(SchemaError, match="absent from the counts catalog"):
        validate_annotations(df, genome=genome)


def test_rejects_genome_not_covering_catalog():
    df = _valid_two_feature_table()
    genome = pd.DataFrame({"feature_id": ["f0"], "genome_id": ["g0"]})
    with pytest.raises(SchemaError, match="missing .* catalog feature"):
        validate_annotations(df, genome=genome)


def test_rejects_feature_in_two_genomes():
    df = _valid_two_feature_table()
    genome = pd.DataFrame(
        {"feature_id": ["f0", "f0", "f1"], "genome_id": ["g0", "g1", "g1"]}
    )
    with pytest.raises(SchemaError, match="more than once"):
        validate_annotations(df, genome=genome)


def test_rejects_duplicate_trait_membership():
    df = _valid_two_feature_table()
    trait = pd.DataFrame(
        {"feature_id": ["f0", "f0"], "trait_id": ["M1", "M1"]}
    )
    with pytest.raises(SchemaError, match="Duplicate .*trait"):
        validate_annotations(df, trait=trait)


def test_rejects_completeness_out_of_range():
    df = _valid_two_feature_table()
    completeness = pd.DataFrame(
        {"genome_id": ["g0"], "trait_id": ["M1"], "completeness": [1.5]}
    )
    with pytest.raises(SchemaError, match=r"\[0, 1\]"):
        validate_annotations(df, completeness=completeness)


def test_rejects_missing_completeness_for_carried_trait():
    df = _valid_two_feature_table()
    genome, family, trait, completeness = _valid_annotations()
    # Drop g1 x M1, which f1 (in g1) carries -> referential gap.
    completeness = completeness.drop(
        completeness[
            (completeness.genome_id == "g1") & (completeness.trait_id == "M1")
        ].index
    )
    with pytest.raises(SchemaError, match="missing genome-trait pair"):
        validate_annotations(
            df, genome=genome, trait=trait, completeness=completeness
        )
