"""Smoke tests for the plotting helpers.

Plots cannot be eyeballed in CI, so these assert only that each helper runs on a
real (tiny) fit and returns an :class:`~matplotlib.axes.Axes` with content.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless backend for CI.

import matplotlib.pyplot as plt
import pytest
from matplotlib.axes import Axes

from mesh.fit import (
    allele_arrays,
    coregion_counts_arrays,
    coregion_feature_order,
    counts_arrays,
    fit_model,
)
from mesh.model import (
    anisotropic_negbinomial,
    coregionalized_negbinomial,
    spatial_betabinomial,
    spatial_negbinomial,
)
from mesh.plots import (
    plot_amplitude_posterior,
    plot_anisotropy,
    plot_field,
    plot_loadings,
    plot_matern_correlation,
    plot_range_posterior,
    plot_samples,
    plot_scale_comparison,
    plot_variance_partition,
    posterior_field_mean,
)
from mesh.simulate import (
    simulate_allele,
    simulate_anisotropic,
    simulate_coregionalized,
    simulate_counts,
)
from mesh.summaries import decompose_variance, summarize_anisotropy, variance_partition


@pytest.fixture(scope="module")
def fit():
    sim = simulate_counts(n_samples=60, range_=200.0, eta=1.0, seed=0)
    arrays = counts_arrays(sim.table)
    idata = fit_model(
        spatial_negbinomial,
        num_warmup=100,
        num_samples=100,
        num_chains=1,
        seed=0,
        **arrays,
    )
    return sim, idata


@pytest.fixture(scope="module")
def allele_fit():
    sim = simulate_allele(n_samples=60, range_=120.0, eta=1.0, seed=1)
    arrays = allele_arrays(sim.table)
    idata = fit_model(
        spatial_betabinomial,
        num_warmup=100,
        num_samples=100,
        num_chains=1,
        seed=1,
        **arrays,
    )
    return sim, idata


@pytest.fixture(scope="module")
def coregion_fit():
    sim = simulate_coregionalized(n_samples=60, ranges=(80.0, 300.0), seed=3)
    arrays = coregion_counts_arrays(sim.table)
    idata = fit_model(
        coregionalized_negbinomial,
        num_warmup=100,
        num_samples=100,
        num_chains=1,
        seed=3,
        n_fields=2,
        **arrays,
    )
    return sim, idata


@pytest.fixture(scope="module")
def anisotropic_fit():
    sim = simulate_anisotropic(
        n_samples=60, lengthscales=(280.0, 100.0), eta=1.2, seed=2
    )
    arrays = counts_arrays(sim.table)
    idata = fit_model(
        anisotropic_negbinomial,
        num_warmup=100,
        num_samples=100,
        num_chains=1,
        seed=2,
        **arrays,
    )
    return sim, idata


def test_plot_samples(fit):
    sim, _ = fit
    ax = plot_samples(sim.table, value="count")
    assert isinstance(ax, Axes)
    assert len(ax.collections) == 1  # the scatter
    plt.close("all")


def test_plot_range_posterior(fit):
    sim, idata = fit
    ax = plot_range_posterior(idata, truth=sim.truth["range"])
    assert isinstance(ax, Axes)
    assert len(ax.lines) >= 1
    plt.close("all")


def test_plot_field_truth_and_inferred(fit):
    sim, idata = fit
    ax = plot_field(sim.coords, sim.field, title="truth")
    assert isinstance(ax, Axes)
    inferred = posterior_field_mean(idata)
    assert inferred.shape == (sim.coords.shape[0],)
    plot_field(sim.coords, inferred, ax=ax)
    plt.close("all")


def test_plot_matern_correlation_both_modes(fit):
    _, idata = fit
    ax = plot_matern_correlation(idata)
    assert isinstance(ax, Axes)
    plot_matern_correlation(range_=200.0)
    with pytest.raises(ValueError):
        plot_matern_correlation(idata, range_=200.0)
    with pytest.raises(ValueError):
        plot_matern_correlation()
    plt.close("all")


def test_plot_amplitude_posterior(fit):
    sim, idata = fit
    ax = plot_amplitude_posterior(idata, truth=sim.truth["eta"])
    assert isinstance(ax, Axes)
    assert len(ax.lines) >= 1
    plt.close("all")


def test_variance_partition_negbinomial(fit):
    _, idata = fit
    parts = decompose_variance(idata)
    fraction = parts["fraction"]
    assert fraction.shape == parts["spatial"].shape
    assert ((fraction >= 0.0) & (fraction <= 1.0)).all()

    table = variance_partition(idata)
    assert set(table["quantity"]) == {
        "spatial_variance",
        "nonspatial_variance",
        "spatial_fraction",
    }
    ax = plot_variance_partition(idata)
    assert isinstance(ax, Axes)
    assert ax.get_xlim() == (0.0, 1.0)
    plt.close("all")


def test_variance_partition_betabinomial(allele_fit):
    _, idata = allele_fit
    fraction = decompose_variance(idata)["fraction"]
    assert ((fraction >= 0.0) & (fraction <= 1.0)).all()
    plt.close("all")


def test_plot_scale_comparison(fit, allele_fit):
    _, idata_counts = fit
    _, idata_allele = allele_fit
    ax = plot_scale_comparison(
        [idata_counts, idata_allele],
        labels=["function", "genotype"],
    )
    assert isinstance(ax, Axes)
    assert len(ax.lines) >= 2  # one mean line per fit
    with pytest.raises(ValueError):
        plot_scale_comparison([idata_counts, idata_allele], labels=["only-one"])
    plt.close("all")


def test_anisotropy_summary_and_plot(anisotropic_fit):
    sim, idata = anisotropic_fit
    summary = summarize_anisotropy(idata).set_index("quantity")
    assert set(summary.index) == {
        "range",
        "ell_x",
        "ell_y",
        "anisotropy_ratio",
        "prob_x_longer",
    }
    assert summary.loc["anisotropy_ratio"]["mean"] >= 1.0
    assert 0.0 <= summary.loc["prob_x_longer"]["mean"] <= 1.0

    ax = plot_anisotropy(idata, truth=sim.truth["lengthscales"])
    assert isinstance(ax, Axes)
    assert len(ax.lines) >= 2  # one mean line per axis
    plt.close("all")


def test_plot_loadings(coregion_fit):
    sim, idata = coregion_fit
    feature_ids = coregion_feature_order(sim.table)
    ax = plot_loadings(idata, feature_ids=feature_ids)
    assert isinstance(ax, Axes)
    assert len(ax.images) == 1  # the heatmap
    # One assignment outline per feature.
    assert len(ax.patches) == len(feature_ids)
    plt.close("all")
