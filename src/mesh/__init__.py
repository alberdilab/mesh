"""MESH: Metagenomic Ecology across Spatial Hierarchies.

Spatial-scale-explicit, coverage-aware Bayesian inference for spatially
resolved shotgun metagenomics. The core model is a hierarchical Bayesian
spatial latent-factor model: latent fields are Gaussian processes with a Matern
covariance whose **range parameter is the patch size, in microns**. Inference
runs on NumPyro / JAX.

This package is **inference-only**: it consumes a validated, analysis-ready
table (see :mod:`mesh.schema`) and produces posterior summaries. It contains no
bioinformatics.
"""

from __future__ import annotations

from .fit import (
    allele_arrays,
    counts_arrays,
    enable_parallel_chains,
    fit_model,
    get_range_posterior,
)
from .kernels import cholesky_factor, matern32_kernel, pairwise_distances
from .model import gp_field, spatial_betabinomial, spatial_negbinomial
from .plots import (
    plot_amplitude_posterior,
    plot_field,
    plot_matern_correlation,
    plot_range_posterior,
    plot_samples,
    plot_scale_comparison,
    plot_variance_partition,
    posterior_field_mean,
)
from .schema import SchemaError, validate_table
from .simulate import SimulatedData, simulate_allele, simulate_counts
from .summaries import (
    decompose_variance,
    summarize_parameters,
    summarize_range,
    variance_partition,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # kernels
    "pairwise_distances",
    "matern32_kernel",
    "cholesky_factor",
    # simulate
    "SimulatedData",
    "simulate_allele",
    "simulate_counts",
    # schema
    "SchemaError",
    "validate_table",
    # model
    "gp_field",
    "spatial_betabinomial",
    "spatial_negbinomial",
    # fit
    "fit_model",
    "enable_parallel_chains",
    "get_range_posterior",
    "counts_arrays",
    "allele_arrays",
    # summaries
    "summarize_range",
    "summarize_parameters",
    "decompose_variance",
    "variance_partition",
    # plots
    "plot_samples",
    "plot_range_posterior",
    "plot_amplitude_posterior",
    "plot_variance_partition",
    "plot_scale_comparison",
    "plot_field",
    "plot_matern_correlation",
    "posterior_field_mean",
]
