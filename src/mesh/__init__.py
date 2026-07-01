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
    compare_smoothness,
    coregion_counts_arrays,
    coregion_feature_order,
    counts_arrays,
    enable_parallel_chains,
    fit_model,
    get_range_posterior,
    nu_label,
)
from .kernels import (
    MATERN_NU,
    anisotropic_matern_kernel,
    anisotropic_scaled_distances,
    cholesky_factor,
    matern32_kernel,
    matern_kernel,
    pairwise_distances,
)
from .model import (
    anisotropic_negbinomial,
    coregionalized_negbinomial,
    gp_field,
    gp_field_anisotropic,
    spatial_betabinomial,
    spatial_negbinomial,
)
from .plots import (
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
from .schema import SchemaError, validate_table
from .simulate import (
    SimulatedData,
    draw_field_anisotropic,
    simulate_allele,
    simulate_anisotropic,
    simulate_coregionalized,
    simulate_counts,
)
from .summaries import (
    decompose_variance,
    summarize_anisotropy,
    summarize_loadings,
    summarize_parameters,
    summarize_range,
    variance_partition,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # kernels
    "pairwise_distances",
    "anisotropic_scaled_distances",
    "matern_kernel",
    "matern32_kernel",
    "anisotropic_matern_kernel",
    "cholesky_factor",
    "MATERN_NU",
    # simulate
    "SimulatedData",
    "simulate_allele",
    "simulate_counts",
    "simulate_anisotropic",
    "simulate_coregionalized",
    "draw_field_anisotropic",
    # schema
    "SchemaError",
    "validate_table",
    # model
    "gp_field",
    "gp_field_anisotropic",
    "spatial_betabinomial",
    "spatial_negbinomial",
    "anisotropic_negbinomial",
    "coregionalized_negbinomial",
    # fit
    "fit_model",
    "enable_parallel_chains",
    "get_range_posterior",
    "counts_arrays",
    "allele_arrays",
    "coregion_counts_arrays",
    "coregion_feature_order",
    "nu_label",
    "compare_smoothness",
    # summaries
    "summarize_range",
    "summarize_parameters",
    "decompose_variance",
    "variance_partition",
    "summarize_anisotropy",
    "summarize_loadings",
    # plots
    "plot_samples",
    "plot_range_posterior",
    "plot_amplitude_posterior",
    "plot_variance_partition",
    "plot_scale_comparison",
    "plot_anisotropy",
    "plot_loadings",
    "plot_field",
    "plot_matern_correlation",
    "posterior_field_mean",
]
