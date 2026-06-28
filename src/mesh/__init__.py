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

from .fit import allele_arrays, counts_arrays, fit_model, get_range_posterior
from .kernels import cholesky_factor, matern32_kernel, pairwise_distances
from .model import gp_field, spatial_betabinomial, spatial_negbinomial
from .schema import SchemaError, validate_table
from .simulate import SimulatedData, simulate_allele, simulate_counts
from .summaries import summarize_parameters, summarize_range

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
    "get_range_posterior",
    "counts_arrays",
    "allele_arrays",
    # summaries
    "summarize_range",
    "summarize_parameters",
]
