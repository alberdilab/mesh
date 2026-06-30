"""NumPyro models for spatial-scale-explicit metagenomic inference.

Two observation models share the same latent-field machinery: a single Matern
3/2 Gaussian process whose **range** is the patch size in microns. The field
uses a **non-centred** parameterisation, ``f = eta * (L @ z)`` with
``z ~ Normal(0, 1)`` and ``L`` the Cholesky factor of the correlation matrix,
which keeps the NUTS geometry well behaved.

* :func:`spatial_betabinomial` -- coverage-aware allele frequencies.
* :func:`spatial_negbinomial` -- abundance counts with a depth offset; this is
  the minimal counts-table entry point.

Priors are weakly informative. The **range** prior sits on the micron scale and
defaults to ``LogNormal(log(150), 1.0)`` (broad: roughly 20-1100 microns over
the central 95%); set ``range_prior`` to match your sampling domain.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax import Array

from .kernels import matern32_kernel

__all__ = ["gp_field", "spatial_betabinomial", "spatial_negbinomial"]

# Default weakly-informative range prior, (loc, scale) of a LogNormal in
# log-microns. Centred below 200 micron so the prior does not sit on any
# particular truth, but broad enough to be data-dominated.
#
# Use ``math.log`` (pure Python) rather than ``jnp.log`` so this module-level
# constant does not run a JAX op at import time. Running one here would
# initialise the JAX backend on ``import mesh`` and lock the host device count,
# preventing ``mesh.enable_parallel_chains`` from taking effect afterwards.
DEFAULT_RANGE_PRIOR: tuple[float, float] = (math.log(150.0), 1.0)


def gp_field(
    name: str,
    coords: Array,
    lengthscale: Array,
    eta: Array,
    jitter: float = 1e-6,
) -> Array:
    """Sample a non-centred Matern 3/2 GP field.

    Draws standard-normal innovations ``z`` and returns ``eta * (L @ z)`` where
    ``L`` is the Cholesky factor of the unit-variance Matern correlation matrix.
    Registers a deterministic site ``name`` for the field and a latent site
    ``f"{name}_z"`` for the innovations.

    Parameters
    ----------
    name : str
        Base name for the sample sites.
    coords : Array
        ``(n, 2)`` coordinates in microns.
    lengthscale : Array
        Matern range (patch size).
    eta : Array
        Field standard deviation.
    jitter : float, optional
        Diagonal jitter for Cholesky stability.

    Returns
    -------
    Array
        Field values with shape ``(n,)``.
    """
    n = coords.shape[0]
    k = matern32_kernel(coords, lengthscale, variance=1.0, jitter=jitter)
    chol = jnp.linalg.cholesky(k)
    z = numpyro.sample(f"{name}_z", dist.Normal(0.0, 1.0).expand([n]).to_event(1))
    f = eta * (chol @ z)
    return numpyro.deterministic(name, f)


def spatial_betabinomial(
    coords: Array,
    alt_count: Array,
    total_count: Array,
    *,
    range_prior: tuple[float, float] = DEFAULT_RANGE_PRIOR,
    eta_scale: float = 1.0,
    jitter: float = 1e-6,
) -> None:
    """Coverage-aware spatial allele-frequency model (beta-binomial).

    The logit allele frequency is ``intercept + f`` for a single Matern field
    ``f``; observed alt counts are beta-binomial given per-site coverage, so
    low-coverage sites contribute less information.

    Parameters
    ----------
    coords : Array
        ``(n, 2)`` coordinates in microns.
    alt_count : Array
        Observed alternate-allele counts, shape ``(n,)``.
    total_count : Array
        Per-site coverage (number of trials), shape ``(n,)``.
    range_prior : tuple of float, optional
        ``(loc, scale)`` of the LogNormal range prior, in log-microns.
    eta_scale : float, optional
        Scale of the ``HalfNormal`` prior on the field standard deviation.
    jitter : float, optional
        Diagonal jitter for Cholesky stability.
    """
    range_ = numpyro.sample("range", dist.LogNormal(*range_prior))
    eta = numpyro.sample("eta", dist.HalfNormal(eta_scale))
    intercept = numpyro.sample("intercept", dist.Normal(0.0, 3.0))
    precision = numpyro.sample("precision", dist.HalfNormal(100.0))

    f = gp_field("f", coords, range_, eta, jitter=jitter)
    p = _sigmoid(intercept + f)

    conc1 = p * precision
    conc0 = (1.0 - p) * precision
    numpyro.sample(
        "obs",
        dist.BetaBinomial(conc1, conc0, total_count),
        obs=alt_count,
    )


def spatial_negbinomial(
    coords: Array,
    counts: Array,
    log_offset: Array,
    *,
    range_prior: tuple[float, float] = DEFAULT_RANGE_PRIOR,
    eta_scale: float = 1.0,
    jitter: float = 1e-6,
) -> None:
    """Spatial abundance-count model with a depth offset (negative-binomial).

    Expected counts follow ``log(mu) = intercept + log_offset + f`` for a
    single Matern field ``f``; ``log_offset`` carries the per-sample sequencing
    depth and feature length, ``log(depth) + log(length)``. This is the minimal
    counts-table entry point.

    Parameters
    ----------
    coords : Array
        ``(n, 2)`` coordinates in microns.
    counts : Array
        Observed integer counts, shape ``(n,)``.
    log_offset : Array
        Per-sample log offset ``log(depth) + log(length)``, shape ``(n,)``.
    range_prior : tuple of float, optional
        ``(loc, scale)`` of the LogNormal range prior, in log-microns.
    eta_scale : float, optional
        Scale of the ``HalfNormal`` prior on the field standard deviation.
    jitter : float, optional
        Diagonal jitter for Cholesky stability.
    """
    range_ = numpyro.sample("range", dist.LogNormal(*range_prior))
    eta = numpyro.sample("eta", dist.HalfNormal(eta_scale))
    # Broad intercept prior: the RPKM-style offset pushes the intercept to
    # large-magnitude (negative) log-rates.
    intercept = numpyro.sample("intercept", dist.Normal(0.0, 25.0))
    concentration = numpyro.sample("concentration", dist.Gamma(2.0, 0.1))

    f = gp_field("f", coords, range_, eta, jitter=jitter)
    mu = jnp.exp(intercept + log_offset + f)
    numpyro.sample(
        "obs",
        dist.NegativeBinomial2(mu, concentration),
        obs=counts,
    )


def _sigmoid(x: Array) -> Array:
    return 1.0 / (1.0 + jnp.exp(-x))
