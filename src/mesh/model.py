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

from .kernels import matern_kernel, pairwise_distances

__all__ = [
    "gp_field",
    "spatial_betabinomial",
    "spatial_negbinomial",
    "coregionalized_negbinomial",
]

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
    nu: float = 1.5,
    jitter: float = 1e-6,
) -> Array:
    """Sample a non-centred Matern GP field at smoothness ``nu``.

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
    nu : float, optional
        Matern smoothness (boundary sharpness), one of
        :data:`mesh.kernels.MATERN_NU`. Default ``1.5``.
    jitter : float, optional
        Diagonal jitter for Cholesky stability.

    Returns
    -------
    Array
        Field values with shape ``(n,)``.
    """
    n = coords.shape[0]
    k = matern_kernel(coords, lengthscale, nu=nu, variance=1.0, jitter=jitter)
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
    nu: float = 1.5,
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
    nu : float, optional
        Matern smoothness (boundary sharpness) of the field, one of
        :data:`mesh.kernels.MATERN_NU`. Default ``1.5``. Fix it per fit and
        model-compare across values (see :func:`mesh.compare_smoothness`).
    jitter : float, optional
        Diagonal jitter for Cholesky stability.
    """
    range_ = numpyro.sample("range", dist.LogNormal(*range_prior))
    eta = numpyro.sample("eta", dist.HalfNormal(eta_scale))
    intercept = numpyro.sample("intercept", dist.Normal(0.0, 3.0))
    precision = numpyro.sample("precision", dist.HalfNormal(100.0))

    f = gp_field("f", coords, range_, eta, nu=nu, jitter=jitter)
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
    nu: float = 1.5,
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
    nu : float, optional
        Matern smoothness (boundary sharpness) of the field, one of
        :data:`mesh.kernels.MATERN_NU`. Default ``1.5``. Fix it per fit and
        model-compare across values to read out boundary sharpness (see
        :func:`mesh.compare_smoothness`).
    jitter : float, optional
        Diagonal jitter for Cholesky stability.
    """
    range_ = numpyro.sample("range", dist.LogNormal(*range_prior))
    eta = numpyro.sample("eta", dist.HalfNormal(eta_scale))
    # Broad intercept prior: the RPKM-style offset pushes the intercept to
    # large-magnitude (negative) log-rates.
    intercept = numpyro.sample("intercept", dist.Normal(0.0, 25.0))
    concentration = numpyro.sample("concentration", dist.Gamma(2.0, 0.1))

    f = gp_field("f", coords, range_, eta, nu=nu, jitter=jitter)
    mu = jnp.exp(intercept + log_offset + f)
    numpyro.sample(
        "obs",
        dist.NegativeBinomial2(mu, concentration),
        obs=counts,
    )


def coregionalized_negbinomial(
    coords: Array,
    counts: Array,
    log_offset: Array,
    *,
    n_fields: int = 2,
    range_prior: tuple[float, float] = DEFAULT_RANGE_PRIOR,
    loadings_scale: float = 1.0,
    jitter: float = 1e-6,
) -> None:
    r"""Coregionalized spatial abundance model: several features, shared fields.

    A **linear model of coregionalization**. ``n_fields`` unit-variance Matern
    3/2 fields, each with its own **range** (patch size), are shared across
    ``J`` features through a loadings matrix ``W`` (``J x n_fields``):

    .. math::

        \log \mu_{ji} = \beta_j + \text{log\_offset}_{ji}
                        + \sum_k W_{jk}\, f_k(\mathbf{x}_i),

    with :math:`f_k` a unit-variance field at range :math:`\ell_k`. The loadings
    carry the per-field amplitude, so each feature's signal is split across
    scales. This is the M1+ *coregionalization* seed: it lets features
    **co-segregate** (share a territory) and lets the inference **separate two
    scales**. The sign and structure of ``W`` read out ecology directly --
    features that load on the *same* field occupy the same patches (cross-feeding,
    syntrophy, a shared niche); features that load with *opposite* sign
    anti-segregate (competition, niche partitioning).

    The ranges are sampled **ordered** (``range[0] < range[1] < ...``). Ordering
    pins field identity -- it removes the label-switching between fields that
    would otherwise make ``range`` and the loading columns exchangeable -- so the
    posterior assigns each feature to a *specific* scale. Each range is also
    softly **bounded at the spatial extent** of the samples: a range beyond the
    farthest pair of points is unidentifiable (the field flattens to a constant
    the intercepts absorb), a runaway basin the ordered-largest field would
    otherwise drift into. Per-field sign (flipping ``f_k`` and column ``k`` of
    ``W`` together) is left free; it does not affect the ranges or which features
    share a field (compare ``|W|``).

    Parameters
    ----------
    coords : Array
        ``(n, 2)`` coordinates in microns (shared across features).
    counts : Array
        Observed integer counts, shape ``(J, n)`` -- one row per feature.
    log_offset : Array
        Per-sample log offset ``log(depth) + log(length)``, shape ``(J, n)``
        (or ``(n,)``, broadcast across features).
    n_fields : int, optional
        Number of shared latent fields (distinct spatial scales). Default 2.
    range_prior : tuple of float, optional
        ``(loc, scale)`` of the LogNormal-style range prior, in log-microns.
        Applied to each ordered field range.
    loadings_scale : float, optional
        Scale of the ``Normal`` prior on the loadings (field amplitudes).
    jitter : float, optional
        Diagonal jitter for Cholesky stability.
    """
    n_features = counts.shape[0]
    loc, scale = range_prior

    # Ordered ranges: range[0] < range[1] < ... Ordering pins which field is
    # which, so the posterior can assign features to a *specific* scale rather
    # than an exchangeable label. Order on the log scale (exp is monotone, so
    # the ranges stay ordered) and expose the positive ranges as `range`.
    log_range = numpyro.sample(
        "log_range",
        dist.TransformedDistribution(
            dist.Normal(loc, scale).expand([n_fields]).to_event(1),
            dist.transforms.OrderedTransform(),
        ),
    )
    range_ = numpyro.deterministic("range", jnp.exp(log_range))

    # Bound each range at the spatial extent of the samples. A range beyond the
    # farthest pair of points is unidentifiable -- the Matern correlation has not
    # decayed there, so the field degenerates into a near-constant the per-feature
    # intercepts absorb. That "runaway" mode is a second basin the ordered-largest
    # field can fall into (sending its range to the prior edge and collapsing the
    # feature assignment); penalising the log-overshoot beyond the extent removes
    # it without distorting the prior for ranges below the extent.
    extent = jnp.max(pairwise_distances(coords))
    log_overshoot = jnp.clip(jnp.log(range_) - jnp.log(extent), min=0.0)
    numpyro.factor("range_extent", -0.5 * jnp.sum((log_overshoot / 0.1) ** 2))

    # Loadings carry the per-field amplitude (fields are unit-variance).
    loadings = numpyro.sample(
        "loadings",
        dist.Normal(0.0, loadings_scale).expand([n_features, n_fields]).to_event(2),
    )
    intercept = numpyro.sample(
        "intercept", dist.Normal(0.0, 25.0).expand([n_features]).to_event(1)
    )
    concentration = numpyro.sample("concentration", dist.Gamma(2.0, 0.1))

    # One unit-variance field per scale; stack to (n_fields, n).
    fields = [
        gp_field(f"field{k}", coords, range_[k], 1.0, jitter=jitter)
        for k in range(n_fields)
    ]
    field_mat = jnp.stack(fields, axis=0)

    latent = loadings @ field_mat  # (J, n)
    mu = jnp.exp(intercept[:, None] + log_offset + latent)
    numpyro.sample(
        "obs",
        dist.NegativeBinomial2(mu, concentration),
        obs=counts,
    )


def _sigmoid(x: Array) -> Array:
    return 1.0 / (1.0 + jnp.exp(-x))
