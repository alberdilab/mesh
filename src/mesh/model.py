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

from .kernels import (
    anisotropic_matern_kernel,
    matern_kernel,
    pairwise_distances,
)

__all__ = [
    "gp_field",
    "gp_field_anisotropic",
    "spatial_betabinomial",
    "spatial_negbinomial",
    "anisotropic_negbinomial",
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


def gp_field_anisotropic(
    name: str,
    coords: Array,
    lengthscales: Array,
    eta: Array,
    nu: float = 1.5,
    jitter: float = 1e-6,
) -> Array:
    """Sample a non-centred **anisotropic** Matern GP field.

    Like :func:`gp_field`, but the correlation uses a **separate range per
    coordinate axis** (:func:`mesh.anisotropic_matern_kernel`), so the patch
    size can differ along ``x`` and ``y`` -- the *direction* axis of spatial
    architecture.

    Parameters
    ----------
    name : str
        Base name for the sample sites.
    coords : Array
        ``(n, 2)`` coordinates in microns.
    lengthscales : Array
        Per-axis Matern ranges ``(ell_x, ell_y)`` (patch size along each axis).
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
    k = anisotropic_matern_kernel(
        coords, lengthscales, nu=nu, variance=1.0, jitter=jitter
    )
    chol = jnp.linalg.cholesky(k)
    z = numpyro.sample(f"{name}_z", dist.Normal(0.0, 1.0).expand([n]).to_event(1))
    f = eta * (chol @ z)
    return numpyro.deterministic(name, f)


def _axis_lengthscales(range_: Array, log_ratio: Array) -> Array:
    """Split a range into per-axis lengthscales given a signed log anisotropy.

    ``ell_x = range * exp(+rho/2)``, ``ell_y = range * exp(-rho/2)``, so the
    geometric mean stays ``range`` and ``exp(rho) = ell_x / ell_y``. Works
    elementwise, so it maps a vector of per-field ranges/ratios to a
    ``(n_fields, 2)`` matrix. See :func:`anisotropic_negbinomial`.
    """
    ell_x = range_ * jnp.exp(0.5 * log_ratio)
    ell_y = range_ * jnp.exp(-0.5 * log_ratio)
    return jnp.stack([ell_x, ell_y], axis=-1)


def _sample_field_latent(
    coords: Array,
    *,
    n_features: int,
    n_fields: int,
    anisotropic: bool,
    nu: float,
    range_prior: tuple[float, float],
    eta_scale: float,
    anisotropy_scale: float,
    loadings_scale: float,
    jitter: float,
) -> Array:
    r"""Sample the latent spatial contribution shared by every likelihood.

    This is the composable core: it builds the latent ``(n_features, n)`` matrix
    that the negative-binomial and beta-binomial likelihoods add to their
    intercept (and offset). Every model in MESH is a choice of three orthogonal
    field knobs handled here -- ``n_fields`` (co-existing scales), ``anisotropic``
    (per-axis ranges, the direction axis) and ``nu`` (boundary sharpness) -- over
    one or more features.

    Two regimes, matching the identifiability structure of the models:

    * **Single field, single feature** (``n_fields == 1`` and
      ``n_features == 1``). The amplitude is a **positive** ``eta``
      (``HalfNormal``); the field's sign is fixable so this stays unimodal. The
      latent is ``eta * f`` for one Matern field ``f`` (isotropic, or anisotropic
      with a signed ``log_ratio``).
    * **Linear model of coregionalization** (otherwise). ``n_fields``
      unit-variance fields at **ordered**, extent-bounded ranges are mixed into
      the features by a **signed** loadings matrix ``W`` (``n_features x
      n_fields``); the loadings carry the per-field amplitude, and their sign is
      free (read magnitudes for assignment). Anisotropy, if on, gives each field
      its own ``log_ratio``.

    Returns the latent ``(n_features, n)``; the likelihood adds the intercept and
    (for counts) the offset.
    """
    simple = n_fields == 1 and n_features == 1

    if simple:
        range_ = numpyro.sample("range", dist.LogNormal(*range_prior))
        eta = numpyro.sample("eta", dist.HalfNormal(eta_scale))
        if anisotropic:
            log_ratio = numpyro.sample("log_ratio", dist.Normal(0.0, anisotropy_scale))
            lengthscales = numpyro.deterministic(
                "lengthscales", _axis_lengthscales(range_, log_ratio)
            )
            numpyro.deterministic("anisotropy", jnp.exp(log_ratio))
            f = gp_field_anisotropic("f", coords, lengthscales, eta, nu=nu, jitter=jitter)
        else:
            f = gp_field("f", coords, range_, eta, nu=nu, jitter=jitter)
        return f[None, :]

    # --- Linear model of coregionalization (K > 1 or J > 1). ---
    loc, scale = range_prior
    # Ordered ranges pin field identity (no label-switching between fields). Order
    # on the log scale (exp is monotone) and expose the positive ranges.
    log_range = numpyro.sample(
        "log_range",
        dist.TransformedDistribution(
            dist.Normal(loc, scale).expand([n_fields]).to_event(1),
            dist.transforms.OrderedTransform(),
        ),
    )
    range_ = numpyro.deterministic("range", jnp.exp(log_range))

    # Bound each range at the spatial extent: a range beyond the farthest pair is
    # unidentifiable (the field flattens to a constant the intercepts absorb), a
    # runaway basin the ordered-largest field would otherwise drift into.
    extent = jnp.max(pairwise_distances(coords))
    log_overshoot = jnp.clip(jnp.log(range_) - jnp.log(extent), min=0.0)
    numpyro.factor("range_extent", -0.5 * jnp.sum((log_overshoot / 0.1) ** 2))

    # Loadings carry the per-field amplitude (fields are unit-variance).
    loadings = numpyro.sample(
        "loadings",
        dist.Normal(0.0, loadings_scale).expand([n_features, n_fields]).to_event(2),
    )

    if anisotropic:
        log_ratio = numpyro.sample(
            "log_ratio", dist.Normal(0.0, anisotropy_scale).expand([n_fields]).to_event(1)
        )
        lengthscales = numpyro.deterministic(
            "lengthscales", _axis_lengthscales(range_, log_ratio)
        )
        numpyro.deterministic("anisotropy", jnp.exp(log_ratio))
        fields = [
            gp_field_anisotropic(
                f"field{k}", coords, lengthscales[k], 1.0, nu=nu, jitter=jitter
            )
            for k in range(n_fields)
        ]
    else:
        fields = [
            gp_field(f"field{k}", coords, range_[k], 1.0, nu=nu, jitter=jitter)
            for k in range(n_fields)
        ]
    field_mat = jnp.stack(fields, axis=0)  # (n_fields, n)
    return loadings @ field_mat  # (n_features, n)


def spatial_betabinomial(
    coords: Array,
    alt_count: Array,
    total_count: Array,
    *,
    anisotropic: bool = False,
    range_prior: tuple[float, float] = DEFAULT_RANGE_PRIOR,
    eta_scale: float = 1.0,
    anisotropy_scale: float = 1.0,
    nu: float = 1.5,
    jitter: float = 1e-6,
) -> None:
    """Coverage-aware spatial allele-frequency model (beta-binomial).

    The logit allele frequency is ``intercept + f`` for a single Matern field
    ``f``; observed alt counts are beta-binomial given per-site coverage, so
    low-coverage sites contribute less information. The field composes the
    boundary-sharpness (``nu``) and direction (``anisotropic``) axes through the
    shared field core; it is single-feature (single-scale) by construction.

    Parameters
    ----------
    coords : Array
        ``(n, 2)`` coordinates in microns.
    alt_count : Array
        Observed alternate-allele counts, shape ``(n,)``.
    total_count : Array
        Per-site coverage (number of trials), shape ``(n,)``.
    anisotropic : bool, optional
        If ``True``, give the field a separate patch size per axis (the
        *direction* axis; see :func:`anisotropic_negbinomial`). Default ``False``.
    range_prior : tuple of float, optional
        ``(loc, scale)`` of the LogNormal range prior, in log-microns.
    eta_scale : float, optional
        Scale of the ``HalfNormal`` prior on the field standard deviation.
    anisotropy_scale : float, optional
        Scale of the ``Normal(0, .)`` prior on the log anisotropy (used only when
        ``anisotropic``); centred at isotropy.
    nu : float, optional
        Matern smoothness (boundary sharpness) of the field, one of
        :data:`mesh.kernels.MATERN_NU`. Default ``1.5``. Fix it per fit and
        model-compare across values (see :func:`mesh.compare_smoothness`).
    jitter : float, optional
        Diagonal jitter for Cholesky stability.
    """
    intercept = numpyro.sample("intercept", dist.Normal(0.0, 3.0))
    precision = numpyro.sample("precision", dist.HalfNormal(100.0))

    latent = _sample_field_latent(
        coords,
        n_features=1,
        n_fields=1,
        anisotropic=anisotropic,
        nu=nu,
        range_prior=range_prior,
        eta_scale=eta_scale,
        anisotropy_scale=anisotropy_scale,
        loadings_scale=1.0,
        jitter=jitter,
    )
    p = _sigmoid(intercept + latent[0])

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
    n_fields: int = 1,
    anisotropic: bool = False,
    range_prior: tuple[float, float] = DEFAULT_RANGE_PRIOR,
    eta_scale: float = 1.0,
    anisotropy_scale: float = 1.0,
    loadings_scale: float = 1.0,
    nu: float = 1.5,
    jitter: float = 1e-6,
) -> None:
    """Spatial abundance-count model with a depth offset (negative-binomial).

    Expected counts follow ``log(mu) = intercept + log_offset + latent``, where
    ``log_offset`` carries the per-sample sequencing depth and feature length,
    ``log(depth) + log(length)``. This is the minimal counts-table entry point
    **and** the composable one: the ``latent`` is built by the shared field core
    (:func:`_sample_field_latent`), so the three field axes combine freely.

    * **Single feature** -- pass ``counts`` of shape ``(n,)``: one patch size
      (``range``), amplitude ``eta``. This is the original single-field model.
    * **Multiple features** -- pass ``counts`` of shape ``(J, n)`` (one row per
      feature, e.g. from :func:`mesh.coregion_counts_arrays`): a linear model of
      coregionalization with ``n_fields`` shared scales and a signed loadings
      matrix (see :func:`coregionalized_negbinomial`).

    Either can be **anisotropic** (``anisotropic=True``, the *direction* axis; see
    :func:`anisotropic_negbinomial`) and at any fixed **smoothness** ``nu`` (the
    *boundary-sharpness* axis; model-compare with :func:`mesh.compare_smoothness`)
    -- combinations that previously needed separate models.

    Parameters
    ----------
    coords : Array
        ``(n, 2)`` coordinates in microns.
    counts : Array
        Observed integer counts, shape ``(n,)`` (single feature) or ``(J, n)``
        (multiple features).
    log_offset : Array
        Per-sample log offset ``log(depth) + log(length)``, shape matching
        ``counts`` (or ``(n,)``, broadcast across features).
    n_fields : int, optional
        Number of shared latent fields (distinct spatial scales). Default ``1``.
        Values ``> 1`` (or multi-feature ``counts``) engage the coregionalization
        core with ordered, extent-bounded ranges.
    anisotropic : bool, optional
        If ``True``, give each field a separate patch size per axis. Default
        ``False``.
    range_prior : tuple of float, optional
        ``(loc, scale)`` of the LogNormal range prior, in log-microns.
    eta_scale : float, optional
        Scale of the ``HalfNormal`` prior on the field standard deviation (single
        field only; with multiple fields the loadings carry the amplitude).
    anisotropy_scale : float, optional
        Scale of the ``Normal(0, .)`` prior on the log anisotropy (used only when
        ``anisotropic``); centred at isotropy.
    loadings_scale : float, optional
        Scale of the ``Normal`` prior on the coregionalization loadings (used
        only in the multi-field/multi-feature regime).
    nu : float, optional
        Matern smoothness (boundary sharpness), one of
        :data:`mesh.kernels.MATERN_NU`. Default ``1.5``.
    jitter : float, optional
        Diagonal jitter for Cholesky stability.
    """
    counts = jnp.asarray(counts)
    single = counts.ndim == 1
    n_features = 1 if single else counts.shape[0]

    concentration = numpyro.sample("concentration", dist.Gamma(2.0, 0.1))
    latent = _sample_field_latent(
        coords,
        n_features=n_features,
        n_fields=n_fields,
        anisotropic=anisotropic,
        nu=nu,
        range_prior=range_prior,
        eta_scale=eta_scale,
        anisotropy_scale=anisotropy_scale,
        loadings_scale=loadings_scale,
        jitter=jitter,
    )

    # Broad intercept prior: the RPKM-style offset pushes the intercept to
    # large-magnitude (negative) log-rates. One per feature.
    if single:
        intercept = numpyro.sample("intercept", dist.Normal(0.0, 25.0))
        mu = jnp.exp(intercept + log_offset + latent[0])
    else:
        intercept = numpyro.sample(
            "intercept", dist.Normal(0.0, 25.0).expand([n_features]).to_event(1)
        )
        mu = jnp.exp(intercept[:, None] + log_offset + latent)
    numpyro.sample(
        "obs",
        dist.NegativeBinomial2(mu, concentration),
        obs=counts,
    )


def anisotropic_negbinomial(
    coords: Array,
    counts: Array,
    log_offset: Array,
    *,
    range_prior: tuple[float, float] = DEFAULT_RANGE_PRIOR,
    eta_scale: float = 1.0,
    anisotropy_scale: float = 1.0,
    nu: float = 1.5,
    jitter: float = 1e-6,
) -> None:
    r"""Directional (anisotropic) spatial abundance model (negative-binomial).

    Same negative-binomial abundance model as :func:`spatial_negbinomial`, but
    the latent field has a **separate patch size along each axis**, so it reads
    out **direction**: does the feature organise along a host axis (proximal--
    distal gut, crypt--villus, depth into a biofilm)? The field is axis-aligned;
    orient the sampling frame to the host axis (a free rotation is a later
    extension).

    The two axis ranges are parameterised by an **overall** patch size and a
    **signed anisotropy**, which are orthogonal and each identified from data:

    .. math::

        \ell_x = \ell\, e^{+\rho/2}, \qquad \ell_y = \ell\, e^{-\rho/2},

    so :math:`\ell = \sqrt{\ell_x \ell_y}` is the geometric-mean range (the same
    ``range`` the isotropic model reports) and :math:`e^{\rho} = \ell_x/\ell_y`
    is the anisotropy. The prior on :math:`\rho` is centred at ``0`` (isotropy),
    so directionality must be supported by the data rather than assumed.

    Parameters
    ----------
    coords : Array
        ``(n, 2)`` coordinates in microns.
    counts : Array
        Observed integer counts, shape ``(n,)``.
    log_offset : Array
        Per-sample log offset ``log(depth) + log(length)``, shape ``(n,)``.
    range_prior : tuple of float, optional
        ``(loc, scale)`` of the LogNormal prior on the **geometric-mean** range,
        in log-microns.
    eta_scale : float, optional
        Scale of the ``HalfNormal`` prior on the field standard deviation.
    anisotropy_scale : float, optional
        Scale of the ``Normal(0, .)`` prior on the log anisotropy
        :math:`\rho = \log(\ell_x/\ell_y)`. The default ``1.0`` is broad (an
        axis ratio of ``e`` at one prior SD) while still shrinking toward
        isotropy.
    nu : float, optional
        Matern smoothness (boundary sharpness) of the field, one of
        :data:`mesh.kernels.MATERN_NU`. Default ``1.5``.
    jitter : float, optional
        Diagonal jitter for Cholesky stability.

    Notes
    -----
    This is a thin preset over :func:`spatial_negbinomial` with
    ``anisotropic=True`` (single feature, single scale). Use
    ``spatial_negbinomial`` directly to combine direction with multiple features
    or scales.
    """
    spatial_negbinomial(
        coords,
        counts,
        log_offset,
        n_fields=1,
        anisotropic=True,
        range_prior=range_prior,
        eta_scale=eta_scale,
        anisotropy_scale=anisotropy_scale,
        nu=nu,
        jitter=jitter,
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

    Notes
    -----
    This is a thin preset over :func:`spatial_negbinomial` with multi-feature
    ``counts`` (shape ``(J, n)``). Pass ``anisotropic=True`` to
    ``spatial_negbinomial`` to make the shared fields directional as well.
    """
    spatial_negbinomial(
        coords,
        counts,
        log_offset,
        n_fields=n_fields,
        anisotropic=False,
        range_prior=range_prior,
        loadings_scale=loadings_scale,
        jitter=jitter,
    )


def _sigmoid(x: Array) -> Array:
    return 1.0 / (1.0 + jnp.exp(-x))
