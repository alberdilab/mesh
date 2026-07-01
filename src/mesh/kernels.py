"""Covariance kernels for spatial latent fields.

The package uses **exact** Gaussian processes: a dense covariance matrix is
formed over the sample coordinates and factorised with a Cholesky
decomposition. For the dataset sizes targeted by MESH (a few hundred to ~2k
patches) this is cheap and avoids the approximations (SPDE / inducing points /
variational GPs) that would obscure the headline output -- the **range**, i.e.
the patch size in microns.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
from jax import Array

__all__ = [
    "pairwise_distances",
    "anisotropic_scaled_distances",
    "matern_kernel",
    "matern32_kernel",
    "anisotropic_matern_kernel",
    "cholesky_factor",
    "MATERN_NU",
]

# The smoothness values with a closed-form covariance (no fractional-order
# modified Bessel function ``K_nu``, which has no autodiff-stable JAX
# implementation). ``math.inf`` is the squared-exponential limit ``nu -> inf``.
# Smaller ``nu`` = rougher field = crisper patch edges; larger ``nu`` = smoother
# field = patches that fade as gradients. This is the **boundary sharpness** axis.
MATERN_NU: tuple[float, ...] = (0.5, 1.5, 2.5, math.inf)


def pairwise_distances(coords: Array) -> Array:
    """Euclidean distances between all pairs of coordinates.

    Parameters
    ----------
    coords : Array
        Coordinates with shape ``(n, d)`` (``d == 2`` for the 2D models).

    Returns
    -------
    Array
        Symmetric ``(n, n)`` matrix of Euclidean distances.
    """
    diff = coords[:, None, :] - coords[None, :, :]
    sq = jnp.sum(diff**2, axis=-1)
    # Clip tiny negative values from round-off before the square root.
    return jnp.sqrt(jnp.clip(sq, min=0.0))


def anisotropic_scaled_distances(coords: Array, lengthscales: Array) -> Array:
    r"""Per-axis-scaled (Mahalanobis) distances for an axis-aligned kernel.

    Divides each coordinate axis by its own lengthscale *before* measuring
    distance, so the returned matrix is the dimensionless separation

    .. math::

        u_{ij} = \sqrt{\sum_a \left(
            \frac{s_{ia} - s_{ja}}{\ell_a} \right)^2 }.

    Feeding :math:`u` to a Matern shape (with unit range) yields an
    **axis-aligned anisotropic** field: the patch size is :math:`\ell_a` along
    axis :math:`a`. With all lengthscales equal this reduces to
    ``pairwise_distances(coords) / lengthscale`` -- the isotropic case.

    Parameters
    ----------
    coords : Array
        Coordinates with shape ``(n, d)``.
    lengthscales : Array
        Per-axis ranges, shape ``(d,)`` (e.g. ``[ell_x, ell_y]`` in microns).

    Returns
    -------
    Array
        Symmetric ``(n, n)`` matrix of scaled distances :math:`u`.
    """
    scaled = coords / lengthscales  # (n, d), each axis in its own range units
    diff = scaled[:, None, :] - scaled[None, :, :]
    sq = jnp.sum(diff**2, axis=-1)
    return jnp.sqrt(jnp.clip(sq, min=0.0))


def _matern_correlation(u: Array, nu: float) -> Array:
    r"""Matern correlation as a function of the dimensionless lag ``u = r/ell``.

    Shared radial shape for the isotropic (:func:`matern_kernel`) and
    anisotropic (:func:`anisotropic_matern_kernel`) kernels, so both agree on
    what a given smoothness ``nu`` means. Only the closed-form members of the
    family are supported (no fractional-order Bessel ``K_nu``).

    Raises
    ------
    ValueError
        If ``nu`` is not one of the closed-form values in :data:`MATERN_NU`.
    """
    if nu == math.inf:
        return jnp.exp(-0.5 * u**2)
    if nu == 0.5:
        return jnp.exp(-u)
    if nu == 1.5:
        s = jnp.sqrt(3.0) * u
        return (1.0 + s) * jnp.exp(-s)
    if nu == 2.5:
        s = jnp.sqrt(5.0) * u
        return (1.0 + s + s**2 / 3.0) * jnp.exp(-s)
    raise ValueError(
        f"nu={nu!r} has no closed-form Matern kernel; choose one of "
        f"{MATERN_NU} (0.5, 1.5, 2.5, or math.inf for squared-exponential)."
    )


def matern_kernel(
    coords: Array,
    lengthscale: Array | float,
    nu: float = 1.5,
    variance: Array | float = 1.0,
    jitter: float = 1e-6,
) -> Array:
    r"""Matern covariance over 2D coordinates at a chosen smoothness ``nu``.

    The smoothness :math:`\nu` is the **boundary sharpness** parameter: it sets
    how rough the field is, i.e. whether patches have crisp edges (small
    :math:`\nu`) or fade as gradients (large :math:`\nu`). Only the closed-form
    members of the Matern family are supported, so the kernel never needs the
    fractional-order modified Bessel function :math:`K_\nu` (which has no
    autodiff-stable JAX implementation):

    .. math::

        \nu = 1/2:\quad & k(r) = \sigma^2 \exp(-r/\ell), \\
        \nu = 3/2:\quad & k(r) = \sigma^2 (1 + s)\,e^{-s},
                          \; s = \sqrt{3}\,r/\ell, \\
        \nu = 5/2:\quad & k(r) = \sigma^2 (1 + s + s^2/3)\,e^{-s},
                          \; s = \sqrt{5}\,r/\ell, \\
        \nu \to \infty:\quad & k(r) = \sigma^2 \exp(-r^2 / 2\ell^2)
                          \quad\text{(squared exponential).}

    The :math:`\sqrt{2\nu}` scaling keeps :math:`\ell` interpretable as the same
    **range** (patch size in microns) across every member, so changing ``nu``
    re-shapes the boundaries without changing what the range *means*.

    Parameters
    ----------
    coords : Array
        Coordinates with shape ``(n, d)``.
    lengthscale : Array or float
        Range parameter :math:`\ell` in the same units as ``coords`` (microns).
    nu : float, optional
        Matern smoothness, one of :data:`MATERN_NU` -- ``0.5`` (exponential,
        roughest/crispest), ``1.5`` (default), ``2.5``, or ``math.inf``
        (squared-exponential, smoothest). Larger means smoother fields.
    variance : Array or float, optional
        Marginal variance :math:`\sigma^2`. Defaults to ``1.0`` so the kernel
        returns a correlation matrix that can be scaled outside (used by the
        non-centred parameterisation).
    jitter : float, optional
        Small value added to the diagonal for numerical stability of the
        Cholesky factorisation.

    Returns
    -------
    Array
        Symmetric positive-definite ``(n, n)`` covariance matrix.

    Raises
    ------
    ValueError
        If ``nu`` is not one of the closed-form values in :data:`MATERN_NU`.
    """
    d = pairwise_distances(coords)
    corr = _matern_correlation(d / lengthscale, nu)
    k = variance * corr
    n = coords.shape[0]
    return k + jitter * jnp.eye(n)


def anisotropic_matern_kernel(
    coords: Array,
    lengthscales: Array,
    nu: float = 1.5,
    variance: Array | float = 1.0,
    jitter: float = 1e-6,
) -> Array:
    r"""Axis-aligned **anisotropic** Matern covariance over 2D coordinates.

    Same Matern family as :func:`matern_kernel`, but with a **separate range per
    coordinate axis**: the patch size is :math:`\ell_x` along ``x`` and
    :math:`\ell_y` along ``y``. This is the **direction** axis of spatial
    architecture -- it answers whether a feature organises along a host axis
    (proximal--distal gut, crypt--villus, depth into a biofilm). The covariance
    is :math:`\sigma^2\, \rho_\nu(u_{ij})` with the per-axis-scaled lag
    :math:`u` from :func:`anisotropic_scaled_distances`; with equal lengthscales
    it is identical to the isotropic :func:`matern_kernel`.

    The axes are **not** rotated: elongation is assumed aligned with ``x``/``y``
    (orient the sampling frame to the host axis). A free rotation is a later
    extension.

    Parameters
    ----------
    coords : Array
        Coordinates with shape ``(n, d)``.
    lengthscales : Array
        Per-axis ranges :math:`(\ell_x, \ell_y)` in the same units as ``coords``
        (microns), shape ``(d,)``.
    nu : float, optional
        Matern smoothness, one of :data:`MATERN_NU`. Default ``1.5``.
    variance : Array or float, optional
        Marginal variance :math:`\sigma^2`. Defaults to ``1.0`` (correlation
        matrix), for the non-centred parameterisation.
    jitter : float, optional
        Small value added to the diagonal for Cholesky stability.

    Returns
    -------
    Array
        Symmetric positive-definite ``(n, n)`` covariance matrix.

    Raises
    ------
    ValueError
        If ``nu`` is not one of the closed-form values in :data:`MATERN_NU`.
    """
    u = anisotropic_scaled_distances(coords, lengthscales)
    k = variance * _matern_correlation(u, nu)
    n = coords.shape[0]
    return k + jitter * jnp.eye(n)


def matern32_kernel(
    coords: Array,
    lengthscale: Array | float,
    variance: Array | float = 1.0,
    jitter: float = 1e-6,
) -> Array:
    r"""Matern 3/2 covariance over 2D coordinates.

    The Matern 3/2 covariance is

    .. math::

        k(r) = \sigma^2 \left(1 + \frac{\sqrt{3}\, r}{\ell}\right)
               \exp\!\left(-\frac{\sqrt{3}\, r}{\ell}\right),

    where :math:`r` is the Euclidean distance, :math:`\sigma^2` the marginal
    variance and :math:`\ell` the **range** (lengthscale) -- the parameter of
    interest, interpreted as the patch size in microns.

    Parameters
    ----------
    coords : Array
        Coordinates with shape ``(n, d)``.
    lengthscale : Array or float
        Range parameter :math:`\ell` in the same units as ``coords`` (microns).
    variance : Array or float, optional
        Marginal variance :math:`\sigma^2`. Defaults to ``1.0`` so the kernel
        returns a correlation matrix that can be scaled outside (used by the
        non-centred parameterisation).
    jitter : float, optional
        Small value added to the diagonal for numerical stability of the
        Cholesky factorisation.

    Returns
    -------
    Array
        Symmetric positive-definite ``(n, n)`` covariance matrix.
    """
    return matern_kernel(coords, lengthscale, nu=1.5, variance=variance, jitter=jitter)


def cholesky_factor(
    coords: Array,
    lengthscale: Array | float,
    nu: float = 1.5,
    variance: Array | float = 1.0,
    jitter: float = 1e-6,
) -> Array:
    """Lower-triangular Cholesky factor of the Matern covariance.

    See :func:`matern_kernel` for the parameters (including the smoothness
    ``nu``). Returned ``L`` satisfies ``L @ L.T == K`` and is used for the
    non-centred GP draw ``f = L @ z``.
    """
    k = matern_kernel(coords, lengthscale, nu=nu, variance=variance, jitter=jitter)
    return jnp.linalg.cholesky(k)
