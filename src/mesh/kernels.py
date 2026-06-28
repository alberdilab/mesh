"""Covariance kernels for spatial latent fields.

The package uses **exact** Gaussian processes: a dense covariance matrix is
formed over the sample coordinates and factorised with a Cholesky
decomposition. For the dataset sizes targeted by MESH (a few hundred to ~2k
patches) this is cheap and avoids the approximations (SPDE / inducing points /
variational GPs) that would obscure the headline output -- the **range**, i.e.
the patch size in microns.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

__all__ = ["pairwise_distances", "matern32_kernel", "cholesky_factor"]


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
    d = pairwise_distances(coords)
    sqrt3 = jnp.sqrt(3.0)
    scaled = sqrt3 * d / lengthscale
    k = variance * (1.0 + scaled) * jnp.exp(-scaled)
    n = coords.shape[0]
    return k + jitter * jnp.eye(n)


def cholesky_factor(
    coords: Array,
    lengthscale: Array | float,
    variance: Array | float = 1.0,
    jitter: float = 1e-6,
) -> Array:
    """Lower-triangular Cholesky factor of the Matern 3/2 covariance.

    See :func:`matern32_kernel` for the parameters. Returned ``L`` satisfies
    ``L @ L.T == K`` and is used for the non-centred GP draw ``f = L @ z``.
    """
    k = matern32_kernel(coords, lengthscale, variance=variance, jitter=jitter)
    return jnp.linalg.cholesky(k)
