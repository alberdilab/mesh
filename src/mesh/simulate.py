"""Synthetic data generators with known ground truth.

MESH is simulation-first: every model ships with a generator whose truth is
known, so that a recovery test can confirm the model recovers its own
parameters. Both generators draw a single latent field from a Matern 3/2
Gaussian process with a known **range** (patch size, microns) and emit a
long-format table matching :mod:`mesh.schema`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.special import expit

from .kernels import matern32_kernel

__all__ = ["SimulatedData", "draw_field", "simulate_allele", "simulate_counts"]


@dataclass
class SimulatedData:
    """Container for a simulated dataset and its ground truth.

    Attributes
    ----------
    table : pandas.DataFrame
        Long-format table conforming to :mod:`mesh.schema`.
    truth : dict
        Ground-truth parameters used to generate the data (notably ``range``).
    coords : numpy.ndarray
        ``(n, 2)`` coordinates, in microns.
    field : numpy.ndarray
        The realised latent field at each coordinate.
    """

    table: pd.DataFrame
    truth: dict
    coords: np.ndarray
    field: np.ndarray = field(repr=False)


def draw_field(
    coords: np.ndarray,
    range_: float,
    eta: float,
    rng: np.random.Generator,
    jitter: float = 1e-6,
) -> np.ndarray:
    """Draw a zero-mean Matern 3/2 GP realisation via the Cholesky factor.

    Parameters
    ----------
    coords : numpy.ndarray
        ``(n, 2)`` coordinates in microns.
    range_ : float
        Matern range (patch size) in microns.
    eta : float
        Marginal standard deviation of the field.
    rng : numpy.random.Generator
        Random generator for reproducibility.
    jitter : float, optional
        Diagonal jitter for Cholesky stability.

    Returns
    -------
    numpy.ndarray
        Field values with shape ``(n,)``.
    """
    k = np.asarray(matern32_kernel(coords, range_, variance=1.0, jitter=jitter))
    chol = np.linalg.cholesky(k)
    z = rng.standard_normal(coords.shape[0])
    return eta * (chol @ z)


def _coords(n_samples: int, domain: float, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(0.0, domain, size=(n_samples, 2))


def simulate_allele(
    n_samples: int = 150,
    range_: float = 200.0,
    eta: float = 1.0,
    domain: float = 1000.0,
    intercept: float = 0.0,
    coverage: int = 60,
    precision: float = 50.0,
    length: int = 1000,
    feature_id: str = "feat0",
    seed: int = 0,
) -> SimulatedData:
    """Simulate beta-binomial allele frequencies over a Matern field.

    The latent field perturbs the logit allele frequency; observed alt counts
    are drawn beta-binomially given per-site coverage, making the generator
    coverage-aware (low-coverage sites are noisier). Ported from the ``m0``
    allele-frequency skeleton.

    Parameters
    ----------
    n_samples : int
        Number of spatial microsamples.
    range_ : float
        True Matern range (patch size) in microns -- the quantity to recover.
    eta : float
        Field standard deviation (on the logit scale).
    domain : float
        Side length of the square sampling domain, in microns.
    intercept : float
        Mean logit allele frequency.
    coverage : int
        Nominal per-site coverage; actual coverage varies around it.
    precision : float
        Beta-binomial precision; larger means closer to binomial.
    length : int
        Feature length (bp), carried through to the table.
    feature_id : str
        Identifier for the (single) feature.
    seed : int
        Random seed.

    Returns
    -------
    SimulatedData
        Table (with ``ref``/``alt`` columns), truth dict, coords and field.
    """
    rng = np.random.default_rng(seed)
    coords = _coords(n_samples, domain, rng)
    f = draw_field(coords, range_, eta, rng)

    p = expit(intercept + f)
    total = rng.integers(coverage // 2, coverage * 2 + 1, size=n_samples)
    # Beta-binomial: draw a per-site probability, then a binomial count.
    a = p * precision
    b = (1.0 - p) * precision
    site_p = rng.beta(a, b)
    alt = rng.binomial(total, site_p)
    ref = total - alt

    table = pd.DataFrame(
        {
            "feature_id": feature_id,
            "sample_id": [f"s{i:04d}" for i in range(n_samples)],
            "x": coords[:, 0],
            "y": coords[:, 1],
            "count": alt.astype(np.int64),
            "depth": total.astype(np.int64),
            "length": np.full(n_samples, length, dtype=np.int64),
            "ref": ref.astype(np.int64),
            "alt": alt.astype(np.int64),
        }
    )
    truth = {
        "range": range_,
        "eta": eta,
        "intercept": intercept,
        "precision": precision,
        "domain": domain,
    }
    return SimulatedData(table=table, truth=truth, coords=coords, field=f)


def simulate_counts(
    n_samples: int = 150,
    range_: float = 200.0,
    eta: float = 1.0,
    domain: float = 1000.0,
    baseline_rate: float = 5e-8,
    concentration: float = 10.0,
    length: int = 1000,
    mean_depth: float = 1e6,
    depth_log_sd: float = 0.3,
    feature_id: str = "feat0",
    seed: int = 0,
) -> SimulatedData:
    """Simulate negative-binomial abundance counts over a Matern field.

    Expected counts follow the standard library-size / feature-length offset:

    .. math::

        \\log \\mu_i = \\beta_0 + \\log(\\text{depth}_i)
                     + \\log(\\text{length}) + f_i,

    where :math:`\\beta_0 = \\log(\\text{baseline\\_rate})`, ``depth`` is the
    per-sample sequencing depth (offset) and :math:`f` the Matern field. Counts
    are negative-binomial (NB2: variance ``mu + mu**2 / concentration``).

    Parameters
    ----------
    n_samples : int
        Number of spatial microsamples.
    range_ : float
        True Matern range (patch size) in microns -- the quantity to recover.
    eta : float
        Field standard deviation (on the log-mean scale).
    domain : float
        Side length of the square sampling domain, in microns.
    baseline_rate : float
        Expected count per (read x bp); sets the intercept ``log(baseline_rate)``.
    concentration : float
        NB2 concentration (dispersion); larger means closer to Poisson.
    length : int
        Feature length (bp), entering the offset.
    mean_depth : float
        Geometric-mean per-sample sequencing depth (reads).
    depth_log_sd : float
        Log-normal scatter of per-sample depth.
    feature_id : str
        Identifier for the (single) feature.
    seed : int
        Random seed.

    Returns
    -------
    SimulatedData
        Table, truth dict (incl. ``intercept`` and ``concentration``), coords
        and field.
    """
    rng = np.random.default_rng(seed)
    coords = _coords(n_samples, domain, rng)
    f = draw_field(coords, range_, eta, rng)

    depth = rng.lognormal(mean=np.log(mean_depth), sigma=depth_log_sd, size=n_samples)
    intercept = float(np.log(baseline_rate))
    log_mu = intercept + np.log(depth) + np.log(length) + f
    mu = np.exp(log_mu)

    # NB2 parameterisation via numpy's (r, p): mean = r (1-p) / p with
    # p = concentration / (concentration + mu) yields mean == mu.
    prob = concentration / (concentration + mu)
    counts = rng.negative_binomial(concentration, prob)

    table = pd.DataFrame(
        {
            "feature_id": feature_id,
            "sample_id": [f"s{i:04d}" for i in range(n_samples)],
            "x": coords[:, 0],
            "y": coords[:, 1],
            "count": counts.astype(np.int64),
            "depth": depth.astype(np.float64),
            "length": np.full(n_samples, length, dtype=np.int64),
        }
    )
    truth = {
        "range": range_,
        "eta": eta,
        "intercept": intercept,
        "concentration": concentration,
        "domain": domain,
    }
    return SimulatedData(table=table, truth=truth, coords=coords, field=f)
