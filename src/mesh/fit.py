"""Thin inference runner: build a model, run NUTS, return ArviZ output.

Also provides helpers that turn a validated single-feature table (see
:mod:`mesh.schema`) into the array inputs each model expects.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Sequence
from typing import Any

import arviz as az
import jax
import numpy as np
import numpyro
import pandas as pd
from numpyro.infer import MCMC, NUTS

from . import schema as _schema
from .kernels import MATERN_NU

__all__ = [
    "fit_model",
    "enable_parallel_chains",
    "get_range_posterior",
    "counts_arrays",
    "allele_arrays",
    "coregion_counts_arrays",
    "coregion_feature_order",
    "hierarchical_counts_arrays",
    "hierarchical_genome_order",
    "nu_label",
    "compare_smoothness",
]


def _jax_initialized() -> bool:
    """Return whether JAX has already initialised its backend.

    Once the backend is live the host device count is locked, so
    :func:`enable_parallel_chains` can no longer change it.
    """
    from jax._src import xla_bridge

    return bool(getattr(xla_bridge, "_backends", {}))


def enable_parallel_chains(n: int) -> int:
    """Expose ``n`` host (CPU) devices so chains can run truly in parallel.

    JAX runs ``chain_method="parallel"`` across *devices*; on CPU it sees a
    single device unless told otherwise. This wraps
    :func:`numpyro.set_host_device_count`, which sets an XLA flag that only
    takes effect **before the JAX backend initialises**. Call it once, early --
    after ``import mesh`` is fine, but before your first :func:`fit_model`.

    Parameters
    ----------
    n : int
        Number of host devices (CPU cores) to expose.

    Returns
    -------
    int
        ``jax.local_device_count()`` after the attempt -- the number of devices
        actually available. If this is below ``n`` the call came too late.

    Warns
    -----
    UserWarning
        If the JAX backend is already initialised, in which case the device
        count cannot be raised and the request has no effect.
    """
    if _jax_initialized():
        warnings.warn(
            f"JAX backend already initialised; enable_parallel_chains({n}) has no "
            "effect. Call it earlier -- after `import mesh` but before the first "
            "fit_model() (or any other JAX operation).",
            UserWarning,
            stacklevel=2,
        )
        return int(jax.local_device_count())
    numpyro.set_host_device_count(n)
    return int(jax.local_device_count())


def _resolve_chain_method(chain_method: str, num_chains: int) -> str:
    """Turn ``chain_method="auto"`` into a concrete method for the runtime.

    ``"auto"`` runs chains in parallel across devices when enough are available,
    otherwise falls back to ``"vectorized"`` (faster than ``"sequential"`` on a
    single device) and warns how to unlock real parallelism. Any explicit
    method is passed through unchanged.
    """
    if chain_method != "auto":
        return chain_method
    if num_chains <= 1:
        return "vectorized"
    n_devices = int(jax.local_device_count())
    if n_devices >= num_chains:
        return "parallel"
    warnings.warn(
        f"chain_method='auto' requested {num_chains} chains but only {n_devices} "
        "JAX device(s) are available; running them on one device with "
        f"chain_method='vectorized'. For true parallelism call "
        f"mesh.enable_parallel_chains({num_chains}) once, before the first "
        "fit_model().",
        UserWarning,
        stacklevel=3,
    )
    return "vectorized"


def fit_model(
    model: Callable[..., Any],
    *,
    num_warmup: int = 500,
    num_samples: int = 500,
    num_chains: int = 1,
    chain_method: str = "auto",
    seed: int = 0,
    target_accept_prob: float = 0.9,
    progress_bar: bool = False,
    log_likelihood: bool = False,
    **model_kwargs: Any,
) -> az.InferenceData:
    """Run NUTS on a MESH model and return an ArviZ ``InferenceData``.

    Parameters
    ----------
    model : callable
        A NumPyro model from :mod:`mesh.model`.
    num_warmup, num_samples, num_chains : int
        NUTS sampling configuration.
    chain_method : str
        How to run multiple chains: ``"auto"`` (default), ``"vectorized"``
        (vmap over chains, single device), ``"sequential"`` or ``"parallel"``.
        ``"auto"`` uses ``"parallel"`` when at least ``num_chains`` JAX devices
        are available and otherwise falls back to ``"vectorized"`` with a
        warning. ``"parallel"`` needs one device per chain; on CPU, call
        :func:`enable_parallel_chains` (or ``numpyro.set_host_device_count(n)``)
        *before* the first fit, otherwise there is only one device to use.
    seed : int
        PRNG seed.
    target_accept_prob : float
        NUTS target acceptance probability; higher reduces divergences in
        awkward GP geometries.
    progress_bar : bool
        Whether to display the sampling progress bar.
    log_likelihood : bool
        Whether to store the pointwise ``log_likelihood`` group. Needed for
        LOO/WAIC model comparison (e.g. :func:`compare_smoothness`); off by
        default to keep fits lean.
    **model_kwargs
        Passed through to ``model`` (e.g. ``coords``, ``counts``, ``log_offset``).

    Returns
    -------
    arviz.InferenceData
        Posterior draws and sample statistics.
    """
    chain_method = _resolve_chain_method(chain_method, num_chains)
    kernel = NUTS(model, target_accept_prob=target_accept_prob)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method=chain_method,
        progress_bar=progress_bar,
    )
    mcmc.run(jax.random.PRNGKey(seed), **model_kwargs)
    # Pass empty dims/pred_dims to skip arviz's automatic dimension inference,
    # which re-traces the model and is brittle across numpyro/arviz versions
    # (and unnecessary here -- all our latent sites are 1D vectors).
    idata = az.from_numpyro(
        mcmc, dims={}, pred_dims={}, log_likelihood=log_likelihood
    )
    if log_likelihood:
        # az.from_numpyro keeps the arrays JAX-backed; arviz-stats' LOO does
        # in-place writes that fail on immutable JAX arrays, so coerce to NumPy.
        idata = idata.map_over_datasets(lambda ds: ds.map(np.asarray))
    return idata


def get_range_posterior(idata: az.InferenceData, var_name: str = "range") -> np.ndarray:
    """Return the flattened posterior draws of the range parameter."""
    return np.asarray(idata.posterior[var_name].values).reshape(-1)


def _single_feature(df: pd.DataFrame) -> pd.DataFrame:
    features = df["feature_id"].unique()
    if len(features) != 1:
        raise ValueError(
            "The M0/M1 models fit a single feature; the table holds "
            f"{len(features)} features. Subset to one feature_id first."
        )
    return df.sort_values("sample_id").reset_index(drop=True)


def counts_arrays(df: pd.DataFrame, *, validate: bool = True) -> dict[str, np.ndarray]:
    """Extract ``coords``, ``counts`` and ``log_offset`` for the NB model.

    Parameters
    ----------
    df : pandas.DataFrame
        Validated single-feature table.
    validate : bool
        Whether to run :func:`mesh.schema.validate_table` first.

    Returns
    -------
    dict
        Keys ``coords`` (``(n, 2)``), ``counts`` (``(n,)``) and ``log_offset``
        (``(n,)`` = ``log(depth) + log(length)``).
    """
    if validate:
        _schema.validate_table(df)
    sub = _single_feature(df)
    coords = sub[["x", "y"]].to_numpy(dtype=np.float64)
    counts = sub["count"].to_numpy(dtype=np.int64)
    log_offset = np.log(sub["depth"].to_numpy(dtype=np.float64)) + np.log(
        sub["length"].to_numpy(dtype=np.float64)
    )
    return {"coords": coords, "counts": counts, "log_offset": log_offset}


def allele_arrays(df: pd.DataFrame, *, validate: bool = True) -> dict[str, np.ndarray]:
    """Extract ``coords``, ``alt_count`` and ``total_count`` for the BB model.

    Parameters
    ----------
    df : pandas.DataFrame
        Validated single-feature table with ``ref``/``alt`` columns.
    validate : bool
        Whether to run :func:`mesh.schema.validate_table` (allele mode) first.

    Returns
    -------
    dict
        Keys ``coords`` (``(n, 2)``), ``alt_count`` and ``total_count``.
    """
    if validate:
        _schema.validate_table(df, require_allele=True)
    sub = _single_feature(df)
    coords = sub[["x", "y"]].to_numpy(dtype=np.float64)
    alt_count = sub["alt"].to_numpy(dtype=np.int64)
    total_count = sub["depth"].to_numpy(dtype=np.int64)
    return {"coords": coords, "alt_count": alt_count, "total_count": total_count}


def coregion_counts_arrays(
    df: pd.DataFrame, *, validate: bool = True
) -> dict[str, np.ndarray]:
    """Build ``coords``, ``counts`` and ``log_offset`` for the coregion model.

    Unlike :func:`counts_arrays` (single feature, ``(n,)`` vectors), this packs
    a **multi-feature** table into matrices for
    :func:`mesh.coregionalized_negbinomial`. Features are taken in sorted
    ``feature_id`` order and samples in sorted ``sample_id`` order; rows of the
    returned ``counts``/``log_offset`` correspond to that feature order.

    Parameters
    ----------
    df : pandas.DataFrame
        Validated multi-feature table (shared catalog across samples).
    validate : bool
        Whether to run :func:`mesh.schema.validate_table` first.

    Returns
    -------
    dict
        Keys ``coords`` (``(n, 2)``), ``counts`` (``(J, n)``) and ``log_offset``
        (``(J, n)`` = ``log(depth) + log(length)``), with ``J`` features ordered
        by sorted ``feature_id``. (See :attr:`feature order
        <mesh.fit.coregion_feature_order>` to recover the labels.)
    """
    if validate:
        _schema.validate_table(df)
    features = sorted(df["feature_id"].unique())
    samples = sorted(df["sample_id"].unique())

    def _matrix(value: str, dtype: type) -> np.ndarray:
        wide = df.pivot(index="feature_id", columns="sample_id", values=value)
        return wide.loc[features, samples].to_numpy(dtype=dtype)

    counts = _matrix("count", np.int64)
    depth = _matrix("depth", np.float64)
    length = _matrix("length", np.float64)
    log_offset = np.log(depth) + np.log(length)

    # Coordinates are shared across features; read them from any single feature.
    coord_rows = (
        df.drop_duplicates("sample_id").set_index("sample_id").loc[samples, ["x", "y"]]
    )
    coords = coord_rows.to_numpy(dtype=np.float64)
    return {"coords": coords, "counts": counts, "log_offset": log_offset}


def coregion_feature_order(df: pd.DataFrame) -> list[str]:
    """Return the feature_id order used by :func:`coregion_counts_arrays`.

    The coregion model's loadings rows follow this order, so use it to map a
    posterior loading row back to its feature.
    """
    return sorted(df["feature_id"].unique())


def hierarchical_genome_order(genome: pd.DataFrame) -> list[str]:
    """Return the genome_id order used by :func:`hierarchical_counts_arrays`.

    The hierarchical model's per-genome ``range``/``eta`` rows follow this order,
    so use it to map a posterior genome index back to its ``genome_id``.
    """
    return sorted(genome["genome_id"].unique())


def hierarchical_counts_arrays(
    df: pd.DataFrame, genome: pd.DataFrame, *, validate: bool = True
) -> dict[str, np.ndarray]:
    """Build arrays for :func:`mesh.hierarchical_coregionalized_negbinomial`.

    Extends :func:`coregion_counts_arrays` with the ``feature -> genome`` (1:1)
    membership: a ``genome_index`` aligned to the sorted-``feature_id`` rows of
    ``counts``, plus ``n_genomes``. Genome codes follow sorted ``genome_id``
    order (see :func:`hierarchical_genome_order`), so ``range[k]``/``eta[k]`` in
    the posterior correspond to the ``k``-th genome in that order.

    Parameters
    ----------
    df : pandas.DataFrame
        Validated multi-feature counts table (shared catalog across samples).
    genome : pandas.DataFrame
        ``feature_id, genome_id`` annotation covering every catalog feature.
    validate : bool
        Whether to run :func:`mesh.validate_annotations` (which also validates
        ``df``) first.

    Returns
    -------
    dict
        ``coords`` ``(n, 2)``, ``counts`` ``(J, n)``, ``log_offset`` ``(J, n)``,
        ``genome_index`` ``(J,)`` and ``n_genomes`` (int), ready to splat into
        the model.
    """
    if validate:
        _schema.validate_annotations(df, genome=genome)
    arrays = coregion_counts_arrays(df, validate=False)
    features = coregion_feature_order(df)
    genome_ids = hierarchical_genome_order(genome)
    code = {g: i for i, g in enumerate(genome_ids)}
    feature_genome = genome.set_index("feature_id")["genome_id"]
    arrays["genome_index"] = np.array(
        [code[feature_genome[f]] for f in features], dtype=np.int64
    )
    arrays["n_genomes"] = len(genome_ids)
    return arrays


def nu_label(nu: float) -> str:
    """Return a short, stable label for a Matern smoothness ``nu``.

    Used as the model key in :func:`compare_smoothness`. ``0.5`` -> ``"matern12"``,
    ``1.5`` -> ``"matern32"``, ``2.5`` -> ``"matern52"`` and ``math.inf`` ->
    ``"se"`` (squared-exponential).
    """
    if nu == math.inf:
        return "se"
    if nu == 0.5:
        return "matern12"
    if nu == 1.5:
        return "matern32"
    if nu == 2.5:
        return "matern52"
    raise ValueError(
        f"nu={nu!r} has no closed-form kernel; choose one of {MATERN_NU}."
    )


def compare_smoothness(
    model: Callable[..., Any],
    *,
    nu_values: Sequence[float] = MATERN_NU,
    num_warmup: int = 500,
    num_samples: int = 500,
    num_chains: int = 2,
    seed: int = 0,
    target_accept_prob: float = 0.95,
    progress_bar: bool = False,
    **model_kwargs: Any,
) -> tuple[pd.DataFrame, dict[str, az.InferenceData]]:
    r"""Compare Matern smoothnesses (boundary sharpness) by LOO cross-validation.

    Fits the **same data** under each fixed smoothness ``nu`` in ``nu_values``
    and ranks the fits with leave-one-out cross-validation (:func:`arviz.compare`,
    PSIS-LOO). This is the *boundary sharpness* readout: the winning ``nu`` says
    whether the patches have crisp edges (small ``nu`` -- competitive exclusion or
    a physical/biofilm barrier) or fade as gradients (large ``nu`` -- a
    diffusion-limited gradient of O2, pH or nutrients).

    Each fit fixes its own ``nu``, so the latent field stays unambiguous (no
    discrete latent marginalised inside a single chain). ``model`` must accept a
    ``nu`` keyword (the single-field models :func:`mesh.spatial_negbinomial` and
    :func:`mesh.spatial_betabinomial` do).

    Parameters
    ----------
    model : callable
        A single-field NumPyro model taking a ``nu`` keyword.
    nu_values : sequence of float, optional
        Smoothnesses to compare; defaults to the full closed-form family
        :data:`mesh.kernels.MATERN_NU` (``0.5, 1.5, 2.5, inf``).
    num_warmup, num_samples, num_chains : int
        NUTS configuration for every fit. ``num_chains`` defaults to 2 so the
        comparison comes with convergence diagnostics.
    seed : int
        Base PRNG seed (shared across fits, so they differ only by ``nu``).
    target_accept_prob : float
        NUTS target acceptance; defaults to ``0.95`` (the GP geometry, and the
        ill-conditioned very-smooth kernels, want a careful sampler).
    progress_bar : bool
        Whether to show each fit's progress bar.
    **model_kwargs
        Passed through to ``model`` (e.g. ``coords``, ``counts``,
        ``log_offset``). A ``nu`` here is overridden per fit.

    Returns
    -------
    comparison : pandas.DataFrame
        :func:`arviz.compare` table, best model first (``rank == 0``), indexed by
        :func:`nu_label`, with an added ``nu`` column. The index of the top row
        is the preferred smoothness.
    idatas : dict of str to arviz.InferenceData
        The per-``nu`` fits (keyed by :func:`nu_label`), each carrying its
        ``log_likelihood`` group.
    """
    model_kwargs.pop("nu", None)
    idatas: dict[str, az.InferenceData] = {}
    labels: dict[str, float] = {}
    for nu in nu_values:
        label = nu_label(nu)
        labels[label] = nu
        idatas[label] = fit_model(
            model,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            seed=seed,
            target_accept_prob=target_accept_prob,
            progress_bar=progress_bar,
            log_likelihood=True,
            nu=nu,
            **model_kwargs,
        )
    comparison = az.compare(idatas)
    comparison["nu"] = [labels[name] for name in comparison.index]
    return comparison, idatas
