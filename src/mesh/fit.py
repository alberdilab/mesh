"""Thin inference runner: build a model, run NUTS, return ArviZ output.

Also provides helpers that turn a validated single-feature table (see
:mod:`mesh.schema`) into the array inputs each model expects.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import arviz as az
import jax
import numpy as np
import numpyro
import pandas as pd
from numpyro.infer import MCMC, NUTS

from . import schema as _schema

__all__ = [
    "fit_model",
    "enable_parallel_chains",
    "get_range_posterior",
    "counts_arrays",
    "allele_arrays",
    "coregion_counts_arrays",
    "coregion_feature_order",
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
            "JAX backend already initialised; enable_parallel_chains(%d) has no "
            "effect. Call it earlier -- after `import mesh` but before the first "
            "fit_model() (or any other JAX operation)." % n,
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
        "chain_method='auto' requested %d chains but only %d JAX device(s) are "
        "available; running them on one device with chain_method='vectorized'. "
        "For true parallelism call mesh.enable_parallel_chains(%d) once, before "
        "the first fit_model()." % (num_chains, n_devices, num_chains),
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
    return az.from_numpyro(mcmc, dims={}, pred_dims={})


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
