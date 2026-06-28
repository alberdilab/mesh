"""Thin inference runner: build a model, run NUTS, return ArviZ output.

Also provides helpers that turn a validated single-feature table (see
:mod:`mesh.schema`) into the array inputs each model expects.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import arviz as az
import jax
import numpy as np
import pandas as pd
from numpyro.infer import MCMC, NUTS

from . import schema as _schema

__all__ = [
    "fit_model",
    "get_range_posterior",
    "counts_arrays",
    "allele_arrays",
]


def fit_model(
    model: Callable[..., Any],
    *,
    num_warmup: int = 500,
    num_samples: int = 500,
    num_chains: int = 1,
    chain_method: str = "vectorized",
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
        How to run multiple chains: ``"vectorized"`` (default; vmap over chains,
        single device), ``"sequential"`` or ``"parallel"``.
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
