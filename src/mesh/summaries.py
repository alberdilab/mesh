"""Posterior summaries of the headline outputs.

The headline output of MESH is the **patch size** (the Matern range, in
microns) with a credible interval. :func:`summarize_range` returns a tidy
one-row table; :func:`summarize_parameters` covers any scalar parameters.
"""

from __future__ import annotations

import arviz as az
import numpy as np
import pandas as pd

__all__ = ["summarize_range", "summarize_parameters"]


def summarize_range(
    idata: az.InferenceData,
    var_name: str = "range",
    hdi_prob: float = 0.95,
) -> pd.DataFrame:
    """Summarise the range (patch size) posterior.

    Parameters
    ----------
    idata : arviz.InferenceData
        Posterior from :func:`mesh.fit.fit_model`.
    var_name : str
        Name of the range variable.
    hdi_prob : float
        Mass of the highest-density credible interval (default 0.95).

    Returns
    -------
    pandas.DataFrame
        One row with ``parameter, mean, median, sd, hdi_low, hdi_high,
        hdi_prob, r_hat, ess_bulk``.
    """
    return summarize_parameters(idata, var_names=[var_name], hdi_prob=hdi_prob)


def summarize_parameters(
    idata: az.InferenceData,
    var_names: list[str] | None = None,
    hdi_prob: float = 0.95,
) -> pd.DataFrame:
    """Summarise scalar posterior parameters with credible intervals.

    Parameters
    ----------
    idata : arviz.InferenceData
        Posterior from :func:`mesh.fit.fit_model`.
    var_names : list of str, optional
        Parameters to summarise. Defaults to ``["range", "eta"]`` where present.
    hdi_prob : float
        Mass of the highest-density credible interval.

    Returns
    -------
    pandas.DataFrame
        One row per parameter.
    """
    if var_names is None:
        var_names = [v for v in ("range", "eta") if v in idata.posterior]

    rows = []
    hdi = az.hdi(idata, var_names=var_names, prob=hdi_prob)
    ess = az.ess(idata, var_names=var_names)
    rhat = az.rhat(idata, var_names=var_names)
    for name in var_names:
        draws = np.asarray(idata.posterior[name].values).reshape(-1)
        bounds = np.asarray(hdi[name].values).reshape(-1)
        rows.append(
            {
                "parameter": name,
                "mean": float(np.mean(draws)),
                "median": float(np.median(draws)),
                "sd": float(np.std(draws)),
                "hdi_low": float(bounds[0]),
                "hdi_high": float(bounds[1]),
                "hdi_prob": hdi_prob,
                "r_hat": float(rhat[name].values),
                "ess_bulk": float(ess[name].values),
            }
        )
    return pd.DataFrame(rows)
