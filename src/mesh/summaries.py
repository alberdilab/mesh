"""Posterior summaries of the headline outputs.

The headline output of MESH is the **patch size** (the Matern range, in
microns) with a credible interval. :func:`summarize_range` returns a tidy
one-row table; :func:`summarize_parameters` covers any scalar parameters.

Patch size is only one axis of spatial architecture. The *same* fit also
reports **how strongly** a feature is patterned -- the field amplitude ``eta``,
already covered by :func:`summarize_parameters` -- and **how much** of its
variation is spatially organised rather than unstructured noise, via
:func:`variance_partition`.
"""

from __future__ import annotations

import arviz as az
import numpy as np
import pandas as pd
from scipy.special import expit, polygamma

__all__ = [
    "summarize_range",
    "summarize_parameters",
    "decompose_variance",
    "variance_partition",
]


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


def decompose_variance(idata: az.InferenceData) -> dict[str, np.ndarray]:
    """Split each posterior draw's latent variance into spatial vs. non-spatial.

    A fit answers more than "how big are the patches". It also says **how much
    of a feature's variation is spatially organised** at all, versus
    unstructured noise below the sampling resolution (the model's
    overdispersion / "nugget"). The ratio is a readout of how *deterministic*
    the spatial architecture is, as opposed to stochastic.

    The split is done on the model's **latent scale** (log-mean for the
    negative-binomial, logit allele frequency for the beta-binomial), where
    both pieces are additive variances:

    * **spatial** -- the Matern field variance, ``eta**2``;
    * **non-spatial** -- the variance the observation model adds beyond the
      field. For the negative-binomial this is the gamma-mixture log-variance
      ``trigamma(concentration)``; for the beta-binomial it is the logit
      variance of a ``Beta`` site probability, ``trigamma(a) + trigamma(b)``
      with ``a = p * precision``, ``b = (1 - p) * precision`` and
      ``p = sigmoid(intercept)``.

    These are exact for the respective mixtures, but mix a latent-field variance
    with an observation-level overdispersion, so treat the fraction as a
    well-defined *index* of spatial determinism rather than a partition of a
    single observable quantity. A clean multi-scale variance partition is a
    later milestone (see the roadmap).

    Parameters
    ----------
    idata : arviz.InferenceData
        Posterior from :func:`mesh.fit.fit_model` for either the
        negative-binomial or the beta-binomial model.

    Returns
    -------
    dict of numpy.ndarray
        Flattened per-draw arrays ``{"spatial", "nonspatial", "fraction"}``,
        where ``fraction = spatial / (spatial + nonspatial)`` is the spatially
        organised share of latent variance.

    Raises
    ------
    ValueError
        If the posterior carries neither a ``concentration`` (negative-binomial)
        nor a ``precision`` (beta-binomial) overdispersion parameter.
    """
    post = idata.posterior
    eta = np.asarray(post["eta"].values).reshape(-1)
    spatial = eta**2

    if "concentration" in post:
        concentration = np.asarray(post["concentration"].values).reshape(-1)
        nonspatial = polygamma(1, concentration)
    elif "precision" in post:
        precision = np.asarray(post["precision"].values).reshape(-1)
        intercept = np.asarray(post["intercept"].values).reshape(-1)
        p = expit(intercept)
        nonspatial = polygamma(1, p * precision) + polygamma(1, (1.0 - p) * precision)
    else:
        raise ValueError(
            "No overdispersion parameter found: expected 'concentration' "
            "(negative-binomial) or 'precision' (beta-binomial) in the posterior."
        )

    fraction = spatial / (spatial + nonspatial)
    return {"spatial": spatial, "nonspatial": nonspatial, "fraction": fraction}


def variance_partition(
    idata: az.InferenceData,
    hdi_prob: float = 0.95,
) -> pd.DataFrame:
    """Summarise the spatial vs. non-spatial variance split as a tidy table.

    Wraps :func:`decompose_variance` into one row per quantity with a credible
    interval, mirroring :func:`summarize_parameters`. The headline row is
    ``spatial_fraction``: the posterior share of latent variance that is
    spatially organised (1 = fully deterministic structure, 0 = spatially
    unpatterned noise at this scale).

    Parameters
    ----------
    idata : arviz.InferenceData
        Posterior from :func:`mesh.fit.fit_model`.
    hdi_prob : float
        Mass of the highest-density credible interval (default 0.95).

    Returns
    -------
    pandas.DataFrame
        Rows ``spatial_variance``, ``nonspatial_variance`` and
        ``spatial_fraction`` with ``mean, median, sd, hdi_low, hdi_high,
        hdi_prob``.
    """
    parts = decompose_variance(idata)
    rows = []
    for name in ("spatial_variance", "nonspatial_variance", "spatial_fraction"):
        key = {"spatial_variance": "spatial", "nonspatial_variance": "nonspatial"}.get(
            name, "fraction"
        )
        draws = parts[key]
        hdi_low, hdi_high = az.hdi(draws, prob=hdi_prob)
        rows.append(
            {
                "quantity": name,
                "mean": float(np.mean(draws)),
                "median": float(np.median(draws)),
                "sd": float(np.std(draws)),
                "hdi_low": float(hdi_low),
                "hdi_high": float(hdi_high),
                "hdi_prob": hdi_prob,
            }
        )
    return pd.DataFrame(rows)
