"""Plotting helpers for the most relevant MESH inputs and outputs.

MESH is inference-only, but a few standard views make a fit interpretable at a
glance. This module collects light matplotlib helpers for:

* the **input** -- the spatial sampling design and the observed signal
  (:func:`plot_samples`);
* the headline **output** -- the patch-size (``range``) posterior with a
  credible interval (:func:`plot_range_posterior`);
* the inferred (or true) **latent field** as a spatial map
  (:func:`plot_field`);
* what the range *means* -- the Matern 3/2 correlation decay implied by the
  range posterior (:func:`plot_matern_correlation`).

Three further views go **beyond patch size**, each a different axis of spatial
architecture from the *same* fit:

* **how strongly** a feature is patterned -- the field-amplitude posterior
  (:func:`plot_amplitude_posterior`);
* **how much** of its variation is spatially organised vs. unstructured noise
  (:func:`plot_variance_partition`);
* **whether two features share a scale** -- e.g. a function's patch vs. a
  genotype's territory (:func:`plot_scale_comparison`).

For the **coregionalization** model (several features on shared fields), one more
view shows **which feature loads on which field** -- the feature-to-field
loading matrix (:func:`plot_loadings`).

Every function takes an optional ``ax`` and returns the :class:`~matplotlib.axes.Axes`
it drew on, so plots compose into multi-panel figures. ``matplotlib`` is pulled
in transitively by ``arviz``; import this module only when you want to plot.
"""

from __future__ import annotations

from collections.abc import Sequence

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from scipy.stats import gaussian_kde

from .summaries import decompose_variance

__all__ = [
    "plot_samples",
    "plot_range_posterior",
    "plot_amplitude_posterior",
    "plot_variance_partition",
    "plot_scale_comparison",
    "plot_loadings",
    "plot_field",
    "plot_matern_correlation",
    "posterior_field_mean",
]


def plot_samples(
    table: pd.DataFrame,
    *,
    value: str = "count",
    as_frequency: bool = False,
    ax: Axes | None = None,
    cmap: str = "viridis",
    s: float = 40.0,
    colorbar: bool = True,
    title: str | None = None,
) -> Axes:
    """Map the input table: sample positions coloured by the observed signal.

    This is the primary view of the **input** -- it shows the sampling design
    (where microsamples sit, in microns) and the spatial pattern in the observed
    signal that the model will try to explain with a latent field.

    Parameters
    ----------
    table : pandas.DataFrame
        Long-format table conforming to :mod:`mesh.schema` (needs ``x``, ``y``
        and ``value``; ``depth`` too when ``as_frequency`` is set).
    value : str
        Column to colour by (default ``"count"``).
    as_frequency : bool
        If ``True``, colour by ``value / depth`` -- e.g. the allele frequency
        (``alt / coverage``) for the beta-binomial model -- rather than the raw
        count.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new one is created if omitted.
    cmap : str
        Matplotlib colormap name.
    s : float
        Marker size.
    colorbar : bool
        Whether to draw a colorbar.
    title : str, optional
        Axes title.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn on.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5.0, 4.2))

    values = table[value].to_numpy(dtype=float)
    label = value
    if as_frequency:
        values = values / table["depth"].to_numpy(dtype=float)
        label = f"{value} / depth"

    sc = ax.scatter(
        table["x"], table["y"], c=values, cmap=cmap, s=s, edgecolors="none"
    )
    ax.set_xlabel("x (microns)")
    ax.set_ylabel("y (microns)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title if title is not None else "Observed samples")
    if colorbar:
        ax.figure.colorbar(sc, ax=ax, label=label, fraction=0.046, pad=0.04)
    return ax


def plot_range_posterior(
    idata: az.InferenceData,
    *,
    var_name: str = "range",
    truth: float | None = None,
    hdi_prob: float = 0.95,
    ax: Axes | None = None,
    color: str = "C0",
    title: str | None = None,
) -> Axes:
    """Plot the patch-size (range) posterior with its credible interval.

    This is the headline **output**: the spatial scale, in microns, at which the
    feature segregates. The highest-density interval is shaded and the posterior
    mean drawn; pass ``truth`` to overlay a known value (e.g. from a simulation).

    Parameters
    ----------
    idata : arviz.InferenceData
        Posterior from :func:`mesh.fit.fit_model`.
    var_name : str
        Name of the range variable (default ``"range"``).
    truth : float, optional
        A known value to mark with a dashed vertical line.
    hdi_prob : float
        Mass of the shaded highest-density interval (default 0.95).
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new one is created if omitted.
    color : str
        Colour for the density and mean line.
    title : str, optional
        Axes title.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn on.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5.0, 3.5))

    draws = np.asarray(idata.posterior[var_name].values).reshape(-1)
    grid = np.linspace(draws.min(), draws.max(), 256)
    density = gaussian_kde(draws)(grid)
    ax.plot(grid, density, color=color)
    ax.fill_between(grid, density, color=color, alpha=0.08)

    hdi_low, hdi_high = az.hdi(draws, prob=hdi_prob)
    ax.axvspan(
        hdi_low,
        hdi_high,
        color=color,
        alpha=0.15,
        label=f"{hdi_prob:.0%} HDI [{hdi_low:.0f}, {hdi_high:.0f}]",
    )
    mean = float(np.mean(draws))
    ax.axvline(mean, color=color, lw=1.5, label=f"mean {mean:.0f}")
    if truth is not None:
        ax.axvline(truth, color="k", ls="--", lw=1.5, label=f"truth {truth:.0f}")

    ax.set_xlabel("range / patch size (microns)")
    ax.set_ylabel("posterior density")
    ax.set_title(title if title is not None else "Patch-size posterior")
    ax.legend(frameon=False, fontsize="small")
    return ax


def plot_amplitude_posterior(
    idata: az.InferenceData,
    *,
    var_name: str = "eta",
    truth: float | None = None,
    hdi_prob: float = 0.95,
    ax: Axes | None = None,
    color: str = "C1",
    title: str | None = None,
) -> Axes:
    """Plot the field-amplitude (``eta``) posterior -- *how strong* the structure is.

    Patch size says how *big* the patches are; amplitude says how *strongly* the
    feature is patterned. They are independent axes: a feature can segregate at
    200 microns with sharp contrast between patches or with contrast so faint it
    is effectively well-mixed. ``eta`` is the Matern field's standard deviation
    on the model's latent scale (log-mean for counts, logit for alleles); near
    zero means "no meaningful spatial structure at this scale".

    Parameters
    ----------
    idata : arviz.InferenceData
        Posterior from :func:`mesh.fit.fit_model`.
    var_name : str
        Name of the amplitude variable (default ``"eta"``).
    truth : float, optional
        A known value to mark with a dashed vertical line.
    hdi_prob : float
        Mass of the shaded highest-density interval (default 0.95).
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new one is created if omitted.
    color : str
        Colour for the density and mean line.
    title : str, optional
        Axes title.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn on.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5.0, 3.5))

    draws = np.asarray(idata.posterior[var_name].values).reshape(-1)
    grid = np.linspace(draws.min(), draws.max(), 256)
    density = gaussian_kde(draws)(grid)
    ax.plot(grid, density, color=color)
    ax.fill_between(grid, density, color=color, alpha=0.08)

    hdi_low, hdi_high = az.hdi(draws, prob=hdi_prob)
    ax.axvspan(
        hdi_low,
        hdi_high,
        color=color,
        alpha=0.15,
        label=f"{hdi_prob:.0%} HDI [{hdi_low:.2f}, {hdi_high:.2f}]",
    )
    mean = float(np.mean(draws))
    ax.axvline(mean, color=color, lw=1.5, label=f"mean {mean:.2f}")
    if truth is not None:
        ax.axvline(truth, color="k", ls="--", lw=1.5, label=f"truth {truth:.2f}")

    ax.set_xlabel("field amplitude eta (latent-scale SD)")
    ax.set_ylabel("posterior density")
    ax.set_title(title if title is not None else "Structure strength (amplitude)")
    ax.legend(frameon=False, fontsize="small")
    return ax


def plot_variance_partition(
    idata: az.InferenceData,
    *,
    hdi_prob: float = 0.95,
    ax: Axes | None = None,
    color: str = "C2",
    title: str | None = None,
) -> Axes:
    """Plot the spatially organised *share* of variance -- *how deterministic*.

    Draws the posterior of the **spatial fraction** from
    :func:`mesh.summaries.decompose_variance`: the share of latent variance that
    the spatial field explains, as opposed to the observation model's
    unstructured overdispersion ("nugget"). A fraction near 1 means the feature's
    arrangement is strongly spatially determined; near 0 means most variation is
    noise below the sampling resolution. The axis is fixed to ``[0, 1]``.

    Parameters
    ----------
    idata : arviz.InferenceData
        Posterior from :func:`mesh.fit.fit_model`.
    hdi_prob : float
        Mass of the shaded highest-density interval (default 0.95).
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new one is created if omitted.
    color : str
        Colour for the density and mean line.
    title : str, optional
        Axes title.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn on.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5.0, 3.5))

    draws = decompose_variance(idata)["fraction"]
    grid = np.linspace(0.0, 1.0, 256)
    density = gaussian_kde(draws)(grid)
    ax.plot(grid, density, color=color)
    ax.fill_between(grid, density, color=color, alpha=0.08)

    hdi_low, hdi_high = az.hdi(draws, prob=hdi_prob)
    ax.axvspan(
        hdi_low,
        hdi_high,
        color=color,
        alpha=0.15,
        label=f"{hdi_prob:.0%} HDI [{hdi_low:.2f}, {hdi_high:.2f}]",
    )
    mean = float(np.mean(draws))
    ax.axvline(mean, color=color, lw=1.5, label=f"mean {mean:.2f}")

    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("spatial fraction of latent variance")
    ax.set_ylabel("posterior density")
    ax.set_title(title if title is not None else "Spatial determinism")
    ax.legend(frameon=False, fontsize="small")
    return ax


def plot_scale_comparison(
    idatas: Sequence[az.InferenceData],
    *,
    labels: Sequence[str] | None = None,
    var_name: str = "range",
    hdi_prob: float = 0.95,
    ax: Axes | None = None,
    colors: Sequence[str] | None = None,
    title: str | None = None,
) -> Axes:
    """Overlay one parameter's posterior across several fits.

    The motivating use is comparing the **scale of a function** (an abundance
    fit) with the **scale of a genotype** (an allele fit) on the same region:
    if a function's patch is larger than any single strain's allele territory,
    the function is shared across strains; if they coincide, it is strain-private.
    More generally this overlays ``var_name`` (``"range"`` or ``"eta"``) for any
    set of fits so their credible intervals can be compared, not just their point
    estimates.

    Parameters
    ----------
    idatas : sequence of arviz.InferenceData
        Posteriors to overlay.
    labels : sequence of str, optional
        One label per fit (e.g. ``["function (abundance)", "genotype (allele)"]``).
        Defaults to ``fit 0``, ``fit 1``, ...
    var_name : str
        Parameter to compare (default ``"range"``).
    hdi_prob : float
        Mass of the highest-density interval reported per fit.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new one is created if omitted.
    colors : sequence of str, optional
        One colour per fit; defaults to the matplotlib ``C0, C1, ...`` cycle.
    title : str, optional
        Axes title.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn on.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 3.5))
    if labels is None:
        labels = [f"fit {i}" for i in range(len(idatas))]
    if len(labels) != len(idatas):
        raise ValueError("`labels` must have one entry per fit in `idatas`.")
    if colors is None:
        colors = [f"C{i}" for i in range(len(idatas))]

    for idata, label, color in zip(idatas, labels, colors):
        draws = np.asarray(idata.posterior[var_name].values).reshape(-1)
        grid = np.linspace(draws.min(), draws.max(), 256)
        density = gaussian_kde(draws)(grid)
        hdi_low, hdi_high = az.hdi(draws, prob=hdi_prob)
        ax.plot(grid, density, color=color, lw=1.8)
        ax.fill_between(grid, density, color=color, alpha=0.10)
        ax.axvspan(hdi_low, hdi_high, color=color, alpha=0.12)
        ax.axvline(
            float(np.mean(draws)),
            color=color,
            lw=1.5,
            label=f"{label}: {np.mean(draws):.0f} [{hdi_low:.0f}, {hdi_high:.0f}]",
        )

    unit = " (microns)" if var_name == "range" else ""
    ax.set_xlabel(f"{var_name}{unit}")
    ax.set_ylabel("posterior density")
    ax.set_title(title if title is not None else f"{var_name} across fits")
    ax.legend(frameon=False, fontsize="small")
    return ax


def plot_loadings(
    idata: az.InferenceData,
    feature_ids: Sequence[str] | None = None,
    *,
    var_name: str = "loadings",
    ax: Axes | None = None,
    cmap: str = "magma",
    title: str | None = None,
) -> Axes:
    """Heatmap of the coregionalization loadings -- which feature shares which field.

    For :func:`mesh.coregionalized_negbinomial`, draws the posterior mean of the
    **loading magnitudes** ``|W|`` (features x fields) as a heatmap, and outlines
    each feature's **assigned field** (the one it loads on most strongly).
    Features sharing a column share a spatial territory; the magnitude is how
    strongly. Magnitudes are used because the per-field sign is not identified
    (see :func:`mesh.summarize_loadings`).

    Parameters
    ----------
    idata : arviz.InferenceData
        Posterior carrying a ``loadings`` variable of shape ``(J, n_fields)``.
    feature_ids : sequence of str, optional
        Feature labels in the model's row order (sorted ``feature_id``; see
        :func:`mesh.fit.coregion_feature_order`). Defaults to ``feature0`` ... .
    var_name : str
        Name of the loadings variable.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new one is created if omitted.
    cmap : str
        Matplotlib colourmap for the magnitudes.
    title : str, optional
        Axes title.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn on.
    """
    draws = np.asarray(idata.posterior[var_name].values)
    flat = draws.reshape(-1, draws.shape[-2], draws.shape[-1])
    n_features, n_fields = flat.shape[1], flat.shape[2]
    if feature_ids is None:
        feature_ids = [f"feature{j}" for j in range(n_features)]

    abs_mean = np.mean(np.abs(flat), axis=0)  # (J, n_fields)
    assigned = np.argmax(abs_mean, axis=1)

    if ax is None:
        _, ax = plt.subplots(figsize=(1.1 * n_fields + 2.5, 0.55 * n_features + 1.6))

    im = ax.imshow(abs_mean, aspect="auto", cmap=cmap, vmin=0.0)
    threshold = 0.6 * float(abs_mean.max())
    for j in range(n_features):
        for k in range(n_fields):
            ax.text(
                k,
                j,
                f"{abs_mean[j, k]:.2f}",
                ha="center",
                va="center",
                color="white" if abs_mean[j, k] < threshold else "black",
                fontsize="small",
            )
        ax.add_patch(
            plt.Rectangle(
                (assigned[j] - 0.5, j - 0.5),
                1,
                1,
                fill=False,
                edgecolor="#39d353",
                lw=2.5,
            )
        )

    ax.set_xticks(range(n_fields))
    ax.set_xticklabels([f"field {k}" for k in range(n_fields)])
    ax.set_yticks(range(n_features))
    ax.set_yticklabels(list(feature_ids))
    ax.set_xlabel("latent field (spatial scale)")
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("posterior mean |loading|")
    ax.set_title(
        title if title is not None else "Feature-to-field loadings (assignment outlined)"
    )
    return ax


def posterior_field_mean(
    idata: az.InferenceData, var_name: str = "f"
) -> np.ndarray:
    """Return the posterior mean of the latent field at each sample.

    Parameters
    ----------
    idata : arviz.InferenceData
        Posterior containing the deterministic field site (default ``"f"``).
    var_name : str
        Name of the field site.

    Returns
    -------
    numpy.ndarray
        Posterior mean field, shape ``(n,)``.
    """
    return np.asarray(idata.posterior[var_name].mean(dim=("chain", "draw")).values)


def plot_field(
    coords: np.ndarray,
    field: np.ndarray,
    *,
    ax: Axes | None = None,
    cmap: str = "RdBu_r",
    s: float = 40.0,
    colorbar: bool = True,
    symmetric: bool = True,
    title: str | None = None,
) -> Axes:
    """Map a latent field over the sample coordinates.

    Works for the **true** field from a simulation (``sim.coords``,
    ``sim.field``) and for the **inferred** field
    (``coords``, :func:`posterior_field_mean`), so the two can be compared
    side by side.

    Parameters
    ----------
    coords : numpy.ndarray
        ``(n, 2)`` coordinates in microns.
    field : numpy.ndarray
        Field value at each coordinate, shape ``(n,)``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new one is created if omitted.
    cmap : str
        Matplotlib colormap name (a diverging map suits a zero-mean field).
    s : float
        Marker size.
    colorbar : bool
        Whether to draw a colorbar.
    symmetric : bool
        Centre the colour scale on zero (recommended for a zero-mean GP).
    title : str, optional
        Axes title.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn on.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5.0, 4.2))

    coords = np.asarray(coords)
    field = np.asarray(field)
    vmax = vmin = None
    if symmetric:
        vmax = float(np.max(np.abs(field)))
        vmin = -vmax

    sc = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=field,
        cmap=cmap,
        s=s,
        vmin=vmin,
        vmax=vmax,
        edgecolors="none",
    )
    ax.set_xlabel("x (microns)")
    ax.set_ylabel("y (microns)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title if title is not None else "Latent field")
    if colorbar:
        ax.figure.colorbar(sc, ax=ax, label="field", fraction=0.046, pad=0.04)
    return ax


def plot_matern_correlation(
    idata: az.InferenceData | None = None,
    *,
    range_: float | None = None,
    var_name: str = "range",
    hdi_prob: float = 0.95,
    max_distance: float | None = None,
    n_draws: int = 200,
    ax: Axes | None = None,
    color: str = "C0",
    title: str | None = None,
) -> Axes:
    """Plot the Matern 3/2 correlation decay implied by the range.

    This translates the headline number into a curve: how spatial correlation
    falls off with distance. Pass an ``idata`` to draw the posterior median
    curve with an ``hdi_prob`` band (propagating range uncertainty), or a single
    ``range_`` for one curve.

    Parameters
    ----------
    idata : arviz.InferenceData, optional
        Posterior to draw the range from. Mutually exclusive with ``range_``.
    range_ : float, optional
        A single range value, in microns. Mutually exclusive with ``idata``.
    var_name : str
        Name of the range variable in ``idata``.
    hdi_prob : float
        Mass of the shaded band around the median curve (``idata`` mode).
    max_distance : float, optional
        Largest distance to plot, in microns. Defaults to ~3x the typical range.
    n_draws : int
        Number of posterior draws to evaluate for the band.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new one is created if omitted.
    color : str
        Curve and band colour.
    title : str, optional
        Axes title.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn on.
    """
    if (idata is None) == (range_ is None):
        raise ValueError("Pass exactly one of `idata` or `range_`.")
    if ax is None:
        _, ax = plt.subplots(figsize=(5.0, 3.5))

    sqrt3 = np.sqrt(3.0)

    def corr(d: np.ndarray, rho: float) -> np.ndarray:
        scaled = sqrt3 * d / rho
        return (1.0 + scaled) * np.exp(-scaled)

    if range_ is not None:
        rng_central = float(range_)
        draws = None
    else:
        all_draws = np.asarray(idata.posterior[var_name].values).reshape(-1)
        rng = np.random.default_rng(0)
        draws = (
            all_draws
            if all_draws.size <= n_draws
            else rng.choice(all_draws, size=n_draws, replace=False)
        )
        rng_central = float(np.median(all_draws))

    if max_distance is None:
        max_distance = 3.0 * rng_central
    d = np.linspace(0.0, max_distance, 200)

    if draws is not None:
        curves = np.stack([corr(d, r) for r in draws])
        lo = (1.0 - hdi_prob) / 2.0
        band_low = np.quantile(curves, lo, axis=0)
        band_high = np.quantile(curves, 1.0 - lo, axis=0)
        ax.fill_between(
            d, band_low, band_high, color=color, alpha=0.15,
            label=f"{hdi_prob:.0%} band",
        )

    ax.plot(d, corr(d, rng_central), color=color, lw=1.8,
            label=f"range {rng_central:.0f} microns")
    ax.axhline(0.1, color="grey", ls=":", lw=1.0)
    ax.set_xlabel("distance (microns)")
    ax.set_ylabel("Matern 3/2 correlation")
    ax.set_ylim(0.0, 1.02)
    ax.set_xlim(0.0, max_distance)
    ax.set_title(title if title is not None else "Spatial correlation decay")
    ax.legend(frameon=False, fontsize="small")
    return ax
