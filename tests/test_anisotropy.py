"""Direction (anisotropy) tests.

Anisotropy is the **direction** axis of spatial architecture: does a feature
organise along a host axis (proximal--distal gut, crypt--villus, depth into a
biofilm)? MESH models it with an **axis-aligned** anisotropic Matern field --
a separate patch size ``ell_x`` / ``ell_y`` per coordinate axis
(:func:`mesh.anisotropic_negbinomial`).

Two checks here:

* :func:`test_anisotropic_kernel_reduces` /
  :func:`test_anisotropic_kernel_directional` -- fast, deterministic kernel
  checks: equal lengthscales reproduce the isotropic kernel exactly, and a
  longer axis decorrelates more slowly along that axis (what makes the model
  directional).
* :func:`test_recovery_anisotropic` -- the gating recovery test: simulate a
  field elongated 3x along ``x`` and confirm both axis ranges land in their 95%
  intervals, the anisotropy is resolved, and its *direction* (``x`` longer) is
  recovered.
"""

from __future__ import annotations

import numpy as np
import numpyro
import pytest

from mesh.fit import counts_arrays, fit_model
from mesh.kernels import (
    anisotropic_matern_kernel,
    anisotropic_scaled_distances,
    matern_kernel,
    pairwise_distances,
)
from mesh.model import anisotropic_negbinomial
from mesh.simulate import simulate_anisotropic
from mesh.summaries import summarize_anisotropy

numpyro.set_host_device_count(1)


def test_anisotropic_kernel_reduces():
    """Equal per-axis lengthscales reproduce the isotropic kernel exactly."""
    rng = np.random.default_rng(0)
    coords = rng.uniform(0.0, 500.0, size=(12, 2))
    ell = 150.0

    # The two paths scale at different points (distance vs. coordinates), so they
    # differ only at float32 round-off, not structurally.
    iso = np.asarray(matern_kernel(coords, ell, nu=1.5))
    aniso = np.asarray(anisotropic_matern_kernel(coords, np.array([ell, ell]), nu=1.5))
    np.testing.assert_allclose(iso, aniso, rtol=1e-5, atol=1e-6)

    # The scaled distance collapses to the isotropic distance / lengthscale.
    u = np.asarray(anisotropic_scaled_distances(coords, np.array([ell, ell])))
    np.testing.assert_allclose(
        u, np.asarray(pairwise_distances(coords)) / ell, rtol=1e-5, atol=1e-4
    )


def test_anisotropic_kernel_directional():
    """A longer axis decorrelates more slowly along that axis.

    Two points separated by the same physical distance -- one pair along ``x``,
    one along ``y`` -- must correlate *more* along the axis with the larger
    lengthscale. This is the property that lets the model read out direction.
    """
    lengthscales = np.array([300.0, 100.0])  # x is the long axis
    lag = 120.0
    along_x = np.array([[0.0, 0.0], [lag, 0.0]])
    along_y = np.array([[0.0, 0.0], [0.0, lag]])

    kx = np.asarray(anisotropic_matern_kernel(along_x, lengthscales, jitter=0.0))
    ky = np.asarray(anisotropic_matern_kernel(along_y, lengthscales, jitter=0.0))
    assert kx[0, 1] > ky[0, 1]
    assert 0.0 < ky[0, 1] < kx[0, 1] < 1.0


@pytest.mark.parametrize("seed", [0])
def test_recovery_anisotropic(seed):
    """Recover both axis ranges, the anisotropy, and its direction.

    Simulate a field elongated 3x along ``x`` (``ell_x = 300``, ``ell_y = 100``)
    and fit :func:`mesh.anisotropic_negbinomial`. The true per-axis ranges must
    lie in their 95% intervals, the folded anisotropy ratio must be clearly
    above 1 (a directional field, not isotropic), and the posterior must place
    the long axis on ``x``.
    """
    sim = simulate_anisotropic(
        n_samples=260,
        lengthscales=(300.0, 100.0),
        eta=1.4,
        domain=1000.0,
        concentration=30.0,
        seed=seed,
    )
    arrays = counts_arrays(sim.table)
    idata = fit_model(
        anisotropic_negbinomial,
        num_warmup=500,
        num_samples=500,
        num_chains=2,
        seed=seed,
        target_accept_prob=0.95,
        **arrays,
    )

    summary = summarize_anisotropy(idata).set_index("quantity")
    true_x, true_y = sim.truth["lengthscales"]

    # Both axis ranges recovered inside their 95% credible intervals.
    row_x = summary.loc["ell_x"]
    row_y = summary.loc["ell_y"]
    assert row_x["hdi_low"] <= true_x <= row_x["hdi_high"], (
        f"true ell_x {true_x} outside CI "
        f"[{row_x['hdi_low']:.0f}, {row_x['hdi_high']:.0f}]"
    )
    assert row_y["hdi_low"] <= true_y <= row_y["hdi_high"], (
        f"true ell_y {true_y} outside CI "
        f"[{row_y['hdi_low']:.0f}, {row_y['hdi_high']:.0f}]"
    )

    # The anisotropy is resolved as directional (the true ratio is 3). Check it
    # on the *signed* ell_x/ell_y, which is better behaved than the folded ratio
    # (folding a ratio near its boundary compresses the interval).
    import arviz as az

    signed = np.asarray(idata.posterior["anisotropy"].values).reshape(-1)
    lo, hi = az.hdi(signed, prob=0.95)
    assert lo <= sim.truth["anisotropy"] <= hi, (
        f"true anisotropy {sim.truth['anisotropy']:.1f} outside CI "
        f"[{lo:.1f}, {hi:.1f}]"
    )
    assert summary.loc["anisotropy_ratio"]["mean"] > 1.5, (
        f"anisotropy not resolved, mean ratio "
        f"{summary.loc['anisotropy_ratio']['mean']:.2f}"
    )

    # ...and its direction: x is confidently the long axis.
    assert summary.loc["prob_x_longer"]["mean"] > 0.9

    # Sampler health.
    r_hat = float(az.rhat(idata)["log_ratio"].values)
    assert r_hat < 1.1, f"poor convergence, r_hat={r_hat:.3f}"
