"""Boundary-sharpness (Matern smoothness ``nu``) tests.

The smoothness ``nu`` is the **boundary sharpness** axis: small ``nu`` = a rough
field with crisp patch edges (competitive exclusion / a physical barrier), large
``nu`` = a smooth field whose patches fade as gradients (a diffusion-limited
gradient of O2, pH or nutrients). Arbitrary ``nu`` needs the fractional-order
Bessel ``K_nu`` (no autodiff-stable JAX form), so MESH offers the closed-form
family {1/2, 3/2, 5/2} plus the squared-exponential ``nu -> inf`` limit and
**selects among them by LOO model comparison** (:func:`mesh.compare_smoothness`).

Two checks here:

* :func:`test_matern_kernel_closed_forms` / :func:`test_smoothness_orders_roughness`
  -- fast, deterministic kernel checks: the closed forms are correct and the
  knob orders roughness monotonically (this is what makes ``nu`` a boundary
  sharpness readout), the SE limit included.
* :func:`test_recovery_smoothness` -- the gating selection test: simulate a
  *rough* field and confirm LOO recovers the rough kernel. Roughness is the
  reliably identifiable direction; confirming extra smoothness from noisy counts
  is much weaker (a rough kernel fits smooth data too), so the gating truth is
  the rough end -- which is also the milestone's headline biological signal.
"""

from __future__ import annotations

import math

import numpy as np
import numpyro
import pytest

from mesh.fit import compare_smoothness, counts_arrays, nu_label
from mesh.kernels import MATERN_NU, matern32_kernel, matern_kernel
from mesh.model import spatial_negbinomial
from mesh.simulate import simulate_counts

numpyro.set_host_device_count(1)


def test_matern_kernel_closed_forms():
    """Each closed form matches its analytic value and matches ``matern32``."""
    coords = np.array([[0.0, 0.0], [30.0, 40.0]])  # the pair is 50 microns apart
    ell = 100.0
    r = 50.0
    k12 = np.asarray(matern_kernel(coords, ell, nu=0.5, jitter=0.0))
    k32 = np.asarray(matern_kernel(coords, ell, nu=1.5, jitter=0.0))
    k52 = np.asarray(matern_kernel(coords, ell, nu=2.5, jitter=0.0))
    kse = np.asarray(matern_kernel(coords, ell, nu=math.inf, jitter=0.0))

    s3 = math.sqrt(3.0) * r / ell
    s5 = math.sqrt(5.0) * r / ell
    assert k12[0, 1] == pytest.approx(math.exp(-r / ell))
    assert k32[0, 1] == pytest.approx((1.0 + s3) * math.exp(-s3))
    assert k52[0, 1] == pytest.approx((1.0 + s5 + s5**2 / 3.0) * math.exp(-s5))
    assert kse[0, 1] == pytest.approx(math.exp(-0.5 * (r / ell) ** 2))

    # The dedicated Matern 3/2 kernel must stay identical to the nu=1.5 path.
    legacy = np.asarray(matern32_kernel(coords, ell))
    parametric = np.asarray(matern_kernel(coords, ell, nu=1.5))
    np.testing.assert_allclose(legacy, parametric)

    # Diagonal is the (unit) variance plus jitter; off-diagonal is a correlation.
    assert np.asarray(matern_kernel(coords, ell, nu=1.5, jitter=1e-6))[0, 0] == (
        pytest.approx(1.0 + 1e-6)
    )

    with pytest.raises(ValueError, match="closed-form"):
        matern_kernel(coords, ell, nu=1.0)


def test_smoothness_orders_roughness():
    """Larger ``nu`` decorrelates more slowly: the boundary-sharpness ordering.

    At a fixed lag (one lengthscale) the correlation is strictly increasing in
    ``nu`` -- a rougher kernel (smaller ``nu``) loses correlation faster, i.e.
    has crisper boundaries. This is the property that makes ``nu`` a readout of
    boundary sharpness, and it holds right up to the SE limit.
    """
    coords = np.array([[0.0, 0.0], [100.0, 0.0]])  # lag == lengthscale
    ell = 100.0
    corr = [
        float(np.asarray(matern_kernel(coords, ell, nu=nu, jitter=0.0))[0, 1])
        for nu in MATERN_NU  # ascending: 0.5, 1.5, 2.5, inf
    ]
    assert corr == sorted(corr), f"correlation not increasing in nu: {corr}"
    assert all(0.0 < c < 1.0 for c in corr)


@pytest.mark.parametrize("seed", [0])
def test_recovery_smoothness(seed):
    """LOO recovers a rough (crisp-edged) field as the rough Matern kernel.

    Simulate a ``nu = 1/2`` field (the roughest, crispest boundaries) and fit the
    same counts under each closed-form Matern smoothness. The generating kernel
    must win the LOO comparison, decisively (the runner-up's ELPD deficit is many
    standard errors below zero), and the winning fit must be converged.
    """
    true_nu = 0.5
    family = (0.5, 1.5, 2.5)
    sim = simulate_counts(
        n_samples=150,
        range_=200.0,
        eta=1.3,
        domain=1000.0,
        concentration=40.0,
        nu=true_nu,
        seed=seed,
        jitter=1e-4,
    )
    arrays = counts_arrays(sim.table)
    comparison, idatas = compare_smoothness(
        spatial_negbinomial,
        nu_values=family,
        num_warmup=350,
        num_samples=350,
        num_chains=2,
        seed=seed,
        target_accept_prob=0.95,
        jitter=1e-4,
        **arrays,
    )

    # The generating smoothness wins.
    best = comparison.index[0]
    assert best == nu_label(true_nu), (
        f"LOO preferred {best!r}, expected {nu_label(true_nu)!r}\n{comparison}"
    )

    # ...and wins decisively: the runner-up sits well below zero ELPD difference,
    # beyond twice its standard error (so this is not a coin-flip among kernels).
    runner_up = comparison.iloc[1]
    assert runner_up["elpd_diff"] + 2.0 * runner_up["dse"] < 0.0, (
        f"smoothness not decisively resolved:\n{comparison}"
    )

    # Sampler health for the winning fit (the non-centred GP at target_accept 0.95).
    import arviz as az

    r_hat = float(az.rhat(idatas[best])["range"].values)
    assert r_hat < 1.1, f"poor convergence in winning fit, r_hat={r_hat:.3f}"
