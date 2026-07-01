"""Composition tests for the unified field core.

The five architecture axes used to live in separate models. After the
consolidation they are orthogonal knobs on one shared field builder
(:func:`mesh.model._sample_field_latent`), reached through
:func:`mesh.spatial_negbinomial`. These tests exercise **crossings that no
single previous model could express**:

* :func:`test_compose_direction_and_scales` -- direction (``anisotropic=True``)
  crossed with multiple scales (``n_fields=2``): the coregionalization still
  resolves both ranges, and the added per-field anisotropy does **not**
  manufacture a spurious direction on isotropic data.
* :func:`test_compose_smoothness_on_direction` -- boundary sharpness
  (:func:`mesh.compare_smoothness`) crossed with direction: LOO still recovers a
  rough field, and the winning anisotropic fit recovers the direction.
"""

from __future__ import annotations

import arviz as az
import numpy as np
import numpyro
import pytest

from mesh.fit import compare_smoothness, coregion_counts_arrays, counts_arrays, fit_model
from mesh.model import spatial_negbinomial
from mesh.simulate import simulate_anisotropic, simulate_coregionalized
from mesh.summaries import summarize_anisotropy

numpyro.set_host_device_count(1)


@pytest.mark.parametrize("seed", [0])
def test_compose_direction_and_scales(seed):
    """anisotropic=True + n_fields=2 recovers both scales without faking direction.

    Fit a *directional multi-scale* model on two-scale (isotropic) data. The two
    ranges must still land in their 95% intervals -- turning on the direction
    knob does not break coregionalization -- and each field's anisotropy interval
    must contain 1 (isotropy), i.e. the extra freedom does not hallucinate a
    direction where there is none.
    """
    true_ranges = (80.0, 300.0)
    sim = simulate_coregionalized(
        n_samples=220, ranges=true_ranges, domain=800.0, seed=seed
    )
    arrays = coregion_counts_arrays(sim.table)
    idata = fit_model(
        spatial_negbinomial,
        num_warmup=600,
        num_samples=600,
        num_chains=2,
        seed=seed,
        target_accept_prob=0.95,
        n_fields=2,
        anisotropic=True,
        **arrays,
    )

    # Both ranges resolved (ordered: range[..., 0] small, range[..., 1] large).
    range_draws = np.asarray(idata.posterior["range"].values)  # (chain, draw, K)
    for k, true_range in enumerate(true_ranges):
        flat = range_draws[:, :, k].reshape(-1)
        hdi_low, hdi_high = az.hdi(flat, prob=0.95)
        assert hdi_low <= true_range <= hdi_high, (
            f"field {k}: true range {true_range} outside 95% CI "
            f"[{hdi_low:.1f}, {hdi_high:.1f}]"
        )
        assert 0.5 * true_range <= float(np.mean(flat)) <= 2.0 * true_range

    # No spurious direction: each field's anisotropy (ell_x/ell_y) CI spans 1.
    aniso = np.asarray(idata.posterior["anisotropy"].values)  # (chain, draw, K)
    for k in range(aniso.shape[-1]):
        lo, hi = az.hdi(aniso[:, :, k].reshape(-1), prob=0.95)
        assert lo <= 1.0 <= hi, (
            f"field {k}: isotropic data but anisotropy CI [{lo:.2f}, {hi:.2f}] "
            "excludes 1"
        )


@pytest.mark.parametrize("seed", [0])
def test_compose_smoothness_on_direction(seed):
    """Smoothness selection composes with direction.

    Simulate a *rough* (``nu = 1/2``) **anisotropic** field and compare Matern
    smoothnesses while keeping the field anisotropic. LOO must recover the rough
    kernel, and the winning fit must still recover the direction (``x`` long).
    """
    from mesh.fit import nu_label

    true_nu = 0.5
    sim = simulate_anisotropic(
        n_samples=180,
        lengthscales=(300.0, 100.0),
        eta=1.4,
        domain=1000.0,
        concentration=40.0,
        nu=true_nu,
        seed=seed,
        jitter=1e-4,
    )
    arrays = counts_arrays(sim.table)
    comparison, idatas = compare_smoothness(
        spatial_negbinomial,
        nu_values=(0.5, 1.5, 2.5),
        num_warmup=350,
        num_samples=350,
        num_chains=2,
        seed=seed,
        target_accept_prob=0.95,
        anisotropic=True,
        jitter=1e-4,
        **arrays,
    )

    best = comparison.index[0]
    assert best == nu_label(true_nu), (
        f"LOO preferred {best!r}, expected {nu_label(true_nu)!r}\n{comparison}"
    )

    # The winning (anisotropic) fit still recovers the direction.
    summary = summarize_anisotropy(idatas[best]).set_index("quantity")
    assert summary.loc["prob_x_longer"]["mean"] > 0.9
    assert summary.loc["anisotropy_ratio"]["mean"] > 1.5
