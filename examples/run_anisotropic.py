"""Direction (anisotropy): recover a per-axis patch size from abundance counts.

Simulates a negative-binomial counts table over an **anisotropic** Matern field
(elongated 3x along ``x``), validates it against the input schema, fits
``anisotropic_negbinomial`` and prints the per-axis patch sizes together with the
directional readout: the anisotropy ratio (how directional) and which axis is the
long one. It also saves a two-panel figure -- the input sampling map and the
per-axis patch-size posteriors -- next to this script.

Run::

    python examples/run_anisotropic.py

The model is axis-aligned: it assumes any elongation lies along ``x``/``y``, so
in a real study orient the sampling frame to the host axis (proximal-distal gut,
crypt-villus, depth into a biofilm). It is parameterised by an overall
(geometric-mean) ``range`` and a signed log anisotropy centred on isotropy, so a
directional reading has to be supported by the data.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from mesh.fit import counts_arrays, fit_model
from mesh.model import anisotropic_negbinomial
from mesh.plots import plot_anisotropy, plot_samples
from mesh.schema import validate_table
from mesh.simulate import simulate_anisotropic
from mesh.summaries import summarize_anisotropy


def main() -> None:
    true_lengthscales = (300.0, 100.0)  # elongated 3x along x
    sim = simulate_anisotropic(
        n_samples=250,
        lengthscales=true_lengthscales,
        eta=1.4,
        domain=1000.0,
        seed=0,
    )

    # The table is the documented interface; validate before fitting.
    validate_table(sim.table)
    arrays = counts_arrays(sim.table)

    idata = fit_model(
        anisotropic_negbinomial,
        num_warmup=500,
        num_samples=500,
        num_chains=2,
        seed=0,
        target_accept_prob=0.95,
        progress_bar=False,
        **arrays,
    )

    summary = summarize_anisotropy(idata).set_index("quantity")

    def _ci(name: str) -> str:
        row = summary.loc[name]
        return f"{row['mean']:.1f} (95% CI {row['hdi_low']:.1f}-{row['hdi_high']:.1f})"

    prob_x = summary.loc["prob_x_longer"]["mean"]
    long_axis = "x" if prob_x >= 0.5 else "y"
    print("Anisotropic (directional) abundance counts")
    print(f"  true patch size:   ell_x={true_lengthscales[0]:.0f}, "
          f"ell_y={true_lengthscales[1]:.0f} microns "
          f"(anisotropy {sim.truth['anisotropy']:.1f}x along x)")
    print(f"  posterior ell_x:   {_ci('ell_x')} microns")
    print(f"  posterior ell_y:   {_ci('ell_y')} microns")
    print(f"  overall range:     {_ci('range')} microns (geometric mean)")
    print(f"  anisotropy ratio:  {_ci('anisotropy_ratio')} (folded, 1 = isotropic)")
    print(f"  long axis:         {long_axis} (P[x longer] = {prob_x:.2f})")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    plot_samples(
        sim.table, value="count", ax=axes[0], title="Input: observed counts"
    )
    plot_anisotropy(idata, truth=true_lengthscales, ax=axes[1])
    fig.tight_layout()

    out = Path(__file__).with_name("anisotropic_panel.png")
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
