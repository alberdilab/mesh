# Installation

MESH requires **Python ≥ 3.11**.

## From source

```bash
git clone https://github.com/anttonalberdi/mesh.git
cd mesh
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

The editable install pulls in the runtime dependencies:

| Purpose      | Packages                          |
|--------------|-----------------------------------|
| Modelling    | `numpyro`, `jax`, `numpy`         |
| Diagnostics  | `arviz`                           |
| Tables / I/O | `pandas`, `pyarrow`               |
| Simulation   | `scipy`                           |

The `[dev]` extra adds `pytest` and `ruff`; the `[docs]` extra adds the Sphinx
toolchain used to build this site.

## JAX backends

MESH runs on the CPU build of JAX by default, which is all the M0/M1 models
need — datasets are a few hundred to ~2k patches and inference uses **exact**
Gaussian processes. No GPU-specific code is required. If you later install a
GPU/TPU build of JAX, MESH will use it without changes, but this is unnecessary
at the current scale.

## Verify the install

```bash
python -c "import mesh; print(mesh.__version__)"
pytest                     # includes the simulation-based recovery tests
ruff check .
```

The recovery tests fit both models to synthetic data with a known patch size
and assert the truth falls inside the posterior credible interval; they are the
package's primary correctness guarantee (see
[Simulation & recovery](../methods/simulation.md)).

## Building the documentation locally

```bash
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
open docs/_build/html/index.html      # Linux: xdg-open
```
