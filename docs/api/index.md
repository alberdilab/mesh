# API reference

Generated from the package's NumPy-style docstrings. The public surface is also
re-exported from the top-level `mesh` namespace (e.g. `from mesh import
fit_model`).

```{list-table}
:header-rows: 1
:widths: 24 76

* - Module
  - Responsibility
* - {doc}`kernels`
  - Matérn 3/2 covariance over 2D coordinates and its Cholesky factor.
* - {doc}`simulate`
  - Synthetic data generators with known truth (allele & counts).
* - {doc}`model`
  - The shared GP field and the two NumPyro observation models.
* - {doc}`schema`
  - Input-contract definition and validation.
* - {doc}`fit`
  - NUTS runner, range-posterior helper, table→arrays converters.
* - {doc}`summaries`
  - Patch-size (range) summaries with credible intervals.
```

```{toctree}
:hidden:

kernels
simulate
model
schema
fit
summaries
```
