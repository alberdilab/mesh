"""Sphinx configuration for the MESH documentation.

Docs are written in Markdown via MyST. The API reference is generated from the
package's NumPy-style docstrings with ``autodoc`` + ``napoleon``.
"""

from __future__ import annotations

import importlib.metadata
import sys
from datetime import datetime
from pathlib import Path

# Make the package importable for autodoc when building without an install
# (Read the Docs installs the package, so this is a local-build convenience).
_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.exists():
    sys.path.insert(0, str(_SRC))

# -- Project information -----------------------------------------------------

project = "MESH"
author = "Antton Alberdi"
copyright = f"{datetime.now():%Y}, {author}"

try:
    release = importlib.metadata.version("mesh")
except importlib.metadata.PackageNotFoundError:  # pragma: no cover - local builds
    release = "0.1.0"
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinxcontrib.mermaid",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- MyST (Markdown) ---------------------------------------------------------

myst_enable_extensions = [
    "amsmath",
    "dollarmath",
    "colon_fence",
    "deflist",
    "fieldlist",
    "substitution",
    "tasklist",
]
myst_heading_anchors = 3

# -- Autodoc / autosummary ---------------------------------------------------

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
# Heavy/compiled dependencies that need not be importable to render the API.
autodoc_mock_imports: list[str] = []

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False
napoleon_use_ivar = True

# -- Intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "arviz": ("https://python.arviz.org/en/stable/", None),
    "numpyro": ("https://num.pyro.ai/en/stable/", None),
    "jax": ("https://docs.jax.dev/en/latest/", None),
}

# -- HTML output -------------------------------------------------------------

html_theme = "furo"
html_title = "MESH"
html_static_path = ["_static"]
html_theme_options = {
    "source_repository": "https://github.com/anttonalberdi/mesh",
    "source_branch": "main",
    "source_directory": "docs/",
}
