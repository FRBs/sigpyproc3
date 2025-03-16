# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

from __future__ import annotations

import datetime
import inspect
import os
import sys
from importlib import import_module
from importlib.metadata import version as meta_version
from pathlib import Path

sys.path.insert(0, Path("../").resolve().as_posix())

# -- Project information -----------------------------------------------------

project = "sigpyproc3"
author = "Fast Radio Burst Software"
year = datetime.datetime.now(tz=datetime.UTC).date().year
copyright = f"{year}, {author}"  # noqa: A001
version = meta_version("sigpyproc")
release = version
master_doc = "index"
repo_url = "https://github.com/FRBs/sigpyproc3"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinx_click",
    "sphinx-prompt",
    "sphinx_copybutton",
    "numpydoc",
    "myst_nb",
    "jupyter_sphinx",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "**.ipynb_checkpoints", "Thumbs.db", ".DS_Store"]
default_role = "any"

rst_epilog = f"""
.. |project| replace:: {project}
"""
nitpicky = True

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

html_theme = "sphinx_book_theme"
html_context = {"default_mode": "light"}
html_title = project
html_theme_options = {
    "repository_url": repo_url,
    "use_repository_button": True,
    "use_issues_button": True,
    "use_download_button": True,
}

# -- Extension configuration -------------------------------------------------

autoclass_content = "class"  # include both class docstring and __init__
autodoc_member_order = "bysource"
autodoc_typehints = "none"

numpydoc_show_class_members = True
numpydoc_show_inherited_class_members = False
numpydoc_class_members_toctree = False
numpydoc_xref_param_type = True
numpydoc_xref_aliases = {
    "ndarray": "numpy.ndarray",
    "dtype": "numpy.dtype",
    "ArrayLike": "numpy.typing.ArrayLike",
    "NDArray": "numpy.typing.NDArray",
    "float32": "numpy.float32",
    "complex64": "numpy.complex64",
    "Figure": "matplotlib.figure.Figure",
    "scipy": "scipy",
    "astropy": "astropy",
    "attrs": "attrs",
    "Path": "pathlib.Path",
    "Buffer": "typing_extensions.Buffer",
    "Iterator": "collections.abc.Iterator",
    "Callable": "collections.abc.Callable",
    "Literal": "typing.Literal",
}
numpydoc_xref_ignore = {
    "of",
    "or",
    "shape",
    "type",
    "optional",
    "scalar",
    "default",
}

coverage_show_missing_items = True

myst_enable_extensions = ["colon_fence", "deflist", "dollarmath", "amsmath"]

nb_execution_mode = "auto"
nb_execution_timeout = -1

copybutton_prompt_text = ">>> "

# -- Options for intersphinx extension ---------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
    "attrs": ("https://www.attrs.org/en/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "pyfftw": ("https://pyfftw.readthedocs.io/en/latest/", None),
    "typing_extensions": ("https://typing-extensions.readthedocs.io/en/stable/", None),
}

# -- Linkcode configuration --------------------------------------------------


def linkcode_resolve(domain: str, info: dict) -> str | None:
    """Point to the source code repository, file and line number."""
    if domain != "py" or not info["module"]:
        return None
    try:
        mod = import_module(info["module"])
        if "." in info["fullname"]:
            objname, attrname = info["fullname"].split(".")
            obj = getattr(getattr(mod, objname), attrname)
        else:
            obj = getattr(mod, info["fullname"])

        file = inspect.getsourcefile(obj)
        lines, start_line = inspect.getsourcelines(obj)
    except (TypeError, AttributeError, ImportError):
        return None

    if not file or not lines:
        return None
    file_path = Path(file).resolve().relative_to(Path("..").resolve())
    end_line = start_line + len(lines) - 1

    # Determine the branch based on RTD version
    rtd_version = os.getenv("READTHEDOCS_VERSION", "latest")
    github_branch = "develop" if rtd_version == "develop" else "main"

    return f"{repo_url}/blob/{github_branch}/{file_path}#L{start_line}-L{end_line}"
