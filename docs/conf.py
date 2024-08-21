# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import datetime
import sys
from importlib.metadata import version as meta_version
from pathlib import Path

sys.path.insert(0, Path("../").resolve().as_posix())

# -- Project information -----------------------------------------------------

project = "sigpyproc3"
author = "Fast Radio Burst Software"
year = datetime.datetime.now(tz=datetime.timezone.utc).date().year
copyright = f"{year}, {author}"  # noqa: A001
version = meta_version("sigpyproc")
release = version
master_doc = "index"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    "sphinx.ext.autodoc",
    "numpydoc",
    "sphinx_autodoc_typehints",
    "sphinx.ext.coverage",
    "sphinx.ext.intersphinx",
    "sphinx_click",
    "sphinx-prompt",
    "sphinx_copybutton",
    "myst_nb",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "**.ipynb_checkpoints", "Thumbs.db", ".DS_Store"]

rst_epilog = f"""
.. |project| replace:: {project}
"""
nitpicky = True

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

html_theme = "sphinx_book_theme"
html_title = project
html_theme_options = {
    "repository_url": "https://github.com/FRBs/sigpyproc3",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_download_button": True,
}
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# \html_static_path = ["_static"]

# -- Extension configuration -------------------------------------------------

autoclass_content = "class"  # include both class docstring and __init__
autodoc_member_order = "bysource"
autodoc_typehints = "none"
autodoc_inherit_docstrings = True

typehints_document_rtype = False

numpydoc_use_plots = True
numpydoc_class_members_toctree = False
numpydoc_show_inherited_class_members = False
numpydoc_xref_param_type = True
numpydoc_xref_aliases = {
    "ndarray": "numpy.ndarray",
    "dtype": "numpy.dtype",
    "ArrayLike": "numpy.typing.ArrayLike",
    "plt": "matplotlib.pyplot",
    "scipy": "scipy",
    "astropy": "astropy",
    "attrs": "attrs",
    "Path": "pathlib.Path",
    "Buffer": "typing_extensions.Buffer",
    "Iterator": "collections.abc.Iterator",
    "Callable": "collections.abc.Callable",
}
numpydoc_xref_ignore = {
    "of",
    "shape",
    "type",
    "optional",
    "default",
}


coverage_show_missing_items = True

myst_enable_extensions = ["colon_fence"]

nb_execution_mode = "auto"
nb_execution_timeout = -1

# -- Options for intersphinx extension ---------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
    "attrs": ("https://www.attrs.org/en/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "typing_extensions": ("https://typing-extensions.readthedocs.io/en/stable/", None),
}
