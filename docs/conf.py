from __future__ import annotations

import ast
import datetime
import inspect
import os
import struct
import sys
from collections.abc import Mapping
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any

DOCS_DIR = Path(__file__).resolve().parent
ROOT_DIR = DOCS_DIR.parent
SRC_DIR = ROOT_DIR / "src"

sys.path.insert(0, str(SRC_DIR))

from sigpyproc import __version__  # noqa: E402

if TYPE_CHECKING:
    from sphinx.application import Sphinx

# -- Project information
project = "sigpyproc3"
author = "Pravir Kumar"
year = datetime.datetime.now(tz=datetime.UTC).date().year
copyright = f"{year}, {author}"  # noqa: A001
release = __version__
version = release
master_doc = "index"
repo_url = "https://github.com/FRBs/sigpyproc3"

# -- General configuration
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "numpydoc",
    "myst_nb",
    "jupyter_sphinx",
    "sphinx_design",
    "sphinx_click",
    "sphinx_copybutton",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]
exclude_patterns = ["_build", "**.ipynb_checkpoints", "Thumbs.db", ".DS_Store"]
default_role = "any"
nitpicky = True
rst_epilog = f"""
.. |project| replace:: {project}
.. _rocket_fft: https://github.com/styfenschaer/rocket-fft
"""

# -- HTML
html_theme = "sphinx_book_theme"
html_context = {"default_mode": "light"}
html_title = project
html_last_updated_fmt = "%b %d, %Y"
html_theme_options = {
    "repository_url": repo_url,
    "use_repository_button": True,
    "use_issues_button": True,
    "use_download_button": True,
    "path_to_docs": "docs",
    "repository_branch": "main",
    "show_toc_level": 2,
    "header_links_before_dropdown": 4,
    "use_fullscreen_button": True,
    "show_navbar_depth": 2,
    "navigation_with_keys": True,
    "toc_title": "On this page",
}

# -- Autodoc / autosummary
autosummary_generate = ["_autosummary_trigger.md"]
autoclass_content = "class"  # include both class docstring and __init__
autodoc_member_order = "bysource"
autodoc_typehints = "none"
autodoc_preserve_defaults = True
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
coverage_show_missing_items = True
suppress_warnings = ["autosummary.import_cycle"]

# -- Numpydoc
numpydoc_show_class_members = True
numpydoc_show_inherited_class_members = False
numpydoc_class_members_toctree = False
numpydoc_xref_param_type = True
numpydoc_xref_aliases = {
    # Force builtins to use py:class role for consistent link color
    "bool": ":py:class:`bool`",
    "int": ":py:class:`int`",
    "str": ":py:class:`str`",
    "bytes": ":py:class:`bytes`",
    "list": ":py:class:`list`",
    "dict": ":py:class:`dict`",
    "tuple": ":py:class:`tuple`",
    "set": ":py:class:`set`",
    "None": ":py:obj:`None`",
    "ndarray": "numpy.ndarray",
    "dtype": "numpy.dtype",
    "ArrayLike": "numpy.typing.ArrayLike",
    "NDArray": "numpy.typing.NDArray",
    "uint8": "numpy.uint8",
    "int32": "numpy.int32",
    "float32": "numpy.float32",
    "float64": "numpy.float64",
    "complex64": "numpy.complex64",
    "Figure": "matplotlib.figure.Figure",
    "Axes": "matplotlib.axes.Axes",
    "scipy": "scipy",
    "astropy": "astropy",
    "attrs": "attrs",
    "Path": "pathlib.Path",
    "Buffer": "collections.abc.Buffer",
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

# -- MyST / notebooks
myst_enable_extensions = [
    "colon_fence",
    "substitution",
    "deflist",
    "dollarmath",
    "amsmath",
]
nb_execution_mode = "auto"
nb_execution_timeout = 300

# -- Sphinx copybutton
copybutton_prompt_text = ">>> "

# -- Intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
    "attrs": ("https://www.attrs.org/en/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "pyfftw": ("https://pyfftw.readthedocs.io/en/latest/", None),
}


# -- Linkcode
def linkcode_resolve(domain: str, info: dict) -> str | None:
    """Point to the source code repository, file and line number."""
    if domain != "py":
        return None
    module_name = info.get("module")
    fullname = info.get("fullname")

    if not module_name or not fullname:
        return None
    try:
        mod = import_module(module_name)
        obj = mod
        for attr in fullname.split("."):
            obj = getattr(obj, attr)
        file = inspect.getsourcefile(obj)
        lines, start_line = inspect.getsourcelines(obj)
    except (TypeError, AttributeError, ImportError, OSError):
        return None

    if not file or not lines:
        return None
    try:
        file_path = Path(file).resolve().relative_to(ROOT_DIR)
    except ValueError:
        return None
    end_line = start_line + len(lines) - 1

    # Determine the branch based on RTD version
    rtd_version = os.getenv("READTHEDOCS_VERSION", "latest")
    rtd_version_type = os.getenv("READTHEDOCS_VERSION_TYPE", "")
    if rtd_version_type == "tag":
        github_ref = rtd_version
    elif rtd_version == "develop":
        github_ref = "develop"
    else:
        github_ref = "main"

    return f"{repo_url}/blob/{github_ref}/{file_path}#L{start_line}-L{end_line}"


def find_data_like_classes(src_path: Path) -> set[str]:  # noqa: C901
    result = set()

    for py_file in src_path.rglob("*.py"):
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001, S112
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for decorator in node.decorator_list:
                    name = None

                    # @dataclass
                    if isinstance(decorator, ast.Name):
                        name = decorator.id

                    # @attrs.define / @attrs.frozen
                    elif isinstance(decorator, ast.Attribute):
                        name = decorator.attr

                    # @dataclass(...)
                    elif isinstance(decorator, ast.Call):
                        if isinstance(decorator.func, ast.Name):
                            name = decorator.func.id
                        elif isinstance(decorator.func, ast.Attribute):
                            name = decorator.func.attr

                    if name in {"dataclass", "define", "frozen"}:
                        result.add(node.name)
                        break

    return sorted(result)


DICT_TABLE_TARGETS = {
    "sigpyproc.io.sigproc.SIGPROC_SCHEMA",
    "sigpyproc.io.sigproc.telescope_ids",
    "sigpyproc.io.sigproc.machine_ids",
}


def _headerfield_to_type_desc(value: Any) -> tuple[str, str]:
    if hasattr(value, "fmt") and hasattr(value, "doc"):
        fmt = value.fmt
        if isinstance(fmt, struct.Struct):
            return f"``{fmt.format}`` ({fmt.size} bytes)", value.doc
        if fmt is None:
            return "string", value.doc
        return str(fmt), value.doc
    return type(value).__name__, repr(value)


def _add_list_table(
    lines: list[str],
    items_dict: dict[str, Any],
    title: str = "",
    key_name: str = "Key",
    value_name: str = "Value",
) -> None:
    lines.extend(
        [
            f".. list-table::{title}",
            "   :header-rows: 1",
            "",
            f"   * - {key_name}",
            f"     - {value_name}",
        ]
    )
    for key, value in sorted(items_dict.items(), key=lambda x: x[1]):
        lines.extend([f"   * - {key}", f"     - {value}"])


def _autodoc_schema_to_table(
    app: Sphinx,
    what: str,
    name: str,
    obj: Any,
    options: Any,
    lines: list[str],
) -> None:
    if (
        what not in {"data", "attribute"}
        or name not in DICT_TABLE_TARGETS
        or not isinstance(obj, Mapping)
    ):
        return
    option_keys = set(options.keys()) if hasattr(options, "keys") else set()
    if "no-value" not in option_keys and "no_value" not in option_keys:
        return

    original = list(lines)
    lines.clear()
    if original:
        lines.extend(original)
        lines.append("")
    # SIGPROC_SCHEMA special rendering
    if name.endswith("SIGPROC_SCHEMA"):
        lines.extend(
            [
                ".. list-table:: SIGPROC Header Schema",
                "   :header-rows: 1",
                "",
                "   * - Key",
                "     - Binary format",
                "     - Description",
            ]
        )

        for key, value in sorted(obj.items()):
            typ, desc = _headerfield_to_type_desc(value)
            lines.extend(
                [
                    f"   * - ``{key}``",
                    f"     - {typ}",
                    f"     - {desc}",
                ]
            )
        return

    # telescope_ids special rendering
    if name.endswith("telescope_ids"):
        _add_list_table(
            lines,
            obj,
            title=" Radio Telescope IDs",
            key_name="Radio Telescope",
            value_name="SIGPROC ID",
        )
        return
    # machine_ids special rendering
    if name.endswith("machine_ids"):
        _add_list_table(
            lines,
            obj,
            title=" Telescope Backend IDs",
            key_name="Radio Telescope Backend",
            value_name="SIGPROC ID",
        )
        return
    # Fallback for other dict-like targets
    _add_list_table(lines, obj, title="", key_name="Key", value_name="Value")
    return


def setup(app: Sphinx) -> None:
    app.connect("autodoc-process-docstring", _autodoc_schema_to_table)
