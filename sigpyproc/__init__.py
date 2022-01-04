import sys

if sys.version_info >= (3, 8):
    from importlib import metadata as importlib_metadata  # noqa: WPS433
else:
    import importlib_metadata  # noqa: WPS440, WPS433

__version__ = importlib_metadata.version(__name__)  # noqa: WPS410
