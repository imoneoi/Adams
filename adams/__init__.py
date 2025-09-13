from .adams_zero import Adams_ZeRO

try:
    from ._version import __version__
except ImportError:
    # Fallback for development installations
    try:
        from setuptools_scm import get_version
        __version__ = get_version(root='..', relative_to=__file__)
    except (ImportError, LookupError):
        __version__ = "unknown"

__all__ = ["Adams_ZeRO"]
