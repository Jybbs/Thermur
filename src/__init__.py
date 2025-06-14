"""
Thermur package root.

Only the public surface (`__version__`, `TrainConfig`) 
is exported here. Everything else is internal.
"""
from importlib import metadata

__all__ = ["__version__", "TrainConfig"]

# Try to get version, default to development version if not installed
try:
    __version__: str = metadata.version("thermur")
except metadata.PackageNotFoundError:
    __version__: str = "0.0.0+dev"

# Lazy imports to avoid circular dependencies
def __getattr__(name):
    if name == "logger":
        from .ops.loguru import logger
        return logger
    elif name == "TrainConfig":
        from .configs import TrainConfig
        return TrainConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
