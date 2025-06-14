"""
Thermur package root.

Only the public surface (`__version__`, `TrainConfig`) 
is exported here. Everything else is internal.
"""
from importlib.metadata   import version, PackageNotFoundError
from src.configs.pydantic import TrainConfig
from src.ops.loguru       import logger

__all__ = ["__version__", "TrainConfig", "logger"]

try:
    __version__: str = version("thermur")

except PackageNotFoundError:
    __version__: str = "0.0.0+dev"
