"""
Thermur package root.

Only the public surface (`__version__`, `ops.get_logger`, `ops.TrainCfg`) 
is exported here. Everything else is internal.
"""

from importlib.metadata import version as _pkg_version

from ops.logger import get_logger
from ops.config import TrainCfg

__all__          = ["__version__", "get_logger", "TrainCfg"]
__version__: str = _pkg_version(__name__)