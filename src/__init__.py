"""
Thermur package root.

Only the public surface (`__version__`, `ops.get_logger`, `ops.TrainCfg`) 
is exported here. Everything else is internal.
"""
from importlib  import metadata
from ops.loguru import get_logger
from ops.config import TrainCfg

__all__          = ["__version__", "get_logger", "TrainCfg"]
__version__: str = metadata.version(__name__)