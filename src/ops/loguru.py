"""
Minimal Loguru bootstrap.
"""

from loguru import logger

logger.remove()
logger.add(
    sink     = lambda m: print(m, end=""),
    format   = "<green>{time:HH:mm:ss}</> | "
               "<level>{level: <8}</> | "
               "<cyan>{name}</>:<cyan>{line}</> - "
               "<level>{message}</>",
    level    = "INFO",
    colorize = True,
)

get_logger = logger.bind