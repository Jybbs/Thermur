"""
Provides a centralized, configuration-driven setup for the Loguru logger.

This module is responsible for initializing the logger at the start of an
application run. It uses a logging configuration to set up sinks for both
console and file-based logging, with configurable levels, formats, and
rotation policies.

At import time, Loguru's default handler is removed to prevent any logging
before the application is properly configured.
"""
from configs.imitation import LoggingModel
from loguru            import logger
from pathlib           import Path
from sys               import stderr

logger.remove()
logger = logger


def configure_loguru(cfg: LoggingModel):
    """
    Configures the global logger based on the provided settings.

    This function sets up sinks for console and optional file logging. It should
    be called once at the beginning of the application's lifecycle.

    Args:
        config: A logging configuration instance (from hydra instantiation)
               containing all necessary settings for levels, colors, file paths,
               rotation, and retention.
    """
    logger.add(
        colorize = cfg.colorize,
        format   = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</> | "
            "<level>{level: <8}</> | "
            "<cyan>{name}</>:<cyan>{function}</>:<cyan>{line}</> - "
            "<level>{message}</>"
        ),
        level    = cfg.level,
        sink     = stderr
    )

    if file_path := cfg.file_path:
        logger.info(f"File logging enabled. Writing logs to {file_path}")
        logger.add(
            diagnose  = cfg.diagnose,
            enqueue   = cfg.enqueue,
            level     = cfg.level,
            retention = cfg.retention,
            rotation  = cfg.rotation,
            sink      = file_path
        )
