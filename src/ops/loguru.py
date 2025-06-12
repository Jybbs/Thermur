"""
Provides a centralized, configuration-driven setup for the Loguru logger.

This module is responsible for initializing the logger at the start of an
application run. It uses a `LoggingConfig` object to set up sinks for both
console and file-based logging, with configurable levels, formats, and
rotation policies.

At import time, Loguru's default handler is removed to prevent any logging
before the application is properly configured.

Usage:
    from thermur.ops.config import AppConfig
    from thermur.ops.loguru import setup_logging, logger

    # At the start of your application (e.g., in __main__):
    cfg = load_config_with_hydra()
    setup_logging(cfg.logging)
    logger.info("Logging is configured.")
"""
from config import LoggingConfig
from loguru import logger
from sys    import stderr

logger.remove()
logger = logger


def setup_logging(config: LoggingConfig):
    """
    Configures the global logger based on the provided settings.

    This function sets up sinks for console and optional file logging. It should
    be called once at the beginning of the application's lifecycle.

    Args:
        config: A `LoggingConfig` instance containing all necessary settings
                for levels, colors, file paths, rotation, and retention.
    """
    logger.add(
        sink     = stderr,
        level    = config.level,
        colorize = config.colorize,
        format   = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</> | "
            "<level>{level: <8}</> | "
            "<cyan>{name}</>:<cyan>{function}</>:<cyan>{line}</> - "
            "<level>{message}</>"
        )
    )

    if config.file_path:
        logger.info(f"File logging enabled. Writing logs to {config.file_path}")
        logger.add(
            sink      = config.file_path,
            level     = config.level,
            rotation  = config.rotation,
            retention = config.retention,
            enqueue   = config.enqueue,
            diagnose  = config.diagnose
        )