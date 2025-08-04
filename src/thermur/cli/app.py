"""
CLI application state and shared components.

This module initializes all CLI components at module level, providing
direct access without indirection. Components are created once on import.
"""
from config.cli.builds import cfg
from .helpers          import *

ui      = ThermurUI(cfg.display)
prompts = CLIPrompts(cfg, ui)
system  = SystemInspector(cfg)

def get_globus() -> GlobusManager:
    """
    Lazy-loaded GlobusManager to avoid secrets directory warning.

    Returns:
        GlobusManager instance, created on first access.
    """
    return GlobusManager(cfg.download)
