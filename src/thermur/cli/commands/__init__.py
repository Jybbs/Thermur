"""
CLI command modules for the Thermur application.

Each module within this package defines a self-contained command and exposes a 
`Typer` instance (e.g., `cmd_train`) that can be registered by the main CLI 
application.
"""
from __future__ import annotations
from .configure import cmd_configure
from .info      import cmd_info
from .monitor   import cmd_monitor
from .train     import cmd_train
from .validate  import cmd_validate

__all__ = [
    "cmd_configure",
    "cmd_info",
    "cmd_monitor",
    "cmd_train",
    "cmd_validate",
]