"""
CLI workload configurations.

This module contains the top-level workload configurations that compose
all necessary components for the CLI system.
"""
from __future__ import annotations

from .cli import cli_config, register_cli_configs

__all__ = [
    "cli_config",
    "register_cli_configs",
]