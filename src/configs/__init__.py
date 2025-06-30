"""
Thermur configuration system.

This package provides a domain-based organization for all Hydra-zen
configurations, with each domain (cli, imitation, etc.) containing
its own schemas, factories, and workloads.

The main exports are the workload configurations and registration functions
that are used by the application entry points.
"""
# Export domain workloads
from .cli import cli_config, register_cli_configs
from .imitation import imitation_config, register_configs

# Export all domains for easy access
from . import cli, imitation

__all__ = [
    # Domains
    "cli",
    "imitation",
    
    # CLI exports
    "cli_config",
    "register_cli_configs",
    
    # Imitation exports
    "imitation_config",
    "register_configs",
]