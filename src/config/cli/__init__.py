"""
CLI configuration system.

This module provides the entry point for CLI configuration, orchestrating
all settings for the terminal interface, user interaction, and system validation.
The CLI configuration uses hydra_zen.just() to wrap pure data structures since
they configure the CLI framework itself rather than instantiatable components.

The configuration is organized as a flat structure:
- display  : Terminal UI themes and formatting
- download : Globus data transfer settings
- messages : User-facing message templates
- prompts  : Interactive dialog configuration
- secrets  : Secure token storage
- system   : Environment validation rules
- wandb    : Experiment tracking access

These configs are accessed directly via instantiation without Hydra's runtime.
"""
from .stores import cli

# The CLI uses the "all" config directly from the store
CLIConfig = cli["all"]

__all__ = ["CLIConfig"]