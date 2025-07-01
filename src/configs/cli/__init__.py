"""
CLI configuration domain.

This package contains all configuration-related code for the Thermur CLI,
including schemas, factories, and workloads organized by functionality.
"""

from . import factories, schemas, workloads

# For convenience, re-export commonly used schemas at the domain level
from .schemas import (
    CLIModel,
    CommandsModel,
    ExplorerModel,
    HeadersModel,
    MessagesModel,
    SystemModel,
    ThemeModel,
    UIModel,
)

# Re-export all factories at the domain level
from .factories import *

# Re-export workload configuration
from .workloads import cli_config, register_cli_configs

__all__ = [
    # Submodules
    "factories",
    "schemas",
    "workloads",
    
    # Workload exports
    "cli_config",
    "register_cli_configs",
    
    # Commonly used schemas
    "CLIModel",
    "CommandsModel",
    "ExplorerModel", 
    "HeadersModel",
    "MessagesModel",
    "SystemModel",
    "ThemeModel",
    "UIModel",
    
    # All factory builders (from factories import)
    "build_cli",
    "build_commands",
    "build_explorer",
    "build_explorer_messages",
    "build_headers",
    "build_messages",
    "build_message_types",
    "build_presets",
    "build_prompts",
    "build_sections",
    "build_status",
    "build_system",
    "build_theme",
    "build_tips",
    "build_training_components",
    "build_ui",
    "build_validation",
    "build_wandb_display",
]