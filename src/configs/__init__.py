"""
Thermur configuration system.

This package provides a domain-based organization for all Hydra-zen
configurations, with each domain (cli, imitation, etc.) containing
its own schemas, factories, and workloads.

The main exports are the workload configurations and registration functions
that are used by the application entry points.
"""
# Domain modules - removed since intermediate __init__.py files are gone

# Registration functions and configs
from .cli.workloads.cli import cli_config, register_cli_configs
from .imitation.workloads.imitation import register_configs

# Imitation learning schemas
# Import from specific schema modules
from .imitation.schemas.control import ControlModel
from .imitation.schemas.learning import LearningModel
from .imitation.schemas.physics import PhysicsModel
from .imitation.schemas.safety import SafetyModel
from .imitation.schemas.swarm import SwarmModel
from .imitation.schemas.specs import SwarmActionModel, SwarmObservationModel
from .imitation.schemas.monitoring import LoggingModel, WandbModel
from .imitation.schemas.visualization import (
    ColorModel,
    GlyphModel,
    GridModel,
    OpacityModel,
    VisualizationModel,
)


# Note: Factory functions are not imported at the top level to avoid circular imports.
# Users should import them from configs.imitation.factories if needed.

# CLI schemas (commonly used)
# Import CLI schemas from specific modules
from .cli.schemas.core import CLIModel, CommandsModel
from .cli.schemas.explorer import ExplorerModel
from .cli.schemas.messages import HeadersModel, MessagesModel
from .cli.schemas.system import SystemModel
from .cli.schemas.ui import ThemeModel, UIModel


__all__ = [
    # Registration functions
    "cli_config",
    "register_cli_configs",
    "register_configs",
    
    # Imitation schemas
    "ControlModel",
    "LearningModel",
    "PhysicsModel",
    "SafetyModel",
    "SwarmModel",
    "LoggingModel",
    "WandbModel",
    "SwarmActionModel",
    "SwarmObservationModel",
    "ColorModel",
    "GlyphModel",
    "GridModel",
    "OpacityModel",
    "VisualizationModel",
    
    # CLI schemas
    "CLIModel",
    "CommandsModel",
    "ExplorerModel",
    "HeadersModel",
    "MessagesModel",
    "SystemModel",
    "ThemeModel",
    "UIModel",
]