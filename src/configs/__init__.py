"""
Thermur configuration system.

This package provides a domain-based organization for all Hydra-zen
configurations, with each domain (cli, imitation, etc.) containing
its own schemas, factories, and workloads.

The main exports are the workload configurations and registration functions
that are used by the application entry points.
"""
from .cli.workloads.cli               import cli_config, register_cli_configs
from .imitation.workloads.imitation   import register_configs

from .imitation.schemas.control       import ControlModel
from .imitation.schemas.learning      import LearningModel
from .imitation.schemas.physics       import PhysicsModel
from .imitation.schemas.safety        import SafetyModel
from .imitation.schemas.swarm         import SwarmModel
from .imitation.schemas.specs         import SwarmActionModel, SwarmObservationModel
from .imitation.schemas.monitoring    import LoggingModel, WandbModel
from .imitation.schemas.visualization import ColorModel, GlyphModel, GridModel,  OpacityModel, VisualizationModel

from .cli.schemas.core                import CLIModel, CommandsModel
from .cli.schemas.explorer            import ExplorerModel
from .cli.schemas.messages            import HeadersModel, MessagesModel
from .cli.schemas.system              import SystemModel
from .cli.schemas.ui                  import ThemeModel, UIModel


__all__ = [
    "cli_config",
    "register_cli_configs",
    "register_configs",
    
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

    "CLIModel",
    "CommandsModel",
    "ExplorerModel",
    "HeadersModel",
    "MessagesModel",
    "SystemModel",
    "ThemeModel",
    "UIModel",
]