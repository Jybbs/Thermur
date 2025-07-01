"""
Thermur configuration system.

This package provides a domain-based organization for all Hydra-zen
configurations, with each domain (cli, imitation, etc.) containing
its own schemas, factories, and workloads.

The main exports are the workload configurations and registration functions
that are used by the application entry points.
"""
from .cli.schemas.core                import CLIModel, CommandsModel
from .cli.schemas.explorer            import ExplorerModel
from .cli.schemas.messages            import HeadersModel, MessagesModel
from .cli.schemas.system              import SystemModel
from .cli.schemas.ui                  import ThemeModel, UIModel
from .cli.workloads.cli               import cli_config, register_cli_configs
from .imitation.schemas.control       import ControlModel
from .imitation.schemas.learning      import LearningModel
from .imitation.schemas.physics       import PhysicsModel
from .imitation.schemas.safety        import SafetyModel
from .imitation.schemas.swarm         import SwarmModel
from .imitation.schemas.specs         import SwarmActionModel, SwarmObservationModel
from .imitation.schemas.monitoring    import LoggingModel, WandbModel
from .imitation.schemas.visualization import ColorModel, GlyphModel, GridModel,  OpacityModel, VisualizationModel
from .imitation.workloads.imitation   import register_imitation_configs


__all__ = [
    "CLIModel",
    "CommandsModel",
    "ExplorerModel",
    "HeadersModel",
    "MessagesModel",
    "SystemModel",
    "ThemeModel",
    "UIModel",
    "cli_config",
    "register_cli_configs",
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
    "register_imitation_configs"
]