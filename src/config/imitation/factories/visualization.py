"""
Hydra-zen factory for Thermur's visualization system.

This module defines the configuration builder for the visualization component,
creating a Hydra-compatible config that instantiates the Visualizer with
parameters validated by the VisualizationModel Pydantic model.
"""
from config.imitation.schemas.visualization     import VisualizationModel
from hydra_zen                                  import builds, zen
from omegaconf                                  import SI
from thermur.imitation.visualization.visualizer import Visualizer


build_visualizer = builds(
    Visualizer,
    max_temperature         = SI("${flock.max_temperature}"),
    simulation              = SI("${simulation}"),
    visualization           = zen(VisualizationModel),
    populate_full_signature = True,
    zen_dataclass           = {
        "module"   : "src.configs.imitation.factories.visualization",
        "cls_name" : "VisualizerBuild"
    }
)
"""
Builder for the flock visualization system.

Provides real-time rendering of agent positions, temperature fields,
and safety boundaries for monitoring training progress and behavior.
"""
