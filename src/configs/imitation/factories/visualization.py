"""
Hydra-zen factory for Thermur's visualization system.

This module defines the configuration builder for the visualization component,
creating a Hydra-compatible config that instantiates the Visualizer with
parameters validated by the VisualizationModel Pydantic model.
"""
from ..schemas             import VisualizationModel
from hydra_zen             import builds, zen
from omegaconf             import SI
from thermur.visualization import Visualizer


build_visualizer = builds(
    Visualizer,
    visualization_config    = zen(VisualizationModel),
    max_temperature         = SI("${swarm.max_temperature}"),
    simulation              = SI("${simulation}"),
    populate_full_signature = True,
    zen_dataclass           = {
        "module"   : "src.configs.factories.visualization",
        "cls_name" : "VisualizerBuild"
    }
)
