"""
Hydra-zen factory for Thermur's visualization system.

This module defines the configuration builder for the visualization component,
creating a Hydra-compatible config that instantiates the Visualizer with
parameters validated by the VisualizationModel Pydantic model.
"""
from ..schemas import ColorModel, GlyphModel, GridModel, OpacityModel, VisualizationModel
from hydra_zen import builds, zen
from omegaconf import SI
from thermur   import Visualizer


build_visualizer = builds(
    Visualizer,
    config                  = zen(VisualizationModel),
    environment             = SI("${simulation}"),
    populate_full_signature = True,
    zen_dataclass           = {
        "module"   : "src.configs.factories.visualization",
        "cls_name" : "VisualizerBuild"
    }
)
