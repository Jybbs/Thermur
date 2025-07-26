"""
Hydra-zen factory for Thermur's visualization system.

This module defines the configuration builder for the visualization component,
creating a Hydra-compatible config that instantiates the Visualizer with
parameters validated by the VisualizationModel Pydantic model.
"""
from config.imitation.schemas.visualization import *
from hydra_zen                              import builds, zen
from omegaconf                              import SI
from thermur.imitation.visualization        import Visualizer


build_visualizer = builds(
    Visualizer,
    colors                  = zen(ColorModel),
    display                 = zen(DisplayModel),
    glyphs                  = zen(GlyphModel),
    grids                   = zen(GridModel),
    opacity                 = zen(OpacityModel),
    simulation              = SI("${simulation}"),
    populate_full_signature = True,
    zen_dataclass           = {
        "module"   : "src.configs.imitation.factories.visualization",
        "cls_name" : "VisualizerBuild"
    }
)
"""
Builder for real-time 3D visualization system.

Creates an interactive PyVista-based renderer that displays the multi-agent flock
navigating through dynamic temperature fields. Visualizes agent positions as glyphs,
temperature distributions as volumetric heatmaps, safety boundaries as isosurfaces,
and communication topology as edge connections. Supports customizable color mappings,
transparency settings, camera controls, and video export for training diagnostics
and paper figures. Essential for debugging emergent behaviors and safety violations.
"""
