"""
Hydra-zen builders for flocking components.

This module defines configuration builders for the expert flocking controller
and related components that implement Reynolds rules and thermal-aware behavior.
These builders leverage Pydantic validation through the zen() wrapper.
"""
from ..schemas       import ControlModel, SwarmModel
from hydra_zen       import builds, zen
from thermur.control import ExpertFlockingController


build_flocking_controller = builds(
    ExpertFlockingController,
    agent_properties        = zen(SwarmModel),
    flocking_params         = zen(ControlModel),
    reynolds_weights        = zen(ControlModel),
    populate_full_signature = True,
    zen_dataclass           = {
        "module"   : "src.configs.factories.flocking",
        "cls_name" : "FlockingControllerBuild"
    }
)
