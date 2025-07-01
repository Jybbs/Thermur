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
    control_config          = zen(ControlModel),
    populate_full_signature = True,
    zen_dataclass           = {
        "module"   : "src.configs.imitation.factories.flocking",
        "cls_name" : "FlockingControllerBuild"
    }
)
"""
Builder for the expert flocking controller.

Implements Reynolds rules augmented with thermal avoidance to generate
expert demonstrations for imitation learning in wildfire scenarios.
"""
