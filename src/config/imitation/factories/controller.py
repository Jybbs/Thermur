"""
Hydra-zen builders for flocking components.

This module defines configuration builders for the expert flocking controller
and related components that implement Reynolds rules and thermal-aware behavior.
These builders leverage Pydantic validation through the zen() wrapper.
"""
from config.imitation.schemas.controller import ControllerModel, SafetyModel
from config.imitation.schemas.flock      import FlockModel
from hydra_zen                           import builds, zen
from omegaconf                           import SI
from thermur.imitation.controller.flock  import FlockController
from thermur.imitation.controller.safety import SafetyFilter


build_flock = builds(
    FlockModel,
    populate_full_signature = True,
    zen_dataclass           = {
        "module"   : "src.configs.imitation.factories.controller",
        "cls_name" : "FlockBuild"
    }
)
"""
Builder for flock configuration.

Creates a Pydantic-validated flock configuration that defines agent properties
including count, spatial dimensions, and temperature constraints.
"""

build_controller = builds(
    FlockController,
    control                 = zen(ControllerModel),
    flock                   = SI("${flock}"),
    safety_filter           = builds(
        SafetyFilter,
        flock  = SI("${flock}"),
        safety = zen(SafetyModel),
    ),
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
