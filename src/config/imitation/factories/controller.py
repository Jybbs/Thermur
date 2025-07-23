"""
Hydra-zen builders for flocking components.

This module defines configuration builders for the expert flocking controller
and related components that implement Reynolds rules and thermal-aware behavior.
These builders leverage Pydantic validation through the zen() wrapper.
"""
from config.imitation.schemas.controller   import ControllerModel, SafetyModel
from config.imitation.schemas.flock        import FlockModel
from hydra_zen                             import builds, zen
from omegaconf                             import SI
from thermur.imitation.controller.flocking import ExpertFlockingController
from thermur.imitation.controller.safety   import SafetyFilter


build_controller = builds(
    ExpertFlockingController,
    agent_properties        = zen(FlockModel),
    control                 = zen(ControllerModel),
    safety_filter           = builds(
        SafetyFilter,
        agent_count          = SI("${flock.agent_count}"),
        max_temperature      = SI("${flock.max_temperature}"),
        safety               = zen(SafetyModel),
        spatial_dims         = SI("${flock.spatial_dims}"),
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
