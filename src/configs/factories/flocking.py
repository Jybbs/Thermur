"""
Hydra-zen builders for flocking components.

This module defines configuration builders for the expert flocking controller
and related components that implement Reynolds rules and thermal-aware behavior.
These builders leverage Pydantic validation through the zen() wrapper.
"""
from ..schemas import AgentModel, FlockingModel, ReynoldsWeightsModel
from hydra_zen import builds, zen
from thermur   import ExpertFlockingController


build_flocking_controller = builds(
    ExpertFlockingController,
    agent_properties        = zen(AgentModel),
    flocking_params         = zen(FlockingModel),
    reynolds_weights        = zen(ReynoldsWeightsModel),
    populate_full_signature = True,
    zen_dataclass           = {
        "module"   : "src.configs.factories.flocking",
        "cls_name" : "FlockingControllerBuild"
    }
)
