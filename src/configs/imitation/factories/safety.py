"""
Hydra-zen builders for safety components.

This module defines configuration builders for the Control Barrier Function
and safety filter components. These builders leverage Pydantic validation
through the zen() wrapper.
"""
from ..schemas import SafetyModel
from hydra_zen import builds, zen
from omegaconf import SI
from thermur   import ThermalBarrierFunction, SafetyFilter


build_thermal_barrier = builds(
    ThermalBarrierFunction,
    max_temperature         = SI("${swarm.max_temperature}"),
    activation_tolerance    = SI("${safety.activation_tolerance}"),
    populate_full_signature = True,
    zen_dataclass           = {
        "module"   : "src.configs.imitation.factories.safety",
        "cls_name" : "ThermalBarrierBuild"
    }
)
"""
Builder for thermal Control Barrier Function (CBF).

Creates a thermal barrier that enforces temperature constraints T ≤ T_max for
all agents in the swarm. The barrier function h(x) = T_max - T(x) defines the
safe set, ensuring thermal safety through quadratic program filtering.
"""

build_safety_filter = builds(
    SafetyFilter,
    barrier                 = build_thermal_barrier,
    agent_count             = SI("${swarm.agent_count}"),
    spatial_dims            = SI("${swarm.spatial_dims}"),
    safety_config           = zen(SafetyModel),
    populate_full_signature = True,
    zen_dataclass           = {
        "module"   : "src.configs.imitation.factories.safety",
        "cls_name" : "SafetyFilterBuild"
    }
)
"""
Builder for the complete safety filtering system.

Wraps the thermal CBF to filter control inputs u_nom → u_safe, solving:
    min ||u - u_nom||²
    s.t. ḣ(x,u) ≥ -α·h(x)
This ensures forward invariance of the safe thermal operating region.
"""
