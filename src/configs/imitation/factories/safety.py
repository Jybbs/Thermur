"""
Hydra-zen builders for safety components.

This module defines configuration builders for the Control Barrier Function
and safety filter components. These builders leverage Pydantic validation
through the zen() wrapper.
"""
from hydra_zen import builds, zen
from thermur   import ThermalBarrierFunction, SafetyFilter


build_thermal_barrier = builds(
    ThermalBarrierFunction,
    config                  = zen("${cbf}"),
    populate_full_signature = True,
    zen_dataclass           = {
        "module"   : "src.configs.factories.safety",
        "cls_name" : "ThermalBarrierBuild"
    }
)

build_safety_filter = builds(
    SafetyFilter,
    barrier                 = build_thermal_barrier,
    config                  = zen("${.}"),
    populate_full_signature = True,
    zen_dataclass           = {
        "module"   : "src.configs.factories.safety",
        "cls_name" : "SafetyFilterBuild"
    }
)
