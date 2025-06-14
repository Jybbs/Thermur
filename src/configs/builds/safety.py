"""
Hydra-zen builder for the Control Barrier Function safety filter.

This module defines the configuration builder for the SafetyFilter, which
ensures all control actions respect thermal safety constraints through
real-time quadratic programming.
"""
from configs.pydantic import SafetyConfig
from hydra_zen        import builds, zen
from thermur          import SafetyFilter


build_safety_filter = builds(
    SafetyFilter,
    config                  = zen(SafetyConfig),
    populate_full_signature = True,
    zen_dataclass           = {
        "module"   : "src.configs.builds.safety",
        "cls_name" : "SafetyFilterConfig"
    }
)
