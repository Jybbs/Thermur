"""
Hydra-zen builder for the Control Barrier Function safety filter.

This module defines the configuration builder for the SafetyFilter, which
ensures all control actions respect thermal safety constraints through
real-time quadratic programming.
"""
from hydra_zen   import builds, zen
from src.configs import SafetyConfig
from src.thermur import SafetyFilter


safety_filter_config = builds(
    SafetyFilter,
    config                  = zen(SafetyConfig),
    populate_full_signature = True,
    zen_dataclass           = {
        "module"   : "src.configs.builds.safety",
        "cls_name" : "SafetyFilterConfig"
    }
)
