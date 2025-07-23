"""
Hydra-zen builders for flock data specifications.

This module provides factory functions that create torchrl spec objects
through builder functions that are compatible with Hydra's serialization.
"""
from config.imitation.schemas.flock import FlockModel
from hydra_zen                      import builds


build_flock = builds(
    FlockModel,
    populate_full_signature = True,
    zen_dataclass           = {
        "module"   : "src.configs.imitation.factories.flock",
        "cls_name" : "FlockBuild"
    }
)
"""
Builder for flock configuration.

Creates a Pydantic-validated flock configuration that defines agent properties
including count, spatial dimensions, and temperature constraints.
"""