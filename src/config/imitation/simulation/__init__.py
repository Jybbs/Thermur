"""
Simulation configuration domain.

Provides schemas and stores for simulation environment components including
physics settings, data loading, and MuJoCo environment configuration.
"""
# Re-export schemas for clean imports
from .schemas import (
    LoaderModel,
    PhysicsModel
)

__all__ = [
    # Schemas
    "LoaderModel",
    "PhysicsModel"
]