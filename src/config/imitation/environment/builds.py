"""
Environment domain builds for hydra-zen configuration.

This module provides pre-built components for trajectory generation and
environmental data access:

- TrajectoryGenerator : Lightweight physics simulation for offline trajectory
                        generation. Manages flock dynamics and returns PyG Data
                        objects for behavioral cloning.

- WRFLoader           : Weather Research and Forecasting (WRF) model data loader
                        that provides wind fields, temperature, and gradient data
                        at agent positions.
"""
from __future__                    import annotations
from .schemas                      import *
from hydra_zen                     import builds, make_config
from thermur.imitation.environment import TrajectoryGenerator, WRFLoader
from typing                        import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from hydra_zen.typing import Builds


ENVIRONMENT_USER_CONFIG = make_config(
    dataset = DatasetModel(),
    physics = PhysicsModel()
)

ENVIRONMENT_SYSTEM_BUILDS: dict[str, type[Builds[Any]]] = {

    "trajectory_generator": builds(
        TrajectoryGenerator,
        agent_count             = "${controller.mmm.agent_count}",
        initial_spacing         = "${controller.mmm.initial_spacing}",
        k_neighbors             = "${controller.mmm.k_neighbors}",
        physics                 = "${environment.physics}",
        wrf                     = "${_system.wrf}",
        populate_full_signature = True
    ),

    "wrf": builds(
        WRFLoader,
        bounds_min              = "${environment.physics.bounds_min}",
        bounds_max              = "${environment.physics.bounds_max}",
        populate_full_signature = True
    )

}
