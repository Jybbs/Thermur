"""
Environment domain builds for hydra-zen configuration.

This module provides pre-built components for trajectory generation and
environmental data access:

- TrajectoryGenerator : Lightweight physics simulation for offline trajectory
                        generation. Manages flock dynamics and returns PyG Data
                        objects without TorchRL overhead.

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
    loader  = LoaderModel(),
    physics = PhysicsModel()
)

ENVIRONMENT_SYSTEM_BUILDS: dict[str, type[Builds[Any]]] = {

    "trajectory_generator": builds(
        TrajectoryGenerator,
        k_neighbors             = "${controller.mmm.k_neighbors}",
        mmm                     = "${controller.mmm}",
        physics                 = "${environment.physics}",
        safety                  = "${controller.safety}",
        wrf                     = "${_system.wrf}",
        populate_full_signature = True
    ),

    "wrf": builds(
        WRFLoader,
        loader                  = "${environment.loader}",
        physics                 = "${environment.physics}",
        populate_full_signature = True
    )

}
