"""
Simulation domain builds for hydra-zen configuration.

This module provides pre-built components for the simulation environment and
data processing pipeline:

- SimulationEnv  : MuJoCo-based physics simulation environment that models drone
                   swarm dynamics in wildfire scenarios. Handles agent physics,
                   collision detection, wind field interactions, and fire spread
                   dynamics.

- WRFDataSource  : Weather Research and Forecasting (WRF) model data loader that
                   ingests high-resolution atmospheric data including wind fields,
                   temperature, humidity, and fire behavior predictions.
"""
from __future__                   import annotations
from .schemas                     import *
from hydra_zen                    import builds, make_config
from thermur.imitation.simulation import SimulationEnv, WRFDataSource
from typing                       import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from hydra_zen.typing import Builds


SIMULATION_USER_CONFIG = make_config(
    loader  = LoaderModel(),
    physics = PhysicsModel()
)

SIMULATION_SYSTEM_BUILDS: dict[str, type[Builds[Any]]] = {
    "env": builds(
        SimulationEnv,
        physics                 = "${simulation.physics}",
        zen_partial             = True,
        populate_full_signature = True
    ),

    "wrf": builds(
        WRFDataSource,
        loader                  = "${simulation.loader}",
        zen_partial             = True,
        populate_full_signature = True
    )
}
