"""
Controller domain builds for hydra-zen configuration.

This module provides pre-built components for the controller domain:

- CBFSafetyFilter       : Enforces thermal safety constraints using Control Barrier
                          Functions (CBF), solving a QP at each timestep to ensure
                          agents never exceed maximum safe temperature.

- MurmurationController : Implements murmuration dynamics with topological interactions
                          based on starling flocks, using k-nearest neighbors rather
                          than metric distances for biologically-inspired collective
                          motion.
"""
from __future__                   import annotations
from .schemas                     import *
from hydra_zen                    import builds, make_config
from thermur.imitation.controller import CBFSafetyFilter, MurmurationController
from typing                       import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from hydra_zen.typing import Builds


CONTROLLER_USER_CONFIG = make_config(
    flock  = FlockModel(),
    mmm    = MurmurationModel(),
    safety = SafetyModel()
)

CONTROLLER_SYSTEM_BUILDS: dict[str, type[Builds[Any]]] = {

    "cbf": builds(
        CBFSafetyFilter,
        flock                   = "${controller.flock}",
        safety                  = "${controller.safety}",
        populate_full_signature = True
    ),
    
    "murmuration_controller": builds(
        MurmurationController,
        cbf                     = "${_system.cbf}",
        flock                   = "${controller.flock}",
        mmm                     = "${controller.mmm}",
        safety                  = "${controller.safety}",
        populate_full_signature = True
    )
    
}
