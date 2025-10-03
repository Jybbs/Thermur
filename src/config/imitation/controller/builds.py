"""
Controller domain builds for hydra-zen configuration.

This module provides pre-built components for the controller domain:

- MurmurationController : Implements murmuration dynamics with topological interactions
                          based on starling flocks, using k-nearest neighbors rather
                          than metric distances for biologically-inspired collective
                          motion.

- ThermalPenalty        : Enforces thermal safety constraints using Kreisselmeier-
                          Steinhauser soft penalties, providing smooth gradient-based
                          corrections without requiring optimization solvers.
"""
from __future__                   import annotations
from .schemas                     import *
from hydra_zen                    import builds, make_config
from thermur.imitation.controller import MurmurationController, ThermalPenalty
from typing                       import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from hydra_zen.typing import Builds


CONTROLLER_USER_CONFIG = make_config(
    mmm    = MurmurationModel(),
    safety = SafetyModel()
)

CONTROLLER_SYSTEM_BUILDS: dict[str, type[Builds[Any]]] = {

    "murmuration": builds(
        MurmurationController,
        mmm                     = "${controller.mmm}",
        penalty                 = "${_system.thermal_penalty}",
        safety                  = "${controller.safety}",
        populate_full_signature = True
    ),

    "thermal_penalty": builds(
        ThermalPenalty,
        safety                  = "${controller.safety}",
        populate_full_signature = True
    )

}
