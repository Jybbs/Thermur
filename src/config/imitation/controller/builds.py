"""
Controller domain builds for hydra-zen configuration.

This module provides pre-built components for the controller domain:

- ExpertController : Coordinates multi-agent behavior using Reynolds flocking rules
                     (separation, alignment, cohesion) with safety filtering to
                     maintain minimum separation distances between agents.

- SafetyFilter     : Monitors agent positions and prevents collisions by overriding
                     control commands when agents get too close. Uses configurable
                     thresholds for minimum separation distances.
"""
from __future__                   import annotations
from .schemas                     import *
from hydra_zen                    import builds, make_config
from thermur.imitation.controller import ExpertController, SafetyFilter
from typing                       import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from hydra_zen.typing import Builds


CONTROLLER_USER_CONFIG = make_config(
    expert     = ExpertModel(),
    flock      = FlockModel(),
    safety     = SafetyModel(),
    thresholds = ThresholdsModel()
)

CONTROLLER_SYSTEM_BUILDS: dict[str, type[Builds[Any]]] = {
    "expert_controller": builds(
        ExpertController,
        expert                  = "${controller.expert}",
        flock                   = "${controller.flock}",
        safety_filter           = "${_system.safety_filter}",
        thresholds              = "${controller.thresholds}",
        zen_partial             = True,
        populate_full_signature = True
    ),

    "safety_filter": builds(
        SafetyFilter,
        flock                   = "${controller.flock}",
        safety                  = "${controller.safety}",
        thresholds              = "${controller.thresholds}",
        zen_partial             = True,
        populate_full_signature = True
    )
}
