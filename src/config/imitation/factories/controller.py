"""
Hydra-zen builders for flocking components.

This module defines configuration builders for the expert flocking controller
and related components that implement Reynolds rules and thermal-aware behavior.
These builders leverage Pydantic validation through the zen() wrapper.
"""
from config.imitation.schemas.controller import *
from hydra_zen                           import builds, zen
from omegaconf                           import SI
from thermur.imitation.controller        import FlockController


build_controller = builds(
    FlockController,
    control                 = zen(ControllerModel),
    flock                   = SI("${flock}"),
    safety                  = SI("${safety}"),
    populate_full_signature = True,
    zen_dataclass           = {
        "module"   : "src.configs.imitation.factories.flocking",
        "cls_name" : "FlockingControllerBuild"
    }
)
"""
Builder for expert demonstration controller.

Orchestrates multi-agent flocking behavior using Reynolds' three rules (cohesion,
separation, alignment) augmented with thermal-aware navigation. Generates optimal
trajectories by computing potential field gradients that balance social forces
with environmental hazard avoidance. The controller's output serves as ground
truth for imitation learning, teaching the neural policy safe collective behavior.
"""

build_flock = builds(
    FlockModel,
    populate_full_signature = True,
    zen_dataclass           = {
        "module"   : "src.configs.imitation.factories.controller",
        "cls_name" : "FlockBuild"
    }
)
"""
Builder for multi-agent flock configuration.

Defines the collective properties of the drone swarm including population size,
communication topology parameters, spatial operating dimensions, thermal tolerance
limits, and heat dissipation dynamics. The communication range determines dynamic
graph connectivity, while thermal parameters establish safety boundaries for the
Control Barrier Function that prevents agent overheating in wildfire scenarios.
"""

build_safety = builds(
    SafetyModel,
    populate_full_signature = True,
    zen_dataclass           = {
        "module"   : "src.configs.imitation.factories.controller",
        "cls_name" : "SafetyBuild"
    }
)
"""
Builder for Control Barrier Function safety configuration.

Specifies parameters for the real-time safety filter that modifies control actions
to guarantee thermal safety. Includes CBF relaxation terms (α), activation thresholds,
quadratic program solver settings (tolerance, iterations), and fallback strategies
for numerical edge cases. Critical for maintaining hard safety guarantees during
both expert demonstration collection and learned policy deployment.
"""
