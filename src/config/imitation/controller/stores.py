"""
Controller configuration stores using hydra-zen.

This module provides store-based configurations for control algorithms
using hydra-zen's decorator pattern. Each component is registered as a separate
build that can be referenced and overridden independently via Hydra's CLI.
"""
from .schemas                     import *
from config.utils.zen             import store, thermur_build, thermur_make_all
from thermur.imitation.controller import ExpertController, SafetyFilter

controller = store()(group="controller")
expert     = ExpertModel()
flock      = FlockModel()
safety     = SafetyModel()
thresholds = ThresholdsModel()


@controller(name="expert")
def expert_build():
    """
    Builder for the expert flocking controller.
    
    Creates a physics-based controller implementing Reynolds flocking rules
    with thermal-aware potential fields. This controller generates optimal
    trajectories for imitation learning by combining:
    
    - Cohesion forces toward neighborhood center of mass
    - Separation forces to prevent collisions
    - Alignment forces to match neighbor velocities
    - Thermal repulsion from high-temperature regions
    
    The controller uses Control Barrier Functions to ensure all commands
    respect thermal safety constraints.
    """
    return thermur_build(
        ExpertController,
        expert        = expert,
        flock         = flock,
        safety_filter = "${controller.safety}",
        thresholds    = thresholds
    )

@controller(name="safety")
def safety_build():
    """
    Builder for the safety filter component.
    
    Creates a Control Barrier Function (CBF) based safety filter that
    ensures all control commands respect thermal constraints. The filter
    solves a quadratic program at each timestep to find the minimally
    invasive safe control:
    
        u* = argmin ||u - u_nom||²
        s.t. ∇h(x)·u ≥ -α·h(x)
    
    where h(x) = T_max - T(x) defines the thermal safety boundary.
    """
    return thermur_build(
        SafetyFilter,
        flock      = flock,
        safety     = safety,
        thresholds = thresholds
    )

@controller(name="thresholds")
def thresholds_build():
    """
    Builder for safety threshold configuration.
    
    Provides centralized threshold values that are used across multiple
    domains to ensure consistency in thermal safety limits and control
    intervention detection.
    """
    return thresholds

thermur_make_all(controller)