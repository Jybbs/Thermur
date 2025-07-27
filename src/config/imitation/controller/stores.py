"""
Controller domain stores for hydra-zen configuration.

This module provides store-based configurations for control algorithms
using simplified domain-level groups with minimal presets.
"""
from hydra_zen import store, builds
from thermur.imitation.controller import FlockController, SafetyFilter

# Import schemas for validation
from . import (
    ControllerModel,
    FlockModel,
    SafetyModel
)

# Pre-configure group for all controller configs
controller = store(group="controller")

@controller(name="default")
def default():
    """
    Standard controller configuration.
    """
    # Use Pydantic models for validation and defaults
    ctrl = ControllerModel()
    flock = FlockModel()
    safety = SafetyModel()
    
    return dict(
        expert=builds(FlockController,
            # Flocking weights
            w_separation=ctrl.w_separation,
            w_alignment=ctrl.w_alignment,
            w_cohesion=ctrl.w_cohesion,
            w_thermal=ctrl.w_thermal,
            
            # Control parameters
            min_distance=ctrl.min_distance,
            temperature_scaling=ctrl.temperature_scaling,
            gradient_step=ctrl.gradient_step,
            epsilon=ctrl.epsilon,
            
            # Flock configuration
            agent_count=flock.agent_count,
            communication_range=flock.communication_range,
            max_temperature=flock.max_temperature,
            thermal_time_constant=flock.thermal_time_constant,
            
            # Safety parameters
            safety_filter=builds(SafetyFilter,
                cbf_alpha=safety.cbf_alpha,
                activation_tolerance=safety.activation_tolerance,
                log_violations=safety.log_violations,
                qp_eps=safety.qp_eps,
                qp_max_iter=safety.qp_max_iter,
                qp_verbose=safety.qp_verbose,
                qp_on_failure=safety.qp_on_failure
            )
        )
    )

@controller(name="debug")
def debug():
    """
    Debug controller configuration.
    """
    # Debug configurations with overrides
    ctrl = ControllerModel(
        w_thermal=1.0,  # Reduced thermal weight for debugging
        epsilon=1e-6
    )
    flock = FlockModel(
        agent_count=3,
        communication_range=10.0,
        formation_scale_factor=0.3
    )
    safety = SafetyModel(
        cbf_alpha=1.0,  # Less aggressive safety
        activation_tolerance=5.0,  # Higher tolerance
        log_violations=False  # Reduce logging
    )
    
    return dict(
        expert=builds(FlockController,
            # Flocking weights
            w_separation=ctrl.w_separation,
            w_alignment=ctrl.w_alignment,
            w_cohesion=ctrl.w_cohesion,
            w_thermal=ctrl.w_thermal,
            
            # Control parameters
            min_distance=ctrl.min_distance,
            temperature_scaling=ctrl.temperature_scaling,
            gradient_step=ctrl.gradient_step,
            epsilon=ctrl.epsilon,
            
            # Small flock for testing
            agent_count=flock.agent_count,
            communication_range=flock.communication_range,
            max_temperature=flock.max_temperature,
            thermal_time_constant=flock.thermal_time_constant,
            
            # Relaxed safety for debugging
            safety_filter=builds(SafetyFilter,
                cbf_alpha=safety.cbf_alpha,
                activation_tolerance=safety.activation_tolerance,
                log_violations=safety.log_violations,
                qp_eps=safety.qp_eps,
                qp_max_iter=safety.qp_max_iter,
                qp_verbose=safety.qp_verbose,
                qp_on_failure=safety.qp_on_failure
            ) if safety.qp_on_failure != "ignore" else None  # Optionally disable
        )
    )