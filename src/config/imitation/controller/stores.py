"""
Controller domain stores for hydra-zen configuration.

This module provides store-based configurations for control algorithms
using simplified domain-level groups with minimal presets.
"""
from hydra_zen                     import store as create_store, builds
from thermur.imitation.controller  import ReynoldsController, SafetyFilter

# Import schemas from __init__ for clean imports
from . import (
    ControllerModel,
    FlockModel,
    SafetyModel
)

# Create domain store
store = create_store()

@store(group="controller", name="default")
def default():
    """
    Standard controller configuration.
    
    Provides default configurations for the Reynolds flocking controller
    with Control Barrier Function safety filtering.
    """
    # Validate configurations with Pydantic
    control = ControllerModel()
    flock   = FlockModel()
    safety  = SafetyModel()
    
    return {
        # Controller
        "controller": builds(
            ReynoldsController,
            control = control.model_dump(),
            flock   = flock.model_dump(),
            safety  = safety.model_dump()
        ),
        
        # Safety filter (optional)
        "safety_filter": builds(
            SafetyFilter,
            safety_config = safety.model_dump()
        ),
        
        # Export individual configs for access
        "control" : control.model_dump(),
        "flock"   : flock.model_dump(),
        "safety"  : safety.model_dump()
    }

@store(group="controller", name="debug")
def debug():
    """
    Debug controller configuration.
    
    Minimal configuration for rapid testing with small flock
    and relaxed safety constraints.
    """
    # Minimal configurations for debugging
    control = ControllerModel(
        w_thermal = 1.0,  # Reduced thermal weight
        epsilon   = 1e-6
    )
    flock = FlockModel(
        agent_count            = 3,
        communication_range    = 10.0,
        formation_scale_factor = 0.3
    )
    safety = SafetyModel(
        cbf_alpha            = 1.0,  # Less aggressive safety
        activation_tolerance = 5.0,   # Higher tolerance
        log_violations       = False  # Reduce logging
    )
    
    return {
        # Simplified controller
        "controller": builds(
            ReynoldsController,
            control = control.model_dump(),
            flock   = flock.model_dump(),
            safety  = safety.model_dump()
        ),
        
        # No safety filter for debugging
        "safety_filter": None,
        
        # Export configs
        "control" : control.model_dump(),
        "flock"   : flock.model_dump(),
        "safety"  : safety.model_dump()
    }