"""
Hydra-zen configuration factories for imitation learning.

This package provides builders organized by functional area, matching
the structure of the business logic components they configure.
"""
from .data          import build_collector, build_replay_buffer
from .flocking      import build_flocking_controller
from .safety        import build_safety_filter, build_thermal_barrier
from .simulation    import build_data_source, build_simulation
from .swarm         import build_action_spec, build_observation_spec
from .imitation     import build_loss, build_optimizer, build_policy
from .visualization import build_visualizer

__all__ = [
    # Data
    "build_collector",
    "build_replay_buffer",
    
    # Expert and safety
    "build_flocking_controller", 
    "build_safety_filter",
    "build_thermal_barrier",
    
    # Environment
    "build_action_spec",
    "build_data_source",
    "build_observation_spec",
    "build_simulation",
    
    # Training
    "build_loss",
    "build_optimizer",
    "build_policy",
    
    # Visualization
    "build_visualizer",
]