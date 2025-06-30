"""
Hydra-zen configuration factories for imitation learning components.

This package provides builders that create Hydra-compatible configurations
for instantiating components needed for imitation learning training.
"""
from .collector     import build_collector
from .flocking      import build_flocking_controller
from .loss          import build_loss
from .monitoring    import build_monitoring
from .optimizer     import build_optimizer
from .policy        import build_policy
from .replay        import build_replay_buffer
from .safety        import build_safety_filter, build_thermal_barrier
from .simulation    import build_data_source, build_simulation
from .swarm         import build_action_spec, build_observation_spec
from .visualization import build_visualizer

__all__ = [
    "build_action_spec",
    "build_collector",
    "build_data_source",
    "build_flocking_controller",
    "build_loss",
    "build_monitoring",
    "build_observation_spec",
    "build_optimizer",
    "build_policy",
    "build_replay_buffer",
    "build_safety_filter",
    "build_simulation",
    "build_thermal_barrier",
    "build_visualizer",
]