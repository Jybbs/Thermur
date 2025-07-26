"""
Hydra-zen builders for monitoring and evaluation components.

This module provides configuration builders for the comprehensive monitoring
system, including metric collection and event logging. These builders integrate
with PyTorch Lightning's logging infrastructure and Weights & Biases for
tracking training progress and agent behavior.
"""
from config.imitation.schemas.monitoring import *
from hydra_zen                           import builds, zen
from omegaconf                           import SI
from thermur.imitation.monitoring        import EventLogger, MetricsCollector


build_events = builds(
    EventLogger,
    activation_tolerance    = SI("${safety.activation_tolerance}"),
    max_temperature         = SI("${flock.max_temperature}"),
    events                  = zen(EventsModel),
    populate_full_signature = True,
    zen_dataclass           = {
        "module"   : "src.configs.imitation.factories.monitoring",
        "cls_name" : "EventsBuild"
    }
)
"""
Builder for critical event detection and logging.

Monitors the flock for safety-critical events including Control Barrier Function
activations (when agents approach thermal limits), thermal violations (overheating),
near-miss incidents, and dynamic topology changes in the communication graph.
Events are logged as rates for dashboards and sampled instances for debugging.
"""

build_metrics = builds(
    MetricsCollector,
    bounds_max              = SI("${physics.bounds_max}"),
    gravity                 = SI("${physics.gravity}"),
    max_temperature         = SI("${flock.max_temperature}"),
    metrics                 = zen(MetricsModel),
    output_dim              = SI("${architecture.output_dim}"),
    populate_full_signature = True,
    zen_dataclass           = {
        "module"   : "src.configs.imitation.factories.monitoring",
        "cls_name" : "MetricsBuild"
    }
)
"""
Builder for comprehensive training metrics.

Centralizes all metric computation including behavioral cloning losses (MSE, MAE),
physical realism metrics (energy consistency, control smoothness), visual quality
measures (SSIM, TVR), and aggregate performance indicators. Integrates seamlessly
with PyTorch Lightning's logging infrastructure and Weights & Biases dashboards.
"""

