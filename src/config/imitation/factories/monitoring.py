"""
Hydra-zen builders for monitoring and evaluation components.

This module provides configuration builders for the comprehensive monitoring
system, including metric collection and event logging. These builders integrate
with PyTorch Lightning's logging infrastructure and Weights & Biases for
tracking training progress and agent behavior.
"""
from config.imitation.schemas.monitoring  import MonitoringModel
from hydra_zen                            import builds, zen
from omegaconf                            import SI
from thermur.imitation.monitoring.events  import EventLogger
from thermur.imitation.monitoring.metrics import MetricsCollector


build_events = builds(
    EventLogger,
    activation_tolerance    = SI("${safety.activation_tolerance}"),
    max_temperature         = SI("${flock.max_temperature}"),
    monitoring              = zen(MonitoringModel),
    populate_full_signature = True,
    zen_dataclass           = {
        "module"   : "src.configs.imitation.factories.monitoring",
        "cls_name" : "EventsBuild"
    }
)
"""
Builder for event logging system.

Tracks and logs critical events during training including CBF activations,
thermal violations, near misses, and topology changes for debugging and analysis.
"""

build_metrics = builds(
    MetricsCollector,
    bounds_max              = SI("${physics.bounds_max}"),
    gravity                 = SI("${physics.gravity}"),
    max_temperature         = SI("${flock.max_temperature}"),
    monitoring              = zen(MonitoringModel),
    output_dim              = SI("${architecture.output_dim}"),
    populate_full_signature = True,
    zen_dataclass           = {
        "module"   : "src.configs.imitation.factories.monitoring",
        "cls_name" : "MetricsBuild"
    }
)
"""
Builder for centralized metrics collection.

Manages all training and evaluation metrics including imitation learning losses,
core performance metrics, and runtime statistics. Integrates with Lightning's
logging system for comprehensive monitoring.
"""