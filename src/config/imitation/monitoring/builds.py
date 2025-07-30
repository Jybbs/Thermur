"""
Monitoring domain builds for hydra-zen configuration.

This module provides pre-built components for metrics tracking and event logging
during training:

- MetricsCollector : Aggregates and tracks training metrics including loss values,
                     learning rates, gradients, and custom performance indicators.
                     Integrates with PyTorch Lightning's metric system.

- EventLogger      : Captures and logs training events such as epoch boundaries,
                     validation runs, checkpoint saves, and early stopping triggers.
"""
from .schemas                     import *
from hydra_zen                    import builds, make_config
from thermur.imitation.monitoring import EventLogger, MetricsCollector


MONITORING_USER_CONFIG = make_config(
    events  = EventsModel(),
    metrics = MetricsModel() 
)

MONITORING_SYSTEM_BUILDS = {
    "collector": builds(
        MetricsCollector,
        metrics                 = "${monitoring.metrics}",
        zen_partial             = True,
        populate_full_signature = True
    ),
    
    "events": builds(
        EventLogger,
        events                  = "${monitoring.events}",
        zen_partial             = True,
        populate_full_signature = True
    )
}