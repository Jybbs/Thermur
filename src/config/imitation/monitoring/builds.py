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
from __future__                   import annotations
from .schemas                     import *
from hydra_zen                    import builds, make_config
from thermur.imitation.monitoring import EventLogger, MetricsCollector
from typing                       import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from hydra_zen.typing import Builds


MONITORING_USER_CONFIG = make_config(
    events  = EventsModel(),
    metrics = MetricsModel()
)

MONITORING_SYSTEM_BUILDS: dict[str, type[Builds[Any]]] = {
    "collector": builds(
        MetricsCollector,
        bounds_max              = "${simulation.physics.bounds_max}",
        gravity                 = "${simulation.physics.gravity}",
        metrics                 = "${monitoring.metrics}",
        mmm                     = "${controller.mmm}",
        safety                  = "${controller.safety}",
        populate_full_signature = True
    ),

    "events": builds(
        EventLogger,
        events                  = "${monitoring.events}",
        safety                  = "${controller.safety}",
        populate_full_signature = True
    )
}
