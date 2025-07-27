"""
Monitoring configuration stores using hydra-zen.

This module provides store-based configurations for monitoring components
using hydra-zen's decorator pattern. Each component is registered as a separate
build that can be referenced and overridden independently via Hydra's CLI.
"""
from .schemas                     import *
from config.utils.zen             import store, build  
from thermur.imitation.monitoring import EventLogger, MetricsCollector

monitoring = store(group="monitoring")
events     = EventsModel()
metrics    = MetricsModel()


@monitoring(name="collector")
def collector_build():
    """
    Builder for unified metrics collector.
    
    Creates a centralized metrics collection system that tracks:
    - Imitation learning metrics (MSE, RMSE, MAE, R²)
    - Core evaluation metrics (thermal safety, legibility, cohesion, energy, color)
    - Runtime performance metrics (CBF activations, trajectories)
    
    The collector integrates with PyTorch Lightning's logging system
    and automatically syncs metrics to Weights & Biases.
    """
    return build(
        MetricsCollector,
        bounds_max      = "${simulation.env.bounds_max}",
        gravity         = "${simulation.env.gravity}",
        max_temperature = "${controller.thresholds.max_temperature}",
        metrics         = metrics
    )

@monitoring(name="events")  
def events_build():
    """
    Builder for event detection and logging system.
    
    Creates an event logger that tracks critical simulation events:
    - Thermal violations when agents exceed T_max
    - Near misses when agents approach thermal limits
    - CBF activations when safety filter modifies control
    
    Events are logged as both aggregate rates and detailed tables
    for debugging and analysis.
    """
    return build(
        EventLogger,
        events     = events,
        thresholds = "${controller.thresholds}"
    )