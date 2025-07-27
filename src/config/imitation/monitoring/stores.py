"""
Monitoring domain stores for hydra-zen configuration.

This module provides store-based configurations for monitoring components
using simplified domain-level groups with minimal presets.
"""
from hydra_zen                         import store as create_store, builds
from thermur.imitation.monitoring      import EventMonitor, MetricsCollector

# Import schemas from __init__ for clean imports
from . import EventsModel, MetricsModel

# Create domain store
store = create_store()

@store(group="monitoring", name="default")
def default():
    """
    Standard monitoring configuration.
    
    Provides default configurations for comprehensive metrics collection
    and event tracking during training.
    """
    # Validate configurations with Pydantic
    events  = EventsModel()
    metrics = MetricsModel()
    
    return {
        # Event monitoring
        "events": builds(
            EventMonitor,
            activation_tolerance = events.activation_tolerance,
            max_temperature     = events.max_temperature
        ),
        
        # Metrics collection
        "collector": builds(
            MetricsCollector,
            bounds_max      = "${physics.bounds_max}",
            gravity         = "${physics.gravity}",
            max_temperature = events.max_temperature,
            output_dim      = 3,  # 3D velocity commands
            metrics         = metrics.model_dump()
        ),
        
        # Export individual configs for access
        "events_config"  : events.model_dump(),
        "metrics_config" : metrics.model_dump()
    }

@store(group="monitoring", name="debug")
def debug():
    """
    Debug monitoring configuration.
    
    Minimal configuration for rapid testing with reduced logging
    and simplified metrics collection.
    """
    # Minimal configurations for debugging
    events = EventsModel(
        activation_tolerance = 5.0,    # Higher tolerance
        max_temperature     = 500.0    # Higher threshold
    )
    metrics = MetricsModel(
        log_every_n_steps    = 1,      # Log every step
        enable_model_summary = False,   # Skip summary
        enable_progress_bar  = True,    # Keep progress
        profiler            = None      # No profiling
    )
    
    return {
        # Simplified event monitoring
        "events": builds(
            EventMonitor,
            activation_tolerance = events.activation_tolerance,
            max_temperature     = events.max_temperature,
            logging_only        = True  # Just log, no stats
        ),
        
        # Lightweight metrics
        "collector": builds(
            MetricsCollector,
            bounds_max      = [50.0, 50.0, 20.0],  # Default bounds
            gravity         = 9.81,
            max_temperature = events.max_temperature,
            output_dim      = 3,
            metrics         = metrics.model_dump(),
            compute_visual  = False  # Skip visual metrics
        ),
        
        # Export configs
        "events_config"  : events.model_dump(),
        "metrics_config" : metrics.model_dump()
    }