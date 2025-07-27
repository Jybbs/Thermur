"""
Monitoring domain stores for hydra-zen configuration.

This module provides store-based configurations for monitoring components
using simplified domain-level groups with minimal presets.
"""
from hydra_zen import store, builds
from thermur.imitation.monitoring import EventLogger, MetricsCollector

# Import schemas for validation
from . import EventsModel, MetricsModel

# Pre-configure group for all monitoring configs
monitoring = store(group="monitoring")

@monitoring(name="default")
def default():
    """
    Standard monitoring configuration.
    """
    # Use Pydantic models for validation and defaults
    events = EventsModel()
    metrics = MetricsModel()
    
    return dict(
        events=builds(EventLogger,
            activation_tolerance=events.activation_tolerance,
            max_temperature=events.max_temperature
        ),
        
        collector=builds(MetricsCollector,
            # Metrics configuration
            log_every_n_steps=metrics.log_every_n_steps,
            logging_interval=metrics.logging_interval,
            enable_model_summary=metrics.enable_model_summary,
            enable_progress_bar=metrics.enable_progress_bar,
            profiler=metrics.profiler,
            
            # Visual metrics parameters
            legibility_grid_size=metrics.legibility_grid_size,
            legibility_kernel_size=metrics.legibility_kernel_size,
            legibility_sigma=metrics.legibility_sigma,
            color_temp_min=metrics.color_temp_min,
            color_temp_max=metrics.color_temp_max,
            power_exponent=metrics.power_exponent,
            
            # Runtime parameters from environment
            max_temperature=events.max_temperature,
            bounds_max="${simulation.env.bounds_max}",
            gravity="${simulation.env.gravity}"
        )
    )

@monitoring(name="debug")
def debug():
    """
    Debug monitoring configuration.
    """
    # Debug configurations with overrides
    events = EventsModel(
        activation_tolerance=5.0,  # Higher tolerance
        max_temperature=500.0  # Higher threshold
    )
    metrics = MetricsModel(
        log_every_n_steps=1,  # Log every step
        enable_model_summary=False,  # Skip summary
        enable_progress_bar=True,  # Keep progress
        profiler=None  # No profiling
    )
    
    return dict(
        events=builds(EventLogger,
            activation_tolerance=events.activation_tolerance,
            max_temperature=events.max_temperature
        ),
        
        collector=builds(MetricsCollector,
            # Simplified metrics for debugging
            log_every_n_steps=metrics.log_every_n_steps,
            logging_interval=metrics.logging_interval,
            enable_model_summary=metrics.enable_model_summary,
            enable_progress_bar=metrics.enable_progress_bar,
            profiler=None,  # No profiling in debug
            
            # Skip visual metrics for speed
            legibility_grid_size=None,
            color_temp_min=metrics.color_temp_min,
            color_temp_max=metrics.color_temp_max,
            
            # Runtime parameters
            max_temperature=events.max_temperature,
            bounds_max="${simulation.env.bounds_max}",
            gravity="${simulation.env.gravity}"
        )
    )