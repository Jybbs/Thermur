"""
Pydantic models for monitoring and evaluation configuration.

This module defines configuration schemas for the comprehensive monitoring
system, including metric collection, event logging, and performance tracking.
The models configure both aggregate metrics and individual agent behavior
monitoring throughout training and evaluation.
"""
from pydantic import BaseModel, Field, PositiveFloat, PositiveInt
from typing   import Any


class MonitoringModel(BaseModel, extra="forbid"):
    """
    Unified monitoring configuration for metrics and event logging.
    
    Consolidates all monitoring parameters into a single model that
    configures metric collection, event detection, and performance
    tracking throughout training and evaluation.
    """
    color_temp_max: PositiveFloat = Field(
        default     = 475.0,
        description = (
            "Maximum temperature for color mapping in Kelvin. Values above this "
            "are clamped to maximum intensity (red in heat colormap)."
        )
    )
    color_temp_min: PositiveFloat = Field(
        default     = 275.0,
        description = (
            "Minimum temperature for color mapping in Kelvin. Values below this "
            "are clamped to minimum intensity (blue in heat colormap)."
        )
    )
    event_sample_every: PositiveInt = Field(
        default     = 100,
        description = (
            "Global steps between detailed event sampling. Controls how often "
            "detailed event data is pushed to logging tables."
        )
    )
    event_types: dict[str, dict[str, Any]] = Field(
        default     = {
            "cbf_activation": {
                "columns"     : ["agent_id", "temperature", "control_diff", "safety_margin"],
                "name"        : "CBF Activations",
                "rate_metric" : "cbf_activation_rate"
            },
            "near_miss": {
                "columns"     : ["agent_id", "temperature", "margin", "position"],
                "name"        : "Near Misses",
                "rate_metric" : "near_miss_rate"
            },
            "thermal_violation": {
                "columns"     : ["agent_id", "temperature", "excess", "position"],
                "name"        : "Thermal Violations",
                "rate_metric" : "thermal_violation_rate"
            },
            "topology_change": {
                "columns"     : ["agent_id", "neighbors_added", "neighbors_lost", "neighbor_count"],
                "name"        : "Topology Changes",
                "rate_metric" : "topology_change_rate"
            }
        },
        description = (
            "Event type definitions for the monitoring system. Maps event type "
            "identifiers to their configuration including columns, display name, "
            "and metric names for rate tracking."
        )
    )
    legibility_grid_size: PositiveInt = Field(
        default     = 64,
        description = (
            "Resolution of the 2D grid for rendering velocity fields. Higher values "
            "provide more detail but increase computation cost."
        )
    )
    legibility_kernel_size: PositiveInt = Field(
        default     = 11,
        description = (
            "Size of the Gaussian kernel for SSIM computation. Must be odd. "
            "Larger kernels consider broader spatial context."
        )
    )
    legibility_sigma: PositiveFloat = Field(
        default     = 2.0,
        description = (
            "Standard deviation for Gaussian kernel in KDE when rendering velocity "
            "fields. Controls smoothness of the rendered field."
        )
    )
    power_exponent: PositiveFloat = Field(
        default     = 1.5,
        description = (
            "Exponent k in power model P ∝ ||u_safe - g||^k. Typically 1.5 for "
            "quadrotors based on momentum theory."
        )
    )
    prefix: str = Field(
        default     = "events/",
        description = (
            "Prefix for event rate metrics in the logging namespace. Helps organize "
            "metrics in tools like Weights & Biases."
        )
    )