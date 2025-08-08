"""
Monitoring domain schemas for Pydantic validation.

This module consolidates all monitoring configuration models including
metrics collection, event tracking, and performance monitoring.
"""
from pydantic import BaseModel, Field, PositiveFloat, PositiveInt
from typing   import Literal


class EventsModel(BaseModel, extra="forbid"):
    """
    Configuration for event detection and logging system.

    Defines parameters for detecting and logging critical events during
    simulation, including safety violations and control interventions.
    """
    event_sample_every: PositiveInt = Field(
        default     = 100,
        description = (
            "Frequency of detailed event sampling for W&B table logging, higher values "
            "reduce data volume but may miss important events."
        )
    )


class MetricsModel(BaseModel, extra="forbid"):
    """
    Configuration for metrics collection and performance monitoring.

    Defines parameters for metric computation, logging frequency, and
    visualization settings used by the MetricsCollector.
    """
    color_temp_max: PositiveFloat = Field(
        default     = 475.0,
        description = (
            "Maximum temperature in Kelvin for color mapping visualization, values "
            "above are clamped to maximum intensity in heat colormap."
        )
    )
    color_temp_min: PositiveFloat = Field(
        default     = 275.0,
        description = (
            "Minimum temperature in Kelvin for color mapping visualization, values "
            "below are clamped to minimum intensity in heat colormap."
        )
    )
    info_propagation_max_speed: PositiveFloat = Field(
        default     = 45.0,
        description = (
            "Maximum expected information propagation speed in m/s for murmuration "
            "dynamics, based on empirical observations from Cavagna et al. (2010)."
        )
    )
    info_propagation_min_speed: PositiveFloat = Field(
        default     = 15.0,
        description = (
            "Minimum expected information propagation speed in m/s for murmuration "
            "dynamics, based on empirical observations from Cavagna et al. (2010)."
        )
    )
    info_propagation_time_step: PositiveFloat = Field(
        default     = 0.05,
        description = (
            "Time step in seconds for estimating information propagation velocity "
            "through the flock by tracking velocity change patterns over time."
        )
    )
    legibility_grid_size: PositiveInt = Field(
        default     = 64,
        description = (
            "Resolution of 2D grid for rendering velocity fields in legibility "
            "metrics, higher values provide more detail but increase computation cost."
        )
    )
    legibility_kernel_size: PositiveInt = Field(
        default     = 11,
        description = (
            "Size of Gaussian kernel for SSIM computation in legibility metrics, "
            "must be odd, larger kernels consider broader spatial context."
        )
    )
    legibility_sigma: PositiveFloat = Field(
        default     = 2.0,
        description = (
            "Standard deviation for Gaussian kernel in KDE velocity field rendering, "
            "controls smoothness of the rendered velocity field representation."
        )
    )
    power_exponent: PositiveFloat = Field(
        default     = 1.5,
        description = (
            "Exponent k in power consumption model P ∝ ||u||^k for energy metrics, "
            "typically 1.5 for quadrotors based on momentum theory analysis."
        )
    )
    profiler: bool | Literal["simple", "advanced", "pytorch"] = Field(
        default     = False,
        description = (
            "PyTorch Lightning profiler for performance analysis. False disables "
            "profiling, True uses 'simple' profiler, or specify 'advanced'/'pytorch' "
            "for detailed profiling."
        )
    )
    susceptibility_max: PositiveFloat = Field(
        default     = 20.0,
        description = (
            "Maximum expected susceptibility χ = N·Var[Φ] for maintaining critical "
            "state dynamics, higher values indicate excessive system responsiveness."
        )
    )
    susceptibility_min: PositiveFloat = Field(
        default     = 5.0,
        description = (
            "Minimum expected susceptibility χ = N·Var[Φ] for maintaining critical "
            "state dynamics, lower values indicate insufficient system responsiveness."
        )
    )
