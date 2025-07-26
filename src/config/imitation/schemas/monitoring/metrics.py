"""
Metrics collection configuration schemas.

This module defines configuration for the comprehensive metrics collection
system used during training and evaluation.
"""
from pydantic import BaseModel, Field, PositiveFloat, PositiveInt
from typing   import Literal, Optional


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
    enable_model_summary: bool = Field(
        default     = True,
        description = (
            "Display comprehensive model architecture summary including parameter counts "
            "and layer shapes before training begins for architecture verification."
        )
    )
    enable_progress_bar: bool = Field(
        default     = True,
        description = (
            "Show real-time progress bar with loss metrics during training epochs "
            "for visual feedback on training progression and convergence."
        )
    )
    legibility_grid_size: PositiveInt = Field(
        default     = 64,
        description = (
            "Resolution of 2D grid for rendering velocity fields in legibility metrics, "
            "higher values provide more detail but increase computation cost."
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
    log_every_n_steps: PositiveInt = Field(
        default     = 50,
        description = (
            "Frequency of metric logging to track training progress, lower values "
            "provide more granular tracking but increase logging overhead."
        )
    )
    logging_interval: Literal["step", "epoch"] = Field(
        default     = "step",
        description = (
            "Interval for learning rate logging to monitor optimization schedule, "
            "choose 'step' for fine-grained tracking or 'epoch' for overview."
        )
    )
    power_exponent: PositiveFloat = Field(
        default     = 1.5,
        description = (
            "Exponent k in power consumption model P ∝ ||u||^k for energy metrics, "
            "typically 1.5 for quadrotors based on momentum theory analysis."
        )
    )
    profiler: Optional[Literal["simple", "advanced"]] = Field(
        default     = None,
        description = (
            "PyTorch Lightning profiler for performance analysis, 'simple' tracks basic "
            "metrics while 'advanced' provides detailed Chrome tracing output."
        )
    )