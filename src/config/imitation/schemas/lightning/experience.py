"""
Experience data handling configuration.

This module defines configuration models for experience data collection,
storage, and sampling during imitation learning from expert demonstrations.
"""
from pydantic import BaseModel, Field, NonNegativeInt, PositiveInt


class ExperienceModel(BaseModel, extra="forbid"):
    """
    Experience data handling and batching configuration.
    
    Manages how experience data is collected, stored, and sampled during
    imitation learning from expert demonstrations.
    """
    batch_size: PositiveInt = Field(
        default     = 256,
        description = (
            "Number of state-action transitions B sampled per gradient update, "
            "balancing computational efficiency with gradient variance."
        )
    )
    buffer_size: PositiveInt = Field(
        default     = 50_000,
        description = (
            "Maximum trajectory transitions |𝒟| stored in circular replay buffer "
            "before oldest experiences are overwritten with new demonstrations."
        )
    )
    frames_per_batch: PositiveInt = Field(
        default     = 1024,
        description = (
            "Environment steps N_batch collected between training updates, controlling "
            "the ratio of environment interaction to gradient computation."
        )
    )
    max_frames_per_traj: int = Field(
        default     = -1,
        description = (
            "Maximum frames per trajectory before episode reset. Use -1 for infinite "
            "episodes that only reset when environment signals done."
        )
    )
    prefetch: NonNegativeInt = Field(
        default     = 8,
        description = (
            "Concurrent batches loaded in background threads, hiding I/O latency "
            "and maintaining GPU utilization during asynchronous data loading."
        )
    )
    total_frames: PositiveInt = Field(
        default     = 200_000,
        description = (
            "Total environment interactions T over entire training run, determining "
            "sample efficiency and final policy performance convergence."
        )
    )