"""
Training and optimization models.

This module defines the Pydantic models for training parameters,
data collection, and replay buffer configuration.
"""
from pydantic import BaseModel, Field, NonNegativeFloat, NonNegativeInt, PositiveFloat, PositiveInt
from typing   import Literal


class CheckpointModel(BaseModel, extra="forbid"):
    """
    Parameters for saving model checkpoints during training.
    """
    interval: PositiveInt = Field(
        default     = 25_000,
        description = "The frequency (in frames) at which to save a model checkpoint."
    )
    path: str = Field(
        default     = "checkpoints/",
        description = "Directory where model checkpoints will be saved."
    )


class CollectorModel(BaseModel, extra="forbid"):
    """
    Parameters for the torchrl SyncDataCollector.

    This configures the data collection loop, which interacts with the
    environment to gather agent experiences.
    """
    frames_per_batch: PositiveInt = Field(
        default     = 1024,
        description = "Number of frames (agent steps) to collect per batch."
    )
    total_frames: PositiveInt = Field(
        default     = 200_000,
        description = "The total number of environment frames to collect for the training run."
    )
    

class HyperparameterModel(BaseModel, extra="forbid"):
    """
    Parameters for the training and optimization loop.

    These settings govern the imitation learning process (behavioral cloning),
    including the optimizer, batching, and total training duration. The loss
    function will minimize the Mean Squared Error (MSE) between the GNN's
    output and the expert's actions: 
    
        L_imitation = ||π_θ(s) - 𝐮_nom||².
    """
    device: Literal["cpu", "cuda", "mps"] = Field(
        default     = "cpu",
        description = "The compute device for training."
    )
    learning_rate: PositiveFloat = Field(
        default     = 3e-4,
        description = "The learning rate for the AdamW optimizer."
    )
    log_interval: PositiveInt = Field(
        default     = 1000,
        description = "The frequency (in frames) at which to log training metrics."
    )
    seed: int = Field(
        default     = 42,
        description = "The global random seed for ensuring reproducibility."
    )
    weight_decay: NonNegativeFloat = Field(
        default     = 1e-5,
        description = "Weight decay (L2 penalty) for the AdamW optimizer."
    )


class StorageModel(BaseModel, extra="forbid"):
    """
    Parameters for the LazyTensorStorage used by the replay buffer.
    """
    max_size: PositiveInt = Field(
        default     = 50_000,
        description = "The maximum number of agent experiences to store in the buffer."
    )


class ReplayBufferModel(BaseModel, extra="forbid"):
    """
    Parameters for the torchrl TensorDictReplayBuffer.

    This configures the experience replay buffer, which stores transitions
    collected from the environment for training.
    """
    batch_size: PositiveInt = Field(
        default     = 256,
        description = "The number of agent experiences per training batch."
    )
    prefetch: NonNegativeInt = Field(
        default     = 8,
        description = "Number of batches to prefetch for training to hide data loading latency."
    )
