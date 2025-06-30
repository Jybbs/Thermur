"""
Training configuration schemas.

This module defines Pydantic models for training hyperparameters, data collection,
and optimization settings used throughout the imitation learning pipeline.
"""
from pydantic import BaseModel, Field, NonNegativeFloat, NonNegativeInt, PositiveFloat, PositiveInt
from typing   import Literal


class TrainingConfig(BaseModel, extra="forbid"):
    """
    Unified configuration for imitation learning training.
    
    These parameters control the training loop, optimization process, and 
    experiment tracking. The configuration is designed to be overrideable via
    CLI using Hydra's interpolation system.
    """
    # Device configuration
    device: Literal["cpu", "cuda", "mps"] = Field(
        default     = "cpu",
        description = "Compute device for training (CPU, CUDA GPU, or Apple Silicon)."
    )
    
    # Core hyperparameters
    learning_rate: PositiveFloat = Field(
        default     = 3e-4,
        description = "Learning rate α for the AdamW optimizer."
    )
    weight_decay: NonNegativeFloat = Field(
        default     = 1e-5,
        description = "L2 regularization coefficient for weight decay."
    )
    seed: int = Field(
        default     = 42,
        description = "Random seed for reproducible training runs."
    )
    
    # Training loop parameters
    total_frames: PositiveInt = Field(
        default     = 200_000,
        description = "Total environment frames T to collect during training."
    )
    frames_per_batch: PositiveInt = Field(
        default     = 1024,
        description = "Number of frames to collect per training iteration."
    )
    log_interval: PositiveInt = Field(
        default     = 1000,
        description = "Frequency (in frames) for logging training metrics."
    )
    
    # Checkpointing
    checkpoint_interval: PositiveInt = Field(
        default     = 25_000,
        description = "Frequency (in frames) for saving model checkpoints."
    )
    checkpoint_path: str = Field(
        default     = "checkpoints/",
        description = "Directory path for saving training checkpoints."
    )


class DataConfig(BaseModel, extra="forbid"):
    """
    Configuration for experience replay and data handling.
    
    Controls the replay buffer size and batching strategy for efficient
    training data management during behavioral cloning.
    """
    buffer_size: PositiveInt = Field(
        default     = 50_000,
        description = "Maximum number of transitions to store in replay buffer."
    )
    batch_size: PositiveInt = Field(
        default     = 256,
        description = "Number of transitions per training batch."
    )
    prefetch: NonNegativeInt = Field(
        default     = 8,
        description = "Number of batches to prefetch for GPU efficiency."
    )


class GNNConfig(BaseModel, extra="forbid"):
    """
    Defines the architecture of the Graph Neural Network (GNN) policy, π_θ.

    This policy is trained to imitate the expert controller. At each step, it
    performs message passing where each node aggregates features from its
    neighbors (𝐚ᵢ = Σ hⱼ) and updates its own hidden state (hᵢ' = GRU(hᵢ, 𝐚ᵢ)).
    """
    activation: Literal["relu", "silu", "tanh"] = Field(
        default     = "silu",
        description = (
            "The nonlinearity used in the GNN's multi-layer perceptrons (MLPs)."
        )
    )
    hidden_dim: PositiveInt = Field(
        default     = 64,
        description = "Dimensionality of the hidden node embeddings and messages.",
    )
    input_dim: PositiveInt = Field(
        default     = 11,  # position(3) + velocity(3) + temperature(1) + temp_grad(3) + energy(1)
        description = "Dimensionality of the input node features (concatenated state vector).",
    )
    num_layers: PositiveInt = Field(
        default     = 3,
        description = (
            "Number of GNN message-passing layers. More layers increase the "
            "agent's receptive field but also computational cost."
        )
    )
    output_dim: PositiveInt = Field(
        default     = 3,
        description = "Dimensionality of the output action (spatial dimensions).",
    )