"""
Learning system model.

This module defines the unified configuration for imitation learning,
including training hyperparameters, data handling, and network architecture.
"""
from pathlib  import Path
from pydantic import BaseModel, Field, NonNegativeFloat
from pydantic import NonNegativeInt, PositiveFloat, PositiveInt
from typing   import Literal


class LearningModel(BaseModel, extra="forbid"):
    """
    Unified configuration for imitation learning system.
    
    Combines training hyperparameters, data collection settings, and
    GNN architecture parameters into a single coherent configuration
    for behavioral cloning from expert demonstrations.
    
    The learning process minimizes the imitation loss:
    
        L_imitation = ||π_θ(s) - u_nom||²
    
    where π_θ is the learned GNN policy and π* is the expert controller.
    
    The GNN architecture processes the dynamic graph G_t = (V, E_t) through
    message passing layers, where each layer performs:
    
        h_i^(l+1) = GRU(h_i^(l), Σ_j∈N(i) MLP(h_j^(l)))
    
    This allows the policy to learn decentralized control while respecting
    the communication topology.
    """
    activation: Literal["relu", "silu", "tanh"] = Field(
        default     = "silu",
        description = "Nonlinearity σ(·) used in GNN message passing MLPs."
    )
    batch_size: PositiveInt = Field(
        default     = 256,
        description = "Number of transitions B per training batch."
    )
    buffer_size: PositiveInt = Field(
        default     = 50_000,
        description = "Maximum transitions |𝒟| to store in replay buffer."
    )
    checkpoint_interval: PositiveInt = Field(
        default     = 25_000,
        description = "Frequency (in frames) for saving model checkpoints."
    )
    checkpoint_path: Path = Field(
        default     = Path("checkpoints"),
        description = "Directory path for saving training checkpoints."
    )
    device: Literal["cpu", "cuda", "mps"] = Field(
        default     = "cpu",
        description = "Compute device for training (CPU, CUDA GPU, or Apple Silicon)."
    )
    frames_per_batch: PositiveInt = Field(
        default     = 1024,
        description = "Number of frames N_batch to collect per training iteration."
    )
    hidden_dim: PositiveInt = Field(
        default     = 64,
        description = "Dimensionality d_h of hidden node embeddings h_i."
    )
    input_dim: PositiveInt = Field(
        default     = 11,
        description = (
            "Dimensionality d_in of input features "
            "[position(3) + velocity(3) + temperature(1) + temp_grad(3) + energy(1)]."
        )
    )
    learning_rate: PositiveFloat = Field(
        default     = 3e-4,
        description = "Learning rate α for the AdamW optimizer."
    )
    log_interval: PositiveInt = Field(
        default     = 1000,
        description = "Frequency (in frames) for logging training metrics."
    )
    num_layers: PositiveInt = Field(
        default     = 3,
        description = (
            "Number of message passing layers L. Receptive field grows as "
            "r = L · r_comm where r_comm is communication range."
        )
    )
    output_dim: PositiveInt = Field(
        default     = 3,
        description = "Dimensionality d_out of output actions (velocity commands)."
    )
    prefetch: NonNegativeInt = Field(
        default     = 8,
        description = "Number of batches to prefetch for GPU efficiency."
    )
    seed: NonNegativeInt = Field(
        default     = 42,
        description = "Random seed for reproducible training runs."
    )
    total_frames: PositiveInt = Field(
        default     = 200_000,
        description = "Total environment frames T to collect during training."
    )
    weight_decay: NonNegativeFloat = Field(
        default     = 1e-5,
        description = "L2 regularization coefficient λ for weight decay."
    )