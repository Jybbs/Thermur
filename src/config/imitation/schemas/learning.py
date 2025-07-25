"""
Learning system models.

This module defines the decomposed configuration models for imitation learning,
including training hyperparameters, data handling, and network architecture.
"""
from pathlib  import Path
from pydantic import BaseModel, Field, NonNegativeFloat
from pydantic import NonNegativeInt, PositiveFloat, PositiveInt
from typing   import Literal, Optional


class ArchitectureModel(BaseModel, extra="forbid"):
    """
    GNN architecture configuration.
    
    Defines the neural network structure for processing the dynamic graph 
    G_t = (V, E_t) through message passing layers, where each layer performs:
    
        h_i^(l+1) = GRU(h_i^(l), Σ_j∈N(i) MLP(h_j^(l)))
    
    This allows the policy to learn decentralized control while respecting
    the communication topology.
    """
    activation: Literal["ReLU", "SiLU", "Tanh"] = Field(
        default     = "SiLU",
        description = "Nonlinearity σ(·) used in GNN message passing MLPs."
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


class CheckpointModel(BaseModel, extra="forbid"):
    """
    Model checkpoint configuration.
    
    Controls how and when model checkpoints are saved during training,
    enabling recovery from interruptions and model selection.
    """
    dirpath: Path = Field(
        default     = Path("checkpoints"),
        description = "Directory path for saving training checkpoints."
    )
    every_n_train_steps: PositiveInt = Field(
        default     = 25_000,
        description = "Frequency in training steps for saving model checkpoints."
    )
    filename: str = Field(
        default     = "checkpoint-{step}",
        description = "Template for checkpoint filenames with step number placeholder."
    )
    save_last: bool = Field(
        default     = True,
        description = (
            "Always save the final model checkpoint at training completion regardless "
            "of whether it achieved the best validation metric."
        )
    )


class ExperienceModel(BaseModel, extra="forbid"):
    """
    Experience data handling and batching configuration.
    
    Manages how experience data is collected, stored, and sampled during
    imitation learning from expert demonstrations.
    """
    batch_size: PositiveInt = Field(
        default     = 256,
        description = "Number of transitions B per training batch."
    )
    buffer_size: PositiveInt = Field(
        default     = 50_000,
        description = "Maximum transitions |𝒟| to store in replay buffer."
    )
    frames_per_batch: PositiveInt = Field(
        default     = 1024,
        description = "Number of frames N_batch to collect per training iteration."
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
        description = "Number of batches to prefetch for GPU efficiency."
    )
    total_frames: PositiveInt = Field(
        default     = 200_000,
        description = "Total environment frames T to collect during training."
    )


class HardwareModel(BaseModel, extra="forbid"):
    """
    Hardware and compute configuration.
    
    Configures the computational resources and strategies used for training,
    including device selection, precision, and distributed training options.
    """
    accelerator: Literal["auto", "cpu", "gpu", "tpu", "mps"] = Field(
        default     = "auto",
        description = (
            "Hardware accelerator type for training computations - automatically "
            "detects and selects the best available device."
        )
    )
    compile_model: bool = Field(
        default     = False,
        description = (
            "Enable PyTorch 2.0 compile for potential speedups. May increase "
            "startup time but can significantly improve training performance."
        )
    )
    devices: PositiveInt = Field(
        default     = 1,
        description = "Number of GPUs/TPUs to use for distributed training."
    )
    precision: Literal["16-mixed", "bf16-mixed", "32-true", "64-true"] = Field(
        default     = "16-mixed",
        description = (
            "Numerical precision mode for training - mixed precision reduces memory "
            "usage and accelerates computation on compatible hardware."
        )
    )
    strategy: Literal["auto", "ddp", "dp", "deepspeed", "fsdp"] = Field(
        default     = "auto",
        description = (
            "Distributed training strategy. 'auto' selects based on hardware, "
            "'ddp' for multi-GPU, 'deepspeed'/'fsdp' for large model training."
        )
    )



class OptimizerModel(BaseModel, extra="forbid"):
    """
    Optimization and learning rate configuration.
    
    Defines the optimization algorithm parameters and learning rate scheduling
    for training the imitation policy to minimize:
    
        L_imitation = ||π_θ(s) - u_nom||²
    
    where π_θ is the learned GNN policy and π* is the expert controller.
    """
    early_stopping_patience: PositiveInt = Field(
        default     = 10,
        description = (
            "Number of epochs without validation improvement before early stopping "
            "triggers to prevent overfitting and save compute."
        )
    )
    gradient_clip_val: PositiveFloat = Field(
        default     = 1.0,
        description = (
            "Maximum gradient norm threshold for clipping to prevent exploding "
            "gradients and maintain stable training dynamics."
        )
    )
    learning_rate: PositiveFloat = Field(
        default     = 3e-4,
        description = "Learning rate α for the AdamW optimizer."
    )
    lr_factor: PositiveFloat = Field(
        default     = 0.5,
        description = (
            "Factor by which learning rate is reduced when validation loss plateaus. "
            "New LR = old LR × factor."
        )
    )
    lr_patience: PositiveInt = Field(
        default     = 5,
        description = (
            "Number of epochs with no improvement before reducing learning rate. "
            "Works with ReduceLROnPlateau scheduler."
        )
    )
    lr_scheduler_verbose: bool = Field(
        default     = True,
        description = (
            "Whether ReduceLROnPlateau scheduler prints messages when learning "
            "rate is reduced."
        )
    )
    metric: str = Field(
        default     = "train/loss",
        description = "Metric name to track for checkpointing and early stopping."
    )
    mode: Literal["min", "max"] = Field(
        default     = "min",
        description = (
            "Optimization direction for monitored metric (minimize or maximize)."
        )
    )
    seed: Optional[NonNegativeInt] = Field(
        default     = 42,
        description = (
            "Random seed for reproducible training runs. Set to None for "
            "non-deterministic behavior."
        )
    )
    weight_decay: NonNegativeFloat = Field(
        default     = 1e-5,
        description = "L2 regularization coefficient λ for weight decay."
    )