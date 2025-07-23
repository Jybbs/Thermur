"""
Learning system model.

This module defines the unified configuration for imitation learning,
including training hyperparameters, data handling, and network architecture.
"""
from pathlib  import Path
from pydantic import BaseModel, Field, NonNegativeFloat
from pydantic import NonNegativeInt, PositiveFloat, PositiveInt
from typing   import Literal, Optional


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
    accelerator: Literal["auto", "cpu", "gpu", "tpu", "mps"] = Field(
        default     = "auto",
        description = (
            "Hardware accelerator type for training computations - automatically "
            "detects and selects the best available device."
        )
    )
    activation: Literal["ReLU", "SiLU", "Tanh"] = Field(
        default     = "SiLU",
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
    dirpath: Path = Field(
        default     = Path("checkpoints"),
        description = "Directory path for saving training checkpoints."
    )
    enable_model_summary: bool = Field(
        default     = True,
        description = (
            "Display comprehensive model architecture summary including parameter "
            "counts and layer shapes before training begins."
        )
    )
    enable_progress_bar: bool = Field(
        default     = True,
        description = "Show real-time progress bar with loss metrics during training."
    )
    every_n_train_steps: PositiveInt = Field(
        default     = 25_000,
        description = "Frequency in training steps for saving model checkpoints."
    )
    filename: str = Field(
        default     = "checkpoint-{step}",
        description = "Template for checkpoint filenames with step number placeholder."
    )
    frames_per_batch: PositiveInt = Field(
        default     = 1024,
        description = "Number of frames N_batch to collect per training iteration."
    )
    gradient_clip_val: PositiveFloat = Field(
        default     = 1.0,
        description = (
            "Maximum gradient norm threshold for clipping to prevent exploding "
            "gradients and maintain stable training dynamics."
        )
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
    log_every_n_steps: PositiveInt = Field(
        default     = 50,
        description = "Frequency of metric logging to track training progress."
    )
    logging_interval: Literal["step", "epoch"] = Field(
        default     = "step",
        description = "Interval for learning rate logging (step or epoch)."
    )
    max_frames_per_traj: int = Field(
        default     = -1,
        description = (
            "Maximum frames per trajectory before episode reset. Use -1 for infinite "
            "episodes that only reset when environment signals done."
        )
    )
    mode: Literal["min", "max"] = Field(
        default     = "min",
        description = (
            "Optimization direction for monitored metric (minimize or maximize)."
        )
    )
    monitor: str = Field(
        default     = "train/loss",
        description = "Metric name to track for checkpointing and early stopping."
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
    patience: PositiveInt = Field(
        default     = 10,
        description = (
            "Number of epochs without validation improvement before early stopping "
            "triggers to prevent overfitting and save compute."
        )
    )
    precision: Literal["16-mixed", "bf16-mixed", "32-true", "64-true"] = Field(
        default     = "16-mixed",
        description = (
            "Numerical precision mode for training - mixed precision reduces memory "
            "usage and accelerates computation on compatible hardware."
        )
    )
    prefetch: NonNegativeInt = Field(
        default     = 8,
        description = "Number of batches to prefetch for GPU efficiency."
    )
    profiler: Optional[Literal["simple", "advanced"]] = Field(
        default     = None,
        description = (
            "Lightning profiler for performance analysis. 'simple' tracks basic "
            "metrics, 'advanced' provides detailed profiling with Chrome tracing."
        )
    )
    save_last: bool = Field(
        default     = True,
        description = (
            "Always save the final model checkpoint at training completion regardless "
            "of whether it achieved the best validation metric."
        )
    )
    seed: Optional[NonNegativeInt] = Field(
        default     = 42,
        description = (
            "Random seed for reproducible training runs. Set to None for "
            "non-deterministic behavior."
        )
    )
    strategy: Literal["auto", "ddp", "dp", "deepspeed", "fsdp"] = Field(
        default     = "auto",
        description = (
            "Distributed training strategy. 'auto' selects based on hardware, "
            "'ddp' for multi-GPU, 'deepspeed'/'fsdp' for large model training."
        )
    )
    total_frames: PositiveInt = Field(
        default     = 200_000,
        description = "Total environment frames T to collect during training."
    )
    weight_decay: NonNegativeFloat = Field(
        default     = 1e-5,
        description = "L2 regularization coefficient λ for weight decay."
    )