"""
Lightning domain schemas for Pydantic validation.

This module consolidates all PyTorch Lightning configuration models including
policy architecture, training hardware, optimization, and experiment tracking.
"""
from pydantic import BaseModel, Field, NonNegativeFloat
from pydantic import NonNegativeInt, PositiveFloat, PositiveInt
from typing   import Literal


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
        description = (
            "Activation function σ(·) applied after linear transformations in message "
            "passing MLPs, with SiLU providing smooth gradients for stable training."
        )
    )
    hidden_dim: PositiveInt = Field(
        default     = 64,
        description = (
            "Hidden state dimensionality d_h for node embeddings h_i^(l) at each "
            "message passing layer, balancing expressiveness against memory usage."
        )
    )
    num_layers: PositiveInt = Field(
        default     = 3,
        description = (
            "Message passing depth L determining multi-hop receptive field radius "
            "r = L × R_comm for aggregating information from distant neighbors."
        )
    )


class CheckpointModel(BaseModel, extra="forbid"):
    """
    Model checkpoint configuration.

    Controls how and when model checkpoints are saved during training,
    enabling recovery from interruptions and model selection.
    """
    dirpath: str = Field(
        default     = "data/checkpoints",
        description = (
            "Directory path for saving model checkpoints during training, enabling "
            "recovery from interruptions and model comparison across runs."
        )
    )
    enabled: bool = Field(
        default     = False,
        description = (
            "Enable or disable checkpoint saving during training. When disabled, "
            "no checkpoints will be saved regardless of other checkpoint settings."
        )
    )
    every_n_train_steps: PositiveInt = Field(
        default     = 25_000,
        description = (
            "Frame interval between checkpoint saves, balancing storage costs with "
            "recovery granularity for long training runs on large datasets."
        )
    )
    save_last: bool = Field(
        default     = True,
        description = (
            "Always save the final model checkpoint at training completion regardless "
            "of whether it achieved the best validation metric."
        )
    )
    save_top_k: int = Field(
        default     = 3,
        description = (
            "Number of best model checkpoints to keep based on monitored metric. "
            "Use -1 to keep all checkpoints, 0 to disable best model saving."
        )
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
    benchmark: bool = Field(
        default     = True,
        description = (
            "Enable algorithm benchmarking to find optimal kernels. Improves "
            "performance for fixed input sizes. Recommended for MPS training."
        )
    )
    deterministic: bool = Field(
        default     = False,
        description = (
            "Enable deterministic mode for reproducible results. May reduce "
            "performance but ensures identical results across runs with same seed."
        )
    )
    detect_anomaly: bool = Field(
        default     = False,
        description = (
            "Enable anomaly detection to find NaN/Inf values in gradients. "
            "Adds significant overhead but helpful for debugging training issues."
        )
    )
    devices: int = Field(
        default     = -1,
        description = (
            "Number of GPUs/TPUs to use for distributed training. Set to 1 for single device "
            "or -1 to use all available devices (not applicable for MPS on Apple Silicon)."
        )
    )
    precision: Literal["16-mixed", "bf16-mixed", "32-true", "64-true", "32"] = Field(
        default     = "32-true",
        description = (
            "Numerical precision mode for training. '32-true' provides full "
            "precision and compatibility across all devices including MPS."
        )
    )
    strategy: Literal["auto", "ddp", "dp", "deepspeed", "fsdp"] = Field(
        default     = "auto",
        description = (
            "Distributed training strategy. 'auto' selects based on hardware, "
            "'ddp' for multi-GPU, 'deepspeed'/'fsdp' for large model training."
        )
    )
    num_workers: NonNegativeInt = Field(
        default     = 8,
        description = (
            "Number of worker processes for data loading. Set to 0 for debugging "
            "or when using CPU. Higher values improve throughput with GPUs."
        )
    )
    pin_memory: bool = Field(
        default     = False,
        description = (
            "Pin memory for faster GPU transfers. Disabled by default to avoid "
            "MPS warnings. Enable for CUDA GPUs if memory permits."
        )
    )


class OptimizerModel(BaseModel, extra="forbid"):
    """
    Optimization and learning rate configuration.

    Defines the optimization algorithm parameters and learning rate scheduling
    for training the imitation policy to minimize:

        L_imitation = ||π_θ(s) - u_nom||²

    where π_θ is the learned GNN policy and π* is the murmuration controller.
    """
    batch_size: PositiveInt = Field(
        default     = 256,
        ge          = 16,
        description = (
            "Number of graph states per training batch. Each state contains "
            "the full flock state at a single frame. Larger batches improve "
            "gradient stability but require more memory."
        )
    )
    early_stopping_patience: PositiveInt = Field(
        default     = 10,
        description = (
            "Number of epochs without validation improvement before early stopping "
            "triggers to prevent overfitting and save compute."
        )
    )
    gradient_clip_val: PositiveFloat = Field(
        default     = 10.0,
        description = (
            "Maximum gradient norm threshold for clipping to prevent exploding "
            "gradients and maintain stable training dynamics."
        )
    )
    learning_rate: PositiveFloat = Field(
        default     = 5e-4,
        description = (
            "Initial learning rate α for AdamW optimizer, controlling gradient step "
            "size in parameter space during gradient descent optimization."
        )
    )
    log_every_n_steps: PositiveInt = Field(
        default     = 1,
        description = (
            "How often to log metrics during training (every N batches). "
            "Set to 1 for logging every frame, useful when training with few batches."
        )
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
    max_epochs: PositiveInt = Field(
        default     = 100,
        description = (
            "Maximum number of training epochs before termination, providing an "
            "upper bound on training time even if convergence isn't reached."
        )
    )
    mode: Literal["min", "max"] = Field(
        default     = "min",
        description = (
            "Optimization direction for monitored metric (minimize or maximize)."
        )
    )
    seed: NonNegativeInt | None = Field(
        default     = 42,
        description = (
            "Random seed for reproducible training runs. Set to None for "
            "non-deterministic behavior."
        )
    )
    train_split: float = Field(
        default     = 0.8,
        ge          = 0.5,
        le          = 0.95,
        description = (
            "Fraction of data reserved for training, with remainder for validation. "
            "Split occurs randomly across all frames to ensure diverse data when "
            "validating."
        )
    )
    val_check_interval: float = Field(
        default     = 1.0,
        description = (
            "How often to check validation set within an epoch. Use 1.0 for once "
            "per epoch, 0.5 for twice, or integer for every N batches."
        )
    )
    weight_decay: NonNegativeFloat = Field(
        default     = 1e-5,
        description = (
            "L2 regularization coefficient λ penalizing large weights to improve "
            "generalization and prevent overfitting to training demonstrations."
        )
    )


class MetricsModel(BaseModel, extra="forbid"):
    """
    Configuration for metrics collection and performance monitoring.

    Defines parameters for metric computation, logging frequency, and
    visualization settings used by the MetricsCollector.
    """
    correlation_exponent: PositiveFloat = Field(
        default     = 0.333,
        description = (
            "Expected exponent γ ≈ 1/3 for scale-free velocity correlations C(r) ~ r^(-γ) "
            "in natural murmurations, per Cavagna et al. (2010)."
        )
    )
    fiedler_shift: PositiveFloat = Field(
        default     = 0.001,
        description = (
            "Diagonal shift λ for computing Fiedler eigenvalue of graph Laplacian "
            "through power iteration, ensures positive definiteness."
        )
    )
    orientation_wave_radius: PositiveFloat = Field(
        default     = 10.0,
        description = (
            "Radius R_wave for computing local orientation gradients ∇θ(𝐫) in "
            "murmuration wave detection, where θ = atan2(v_y, v_x)."
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
    velocity_threshold: PositiveFloat = Field(
        default     = 1e-3,
        description = (
            "Minimum velocity magnitude in m/s below which agents are considered "
            "stationary for orientation wave and heading computations."
        )
    )
