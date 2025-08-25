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
    compile: bool = Field(
        default     = False,
        description = (
            "Enable torch.compile optimization for the forward pass. Provides significant "
            "speedup on modern GPUs but increases initial compilation time. Recommended "
            "for production training but not for debugging."
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
        default     = "checkpoints",
        description = (
            "Directory path for saving model checkpoints during training, enabling "
            "recovery from interruptions and model comparison across runs."
        )
    )
    every_n_train_steps: PositiveInt = Field(
        default     = 25_000,
        description = (
            "Step interval between checkpoint saves, balancing storage costs with "
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
    devices: PositiveInt = Field(
        default     = 1,
        description = "Number of GPUs/TPUs to use for distributed training."
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
        default     = True,
        description = (
            "Pin memory for faster GPU transfers. Disable for CPU-only training "
            "or when memory is limited."
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
        default     = 2e-3,
        description = (
            "Initial learning rate α for AdamW optimizer, controlling step size "
            "in parameter space during gradient descent optimization."
        )
    )
    log_every_n_steps: PositiveInt = Field(
        default     = 1,
        description = (
            "How often to log metrics during training (every N batches). "
            "Set to 1 for logging every step, useful when training with few batches."
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
    training_metric: str = Field(
        default     = "training/loss",
        description = (
            "Primary metric monitored during training for logging and model "
            "selection. Also used by early stopping callback."
        )
    )
    mode: Literal["min", "max"] = Field(
        default     = "min",
        description = (
            "Optimization direction for monitored metric (minimize or maximize)."
        )
    )
    scheduler_metric: str = Field(
        default     = "validation/loss",
        description = (
            "Metric monitored by learning rate scheduler for reducing learning rate "
            "on plateau. Typically a validation metric to avoid overfitting."
        )
    )
    seed: NonNegativeInt | None = Field(
        default     = 42,
        description = (
            "Random seed for reproducible training runs. Set to None for "
            "non-deterministic behavior."
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
    info_propagation_max_speed: PositiveFloat = Field(
        default     = 45.0,
        description = (
            "Maximum information propagation speed in m/s through the flock, "
            "corresponding to alert state responsiveness (Attanasi et al. 2014)."
        )
    )
    info_propagation_min_speed: PositiveFloat = Field(
        default     = 15.0,
        description = (
            "Minimum information propagation speed in m/s through the flock, "
            "corresponding to relaxed state responsiveness (Attanasi et al. 2014)."
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
