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


class ExperienceModel(BaseModel, extra="forbid"):
    """
    Experience data handling and batching configuration.

    Manages how experience data is collected, stored, and sampled during
    imitation learning from expert demonstrations.
    """
    batch_size: PositiveInt = Field(
        default     = 256,
        ge          = 16,
        description = (
            "Number of state-action transitions B sampled per gradient update, "
            "balancing computational efficiency with gradient variance. "
            "Minimum of 16 ensures stable gradients and efficient GPU utilization."
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
    validation_batches: PositiveInt = Field(
        default     = 2,
        description = (
            "Number of batches to sample for validation from the replay buffer. "
            "Validation samples from the oldest portion of the buffer to avoid overlap "
            "with recent training data."
        )
    )
    validation_split: float = Field(
        default     = 0.2,
        ge          = 0.0,
        le          = 0.5,
        description = (
            "Fraction of replay buffer reserved for validation sampling. "
            "A value of 0.2 means validation samples from the oldest 20% of the buffer, "
            "ensuring temporal separation from recent training data."
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
        default     = False,
        description = (
            "Enable cuDNN benchmarking to find optimal algorithms. Improves "
            "performance for fixed input sizes but adds startup overhead."
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
        default     = "32",
        description = (
            "Numerical precision mode for training. Default '32' lets Lightning "
            "choose optimal precision. Mixed precision reduces memory usage."
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
        default     = 1.0,
        description = (
            "Maximum gradient norm threshold for clipping to prevent exploding "
            "gradients and maintain stable training dynamics."
        )
    )
    learning_rate: PositiveFloat = Field(
        default     = 3e-4,
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
        default     = "train/loss",
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
        default     = "val/loss",
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


class WandbModel(BaseModel, extra="forbid"):
    """
    Configuration for Weights & Biases experiment tracking.

    W&B provides comprehensive experiment tracking for machine learning workflows,
    including metric logging, hyperparameter tracking, model versioning, and
    visualization. This configuration controls how Lightning integrates with W&B
    during training and how the CLI accesses W&B for monitoring.

    The mode parameter allows flexible deployment:
    - "online": Full cloud synchronization for collaborative work
    - "offline": Local logging for air-gapped environments
    - "disabled": No W&B integration (useful for debugging)

    The API key is read from an environment variable to avoid hardcoding
    credentials in configuration files.
    """
    log_model: Literal["all", "false"] | bool = Field(
        default     = "all",
        description = (
            "Model checkpoint logging policy - 'all' saves every checkpoint, "
            "'false' or False disables model logging to save bandwidth."
        )
    )
    mode: Literal["online", "offline", "disabled"] = Field(
        default     = "online",
        description = (
            "W&B tracking mode - online syncs to cloud, offline stores locally, "
            "disabled skips W&B integration entirely."
        )
    )
    project: str = Field(
        default     = "thermur-imitation",
        description = (
            "W&B project name for organizing experiments - groups related training "
            "runs for easier comparison and analysis."
        )
    )
    run_name: str | None = Field(
        default     = None,
        description = (
            "Explicit run name (e.g., 'IM001', 'murmuration-test-v2'). Propagates to "
            "WandB, Hydra output directories, and checkpoint paths. If not set, "
            "WandB auto-generates a random name like 'cosmic-sunset-42'."
        )
    )


class WatchModel(BaseModel, extra="forbid"):
    """
    Watch configuration for real-time training visualization.

    Controls the integration of PyVista 3D visualization into the training loop,
    allowing researchers to observe emergent flock behaviors and thermal dynamics
    as the policy learns. Visualization frames can be automatically logged to
    WandB for later review without re-running simulations.
    """
    auto_close: bool = Field(
        default     = True,
        description = (
            "Automatically close visualization window when training completes. "
            "Disable to keep window open for final state inspection."
        )
    )
    fps: PositiveInt = Field(
        default     = 30,
        description = (
            "Frames per second for video encoding when logging to WandB. "
            "Standard video framerates are 24, 30, or 60 fps."
        )
    )
    start_epoch: NonNegativeInt = Field(
        default     = 0,
        description = (
            "Epoch at which to start visualization. Delaying start can skip early "
            "random policy behavior and focus on emergent learned behaviors."
        )
    )
    update_frequency: PositiveInt = Field(
        default     = 10,
        description = (
            "Batch interval between visualization updates. Higher values reduce "
            "computational overhead but provide less frequent visual feedback."
        )
    )
    video_duration: float = Field(
        default     = 30.0,
        gt          = 0,
        description = (
            "Duration in seconds of each video clip logged to WandB. "
            "Training progress is captured as a series of video clips, each "
            "showing this many seconds of simulation at different training steps."
        )
    )
    watch_run: bool = Field(
        default     = False,
        description = (
            "Display a live visualization window during training. Typically controlled "
            "via the --watch CLI flag rather than config files."
        )
    )