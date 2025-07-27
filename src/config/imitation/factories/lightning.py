"""
Hydra-zen builders for PyTorch Lightning components.

This module provides configuration builders for Lightning-specific training
components including the GNN policy network, data module, trainer, and callbacks.
"""
from config.imitation.schemas.lightning import *
from hydra_zen                          import builds, zen
from omegaconf                          import SI
from pytorch_lightning                  import Trainer
from pytorch_lightning.callbacks        import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers          import WandbLogger
from thermur.imitation.lightning        import DataModule, GNNPolicy, MonitoringCallback


build_checkpoint = builds(
    CheckpointModel,
    populate_full_signature = True,
    zen_dataclass           = {
        "module"   : "src.configs.imitation.factories.lightning",
        "cls_name" : "CheckpointBuild"
    }
)
"""
Builder for model persistence configuration.

Manages checkpoint saving strategy including frequency (every N steps/epochs),
file naming patterns, directory structure, and retention policies. Ensures training
can resume after interruptions and facilitates model selection by preserving best
and latest checkpoints. Integrates with W&B for checkpoint versioning and backup.
"""

build_experience = builds(
    ExperienceModel,
    populate_full_signature = True,
    zen_dataclass           = {
        "module"   : "src.configs.imitation.factories.lightning",
        "cls_name" : "ExperienceBuild"
    }
)
"""
Builder for experience replay configuration.

Controls expert demonstration storage and sampling including buffer capacity,
batch dimensions, trajectory segmentation, and prefetch optimization. Balances
memory usage against data diversity by managing how many frames to collect before
training begins and how to sample mini-batches that preserve temporal correlations
while ensuring stable gradient estimates during behavioral cloning.
"""

build_hardware = builds(
    HardwareModel,
    populate_full_signature = True,
    zen_dataclass           = {
        "module"   : "src.configs.imitation.factories.lightning",
        "cls_name" : "HardwareBuild"
    }
)
"""
Builder for computational resource allocation.

Optimizes training for available hardware by configuring device placement (CPU/GPU),
mixed precision training (fp16/bf16 for memory efficiency), distributed strategies
(DDP, FSDP), and model compilation options. Automatically detects CUDA availability
and adjusts batch sizes and gradient accumulation to maximize throughput while
preventing out-of-memory errors on resource-constrained systems.
"""

build_optimizer = builds(
    OptimizerModel,
    populate_full_signature = True,
    zen_dataclass           = {
        "module"   : "src.configs.imitation.factories.lightning",
        "cls_name" : "OptimizerBuild"
    }
)
"""
Builder for optimization strategy configuration.

Defines the gradient descent algorithm (AdamW) hyperparameters including learning
rate, weight decay regularization, gradient clipping thresholds, and adaptive
scheduling policies (ReduceLROnPlateau). Includes early stopping patience to
prevent overfitting and learning rate warmup for stable convergence. Critical
for balancing exploration vs exploitation during behavioral cloning.
"""

build_checkpoint_callback = builds(
    ModelCheckpoint,
    dirpath                 = SI("${checkpoint.dirpath}"),
    every_n_train_steps     = SI("${checkpoint.every_n_train_steps}"),
    filename                = SI("${checkpoint.filename}"),
    save_last               = SI("${checkpoint.save_last}"),
    populate_full_signature = True,
    zen_dataclass           = {
        "module"   : "src.configs.imitation.factories.lightning",
        "cls_name" : "CheckpointCallbackBuild"
    }
)
"""
Builder for Lightning checkpoint callback.

Automates model persistence by saving checkpoints at specified intervals (steps
or epochs) with configurable naming schemes. Monitors validation metrics to
preserve best-performing models while maintaining recent checkpoints for recovery.
Integrates with PyTorch Lightning's fault-tolerant training to enable seamless
resumption after hardware failures or preemption.
"""

build_datamodule = builds(
    DataModule,
    controller              = SI("${controller}"),
    env                     = SI("${simulation}"),
    experience              = zen(ExperienceModel),
    populate_full_signature = True,
    zen_dataclass           = {
        "module"   : "src.configs.imitation.factories.lightning",
        "cls_name" : "DataModuleBuild"
    }
)
"""
Builder for behavioral cloning data pipeline.

Bridges TorchRL's replay buffer with Lightning's data loading interface to stream
expert demonstrations during training. Manages parallel trajectory collection from
the physics-based controller, efficient batch sampling that preserves temporal
structure, and memory-mapped storage for large demonstration datasets. Handles
dynamic graph padding for variable agent counts across episodes.
"""

build_early_stopping_callback = builds(
    EarlyStopping,
    mode                    = SI("${optimizer.mode}"),
    monitor                 = SI("${optimizer.metric}"),
    patience                = SI("${optimizer.early_stopping_patience}"),
    populate_full_signature = True,
    zen_dataclass           = {
        "module"   : "src.configs.imitation.factories.lightning",
        "cls_name" : "EarlyStoppingCallbackBuild"
    }
)
"""
Builder for training termination callback.

Implements patience-based early stopping by monitoring validation metrics (loss,
accuracy) for sustained improvement. Prevents overfitting by halting training
when the model stops learning, saving computational resources and avoiding
degeneracy. Configurable patience window and improvement thresholds balance
between premature stopping and wasteful overtraining.
"""

build_logger = builds(
    WandbLogger,
    log_model               = SI("${wandb.log_model}"),
    mode                    = SI("${wandb.mode}"),
    project                 = SI("${wandb.project}"),
    save_dir                = SI("${checkpoint.dirpath}"),
    populate_full_signature = True,
    zen_dataclass           = {
        "module"   : "src.configs.imitation.factories.lightning",
        "cls_name" : "LoggerBuild"
    }
)
"""
Builder for Weights & Biases experiment logger.

Establishes connection to W&B cloud service for comprehensive experiment tracking
including hyperparameter logging, real-time metric visualization, model artifact
versioning, and collaborative dashboards. Configures project organization, run
naming conventions, and offline fallback modes for air-gapped environments.
Essential for experiment reproducibility and team collaboration.
"""

build_lr_monitor_callback = builds(
    LearningRateMonitor,
    logging_interval        = SI("${metrics.logging_interval}"),
    populate_full_signature = True,
    zen_dataclass           = {
        "module"   : "src.configs.imitation.factories.lightning",
        "cls_name" : "LRMonitorCallbackBuild"
    }
)
"""
Builder for learning rate tracking callback.

Monitors and logs learning rate evolution throughout training, capturing scheduler
dynamics (plateaus, reductions, warmup). Critical for debugging convergence issues
and understanding when the model transitions between exploration and exploitation
phases. Integrates with tensorboard and W&B for visualization of LR schedules
alongside loss curves.
"""

build_architecture = builds(
    ArchitectureModel,
    populate_full_signature = True,
    zen_dataclass           = {
        "module"   : "src.configs.imitation.factories.lightning",
        "cls_name" : "ArchitectureBuild"
    }
)
"""
Builder for Graph Neural Network architecture specification.

Configures the permutation-equivariant GNN that processes dynamic agent graphs
to predict control actions. Defines layer depth, hidden dimensionality, activation
functions (ReLU, SiLU, GELU), and message-passing mechanics. The architecture must
handle variable-sized graphs as agents enter/exit communication range, making
equivariance critical for generalization across different flock configurations.
"""

build_policy = builds(
    GNNPolicy,
    architecture            = SI("${architecture}"),
    collector               = SI("${collector}"),
    optimizer               = zen(OptimizerModel),
    populate_full_signature = True,
    zen_dataclass           = {
        "module"   : "src.configs.imitation.factories.lightning",
        "cls_name" : "PolicyBuild"
    }
)
"""
Builder for imitation learning policy network.

Constructs the core Graph Neural Network that maps agent observations (positions,
velocities, temperatures) to control actions via message-passing on the dynamic
communication graph. Implements behavioral cloning loss functions, metric tracking,
and optimization schedules. The GNN's equivariance ensures consistent behavior
regardless of agent ordering, critical for sim-to-real transfer.
"""

build_monitoring_callback = builds(
    MonitoringCallback,
    events                  = SI("${events}"),
    collector               = SI("${collector}"),
    populate_full_signature = True,
    zen_dataclass           = {
        "module"   : "src.configs.imitation.factories.lightning",
        "cls_name" : "MonitoringCallbackBuild"
    }
)
"""
Builder for unified monitoring callback.

Consolidates metrics collection and event logging into a single Lightning callback
that orchestrates all monitoring activities. Tracks training/validation losses,
physical metrics (energy, control smoothness), safety events (CBF activations,
thermal violations), and visual quality indicators. Handles metric aggregation
across distributed training and manages tensorboard/W&B logging interfaces.
"""

build_wandb = builds(
    WandbModel,
    populate_full_signature = True,
    zen_dataclass           = {
        "module"   : "src.config.imitation.factories.lightning",
        "cls_name" : "WandbBuild"
    }
)
"""
Builder for Weights & Biases integration settings.

Configures the W&B service connection including project naming, run grouping,
API authentication, and cloud synchronization modes. Controls artifact logging
policies for model checkpoints, visualization outputs, and configuration files.
Supports offline mode for secure environments and manages bandwidth usage through
selective metric/media logging. Foundation for experiment tracking ecosystem.
"""

build_trainer = builds(
    Trainer,
    accelerator             = SI("${hardware.accelerator}"),
    callbacks               = [
        build_checkpoint_callback, 
        build_early_stopping_callback,
        build_lr_monitor_callback,
        build_monitoring_callback
    ],
    devices                 = SI("${hardware.devices}"),
    enable_model_summary    = SI("${metrics.enable_model_summary}"),
    enable_progress_bar     = SI("${metrics.enable_progress_bar}"),
    gradient_clip_val       = SI("${optimizer.gradient_clip_val}"),
    log_every_n_steps       = SI("${metrics.log_every_n_steps}"),
    logger                  = SI("${logger}"),
    precision               = SI("${hardware.precision}"),
    profiler                = SI("${metrics.profiler}"),
    strategy                = SI("${hardware.strategy}"),
    populate_full_signature = True,
    zen_dataclass           = {
        "module"   : "src.configs.imitation.factories.lightning",
        "cls_name" : "TrainerBuild"
    }
)
"""
Builder for PyTorch Lightning training orchestrator.

Centralizes all training loop configuration including hardware allocation, mixed
precision settings, gradient accumulation strategies, distributed training backends,
and callback scheduling. Manages the complex interplay between data loading, forward
passes, loss computation, backpropagation, optimizer steps, and metric logging.
Provides fault tolerance, automatic batching, and profiling hooks while abstracting
away PyTorch boilerplate for cleaner scientific code.
"""