"""
Hydra-zen builders for PyTorch Lightning components.

This module provides configuration builders for Lightning-specific training
components including the GNN policy network, data module, trainer, and callbacks.
"""
from config.imitation.schemas.learning      import *
from config.imitation.schemas.wandb         import WandbModel
from hydra_zen                              import builds, zen
from omegaconf                              import SI
from pytorch_lightning                      import Trainer
from pytorch_lightning.callbacks            import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers              import WandbLogger
from thermur.imitation.lightning.callback   import MonitoringCallback
from thermur.imitation.lightning.experience import DataModule
from thermur.imitation.lightning.policy     import GNNPolicy


build_checkpoint = builds(
    CheckpointModel,
    populate_full_signature = True,
    zen_dataclass           = {
        "module"   : "src.configs.imitation.factories.lightning",
        "cls_name" : "CheckpointBuild"
    }
)
"""
Builder for checkpoint configuration.

Controls model checkpoint saving frequency, location, and retention policy.
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
Builder for experience data configuration.

Manages batch sizes, buffer capacity, and data collection parameters.
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
Builder for hardware configuration.

Specifies compute resources, precision, and distributed training strategy.
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
Builder for optimizer configuration.

Sets learning rate, weight decay, gradient clipping, and scheduling parameters.
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
Builder for model checkpointing callback.

Saves model checkpoints at regular intervals during training, enabling
recovery from failures and model selection.
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
Builder for Lightning data module.

Manages expert demonstration collection and replay buffer for imitation
learning, wrapping TorchRL components in Lightning's DataModule interface.
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
Builder for early stopping callback.

Monitors training loss and stops training if no improvement is seen,
preventing overfitting and saving compute resources.
"""

build_lr_monitor_callback = builds(
    LearningRateMonitor,
    logging_interval        = SI("${monitoring.logging_interval}"),
    populate_full_signature = True,
    zen_dataclass           = {
        "module"   : "src.configs.imitation.factories.lightning",
        "cls_name" : "LRMonitorCallbackBuild"
    }
)
"""
Builder for learning rate monitor callback.

Tracks learning rate changes during training, particularly useful when
using schedulers like ReduceLROnPlateau.
"""

build_policy = builds(
    GNNPolicy,
    architecture            = zen(ArchitectureModel),
    metrics                 = SI("${metrics}"),
    optimizer               = zen(OptimizerModel),
    populate_full_signature = True,
    zen_dataclass           = {
        "module"   : "src.configs.imitation.factories.lightning",
        "cls_name" : "PolicyBuild"
    }
)
"""
Builder for the Graph Neural Network policy.

Creates a permutation-equivariant GNN that processes the flock graph
structure to output nominal control actions u_nom for each agent.
"""

build_monitoring_callback = builds(
    MonitoringCallback,
    events                  = SI("${events}"),
    metrics                 = SI("${metrics}"),
    populate_full_signature = True,
    zen_dataclass           = {
        "module"   : "src.configs.imitation.factories.lightning",
        "cls_name" : "MonitoringCallbackBuild"
    }
)
"""
Builder for unified monitoring callback.

Creates a single Lightning callback that handles both metrics collection
and event logging, reducing code duplication and simplifying integration.

Creates a Lightning callback that integrates MetricsCollector into the training
loop, updating evaluation metrics and handling epoch boundaries.
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
Builder for W&B experiment tracking configuration.

Creates a validated configuration that controls how Lightning integrates
with Weights & Biases for metric logging, hyperparameter tracking, and
experiment organization during training.
"""

build_trainer = builds(
    Trainer,
    accelerator             = SI("${hardware.accelerator}"),
    callbacks               = [
        build_checkpoint_callback, 
        build_early_stopping_callback,
        build_lr_monitor_callback,
        SI("${monitoring_callback}")
    ],
    devices                 = SI("${hardware.devices}"),
    enable_model_summary    = SI("${monitoring.enable_model_summary}"),
    enable_progress_bar     = SI("${monitoring.enable_progress_bar}"),
    gradient_clip_val       = SI("${optimizer.gradient_clip_val}"),
    log_every_n_steps       = SI("${monitoring.log_every_n_steps}"),
    logger                  = builds(
        WandbLogger,
        log_model   = "all",
        mode        = SI("${wandb.mode}"),
        project     = SI("${wandb.project}"),
        save_dir    = SI("${checkpoint.dirpath}")
    ),
    precision               = SI("${hardware.precision}"),
    profiler                = SI("${monitoring.profiler}"),
    strategy                = SI("${hardware.strategy}"),
    populate_full_signature = True,
    zen_dataclass           = {
        "module"   : "src.configs.imitation.factories.lightning",
        "cls_name" : "TrainerBuild"
    }
)
"""
Builder for PyTorch Lightning Trainer.

Configures the Lightning Trainer with automatic mixed precision, gradient
clipping, logging, and checkpointing. Handles all training loop boilerplate
including device placement, backward passes, and metric tracking.
"""