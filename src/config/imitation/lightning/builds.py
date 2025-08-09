"""
Lightning domain builds for hydra-zen configuration.

This module provides pre-built components for PyTorch Lightning training infrastructure:

Training Components:
- Trainer                : PyTorch Lightning trainer with hardware configuration,
                           gradient clipping, distributed training support, and
                           callback management.
- DataModule             : Handles experience replay buffer management, batch
                           generation, and data loading for the imitation learning
                           pipeline.
- GNNPolicy              : Graph Neural Network policy that processes agent
                           observations and produces control actions using attention
                           mechanisms.

Optimization:
- AdamW                  : Adaptive optimizer with weight decay for training the
                           policy network.
- ReduceLROnPlateau      : Learning rate scheduler that reduces LR when validation
                           metrics plateau.

Callbacks:
- ModelCheckpoint        : Saves model checkpoints based on validation metrics with
                           configurable retention policies.
- EarlyStopping          : Monitors validation metrics and stops training when no
                           improvement is detected.
- LearningRateMonitor    : Tracks and logs learning rate changes during training.
- MonitoringCallback     : Integrates with the metrics collector for training
                           analytics.

Logging:
- WandbLogger            : Weights & Biases integration for experiment tracking and
                           metric visualization.
"""
from __future__                  import annotations
from .schemas                    import *
from hydra_zen                   import builds, make_config
from pytorch_lightning           import Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers   import WandbLogger
from thermur.imitation.lightning import DataModule, GNNPolicy, MonitoringCallback, VisualizationCallback
from torch.optim                 import AdamW
from torch.optim.lr_scheduler    import ReduceLROnPlateau
from typing                      import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from hydra_zen.typing import Builds


LIGHTNING_USER_CONFIG = make_config(
    architecture = ArchitectureModel(),
    checkpoint   = CheckpointModel(),
    experience   = ExperienceModel(),
    hardware     = HardwareModel(),
    optimizer    = OptimizerModel(),
    wandb        = WandbModel(),
    watch        = WatchModel()
)

LIGHTNING_SYSTEM_BUILDS: dict[str, type[Builds[Any]]] = {
    "checkpoint_callback": builds(
        ModelCheckpoint,
        dirpath                 = "${lightning.checkpoint.dirpath}",
        every_n_train_steps     = "${lightning.checkpoint.every_n_train_steps}",
        filename                = "checkpoint-{step}",
        monitor                 = "${lightning.optimizer.training_metric}",
        mode                    = "${lightning.optimizer.mode}",
        save_last               = "${lightning.checkpoint.save_last}",
        save_top_k              = "${lightning.checkpoint.save_top_k}",
        populate_full_signature = True
    ),

    "datamodule": builds(
        DataModule,
        env                     = "${_system.env}",
        experience              = "${lightning.experience}",
        expert                  = "${_system.murmuration_controller}",
        populate_full_signature = True
    ),

    "early_stopping_callback": builds(
        EarlyStopping,
        monitor                 = "${lightning.optimizer.training_metric}",
        mode                    = "${lightning.optimizer.mode}",
        patience                = "${lightning.optimizer.early_stopping_patience}",
        populate_full_signature = True
    ),

    "logger": builds(
        WandbLogger,
        log_model               = "${lightning.wandb.log_model}",
        mode                    = "${lightning.wandb.mode}",
        name                    = "${lightning.wandb.run_name}",
        project                 = "${lightning.wandb.project}",
        populate_full_signature = True
    ),

    "lr_monitor_callback": builds(
        LearningRateMonitor,
        populate_full_signature = True
    ),

    "monitoring_callback": builds(
        MonitoringCallback,
        collector               = "${_system.collector}",
        events                  = "${_system.events}",
        populate_full_signature = True
    ),

    "optimizer": builds(
        AdamW,
        lr                      = "${lightning.optimizer.learning_rate}",
        weight_decay            = "${lightning.optimizer.weight_decay}",
        zen_partial             = True,
        populate_full_signature = True
    ),

    "policy": builds(
        GNNPolicy,
        architecture            = "${lightning.architecture}",
        collector               = "${_system.collector}",
        optimizer               = "${_system.optimizer}",
        scheduler               = "${_system.scheduler}",
        scheduler_metric        = "${lightning.optimizer.scheduler_metric}",
        training_metric         = "${lightning.optimizer.training_metric}",
        populate_full_signature = True
    ),

    "scheduler": builds(
        ReduceLROnPlateau,
        factor                  = "${lightning.optimizer.lr_factor}",
        patience                = "${lightning.optimizer.lr_patience}",
        mode                    = "${lightning.optimizer.mode}",
        zen_partial             = True,
        populate_full_signature = True
    ),

    "trainer": builds(
        Trainer,
        accelerator             = "${lightning.hardware.accelerator}",
        benchmark               = "${lightning.hardware.benchmark}",
        callbacks               = [
            "${_system.checkpoint_callback}",
            "${_system.early_stopping_callback}",
            "${_system.lr_monitor_callback}",
            "${_system.monitoring_callback}",
            "${_system.watch_callback}"
        ],
        detect_anomaly          = "${lightning.hardware.detect_anomaly}",
        deterministic           = "${lightning.hardware.deterministic}",
        devices                 = "${lightning.hardware.devices}",
        gradient_clip_val       = "${lightning.optimizer.gradient_clip_val}",
        log_every_n_steps       = "${lightning.optimizer.log_every_n_steps}"
        logger                  = "${_system.logger}",
        max_epochs              = "${lightning.optimizer.max_epochs}",
        precision               = "${lightning.hardware.precision}",
        profiler                = "${monitoring.metrics.profiler}",
        strategy                = "${lightning.hardware.strategy}",
        val_check_interval      = "${lightning.optimizer.val_check_interval}",
        populate_full_signature = True
    ),

    "watch_callback": builds(
        VisualizationCallback,
        auto_close              = "${lightning.watch.auto_close}",
        fps                     = "${lightning.watch.fps}",
        start_epoch             = "${lightning.watch.start_epoch}",
        update_frequency        = "${lightning.watch.update_frequency}",
        video_duration          = "${lightning.watch.video_duration}",
        visualizer              = "${_system.visualizer}",
        watch_run               = "${lightning.watch.watch_run}",
        populate_full_signature = True
    )
}
