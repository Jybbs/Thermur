"""
Training domain builds for hydra-zen configuration.

This module provides pre-built components for the training infrastructure:

Training Components:
- DemonstrationsDataset  : PyG InMemoryDataset that generates and caches expert
                           trajectories with automatic WRF data discovery.
- GNNPolicy              : Graph Neural Network policy that processes agent
                           observations and produces control actions using attention
                           mechanisms.
- Trainer                : PyTorch Lightning trainer with hardware configuration,
                           gradient clipping, distributed training support, and
                           callback management.

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
from __future__                   import annotations
from .schemas                     import *
from config.cli.schemas           import WandbModel
from hydra_zen                    import builds, make_config
from pytorch_lightning            import Trainer
from pytorch_lightning.callbacks  import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers    import WandbLogger
from thermur.imitation.controller import DemonstrationsDataset
from thermur.imitation.training   import GNNPolicy, MetricsCollector, MonitoringCallback
from torch.optim                  import AdamW
from torch.optim.lr_scheduler     import ReduceLROnPlateau
from typing                       import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from hydra_zen.typing import Builds


TRAINING_USER_CONFIG = make_config(
    architecture    = ArchitectureModel(),
    checkpoint      = CheckpointModel(),
    hardware        = HardwareModel(),
    metrics         = MetricsModel(),
    optimizer       = OptimizerModel(),
    wandb           = WandbModel()
)

TRAINING_SYSTEM_BUILDS: dict[str, type[Builds[Any]]] = {

    "checkpoint_callback": builds(
        ModelCheckpoint,
        dirpath                 = "${training.checkpoint.dirpath}",
        every_n_train_steps     = "${training.checkpoint.every_n_train_steps}",
        filename                = "checkpoint-{step}",
        monitor                 = "${training.optimizer.training_metric}",
        mode                    = "${training.optimizer.mode}",
        save_last               = "${training.checkpoint.save_last}",
        save_top_k              = "${training.checkpoint.save_top_k}",
        populate_full_signature = True
    ),

    "collector": builds(
        MetricsCollector,
        agent_count             = "${controller.flock.agent_count}",
        bounds_max              = "${environment.physics.bounds_max}",
        gravity                 = "${environment.physics.gravity}",
        metrics                 = "${training.metrics}",
        mmm                     = "${controller.mmm}",
        safety                  = "${controller.safety}",
        populate_full_signature = True
    ),

    "datamodule": builds(
        DemonstrationsDataset.as_lightning_datamodule,
        batch_size              = "${training.optimizer.batch_size}",
        controller              = "${controller}",
        environment             = "${environment}",
        generator               = "${_system.trajectory_generator}",
        hardware                = "${training.hardware}",
        murmuration             = "${_system.murmuration}",
        train_split             = "${training.optimizer.train_split}",
        populate_full_signature = True
    ),

    "early_stopping_callback": builds(
        EarlyStopping,
        monitor                 = "${training.optimizer.training_metric}",
        mode                    = "${training.optimizer.mode}",
        patience                = "${training.optimizer.early_stopping_patience}",
        populate_full_signature = True
    ),

    "logger": builds(
        WandbLogger,
        log_model               = "${training.wandb.log_model}",
        mode                    = "${training.wandb.mode}",
        name                    = "${training.wandb.run_name}",
        notes                   = "${training.wandb.notes}",
        project                 = "${training.wandb.project}",
        populate_full_signature = True
    ),

    "lr_monitor_callback": builds(
        LearningRateMonitor,
        populate_full_signature = True
    ),

    "monitoring_callback": builds(
        MonitoringCallback,
        collector               = "${_system.collector}",
        populate_full_signature = True
    ),

    "optimizer": builds(
        AdamW,
        lr                      = "${training.optimizer.learning_rate}",
        weight_decay            = "${training.optimizer.weight_decay}",
        zen_partial             = True,
        populate_full_signature = True
    ),

    "policy": builds(
        GNNPolicy,
        architecture            = "${training.architecture}",
        collector               = "${_system.collector}",
        optimizer               = "${_system.optimizer}",
        scheduler               = "${_system.scheduler}",
        scheduler_metric        = "${training.optimizer.scheduler_metric}",
        training_metric         = "${training.optimizer.training_metric}",
        populate_full_signature = True
    ),

    "scheduler": builds(
        ReduceLROnPlateau,
        factor                  = "${training.optimizer.lr_factor}",
        patience                = "${training.optimizer.lr_patience}",
        mode                    = "${training.optimizer.mode}",
        zen_partial             = True,
        populate_full_signature = True
    ),

    "trainer": builds(
        Trainer,
        accelerator             = "${training.hardware.accelerator}",
        benchmark               = "${training.hardware.benchmark}",
        callbacks               = [
            "${_system.checkpoint_callback}",
            "${_system.early_stopping_callback}",
            "${_system.monitoring_callback}"
        ],
        detect_anomaly          = "${training.hardware.detect_anomaly}",
        deterministic           = "${training.hardware.deterministic}",
        devices                 = "${training.hardware.devices}",
        gradient_clip_val       = "${training.optimizer.gradient_clip_val}",
        log_every_n_steps       = "${training.optimizer.log_every_n_steps}",
        logger                  = "${_system.logger}",
        max_epochs              = "${training.optimizer.max_epochs}",
        precision               = "${training.hardware.precision}",
        profiler                = "${training.metrics.profiler}",
        strategy                = "${training.hardware.strategy}",
        val_check_interval      = "${training.optimizer.val_check_interval}",
        populate_full_signature = True
    )

}
