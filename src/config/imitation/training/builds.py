"""
Training domain builds for hydra-zen configuration.

This module provides pre-built components for the training infrastructure:

Training Components:
- ExpertDataset       : PyG InMemoryDataset that generates and caches expert
                        trajectories with automatic WRF data discovery.
- GNNPolicy           : Graph Neural Network policy that processes agent
                        observations and produces control actions.
- Trainer             : PyTorch Lightning trainer with hardware configuration,
                        gradient clipping, distributed training, and callbacks.

Optimization:
- AdamW               : Adaptive optimizer with weight decay for training the policy
                        network.
- ReduceLROnPlateau   : Learning rate scheduler that reduces LR when validation
                        metrics plateau.

Callbacks:
- ModelCheckpoint     : Saves model checkpoints based on validation metrics with
                        configurable retention policies.
- EarlyStopping       : Monitors validation metrics and stops training when no
                        improvement is detected.
- LearningRateMonitor : Tracks and logs learning rate changes during training.

Logging:
- WandbLogger         : Weights & Biases integration for experiment tracking and
                        metric visualization.
"""
from __future__                   import annotations
from .schemas                     import *
from config.cli.schemas           import DisplayModel, WandbModel
from hydra_zen                    import builds, make_config
from pytorch_lightning            import Trainer
from pytorch_lightning.callbacks  import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers    import WandbLogger
from thermur.imitation.controller import ExpertDataset
from thermur.imitation.training   import CallbackFactory, GNNPolicy, MetricsFactory
from torch.optim                  import AdamW
from torch.optim.lr_scheduler     import ReduceLROnPlateau
from typing                       import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from hydra_zen.typing import Builds


TRAINING_USER_CONFIG = make_config(
    architecture = ArchitectureModel(),
    checkpoint   = CheckpointModel(),
    display      = DisplayModel(),
    hardware     = HardwareModel(),
    metrics      = MetricsModel(),
    optimizer    = OptimizerModel(),
    wandb        = WandbModel()
)

TRAINING_SYSTEM_BUILDS: dict[str, type[Builds[Any]]] = {

    "checkpoint_callback": builds(
        ModelCheckpoint,
        dirpath                 = "${training.checkpoint.dirpath}",
        every_n_train_steps     = "${training.checkpoint.every_n_train_steps}",
        filename                = "checkpoint-{step}",
        mode                    = "${training.optimizer.mode}",
        monitor                 = "validation/mse",
        save_last               = "${training.checkpoint.save_last}",
        save_top_k              = "${training.checkpoint.save_top_k}",
        populate_full_signature = True
    ),

    "datamodule": builds(
        ExpertDataset.as_lightning_datamodule,
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
        monitor                 = "validation/mse",
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
        settings                = {"quiet": "${training.wandb.quiet}"},
        populate_full_signature = True
    ),

    "lr_monitor_callback": builds(
        LearningRateMonitor,
        populate_full_signature = True
    ),

    "metrics": builds(
        MetricsFactory,
        agent_count             = "${controller.mmm.agent_count}",
        metrics                 = "${training.metrics}",
        murmuration             = "${controller.mmm}",
        physics                 = "${environment.physics}",
        safety                  = "${controller.safety}",
        populate_full_signature = True
    ),

    "model_summary_callback": builds(
        CallbackFactory.create_model_summary,
        display                 = "${training.display}",
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
        metrics                 = "${_system.metrics}",
        optimizer               = "${_system.optimizer}",
        scheduler               = "${_system.scheduler}",
        populate_full_signature = True
    ),

    "progress_bar_callback": builds(
        CallbackFactory.create_progress_bar,
        display                 = "${training.display}",
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
            "${_system.model_summary_callback}",
            "${_system.progress_bar_callback}"
        ],
        detect_anomaly          = "${training.hardware.detect_anomaly}",
        deterministic           = "${training.hardware.deterministic}",
        enable_model_summary    = False,
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
