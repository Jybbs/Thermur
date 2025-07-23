"""
Hydra-zen builders for PyTorch Lightning components.

This module provides configuration builders for Lightning-specific training
components including the GNN policy network, data module, trainer, and callbacks.
"""
from config.imitation.schemas.learning      import LearningModel
from hydra_zen                              import builds, zen
from omegaconf                              import SI
from pytorch_lightning                      import Trainer
from pytorch_lightning.callbacks            import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers              import WandbLogger
from thermur.imitation.lightning.experience import ExperienceModule
from thermur.imitation.lightning.policy     import GNNPolicy


build_checkpoint_callback = builds(
    ModelCheckpoint,
    dirpath                 = SI("${learning.dirpath}"),
    every_n_train_steps     = SI("${learning.every_n_train_steps}"),
    filename                = SI("${learning.filename}"),
    save_last               = SI("${learning.save_last}"),
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

build_early_stopping_callback = builds(
    EarlyStopping,
    mode                    = SI("${learning.mode}"),
    monitor                 = SI("${learning.monitor}"),
    patience                = SI("${learning.patience}"),
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
    logging_interval        = SI("${learning.logging_interval}"),
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

build_data_module = builds(
    ExperienceModule,
    env                     = SI("${simulation}"),
    learning                = zen(LearningModel),
    policy                  = SI("${controller}"),
    populate_full_signature = True,
    zen_dataclass           = {
        "module"   : "src.configs.imitation.factories.lightning",
        "cls_name" : "DataModuleBuild"
    }
)
"""
Builder for experience data module.

Manages expert demonstration collection and replay buffer for imitation
learning, wrapping TorchRL components in Lightning's DataModule interface.
"""

build_learning = builds(
    LearningModel,
    populate_full_signature = True,
    zen_dataclass           = {
        "module"   : "src.configs.imitation.factories.lightning",
        "cls_name" : "LearningBuild"
    }
)
"""
Builder for learning configuration.

Creates a Pydantic-validated learning configuration that manages training
hyperparameters, batch sizes, and optimization settings.
"""

build_policy = builds(
    GNNPolicy,
    learning                = zen(LearningModel),
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

build_trainer = builds(
    Trainer,
    accelerator             = SI("${learning.accelerator}"),
    callbacks               = [
        build_checkpoint_callback, 
        build_early_stopping_callback,
        build_lr_monitor_callback
    ],
    devices                 = SI("${learning.devices}"),
    enable_model_summary    = SI("${learning.enable_model_summary}"),
    enable_progress_bar     = SI("${learning.enable_progress_bar}"),
    gradient_clip_val       = SI("${learning.gradient_clip_val}"),
    log_every_n_steps       = SI("${learning.log_every_n_steps}"),
    logger                  = builds(
        WandbLogger,
        log_model   = "all",
        mode        = SI("${wandb.mode}"),
        project     = SI("${wandb.project}"),
        save_dir    = SI("${learning.dirpath}")
    ),
    precision               = SI("${learning.precision}"),
    profiler                = SI("${learning.profiler}"),
    strategy                = SI("${learning.strategy}"),
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