"""
Lightning configuration stores using hydra-zen.

This module provides store-based configurations for PyTorch Lightning components
using hydra-zen's decorator pattern. Each component is registered as a separate
thermur_build that can be referenced and overridden independently via Hydra's CLI.

The stores follow a flat structure where each component (optimizer, policy, 
trainer, etc.) is defined as a function decorated with @lightning(name=...).
This allows for clean interpolation references like ${lightning.optimizer}
without nested builds, improving configuration clarity and override flexibility.
"""
from .schemas                     import *
from config.utils.zen             import store, thermur_build, thermur_make_all
from pytorch_lightning            import Trainer
from pytorch_lightning.callbacks  import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers    import WandbLogger
from thermur.imitation.lightning  import DataModule, GNNPolicy, MonitoringCallback
from thermur.imitation.monitoring import MetricsModel
from torch.optim                  import AdamW
from torch.optim.lr_scheduler     import ReduceLROnPlateau

lightning    = store()(group="lightning")
architecture = ArchitectureModel()
checkpoint   = CheckpointModel()
experience   = ExperienceModel()
hardware     = HardwareModel()
metrics      = MetricsModel()
optimizer    = OptimizerModel()
wandb        = WandbModel()


@lightning(name="checkpoint_callback")
def checkpoint_callback_build():
    """
    Builder for model checkpointing callback.
    
    Saves model checkpoints at regular intervals during training, enabling
    recovery from failures and model selection. Uses Pydantic defaults for
    save frequency, directory path, and retention policy.
    
    The callback monitors the metric specified in OptimizerModel (default: 
    train/loss) to optionally save the best performing model in addition
    to periodic checkpoints.
    """
    return thermur_build(
        ModelCheckpoint,
        dirpath             = str(checkpoint.dirpath),
        every_n_train_steps = checkpoint.every_n_train_steps,
        filename            = checkpoint.filename,
        mode                = optimizer.mode,
        monitor             = optimizer.training_metric,
        save_last           = checkpoint.save_last,
        save_top_k          = checkpoint.save_top_k
    )

@lightning(name="datamodule")
def datamodule_build():
    """
    Builder for Lightning data module.
    
    Manages expert demonstration collection and replay buffer for imitation
    learning, wrapping TorchRL components in Lightning's DataModule interface.
    The module handles batch creation, shuffling, and multi-worker data loading.
    
    References external components via interpolation:
    - ${controller.expert}: Expert controller for demonstration collection
    - ${simulation.env}: Environment for trajectory rollouts
    """
    return thermur_build(
        DataModule,
        env        = "${simulation.env}",
        experience = experience,
        expert     = "${controller.expert}",
    )

@lightning(name="early_stopping_callback")
def early_stopping_callback_build():
    """
    Builder for early stopping callback.
    
    Monitors validation metrics and stops training if no improvement is seen
    for a specified number of epochs, preventing overfitting and saving compute
    resources. Uses the same metric as checkpointing for consistency.
    """
    return thermur_build(
        EarlyStopping,
        mode     = optimizer.mode,
        monitor  = optimizer.training_metric,
        patience = optimizer.early_stopping_patience
    )

@lightning(name="logger")
def logger_build():
    """
    Builder for Weights & Biases logger.
    
    Creates a WandbLogger for experiment tracking if mode is not 'disabled'.
    Returns None when W&B integration is disabled, allowing for offline
    training or debugging without external dependencies.
    
    The logger automatically tracks hyperparameters, metrics, and optionally
    model checkpoints based on the log_model setting.
    """
    if wandb.mode != "disabled":
        return thermur_build(
            WandbLogger, 
            log_model = wandb.log_model,
            mode      = wandb.mode,
            project   = wandb.project
        )
    return None

@lightning(name="lr_monitor_callback")
def lr_monitor_callback_build():
    """
    Builder for learning rate monitor callback.
    
    Tracks learning rate changes during training, particularly useful when
    using schedulers like ReduceLROnPlateau. Logs LR at step granularity
    for detailed optimization analysis.
    """
    return thermur_build(
        LearningRateMonitor, 
        logging_interval = metrics.logging_interval
    )

@lightning(name="monitoring_callback")
def monitoring_callback_build():
    """
    Builder for unified monitoring callback.
    
    Integrates with the monitoring domain to track custom metrics and events
    during training. References external monitoring components that handle
    safety violations, control interventions, and performance metrics.
    
    References:
    - ${monitoring.events}: Event logger for safety violations
    - ${monitoring.collector}: Metrics collector for performance tracking
    """
    return thermur_build(
        MonitoringCallback,
        collector = "${monitoring.collector}",
        events    = "${monitoring.events}"
    )

@lightning(name="optimizer")
def optimizer_build():
    """
    Builder for AdamW optimizer.
    
    Creates an AdamW optimizer with learning rate and weight decay from
    OptimizerModel defaults. The zen_partial flag (enabled by default in our
    custom thermur_build function) allows the model parameters to be injected at runtime.
    
    This can be swapped for other optimizers (SGD, Adam, etc.) via CLI overrides.
    """
    return thermur_build(
        AdamW,
        lr           = optimizer.learning_rate,
        weight_decay = optimizer.weight_decay
    )

@lightning(name="policy")
def policy_build():
    """
    Builder for the Graph Neural Network policy.
    
    Creates a permutation-equivariant GNN that processes the flock graph
    structure G_t = (V, E_t) to output nominal control actions u_nom for 
    each agent. The architecture uses message passing layers with GRU-based
    state updates.
    
    The policy configures its optimizer using the provided optimizer and
    scheduler configurations from the lightning domain.
    
    References:
    - ${monitoring.collector}: Metrics collector for loss computation
    - ${lightning.optimizer}: Optimizer configuration
    - ${lightning.scheduler}: Learning rate scheduler configuration
    """
    return thermur_build(
        GNNPolicy,
        architecture     = architecture,
        collector        = "${monitoring.collector}",
        optimizer        = "${lightning.optimizer}",
        scheduler        = "${lightning.scheduler}",
        scheduler_metric = optimizer.scheduler_metric,
        training_metric  = optimizer.training_metric
    )

@lightning(name="scheduler")
def scheduler_build():
    """
    Builder for ReduceLROnPlateau learning rate scheduler.
    
    Creates a learning rate scheduler that reduces the learning rate when
    the monitored metric plateaus. The zen_partial flag (enabled by default)
    allows the optimizer instance to be injected at runtime.
    
    The scheduler monitors the validation version of the metric specified
    in OptimizerModel and reduces learning rate by lr_factor after patience
    epochs without improvement.
    """
    return thermur_build(
        ReduceLROnPlateau,
        factor   = optimizer.lr_factor,
        mode     = optimizer.mode,
        patience = optimizer.lr_patience
    )

@lightning(name="trainer")
def trainer_build():
    """
    Builder for PyTorch Lightning Trainer.
    
    Configures the Lightning Trainer with automatic mixed precision, gradient
    clipping, logging, and checkpointing. Handles all training loop boilerplate
    including device placement, backward passes, and metric tracking.
    
    Callbacks are referenced via interpolation to allow independent override
    of each callback's configuration. The trainer uses hardware settings from
    HardwareModel and training parameters from OptimizerModel.
    """
    return thermur_build(
        Trainer,
        accelerator          = hardware.accelerator,
        benchmark            = hardware.benchmark,
        callbacks            = [
            "${lightning.checkpoint_callback}",
            "${lightning.early_stopping_callback}",
            "${lightning.lr_monitor_callback}",
            "${lightning.monitoring_callback}"
        ],
        detect_anomaly       = hardware.detect_anomaly,
        deterministic        = hardware.deterministic,
        devices              = hardware.devices,
        gradient_clip_val    = optimizer.gradient_clip_val,
        log_every_n_steps    = metrics.log_every_n_steps,
        logger               = "${lightning.logger}",
        max_epochs           = optimizer.max_epochs,
        precision            = hardware.precision,
        profiler             = metrics.profiler,
        strategy             = hardware.strategy,
        val_check_interval   = optimizer.val_check_interval
    )

thermur_make_all(lightning)