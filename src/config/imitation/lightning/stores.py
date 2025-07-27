"""
Lightning domain stores for hydra-zen configuration.

This module provides store-based configurations for PyTorch Lightning components
using simplified domain-level groups with minimal presets.
"""
from hydra_zen                          import store as create_store, builds
from pytorch_lightning                  import Trainer
from pytorch_lightning.callbacks        import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers          import WandbLogger
from thermur.imitation.lightning        import DataModule, GNNPolicy, MonitoringCallback
from torch.optim                        import Adam

# Import schemas from __init__ for clean imports
from . import (
    ArchitectureModel,
    CheckpointModel, 
    ExperienceModel,
    HardwareModel,
    OptimizerModel,
    WandbModel
)

# Create domain store
store = create_store()

@store(group="lightning", name="default")
def default():
    """
    Standard Lightning configuration.
    
    Provides default configurations for all PyTorch Lightning components
    including model architecture, optimization, hardware settings, and callbacks.
    """
    # Validate configurations with Pydantic
    architecture = ArchitectureModel()
    optimizer    = OptimizerModel()
    hardware     = HardwareModel()
    checkpoint   = CheckpointModel()
    experience   = ExperienceModel()
    wandb        = WandbModel()
    
    return {
        # Model configuration
        "policy": builds(
            GNNPolicy,
            architecture = architecture.model_dump(),
            optimizer    = builds(
                Adam,
                lr           = optimizer.learning_rate,
                weight_decay = optimizer.weight_decay
            ),
            collector    = "${collector}"
        ),
        
        # Data module
        "datamodule": builds(
            DataModule,
            controller = "${controller}",
            env        = "${simulation}",
            experience = experience.model_dump()
        ),
        
        # Trainer
        "trainer": builds(
            Trainer,
            # Hardware settings
            accelerator  = hardware.accelerator,
            devices      = hardware.devices,
            precision    = hardware.precision,
            strategy     = hardware.strategy,
            
            # Training settings  
            max_epochs        = 100,
            gradient_clip_val = optimizer.gradient_clip_val,
            log_every_n_steps = 50,
            
            # Callbacks
            callbacks = [
                builds(
                    ModelCheckpoint,
                    dirpath             = str(checkpoint.dirpath),
                    filename            = checkpoint.filename,
                    every_n_train_steps = checkpoint.every_n_train_steps,
                    save_last           = checkpoint.save_last,
                    monitor             = optimizer.metric,
                    mode                = optimizer.mode
                ),
                builds(
                    EarlyStopping,
                    monitor  = optimizer.metric,
                    mode     = optimizer.mode,
                    patience = optimizer.early_stopping_patience
                ),
                builds(
                    LearningRateMonitor,
                    logging_interval = "step"
                ),
                builds(
                    MonitoringCallback,
                    events    = "${events}",
                    collector = "${collector}"
                )
            ],
            
            # Logger
            logger = builds(
                WandbLogger,
                project   = wandb.project,
                log_model = wandb.log_model,
                mode      = wandb.mode,
                save_dir  = str(checkpoint.dirpath)
            ),
            
            # Other settings
            enable_model_summary = True,
            enable_progress_bar  = True
        ),
        
        # Export individual configs for access
        "architecture" : architecture.model_dump(),
        "optimizer"    : optimizer.model_dump(),
        "hardware"     : hardware.model_dump(),
        "checkpoint"   : checkpoint.model_dump(),
        "experience"   : experience.model_dump(),
        "wandb"        : wandb.model_dump()
    }

@store(group="lightning", name="debug")
def debug():
    """
    Debug Lightning configuration.
    
    Minimal configuration for rapid testing and debugging with reduced
    model size, limited batches, and CPU execution.
    """
    # Minimal configurations for debugging
    architecture = ArchitectureModel(hidden_dim=32, num_layers=2)
    optimizer    = OptimizerModel(learning_rate=3e-4)
    hardware     = HardwareModel(accelerator="cpu", devices=1, precision="32-true")
    checkpoint   = CheckpointModel(every_n_train_steps=100)
    experience   = ExperienceModel(batch_size=32, buffer_size=1000, total_frames=1000)
    wandb        = WandbModel(mode="disabled")
    
    return {
        # Simplified model
        "policy": builds(
            GNNPolicy,
            architecture = architecture.model_dump(),
            optimizer    = builds(Adam, lr=optimizer.learning_rate),
            collector    = "${collector}"
        ),
        
        # Data module with small batches
        "datamodule": builds(
            DataModule,
            controller = "${controller}",
            env        = "${simulation}",
            experience = experience.model_dump()
        ),
        
        # Debug trainer
        "trainer": builds(
            Trainer,
            # Hardware
            accelerator = hardware.accelerator,
            devices     = hardware.devices,
            precision   = hardware.precision,
            
            # Debug settings
            max_epochs          = 2,
            limit_train_batches = 10,
            limit_val_batches   = 5,
            log_every_n_steps   = 1,
            detect_anomaly      = True,
            
            # Minimal callbacks
            callbacks = [
                builds(
                    ModelCheckpoint,
                    dirpath             = str(checkpoint.dirpath),
                    every_n_train_steps = checkpoint.every_n_train_steps,
                    save_last           = True
                )
            ],
            
            # No logger for debugging
            logger = None,
            
            # UI settings
            enable_model_summary = True,
            enable_progress_bar  = True
        ),
        
        # Export configs
        "architecture" : architecture.model_dump(),
        "optimizer"    : optimizer.model_dump(),
        "hardware"     : hardware.model_dump(),
        "checkpoint"   : checkpoint.model_dump(),
        "experience"   : experience.model_dump(),
        "wandb"        : wandb.model_dump()
    }