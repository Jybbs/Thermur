"""
Entry point configuration for imitation learning training.

This module assembles only the necessary components for training the GNN policy
via behavioral cloning, avoiding unnecessary components and providing a focused
configuration for the training script.
"""
from configs.builds import *
from configs.models import *
from hydra_zen      import make_config, store, zen


imitation_train_config = make_config(

    # Training hyperparameters wrapped with zen
    hyperparameters = zen(HyperparameterModel),
    collector       = zen(CollectorModel),
    replay_buffer   = zen(ReplayBufferModel),
    logging         = zen(LoggingModel),
    wandb           = zen(WandbModel),
    
    # Component builders
    environment       = build_environment,
    expert_policy     = build_expert_controller,
    policy            = build_policy,
    data_collector    = build_collector,
    experience_buffer = build_replay_buffer,
    loss_function     = build_loss,
    optimizer         = build_optimizer,
    
    # Hydra defaults
    defaults = ["_self_"],
)

def register_imitation_training_config():
    """
    Register the imitation training configuration with Hydra's ConfigStore.
    """
    store(
        imitation_train_config,
        name    = "imitation_train",
        group   = "config",
        package = "_global_"
    )
    
    store.add_to_hydra_store(overwrite_ok=True)
