"""
Entry point configuration for imitation learning training.

This module assembles only the necessary components for training the GNN policy
via behavioral cloning, avoiding unnecessary components and providing a focused
configuration for the training script.
"""
from configs.builds import *
from configs.models import *
from hydra_zen      import builds, make_config, store


imitation_train_config = make_config(

    # Training hyperparameters as builds
    hyperparameters = builds(HyperparameterModel),
    collector       = builds(CollectorModel),
    replay_buffer   = builds(ReplayBufferModel),
    logging         = builds(LoggingModel),
    wandb           = builds(WandbModel),
    
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
    # Use a store that allows overwriting
    from hydra_zen import ZenStore
    
    zen_store = ZenStore(overwrite_ok=True)
    zen_store(
        imitation_train_config,
        name    = "imitation_train",
        group   = "config",
        package = "_global_"
    )
    zen_store.add_to_hydra_store()
