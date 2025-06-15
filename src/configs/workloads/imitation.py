"""
Imitation learning configuration for Thermur.

This module defines the configuration for training the GNN policy via 
behavioral cloning from the expert flocking controller.
"""
from ..factories import *
from ..schemas   import *
from hydra_zen   import builds, make_config, ZenStore


imitation_config = make_config(
    # Parameter models
    hyperparameters = builds(HyperparameterModel),
    collector       = builds(CollectorModel),
    replay_buffer   = builds(ReplayBufferModel),
    logging         = builds(LoggingModel),
    wandb           = builds(WandbModel),
    environment     = builds(EnvironmentModel),
    swarm           = builds(SwarmModel),
    agent           = builds(AgentModel),
    
    # Component builders
    simulation        = build_environment,
    expert_policy     = build_expert_controller,
    policy            = build_policy,
    data_collector    = build_collector,
    experience_buffer = build_replay_buffer,
    loss_function     = build_loss,
    optimizer         = build_optimizer,
    
    # Hydra defaults
    defaults = ["_self_"],
)

def register_configs():
    """
    Register all configurations with Hydra's ConfigStore.
    """
    store = ZenStore(overwrite_ok=True)
    
    # Register the main training config
    store(
        imitation_config, 
        name    = "train", 
        group   = "config", 
        package = "_global_"
    )
    
    # Add to Hydra
    store.add_to_hydra_store()
