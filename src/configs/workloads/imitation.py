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
    agent           = builds(AgentModel),
    collector       = builds(CollectorModel),
    environment     = builds(EnvironmentModel),
    hyperparameters = builds(HyperparameterModel),
    logging         = builds(LoggingModel),
    replay_buffer   = builds(ReplayBufferModel),
    swarm           = builds(SwarmModel),
    visualization   = builds(VisualizationModel),
    wandb           = builds(WandbModel),
    
    # Component builders
    data_collector    = build_collector,
    environment       = build_environment,
    experience_buffer = build_replay_buffer,
    expert_policy     = build_flocking_controller,
    loss_function     = build_loss,
    optimizer         = build_optimizer,
    policy            = build_policy,
    visualizer        = build_visualizer,
    
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
