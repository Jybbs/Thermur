"""
Imitation learning configuration workload.

This module defines the top-level configuration structure for training
the GNN policy via behavioral cloning from expert demonstrations.
"""
from ..factories import *
from hydra_zen   import make_config, ZenStore

imitation_config = make_config(
    controller        = build_controller,
    data_collector    = build_data_collector,
    environment       = build_environment,
    experience_buffer = build_experience_buffer,
    flock             = build_flock,
    learning          = build_learning,
    logging           = build_logging,
    loss              = build_loss,
    optimizer         = build_optimizer,
    physics           = build_physics,
    policy            = build_policy,
    visualizer        = build_visualizer,
    wandb             = build_wandb,
    defaults          = ["_self_"],
)
    

def register_imitation_configs():
    """
    Register imitation learning configurations with Hydra's ConfigStore.
    
    This function adds the complete imitation learning configuration to Hydra's
    global ConfigStore under the name "train". The configuration orchestrates
    all components needed for behavioral cloning: the simulation environment,
    expert controller, GNN policy, optimizer, loss function, data collection,
    and visualization.
    
    The registered config enables training runs via:
    
        @hydra.main(version_base=None, config_path="...", config_name="train")
    """
    store = ZenStore(overwrite_ok=True)
    
    store(
        imitation_config, 
        name    = "train", 
        group   = "config", 
        package = "_global_"
    )
    
    store.add_to_hydra_store()
