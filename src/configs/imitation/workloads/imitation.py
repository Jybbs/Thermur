"""
Imitation learning configuration workload.

This module defines the top-level configuration structure for training
the GNN policy via behavioral cloning from expert demonstrations.
"""
from ..schemas import *
from hydra_zen import make_config, zen, ZenStore


def register_configs():
    """
    Register all configurations with Hydra's ConfigStore.
    """
    # Import factories here to avoid circular imports
    from ..factories import (
        build_collector,
        build_replay_buffer,
        build_flocking_controller,
        build_loss,
        build_optimizer,
        build_policy,
        build_simulation,
        build_visualizer,
    )
    
    imitation_config = make_config(
        # Configuration models (validated but not instantiated)
        control         = zen(ControlModel),
        learning        = zen(LearningModel),
        logging         = zen(LoggingModel),
        physics         = zen(PhysicsModel),
        safety          = zen(SafetyModel),
        swarm           = zen(SwarmModel),
        visualization   = zen(VisualizationModel),
        wandb           = zen(WandbModel),
        
        # Instantiatable components
        data_collector    = build_collector,
        experience_buffer = build_replay_buffer,
        expert_policy     = build_flocking_controller,
        loss_function     = build_loss,
        optimizer         = build_optimizer,
        policy            = build_policy,
        simulation        = build_simulation,
        visualizer        = build_visualizer,
        
        # Hydra defaults
        defaults = ["_self_"],
    )
    
    store = ZenStore(overwrite_ok=True)
    
    store(
        imitation_config, 
        name    = "train", 
        group   = "config", 
        package = "_global_"
    )
    
    store.add_to_hydra_store()

# Create module-level config for backwards compatibility
imitation_config = None
