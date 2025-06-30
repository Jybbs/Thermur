"""
Imitation learning configuration workload.

This module defines the top-level configuration structure for training
the GNN policy via behavioral cloning from expert demonstrations.
"""
from ..factories import *
from ..schemas   import *
from hydra_zen   import make_config, zen, ZenStore


imitation_config = make_config(
    # Configuration models (validated but not instantiated)
    agent           = zen(AgentModel),
    cbf             = zen(CBFModel),
    data            = zen(DataConfig),
    environment     = zen(EnvironmentModel),
    gnn             = zen(GNNConfig),
    logging         = zen(LoggingModel),
    qp_solver       = zen(QPSolverModel),
    swarm           = zen(SwarmModel),
    training        = zen(TrainingConfig),
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

def register_configs():
    """
    Register all configurations with Hydra's ConfigStore.
    """
    store = ZenStore(overwrite_ok=True)
    
    store(
        imitation_config, 
        name    = "train", 
        group   = "config", 
        package = "_global_"
    )
    
    store.add_to_hydra_store()
