"""
Imitation learning configuration workload.

This module defines the top-level configuration structure for training
the GNN policy via behavioral cloning from expert demonstrations.
"""
from ..factories.data          import build_collector, build_replay_buffer
from ..factories.flocking      import build_flocking_controller
from ..factories.imitation     import build_loss, build_optimizer, build_policy
from ..factories.simulation    import build_simulation
from ..factories.visualization import build_visualizer
from hydra_zen                 import make_config, ZenStore

imitation_config = make_config(
    data_collector    = build_collector,
    experience_buffer = build_replay_buffer,
    expert_policy     = build_flocking_controller,
    loss_function     = build_loss,
    optimizer         = build_optimizer,
    policy            = build_policy,
    simulation        = build_simulation,
    visualizer        = build_visualizer,
    defaults          = ["_self_"],
)
    

def register_imitation_configs():
    """
    Register imitation configurations with Hydra's ConfigStore.
    """
    store = ZenStore(overwrite_ok=True)
    
    store(
        imitation_config, 
        name    = "train", 
        group   = "config", 
        package = "_global_"
    )
    
    store.add_to_hydra_store()
