"""
Imitation learning configuration workload.

This module defines the top-level configuration structure for training
the GNN policy via behavioral cloning from expert demonstrations.
"""
from ..factories.controller    import *
from ..factories.lightning     import *
from ..factories.monitoring    import *
from ..factories.simulation    import *
from ..factories.visualization import *
from hydra_zen                 import make_config, ZenStore

imitation_cfg = make_config(
    checkpoint  = build_checkpoint,
    controller  = build_controller,
    datamodule  = build_datamodule,
    events      = build_events,
    experience  = build_experience,
    flock       = build_flock,
    hardware    = build_hardware,
    metrics     = build_metrics,
    monitor     = build_monitor,
    optimizer   = build_optimizer,
    policy      = build_policy,
    simulation  = build_simulation,
    trainer     = build_trainer,
    visualizer  = build_visualizer,
    wandb       = build_wandb,
    defaults    = ["_self_"],
)
    

def register_imitation_cfgs():
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
        imitation_cfg, 
        group   = "config", 
        name    = "train", 
        package = "_global_"
    )
    
    store.add_to_hydra_store()
