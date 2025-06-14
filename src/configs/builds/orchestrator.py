"""
Hydra-zen builder for the training orchestrator.

This module defines the configuration builder for TrainingOrchestrator,
which manages the entire imitation learning pipeline with fully instantiated
components.
"""
from hydra_zen   import builds, zen
from omegaconf   import SI
from configs.pydantic import TrainConfig, WandbConfig
from thermur          import TrainingOrchestrator


build_orchestrator = builds(
    TrainingOrchestrator,
    env                     = SI("${env}"),
    expert_policy           = SI("${expert_policy}"),
    policy                  = SI("${gnn_policy}"),
    collector               = SI("${collector}"),
    replay_buffer           = SI("${replay_buffer}"),
    loss_module             = SI("${loss_module}"),
    optimizer               = SI("${optimizer}"),
    train_config            = zen(TrainConfig),
    wandb_config            = zen(WandbConfig),
    populate_full_signature = True,
    zen_dataclass           = {
        "module"   : "src.configs.builds.orchestrator",
        "cls_name" : "OrchestratorConfig"
    }
)
