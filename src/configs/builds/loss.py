"""
Hydra-zen builder for the imitation learning loss module.

This module defines the configuration builder for ImitationLoss, which
computes the MSE between predicted and expert actions for behavioral cloning.
"""
from hydra_zen   import builds
from omegaconf   import SI
from src.thermur import ImitationLoss


loss_config = builds(
    ImitationLoss,
    policy_network          = SI("${gnn_policy}"),  # Reference to instantiated GNN policy
    populate_full_signature = True,
    zen_dataclass           = {
        "module"   : "src.configs.builds.loss",
        "cls_name" : "LossConfig"
    }
)
