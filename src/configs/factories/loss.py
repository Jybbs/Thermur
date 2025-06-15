"""
Hydra-zen builder for the imitation learning loss module.

This module defines the configuration builder for ImitationLoss, which
computes the MSE between predicted and expert actions for behavioral cloning.
"""
from hydra_zen         import builds
from omegaconf         import SI
from thermur.training  import ImitationLoss


build_loss = builds(
    ImitationLoss,
    policy_network          = SI("${policy}"),
    populate_full_signature = True,
    zen_dataclass           = {
        "module"   : "src.configs.factories.loss",
        "cls_name" : "LossBuild"
    }
)
