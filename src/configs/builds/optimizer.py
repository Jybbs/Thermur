"""
Hydra-zen builder for the AdamW optimizer.

This module defines the configuration builder for the optimizer used in
training the GNN policy through imitation learning.
"""
from hydra_zen   import builds, zen
from omegaconf   import SI
from src.configs import TrainConfig
from torch.optim import AdamW


optimizer_config = builds(
    AdamW,
    params                  = SI("${loss_module.parameters()}"),  # Get parameters from loss module
    lr                      = SI("${train_config.learning_rate}"),
    weight_decay            = SI("${train_config.weight_decay}"),
    populate_full_signature = False,  # AdamW has many optional params we don't need
    zen_dataclass           = {
        "module"   : "src.configs.builds.optimizer",
        "cls_name" : "OptimizerConfig"
    }
)
