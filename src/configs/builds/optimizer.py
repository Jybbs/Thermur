"""
Hydra-zen builder for the AdamW optimizer.

This module defines the configuration builder for the optimizer used in
training the GNN policy through imitation learning.
"""
from hydra_zen    import builds
from omegaconf    import SI
from torch.optim  import AdamW


build_optimizer = builds(
    AdamW,
    params                  = SI("${policy.parameters()}"),
    lr                      = SI("${hyperparameters.learning_rate}"),
    weight_decay            = SI("${hyperparameters.weight_decay}"),
    populate_full_signature = False,
    zen_dataclass           = {
        "module"   : "src.configs.builds.optimizer",
        "cls_name" : "OptimizerBuild"
    }
)
