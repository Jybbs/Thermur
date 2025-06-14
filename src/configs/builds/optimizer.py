"""
Hydra-zen builder for the AdamW optimizer.

This module defines the configuration builder for the optimizer used in
training the GNN policy through imitation learning.
"""
from configs.models import HyperparameterModel
from hydra_zen      import builds, zen
from omegaconf      import SI
from torch.optim    import AdamW


build_optimizer = builds(
    AdamW,
    params                  = SI("${policy.parameters()}"),
    lr                      = zen(HyperparameterModel).learning_rate,
    weight_decay            = zen(HyperparameterModel).weight_decay,
    populate_full_signature = False,
    zen_dataclass           = {
        "module"   : "src.configs.builds.optimizer",
        "cls_name" : "OptimizerBuild"
    }
)
