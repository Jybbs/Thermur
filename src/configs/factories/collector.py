"""
Hydra-zen builder for the TorchRL data collector.

This module defines the configuration builder for SyncDataCollector, which
gathers experience from the environment using the expert policy.
"""
from ..schemas          import CollectorModel
from hydra_zen          import builds, zen
from omegaconf          import SI
from torchrl.collectors import SyncDataCollector


build_collector = builds(
    SyncDataCollector,
    create_env_fn           = SI("${environment}"),
    device                  = SI("${hyperparameters.device}"),
    policy                  = SI("${expert_policy}"),
    populate_full_signature = False,
    zen_dataclass           = {
        "module"   : "src.configs.factories.collector",
        "cls_name" : "CollectorBuild"
    },
    **zen(CollectorModel)
)
