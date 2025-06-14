"""
Hydra-zen builder for the TorchRL data collector.

This module defines the configuration builder for SyncDataCollector, which
gathers experience from the environment using the expert policy.
"""
from configs.models     import CollectorModel, HyperparameterModel
from hydra_zen          import builds, zen
from omegaconf          import SI
from torchrl.collectors import SyncDataCollector


build_collector = builds(
    SyncDataCollector,
    create_env_fn           = SI("${lambda: environment}"),
    policy                  = SI("${expert_policy}"),
    total_frames            = zen(CollectorModel).total_frames,
    frames_per_batch        = zen(CollectorModel).frames_per_batch,
    device                  = zen(HyperparameterModel).device,
    populate_full_signature = False,
    zen_dataclass           = {
        "module"   : "src.configs.builds.collector",
        "cls_name" : "CollectorBuild"
    }
)
