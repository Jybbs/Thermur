"""
Hydra-zen builder for the TorchRL data collector.

This module defines the configuration builder for SyncDataCollector, which
gathers experience from the environment using the expert policy.
"""
from hydra_zen          import builds, zen
from omegaconf          import SI
from configs.pydantic   import CollectorConfig
from torchrl.collectors import SyncDataCollector


build_collector = builds(
    SyncDataCollector,
    create_env_fn           = SI("${lambda: env}"),  # Wrap the instantiated env
    policy                  = SI("${expert_policy}"),
    total_frames            = SI("${collector_config.total_frames}"),
    frames_per_batch        = SI("${collector_config.frames_per_batch}"),
    device                  = SI("${train_config.device}"),
    populate_full_signature = False,  # SyncDataCollector has many optional params
    zen_dataclass           = {
        "module"   : "src.configs.builds.collector",
        "cls_name" : "CollectorConfig"
    }
)
