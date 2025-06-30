"""
Hydra-zen builder for monitoring and tracking configuration.

This module defines configuration builders for both Loguru logging setup
and Weights & Biases experiment tracking, which together handle all
monitoring needs for the application.
"""
from ..schemas import LoggingModel, WandbModel
from hydra_zen import builds, zen


build_monitoring = builds(
    dict,
    logging = zen(LoggingModel),
    wandb   = zen(WandbModel),
    populate_full_signature = False,
    zen_dataclass           = {
        "module"   : "src.configs.factories.monitoring",
        "cls_name" : "MonitoringConfigBuild"
    },
)
