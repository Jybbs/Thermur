"""
Hydra-zen builder for the Thermur simulation environment.

This module defines the configuration builder for ThermurEnv, which creates
a Hydra-compatible config that instantiates the environment with validated
parameters from the EnvironmentConfig Pydantic model.
"""
from hydra_zen   import builds, zen
from src.configs import EnvironmentConfig
from src.thermur import ThermurEnv


env_config = builds(
    ThermurEnv,
    config                  = zen(EnvironmentConfig),
    populate_full_signature = True,
    zen_dataclass           = {
        "module"   : "src.configs.builds.environment",
        "cls_name" : "EnvConfig"
    }
)
