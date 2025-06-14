"""
Hydra-zen builder for the Thermur simulation environment.

This module defines the configuration builder for ThermurEnv, which creates
a Hydra-compatible config that instantiates the environment with validated
parameters from the EnvironmentModel Pydantic model.
"""
from configs.models import EnvironmentModel
from hydra_zen      import builds, zen
from thermur        import ThermurEnv


build_environment = builds(
    ThermurEnv,
    config                  = zen(EnvironmentModel),
    populate_full_signature = True,
    zen_dataclass           = {
        "module"   : "src.configs.builds.environment",
        "cls_name" : "EnvironmentBuild"
    }
)
