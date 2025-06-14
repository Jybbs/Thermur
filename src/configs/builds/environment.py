"""
Hydra-zen builder for the Thermur simulation environment.

This module defines the configuration builder for SimulationEnv, which creates
a Hydra-compatible config that instantiates the environment with validated
parameters from the EnvironmentModel Pydantic model.

The environment follows dependency injection principles, so all dependencies
are provided as arguments rather than imported directly.
"""
from configs.models import EnvironmentModel
from hydra_zen      import builds, zen
from thermur        import (
    compute_edge_index,
    EnvironmentDataSource,
    set_seed,
    SimulationEnv,
    SwarmDataSpec,
)


build_action_spec = builds(
    SwarmDataSpec.get_action_spec,
    swarm_config = zen(EnvironmentModel).swarm,
)

build_data_source = builds(
    EnvironmentDataSource,
    config = zen(EnvironmentModel).data_source,
)

build_observation_spec = builds(
    SwarmDataSpec.get_observation_spec,
    swarm_config = zen(EnvironmentModel).swarm,
)

build_environment = builds(
    SimulationEnv,
    action_spec             = build_action_spec,
    compute_edge_index      = compute_edge_index,
    config                  = zen(EnvironmentModel),
    data_source             = build_data_source,
    observation_spec        = build_observation_spec,
    populate_full_signature = True,
    seed_fn                 = set_seed,
    zen_dataclass           = {
        "module"   : "src.configs.builds.environment",
        "cls_name" : "EnvironmentBuild"
    }
)
