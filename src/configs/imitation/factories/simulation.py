"""
Hydra-zen builder for the Thermur simulation environment.

This module defines the configuration builder for SimulationEnv, which creates
a Hydra-compatible config that instantiates the environment with validated
parameters from the PhysicsModel Pydantic model.

The environment follows dependency injection principles, so all dependencies
are provided as arguments rather than imported directly.
"""
from ..schemas  import PhysicsModel
from .swarm     import build_action_spec, build_observation_spec
from hydra_zen  import builds, zen
from omegaconf  import SI
from thermur    import compute_edge_index, SimulationEnv, EnvironmentDataSource, set_seed


build_data_source = builds(
        EnvironmentDataSource,
        data_path      = SI("${physics.thermal_data_source}"),
        physics_config = zen(PhysicsModel),
        populate_full_signature = True,
    )

build_simulation = builds(
    SimulationEnv,
    action_spec             = build_action_spec,
    compute_edge_index      = compute_edge_index,
    data_source             = build_data_source,
    observation_spec        = build_observation_spec,
    seed_fn                 = set_seed,

    # Physics parameters from schema
    assets_dir              = SI("${physics.assets_dir}"),
    simulation_step         = SI("${physics.simulation_step}"),

    # Swarm parameters from schema
    agent_count             = SI("${swarm.agent_count}"),
    communication_range     = SI("${swarm.communication_range}"),
    formation_scale_factor  = SI("${swarm.formation_scale_factor}"),
    initial_formation       = SI("${swarm.initial_formation}"),
    spatial_dims            = SI("${swarm.spatial_dims}"),
    populate_full_signature = True,
    zen_dataclass           = {
        "module"   : "src.configs.factories.simulation",
        "cls_name" : "SimulationBuild"
    }
)
