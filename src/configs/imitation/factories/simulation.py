"""
Hydra-zen builder for the Thermur simulation environment.

This module defines the configuration builder for SimulationEnv, which creates
a Hydra-compatible config that instantiates the environment with validated
parameters from the PhysicsModel Pydantic model.

The environment follows dependency injection principles, so all dependencies
are provided as arguments rather than imported directly.
"""
from ..schemas          import PhysicsModel
from .flock             import build_action_spec, build_observation_spec
from hydra_zen          import builds, zen
from omegaconf          import SI
from thermur.simulation import compute_edge_index, SimulationEnv
from thermur.utils      import EnvironmentDataSource, set_seed


build_data_source = builds(
        EnvironmentDataSource,
        data_path      = SI("${physics.thermal_data_source}"),
        physics_config = zen(PhysicsModel),
        populate_full_signature = True,
    )
"""
Builder for the environmental data source.

Loads and interpolates time-varying temperature and wind field data
from external wildfire simulations (e.g., WRF-Fire outputs).
"""

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

    # Flock parameters from schema
    agent_count             = SI("${flock.agent_count}"),
    communication_range     = SI("${flock.communication_range}"),
    formation_scale_factor  = SI("${flock.formation_scale_factor}"),
    initial_formation       = SI("${flock.initial_formation}"),
    spatial_dims            = SI("${flock.spatial_dims}"),
    populate_full_signature = True,
    zen_dataclass           = {
        "module"   : "src.configs.imitation.factories.simulation",
        "cls_name" : "SimulationBuild"
    }
)
"""
Builder for the main simulation environment.

Creates a TorchRL-compatible environment managing multi-agent flock dynamics
within MuJoCo physics, integrating real-time environmental hazards.
"""
