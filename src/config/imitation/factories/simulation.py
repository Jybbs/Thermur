"""
Hydra-zen builder for the Thermur simulation environment.

This module defines the configuration builder for SimulationEnv, which creates
a Hydra-compatible config that instantiates the environment with validated
parameters from the PhysicsModel Pydantic model.

The environment follows dependency injection principles, so all dependencies
are provided as arguments rather than imported directly.
"""
from config.imitation.schemas.physics         import PhysicsModel
from hydra_zen                                import builds
from omegaconf                                import SI
from thermur.imitation.simulation.environment import SimulationEnv
from thermur.imitation.sources.loaders        import WRFDataSource


build_physics = builds(
    PhysicsModel,
    populate_full_signature = True,
    zen_dataclass           = {
        "module"   : "src.configs.imitation.factories.simulation",
        "cls_name" : "PhysicsBuild"
    }
)
"""
Builder for physics configuration.

Defines physical simulation parameters and thermal constraints for
the environment.
"""

build_data_source = builds(
        WRFDataSource,
        data_path               = SI("${source.data_path}"),
        physics                 = SI("${physics}"),
        wrf_data                = SI("${source}"),
        populate_full_signature = True,
    )
"""
Builder for the environmental data source.

Loads and interpolates time-varying temperature and wind field data
from external wildfire simulations (e.g., WRF-Fire outputs).
"""

build_simulation = builds(
    SimulationEnv,
    data_source             = build_data_source,
    flock                   = SI("${flock}"),
    physics                 = SI("${physics}"),
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
