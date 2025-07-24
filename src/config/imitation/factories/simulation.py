"""
Hydra-zen builder for the Thermur simulation environment.

This module defines the configuration builder for SimulationEnv, which creates
a Hydra-compatible config that instantiates the environment with validated
parameters from the PhysicsModel Pydantic model.

The environment follows dependency injection principles, so all dependencies
are provided as arguments rather than imported directly.
"""
from config.imitation.schemas.loader          import LoaderModel
from config.imitation.schemas.physics         import PhysicsModel
from hydra_zen                                import builds, zen
from omegaconf                                import SI
from thermur.imitation.simulation.environment import SimulationEnv
from thermur.imitation.simulation.loader      import WRFDataSource

build_simulation = builds(
    SimulationEnv,
    flock                   = SI("${flock}"),
    loader                  = builds(
        WRFDataSource,
        physics = zen(PhysicsModel),
        wrf     = zen(LoaderModel)
    ),
    physics                 = zen(PhysicsModel),
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
