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

Defines the physical constraints and dynamics of the simulation environment,
including gravitational acceleration, spatial boundaries, time discretization,
and temperature thresholds used throughout the simulation and metrics.
"""

build_loader = builds(
    WRFDataSource,
    physics                 = SI("${physics}"),
    wrf                     = zen(LoaderModel),
    populate_full_signature = True,
    zen_dataclass           = {
        "module"   : "src.configs.imitation.factories.simulation",
        "cls_name" : "LoaderBuild"
    }
)
"""
Builder for WRF-Fire data loader.

Configures the data pipeline that ingests NetCDF files from wildfire simulations,
extracting temperature fields, wind vectors, and fire heat flux. Supports domain
randomization and noise injection for robust policy training.
"""

build_simulation = builds(
    SimulationEnv,
    flock                   = SI("${flock}"),
    loader                  = SI("${loader}"),
    physics                 = SI("${physics}"),
    populate_full_signature = True,
    zen_dataclass           = {
        "module"   : "src.configs.imitation.factories.simulation",
        "cls_name" : "SimulationBuild"
    }
)
"""
Builder for the MuJoCo simulation environment.

Orchestrates the complete simulation pipeline: initializes the flock formation,
steps the physics engine, queries environmental hazards from WRF data, computes
agent observations including temperature gradients, and maintains the dynamic
communication graph based on proximity.
"""
