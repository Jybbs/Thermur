"""
Simulation configuration stores using hydra-zen.

This module provides store-based configurations for simulation components
using hydra-zen's decorator pattern. Each component is registered as a separate
build that can be referenced and overridden independently via Hydra's CLI.

The stores follow a flat structure where each component (data_source, env)
is defined as a function decorated with @simulation(name=...). This allows
for clean interpolation references like ${simulation.data_source} without
nested builds, improving configuration clarity and override flexibility.
"""
from .schemas                     import *
from config.utils.zen             import store, thermur_build, thermur_make_all
from thermur.imitation.simulation import SimulationEnv, WRFDataSource

simulation = store()(group="simulation")
loader     = LoaderModel()
physics    = PhysicsModel()


@simulation(name="env")
def env_build():
    """
    Builder for the MuJoCo simulation environment.
    
    Creates the primary simulation environment that manages drone physics,
    thermal field interactions, and multi-agent dynamics. The environment
    integrates MuJoCo for rigid body dynamics with WRF data for realistic
    thermal conditions.
    
    References:
    - ${simulation.wrf}: WRF data source for thermal fields
    - ${controller.flock}: Flock configuration from controller domain
    """
    return thermur_build(
        SimulationEnv,
        flock   = "${controller.flock}",
        physics = physics,
        wrf     = "${simulation.wrf}"
    )

@simulation(name="wrf")
def wrf_build():
    """
    Builder for WRF data source.
    
    Creates a data loader that reads WRF-Fire NetCDF files and provides
    thermal field interpolation for the simulation environment. Supports
    both real WRF data and synthetic test data.
    """
    return thermur_build(
        WRFDataSource,
        loader  = loader,
        physics = physics
    )

thermur_make_all(simulation)