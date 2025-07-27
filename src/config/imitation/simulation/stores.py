"""
Configuration for simulation environment and physics.

This module provides hydra-zen configurations for the MuJoCo simulation
environment, WRF data loading, and physics parameters using the store pattern.
"""
from hydra_zen                    import store as create_store, builds
from thermur.imitation.simulation import SimulationEnv, WRFDataSource

# Import schemas from __init__ for clean imports
from . import LoaderModel, PhysicsModel

# Import FlockModel from controller domain
from ..controller import FlockModel

# Create domain store
store = create_store()

@store(group="simulation", name="default")
def default():
    """
    Standard simulation configuration.
    
    Provides default configurations for the MuJoCo simulation environment
    including physics settings, data loading, and flock parameters.
    """
    # Validate configurations with Pydantic
    physics = PhysicsModel()
    loader  = LoaderModel()
    flock   = FlockModel()
    
    return {
        # Environment
        "env": builds(
            SimulationEnv,
            flock   = flock.model_dump(),
            loader  = builds(
                WRFDataSource,
                physics = physics.model_dump(),
                wrf     = loader.model_dump()
            ),
            physics = physics.model_dump()
        ),
        
        # Export individual configs for access
        "physics" : physics.model_dump(),
        "loader"  : loader.model_dump(),
        "flock"   : flock.model_dump()
    }

@store(group="simulation", name="debug")
def debug():
    """
    Debug simulation configuration.
    
    Minimal configuration for rapid testing with small flock,
    synthetic data, and headless rendering.
    """
    # Minimal configurations for debugging
    physics = PhysicsModel(simulation_step=0.1)  # Faster timestep
    loader  = LoaderModel(domain_randomization=False)  # No randomization
    flock   = FlockModel(
        agent_count            = 3,
        communication_range    = 10.0,
        formation_scale_factor = 0.3
    )
    
    # Create synthetic loader
    loader_dict = loader.model_dump()
    loader_dict["synthetic"] = True
    
    return {
        # Headless environment for speed
        "env": builds(
            SimulationEnv,
            flock    = flock.model_dump(),
            loader   = builds(
                WRFDataSource,
                physics = physics.model_dump(),
                wrf     = loader_dict
            ),
            physics  = physics.model_dump(),
            headless = True
        ),
        
        # Export configs
        "physics" : physics.model_dump(),
        "loader"  : loader_dict,
        "flock"   : flock.model_dump()
    }