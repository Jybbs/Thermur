"""
Configuration for simulation environment and physics.

This module provides hydra-zen configurations for the MuJoCo simulation
environment, WRF data loading, and physics parameters using the store pattern.
"""
from hydra_zen import store, builds
from thermur.imitation.simulation import SimulationEnv, WRFDataSource

# Import schemas for validation
from . import LoaderModel, PhysicsModel
from ..controller import FlockModel

# Pre-configure group for all simulation configs
simulation = store(group="simulation")

@simulation(name="default")
def default():
    """
    Standard simulation configuration.
    """
    # Use Pydantic models for validation and defaults
    physics = PhysicsModel()
    loader = LoaderModel()
    flock = FlockModel()
    
    return dict(
        env=builds(SimulationEnv,
            # Physics parameters
            simulation_step=physics.simulation_step,
            gravity=physics.gravity,
            bounds_min=physics.bounds_min,
            bounds_max=physics.bounds_max,
            
            # Flock parameters
            agent_count=flock.agent_count,
            communication_range=flock.communication_range,
            formation_scale_factor=flock.formation_scale_factor,
            initial_formation=flock.initial_formation,
            max_temperature=flock.max_temperature,
            thermal_time_constant=flock.thermal_time_constant,
            spatial_dims=flock.spatial_dims,
            
            # Data source
            data_source=builds(WRFDataSource,
                data_path=loader.data_path,
                temperature_variable=physics.temperature_variable,
                x_dimension=physics.x_dimension,
                y_dimension=physics.y_dimension,
                z_dimension=physics.z_dimension,
                u_wind_variable=loader.u_wind_variable,
                v_wind_variable=loader.v_wind_variable,
                w_wind_variable=loader.w_wind_variable,
                fire_heat_variable=loader.fire_heat_variable,
                domain_randomization=loader.domain_randomization,
                temperature_noise_std=loader.temperature_noise_std,
                wind_noise_std=loader.wind_noise_std,
                fallback_temperature=physics.fallback_temperature,
                epsilon=physics.epsilon
            ),
            
            # MuJoCo assets
            assets_dir=physics.assets_dir
        )
    )

@simulation(name="debug")
def debug():
    """
    Debug simulation configuration.
    """
    # Debug configurations with overrides
    physics = PhysicsModel(simulation_step=0.1)  # Faster timestep
    loader = LoaderModel(domain_randomization=False)  # No randomization
    flock = FlockModel(
        agent_count=3,
        communication_range=10.0,
        formation_scale_factor=0.3
    )
    
    return dict(
        env=builds(SimulationEnv,
            # Physics parameters
            simulation_step=physics.simulation_step,
            gravity=physics.gravity,
            bounds_min=physics.bounds_min,
            bounds_max=physics.bounds_max,
            
            # Small flock for testing
            agent_count=flock.agent_count,
            communication_range=flock.communication_range,
            formation_scale_factor=flock.formation_scale_factor,
            initial_formation=flock.initial_formation,
            max_temperature=flock.max_temperature,
            thermal_time_constant=flock.thermal_time_constant,
            spatial_dims=flock.spatial_dims,
            
            # Synthetic data source for testing
            data_source=builds(WRFDataSource,
                synthetic=True,  # Use synthetic data
                temperature_variable=physics.temperature_variable,
                x_dimension=physics.x_dimension,
                y_dimension=physics.y_dimension,
                z_dimension=physics.z_dimension,
                fallback_temperature=physics.fallback_temperature,
                epsilon=physics.epsilon,
                domain_randomization=loader.domain_randomization
            ),
            
            # MuJoCo assets
            assets_dir=physics.assets_dir,
            
            # Debug settings
            headless=True  # No visualization for speed
        )
    )