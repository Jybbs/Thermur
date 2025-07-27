"""
Unified configuration orchestration for Thermur imitation learning.

This module provides the main entry point for Hydra-based configuration,
orchestrating all domain-specific stores and defining the top-level structure.
The configuration system uses hydra-zen's store pattern for clean, modular
configurations while maintaining Pydantic validation for type safety.

The system is organized by domains:
- Lightning: PyTorch Lightning training components
- Simulation: MuJoCo environment and physics
- Controller: Expert controller and safety systems  
- Monitoring: Metrics and event tracking
- Visualization: 3D rendering and display

Each domain uses ZenStore for configuration management, enabling:
- Modular, swappable components via Hydra's choice system
- Type-safe configuration with Pydantic validation
- Clean CLI interface with intuitive overrides
- No YAML files - all configuration in Python
"""
from hydra_zen import make_config, store
from pathlib   import Path

# Import all domain stores
from .controller.stores    import controller as controller_store
from .lightning.stores     import lightning as lightning_store
from .monitoring.stores    import monitoring as monitoring_store
from .simulation.stores    import simulation as simulation_store
from .visualization.stores import visualization as visualization_store

# Create main imitation learning config
ImitationConfig = make_config(
    # Hydra defaults for swappable components
    defaults=[
        "_self_",
        {"lightning": "default"},
        {"simulation": "default"},
        {"controller": "default"},
        {"monitoring": "default"},
        {"visualization": "default"},
    ],
    
    # Direct configuration values
    seed           = 42,
    experiment_dir = Path("${hydra:runtime.output_dir}"),
    experiment_name = "${hydra:job.name}",
    
    # Documentation
    __doc__="""
    Main configuration for imitation learning training.
    
    This configuration orchestrates all components needed for behavioral cloning:
    simulation environment, expert controller, GNN policy, optimization strategy,
    data collection, and experiment tracking.
    
    The configuration uses Hydra's choice system for modular components, allowing
    easy swapping via CLI overrides.
    
    Example CLI usage:
        # Standard training
        thermur train
        
        # Use debug configurations
        thermur train lightning=debug simulation=debug
        
        # Override specific parameters
        thermur train lightning.optimizer.learning_rate=1e-3
        thermur train controller.flock.agent_count=50
        
        # Use presets
        thermur train +preset=quick
    """
)

# Create minimal presets (only essential ones from original design)
@store(group="preset", name="quick")
def quick_preset() -> dict:
    """Quick experiment preset for rapid testing."""
    return {
        "lightning": "debug",
        "simulation": "debug",
        "controller": "debug",
        "monitoring": "debug",
        "visualization": "debug"
    }

@store(group="preset", name="standard")
def standard_preset() -> dict:
    """Standard training preset with default settings."""
    return {
        "lightning": "default",
        "simulation": "default",
        "controller": "default",
        "monitoring": "default",
        "visualization": "default"
    }

def register_all_stores():
    """
    Register all domain configurations with Hydra's global store.
    
    This function consolidates store registration, ensuring all configurations
    are available for Hydra's resolution and override system.
    """
    stores = [
        controller_store,
        lightning_store,
        monitoring_store,
        simulation_store,
        visualization_store
    ]
    
    # Register all stores
    for store_instance in stores:
        store_instance.add_to_hydra_store()

# Register main config
main_store = store()
main_store(ImitationConfig, name="train", group="config", package="_global_")

# Auto-register all stores when module is imported
register_all_stores()
main_store.add_to_hydra_store()

# Export key components for easy access
__all__ = [
    "ImitationConfig",
    "register_all_stores"
]