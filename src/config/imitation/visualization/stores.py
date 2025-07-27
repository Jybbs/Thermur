"""
Visualization domain stores for hydra-zen configuration.

This module provides store-based configurations for visualization components
using simplified domain-level groups with minimal presets.
"""
from hydra_zen                          import store as create_store, builds
from thermur.imitation.visualization    import SwarmVisualizer

# Import schemas from __init__ for clean imports
from . import SamplingModel, VisualizerModel

# Create domain store
store = create_store()

@store(group="visualization", name="default")
def default():
    """
    Standard visualization configuration.
    
    Provides default configurations for 3D rendering with balanced
    quality and performance settings.
    """
    # Validate configurations with Pydantic
    sampling   = SamplingModel()
    visualizer = VisualizerModel()
    
    return {
        # Visualizer
        "visualizer": builds(
            SwarmVisualizer,
            simulation = "${simulation}",
            sampling   = sampling.model_dump(),
            config     = visualizer.model_dump()
        ),
        
        # Export individual configs for access
        "sampling_config"   : sampling.model_dump(),
        "visualizer_config" : visualizer.model_dump()
    }

@store(group="visualization", name="debug")
def debug():
    """
    Debug visualization configuration.
    
    Minimal configuration for rapid testing with reduced quality
    and disabled frame saving.
    """
    # Minimal configurations for debugging
    sampling = SamplingModel(
        render_every_n_steps = 50,  # Less frequent rendering
        save_frames         = False
    )
    visualizer = VisualizerModel(
        window_size      = (640, 480),  # Smaller window
        show_temperature = False,        # Skip temperature field
        agent_size       = 0.2,         # Larger agents for visibility
        colormap         = "viridis"    # Simple colormap
    )
    
    return {
        # Simplified visualizer
        "visualizer": builds(
            SwarmVisualizer,
            simulation = "${simulation}",
            sampling   = sampling.model_dump(),
            config     = visualizer.model_dump(),
            headless   = True  # No display for CI/CD
        ),
        
        # Export configs
        "sampling_config"   : sampling.model_dump(),
        "visualizer_config" : visualizer.model_dump()
    }