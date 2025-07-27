"""
Visualization domain stores for hydra-zen configuration.

This module provides store-based configurations for visualization components
using simplified domain-level groups with minimal presets.
"""
from hydra_zen import store, builds
from thermur.imitation.visualization import Visualizer

# Import schemas for validation
from . import SamplingModel, VisualizerModel

# Pre-configure group for all visualization configs
visualization = store(group="visualization")

@visualization(name="default")
def default():
    """
    Standard visualization configuration.
    """
    # Use Pydantic models for validation and defaults
    sampling = SamplingModel()
    viz = VisualizerModel()
    
    return dict(
        visualizer=builds(Visualizer,
            # Sampling configuration
            render_every_n_steps=sampling.render_every_n_steps,
            save_frames=sampling.save_frames,
            output_dir=sampling.output_dir,
            
            # Display settings
            window_size=viz.window_size,
            window_title=viz.window_title,
            dark_mode=viz.dark_mode,
            
            # Visual elements
            show_agents=viz.show_agents,
            show_graph=viz.show_graph,
            show_temperature=viz.show_temperature,
            
            # Agent rendering
            agent_size=viz.agent_size,
            agent_opacity=viz.agent_opacity,
            agent_color=viz.agent_color,
            
            # Temperature field
            colormap=viz.colormap,
            
            # Runtime linkage
            env="${simulation.env}"
        )
    )

@visualization(name="debug")
def debug():
    """
    Debug visualization configuration.
    """
    # Debug configurations with overrides
    sampling = SamplingModel(
        render_every_n_steps=50,  # Less frequent rendering
        save_frames=False
    )
    viz = VisualizerModel(
        window_size=(640, 480),  # Smaller window
        show_temperature=False,  # Skip temperature field for speed
        agent_size=0.2,  # Larger agents for visibility
        colormap="viridis"  # Simple colormap
    )
    
    return dict(
        visualizer=builds(Visualizer,
            # Reduced sampling for speed
            render_every_n_steps=sampling.render_every_n_steps,
            save_frames=sampling.save_frames,
            output_dir=sampling.output_dir,
            
            # Smaller window
            window_size=viz.window_size,
            window_title=viz.window_title,
            dark_mode=viz.dark_mode,
            
            # Simplified visuals
            show_agents=viz.show_agents,
            show_graph=viz.show_graph,
            show_temperature=viz.show_temperature,
            
            # Debug rendering
            agent_size=viz.agent_size,
            agent_opacity=viz.agent_opacity,
            agent_color=viz.agent_color,
            colormap=viz.colormap,
            
            # Runtime linkage
            env="${simulation.env}",
            
            # Debug mode
            headless=True  # No display for CI/CD
        )
    )