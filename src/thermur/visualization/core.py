"""
3D visualization display manager for the Thermur simulation.

This module provides the `Visualizer` class, which serves as the primary interface 
for creating and updating 3D visualizations of the swarm simulation. It coordinates
the rendering of swarm agents, thermal fields, wind vectors, safety boundaries,
and communication graphs.
"""
import pyvista as pv
import torch

from pathlib            import Path
from pyvista            import Plotter
from tensordict         import TensorDictBase
from torch              import Tensor
from typing             import Optional

from .colors     import create_temperature_colormap
from .renderers  import (
    render_agents,
    render_communication_graph,
    render_safety_boundary, 
    render_temperature_field,
    render_wind_field
)
from .sampling   import create_temperature_grid, create_wind_grid


class Visualizer:
    """
    Main visualization class for the Thermur simulation.
    
    This class manages a PyVista plotter window and provides methods to render
    and update various aspects of the simulation. It handles the rendering of
    agents as glyphs, thermal coloring, wind field visualization, safety
    boundaries, and communication graph visualization.
    
    The visualizer maintains internal state for efficient updates, only recreating
    elements when necessary, and supports toggling different visualization
    components on and off as needed.
    """
    
    def __init__(
        self,
        config,
        environment = None,
    ):
        """
        Initialize the visualizer with the configuration.
        
        Creates a PyVista rendering window and sets up the initial visualization
        state based on the provided configuration. The environment is accessed
        through the configuration system rather than passed directly, ensuring
        proper dependency management.
        
        Args:
            config      : Configuration object containing visualization settings
                          and access to the environment
            environment : Optional direct environment reference (deprecated)
        """
        self.config      = config
        self.environment = environment
        
        # Initialize rendering state
        self._plotter       = None
        self._agent_actors  = None
        self._wind_actors   = None
        self._safety_actors = None
        self._graph_actors  = None
        self._colormap      = None
        
        # Initialize the plotter
        self._initialize_plotter()
    
    def _initialize_plotter(self):
        """
        Initialize the PyVista plotter with appropriate settings.
        
        This sets up the 3D rendering environment with the correct theme,
        lighting, and camera position for visualizing the swarm.
        """
        theme = pv.themes.DarkTheme() if self.config.dark_mode else pv.themes.DocumentTheme()
        pv.global_theme.load_theme(theme)
        
        self._plotter = Plotter(
            window_size = self.config.window_size,
            title       = self.config.window_title,
            lighting    = "three lights",
            off_screen  = False,
        )
        
        # Set up initial view
        self._plotter.camera_position = 'xy'
        self._plotter.camera.zoom(1.5)
        
        # Create colormap based on configuration
        self._colormap = create_temperature_colormap(
            cmap_name = self.config.colormap
        )
    
    def update(self, observation: TensorDictBase) -> None:
        """
        Update the visualization with new simulation data.
        
        This method is called during the simulation loop to update the
        visualization with the latest state of the swarm. It updates all
        active visualization elements, including agent positions, thermal
        coloring, wind field, safety boundaries, and communication graph.
        
        Args:
            observation: A TensorDict containing the current simulation state
        """
        if self._plotter is None or not self._plotter.window_exists:
            self._initialize_plotter()
        
        # Extract tensor data from observation
        position         = observation.get("position")
        velocity         = observation.get("velocity")
        temperature      = observation.get("temperature")
        temperature_grad = observation.get("temperature_grad")
        edge_index       = observation.get("edge_index")
        
        # Skip if window was closed
        if not self._plotter.window_exists:
            return
            
        self._plotter.clear_actors()
        
        # Render agent glyphs if enabled
        if self.config.show_agents:
            self._agent_actors = render_agents(
                plotter      = self._plotter,
                position     = position,
                velocity     = velocity,
                temperature  = temperature,
                colormap     = self._colormap if self.config.show_thermal else None,
                glyph_type   = self.config.glyph_type,
                glyph_size   = self.config.glyph_size,
                show_trails  = self.config.show_trails,
            )
        
        # Render wind field vectors if enabled
        if self.config.show_wind:
            wind_grid = create_wind_grid(
                environment     = self.environment,
                position        = position,
                grid_resolution = self.config.wind_grid_resolution,
                padding         = self.config.grid_padding,
            )
            self._wind_actors = render_wind_field(
                plotter     = self._plotter,
                wind_grid   = wind_grid,
            )
        
        # Render thermal safety boundary if enabled
        if self.config.show_safety:
            self._safety_actors = render_safety_boundary(
                plotter          = self._plotter,
                position         = position,
                temperature      = temperature,
                temperature_grad = temperature_grad,
                max_temperature  = self.environment.config.safety.max_temperature,
            )
        
        # Render communication graph edges if enabled
        if self.config.show_graph:
            self._graph_actors = render_communication_graph(
                plotter    = self._plotter,
                position   = position,
                edge_index = edge_index,
            )
    
    def render(self) -> None:
        """
        Render the current visualization state.
        
        This method should be called after `update()` to actually display
        the visualization. It triggers a render pass in the PyVista plotter.
        """
        if self._plotter is not None and self._plotter.window_exists:
            self._plotter.render()
    
    def toggle_agents(self, show: Optional[bool] = None) -> bool:
        """
        Toggle visibility of agent glyphs.
        
        Args:
            show: Explicit visibility state or None to toggle current state
        
        Returns:
            New visibility state
        """
        self.config.show_agents = not self.config.show_agents if show is None else show
        return self.config.show_agents
    
    def toggle_graph(self, show: Optional[bool] = None) -> bool:
        """
        Toggle visibility of communication graph visualization.
        
        Args:
            show: Explicit visibility state or None to toggle current state
        
        Returns:
            New visibility state
        """
        self.config.show_graph = not self.config.show_graph if show is None else show
        return self.config.show_graph
    
    def toggle_safety(self, show: Optional[bool] = None) -> bool:
        """
        Toggle visibility of safety boundary visualization.
        
        Args:
            show: Explicit visibility state or None to toggle current state
        
        Returns:
            New visibility state
        """
        self.config.show_safety = not self.config.show_safety if show is None else show
        return self.config.show_safety
        
    def toggle_thermal(self, show: Optional[bool] = None) -> bool:
        """
        Toggle thermal coloring of agents.
        
        Args:
            show: Explicit visibility state or None to toggle current state
        
        Returns:
            New visibility state
        """
        self.config.show_thermal = not self.config.show_thermal if show is None else show
        return self.config.show_thermal
    
    def toggle_wind(self, show: Optional[bool] = None) -> bool:
        """
        Toggle visibility of wind field visualization.
        
        Args:
            show: Explicit visibility state or None to toggle current state
        
        Returns:
            New visibility state
        """
        self.config.show_wind = not self.config.show_wind if show is None else show
        return self.config.show_wind
    
    def close(self) -> None:
        """
        Close the visualization window.
        
        This method should be called when the simulation is complete to
        properly clean up resources.
        """
        if self._plotter is not None and self._plotter.window_exists:
            self._plotter.close()
