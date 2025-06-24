"""
3D visualization display manager for the Thermur simulation.

This module provides the `Visualizer` class, which serves as the primary interface 
for creating and updating 3D visualizations of the swarm simulation. It coordinates
the rendering of swarm agents, thermal fields, wind vectors, safety boundaries,
and communication graphs.
"""
import pyvista as pv

from pyvista            import Plotter
from tensordict         import TensorDictBase
from typing             import Any, Optional

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
    
    Manages a PyVista plotter window and provides methods for rendering
    and updating various aspects of the simulation state, including agent
    glyphs, thermal fields, wind vectors, safety boundaries, and the
    communication graph topology.
    """
    
    def __init__(
        self,
        config      : Any,
        environment : Optional[Any] = None,
    ):
        """
        Initialize the visualizer with configuration settings.
        
        Creates the rendering window and sets up initial visualization state.
        
        Args:
            config      : Configuration object with visualization settings
            environment : Optional environment reference for data access
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
        Set up the PyVista plotter with appropriate theme and camera settings.
        
        Creates the visualization window with the configured size, theme, and
        lighting based on user preferences.
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

    def update(self, observation: TensorDictBase):
        """
        Update the visualization with new simulation data.
        
        Processes the latest observation data and updates all active
        visualization elements accordingly.
        
        Args:
            observation: Current simulation state containing agent data
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
                plotter   = self._plotter,
                wind_grid = wind_grid,
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
    
    def render(self):
        """
        Render the current visualization state.
        
        Triggers a render pass in the PyVista plotter to display
        the updated visualization.
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
    
    def close(self):
        """
        Close the visualization window.
        
        Properly cleans up resources when visualization is no longer needed.
        """
        if self._plotter is not None and self._plotter.window_exists:
            self._plotter.close()
