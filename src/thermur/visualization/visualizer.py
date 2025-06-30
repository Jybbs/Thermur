"""
3D visualization display manager for the Thermur simulation.

This module provides the `Visualizer` class, which serves as the primary interface 
for creating and updating 3D visualizations of the swarm simulation. It coordinates
the rendering of swarm agents, thermal fields, wind vectors, safety boundaries,
and communication graphs.

The visualizer manages a PyVista plotter window and handles the lifecycle of
various visual elements, including their creation, updates, and cleanup. It
provides runtime toggles for different visualization features and supports
both light and dark themes.
"""
from __future__ import annotations
from .renderers import *
from .sampling  import create_wind_grid
from pydantic   import BaseModel
from pyvista    import Actor, Plotter
from tensordict import TensorDictBase
from typing     import Optional

import pyvista as pv


class Visualizer:
    """
    Main visualization class for the Thermur simulation.
    
    This class manages a PyVista plotter window and provides methods for rendering
    and updating various aspects of the simulation state. It serves as the central
    coordinator for all visualization elements, handling their lifecycle from
    initialization through updates to cleanup.
    
    The visualizer supports real-time rendering of:
    - Agent positions and velocities (as glyphs)
    - Temperature-based agent coloring
    - Wind field vectors
    - Thermal safety boundaries (as isosurfaces)
    - Communication graph topology
    - Agent motion trails
    
    All visualization elements can be toggled at runtime, allowing users to
    focus on specific aspects of the simulation. The rendering uses PyVista's
    efficient update mechanisms to maintain performance even with large swarms.
    """
    
    def __init__(
        self,
        colors          : BaseModel,
        glyphs          : BaseModel,
        grids           : BaseModel,
        opacity         : BaseModel,
        visualization   : BaseModel,
        max_temperature : float,
        simulation      : Optional[object] = None,
    ):
        """
        Initialize the visualizer with configuration settings.
        
        Creates the rendering window, sets up the initial visualization state,
        and configures the rendering theme based on user preferences. The
        visualizer maintains references to all rendered actors for efficient
        updates and cleanup.
        
        Args:
            colors          : Color configuration model for visualization elements
            glyphs          : Glyph configuration model for agent representation
            grids           : Grid configuration model for sampling parameters
            opacity         : Opacity configuration model for transparency settings
            visualization   : Main visualization configuration with feature toggles
            max_temperature : Maximum safe temperature (T_max) for safety visualization
            simulation      : Optional simulation reference for accessing environment data
        """
        # Store configuration models
        self.colors          = colors
        self.glyphs          = glyphs
        self.grids           = grids
        self.opacity         = opacity
        self.visualization   = visualization
        self.max_temperature = max_temperature
        self.simulation      = simulation
        
        # Initialize rendering state
        self._plotter       : Optional[Plotter]      = None
        self._agent_actors  : Optional[list[Actor]]  = None
        self._wind_actors   : Optional[list[Actor]]  = None
        self._safety_actors : Optional[list[Actor]]  = None
        self._graph_actors  : Optional[list[Actor]]  = None
        self._colormap      : Optional[str]          = None
        
        # Initialize the plotter
        self._initialize_plotter()
    
    def _initialize_plotter(self):
        """
        Set up the PyVista plotter with appropriate theme and camera settings.
        
        This method creates the visualization window with the configured size,
        applies the selected theme (dark or light), and sets up appropriate
        lighting for 3D rendering. The camera is positioned to provide a
        clear initial view of the swarm, with zoom adjusted for typical
        simulation bounds.
        
        The method also initializes the temperature colormap that will be
        used for thermal visualization of agents throughout the simulation.
        """
        theme = (
            pv.themes.DarkTheme() if self.visualization.dark_mode 
            else pv.themes.DocumentTheme()
        )
        pv.global_theme.load_theme(theme)
        
        self._plotter = Plotter(
            window_size = self.visualization.window_size,
            title       = self.visualization.window_title,
            lighting    = "three lights",
            off_screen  = False,
        )
        
        # Set up initial view
        self._plotter.camera_position = 'xy'
        self._plotter.camera.zoom(1.5)
        
        # Use colormap from configuration
        self._colormap = self.colors.colormap

    def update(self, observation: TensorDictBase) -> None:
        """
        Update the visualization with new simulation data.
        
        This method processes the latest observation data from the simulation
        and updates all active visualization elements. It efficiently manages
        the rendering pipeline by clearing previous actors and creating new
        ones based on the current configuration settings.
        
        The update process includes:
        1. Extracting tensor data from the observation
        2. Clearing previous frame's actors
        3. Conditionally rendering each visualization element
        4. Managing actor references for future updates
        
        Args:
            observation: Current simulation state containing:
                - position         : Agent positions (N, 3)
                - velocity         : Agent velocities (N, 3) 
                - temperature      : Agent temperatures (N, 1)
                - temperature_grad : Temperature gradients (N, 3)
                - edge_index       : Communication graph edges (2, E)
        """
        if self._plotter is None:
            self._initialize_plotter()
        
        # Extract tensor data from observation
        position         = observation.get("position")
        velocity         = observation.get("velocity")
        temperature      = observation.get("temperature")
        temperature_grad = observation.get("temperature_grad")
        edge_index       = observation.get("edge_index")
        
        # Skip if window was closed
        if self._plotter.ren_win is None:
            return
            
        self._plotter.clear_actors()
        
        # Render agent glyphs if enabled
        if self.visualization.show_agents:
            colormap = self._colormap if self.visualization.show_thermal else None
            self._agent_actors = render_agents(
                plotter     = self._plotter,
                position    = position,
                velocity    = velocity,
                temperature = temperature,
                colormap    = colormap,
                glyphs      = self.glyphs,
                colors      = self.colors,
                opacities   = self.opacity,
                show_trails = self.visualization.show_trails,
            )
        
        # Render wind field vectors if enabled
        if self.visualization.show_wind:
            wind_grid = create_wind_grid(
                simulation  = self.simulation,
                position    = position,
                grid_config = self.grids,
            )
            self._wind_actors = render_wind_field(
                plotter   = self._plotter,
                wind_grid = wind_grid,
                glyphs    = self.glyphs,
                colors    = self.colors,
                opacities = self.opacity,
            )
        
        # Render thermal safety boundary if enabled
        if self.visualization.show_safety and self.simulation is not None:
            self._safety_actors = render_safety_boundary(
                plotter          = self._plotter,
                position         = position,
                temperature      = temperature,
                temperature_grad = temperature_grad,
                max_temperature  = self.max_temperature,
                grids            = self.grids,
                colors           = self.colors,
                opacities        = self.opacity,
            )
        
        # Render communication graph edges if enabled
        if self.visualization.show_graph:
            self._graph_actors = render_communication_graph(
                plotter    = self._plotter,
                position   = position,
                edge_index = edge_index,
                colors     = self.colors,
                opacities  = self.opacity,
            )
    
    def render(self):
        """
        Render the current visualization state.
        
        This method triggers a render pass in the PyVista plotter to display
        the updated visualization. It should be called after update() to
        reflect changes in the display window. The method includes safety
        checks to ensure the plotter is initialized and the window is still
        open before attempting to render.
        """
        if self._plotter is not None:
            self._plotter.render()
    
    def toggle(
        self, 
        feature : str, 
        show    : Optional[bool] = None
    ) -> bool:
        """
        Toggle visibility of a visualization feature.
        
        This generic method handles toggling for all visualization elements,
        reducing code duplication. It modifies the appropriate config attribute
        based on the feature name.
        
        Args:
            feature : Name of the feature to toggle. Valid options:
                     'agents'   - Agent glyphs (spheres or arrows)
                     'graph'    - Communication graph edges
                     'safety'   - Thermal safety boundary (T_max isosurface)
                     'thermal'  - Temperature-based agent coloring
                     'wind'     - Wind field vector arrows
                     'trails'   - Agent motion trails
            show    : Explicit visibility state. If None, toggles current state.
                     If True, enables the feature. If False, disables it.
        
        Returns:
            New visibility state after the toggle operation
            
        Raises:
            ValueError: If feature name is not recognized
        """
        attr_name = f"show_{feature}"
        if not hasattr(self.visualization, attr_name):
            raise ValueError(
                f"Unknown visualization feature: '{feature}'. "
                f"Valid options: agents, graph, safety, thermal, wind, trails"
            )
        
        current   = getattr(self.visualization, attr_name)
        new_state = not current if show is None else show
        setattr(self.visualization, attr_name, new_state)
        
        return new_state
    
    def close(self):
        """
        Close the visualization window and clean up resources.
        
        This method properly shuts down the PyVista plotter and releases
        all associated resources. It should be called when the visualization
        is no longer needed, such as at the end of a training run or when
        the user requests to close the window. The method includes safety
        checks to avoid errors if the plotter is already closed.
        """
        if self._plotter is not None:
            self._plotter.close()
