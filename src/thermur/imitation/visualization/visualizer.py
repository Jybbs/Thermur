"""
3D visualization display manager for the Thermur simulation.

This module provides the `Visualizer` class, which serves as the primary interface 
for creating and updating 3D visualizations of the flock simulation. It coordinates
the rendering of flock agents, thermal fields, wind vectors, safety boundaries,
and communication graphs.

The visualizer manages a PyVista plotter window and handles the lifecycle of
various visual elements, including their creation, updates, and cleanup. It
provides runtime toggles for different visualization features and supports
both light and dark themes through pre-configured theme objects.
"""
from __future__ import annotations
from .renderer  import Renderer
from .sampler   import Sampler
from itertools  import count
from pathlib    import Path
from pyvista    import Plotter
from tensordict import TensorDictBase
from typing     import TYPE_CHECKING

if TYPE_CHECKING:
    from ..simulation.environment       import SimulationEnv
    from config.imitation.visualization import VistaModel


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
    efficient update mechanisms to maintain performance even with large flocks.
    """
    
    def __init__(
        self,
        max_temperature : float,
        plotter         : Plotter,
        renderer        : Renderer,
        sampler         : Sampler,
        simulation      : SimulationEnv,
        vista           : VistaModel
    ):
        """
        Initialize the visualizer with configuration settings.
        
        Creates the rendering window, sets up the initial visualization state,
        and configures the rendering theme based on user preferences. The
        visualizer maintains references to all rendered actors for efficient
        updates and cleanup.
        
        Args:
            max_temperature : Critical temperature threshold for safety boundary
            plotter         : Pre-built PyVista plotter window
            renderer        : Pre-built renderer for visualization elements
            sampler         : Pre-built grid sampler for spatial data sampling
            simulation      : Simulation reference for accessing environment data
            vista           : Unified visualization configuration
        """
        self.max_temperature = max_temperature
        self.plotter         = plotter
        self.renderer        = renderer
        self.sampler         = sampler
        self.simulation      = simulation
        self.vista           = vista
        
        self.agent_actors          = None
        self.frame_capture_enabled = False
        self.frame_counter         = None
        self.frame_dir             = None
        self.graph_actors          = None
        self.safety_actors         = None
        self.temperature_actors    = None
        self.wind_actors           = None
        
        self._initialize_display()
        
        if self.vista.auto_save_frames:
            self.enable_frame_capture()
    
    def _initialize_display(self):
        """
        Set up the PyVista display and camera settings.
        
        This method sets up the camera position for an optimal initial view
        of the flock. The theme has already been applied to the plotter
        during the build phase.
        """
        self.plotter.camera_position = 'xy'
        if self.plotter.camera:
            self.plotter.camera.zoom(1.5)

    def close(self):
        """
        Close the visualization window and clean up resources.
        
        This method properly shuts down the PyVista plotter and releases
        all associated resources. It should be called when the visualization
        is no longer needed, such as at the end of a training run or when
        the user requests to close the window.
        """
        self.plotter.close()

    def enable_frame_capture(self, output_dir: Path | None = None):
        """
        Enable frame capture for creating animations.
        
        Sets up the frame capture system with an output directory and
        initializes the frame counter. Once enabled, frames can be saved
        manually or automatically during rendering.
        
        Args:
            output_dir : Directory to save frames. Defaults to configured path
        """
        self.frame_dir = Path(output_dir or self.vista.frame_output_dir)
        self.frame_dir.mkdir(parents=True, exist_ok=True)
        self.frame_counter = count()
        self.frame_capture_enabled = True
    
    def render(self):
        """
        Render the current visualization state.
        
        This method triggers a render pass in the PyVista plotter to display
        the updated visualization. It should be called after update() to
        reflect changes in the display window. The method includes safety
        checks to ensure the plotter is initialized and the window is still
        open before attempting to render.
        
        If auto_save_frames is enabled, automatically captures and saves
        a screenshot after rendering.
        """
        self.plotter.render()
        
        if self.vista.auto_save_frames and self.frame_capture_enabled:
            self.save_frame()
    
    def save_frame(self, filename: str | None = None) -> Path | None:
        """
        Save the current visualization frame as an image.
        
        Captures the current state of the visualization window and saves it
        as a PNG image. Frame capture must be enabled first via
        enable_frame_capture() method.
        
        Args:
            filename : Optional custom filename. If None, uses frame counter.
                      Should not include directory path or extension.
        
        Returns:
            Path to the saved image file, or None if capture not enabled
        """
        if not self.frame_capture_enabled or self.frame_dir is None:
            return None
        
        if filename is None:
            if self.frame_counter is None:
                return None
            filename = f"frame_{next(self.frame_counter):06d}"
        
        filepath = self.frame_dir / f"{filename}.png"
        self.plotter.screenshot(filepath)
        
        return filepath
    
    def toggle(
        self, 
        feature : str, 
        show    : bool | None = None
    ) -> bool:
        """
        Toggle visibility of a visualization feature.
        
        This generic method handles toggling for all visualization elements,
        reducing code duplication. It modifies the appropriate config attribute
        based on the feature name.
        
        Args:
            feature : Name of the feature to toggle. Valid options:
                     'agents'  - Agent glyphs (spheres or arrows)
                     'graph'   - Communication graph edges
                     'safety'  - Thermal safety boundary (T_max isosurface)
                     'thermal' - Temperature-based agent coloring
                     'trails'  - Agent motion trails
                     'wind'    - Wind field vector arrows
            show    : Explicit visibility state. If None, toggles current state.
                     If True, enables the feature. If False, disables it.
        
        Returns:
            New visibility state after the toggle operation
            
        Raises:
            ValueError: If feature name is not recognized
        """
        attr_name = f"show_{feature}"
        if not hasattr(self.vista, attr_name):
            raise ValueError(
                f"Unknown visualization feature: '{feature}'. "
                f"Valid options: agents, graph, safety, thermal, wind, trails"
            )
        
        current   = getattr(self.vista, attr_name)
        new_state = not current if show is None else show
        setattr(self.vista, attr_name, new_state)
        
        return new_state

    def update(self, observation: TensorDictBase):
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
                - edge_index  : Communication graph edges (2, E)
                - gradient    : Temperature gradients (N, 3)
                - position    : Agent positions (N, 3)
                - temperature : Agent temperatures (N, 1)
                - velocity    : Agent velocities (N, 3) 
        """
        edge_index  = observation.get("edge_index")
        position    = observation.get("position")
        temperature = observation.get("temperature")
        velocity    = observation.get("velocity")
        
        if not self.plotter.ren_win:
            return
            
        self.plotter.clear_actors()
        
        if self.vista.show_agents:
            colormap = self.vista.colormap if self.vista.show_thermal_colors else None
            self.agent_actors = self.renderer.add_agents(
                colormap    = colormap,
                plotter     = self.plotter,
                position    = position,
                show_trails = self.vista.show_trails,
                temperature = temperature,
                velocity    = velocity
            )
        
        if self.vista.show_wind_arrows:
            wind_grid = self.sampler.create_wind_grid(
                position   = position,
                simulation = self.simulation
            )
            self.wind_actors = self.renderer.add_wind_vectors(
                plotter   = self.plotter,
                wind_grid = wind_grid
            )
        
        if self.vista.show_safety_boundary:
            self.safety_actors = self.renderer.add_safety_boundary(
                max_temperature = self.max_temperature,
                plotter         = self.plotter,
                position        = position,
                temperature     = temperature
            )
        
        if self.vista.show_graph:
            self.graph_actors = self.renderer.add_communication_graph(
                edge_index = edge_index,
                plotter    = self.plotter,
                position   = position
            )
        
        if self.vista.show_temperature_volume:
            temp_grid = self.sampler.create_temperature_grid(
                environment = self.simulation,
                position    = position
            )
            self.temperature_actors = self.renderer.add_temperature_volume(
                plotter   = self.plotter,
                temp_grid = temp_grid
            )