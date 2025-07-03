"""
Rendering functions for the Thermur visualization system.

This module provides specialized functions for rendering the different elements 
of the simulation. Each function creates PyVista actors that represent a 
particular aspect of the simulation state, such as agent positions and 
orientations, wind vectors, temperature fields, safety boundaries, and 
communication networks.

The rendering functions are designed to be efficient and scalable, using
vectorized operations where possible and PyVista's optimized rendering
pipeline. Each function returns a list of actors that can be managed by
the main visualizer for updates and cleanup.
"""
from configs.imitation import ColorModel, GlyphModel, GridModel, OpacityModel
from pyvista           import Actor, ImageData, Plotter, PolyData
from torch             import Tensor
from typing            import Optional

import numpy   as np
import pyvista as pv


class Renderer:
    """
    Manages rendering of simulation elements for visualization.
    
    This class provides methods for rendering various aspects of the simulation
    state including agents, temperature fields, wind vectors, safety boundaries,
    and communication networks. Each rendering method creates PyVista actors
    that can be displayed in the visualization window.
    
    The renderer uses efficient vectorized operations and PyVista's optimized
    rendering pipeline to handle large-scale simulations while maintaining
    interactive frame rates.
    """
    
    def __init__(
        self,
        colors    : Optional[ColorModel] = None,
        glyphs    : Optional[GlyphModel] = None,
        opacities : Optional[OpacityModel] = None
    ):
        """
        Initialize the renderer with visual configuration.
        
        Args:
            colors    : Configuration for color settings
            glyphs    : Configuration for glyph appearance
            opacities : Configuration for opacity values
        """
        self.colors    = colors
        self.glyphs    = glyphs
        self.opacities = opacities
    
    def _create_agent_trails(
        self,
        colormap    : Optional[str],
        plotter     : Plotter,
        positions   : np.ndarray,
        temperature : Optional[np.ndarray],
        velocities  : np.ndarray
    ) -> list[Actor]:
        """
        Create motion trail visualization for agents.
        
        Creates fading trail lines behind each agent based on their velocity.
        Trails fade out with distance/time for visual clarity. This is a
        private helper method used by the agents rendering method.
        
        Args:
            colormap    : Temperature colormap (optional)
            plotter     : PyVista Plotter instance
            positions   : Agent positions array [N, 3]
            temperature : Agent temperatures array [N] (optional)
            velocities  : Agent velocities array [N, 3]
            
        Returns:
            List of actors for the trails
        """
        n_agents = len(positions)
        n_points = self.glyphs.trail_length
        
        t          = np.linspace(0, 0.1 * n_points, n_points)
        decay      = np.linspace(1, 0, n_points)[:, None]
        offsets    = velocities[:, None] * t * decay
        points     = (positions[:, None] - offsets).reshape(-1, 3)
        indices    = np.arange(n_agents * n_points).reshape(n_agents, n_points)
        trail_mesh = pv.MultipleLines(points=points, lines=indices)
        
        if temperature is not None:
            trail_mesh["temperature"] = np.repeat(temperature, n_points)
        
        params = {
            "line_width"            : 2,
            "opacity"               : self.opacities.trails,
            "render_lines_as_tubes" : True,
        }
        
        if temperature is not None and colormap:
            params.update(
                clim    = (temperature.min(), temperature.max()),
                cmap    = colormap,
                scalars = "temperature",
            )
        else:
            params["color"] = self.colors.trail_default
        
        return [plotter.add_mesh(trail_mesh, **params)]
    
    def add_agents(
        self,
        plotter     : Plotter,
        position    : Tensor,
        colormap    : Optional[str]    = None,
        show_trails : bool             = False,
        temperature : Optional[Tensor] = None,
        velocity    : Optional[Tensor] = None
    ) -> list[Actor]:
        """
        Add agent visualizations to the plotter.
        
        Creates visual representations of each agent in the flock using either sphere
        glyphs (showing position only) or arrow glyphs (showing position and direction).
        Agents can be colored by temperature using the provided colormap, and optional
        motion trails can be drawn to show recent movement patterns.
        
        The method optimizes rendering performance by using PyVista's efficient
        glyph operations and batching similar operations together. Temperature-based
        coloring uses the simulation's thermal data to provide immediate visual
        feedback about each agent's thermal state.
        
        Args:
            plotter     : PyVista Plotter instance to render to
            position    : Agent positions tensor of shape [N, 3]
            colormap    : Temperature colormap for thermal visualization
            show_trails : Whether to render motion trails behind agents
            temperature : Agent temperatures tensor of shape [N, 1] (optional)
            velocity    : Agent velocities tensor of shape [N, 3] (optional)
            
        Returns:
            List of PyVista actors created for the agents
        """
        actors      = []
        positions   = position.detach().cpu().numpy()
        point_cloud = pv.PolyData(positions)

        if temperature is not None:
            temps = temperature.detach().cpu().numpy().flatten()
            point_cloud["temperature"] = temps
        
        if velocity is not None:
            velocities = velocity.detach().cpu().numpy()
            point_cloud["velocity"] = velocities
        
        glyph_geom = (
            pv.Sphere(radius=self.glyphs.size) 
            if self.glyphs.type == "sphere" 
            else pv.Arrow()
        )
        
        if self.glyphs.type == "arrow" and velocity is not None:
            norms = np.linalg.norm(velocities, axis=1, keepdims=True)
            safe_norms = np.maximum(norms, 1e-6)
            point_cloud["direction"] = velocities / safe_norms
            
            agent_glyphs = point_cloud.glyph(
                geom   = glyph_geom, 
                orient = "direction",
                scale  = False
            )
            
        else:
            agent_glyphs = point_cloud.glyph(
                geom   = glyph_geom, 
                orient = False,
                scale  = False
            )
        
        mesh_params = {
            "render_points_as_spheres": self.glyphs.type == "sphere",
        }
        
        if temperature is not None and colormap:
            mesh_params.update(
                clim    = (temps.min(), temps.max()),
                cmap    = colormap,
                scalars = "temperature",
            )
        else:
            mesh_params["color"] = self.colors.agent_default
        
        actors.append(plotter.add_mesh(agent_glyphs, **mesh_params))
        
        if show_trails and velocity is not None:
            trail_actors = self._create_agent_trails(
                colormap    = colormap,
                plotter     = plotter,
                positions   = positions,
                temperature = temps if temperature is not None else None,
                velocities  = velocities
            )
            actors.extend(trail_actors)
        
        return actors
    
    def add_communication_graph(
        self,
        edge_index : Tensor,
        plotter    : Plotter,
        position   : Tensor
    ) -> list[Actor]:
        """
        Add communication graph visualization to the plotter.
        
        Visualizes the graph connectivity of the flock by drawing lines between
        agents that are within communication range. This visualization reveals
        the topology of information flow within the flock and helps debug
        connectivity issues or understand emergent behaviors based on communication
        patterns.
        
        The method efficiently handles large graphs by batching line creation
        and using PyVista's optimized rendering for line meshes. The visual
        style can be configured to match the overall visualization theme.
        
        Args:
            edge_index : Connectivity graph tensor of shape [2, E]
            plotter    : PyVista Plotter instance to render to
            position   : Agent positions tensor of shape [N, 3]
            
        Returns:
            List of PyVista actors created for the communication graph
        """
        if not edge_index.numel():
            return []
        
        edges = edge_index.cpu().numpy()
        lines = np.column_stack(
            [
                np.full(edges.shape[1], 2),
                edges[0],
                edges[1]
            ]
        ).ravel()
        
        mesh = pv.PolyData(position.cpu().numpy())
        mesh.lines = lines
        
        return [
            plotter.add_mesh(
                color                  = self.colors.graph_default,
                line_width             = 2,
                mesh                   = mesh,
                opacity                = self.opacities.graph,
                render_lines_as_tubes  = True,
            )
        ]
    
    def add_safety_boundary(
        self,
        grids            : GridModel,
        max_temperature  : float,
        plotter          : Plotter,
        position         : Tensor,
        temperature      : Tensor
    ) -> list[Actor]:
        """
        Add safety boundary isosurface to the plotter.
        
        Creates a semi-transparent isosurface representing the boundary where
        temperature equals T_max, which is the safety limit for the flock.
        This visualization helps to verify that the CBF (Control Barrier Function)
        is working correctly by showing the areas that agents should avoid.
        
        The method uses PyVista's built-in interpolation for temperature field
        estimation and isosurface extraction for efficient rendering.
        
        Args:
            grids            : Configuration for grid sampling parameters
            max_temperature  : Maximum safe temperature (T_max)
            plotter          : PyVista Plotter instance to render to
            position         : Agent positions tensor of shape [N, 3]
            temperature      : Agent temperatures tensor of shape [N, 1]
            
        Returns:
            List of PyVista actors created for the safety boundary
        """
        positions   = position.cpu().numpy()
        point_cloud = pv.PolyData(positions)
        point_cloud["temperature"] = temperature.cpu().numpy().ravel()
        
        min_bounds = np.array([positions[:, i].min() - grids.padding for i in range(3)])
        max_bounds = np.array([positions[:, i].max() + grids.padding for i in range(3)])
        resolution = np.array(grids.temperature_resolution)
        
        target_grid = pv.ImageData(
            dimensions = grids.temperature_resolution,
            origin     = min_bounds,
            spacing    = (max_bounds - min_bounds) / (resolution - 1),
        )
        
        # Use sample method to interpolate temperature values
        grid = target_grid.sample(point_cloud)
        
        try:
            contour = grid.contour([max_temperature])
            if contour.n_points == 0:
                return []
        except:
            return []
        
        return [
            plotter.add_mesh(
                color   = self.colors.safety_default,
                mesh    = contour.smooth(n_iter=50),
                opacity = self.opacities.safety,
            )
        ]
    
    def add_temperature_volume(
        self,
        plotter   : Plotter,
        temp_grid : ImageData,
        max_temp  : Optional[float] = None,
        min_temp  : Optional[float] = None
    ) -> list[Actor]:
        """
        Add temperature field volume rendering to the plotter.
        
        Creates a volume rendering of a temperature field, showing thermal gradients
        throughout the simulation space. This visualization provides insights into
        the thermal environment that the flock navigates through, helping to understand
        the thermal currents, hot spots, and thermal gradients.
        
        The volume rendering uses PyVista's efficient GPU-accelerated rendering
        pipeline to display large temperature fields in real-time. The colormap
        and opacity settings can be configured to highlight specific temperature
        ranges or features of interest.
        
        Args:
            plotter   : PyVista Plotter instance to render to
            temp_grid : PyVista UniformGrid with temperature data
            max_temp  : Maximum temperature for colormap scaling (auto if None)
            min_temp  : Minimum temperature for colormap scaling (auto if None)
            
        Returns:
            List of PyVista actors created for the temperature field
        """
        bounds = temp_grid.get_data_range("temperature")
        
        return [
            plotter.add_volume(
                cmap            = self.colors.colormap,
                clim            = (min_temp or bounds[0], max_temp or bounds[1]),
                opacity         = "linear",
                scalar_bar_args = {
                    "position_x" : self.colors.scalar_bar_position_x,
                    "position_y" : self.colors.scalar_bar_position_y,
                    "title"      : self.colors.scalar_bar_title,
                },
                volume          = temp_grid,
            )
        ]
    
    def add_wind_vectors(
        self,
        plotter   : Plotter,
        wind_grid : PolyData
    ) -> list[Actor]:
        """
        Add wind field vector visualization to the plotter.
        
        Visualizes wind vectors as directional arrows that show both orientation
        and magnitude of airflow. Arrows are placed on a grid with density based
        on the provided wind_grid. Only significant wind vectors are displayed to
        reduce visual clutter.
        
        The method optimizes performance by filtering out small wind vectors
        and using PyVista's efficient glyph operations. The arrow size and color
        can be configured to match the overall visualization style.
        
        Args:
            plotter   : PyVista Plotter instance to render to
            wind_grid : PyVista PolyData with wind vector data
            
        Returns:
            List of PyVista actors created for the wind field
        """
        velocities = wind_grid["wind_velocity"]
        magnitudes = np.linalg.norm(velocities, axis=1)
        wind_grid["wind_magnitude"] = magnitudes
        
        # Filter insignificant vectors to reduce clutter
        threshold = 0.1 * magnitudes.max()
        filtered_grid = wind_grid.threshold(
            scalars = "wind_magnitude",
            value   = threshold, 
        )
        
        if not filtered_grid.n_points:
            return []
        
        wind_glyphs = filtered_grid.glyph(
            factor = self.glyphs.arrow_scale,
            geom   = pv.Arrow(),
            orient = "wind_velocity",
            scale  = "wind_magnitude",
        )
        
        mesh_params = {
            "color"   : self.colors.wind_default,
            "opacity" : self.opacities.wind,
        }
        
        return [plotter.add_mesh(mesh=wind_glyphs, **mesh_params)]