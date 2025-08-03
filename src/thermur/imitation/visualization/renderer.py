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
from config.imitation.visualization import VistaModel
from contextlib                     import suppress
from numpy.typing                   import NDArray
from pyvista                        import Actor, ImageData, Plotter, PolyData
from torch                          import Tensor

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
    interactive frame rates. Glyph geometries are pre-built and cached for
    improved performance.
    """
    
    def __init__(
        self,
        agent_glyph : object,
        vista       : VistaModel,
        wind_glyph  : object
    ):
        """
        Initialize the renderer with visual configuration and cached geometries.
        
        Args:
            agent_glyph : Pre-built geometry for agent visualization
            vista       : Unified visualization configuration
            wind_glyph  : Pre-built arrow geometry for wind vectors
        """
        self.agent_glyph = agent_glyph
        self.vista       = vista
        self.wind_glyph  = wind_glyph
        
        self.init_cached_values()
    
    def _create_agent_trails(
        self,
        colormap    : str | None,
        plotter     : Plotter,
        positions   : NDArray,
        temperature : NDArray | None,
        velocities  : NDArray
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
        n_points = self.vista.trail_length
        
        offsets    = velocities[:, None] * self.trail_t * self.trail_decay
        points     = (positions[:, None] - offsets).reshape(-1, 3)
        indices    = np.arange(n_agents * n_points).reshape(n_agents, n_points)
        trail_mesh = pv.MultipleLines(points=points, lines=indices)
        
        if temperature is not None:
            trail_mesh["temperature"] = np.repeat(temperature, n_points)
        
        params = self.trail_params.copy()
        
        if temperature is not None and colormap:
            params |= {
                "clim"    : (temperature.min(), temperature.max()),
                "cmap"    : colormap,
                "scalars" : "temperature",
            }
        else:
            params["color"] = self.color_trail
        
        return [plotter.add_mesh(trail_mesh, **params)]
    
    def add_agents(
        self,
        plotter     : Plotter,
        position    : Tensor,
        colormap    : str | None    = None,
        show_trails : bool          = False,
        temperature : Tensor | None = None,
        velocity    : Tensor | None = None
    ) -> list[Actor]:
        """
        Add agent visualizations to the plotter.
        
        Creates visual representations of each agent in the flock using arrow
        glyphs (showing both position and velocity direction when available).
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
        
        glyph_geom = self.agent_glyph
        
        if velocity is not None:
            norms      = np.linalg.norm(velocities, axis=1, keepdims=True)
            safe_norms = np.maximum(norms, 1e-6)
            point_cloud["direction"] = velocities / safe_norms
            orient = "direction"
        else:
            orient = False
        
        agent_glyphs = point_cloud.glyph(
            geom   = glyph_geom, 
            orient = orient,
            scale  = False
        )
        
        mesh_params = {"opacity": self.vista.agent_opacity}
        
        if temperature is not None and colormap:
            mesh_params |= {
                "clim"    : (temps.min(), temps.max()),
                "cmap"    : colormap,
                "scalars" : "temperature",
            }
        else:
            mesh_params["color"] = self.vista.agent_color
        
        actors.append(plotter.add_mesh(agent_glyphs, **mesh_params))
        
        if show_trails and velocity is not None:
            actors.extend(
                self._create_agent_trails(
                    colormap    = colormap,
                    plotter     = plotter,
                    positions   = positions,
                    temperature = temps if temperature is not None else None,
                    velocities  = velocities
                )
            )
        
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
                np.full(edges.shape[1], self.line_segment_size),
                edges[0],
                edges[1]
            ]
        ).ravel()
        
        mesh = pv.PolyData(position.cpu().numpy())
        mesh.lines = lines
        
        return [
            plotter.add_mesh(
                color                  = self.color_graph,
                line_width             = 2,
                mesh                   = mesh,
                opacity                = self.vista.graph_opacity,
                render_lines_as_tubes  = True,
            )
        ]
    
    def add_safety_boundary(
        self,
        max_temperature : float,
        plotter         : Plotter,
        position        : Tensor,
        temperature     : Tensor
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
        
        min_bounds = positions.min(axis=0) - self.vista.grid_padding
        max_bounds = positions.max(axis=0) + self.vista.grid_padding
        resolution = np.array(self.vista.temperature_resolution)
        
        target_grid = pv.ImageData(
            dimensions = self.vista.temperature_resolution,
            origin     = min_bounds,
            spacing    = (max_bounds - min_bounds) / (resolution - 1),
        )
        
        grid = target_grid.sample(point_cloud)
        
        with suppress(Exception):
            if (contour := grid.contour([max_temperature])).n_points == 0:
                return []
        return [
            plotter.add_mesh(
                color   = self.color_safety,
                mesh    = contour.smooth(n_iter=50),
                opacity = 0.3,
            )
        ]
    
    def add_temperature_volume(
        self,
        plotter   : Plotter,
        temp_grid : ImageData
    ) -> list[Actor]:
        """
        Add volumetric temperature field rendering to the plotter.
        
        Creates a semi-transparent 3D volume showing the thermal structure of the
        environment. This visualization reveals thermal columns, temperature gradients,
        and mixing regions that influence agent navigation. The volume rendering
        provides critical insight into the environmental conditions that drive
        the flock's thermal soaring behavior.
        
        Args:
            plotter   : PyVista Plotter instance to render to
            temp_grid : Volumetric temperature data sampled on a regular grid
            
        Returns:
            List of PyVista actors created for the temperature volume
        """
        bounds = temp_grid.get_data_range("temperature")
        
        return [
            plotter.add_volume(
                clim                  = bounds,
                cmap                  = self.vista.colormap,
                opacity               = "sigmoid",
                opacity_unit_distance = 0.1,
                volume                = temp_grid,
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
        
        threshold = self.wind_threshold_factor * magnitudes.max()
        filtered_grid = wind_grid.threshold(
            scalars = "wind_magnitude",
            value   = threshold, 
        )
        
        if not filtered_grid.n_points:
            return []
        
        wind_glyphs = filtered_grid.glyph(
            factor = self.vista.arrow_scale,
            geom   = self.wind_glyph,
            orient = "wind_velocity",
            scale  = "wind_magnitude",
        )
        
        return [
            plotter.add_mesh(
                mesh    = wind_glyphs,
                color   = self.color_wind,
                opacity = 0.8,
            )
        ]
    
    def init_cached_values(self):
        """
        Initialize cached values for rendering performance.
        
        Pre-computes and caches frequently-used values to avoid repeated
        allocations and computations during rendering. This includes trail
        parameters, rendering options, and color constants.
        """
        n_points = self.vista.trail_length
        self.trail_t     = np.linspace(0, 0.1 * n_points, n_points)
        self.trail_decay = np.linspace(1, 0, n_points)[:, None]
        
        self.trail_params = {
            "line_width"            : 2,
            "opacity"               : 0.5,
            "render_lines_as_tubes" : True,
        }
        
        self.color_graph  = (0.7, 0.7, 0.9)
        self.color_safety = (0.9, 0.3, 0.3)
        self.color_trail  = (0.8, 0.8, 0.8)
        self.color_wind   = (0.7, 0.7, 0.7)
        
        self.line_segment_size     = 2
        self.wind_threshold_factor = 0.1