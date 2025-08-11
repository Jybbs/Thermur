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
from __future__ import annotations
from pyvista    import Color
from typing     import TYPE_CHECKING

import numpy   as np
import pyvista as pv

if TYPE_CHECKING:
    from config.imitation.visualization import VistaModel
    from pyvista                        import Actor, ImageData, Plotter, PolyData
    from torch                          import Tensor


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
        plotter     : Plotter,
        point_cloud : PolyData
    ) -> list[Actor]:
        """
        Create motion trail visualization for agents.

        Creates fading trail lines behind each agent based on their velocity.
        Trails fade out with distance/time for visual clarity.

        Args:
            plotter     : PyVista Plotter instance
            point_cloud : PolyData containing agent positions, velocities, and temperatures

        Returns:
            List of actors for the trails
        """
        positions    = point_cloud.points
        n_agents     = len(positions)
        n_points     = self.vista.trail_length
        velocities   = point_cloud["velocity"]
        temperatures = point_cloud["temperature"]

        trail_mesh = pv.PolyData(
            (
                positions[:, None]
                - velocities[:, None] * self.trail_t * self.trail_decay
            ).reshape(-1, 3)
        )

        trail_mesh.lines = np.column_stack([
            np.full(n_agents, n_points),
            np.arange(0, n_agents * n_points, n_points),
            np.arange(0, n_agents * n_points, n_points) + np.arange(n_points)
        ]).ravel()
        trail_mesh["temperature"] = np.repeat(temperatures, n_points)

        return [
            plotter.add_mesh(
                clim                  = (
                    float(temperatures.min()),
                    float(temperatures.max())
                ),
                cmap                  = "plasma",
                line_width            = self.trail_params["line_width"],
                mesh                  = trail_mesh,
                opacity               = self.trail_params["opacity"],
                render_lines_as_tubes = True,
                scalars               = "temperature",
            )
        ]

    def add_agents(
        self,
        plotter     : Plotter,
        position    : Tensor,
        temperature : Tensor,
        velocity    : Tensor,
        colormap    : str | None = None,
        show_trails : bool       = False
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
            temperature : Agent temperatures tensor of shape [N, 1]
            velocity    : Agent velocities tensor of shape [N, 3]
            colormap    : Temperature colormap for thermal visualization
            show_trails : Whether to render motion trails behind agents

        Returns:
            List of PyVista actors created for the agents
        """
        point_cloud = pv.PolyData(position.detach().cpu().numpy())
        point_cloud["temperature"] = temperature.detach().cpu().numpy().flatten()
        point_cloud["velocity"]    = velocity.detach().cpu().numpy()
        velocity_norms = np.linalg.norm(
            point_cloud["velocity"], axis=1, keepdims=True
        )
        point_cloud["direction"] = (
            point_cloud["velocity"] / np.maximum(velocity_norms, 1e-8)
        )

        agent_glyphs = point_cloud.glyph(
            geom   = self.agent_glyph,
            orient = "direction",
            scale  = False
        )

        if colormap:
            actors = [
                plotter.add_mesh(
                    clim    = (
                        float(point_cloud["temperature"].min()),
                        float(point_cloud["temperature"].max())
                    ),
                    cmap    = "plasma",
                    mesh    = agent_glyphs,
                    opacity = self.vista.agent_opacity,
                    scalars = "temperature"
                )
            ]
        else:
            actors = [
                plotter.add_mesh(
                    color   = self.vista.agent_color,
                    mesh    = agent_glyphs,
                    opacity = self.vista.agent_opacity
                )
            ]

        if show_trails and colormap:
            actors.extend(self._create_agent_trails(plotter, point_cloud))

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
        lines = np.column_stack([
            np.full(edges.shape[1], self.line_segment_size),
            edges[0],
            edges[1]
        ]).ravel()

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

        try:
            contour = target_grid.sample(point_cloud).contour([max_temperature])
            smoothed = (
                contour.smooth(n_iter=50) 
                if contour.n_points > 0 
                else contour
            )
            return [
                plotter.add_mesh(
                    color   = self.color_safety,
                    mesh    = smoothed,
                    opacity = 0.3
                )
            ]
        except Exception:
            return []

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
        result = plotter.add_volume(
            clim                  = temp_grid.get_data_range("temperature"),
            cmap                  = "plasma",
            opacity               = "sigmoid",
            opacity_unit_distance = 0.1,
            volume                = temp_grid
        )
        assert not isinstance(result, list)
        return [result]

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
            "line_width" : 2,
            "opacity"    : 0.5,
        }

        self.color_graph  = Color((0.7, 0.7, 0.9))
        self.color_safety = Color((0.9, 0.3, 0.3))
        self.color_trail  = Color((0.8, 0.8, 0.8))
        self.color_wind   = Color((0.7, 0.7, 0.7))

        self.line_segment_size     = 2
        self.wind_threshold_factor = 0.1
