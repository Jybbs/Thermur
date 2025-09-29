"""
Trajectory generation for offline expert demonstrations.

This module provides lightweight trajectory generation without the overhead
of TorchRL environments. It manages flock physics simulation and returns
PyG Data objects suitable for offline imitation learning.
"""
from __future__           import annotations
from torch                import Tensor
from torch_geometric.data import Data
from typing               import TYPE_CHECKING

import torch as th

if TYPE_CHECKING:
    from .loader                      import WRFLoader
    from config.imitation.environment import PhysicsModel


class TrajectoryGenerator:
    """
    Generates expert trajectories for behavioral cloning.

    Provides lightweight physics simulation for drone flocks, managing agent
    positions, velocities, and environmental interactions. Returns PyG Data
    objects suitable for imitation learning without reinforcement learning
    overhead like rewards or termination flags.
    """

    def __init__(
        self,
        agent_count         : int,
        communication_range : float,
        k_neighbors         : int,
        physics             : PhysicsModel,
        wrf                 : WRFLoader,
    ):
        """
        Initialize the trajectory generator.

        Args:
            agent_count         : Number of agents in the flock
            communication_range : Metric interaction radius for initial spacing
            k_neighbors         : Number of topological neighbors for connectivity
            physics             : Physics simulation parameters
            wrf                 : WRF data source for environmental queries
        """
        self.agent_count         = agent_count
        self.communication_range = communication_range
        self.frame               = 0
        self.k_neighbors         = k_neighbors
        self.physics             = physics
        self.positions           = th.zeros(agent_count, 3)
        self.time                = 0.0
        self.velocities          = th.zeros(agent_count, 3)
        self.velocity_rng        = th.Generator()
        self.wrf                 = wrf

    def _compute_edge_index(self, distances: Tensor) -> Tensor:
        """
        Compute topological k-nearest neighbor connectivity.

        Following Ballerini et al. (2008), builds graph connectivity based on
        k-nearest neighbors rather than metric distance.

        Args:
            distances: Pairwise distance matrix [N, N]

        Returns:
            Edge index tensor [2, E] for PyG Data objects
        """
        _, indices = distances.topk(self.k_neighbors + 1, largest=False)

        return th.stack([
            th.arange(self.agent_count).repeat_interleave(self.k_neighbors),
            indices[:, 1:].flatten()
        ])

    def _compute_forces(
        self,
        actions    : Tensor,
        velocities : Tensor,
        wind       : Tensor,
    ) -> Tensor:
        """
        Compute total acceleration from control and environmental forces.

        Aggregates forces following Newton's second law:
            𝐚_total = 𝐚_control + 𝐚_gravity + 𝐚_drag + 𝐚_wind

        Args:
            actions    : Control accelerations    [N, 3]
            velocities : Agent velocities         [N, 3]
            wind       : Environmental wind field [N, 3]

        Returns:
            Total accelerations [N, 3]
        """
        gravity = th.zeros_like(velocities)
        gravity[:, 2] = -self.physics.gravity

        speed      = velocities.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        drag_force = -self.physics.drag_coefficient * velocities * speed
        wind_force = self.physics.wind_coupling_coefficient * (wind - velocities)

        return actions + gravity + drag_force + wind_force

    def _fibonacci_lattice(self) -> Tensor:
        """
        Generate initial positions using Fibonacci lattice on a sphere.

        Creates a nearly-uniform distribution on a sphere using the golden
        ratio for optimal angular spacing.

        Returns:
            Initial positions [N, 3]
        """
        indices      = th.linspace(0, self.agent_count - 1, self.agent_count)
        z            = 1 - (2 * indices) / (self.agent_count - 1)
        radius       = th.sqrt(1 - z*z)
        golden_angle = th.pi * (3. - (5.**0.5))
        theta        = golden_angle * indices

        return th.stack(
            dim     = 1,
            tensors = (
                th.cos(theta) * radius,
                th.sin(theta) * radius,
                z
            ),
        )

    def _integrate_positions(
        self,
        positions  : Tensor,
        timeframe  : float,
        velocities : Tensor,
    ) -> Tensor:
        """
        Update positions with boundary constraints.

        Performs forward Euler integration with hard boundary clamping.

        Args:
            positions  : Current positions [N, 3]
            timeframe  : Integration timeframe
            velocities : Current velocities [N, 3]

        Returns:
            Updated positions [N, 3]
        """
        new_positions = positions + velocities * timeframe
        bounds_min    = th.as_tensor(self.physics.bounds_min, device=positions.device)
        bounds_max    = th.as_tensor(self.physics.bounds_max, device=positions.device)
        new_positions = new_positions.clamp(bounds_min, bounds_max)

        # Zero velocities at boundaries
        at_bounds = (
            (new_positions == bounds_min) |
            (new_positions == bounds_max)
        )
        velocities.masked_fill_(at_bounds, 0.0)

        return new_positions

    def _integrate_velocities(
        self,
        acceleration : Tensor,
        timeframe    : float,
        velocities   : Tensor,
    ) -> Tensor:
        """
        Update velocities with maximum speed constraint.

        Args:
            acceleration : Total accelerations [N, 3]
            timeframe    : Integration timeframe
            velocities   : Current velocities  [N, 3]

        Returns:
            Updated velocities [N, 3]
        """
        new_velocities = velocities + acceleration * timeframe

        # Apply maximum speed constraint
        speed = new_velocities.norm(dim=-1, keepdim=True)
        scale = (self.physics.max_speed / speed.clamp(min=self.physics.max_speed))

        return new_velocities * scale.clamp(max=1.0)

    def reset(self, snapshot_idx: int) -> Data:
        """
        Reset the trajectory generator to initial conditions.

        Pins the WRF loader to the specified snapshot index, ensuring all
        environmental queries throughout the trajectory use consistent
        conditions.

        Args:
            snapshot_idx: WRF snapshot index for this trajectory

        Returns:
            Initial state as PyG Data object with:
                - edge_index  : Topological connectivity [2, E]
                - position    : Agent positions          [N, 3]
                - velocity    : Agent velocities         [N, 3]
                - temperature : Temperature at positions [N, 1]
                - gradient    : Temperature gradient     [N, 3]
                - wind        : Wind field at positions  [N, 3]
        """
        self.wrf.snapshot_idx = snapshot_idx

        positions = self._fibonacci_lattice()
        positions *= (
            self.communication_range *
            self.physics.initial_spacing_factor
        )
        positions[:, 2] += self.physics.initial_altitude

        self.frame = 0
        self.time  = 0.0

        self.velocity_rng.manual_seed(self.frame)
        self.positions  = positions
        self.velocities = th.randn(
            generator = self.velocity_rng,
            size      = positions.shape
        ) * 2.0

        gradient, temperature = self.wrf.query_thermal(self.positions)
        distances = th.cdist(self.positions, self.positions)

        return Data(
            distances   = distances,
            edge_index  = self._compute_edge_index(distances),
            frame       = self.frame,
            gradient    = gradient,
            position    = self.positions.clone(),
            temperature = temperature,
            velocity    = self.velocities.clone(),
            wind        = self.wrf.query_wind(self.positions)
        )

    def step(self, action: Tensor) -> Data:
        """
        Advance the simulation by one frame.

        Args:
            action: Control actions (accelerations) [N, 3]

        Returns:
            Next state as PyG Data object
        """
        wind         = self.wrf.query_wind(self.positions)
        acceleration = self._compute_forces(
            actions    = action,
            velocities = self.velocities,
            wind       = wind
        )

        self.velocities = self._integrate_velocities(
            acceleration = acceleration,
            timeframe    = self.physics.timeframe,
            velocities   = self.velocities
        )

        self.positions = self._integrate_positions(
            positions  = self.positions,
            timeframe  = self.physics.timeframe,
            velocities = self.velocities
        )

        gradient, temperature = self.wrf.query_thermal(self.positions)
        distances   = th.cdist(self.positions, self.positions)
        self.frame += 1
        self.time  += self.physics.timeframe

        return Data(
            distances   = distances,
            edge_index  = self._compute_edge_index(distances),
            frame       = self.frame,
            gradient    = gradient,
            position    = self.positions.clone(),
            temperature = temperature,
            velocity    = self.velocities.clone(),
            wind        = wind
        )
