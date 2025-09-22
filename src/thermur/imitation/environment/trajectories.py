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
    from config.imitation.controller  import MurmurationModel, SafetyModel
    from config.imitation.environment import PhysicsModel


class TrajectoryGenerator:
    """
    Generates expert demonstration trajectories for offline training.

    This class provides physics simulation for drone flocks without the
    overhead of a full RL environment. It manages agent positions, velocities,
    and environmental interactions, returning PyG Data objects suitable for
    behavioral cloning.

    Unlike the previous TorchRL environment, this generator:
    - Returns PyG Data objects directly (no TensorDict)
    - Focuses solely on trajectory generation (no rewards/done flags)
    - Simplified API for offline demonstration collection
    """

    def __init__(
        self,
        k_neighbors : int,
        mmm         : MurmurationModel,
        physics     : PhysicsModel,
        safety      : SafetyModel,
        wrf         : WRFLoader,
    ):
        """
        Initialize the trajectory generator.

        Args:
            k_neighbors : Number of topological neighbors for connectivity
            mmm         : Murmuration model with agent count and parameters
            physics     : Physics simulation parameters
            safety      : Safety thresholds (for reference, not enforced here)
            wrf         : WRF data source for environmental queries
        """
        self.episode_time = 0.0
        self.k_neighbors  = k_neighbors
        self.mmm          = mmm
        self.physics      = physics
        self.positions    = th.zeros(mmm.agent_count, 3)
        self.safety       = safety
        self.velocities   = th.zeros(mmm.agent_count, 3)
        self.wrf          = wrf

        # Temporal tracking
        self.timestep = 0

    def _compute_edge_index(self, position: Tensor) -> Tensor:
        """
        Compute topological k-nearest neighbor connectivity.

        Following Ballerini et al. (2008), builds graph connectivity based on
        k-nearest neighbors rather than metric distance.

        Args:
            position: Node positions [N, 3]

        Returns:
            Edge index tensor [2, E] for PyG Data objects
        """
        distances  = th.cdist(position, position)
        _, indices = distances.topk(self.k_neighbors + 1, largest=False)

        return th.stack([
            th.arange(self.mmm.agent_count).repeat_interleave(self.k_neighbors),
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
        indices      = th.arange(self.mmm.agent_count, dtype=th.float32)
        z            = 1 - (2 * indices) / (self.mmm.agent_count - 1)
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
        timestep   : float,
        velocities : Tensor,
    ) -> Tensor:
        """
        Update positions with boundary constraints.

        Performs forward Euler integration with hard boundary clamping.

        Args:
            positions  : Current positions [N, 3]
            timestep   : Integration timestep
            velocities : Current velocities [N, 3]

        Returns:
            Updated positions [N, 3]
        """
        new_positions = positions + velocities * timestep

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
        timestep     : float,
        velocities   : Tensor,
    ) -> Tensor:
        """
        Update velocities with maximum speed constraint.

        Args:
            acceleration : Total accelerations [N, 3]
            timestep     : Integration timestep
            velocities   : Current velocities  [N, 3]

        Returns:
            Updated velocities [N, 3]
        """
        new_velocities = velocities + acceleration * timestep

        # Apply maximum speed constraint
        speed = new_velocities.norm(dim=-1, keepdim=True)
        scale = (self.physics.max_speed / speed.clamp(min=self.physics.max_speed))

        return new_velocities * scale.clamp(max=1.0)

    def reset(self) -> Data:
        """
        Reset the trajectory generator to initial conditions.

        Returns:
            Initial state as PyG Data object with:
                - edge_index  : Topological connectivity [2, E]
                - position    : Agent positions          [N, 3]
                - velocity    : Agent velocities         [N, 3]
                - temperature : Temperature at positions [N, 1]
                - gradient    : Temperature gradient     [N, 3]
                - wind        : Wind field at positions  [N, 3]
        """
        # Generate initial positions using Fibonacci lattice
        positions = self._fibonacci_lattice()
        positions *= (
            self.mmm.communication_range *
            self.physics.initial_spacing_factor
        )
        positions[:, 2] += self.physics.initial_altitude

        self.episode_time     = 0.0
        self.velocities       = th.randn_like(positions) * 2.0
        self.positions        = positions.clone()
        self.timestep         = 0
        self.wrf.current_time = self.episode_time
        gradient, temperature = self.wrf.query_thermal(self.positions)
        wind                  = self.wrf.query_wind(self.positions)

        return Data(
            edge_index   = self._compute_edge_index(self.positions),
            position     = self.positions.clone(),
            velocity     = self.velocities.clone(),
            temperature  = temperature.clone(),
            gradient     = gradient.clone(),
            wind         = wind.clone()
        )

    def step(self, action: Tensor) -> Data:
        """
        Advance the simulation by one timestep.

        Args:
            action: Control actions (accelerations) [N, 3]

        Returns:
            Next state as PyG Data object
        """
        self.wrf.current_time = self.episode_time
        wind                  = self.wrf.query_wind(self.positions)

        acceleration = self._compute_forces(
            actions    = action,
            velocities = self.velocities,
            wind       = wind
        )
        self.velocities = self._integrate_velocities(
            acceleration = acceleration,
            timestep     = self.physics.timestep,
            velocities   = self.velocities
        )

        self.positions = self._integrate_positions(
            positions  = self.positions,
            timestep   = self.physics.timestep,
            velocities = self.velocities
        )

        gradient, temperature = self.wrf.query_thermal(self.positions)

        self.timestep     += 1
        self.episode_time += self.physics.timestep

        return Data(
            edge_index   = self._compute_edge_index(self.positions),
            position     = self.positions.clone(),
            velocity     = self.velocities.clone(),
            temperature  = temperature,
            gradient     = gradient,
            wind         = wind
        )
