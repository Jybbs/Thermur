"""
Defines the core `torchrl` simulation environment for the Thermur project.

This module provides the `SimulationEnv` class, which serves as the primary
interface for training and evaluating flock policies. It implements the
`torchrl.EnvBase` API, making it compatible with the broader `torchrl`
ecosystem of collectors, replay buffers, and trainers.

The environment manages the state of the multi-agent system and steps the
simulation forward using Euler integration of agent dynamics. It
interfaces with a dynamic environmental data source (e.g., WRF-Fire data)
to provide thermal and wind field information.
"""
from __future__   import annotations
from torch        import Size
from torchrl.data import Bounded, Composite, Unbounded
from torchrl.envs import EnvBase
from typing       import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .loader                     import WRFDataSource
    from config.imitation.controller import FlockModel
    from config.imitation.simulation import PhysicsModel
    from tensordict                  import TensorDictBase
    from torch                       import Tensor
    from torchrl.data                import TensorSpec

import torch as th


class SimulationEnv(EnvBase):
    """
    A thermally-aware multi-agent flocking environment for `torchrl`.

    This environment simulates a flock of `N` agents navigating a 3D space
    characterized by a pre-computed wind and temperature field. Agents must
    learn to move in a way that is legible to an observer while respecting
    strict thermal safety constraints.

    The state is represented by a `TensorDict` matching the specification
    defined externally, which includes agent kinematics, local thermal
    data, and the communication graph topology.
    """

    def __init__(
        self,
        flock   : FlockModel,
        physics : PhysicsModel,
        wrf     : WRFDataSource,
    ):
        """
        Initializes the Thermur environment with dependency injection.

        Args:
            flock   : Flock parameters configuration.
            physics : Physics simulation configuration.
            wrf     : WRF data source providing environmental data queries.
        """
        self.flock   = flock
        self.physics = physics
        self.wrf     = wrf
        
        # Initialize state tensors for physics integration
        self.positions  = None
        self.velocities = None
        
        super().__init__(device="cpu")

    def _compute_edge_index(
        self,
        position : Tensor,
        radius   : float
    ) -> Tensor:
        """
        Computes the graph connectivity based on metric distance.

        This function builds an `edge_index` for `torch-geometric` by finding all
        pairs of nodes (i, j) where the Euclidean distance is less than `r`.
        It avoids self-loops.

        Args:
            position : A tensor of node positions, shape (num_nodes, num_dims).
            radius   : The communication radius.

        Returns:
            An `edge_index` tensor of shape (2, num_edges), suitable for a
            `torch_geometric.data.Data` object.
        """
        distances = th.cdist(position, position)
        mask      = (distances < radius) & (distances > 0)
        return th.nonzero(mask, as_tuple=False).t().contiguous()

    def _create_action_space(self) -> TensorSpec:
        """
        Defines the action space structure for agent control.

        The action space consists of velocity commands u_nom ∈ ℝ^d for each agent,
        which are processed by the safety filter before execution to ensure thermal
        constraint satisfaction.

        Returns:
            Composite tensor specification defining the action structure
        """
        return Composite(
            action = Unbounded(
                dtype = th.float32,
                shape = Size([self.flock.agent_count, 3])
            )
        )

    def _create_observation_space(self) -> TensorSpec:
        """
        Defines the observation space structure for the flock system.

        Creates a complete state representation including agent kinematics,
        thermal data, and communication topology. This structure serves as the
        contract for data exchange between simulation components.

        The observation space includes:
        - battery     : Energy remaining ∈ [0, 1] for each agent
        - done        : Episode termination flag
        - edge_index  : Dynamic graph connectivity for communication
        - gradient    : Temperature gradient ∇T at agent positions
        - position    : Spatial coordinates in ℝ^d
        - reward      : Reward signal for each agent
        - temperature : Thermal state in Kelvin
        - velocity    : Motion vectors in ℝ^d
        - wind        : Environmental wind field at agent positions

        Returns:
            Composite tensor specification defining the observation structure
        """
        n = self.flock.agent_count

        return Composite(
            battery     = Bounded(0, 1,      Size([n, 1]),       dtype=th.float32),
            done        = Bounded(0, 1,      Size([1]),          dtype=th.bool),
            edge_index  = Bounded(0, n-1,    Size([2, n*(n-1)]), dtype=th.int64),
            temperature = Bounded(0, th.inf, Size([n, 1]),       dtype=th.float32),

            gradient    = Unbounded(shape=Size([n, 3]), dtype=th.float32),
            position    = Unbounded(shape=Size([n, 3]), dtype=th.float32),
            reward      = Unbounded(shape=Size([n]),    dtype=th.float32),
            velocity    = Unbounded(shape=Size([n, 3]), dtype=th.float32),
            wind        = Unbounded(shape=Size([n, 3]), dtype=th.float32),
        )


    def _generate_initial_positions(self, n: int) -> Tensor:
        """
        Generates initial positions for murmuration dynamics.

        Uses the Fibonacci lattice method to create a compact spherical
        distribution suitable for topological interactions. This provides
        excellent uniformity while maintaining close proximity between
        agents, essential for k-nearest neighbor connectivity.

        The Fibonacci lattice method creates a nearly-uniform distribution by:
            - Using the golden ratio φ = (1 + √5)/2 to create optimal angular spacing
            - Setting z-coordinates using a linear spacing from -1 to 1
            - Computing radius at each z value: r = √(1 - z²)

        Points are then placed at:

            (r·cos(θ), r·sin(θ), z) where θ = 2π·k/φ

        This initial configuration promotes rapid emergence of collective
        murmuration behavior through strong topological connectivity.

        Args:
            n: Number of agents in the flock

        Returns:
            A tensor of shape [n, 3] containing agent positions
        """
        indices      = th.arange(n, dtype=th.float32)
        z            = 1 - (2 * indices) / (n - 1)
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


    def _make_spec(self, td_params: Any | None = None):
        """
        Creates action and observation specs for the environment.

        Called by torchrl.EnvBase during initialization to set up the
        action_spec and observation_spec attributes that define the
        environment's input/output structure.

        Args:
            td_params: Unused parameter required by base class interface
        """
        self.action_spec      = self._create_action_space()
        self.observation_spec = self._create_observation_space()

    def _reset(self, *args: Any, **kwargs: Any) -> TensorDictBase:
        """
        Resets the environment to an initial state for a new episode.

        This method creates an initial `TensorDict` observation by placing agents
        in a compact spherical formation suitable for murmuration dynamics. The
        initial positions use the Fibonacci lattice method to ensure uniform
        distribution while maintaining close proximity for topological interactions.

        The formation is scaled to approximately 0.3 times the communication range
        to ensure strong initial connectivity between agents, promoting the emergence
        of collective murmuration behavior.

        Returns:
            A `TensorDict` containing the initial observation of the flock.
        """
        positions = self._generate_initial_positions(self.flock.agent_count)
        
        # Scale positions by communication range and spacing factor
        scaled_positions = (
            positions * self.flock.communication_range * 
            self.physics.initial_spacing_factor
        )

        initial_observation = self.observation_spec.zero()
        initial_observation.update({
            "battery"  : th.ones(self.flock.agent_count, 1),
            "position" : scaled_positions,
        })

        positions             = initial_observation["position"]
        temperature, gradient = self.wrf.query_thermal(positions)
        wind                  = self.wrf.query_wind(positions)

        initial_observation.update(
            {
                "edge_index"  : self._compute_edge_index(
                    position = positions,
                    radius   = self.flock.communication_range
                ),
                "gradient"    : gradient,
                "temperature" : temperature,
                "wind"        : wind
            }
        )

        # Initialize physics state
        self.positions  = scaled_positions.clone()
        self.velocities = th.zeros_like(scaled_positions)

        return initial_observation


    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """
        Performs one discrete time step using Euler integration.

        The process follows these steps:
            1. Extract control actions (accelerations) from the input
            2. Update velocities: v(t+dt) = v(t) + a(t) * dt
            3. Update positions: x(t+dt) = x(t) + v(t+dt) * dt
            4. Query environmental data at new positions
            5. Create observation with updated state

        Args:
            tensordict: A `TensorDict` containing the control `action` to apply,
                        with shape [n_agents, 3] representing accelerations

        Returns:
            A `TensorDict` for the next state, including the new observation
        """
        # Extract control actions (accelerations)
        actions = tensordict.get("action")
        
        # Ensure velocities and positions are initialized
        if self.velocities is None:
            self.velocities = th.zeros_like(actions)
        if self.positions is None:
            self.positions = th.zeros_like(actions)
        
        # Euler integration with forces
        dt = self.physics.simulation_step
        
        # Apply gravity force (downward in z-direction)
        gravity_force       = th.zeros_like(self.velocities)
        gravity_force[:, 2] = -self.physics.gravity
        
        # Apply drag force proportional to velocity squared
        drag_coefficient = self.physics.drag_coefficient
        speed            = self.velocities.norm(dim=1, keepdim=True)
        drag_force       = -drag_coefficient * self.velocities * speed
        
        # Total acceleration = control input + gravity + drag
        total_acceleration = actions + gravity_force + drag_force
        
        # Update velocities: v(t+dt) = v(t) + a(t) * dt
        self.velocities = self.velocities + total_acceleration * dt
        
        # Clamp velocities to reasonable limits
        max_speed = self.physics.max_speed
        speed     = self.velocities.norm(dim=1, keepdim=True)
        self.velocities = th.where(
            speed > max_speed,
            self.velocities * max_speed / speed,
            self.velocities
        )
        
        # Update positions: x(t+dt) = x(t) + v(t+dt) * dt
        self.positions = self.positions + self.velocities * dt
        
        # Enforce boundary constraints
        for i in range(3):
            self.positions[:, i] = self.positions[:, i].clamp(
                min=self.physics.bounds_min[i],
                max=self.physics.bounds_max[i]
            )
            # Zero out velocity component if hitting boundary
            at_min = self.positions[:, i] == self.physics.bounds_min[i]
            at_max = self.positions[:, i] == self.physics.bounds_max[i]
            self.velocities[at_min | at_max, i] = 0
        
        # Query environmental data at new positions
        temperature, gradient = self.wrf.query_thermal(self.positions)
        wind                  = self.wrf.query_wind(self.positions)
        edge_index            = self._compute_edge_index(
            position = self.positions,
            radius   = self.flock.communication_range
        )
        
        # Create next observation
        next_observation = self.observation_spec.zero()
        next_observation.update({
            "battery"     : th.ones(self.flock.agent_count, 1),
            "edge_index"  : edge_index,
            "gradient"    : gradient,
            "position"    : self.positions.clone(),
            "temperature" : temperature,
            "velocity"    : self.velocities.clone(),
            "wind"        : wind,
        })
        
        # TODO: Consider the following for full TorchRL compatibility:
        # - Add done/terminated flags based on bounds or thermal limits
        # - Update input tensordict with next observation (TorchRL convention)
        # - Add any additional physics constraints (e.g., velocity limits)
        # - Handle episode resets when agents exceed boundaries
        
        return next_observation
