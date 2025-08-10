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
from torch        import Size, Tensor
from torchrl.data import Bounded, Composite, Unbounded
from torchrl.envs import EnvBase
from typing       import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .loader                     import WRFDataSource
    from config.imitation.controller import FlockModel, SafetyModel
    from config.imitation.simulation import PhysicsModel
    from tensordict                  import TensorDictBase
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
        flock       : FlockModel,
        k_neighbors : int,
        physics     : PhysicsModel,
        safety      : SafetyModel,
        wrf         : WRFDataSource,
    ):
        """
        Initializes the Thermur environment with dependency injection.

        Args:
            flock       : Flock parameters configuration.
            k_neighbors : Number of topological neighbors for murmuration.
            physics     : Physics simulation configuration.
            safety      : Safety configuration with temperature thresholds.
            wrf         : WRF data source providing environmental data queries.
        """
        self.episode_time = 0.0
        self.flock        = flock
        self.k_neighbors  = k_neighbors
        self.physics      = physics
        self.positions    = th.zeros(flock.agent_count, 3)
        self.safety       = safety
        self.velocities   = th.zeros(flock.agent_count, 3)
        self.wrf          = wrf
        
        # Temporal tracking
        self.timestep      = 0
        self.trajectory_id = th.randint(0, 100000, (1,)).item()
        
        super().__init__(device="cpu")

    def _compute_edge_index(self, position: Tensor) -> Tensor:
        """
        Computes topological k-nearest neighbor connectivity.

        Following Ballerini et al. (2008), builds graph connectivity based on
        k-nearest neighbors rather than metric distance. Each agent connects
        to exactly k nearest neighbors regardless of distance, matching
        the topological interaction rule observed in real murmurations.

        Args:
            position: A tensor of node positions, shape (num_nodes, num_dims).

        Returns:
            An `edge_index` tensor of shape (2, num_edges), suitable for a
            `torch_geometric.data.Data` object.
        """
        distances   = th.cdist(position, position)
        _, indices  = distances.topk(self.k_neighbors + 1, largest=False)
        n_agents    = len(position)
        edge_source = th.arange(n_agents).repeat_interleave(self.k_neighbors)
        edge_target = indices[:, 1:].flatten()
        
        return th.stack([edge_source, edge_target])

    def _compute_forces(
        self,
        actions    : Tensor,
        velocities : Tensor,
        wind       : Tensor,
    ) -> Tensor:
        """
        Compute total acceleration from control and environmental forces.

        Aggregates forces following Newton's second law to determine the
        net acceleration on each agent:

            𝐚_total = 𝐚_control + 𝐚_gravity + 𝐚_drag + 𝐚_wind

        The drag model assumes quadratic resistance proportional to speed,
        while wind coupling uses a reduced coefficient to model partial
        sheltering effects in the flock.

        Args:
            actions    : Control accelerations 𝐚_nom ∈ ℝ^(n×3) [m/s²]
            velocities : Agent velocities 𝐯 ∈ ℝ^(n×3) [m/s]
            wind       : Environmental wind field 𝐰 ∈ ℝ^(n×3) [m/s]

        Returns:
            Tensor [n, 3] of total accelerations [m/s²]
        """
        gravity = th.zeros_like(velocities)
        gravity[:, 2] = -self.physics.gravity
        
        drag_force = (
            -self.physics.drag_coefficient * velocities *
            velocities.norm(dim=1, keepdim=True)
        )
        wind_force = wind * self.physics.drag_coefficient * 0.5
        
        return actions + gravity + drag_force + wind_force

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

            action        = Unbounded(shape=Size([n, 3]), dtype=th.float32),
            gradient      = Unbounded(shape=Size([n, 3]), dtype=th.float32),
            position      = Unbounded(shape=Size([n, 3]), dtype=th.float32),
            reward        = Unbounded(shape=Size([n]),    dtype=th.float32), 
            timestep      = Unbounded(shape=Size([1]),    dtype=th.int64),
            trajectory_id = Unbounded(shape=Size([1]),    dtype=th.int64),
            velocity      = Unbounded(shape=Size([n, 3]), dtype=th.float32),
            wind          = Unbounded(shape=Size([n, 3]), dtype=th.float32),
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

    def _integrate_positions(
        self,
        positions  : Tensor,
        timestep   : float,
        velocities : Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Update positions with hard boundary constraints.

        Performs forward Euler integration of positions with collision
        detection at simulation boundaries:

            𝐱(t+Δt) = 𝐱(t) + 𝐯(t)·Δt

        When agents reach boundaries, positions are clamped and velocities
        are zeroed in the constrained dimensions to prevent penetration.
        This implements a perfectly inelastic collision model.

        Args:
            positions  : Current positions 𝐱 ∈ ℝ^(n×3) [m]
            timestep   : Integration timestep Δt [s]
            velocities : Current velocities 𝐯 ∈ ℝ^(n×3) [m/s]

        Returns:
            Tuple of (new_positions, at_bounds) where:
                new_positions : Tensor [n, 3] of updated positions [m]
                at_bounds     : Tensor [n, 1] boundary collision indicators
        """
        new_positions = positions + velocities * timestep
        
        bounds_min    = th.as_tensor(self.physics.bounds_min, device=positions.device)
        bounds_max    = th.as_tensor(self.physics.bounds_max, device=positions.device)
        new_positions = new_positions.clamp(bounds_min, bounds_max)
        at_bounds     = (
            (new_positions == bounds_min) | 
            (new_positions == bounds_max)
        ).any(dim=-1, keepdim=True)
        
        velocities.masked_fill_(at_bounds.expand_as(velocities), 0.0)
        
        return new_positions, at_bounds

    def _integrate_velocities(
        self,
        acceleration : Tensor,
        timestep     : float,
        velocities   : Tensor,
    ) -> Tensor:
        """
        Update velocities with maximum speed constraint.

        Performs forward Euler integration with speed limiting:

            𝐯(t+Δt) = 𝐯(t) + 𝐚(t)·Δt
            
        If |𝐯| > v_max, the velocity is rescaled to preserve direction:

            𝐯_limited = 𝐯 · (v_max / |𝐯|)

        This maintains smooth trajectories at the speed boundary rather
        than component-wise clamping which would cause discontinuities.

        Args:
            acceleration : Total acceleration 𝐚 ∈ ℝ^(n×3) [m/s²]
            timestep     : Integration timestep Δt [s]
            velocities   : Current velocities 𝐯 ∈ ℝ^(n×3) [m/s]

        Returns:
            Tensor [n, 3] of updated velocities [m/s]
        """
        new_velocities = velocities + acceleration * timestep
        speed          = new_velocities.norm(dim=1, keepdim=True)
        
        scale = th.minimum(
            th.ones_like(speed),
            self.physics.max_speed / speed.clamp_min(self.physics.epsilon)
        )
        
        return new_velocities * scale

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
        positions = (
            self._generate_initial_positions(self.flock.agent_count) *
            self.flock.communication_range * 
            self.physics.initial_spacing_factor
        )

        self.episode_time     = 0.0
        self.velocities       = th.zeros_like(positions)
        self.positions        = positions.clone()
        self.timestep         = 0
        self.trajectory_id    = th.randint(0, 100000, (1,)).item()
        self.wrf.current_time = self.episode_time
        thermal               = self.wrf.query_thermal(positions)
        initial_obs           = self.observation_spec.zero()
        
        initial_obs.update({
            "action"        : th.zeros(self.flock.agent_count, 3),
            "battery"       : th.ones(self.flock.agent_count, 1),
            "edge_index"    : self._compute_edge_index(positions),
            "gradient"      : thermal[0],
            "position"      : self.positions,
            "temperature"   : thermal[1],
            "timestep"      : th.tensor([self.timestep]),
            "trajectory_id" : th.tensor([self.trajectory_id]),
            "velocity"      : self.velocities,
            "wind"          : self.wrf.query_wind(positions),
        })

        return initial_obs

    def _set_seed(self, seed: int | None):
        """
        Sets the random seed for reproducible environment dynamics.
        
        This method ensures deterministic behavior across the environment
        and its components. It coordinates with PyTorch's global random
        state to maintain consistency with the broader training pipeline.
        
        Args:
            seed: The random seed to set. If None, generates a random seed
                  from PyTorch's current random state.
        """
        if seed is None:
            seed = int(th.empty((), dtype=th.int64).random_().item())
        
        th.manual_seed(seed)
        
        if th.cuda.is_available():
            th.cuda.manual_seed(seed)
            th.cuda.manual_seed_all(seed)
        
        self._seed = seed

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """
        Performs one discrete time step using Euler integration.

        The process follows these steps:
            1. Extract control actions (accelerations) from the input
            2. Update velocities : v(t+dt) = v(t) + a(t)    * dt
            3. Update positions  : x(t+dt) = x(t) + v(t+dt) * dt
            4. Query environmental data at new positions
            5. Create observation with updated state

        Args:
            tensordict: A `TensorDict` containing the control `action` to apply,
                        with shape [n_agents, 3] representing accelerations

        Returns:
            A `TensorDict` for the next state, including the new observation
        """
        actions = tensordict.get("action")
        
        self.wrf.current_time = self.episode_time
        wind                  = self.wrf.query_wind(self.positions)
        
        self.velocities = self._integrate_velocities(
            acceleration = self._compute_forces(
                actions    = actions,
                velocities = self.velocities,
                wind       = wind
            ),
            timestep     = self.physics.simulation_step,
            velocities   = self.velocities
        )
        
        self.positions, at_bounds = self._integrate_positions(
            positions  = self.positions,
            timestep   = self.physics.simulation_step,
            velocities = self.velocities
        )
        
        thermal        = self.wrf.query_thermal(self.positions)
        next_obs       = self.observation_spec.zero()
        self.timestep += 1
        
        next_obs.update({
            "action"       : actions.clone(),
            "battery"      : th.ones(self.flock.agent_count, 1),
            "done"         : at_bounds.any().unsqueeze(0),
            "edge_index"   : self._compute_edge_index(self.positions),
            "gradient"     : thermal[0],
            "position"     : self.positions.clone(),
            "reward"       : th.zeros(self.flock.agent_count, dtype=th.float32),
            "temperature"  : thermal[1],
            "timestep"     : th.tensor([self.timestep]),
            "trajectory_id": th.tensor([self.trajectory_id]),
            "velocity"     : self.velocities.clone(),
            "wind"         : wind,
        })
        
        self.episode_time += self.physics.simulation_step
        return next_obs
