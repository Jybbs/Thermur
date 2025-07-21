"""
Defines the core `torchrl` simulation environment for the Thermur project.

This module provides the `SimulationEnv` class, which serves as the primary
interface for training and evaluating flock policies. It implements the
`torchrl.EnvBase` API, making it compatible with the broader `torchrl`
ecosystem of collectors, replay buffers, and trainers.

The environment's main responsibility is to manage the state of the multi-agent
system, step the simulation forward in time, and provide observations and
rewards to the learning algorithm. It couples a rigid-body physics engine
(MuJoCo) with a dynamic environmental data source (e.g., WRF-Fire data).
"""
from ..utils           import generate_flock_xml, load_flock_model
from configs.imitation import FlockModel, PhysicsModel
from operator          import itemgetter
from tensordict        import TensorDictBase
from torch             import bool, cdist, float32, inf, int64, nonzero, Tensor
from torchrl.data      import Bounded, Composite, Unbounded
from torchrl.envs      import EnvBase
from typing            import Any, Callable

import math
import mujoco as mj
import torch


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
        data_source : Any,
        flock       : FlockModel,
        physics     : PhysicsModel,
        seed_fn     : Callable,
    ):
        """
        Initializes the Thermur environment with dependency injection.

        Args:
            data_source : A callable that provides environmental data queries.
            flock       : Flock parameters configuration.
            physics     : Physics simulation configuration.
            seed_fn     : Callable for setting random seeds.
        """
        super().__init__(device="cpu")
        self.action_spec      = self._create_action_space()
        self.data_source      = data_source
        self.flock            = flock
        self.observation_spec = self._create_observation_space()
        self.physics_model    = self._initialize_physics(physics)
        self.seed_fn          = seed_fn
    
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
        distances = cdist(position, position)
        mask      = (distances < radius) & (distances > 0)
        return nonzero(mask, as_tuple=False).t().contiguous()
    
    def _create_action_space(self) -> TensorDictBase:
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
                dtype = float32,
                shape = self.flock.shape,
            ),
        )
    
    def _create_observation_space(self) -> TensorDictBase:
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
        n, dims = self.flock.shape
        
        bounded_tensors = {
            "battery"     : Bounded(0, 1,   (n, 1),       dtype=float32),
            "done"        : Bounded(0, 1,   (1,),         dtype=bool),
            "edge_index"  : Bounded(0, n-1, (2, n*(n-1)), dtype=int64),
            "temperature" : Bounded(0, inf, (n, 1),       dtype=float32),
        }
        
        unbounded = lambda shape: Unbounded(shape=shape, dtype=float32)
        unbounded_tensors = {
            "gradient" : unbounded((n, dims)),
            "position" : unbounded((n, dims)),
            "reward"   : unbounded((n,  )),
            "velocity" : unbounded((n, dims)),
            "wind"     : unbounded((n, dims)),
        }
        
        return Composite(**bounded_tensors, **unbounded_tensors)
    
    def _extract_agent_states(self, data) -> tuple:
        """
        Extracts the position and velocity states for all agents from MuJoCo data.
        
        This method reads the actual physical state of each agent from the MuJoCo
        simulation data, enabling true multi-agent physics with individual states.
        
        Args:
            data: MjData object with current simulation state
            
        Returns:
            positions  : Tensor of shape [agent_count, spatial_dims]
            velocities : Tensor of shape [agent_count, spatial_dims]
        """
        positions = torch.from_numpy(
            data.qpos[:self.flock.state_size].copy().reshape(self.flock.shape)
        )
        velocities = torch.from_numpy(
            data.qvel[:self.flock.state_size].copy().reshape(self.flock.shape)
        )
    
        return positions, velocities
    
    def _generate_cube_formation(self, shape: tuple[int, int]) -> Tensor:
        """
        Generates points distributed in a hypercube grid formation.
        
        Creates a regular grid in N-dimensional space with points arranged
        in a hypercube. The algorithm:
            - Calculates the side length needed to accommodate N agents in a
              d-dimensional space: ceil(N^(1/d))
            - Creates a coordinate grid spanning [-1, 1] in each dimension
            - Returns the first N points from the flattened grid
        
        Args:
            shape : (n, dims) tuple from FlockModel

        Returns:
            A tensor of shape [n, dims] containing agent positions
        """
        n, dims     = shape
        side_length = math.ceil(n ** (1./dims))
        coords      = torch.linspace(-1, 1, side_length)
        grid        = torch.stack(
            dim     = -1,
            tensors = torch.meshgrid(*([coords] * dims), indexing='ij')
        )

        return grid.reshape(-1, dims)[:n]
    
    def _generate_sphere_formation(self, shape: tuple[int, int]) -> Tensor:
        """
        Generates points distributed evenly on a sphere (3D) or circle (2D).

        For 2D, places points evenly on a circle. For 3D, uses the Fibonacci
        lattice method, which provides excellent uniformity for arbitrary N.
        
        The Fibonacci lattice method creates a nearly-uniform distribution by:
            - Using the golden ratio φ = (1 + √5)/2 to create optimal angular spacing
            - Setting z-coordinates using a linear spacing from -1 to 1
            - Computing radius at each z value: r = √(1 - z²)
        
        Points are then placed at:
         
            (r·cos(θ), r·sin(θ), z) where θ = 2π·k/φ
        
        Args:
            shape : (n, dims) tuple from FlockModel

        Returns:
            A tensor of shape [n, dims] containing agent positions
        """
        n, dims = shape
        if dims == 2:
            thetas = torch.linspace(0, 2 * torch.pi, n, endpoint=False)
            return torch.stack(
                dim     = 1,
                tensors = (torch.cos(thetas), torch.sin(thetas))
            )

        indices      = torch.arange(n, dtype=float32)
        z            = 1 - (2 * indices) / (n - 1)
        radius       = torch.sqrt(1 - z*z)
        golden_angle = torch.pi * (3. - (5.**0.5))
        theta        = golden_angle * indices

        return torch.stack(
            dim     = 1,
            tensors = (
                torch.cos(theta) * radius,
                torch.sin(theta) * radius,
                z
            ),
        )
    
    def _initialize_physics(self, physics: PhysicsModel):
        """
        Dynamically generates and loads the MuJoCo physics model.

        This method creates a MuJoCo model with N distinct drone bodies based on
        the `agent_count` configuration parameter. Each drone has its own set of
        joints and actuators, enabling true multi-agent physics simulation.
        
        Args:
            physics: Physics configuration containing assets_dir and simulation_step

        Returns:
            A dictionary containing the MuJoCo model and data instances.
        """
        xml_string = generate_flock_xml(
            assets_dir      = physics.assets_dir,
            shape           = self.flock.shape,
            simulation_step = physics.simulation_step
        )
        
        return load_flock_model(xml_string)

    def _make_spec(self, td_params: TensorDictBase):
        """
        Implements the `torchrl.EnvBase` API for spec creation.
        
        In this project, specs are created internally during initialization
        via the _create_action_space and _create_observation_space methods.
        This method is provided to satisfy the EnvBase interface.
        """
        pass

    def _reset(self, tensordict=None, **kwargs) -> TensorDictBase:
        """
        Resets the environment to an initial state for a new episode.

        This method creates an initial `TensorDict` observation by placing agents
        according to the `initial_formation` specified in the flock config and
        querying the environmental data at these starting positions.
        
        The supported formation types include:
        - 'sphere'      : Agents are distributed evenly on a sphere surface using 
                          the Fibonacci lattice method for uniform distribution
        - 'cube'        : Agents are arranged in a uniform N-dimensional grid
        - 'murmuration' : (TBD) Agents mimic the controlled chaos of starlings

        The formation is scaled by the communication range and formation scale
        factor to ensure appropriate initial connectivity between agents.

        Returns:
            A `TensorDict` containing the initial observation of the flock.
        """
        match self.flock.initial_formation:
            case "cube":
                positions = self._generate_cube_formation(self.flock.shape)
            case "sphere" | _:
                positions = self._generate_sphere_formation(self.flock.shape)

        scaled_positions = (
            positions * 
            self.flock.communication_range *
            self.flock.formation_scale_factor
        )
        
        initial_observation = self.observation_spec.zero()
        initial_observation.update({
            "battery"  : torch.ones(self.flock.agent_count, 1),
            "position" : scaled_positions,
        })
        
        positions             = initial_observation["position"]
        temperature, gradient = self.data_source.query_thermal(positions)
        wind                  = self.data_source.query_wind(positions)
        
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
        
        self._set_physics_state(positions, torch.zeros_like(positions))
        
        return initial_observation

    def _set_physics_state(
        self, 
        positions  : Tensor, 
        velocities : Tensor
    ):
        """
        Sets the physics state for all agents in the MuJoCo simulation.
        
        This method updates the MuJoCo simulation state to match the provided
        positions and velocities for all agents. This is used both during reset
        to set the initial state and can be used to manually update agent positions.
        
        Args:
            positions  : Tensor [N, spatial_dims] containing agent positions
            velocities : Tensor [N, spatial_dims] containing agent velocities
        """
        data = self.physics_model["data"]
        mj.mj_resetData(self.physics_model["model"], data)
        
        data.qpos[:self.flock.state_size] = positions.reshape(-1).cpu().numpy()
        data.qvel[:self.flock.state_size] = velocities.reshape(-1).cpu().numpy()
        
        # Forward kinematics to update all derived quantities
        mj.mj_forward(self.physics_model["model"], data)

    def _set_seed(self, seed: int):
        """
        Sets the random seed for the environment.
        
        This method delegates to the provided seed_fn that was injected
        during initialization.
        
        Args:
            seed: Integer seed value to use for random number generation
        """
        self.seed_fn(seed)

    def _step(self, td: TensorDictBase) -> TensorDictBase:
        """
        Performs one discrete time step in the environment with true multi-agent physics.
        
        The process follows these steps:
            1. Extract individual agent control actions from the input `td`
            2. Apply each action to its corresponding agent's actuators
            3. Step the physics engine forward by `simulation_step` seconds
            4. Extract the updated agent states (positions, velocities)
            5. Create a new observation with environmental data at these positions
            6. Compute "reward" and "done" flags for each agent
            7. Return a `td` with the complete next state
        
        Args:
            td: A `TensorDict` containing the control `action` to apply,
                with shape [n_agents, action_dims]

        Returns:
            A `TensorDict` for the `next` state, including the new
            observation, reward, and done flag.
        """
        actions     = td.get("action")
        model, data = itemgetter("model", "data")(self.physics_model)
        reshaped    = actions[:, :self.flock.spatial_dims].flatten().cpu().numpy()

        if ctrl_count := min(len(reshaped), len(data.ctrl)):
            data.ctrl[:ctrl_count] = reshaped[:ctrl_count]
        
        mj.mj_step(model, data)
        
        position, velocity    = self._extract_agent_states(data)
        temperature, gradient = self.data_source.query_thermal(position)
        wind                  = self.data_source.query_wind(position)
        edge_index            = self._compute_edge_index(
            position = position,
            radius   = self.flock.communication_range
        )
        
        next_observation = self.observation_spec.zero()
        next_observation.update({
            "battery"     : torch.ones(self.flock.agent_count, 1),
            "edge_index"  : edge_index,
            "gradient"    : gradient,
            "position"    : position,
            "temperature" : temperature,
            "velocity"    : velocity,
            "wind"        : wind,
        })
    
        return next_observation