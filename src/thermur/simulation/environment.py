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
from configs.imitation import PhysicsModel, FlockModel
from operator          import itemgetter
from tensordict        import TensorDict, TensorDictBase
from torch             import Tensor
from torchrl.envs      import EnvBase
from torch             import cdist, nonzero
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
        action_spec      : TensorDictBase,
        data_source      : Any,
        flock            : FlockModel,
        observation_spec : TensorDictBase,
        physics          : PhysicsModel,
        seed_fn          : Callable,
    ):
        """
        Initializes the Thermur environment with dependency injection.

        Args:
            action_spec      : The action space specification.
            data_source      : A callable that provides environmental data queries.
            flock            : Flock parameters configuration.
            observation_spec : The observation space specification.
            physics          : Physics simulation configuration.
            seed_fn          : Callable for setting random seeds.
        """
        super().__init__(device="cpu")
        self.action_spec      = action_spec
        self.data_source      = data_source
        self.flock            = flock
        self.observation_spec = observation_spec
        self.physics          = physics
        self.physics_model    = self._initialize_physics()
        self.seed_fn          = seed_fn
    
    def _compute_edge_index(self, pos: Tensor, r: float) -> Tensor:
        """
        Computes the graph connectivity based on metric distance.

        This function builds an `edge_index` for `torch-geometric` by finding all
        pairs of nodes (i, j) where the Euclidean distance is less than `r`.
        It avoids self-loops.

        Args:
            pos : A tensor of node positions, shape (num_nodes, num_dims).
            r   : The communication radius.

        Returns:
            An `edge_index` tensor of shape (2, num_edges), suitable for a
            `torch_geometric.data.Data` object.
        """
        distances = cdist(pos, pos)
        mask      = (distances < r) & (distances > 0)
        return nonzero(mask, as_tuple=False).t().contiguous()
    
    def _create_base_observation(self, include_defaults: bool = False) -> dict:
        """
        Creates the base observation dictionary with zero-initialized tensors.
        
        This helper method DRYs up the common tensor initialization patterns
        used in both _reset and _step methods.
        
        Args:
            include_defaults: If True, includes default zero tensors for all fields
        
        Returns:
            Dictionary with zero-initialized tensors for the observation
        """
        n = self.flock.agent_count
        d = self.flock.spatial_dims
        
        base = {
            "_done"  : torch.zeros(n, dtype=torch.bool),
            "done"   : torch.zeros(1, dtype=torch.bool),
            "reward" : torch.zeros(n),
        }
        
        if include_defaults:
            base.update({
                "edge_index"       : torch.zeros((2, 0), dtype=torch.long),
                "temperature"      : torch.zeros(n),
                "temperature_grad" : torch.zeros((n, d)),
            })
        
        return base
    
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
        end_idx      = self.flock.agent_count * self.flock.spatial_dims

        reshape_dims = (self.flock.agent_count, self.flock.spatial_dims)
        positions = torch.from_numpy(
            data.qpos[:end_idx].copy().reshape(reshape_dims)
        )
        velocities = torch.from_numpy(
            data.qvel[:end_idx].copy().reshape(reshape_dims)
        )
    
        return positions, velocities
    
    def _generate_cube_formation(
        self,
        dims     : int,
        n_agents : int
    ) -> Tensor:
        """
        Generates points distributed in a hypercube grid formation.
        
        Creates a regular grid in N-dimensional space with points arranged
        in a hypercube. The algorithm:
            - Calculates the side length needed to accommodate N agents in a
              d-dimensional space: ceil(N^(1/d))
            - Creates a coordinate grid spanning [-1, 1] in each dimension
            - Returns the first N points from the flattened grid
        
        Args:
            dims     : Spatial dimensions for the hypercube (2 or 3)
            n_agents : Number of agents to place [N]

        Returns:
            A tensor of shape [n_agents, dims] containing agent positions
        """
        side_length = math.ceil(n_agents ** (1./dims))
        coords      = torch.linspace(-1, 1, side_length)
        grid        = torch.stack(
            dim     = -1,
            tensors = torch.meshgrid(*([coords] * dims), indexing='ij')
        )

        return grid.reshape(-1, dims)[:n_agents]
    
    def _generate_sphere_formation(
        self,
        dims     : int,
        n_agents : int
    ) -> Tensor:
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
            dims     : Spatial dimensions (2 or 3)
            n_agents : Number of agents to place [N]

        Returns:
            A tensor of shape [n_agents, dims] containing agent positions
        """
        if dims == 2:
            thetas = torch.linspace(0, 2 * torch.pi, n_agents, endpoint=False)
            return torch.stack(
                dim     = 1,
                tensors = (torch.cos(thetas), torch.sin(thetas))
            )

        indices      = torch.arange(n_agents, dtype=torch.float32)
        z            = 1 - (2 * indices) / (n_agents - 1)
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
    
    def _initialize_physics(self):
        """
        Dynamically generates and loads the MuJoCo physics model.

        This method creates a MuJoCo model with N distinct drone bodies based on
        the `agent_count` configuration parameter. Each drone has its own set of
        joints and actuators, enabling true multi-agent physics simulation.

        Returns:
            A dictionary containing the MuJoCo model and data instances.
        """
        # Generate XML model with N distinct agent bodies
        xml_string = generate_flock_xml(
            agent_count     = self.flock.agent_count,
            assets_dir      = self.physics.assets_dir,
            simulation_step = self.physics.simulation_step,
            spatial_dims    = self.flock.spatial_dims
        )
        
        return load_flock_model(xml_string)

    def _make_spec(self, td_params: TensorDictBase):
        """
        Implements the `torchrl.EnvBase` API for spec creation.
        
        In this project, this method is a no-op because the observation and
        action specifications are pre-built via hydra-zen and injected into
        the constructor.
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
        formation_args = (self.flock.spatial_dims, self.flock.agent_count)
        match self.flock.initial_formation:
            case "cube":
                positions = self._generate_cube_formation(*formation_args)
            case "sphere" | _:
                positions = self._generate_sphere_formation(*formation_args)

        # Create a fresh TensorDict with proper structure and shape
        scale_factor     = self.flock.communication_range * self.flock.formation_scale_factor
        scaled_positions = positions * scale_factor
        
        base_obs = self._create_base_observation(include_defaults=True)
        base_obs.update({
            "position" : scaled_positions,
            "velocity" : torch.zeros_like(scaled_positions)
        })
        
        initial_observation = TensorDict(base_obs, batch_size=[])
        
        # Update with thermal data and edge_index
        positions       = initial_observation["position"]
        temp, temp_grad = self.data_source.query_thermal(positions)
        wind            = self.data_source.query_wind(positions)
        
        initial_observation.update(
            {
                "edge_index"       : self._compute_edge_index(
                    pos = positions,
                    r   = self.flock.communication_range
                ),
                "temperature"      : temp,
                "temperature_grad" : temp_grad,
                "wind"             : wind
            }
        )
        
        # Set the initial positions in the MuJoCo simulation
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
        data         = self.physics_model["data"]
        end_idx      = self.flock.agent_count * self.flock.spatial_dims
        
        mj.mj_resetData(self.physics_model["model"], data)
        
        data.qpos[:end_idx] = positions.reshape(-1).cpu().numpy()
        data.qvel[:end_idx] = velocities.reshape(-1).cpu().numpy()
        
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
        
        pos, vel         = self._extract_agent_states(data)
        temp, temp_grad  = self.data_source.query_thermal(pos)
        wind             = self.data_source.query_wind(pos)
        edge_index       = self._compute_edge_index(
            pos = pos,
            r   = self.flock.communication_range
        )
        
        base_obs = self._create_base_observation()
        base_obs.update({
            "edge_index"       : edge_index,
            "position"         : pos,
            "temperature"      : temp,
            "temperature_grad" : temp_grad,
            "velocity"         : vel,
            "wind"             : wind
        })
        
        next_observation = TensorDict(base_obs, batch_size=[])
    
        return next_observation
