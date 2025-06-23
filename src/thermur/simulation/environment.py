"""
Defines the core `torchrl` simulation environment for the Thermur project.

This module provides the `SimulationEnv` class, which serves as the primary
interface for training and evaluating swarm policies. It implements the
`torchrl.EnvBase` API, making it compatible with the broader `torchrl`
ecosystem of collectors, replay buffers, and trainers.

The environment's main responsibility is to manage the state of the multi-agent
system, step the simulation forward in time, and provide observations and
rewards to the learning algorithm. It couples a rigid-body physics engine
(MuJoCo) with a dynamic environmental data source (e.g., WRF-Fire data).
"""
import mujoco
import torch

from pathlib      import Path
from tensordict   import TensorDict, TensorDictBase
from torch        import Tensor
from torchrl.envs import EnvBase
from typing       import Callable, Optional


class SimulationEnv(EnvBase):
    """
    A thermally-aware multi-agent flocking environment for `torchrl`.

    This environment simulates a swarm of `N` agents navigating a 3D space
    characterized by a pre-computed wind and temperature field. Agents must
    learn to move in a way that is legible to an observer while respecting
    strict thermal safety constraints.

    The state is represented by a `TensorDict` matching the specification
    defined externally, which includes agent kinematics, local thermal
    data, and the communication graph topology.
    """

    def __init__(
        self,
        config,
        action_spec        : TensorDictBase,
        compute_edge_index : Callable,
        data_source        : Callable,
        observation_spec   : TensorDictBase,
        seed_fn            : Optional[Callable] = None,
    ):
        """
        Initializes the Thermur environment with dependency injection.

        Args:
            config             : An instance containing simulation parameters.
            action_spec        : The action space specification.
            compute_edge_index : A callable that computes communication graph edges.
            data_source        : A callable that provides environmental data queries.
            observation_spec   : The observation space specification.
            seed_fn            : Optional callable for setting random seeds.
        """
        super().__init__(device="cpu")
        self.config             = config
        self.action_spec        = action_spec
        self.compute_edge_index = compute_edge_index
        self.data_source        = data_source
        self.observation_spec   = observation_spec
        self.seed_fn            = seed_fn
        self.physics_model      = self._initialize_physics()

    def _initialize_physics(self):
        """
        Dynamically generates and loads the MuJoCo physics model.

        This method creates a MuJoCo model with N distinct drone bodies based on
        the `agent_count` configuration parameter. Each drone has its own set of
        joints and actuators, enabling true multi-agent physics simulation.

        Returns:
            A dictionary containing the MuJoCo model and data instances.
        """
        from thermur import generate_swarm_xml, load_swarm_model
        
        agent_count     = self.config.swarm.agent_count
        spatial_dims    = self.config.swarm.spatial_dims
        simulation_step = self.config.environment.simulation_step
        assets_dir      = self.config.environment.assets_dir
        
        # Generate XML model with N distinct agent bodies
        xml_string = generate_swarm_xml(
            assets_dir      = assets_dir,
            agent_count     = agent_count,
            spatial_dims    = spatial_dims,
            simulation_step = simulation_step
        )
        
        return load_swarm_model(xml_string)

    def _reset(self, tensordict=None, **kwargs) -> TensorDictBase:
        """
        Resets the environment to an initial state for a new episode.

        This method creates an initial `TensorDict` observation by placing agents
        according to the `initial_formation` specified in the swarm config and
        querying the environmental data at these starting positions.
        
        The supported formation types include:
        - 'sphere'      : Agents are distributed evenly on a sphere surface using 
                          the Fibonacci lattice method for uniform distribution
        - 'cube'        : Agents are arranged in a uniform N-dimensional grid
        - 'murmuration' : (TBD) Agents mimic the controlled chaos of starlings

        The formation is scaled by the communication range and formation scale
        factor to ensure appropriate initial connectivity between agents.

        Returns:
            A `TensorDict` containing the initial observation of the swarm.
        """
        agent_count         = self.config.swarm.agent_count
        spatial_dims        = self.config.swarm.spatial_dims
        communication_range = self.config.swarm.communication_range
        formation_scale     = self.config.swarm.formation_scale_factor

        if self.config.swarm.initial_formation == "cube":
            positions = self._generate_cube_formation(agent_count, spatial_dims)

        else:
            positions = self._generate_sphere_formation(agent_count, spatial_dims)

        # Create a fresh TensorDict with proper structure and shape
        scaled_positions    = positions * communication_range * formation_scale
        initial_observation = TensorDict(
            {
                "position"         : scaled_positions,
                "velocity"         : torch.zeros_like(scaled_positions),
                "temperature"      : torch.zeros(agent_count),
                "temperature_grad" : torch.zeros((agent_count, spatial_dims)),
                "edge_index"       : torch.zeros((2, 0), dtype=torch.long),
                "reward"           : torch.zeros(agent_count),
                "done"             : torch.zeros(1, dtype=torch.bool),
                "_done"            : torch.zeros(agent_count, dtype=torch.bool)
            }, 
            batch_size = []
        )
        
        # Update with thermal data and edge_index
        positions       = initial_observation["position"]
        temp, temp_grad = self.data_source(positions)
        
        initial_observation.update(
            {
                "temperature"      : temp,
                "temperature_grad" : temp_grad,
                "edge_index"       : self.compute_edge_index(
                    pos = positions,
                    r   = self.config.swarm.communication_range
                )
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
        agent_count  = self.config.swarm.agent_count
        spatial_dims = self.config.swarm.spatial_dims
        data         = self.physics_model["data"]
        end_idx      = agent_count * spatial_dims
        
        mujoco.mj_resetData(self.physics_model["model"], data)
        
        data.qpos[:end_idx] = positions.reshape(-1).cpu().numpy()
        data.qvel[:end_idx] = velocities.reshape(-1).cpu().numpy()
        
        # Forward kinematics to update all derived quantities
        mujoco.mj_forward(self.physics_model["model"], data)
        
    def _generate_sphere_formation(
        self,
        n_agents : int,
        dims     : int
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
            n_agents : Number of agents to place [N]
            dims     : Spatial dimensions (2 or 3)

        Returns:
            A tensor of shape [n_agents, dims] containing agent positions
        """
        if dims == 2:
            thetas = torch.linspace(0, 2 * torch.pi, n_agents, endpoint=False)
            return torch.stack((torch.cos(thetas), torch.sin(thetas)), dim=1)

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
                z),
            )
    
    def _generate_cube_formation(
        self,
        n_agents : int,
        dims     : int
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
            n_agents : Number of agents to place [N]
            dims     : Spatial dimensions for the hypercube (2 or 3)

        Returns:
            A tensor of shape [n_agents, dims] containing agent positions
        """
        side_length = int(torch.ceil(Tensor(n_agents).float().pow(1./dims)))
        coords      = torch.linspace(-1, 1, side_length)
        grid        = torch.stack(
            dim     = -1,
            tensors = torch.meshgrid(*([coords] * dims), indexing='ij')
        )

        return grid.reshape(-1, dims)[:n_agents]
    
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
        agent_count  = self.config.swarm.agent_count
        spatial_dims = self.config.swarm.spatial_dims
        end_idx      = agent_count * spatial_dims

        positions = torch.from_numpy(
            data.qpos[:end_idx].copy().reshape(agent_count, spatial_dims)
        )
        velocities = torch.from_numpy(
            data.qvel[:end_idx].copy().reshape(agent_count, spatial_dims)
        )
    
        return positions, velocities

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
        actions      = td.get("action")
        model        = self.physics_model["model"]
        data         = self.physics_model["data"]
        agent_count  = self.config.swarm.agent_count
        spatial_dims = self.config.swarm.spatial_dims
        
        # Reshape actions to match control array layout
        reshaped_actions = actions[:, :spatial_dims].reshape(-1).cpu().numpy()
        
        ctrl_count       = min(len(reshaped_actions), len(data.ctrl))
        data.ctrl[:ctrl_count] = reshaped_actions[:ctrl_count]
        
        mujoco.mj_step(model, data)
        
        positions, velocities = self._extract_agent_states(data)
        temp, temp_grad       = self.data_source(positions)
        next_observation      = TensorDict(
            {
                "position"         : positions,
                "velocity"         : velocities,
                "temperature"      : temp,
                "temperature_grad" : temp_grad,
                "edge_index"       : self.compute_edge_index(
                    pos = positions,
                    r   = self.config.swarm.communication_range
                ),
                "reward"           : torch.zeros(agent_count),
                "done"             : torch.zeros(1, dtype=torch.bool),
                "_done"            : torch.zeros(agent_count, dtype=torch.bool)
            }, 
            batch_size = []
        )
    
        return next_observation

    def _make_spec(self, td_params: TensorDictBase):
        """
        Implements the `torchrl.EnvBase` API for spec creation.
        
        In this project, this method is a no-op because the observation and
        action specifications are pre-built via hydra-zen and injected into
        the constructor.
        """
        pass

    def _set_seed(self, seed: int):
        """
        Sets the random seed for the environment.
        
        This method delegates to the provided seed_fn if one was injected
        during initialization.
        
        Args:
            seed: Integer seed value to use for random number generation
        """
        if self.seed_fn is not None:
            self.seed_fn(seed)
