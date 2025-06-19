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

from tensordict   import TensorDict, TensorDictBase
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
        Loads and configures the MuJoCo physics model.

        This method initializes the MuJoCo simulation from the swarm XML model.
        It dynamically sets the number of agents in the model based on the
        provided configuration, ensuring the physics engine is correctly
        instantiated for the specified swarm size.

        Returns:
            A dictionary containing the MuJoCo model and data instances.
        """
        model_path = self.config.environment.assets_dir / "swarm.xml"
        model      = mujoco.MjModel.from_xml_path(model_path.as_posix())

        model.opt.timestep = self.config.environment.simulation_step

        return {
            "model" : model,
            "data"  : mujoco.MjData(model)
        }

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

        # Scale the positions
        scaled_positions = positions * communication_range * formation_scale
        
        # Create a fresh TensorDict with proper structure and shape
        initial_obs = TensorDict(
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
        positions       = initial_obs["position"]
        temp, temp_grad = self.data_source(positions)
        
        initial_obs.update(
            {
                "temperature"      : temp,
                "temperature_grad" : temp_grad,
                "edge_index"       : self.compute_edge_index(
                    pos = positions,
                    r   = self.config.swarm.communication_range
                )
            }
        )
        
        return initial_obs
        
    def _generate_sphere_formation(
        self,
        n_agents : int,
        dims     : int
    ) -> torch.Tensor:
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

        return torch.stack((
            torch.cos(theta) * radius,
            torch.sin(theta) * radius,
            z
        ), dim=1)
    
    def _generate_cube_formation(
        self,
        n_agents : int,
        dims     : int
    ) -> torch.Tensor:
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
        side_length = int(torch.ceil(torch.tensor(n_agents).float().pow(1./dims)))
        coords      = torch.linspace(-1, 1, side_length)
        grid        = torch.stack(
            dim     = -1,
            tensors = torch.meshgrid(*([coords] * dims), indexing='ij')
        )

        return grid.reshape(-1, dims)[:n_agents]

    def _step(self, td: TensorDictBase) -> TensorDictBase:
        """
        Performs one discrete time step in the environment.
        
        The process follows these steps:
            1. Extract agent control actions from the input `td`
            2. Apply these actions to the MuJoCo simulation
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
        actions = td.get("action")
        model   = self.physics_model["model"]
        data    = self.physics_model["data"]

        # Note: The MuJoCo model is set up with only 3 actuators in total (one for each dimension),
        # not per-agent actuators. For now, we'll use the average of all agent actions.
        #
        # In a future branch, we'll need to adjust the model to have separate actuators per agent.
        mean_action = actions.mean(dim=0).cpu().numpy()
        data.ctrl[:] = mean_action
        
        mujoco.mj_step(model, data)
        
        # Since there's only one agent in the physics model, we'll duplicate its state for all agents
        # with small offsets to simulate multiple agents
        qpos = torch.from_numpy(data.qpos).view(-1)
        qvel = torch.from_numpy(data.qvel).view(-1)
        
        agent_count      = self.config.swarm.agent_count
        spatial_dims     = self.config.swarm.spatial_dims
        agent_positions  = []
        agent_velocities = []
        
        for i in range(agent_count):

            # Add a small offset to each agent's position to spread them out
            offset = torch.tensor([
                0.2 * (i % 4 - 1.5),
                0.2 * ((i // 4) % 3 - 1),
                0.0
            ])
            agent_positions.append(qpos + offset)
            agent_velocities.append(qvel)
        
        positions       = torch.stack(agent_positions)
        velocities      = torch.stack(agent_velocities)
        temp, temp_grad = self.data_source(positions)
        
        # Create the next observation TensorDict
        next_obs = TensorDict(
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
    

        return next_obs

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
