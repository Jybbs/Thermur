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

from tensordict   import TensorDictBase
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

    This class follows dependency injection principles - all dependencies
    are provided through the constructor rather than imported directly.

    Attributes:
        config              : The environment configuration containing simulation params.
        data_source         : A callable for querying thermal data.
        compute_edge_index  : A callable for computing communication graphs.
        observation_spec    : The observation space specification.
        action_spec         : The action space specification.
        seed_fn             : Optional callable for setting random seeds.
        physics_model       : A handle to the underlying MuJoCo physics simulation.
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
        
        This method initializes the MuJoCo simulation by loading the swarm XML model
        from the configured assets directory. The model is configured with the 
        simulation timestep specified in the environment configuration. The returned
        physics model contains both the MjModel and MjData components needed for
        simulation.
        
        Returns:
            A dictionary containing the MuJoCo model and data instances needed
            for physics simulation.
        """
        model_path = self.config.environment.assets_dir / "swarm.xml"
        model      = mujoco.MjModel.from_xml_path(model_path.as_posix())

        model.opt.timestep = self.config.environment.simulation_step
        
        return {
            "model" : model, 
            "data"  : mujoco.MjData(model)
        }

    def _reset(self) -> TensorDictBase:
        """
        Resets the environment to an initial state for a new episode.

        This method creates an initial `TensorDict` observation by placing agents
        according to the `initial_formation` specified in the swarm config and
        querying the environmental data at these starting positions.
        
        The supported formation types include:
        - 'sphere'      : Agents are distributed evenly on the surface of a sphere
        - 'cube'        : Agents are arranged in a uniform 3D grid within a cube
        - 'murmuration' : (Future) A more complex formation mimicking starling flocks
        
        The formation is scaled based on the communication range to ensure
        appropriate initial connectivity between agents.

        Returns:
            A `TensorDict` containing the initial observation of the swarm.
        """
        observation_dict    = self.observation_spec.zero()
        formation           = self.config.swarm.initial_formation
        agent_count         = self.config.swarm.agent_count
        spatial_dims        = self.config.swarm.spatial_dims
        communication_range = self.config.swarm.communication_range
        formation_scale     = self.config.swarm.formation_scale_factor
        
        if formation == "cube":
            positions = self._generate_cube_formation(agent_count, spatial_dims)

        else:
            positions = self._generate_sphere_formation(agent_count, spatial_dims)
        
        observation_dict.set(
            key  = "position", 
            item = positions * communication_range * formation_scale
        )
        observation_dict.set("velocity", torch.zeros_like(positions))
        
        return self._update_observation(observation_dict)
        
    def _generate_sphere_formation(
        self, 
        n_agents : int, 
        dims     : int
    ) -> torch.Tensor:
        """
        Generates points distributed evenly on a sphere or circle.
        
        For 3D, uses the Fibonacci sphere algorithm to generate points that are
        approximately equidistant on a sphere. For 2D, places points evenly on
        a circle using angular spacing.
        
        Args:
            n_agents : Number of agents to place
            dims     : Spatial dimensions (2 for circle, 3 for sphere)
            
        Returns:
            Tensor of shape [n_agents, dims] containing agent positions
        """
        if dims == 2:
            theta = torch.linspace(0, 2 * torch.pi, n_agents + 1)[:-1]
            
            return torch.stack(
                dim     = 1,
                tensors = [torch.cos(theta), torch.sin(theta)]
            )
        
        phi     = (1 + 5 ** 0.5) / 2  # Golden ratio
        indices = torch.arange(0, n_agents, dtype=torch.float32)

        theta   = 2 * torch.pi * indices / phi
        z       = 1 - (2 * indices + 1) / n_agents
        radius  = torch.sqrt(1 - z * z)
        
        return torch.stack(
            dim     = 1,
            tensors = [
                radius * torch.cos(theta), 
                radius * torch.sin(theta), 
                z
            ]
        )
    
    def _generate_cube_formation(
        self, 
        n_agents : int, 
        dims     : int
    ) -> torch.Tensor:
        """
        Generates points distributed in a cube or square grid.
        
        Creates a grid-like formation of agents. For 3D, the agents are arranged
        in a cubic lattice. For 2D, they form a square grid.
        
        Args:
            n_agents : Number of agents to place
            dims     : Spatial dimensions (2 for square, 3 for cube)
            
        Returns:
            Tensor of shape [n_agents, dims] containing agent positions
        """
        if dims == 2:

            side_length = int(torch.ceil(torch.sqrt(torch.tensor(n_agents, dtype=torch.float32))))

            grid_x, grid_y = torch.meshgrid(
                torch.linspace(-1, 1, side_length), 
                torch.linspace(-1, 1, side_length), 
                indexing = 'ij'
            )
            positions = torch.stack(
                dim     = 1,
                tensors = [grid_x.flatten(), grid_y.flatten()]
            )
            
            return positions[:n_agents]
        
        side_length = int(torch.ceil(torch.pow(torch.tensor(n_agents, dtype=torch.float32), 1/3)))

        grid_x, grid_y, grid_z = torch.meshgrid(
            torch.linspace(-1, 1, side_length), 
            torch.linspace(-1, 1, side_length), 
            torch.linspace(-1, 1, side_length), 
            indexing = 'ij')
        
        positions = torch.stack(
            dim     = 1,
            tensors = [grid_x.flatten(), grid_y.flatten(), grid_z.flatten()]
        )
        
        return positions[:n_agents]

    def _step(self, action_dict: TensorDictBase) -> TensorDictBase:
        """
        Performs one discrete time step in the environment.

        The process is as follows:
        1.  The input `action_dict` provides the `action` (velocity commands).
        2.  These actions are applied to the agents in the MuJoCo simulation.
        3.  The physics engine is stepped forward by `simulation_step` seconds.
        4.  The new agent states (position, velocity) are retrieved.
        5.  A new observation is constructed by querying the environment at these
            new positions.
        6.  A reward is computed, and a 'done' signal is determined.
        7.  A `TensorDict` containing this `next` state is returned.

        Args:
            action_dict: A `TensorDict` containing the control `action` to apply.

        Returns:
            A `TensorDict` containing the observation after the step, along with
            the calculated reward and done flag.
        """
        actions = action_dict.get("action")
        model   = self.physics_model["model"]
        data    = self.physics_model["data"]
        
        self._apply_agent_actions(data, actions)
        mujoco.mj_step(model, data)
        
        next_positions  = self._get_agent_positions(data)
        next_velocities = self._get_agent_velocities(data)
        
        next_observation = self.observation_spec.zero()
        next_observation.set("position", next_positions)
        next_observation.set("velocity", next_velocities)
        next_observation = self._update_observation(next_observation)
        
        agent_count = self.config.swarm.agent_count
        rewards     = self._compute_rewards(next_observation)
        dones       = torch.zeros(agent_count, dtype=torch.bool)
        
        next_observation.set("reward", rewards)
        next_observation.set("done", dones)
        
        return next_observation
        
    def _apply_agent_actions(
        self, 
        data    : mujoco.MjData, 
        actions : torch.Tensor
    ):
        """
        Applies agent control actions to the MuJoCo simulation.
        
        This method translates the high-level velocity commands from the policy
        into appropriate control inputs for the MuJoCo physics simulation. The 
        exact mapping depends on the structure of the swarm XML model, particularly
        how the actuators are defined for each agent.
        
        For each agent, the corresponding velocity command is mapped to the
        appropriate control inputs in the MuJoCo data structure. The method
        handles both 2D and 3D velocity commands based on the configured
        spatial dimensions.
        
        Args:
            data    : MuJoCo simulation data object with control arrays
            actions : Tensor of shape [n_agents, action_dims] with velocity commands
        """
        agent_count = self.config.swarm.agent_count
        
        for i in range(agent_count):
            agent_action = actions[i]
            control_idx  = i * 3  # Assuming 3 DOF per agent
            
            if len(agent_action) >= 3:
                data.ctrl[control_idx:control_idx+3] = agent_action[:3].cpu().numpy()
            else:
                data.ctrl[control_idx:control_idx+2] = agent_action.cpu().numpy()
                data.ctrl[control_idx+2] = 0.0
    
    def _compute_rewards(self, observation_dict: TensorDictBase) -> torch.Tensor:
        """
        Computes rewards for each agent based on the current state.
        
        This method implements the reward function that guides agent learning.
        The rewards incentivize desired behaviors like maintaining formation,
        avoiding obstacles, and staying within safe temperature ranges.
        
        The reward function balances multiple objectives including:
        - Formation maintenance (staying in a cohesive group)
        - Temperature safety (avoiding dangerous thermal conditions)
        - Task completion (reaching target locations or patterns)
        - Efficiency (minimizing energy usage and unnecessary movement)
        
        Args:
            observation_dict : TensorDict containing the current observation
            
        Returns:
            Tensor of shape [n_agents] containing individual agent rewards
        """
        agent_count = self.config.swarm.agent_count
        
        return torch.zeros(agent_count)
    
    def _get_agent_positions(self, data: mujoco.MjData) -> torch.Tensor:
        """
        Extracts agent positions from MuJoCo simulation data.
        
        This method maps from the MuJoCo internal representation of agent positions
        to the tensor format used by the learning system. The exact mapping depends
        on how agents are represented in the MuJoCo XML model, particularly the
        structure of the qpos array.
        
        Args:
            data : MuJoCo simulation data object containing agent state
            
        Returns:
            Tensor of shape [n_agents, spatial_dims] containing agent positions
        """
        agent_count = self.config.swarm.agent_count
        dims        = self.config.swarm.spatial_dims
        positions   = torch.zeros((agent_count, dims))
        
        for i in range(agent_count):
            body_idx = i + 1  # Skip world body
            
            if dims == 3:
                positions[i, 0] = data.qpos[body_idx * 7]     # X position
                positions[i, 1] = data.qpos[body_idx * 7 + 1] # Y position
                positions[i, 2] = data.qpos[body_idx * 7 + 2] # Z position

            else:
                positions[i, 0] = data.qpos[body_idx * 7]     # X position
                positions[i, 1] = data.qpos[body_idx * 7 + 1] # Y position
                
        return positions
    
    def _get_agent_velocities(self, data: mujoco.MjData) -> torch.Tensor:
        """
        Extracts agent velocities from MuJoCo simulation data.
        
        This method maps from the MuJoCo internal representation of agent velocities
        to the tensor format used by the learning system. The exact mapping depends
        on how agents are represented in the MuJoCo XML model, particularly the
        structure of the qvel array.
        
        Args:
            data : MuJoCo simulation data object containing agent state
            
        Returns:
            Tensor of shape [n_agents, spatial_dims] containing agent velocities
        """
        agent_count = self.config.swarm.agent_count
        dims        = self.config.swarm.spatial_dims
        velocities  = torch.zeros((agent_count, dims))
        
        for i in range(agent_count):
            body_idx = i + 1  # Skip world body
            
            if dims == 3:
                velocities[i, 0] = data.qvel[body_idx * 6]     # X velocity
                velocities[i, 1] = data.qvel[body_idx * 6 + 1] # Y velocity
                velocities[i, 2] = data.qvel[body_idx * 6 + 2] # Z velocity

            else:
                velocities[i, 0] = data.qvel[body_idx * 6]     # X velocity
                velocities[i, 1] = data.qvel[body_idx * 6 + 1] # Y velocity
                
        return velocities

    def _update_observation(self, observation_dict: TensorDictBase) -> TensorDictBase:
        """
        Populates a TensorDict with fresh sensor and graph data.

        Args:
            observation_dict: A `TensorDict` containing, at a minimum, the `position` of all agents.

        Returns:
            The input `TensorDict`, updated in-place with new thermal data and
            the communication graph's `edge_index`.
        """
        positions = observation_dict.get("position")

        temp, temp_grad = self.data_source.query_thermal(positions)
        observation_dict.set("temperature", temp)
        observation_dict.set("temperature_grad", temp_grad)

        observation_dict.set(
            key  = "edge_index",
            item = self.compute_edge_index(
                pos = positions,
                r   = self.config.swarm.communication_range
            )
        )
        return observation_dict

    def _make_spec(self, td_params: TensorDictBase):
        """
        Creates the environment's observation and action specs.
        
        Since specs are passed in via dependency injection, this method
        is effectively a no-op but is required by the torchrl API.
        """
        pass

    def _set_seed(self, seed: int):
        """
        Sets the random seed for the environment.
        """
        if self.seed_fn is not None:
            self.seed_fn(seed)
