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
from __future__   import annotations
from .generator   import XMLGenerator
from math         import ceil
from operator     import itemgetter
from pathlib      import Path
from torch        import Size
from torchrl.data import Bounded, Composite, Unbounded
from torchrl.envs import EnvBase
from typing       import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .loader                     import WRFDataSource
    from config.imitation.controller import FlockModel
    from config.imitation.simulation import PhysicsModel
    from config.types                import MujocoModel
    from tensordict                  import TensorDictBase
    from torch                       import Tensor
    from torchrl.data                import TensorSpec

import mujoco as mj
import torch  as th


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
    
    physics_model: MujocoModel

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
        self.flock         = flock
        self.physics       = physics
        self.wrf           = wrf
        self.xml_generator = XMLGenerator(Path(self.physics.assets_dir))
        self.physics_model = self._initialize_physics()
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
    
    def _extract_agent_states(self, data: Any) -> tuple[Tensor, Tensor]:  # MjData
        """
        Extracts the position and velocity states for all agents from MuJoCo data.
        
        This method reads the actual physical state of each agent from the MuJoCo
        simulation data, enabling true multi-agent physics with individual states.
        
        Args:
            data: MjData object with current simulation state
            
        Returns:
            positions  : Tensor of shape [agent_count, 3]
            velocities : Tensor of shape [agent_count, 3]
        """
        positions = th.from_numpy(
            data.qpos[:self.flock.agent_count * 3].copy().reshape(self.flock.agent_count, 3)
        )
        velocities = th.from_numpy(
            data.qvel[:self.flock.agent_count * 3].copy().reshape(self.flock.agent_count, 3)
        )
    
        return positions, velocities
    
    def _generate_cube_formation(self, n: int) -> Tensor:
        """
        Generates points distributed in a hypercube grid formation.
        
        Creates a regular grid in N-dimensional space with points arranged
        in a hypercube. The algorithm:
            - Calculates the side length needed to accommodate N agents in a
              d-dimensional space: ceil(N^(1/d))
            - Creates a coordinate grid spanning [-1, 1] in each dimension
            - Returns the first N points from the flattened grid
        
        Args:
            n: Number of agents in the flock

        Returns:
            A tensor of shape [n, 3] containing agent positions
        """
        side_length = ceil(n ** (1./3))
        coords      = th.linspace(-1, 1, side_length)
        grid        = th.stack(
            dim     = -1,
            tensors = th.meshgrid(coords, coords, coords, indexing='ij')
        )

        return grid.reshape(-1, 3)[:n]
    
    def _generate_sphere_formation(self, n: int) -> Tensor:
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
    
    def _initialize_physics(self):
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
        xml_string = self.xml_generator.generate_xml(
            agent_count     = self.flock.agent_count,
            simulation_step = self.physics.simulation_step
        )
        return self.xml_generator.load_model(xml_string)

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
                positions = self._generate_cube_formation(self.flock.agent_count)
            case "sphere" | _:
                positions = self._generate_sphere_formation(self.flock.agent_count)

        scaled_positions = (
            positions * 
            self.flock.communication_range *
            self.flock.formation_scale_factor
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
        
        self._set_physics_state(positions, th.zeros_like(positions))
        
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
            positions  : Tensor [N, 3] containing agent positions
            velocities : Tensor [N, 3] containing agent velocities
        """
        data = self.physics_model["data"]
        getattr(mj, 'mj_resetData')(self.physics_model["model"], data)
        
        data.qpos[:self.flock.agent_count * 3] = positions.reshape(-1).cpu().numpy()
        data.qvel[:self.flock.agent_count * 3] = velocities.reshape(-1).cpu().numpy()
        
        # Forward kinematics to update all derived quantities
        getattr(mj, 'mj_forward')(self.physics_model["model"], data)

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
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
        actions     = tensordict.get("action")
        model, data = itemgetter("model", "data")(self.physics_model)
        reshaped    = actions[:, :3].flatten().cpu().numpy()

        if ctrl_count := min(len(reshaped), len(data.ctrl)):
            data.ctrl[:ctrl_count] = reshaped[:ctrl_count]
        
        getattr(mj, 'mj_step')(model, data)
        
        position, velocity    = self._extract_agent_states(data)
        temperature, gradient = self.wrf.query_thermal(position)
        wind                  = self.wrf.query_wind(position)
        edge_index            = self._compute_edge_index(
            position = position,
            radius   = self.flock.communication_range
        )
        
        next_observation = self.observation_spec.zero()
        next_observation.update({
            "battery"     : th.ones(self.flock.agent_count, 1),
            "edge_index"  : edge_index,
            "gradient"    : gradient,
            "position"    : position,
            "temperature" : temperature,
            "velocity"    : velocity,
            "wind"        : wind,
        })
    
        return next_observation