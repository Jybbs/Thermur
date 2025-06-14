"""
Defines the canonical data structures for the project using torchrl's
TensorDict.

This module is the definitive source for the shape, dtype, and semantics of
all data passed between the environment, policy, and replay buffer. It is
intended to house the primary data structure definitions for the project.
"""
from functools            import partial
from tensordict           import TensorDict, TensorDictBase
from torch                import Tensor, cat, float32, int64
from torch_geometric.data import Data
from torchrl.data         import Composite, TensorSpec


class SwarmData(TensorDict):
    """
    A custom TensorDict subclass for swarm data.

    This class provides a type-safe, property-based interface for accessing
    the canonical keys defined in SwarmDataSpec. It eliminates the need for
    "magic strings" when consuming observation data.
    """
    @property
    def position(self) -> Tensor:
        return self.get("position")

    @property
    def velocity(self) -> Tensor:
        return self.get("velocity")

    @property
    def temperature(self) -> Tensor:
        return self.get("temperature")

    @property
    def temperature_grad(self) -> Tensor:
        return self.get("temperature_grad")

    @property
    def battery(self) -> Tensor:
        return self.get("battery")

    @property
    def edge_index(self) -> Tensor:
        return self.get("edge_index")


class SwarmDataSpec:
    """
    A namespace for defining the project's primary TensorDict specification.

    This class centralizes the definition of the swarm's data structure. It
    provides class methods to create `Composite` spec objects that `torchrl`
    environments and replay buffers use for validation and memory allocation.
    """

    @classmethod
    def get_action_spec(cls, config) -> Composite:
        """
        Returns the torchrl spec for an action TensorDict.

        The action is the desired N-dimensional velocity vector (𝐮) for each
        agent that is output by the policy. This spec's shape is derived
        directly from the provided swarm configuration.

        Args:
            config: A swarm configuration instance (from environment config)
                   containing the agent count and number of spatial dimensions.

        Returns:
            A `Composite` object defining the action structure.
        """
        return Composite(
            shape  = (config.agent_count,),
            action = TensorSpec(
                shape = (config.spatial_dims,),
                dtype = float32,
            )
        )

    @classmethod
    def get_observation_spec(cls, config) -> Composite:
        """
        Returns the torchrl spec for an observation TensorDict.

        An observation consists of agent-specific state vectors and the shared
        graph topology (edge_index). This spec defines the structure of the
        agent state vector s, where s = [𝐩, 𝐯, T, ∇T, E].

        Args:
            config: A swarm configuration instance (from environment config)
                   containing the agent count and number of spatial dimensions.

        Returns:
            A `Composite` object defining the observation structure.
        """
        agent_count = config.swarm.agent_count
        dims        = config.swarm.spatial_dims
        AgentSpec   = partial(TensorSpec, dtype=float32)

        return Composite(
            shape            = (agent_count,),
            position         = AgentSpec(shape=(dims,)), # 𝐩
            velocity         = AgentSpec(shape=(dims,)), # 𝐯
            temperature      = AgentSpec(shape=(1,)),    # T
            temperature_grad = AgentSpec(shape=(dims,)), # ∇T
            battery          = AgentSpec(shape=(1,)),    # E
            edge_index       = TensorSpec(
                shape  = (2, agent_count * (agent_count - 1)),
                device = "cpu",
                dtype  = int64,
            ).to_owned_by(()),
        )

    @staticmethod
    def to_torch_geometric(td: TensorDictBase) -> Data:
        """
        Converts a swarm-structured TensorDict to a PyG Data object.

        This is a crucial utility for interfacing with the GNN policy, which
        expects `torch_geometric.data.Data` as input. It explicitly
        concatenates the individual state tensors from the TensorDict into a
        single node feature matrix, `x`, as required by most GNN layers.

        Args:
            td: A TensorDict containing the observation spec keys. Assumed to
                have a leading dimension corresponding to the agent count.

        Returns:
            A `torch_geometric.data.Data` object ready for GNN processing.
        """
        node_features: list[Tensor | None] = [
            td.get("position"),
            td.get("velocity"),
            td.get("temperature"),
            td.get("temperature_grad"),
            td.get("battery"),
        ]

        return Data(
            x          = cat([f for f in node_features if f is not None], dim=-1),
            edge_index = td.get("edge_index"),
            pos        = td.get("position"),
        )
