"""
Hydra-zen builders for swarm data specifications.

This module provides factory functions that create torchrl spec objects
directly, defining the structure and constraints for actions and observations
in the simulation without intermediate Pydantic models.
"""
from functools    import partial
from hydra_zen    import builds, SI
from torch        import float32, int64
from torchrl.data import Composite, TensorSpec


def _edge_index_spec(n: int):
    """
    Creates the edge index spec for a fully connected graph.
    """
    return TensorSpec(
        shape  = (2, n * (n - 1)),
        device = "cpu",
        dtype  = int64,
    ).to_owned_by(())


def _float_spec(shape=1):
    """
    Helper to create float32 TensorSpec with configurable shape.
    """
    return builds(
        TensorSpec, 
        shape = shape, 
        dtype = float32
    )


build_action_spec = builds(
    Composite,
    shape       = SI("${swarm.agent_count}"),
    action      = _float_spec(SI("${swarm.spatial_dims}")),
    zen_partial = True,
)

build_observation_spec = builds(
    Composite,
    shape            = SI("${swarm.agent_count}"),
    battery          = _float_spec(),
    edge_index       = builds(_edge_index_spec, n = SI("${swarm.agent_count}")),
    position         = _float_spec(SI("${swarm.spatial_dims}")),
    temperature      = _float_spec(),
    temperature_grad = _float_spec(SI("${swarm.spatial_dims}")),
    velocity         = _float_spec(SI("${swarm.spatial_dims}")),
    zen_partial      = True,
)
