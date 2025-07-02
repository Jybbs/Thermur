"""
Provides stateless utility functions for geometric computations.

These functions are designed to be pure, vectorized operations on torch
Tensors, suitable for use in environments, policies, and analysis scripts.
"""
from torch import cdist, nonzero, Tensor


def compute_edge_index(pos: Tensor, r: float) -> Tensor:
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
    return nonzero(
        input    = (cdist(pos, pos) < r) & (cdist(pos, pos) > 0), 
        as_tuple = False
    ).t().contiguous()
