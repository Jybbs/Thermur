# File: src/core/geometry.py

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
    dist_matrix = cdist(pos, pos, p=2.0)
    adj         = (dist_matrix < r) & (dist_matrix > 0)
    edge_index  = nonzero(adj, as_tuple=False).t().contiguous()

    return edge_index
