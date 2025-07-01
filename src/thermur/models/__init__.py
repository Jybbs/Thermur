"""
Neural network architectures for swarm control.

The GNN policy π_θ processes the swarm state as a dynamic graph G_t = (V, E_t)
where vertices represent agents and edges encode communication links. The 
architecture consists of:

1. Node encoder: Projects agent features [𝐩, 𝐯, T, ∇T, E] to hidden dimension
2. Message passing: Alternating GCN layers and GRU cells for temporal modeling  
3. Action decoder: Maps hidden states to nominal velocity commands 𝐮_nom

The policy learns to approximate the expert controller through behavioral cloning
on collected demonstration trajectories.
"""
from .gnn_policy import GNNPolicy