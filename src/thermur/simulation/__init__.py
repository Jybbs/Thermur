"""
MuJoCo-based simulation environment for drone swarms.

This package provides the physics simulation for training and evaluation. The
environment models N quadrotor agents navigating through thermal fields while
maintaining flocking behavior. Key features include:

- Dynamic graph topology based on communication range
- Realistic thermal field generation with configurable hazards
- Vectorized physics integration for efficient parallel simulation
- TorchRL-compatible interface for seamless integration with learning algorithms

The edge index computation uses spatial proximity to determine the communication
graph G_t at each timestep, enabling decentralized coordination.
"""
from .environment import SimulationEnv
from .geometry    import compute_edge_index