"""
Environment components for offline trajectory generation.

This package provides lightweight physics simulation and environmental data
access for generating expert demonstrations without TorchRL overhead.

Key components:
- WRFLoader           : Environmental data loader (wind, temperature fields)
- TrajectoryGenerator : Physics simulation for offline trajectory generation  

The trajectory generator uses Euler integration for agent dynamics and returns
PyG Data objects suitable for offline imitation learning.
"""
from .loader       import *
from .trajectories import *
