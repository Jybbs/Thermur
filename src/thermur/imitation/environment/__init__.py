"""
Environment components for offline trajectory generation.

This package provides lightweight physics simulation and environmental data
access for generating expert trajectories for behavioral cloning.

Key components:
- TrajectoryGenerator : Physics simulation for offline trajectory generation
- WRFLoader           : Environmental data loader (wind, temperature fields)

The trajectory generator uses Euler integration for agent dynamics and returns
PyG Data objects suitable for offline imitation learning.
"""
from .generator import *
from .loader    import *
