"""
Imitation learning configuration domain.

This package structures the configuration for behavioral cloning from expert demonstrations
in wildfire flock scenarios. The configuration hierarchy includes:

- Control schemas for Reynolds flocking rules with thermal constraints
- Learning schemas for GNN policy training hyperparameters
- Safety schemas for thermal barrier functions and collision avoidance
- Monitoring schemas for experiment tracking and visualization

The expert controller implements potential-based flocking where nominal actions are
computed as 𝐮_nom = -∇_x U(S_t), combining classical Reynolds rules with thermal
avoidance. The GNN policy π_θ learns to approximate this expert behavior through
supervised learning on collected trajectories.
"""
from .schemas   import *
from .workloads import imitation_cfg, register_imitation_cfgs