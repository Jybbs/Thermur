"""
Offline imitation learning for flocking control.

This package provides expert demonstrations and training infrastructure for
imitation learning of murmuration dynamics. The pipeline uses PyTorch Geometric
for efficient graph-based learning from offline trajectories.

Key components:
- controller/  : Expert murmuration controller with topological interactions
- environment/ : Trajectory generation and WRF environmental data loading
- training/    : PyTorch Lightning modules for behavioral cloning

The training process:
1. Generate offline trajectories using the expert controller
2. Train GNN policy via behavioral cloning on PyG Data objects
3. Evaluate policy performance against expert demonstrations
"""
