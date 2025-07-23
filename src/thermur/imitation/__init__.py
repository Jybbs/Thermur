"""
Training algorithms for imitation learning.

This package implements the behavioral cloning pipeline for training the GNN
policy from expert demonstrations. The training process involves:

1. Collecting trajectories using the expert controller
2. Storing experiences in a replay buffer for sampling
3. Minimizing the MSE loss between policy and expert actions
4. Periodic evaluation and checkpointing

The package is organized into:
- controller/    : Expert controller implementations (flocking, safety)
- lightning/     : PyTorch Lightning components (models, data, training)
- simulation/    : Environment and physics simulation (MuJoCo, XML generation)
- sources/       : Environmental data sources (WRF-Fire loaders)
- visualization/ : 3D rendering and monitoring (PyVista-based)

Import components directly from their respective submodules.
"""
