"""
Thermur: Thermally-aware drone flock control via imitation learning.

This package implements a complete system for training autonomous drone flocks
to navigate wildfire scenarios using graph neural networks and behavioral cloning.

Key components:
- cli/: Command-line interface for training, validation, and monitoring
- imitation/: Complete imitation learning pipeline including:
  - controller/: Expert controllers with Reynolds flocking and thermal constraints
  - lightning/: PyTorch Lightning training infrastructure
  - simulation/: MuJoCo-based physics environment
  - sources/: WRF-Fire data loading and interpolation
  - visualization/: Real-time 3D rendering with PyVista

The system uses PyTorch Lightning to streamline the training pipeline, reducing
boilerplate while maintaining flexibility. Configuration is managed through
Hydra-zen with Pydantic validation.

Import components directly from their respective submodules.
"""
