"""
Thermur: Thermally-aware drone flock control via imitation learning.

This package implements a complete system for training autonomous drone flocks
to navigate wildfire scenarios using graph neural networks and behavioral cloning.

Key components:
- cli/: Command-line interface for training and validation
- imitation/: Complete imitation learning pipeline including:
  - controller/: Expert controllers with Reynolds flocking and thermal constraints
  - simulation/: Physics environment with Euler integration
  - training/: Training infrastructure with metrics collection

The system uses PyTorch Lightning to streamline the training pipeline, reducing
boilerplate while maintaining flexibility. Configuration is managed through
Hydra-zen with Pydantic validation.

Import components directly from their respective submodules.
"""
