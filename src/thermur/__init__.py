"""
Thermur: Thermally-aware drone flock control via imitation learning.

This package implements a complete system for training autonomous drone flocks
to navigate wildfire scenarios using graph neural networks and behavioral cloning.

Key components:
- cli/       : Command-line interface for training, validation, monitoring, and run management
- imitation/ : Complete imitation learning pipeline including:
  - controller/  : Offline expert demonstration generation with murmuration behaviors and thermal safety
  - environment/ : WRF-Fire NetCDF data loading and flock physics trajectory simulation
  - training/    : GNN policy networks (π_θ) with PyTorch Lightning modules and metrics tracking

The system uses PyTorch Lightning to streamline the training pipeline, reducing
boilerplate while maintaining flexibility. Configuration is managed through
Hydra-zen with Pydantic validation.
"""
