"""
PyTorch Lightning components for imitation learning.

This package provides the core Lightning infrastructure for training GNN policies
through behavioral cloning. It leverages PyTorch Lightning's abstractions to
eliminate training loop boilerplate while maintaining flexibility.

The module includes:
- ExperienceModule: A LightningDataModule that manages trajectory collection
  from expert demonstrations and maintains a replay buffer for efficient
  batch sampling during training
- GNNPolicy: A LightningModule implementing the graph neural network that
  learns decentralized control policies from expert demonstrations
- Monitoring callbacks: Lightning callbacks for metrics collection and event
  logging that integrate seamlessly with the training loop
- Training utilities: High-level functions that coordinate Lightning's Trainer
  with the policy and data modules

Lightning handles many concerns automatically:
- Device placement and distributed training
- Mixed precision training and gradient scaling
- Logging integration with Weights & Biases
- Automatic checkpointing and early stopping
- Progress bars and training metrics

This design allows researchers to focus on the algorithmic aspects of
imitation learning while Lightning manages the engineering complexity.
"""
from .callback   import *
from .experience import *
from .policy     import *
