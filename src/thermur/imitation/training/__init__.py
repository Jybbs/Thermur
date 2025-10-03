"""
Training components for imitation learning.

This package provides the complete training infrastructure for GNN policies
through behavioral cloning, including PyTorch Lightning modules and metrics.

The module includes:

callbacks:
- CallbackFactory: Factory for creating customized Lightning callbacks with
  Thermur's visual styling for progress bars and model summaries

metrics:
- MetricsFactory: Factory for creating training and validation metric collections
- Various metric classes for measuring scale-free correlations, information
  propagation, cohesion, energy consumption, and other emergent properties

policy:
- GNNPolicy: A LightningModule implementing the graph neural network that
  learns decentralized control policies from expert demonstrations

The infrastructure handles many concerns automatically:
- Device placement and distributed training
- Mixed precision training and gradient scaling
- Logging integration with Weights & Biases
- Automatic checkpointing and early stopping
- Progress bars and training metrics with thermal styling

This design allows researchers to focus on the algorithmic aspects of
imitation learning while the framework manages engineering complexity.
"""
from .callbacks import *
from .metrics   import *
from .policy    import *
