"""
Training algorithms for imitation learning.

This package implements the behavioral cloning pipeline for training the GNN
policy from expert demonstrations. The training process involves:

1. Collecting trajectories using the expert controller
2. Storing experiences in a replay buffer for sampling
3. Minimizing the MSE loss between policy and expert actions
4. Periodic evaluation and checkpointing

The loss function includes optional regularization terms and can be extended
with auxiliary objectives for improved learning stability.
"""
from .train import save_checkpoint, train_imitation_learning
from .loss      import ImitationLoss
from .policy    import GNNPolicy