"""
Loss functions for imitation learning.

This module provides loss functions used in training policies
via behavioral cloning.
"""
from __future__         import annotations
from torch.nn           import functional, Module
from torchrl.data       import TensorDict
from torchrl.objectives import LossModule
from typing             import Any


class ImitationLoss(LossModule):
    """
    Imitation learning loss for behavioral cloning.
    
    Computes the mean squared error between the policy's predicted actions
    and the expert's demonstrated actions.
    """
    
    def __init__(self, policy_network: Module):
        """
        Initialize the imitation loss module.
        
        Args:
            policy_network: The policy network being trained.
        """
        super().__init__()
        self.policy_network = policy_network
        
    def forward(self, tensordict: TensorDict) -> dict[str, Any]:
        """
        Compute the imitation loss.
        
        Args:
            tensordict: Batch of experiences containing observations and expert actions.
            
        Returns:
            Dictionary containing the computed loss.
        """
        predicted_actions = self.policy_network(tensordict)["action"]
        expert_actions    = tensordict["action"]
        loss              = functional.mse_loss(predicted_actions, expert_actions)
        
        return {
            "loss"           : loss,
            "loss_imitation" : loss,
        }
