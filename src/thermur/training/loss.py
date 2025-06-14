"""
Loss functions for imitation learning.

This module provides loss functions used in training policies
via behavioral cloning.
"""
import torch
from torch        import nn
from torchrl.objectives import LossModule


class ImitationLoss(LossModule):
    """
    Imitation learning loss for behavioral cloning.
    
    Computes the mean squared error between the policy's predicted actions
    and the expert's demonstrated actions.
    """
    
    def __init__(self, policy_network: nn.Module):
        """
        Initialize the imitation loss module.
        
        Args:
            policy_network: The policy network being trained.
        """
        super().__init__()
        self.policy_network = policy_network
        
    def forward(self, tensordict):
        """
        Compute the imitation loss.
        
        Args:
            tensordict: Batch of experiences containing observations and expert actions.
            
        Returns:
            Dictionary containing the computed loss.
        """
        # Get policy predictions
        policy_out = self.policy_network(tensordict)
        predicted_actions = policy_out["action"]
        
        # Get expert actions
        expert_actions = tensordict["action"]
        
        # Compute MSE loss
        loss = torch.nn.functional.mse_loss(predicted_actions, expert_actions)
        
        return {
            "loss": loss,
            "loss_imitation": loss,  # For logging
        }
