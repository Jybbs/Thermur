"""
Loss functions for imitation learning.

This module provides loss functions used in training policies
via behavioral cloning.
"""
from tensordict         import TensorDict
from torch.nn           import functional, Module
from torchrl.objectives import LossModule


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
        
    def forward(self, tensordict: TensorDict) -> TensorDict:
        """
        Compute the imitation loss.
        
        Args:
            tensordict: Batch of experiences containing observations and expert actions.
            
        Returns:
            Dictionary containing the computed loss.
        """
        expert_actions    = tensordict["action"]
        predicted_actions = self.policy_network(tensordict)["action"]
        loss              = functional.mse_loss(predicted_actions, expert_actions)
        
        return {
            "loss"           : loss,
            "loss_imitation" : loss,
        }
