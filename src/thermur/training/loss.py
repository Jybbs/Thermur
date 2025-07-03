"""
Loss functions for imitation learning.

This module provides loss functions used in training policies via behavioral 
cloning. The primary loss function implemented here is the ImitationLoss, which
measures how well a policy network mimics expert demonstrations by computing
the mean squared error between predicted and expert actions.

The loss functions are designed to work with TorchRL's training infrastructure,
returning dictionaries that contain both the total loss for backpropagation and
individual loss components for monitoring and logging.
"""
from tensordict         import TensorDict
from torch.nn           import Module, functional
from torchrl.objectives import LossModule


class ImitationLoss(LossModule):
    """
    Imitation learning loss for behavioral cloning.
    
    This loss module implements the standard behavioral cloning objective, which
    trains a policy network to mimic expert demonstrations by minimizing the
    mean squared error between the policy's predicted actions and the expert's
    demonstrated actions.
    
    The loss function computes:
        L = MSE(π_θ(s), a*)
    
    Where:
        - π_θ(s) is the policy network's predicted action given state s
        - a* is the expert's demonstrated action
        - MSE is the mean squared error function
    
    This approach is suitable for continuous action spaces and assumes that
    the expert demonstrations provide a good coverage of the state space.
    """
    
    def __init__(self, policy_network: Module):
        """
        Initialize the imitation loss module.
        
        Sets up the loss function with a reference to the policy network that
        will be trained. The policy network should accept TensorDict inputs
        and return a TensorDict containing at least an "action" key.
        
        Args:
            policy_network: The policy network to be trained. Expected to be a
                            torch.nn.Module that processes TensorDict inputs and
                            outputs action predictions in a TensorDict format.
        """
        super().__init__()
        self.policy_network = policy_network
        
    def forward(self, td: TensorDict) -> TensorDict:
        """
        Compute the imitation loss for a batch of experiences.
        
        Processes a batch of state-action pairs from expert demonstrations,
        generates action predictions using the policy network, and computes
        the mean squared error between predicted and expert actions.
        
        The method returns both a total loss (for gradient computation) and
        component losses (for logging). For pure behavioral cloning, these 
        values are identical since there's only one loss component.
        
        Args:
            td: A TensorDict containing a batch of experiences with at least:
                - Observation data (format depends on policy network)
                - "action": Expert demonstrated actions [batch_size, action_dim]
                
        Returns:
            A TensorDict containing:
                - loss           : Total loss value for backpropagation
                - loss_imitation : Imitation loss component
        """
        imitation_loss = functional.mse_loss(
            input  = self.policy_network(td)["action"], 
            target = td["action"]
        )
        
        return {
            "loss"           : imitation_loss,
            "loss_imitation" : imitation_loss,
        }