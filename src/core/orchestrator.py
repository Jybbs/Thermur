"""
Training orchestrator for the Thermur project.

This module provides the TrainingOrchestrator class that manages the entire
imitation learning pipeline. It receives fully instantiated components from
Hydra and coordinates the training loop.
"""
from __future__ import annotations

import torch
import wandb
from loguru                 import logger
from src.configs            import TrainConfig, WandbConfig
from src.core.env           import ThermurEnv
from src.models.gnn_policy  import GNNPolicy
from src.ops.loguru         import configure_loguru
from src.ops.seed           import set_seed
from src.physics.potentials import ExpertFlockingController
from torch.nn               import Module
from torch.optim            import Optimizer
from torchrl.collectors     import SyncDataCollector
from torchrl.data           import TensorDictReplayBuffer
from typing                 import TYPE_CHECKING

if TYPE_CHECKING:
    from tensordict import TensorDict


class ImitationLoss(Module):
    """
    Computes the imitation learning loss for behavioral cloning.
    
    This loss module wraps the GNN policy and computes the MSE between
    predicted actions and expert demonstrations.
    """
    
    def __init__(self, policy_network: GNNPolicy):
        """
        Initialize the loss module.
        
        Args:
            policy_network: The GNN policy to train.
        """
        super().__init__()
        self.policy_network = policy_network
    
    def forward(self, batch: TensorDict) -> torch.Tensor:
        """
        Compute the imitation loss on a batch of data.
        
        Args:
            batch: A TensorDict containing observations and expert actions.
            
        Returns:
            The MSE loss between predicted and expert actions.
        """
        # Extract observation data and convert to graph format
        obs_data = self._tensordict_to_graph(batch)
        
        # Get policy predictions
        predicted_actions = self.policy_network(obs_data)
        
        # Extract expert actions from batch
        expert_actions = batch["action"]
        
        # Compute MSE loss
        loss = torch.nn.functional.mse_loss(predicted_actions, expert_actions)
        
        return loss
    
    def _tensordict_to_graph(self, td: TensorDict):
        """Convert TensorDict observation to torch_geometric Data."""
        # This would need proper implementation based on SwarmData structure
        raise NotImplementedError("Graph conversion logic to be implemented")


class TrainingOrchestrator:
    """
    Orchestrates the entire training pipeline with fully instantiated components.
    
    This class receives all components as instantiated objects from Hydra,
    eliminating the need for manual instantiation based on configs.
    """
    
    def __init__(
        self,
        env           : ThermurEnv,
        expert_policy : ExpertFlockingController,
        policy        : GNNPolicy,
        collector     : SyncDataCollector,
        replay_buffer : TensorDictReplayBuffer,
        loss_module   : ImitationLoss,
        optimizer     : Optimizer,
        train_config  : TrainConfig,
        wandb_config  : WandbConfig,
    ):
        """
        Initialize the orchestrator with fully instantiated components.
        
        All components are built by Hydra based on the declarative configuration,
        demonstrating the power of hydra-zen's approach.
        
        Args:
            env          : The instantiated Thermur environment.
            expert_policy: The instantiated expert controller.
            policy       : The instantiated GNN policy to train.
            collector    : The instantiated data collector.
            replay_buffer: The instantiated replay buffer.
            loss_module  : The instantiated loss computation module.
            optimizer    : The instantiated optimizer.
            train_config : Training configuration parameters.
            wandb_config : Weights & Biases configuration.
        """
        self.env           = env
        self.expert_policy = expert_policy
        self.policy        = policy
        self.collector     = collector
        self.replay_buffer = replay_buffer
        self.loss_module   = loss_module
        self.optimizer     = optimizer
        self.train_config  = train_config
        self.wandb_config  = wandb_config
        
        # Setup logging and tracking
        self._setup()
    
    def _setup(self):
        """Initialize logging, random seeds, and experiment tracking."""
        # Configure logging
        configure_loguru(self.train_config.logging_config)
        
        # Set random seed
        set_seed(self.train_config.seed)
        
        # Initialize W&B if enabled
        if self.wandb_config.mode != "disabled":
            wandb.init(
                project = self.wandb_config.project,
                entity  = self.wandb_config.entity,
                mode    = self.wandb_config.mode,
                config  = {
                    "train_config" : self.train_config.dict(),
                    "wandb_config" : self.wandb_config.dict(),
                }
            )
    
    def run(self):
        """
        Execute the main training loop.
        
        This method coordinates data collection, training, checkpointing,
        and logging throughout the training process.
        """
        logger.info("Starting training orchestration")
        
        # Main training loop
        total_frames = 0
        epoch = 0
        
        while total_frames < self.train_config.collector.total_frames:
            # Collect data using expert policy
            logger.info(f"Collecting data for epoch {epoch}")
            batch = self.collector.next()
            
            # Add to replay buffer
            self.replay_buffer.extend(batch)
            
            # Training iterations
            for _ in range(self.train_config.collector.frames_per_batch):
                # Sample from replay buffer
                train_batch = self.replay_buffer.sample()
                
                # Compute loss
                loss = self.loss_module(train_batch)
                
                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Logging
                if total_frames % self.train_config.log_interval == 0:
                    logger.info(f"Frame {total_frames}: Loss = {loss.item():.4f}")
                    
                    if self.wandb_config.mode != "disabled":
                        wandb.log({
                            "loss"        : loss.item(),
                            "total_frames": total_frames,
                            "epoch"       : epoch,
                        })
                
                total_frames += 1
                
                # Checkpointing
                if total_frames % self.train_config.checkpoint.interval == 0:
                    self._save_checkpoint(total_frames)
            
            epoch += 1
        
        logger.info("Training completed")
        
        if self.wandb_config.mode != "disabled":
            wandb.finish()
    
    def _save_checkpoint(self, frame_count: int):
        """Save a model checkpoint."""
        checkpoint_path = (
            f"{self.train_config.checkpoint.path}/model_{frame_count}.pt"
        )
        
        torch.save({
            "frame_count"  : frame_count,
            "policy_state" : self.policy.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
        }, checkpoint_path)
        
        logger.info(f"Saved checkpoint to {checkpoint_path}")
