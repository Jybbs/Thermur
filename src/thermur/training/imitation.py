"""
Imitation learning training utilities.

This module provides clean, focused functions for training policies
via behavioral cloning, without the overhead of an orchestrator class.
"""
import os
import torch
import wandb

from loguru             import logger
from torch.nn           import Module
from torch.optim        import Optimizer
from torchrl.collectors import SyncDataCollector
from torchrl.data       import TensorDictReplayBuffer
from torchrl.envs       import EnvBase
from torchrl.modules    import SafeModule
from torchrl.objectives import LossModule
from tqdm               import tqdm


def train_imitation_learning(
    environment       : EnvBase,
    expert_policy     : SafeModule,
    policy            : Module,
    data_collector    : SyncDataCollector,
    experience_buffer : TensorDictReplayBuffer,
    loss_function     : LossModule,
    optimizer         : Optimizer,
    hyperparameters,
    wandb_config,
):
    """
    Train a policy via imitation learning (behavioral cloning).

    This function implements the training loop for imitation learning,
    collecting expert demonstrations and training the policy to match
    the expert's actions.

    Args:
        environment       : The simulation environment.
        expert_policy     : The expert policy generating demonstrations.
        policy            : The GNN policy to be trained.
        data_collector    : Collects experiences from the environment.
        experience_buffer : Stores collected experiences.
        loss_function     : Computes the imitation loss.
        optimizer         : Updates the policy parameters.
        hyperparameters   : Training hyperparameters.
        wandb_config      : Weights & Biases configuration.
    """
    device = torch.device(hyperparameters.device)
    
    # Initialize Weights & Biases if enabled
    if wandb_config.mode != "disabled":
        wandb.init(
            project = wandb_config.project,
            entity  = wandb_config.entity,
            mode    = wandb_config.mode,
            config  = {
                "hyperparameters": hyperparameters.__dict__,
                "wandb"          : wandb_config.__dict__,
            }
        )
        logger.info("Weights & Biases initialized for experiment tracking.")
    
    # Main training loop
    logger.info(f"Starting training for {data_collector.total_frames} frames.")
    pbar = tqdm(total=data_collector.total_frames)
    
    total_frames = 0
    for i, data in enumerate(data_collector):
        # Add collected data to replay buffer
        experience_buffer.extend(data.to("cpu"))
        current_frames = data.numel()
        total_frames  += current_frames
        
        pbar.update(current_frames)
        
        # Train when we have enough data
        if total_frames > experience_buffer.batch_size:
            batch     = experience_buffer.sample().to(device)
            loss_dict = loss_function(batch)
            loss      = loss_dict["loss"]
            
            # Optimization step
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # Logging
            if i % hyperparameters.log_interval == 0:
                if wandb_config.mode != "disabled":
                    wandb.log({"train/loss": loss.item()}, step=total_frames)
                pbar.set_description(f"Loss: {loss.item():.4f}")
    
    data_collector.shutdown()
    pbar.close()
    
    # Save final model
    save_checkpoint(policy, optimizer, total_frames, "checkpoints", is_final=True)
    logger.info("Training finished successfully.")


def save_checkpoint(
    policy      : Module,
    optimizer   : Optimizer, 
    frame_count : int,
    save_path   : str,
    is_final    : bool = False
):
    """
    Save a model checkpoint.

    Args:
        policy      : The policy network to save.
        optimizer   : The optimizer state to save.
        frame_count : Current training frame count.
        save_path   : Directory to save checkpoints.
        is_final    : Whether this is the final checkpoint.
    """
    os.makedirs(save_path, exist_ok=True)
    
    filename  = "final.pt" if is_final else f"checkpoint_{frame_count}.pt"
    full_path = os.path.join(save_path, filename)
    
    torch.save(
        {
            'frame'                : frame_count,
            'model_state_dict'     : policy.state_dict(),
            'optimizer_state_dict' : optimizer.state_dict(),
        }, 
        full_path
    )
    logger.info(f"Checkpoint saved to {full_path}")
