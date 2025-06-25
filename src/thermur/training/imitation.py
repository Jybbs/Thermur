"""
Imitation learning training utilities.

This module provides clean, focused functions for training policies
via behavioral cloning, without the overhead of an orchestrator class.
It includes functions for managing the training loop, collecting expert 
demonstrations, and handling model checkpoints.
"""
import os
import torch
import wandb

from loguru             import logger
from pydantic           import BaseModel
from torch.nn           import Module
from torch.optim        import Optimizer
from torchrl.collectors import SyncDataCollector
from torchrl.data       import TensorDictReplayBuffer
from torchrl.envs       import EnvBase
from torchrl.modules    import SafeModule
from torchrl.objectives import LossModule
from tqdm               import tqdm
from typing             import Any, Optional


def cleanup_resources(
    data_collector : SyncDataCollector,
    visualizer     : Optional[Any],
    pbar           : tqdm
) -> None:
    """
    Clean up resources used during training.
    
    Properly shuts down the data collector, closes the progress bar,
    and closes the visualizer if it exists.
    
    Args:
        data_collector : The experience collector to shut down
        visualizer     : The visualization module instance or None
        pbar           : The progress bar to close
    """
    data_collector.shutdown()
    pbar.close()
    
    if visualizer is not None:
        visualizer.close()


def initialize_wandb(
    hyperparameters : BaseModel,
    wandb_config    : BaseModel
) -> None:
    """
    Initialize Weights & Biases for experiment tracking.
    
    Sets up the W&B experiment with the appropriate project, entity, and
    configuration settings. Only initializes if W&B is enabled in the config.
    
    Args:
        hyperparameters : Training hyperparameters model
        wandb_config    : W&B configuration model
    """
    if wandb_config.mode != "disabled":
        wandb.init(
            config  = {
                "hyperparameters": hyperparameters.__dict__,
                "wandb"          : wandb_config.__dict__,
            },
            entity  = wandb_config.entity,
            mode    = wandb_config.mode,
            project = wandb_config.project,
        )
        logger.info("Weights & Biases initialized for experiment tracking.")


def save_checkpoint(
    policy      : Module,
    optimizer   : Optimizer, 
    frame_count : int,
    save_path   : str,
    is_final    : bool = False
) -> None:
    """
    Save a model checkpoint.

    Creates a checkpoint containing the policy model weights, optimizer state,
    and current frame count. The checkpoint can be used to resume training or
    for model evaluation. The function creates the save directory if it doesn't
    exist.

    Args:
        policy      : The policy network to save
        optimizer   : The optimizer state to save
        frame_count : Current training frame count
        save_path   : Directory to save checkpoints
        is_final    : Whether this is the final checkpoint
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


def train_imitation_learning(
    environment       : EnvBase,
    expert_policy     : SafeModule,
    policy            : Module,
    data_collector    : SyncDataCollector,
    experience_buffer : TensorDictReplayBuffer,
    loss_function     : LossModule,
    optimizer         : Optimizer,
    hyperparameters   : BaseModel,
    wandb_config      : BaseModel,
    visualizer        : Optional[Any] = None,
) -> None:
    """
    Train a policy via imitation learning (behavioral cloning).

    This function implements the training loop for imitation learning,
    collecting expert demonstrations and training the policy to match
    the expert's actions. It manages the data collection, optimization,
    visualization, and logging processes.

    Args:
        environment       : The simulation environment
        expert_policy     : The expert policy generating demonstrations
        policy            : The GNN policy to be trained
        data_collector    : Collects experiences from the environment
        experience_buffer : Stores collected experiences
        loss_function     : Computes the imitation loss
        optimizer         : Updates the policy parameters
        hyperparameters   : Training hyperparameters
        wandb_config      : Weights & Biases configuration
        visualizer        : Optional visualization module (None to disable)
    """
    device = torch.device(hyperparameters.device)
    
    # Initialize experiment tracking
    initialize_wandb(hyperparameters, wandb_config)
    
    logger.info(f"Starting training for {data_collector.total_frames} frames.")
    pbar = tqdm(total=data_collector.total_frames)
    
    total_frames = 0
    for i, data in enumerate(data_collector):
        experience_buffer.extend(data.to("cpu"))
        current_frames = data.numel()
        total_frames  += current_frames
        
        # Update visualization if enabled
        if visualizer is not None:
            latest_observation = data[-1].get("next")
            update_visualization(visualizer, latest_observation)
        
        pbar.update(current_frames)
        
        if total_frames > experience_buffer.batch_size:
            batch     = experience_buffer.sample().to(device)
            loss_dict = loss_function(batch)
            loss      = loss_dict["loss"]
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if i % hyperparameters.log_interval == 0:
                if wandb_config.mode != "disabled":
                    wandb.log({"train/loss": loss.item()}, step=total_frames)
                pbar.set_description(f"Loss: {loss.item():.4f}")
    
    # Clean up resources
    cleanup_resources(data_collector, visualizer, pbar)
    
    # Save final checkpoint
    save_checkpoint(policy, optimizer, total_frames, "checkpoints", is_final=True)
    logger.info("Training finished successfully.")


def update_visualization(
    visualizer         : Optional[Any],
    latest_observation : dict[str, Any]
) -> None:
    """
    Update the visualization with the latest observation.
    
    Updates the visualizer's state with the current simulation data
    and renders the updated visualization. Only performs updates if
    a visualizer is provided.
    
    Args:
        visualizer         : The visualization module instance or None
        latest_observation : The most recent observation from the environment
    """
    if visualizer is not None:
        visualizer.update(latest_observation)
        visualizer.render()
