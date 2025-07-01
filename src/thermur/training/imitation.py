"""
Imitation learning training loop.

This module implements behavioral cloning for training a GNN policy
to mimic expert flocking behavior while respecting thermal constraints.
"""
from ..visualization.visualizer import Visualizer
from configs                    import LearningModel, WandbModel
from loguru                     import logger
from pathlib                    import Path
from torch.nn                   import Module
from torch.optim                import Optimizer
from torchrl.collectors         import SyncDataCollector
from torchrl.data               import TensorDictReplayBuffer
from torchrl.envs               import EnvBase
from torchrl.modules            import SafeModule
from torchrl.objectives         import LossModule
from tqdm                       import tqdm

import torch
import wandb


def cleanup_resources(
    data_collector : SyncDataCollector,
    visualizer     : Visualizer | None,
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
    learning     : LearningModel,
    wandb_config : WandbModel
) -> None:
    """
    Initialize Weights & Biases for experiment tracking.
    
    Sets up the W&B experiment with the appropriate project, entity, and
    configuration settings. Only initializes if W&B is enabled in the config.
    
    Args:
        learning     : Learning hyperparameters and settings
        wandb_config : W&B configuration model
    """
    if wandb_config.mode != "disabled":
        wandb.init(
            project = wandb_config.project,
            entity  = wandb_config.entity,
            mode    = wandb_config.mode,
            config  = {
                "learning" : learning.model_dump(),
                "wandb"    : wandb_config.model_dump(),
            }
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
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    filename  = "final.pt" if is_final else f"checkpoint_{frame_count}.pt"
    full_path = save_dir / filename
    
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
    learning          : LearningModel,
    wandb_config      : WandbModel,
    visualizer        : Visualizer | None = None,
) -> None:
    """
    Train a policy via imitation learning (behavioral cloning).
    
    Implements the training loop that collects expert demonstrations and
    trains the GNN policy to minimize:
    
        L_imitation = 𝔼_𝒟[||π_θ(s) - π*(s)||²]
    
    where π_θ is the learned policy and π* is the expert policy.
    
    Args:
        environment       : Simulation environment for data collection
        expert_policy     : Expert controller providing demonstrations
        policy            : GNN policy network to train
        data_collector    : Manages environment interaction loop
        experience_buffer : Stores transitions for replay
        loss_function     : Computes imitation loss
        optimizer         : Updates policy parameters
        learning          : Learning hyperparameters and settings
        wandb_config      : Experiment tracking configuration
        visualizer        : Optional 3D visualization module
    """
    device = torch.device(learning.device)
    
    # Initialize experiment tracking
    initialize_wandb(learning, wandb_config)
    
    logger.info(f"Starting training for {learning.total_frames} frames.")
    pbar = tqdm(total=learning.total_frames)
    
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
            
            if i % learning.log_interval == 0:
                if wandb_config.mode != "disabled":
                    wandb.log({"train/loss": loss.item()}, step=total_frames)
                pbar.set_description(f"Loss: {loss.item():.4f}")
        
        # Save checkpoint if needed
        if total_frames % learning.checkpoint_interval == 0:
            save_checkpoint(
                policy, 
                optimizer, 
                total_frames, 
                learning.checkpoint_path
            )
    
    # Clean up resources
    cleanup_resources(data_collector, visualizer, pbar)
    
    # Save final checkpoint
    save_checkpoint(
        policy, 
        optimizer, 
        total_frames, 
        learning.checkpoint_path, 
        is_final=True
    )
    logger.info("Training finished successfully.")


def update_visualization(
    visualizer         : Visualizer,
    latest_observation : dict[str, any]
) -> None:
    """
    Update the visualization with the latest observation.
    
    Updates the visualizer's state with the current simulation data
    and renders the updated visualization. Only performs updates if
    a visualizer is provided.
    
    Args:
        visualizer         : The visualization module instance
        latest_observation : The most recent observation from the environment
    """
    if visualizer is not None:
        visualizer.update(latest_observation)
        visualizer.render()