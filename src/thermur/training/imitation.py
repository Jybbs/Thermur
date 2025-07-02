"""
Imitation learning training loop.

This module implements behavioral cloning for training a GNN policy
to mimic expert flocking behavior while respecting thermal constraints.
"""
from ..visualization    import Visualizer
from configs.imitation  import LearningModel, WandbModel
from loguru             import logger
from pathlib            import Path
from torch.nn           import Module
from torch.optim        import Optimizer
from torchrl.collectors import SyncDataCollector
from torchrl.data       import TensorDictReplayBuffer
from torchrl.objectives import LossModule
from tqdm               import tqdm

import torch
import wandb as wb


def cleanup_resources(
    data_collector : SyncDataCollector,
    pbar           : tqdm,
    visualizer     : Visualizer | None
):
    """
    Clean up resources used during training.
    
    Properly shuts down the data collector, closes the progress bar,
    and closes the visualizer if it exists.
    
    Args:
        data_collector : The experience collector to shut down
        pbar           : The progress bar to close
        visualizer     : The visualization module instance or None
    """
    data_collector.shutdown()
    pbar.close()
    
    if visualizer is not None:
        visualizer.close()


def initialize_wandb(
    learning : LearningModel,
    wandb    : WandbModel
):
    """
    Initialize Weights & Biases for experiment tracking.
    
    Sets up the W&B experiment with the appropriate project, entity, and
    configuration settings. Only initializes if W&B is enabled in the config.
    
    Args:
        learning : Learning hyperparameters and settings
        wandb    : W&B configuration model
    """
    if wandb.mode != "disabled":
        wb.init(
            config  = {
                "learning" : learning.model_dump(),
                "wandb"    : wandb.model_dump(),
            },
            entity  = wandb.entity,
            mode    = wandb.mode,
            project = wandb.project
        )
        logger.info("Weights & Biases initialized for experiment tracking.")


def save_checkpoint(
    frame_count : int,
    optimizer   : Optimizer,
    policy      : Module,
    save_path   : str,
    is_final    : bool = False
):
    """
    Save a model checkpoint.

    Creates a checkpoint containing the policy model weights, optimizer state,
    and current frame count. The checkpoint can be used to resume training or
    for model evaluation. The function creates the save directory if it doesn't
    exist.

    Args:
        frame_count : Current training frame count
        optimizer   : The optimizer state to save
        policy      : The policy network to save
        save_path   : Directory to save checkpoints
        is_final    : Whether this is the final checkpoint
    """
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    filename  = "final.pt" if is_final else f"checkpoint_{frame_count}.pt"
    full_path = save_dir / filename
    
    torch.save(
        f   = full_path,
        obj = {
            'frame'                : frame_count,
            'model_state_dict'     : policy.state_dict(),
            'optimizer_state_dict' : optimizer.state_dict(),
        }
    )
    logger.info(f"Checkpoint saved to {full_path}")


def train_imitation_learning(
    data_collector    : SyncDataCollector,
    experience_buffer : TensorDictReplayBuffer,
    learning          : LearningModel,
    loss              : LossModule,
    optimizer         : Optimizer,
    policy            : Module,
    wandb             : WandbModel,
    visualizer        : Visualizer | None = None,
):
    """
    Train a policy via imitation learning (behavioral cloning).
    
    Implements the training loop that collects expert demonstrations and
    trains the GNN policy to minimize:
    
        L_imitation = 𝔼_𝒟[||π_θ(s) - π*(s)||²]
    
    where π_θ is the learned policy and π* is the expert controller.
    
    Args:
        data_collector    : Manages environment interaction loop
        experience_buffer : Stores and samples demonstration data
        learning          : Training hyperparameters and settings
        loss              : Behavioral cloning loss module
        optimizer         : Gradient-based optimizer
        policy            : GNN policy network to train
        wandb             : Experiment tracking configuration
        visualizer        : Optional 3D visualization module
    """
    device = torch.device(learning.device)
    initialize_wandb(learning, wandb)
    
    logger.info(f"Starting training for {learning.total_frames} frames.")
    pbar = tqdm(total=learning.total_frames)
    
    total_frames = 0
    for i, data in enumerate(data_collector):
        experience_buffer.extend(data.to("cpu"))
        current_frames = data.numel()
        total_frames  += current_frames
        
        if visualizer is not None:
            latest_observation = data[-1].get("next")
            update_visualization(
                latest_observation = latest_observation,
                visualizer         = visualizer
            )
        
        pbar.update(current_frames)
        
        if total_frames > experience_buffer.batch_size:
            batch     = experience_buffer.sample().to(device)
            loss_dict = loss(batch)
            loss      = loss_dict["loss"]
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if i % learning.log_interval == 0:
                if wandb.mode != "disabled":
                    wb.log({"train/loss": loss.item()}, step=total_frames)
                pbar.set_description(f"Loss: {loss.item():.4f}")
        
        if total_frames % learning.checkpoint_interval == 0:
            save_checkpoint(
                frame_count = total_frames,
                optimizer   = optimizer,
                policy      = policy,
                save_path   = learning.checkpoint_path
            )
    
    cleanup_resources(
        data_collector = data_collector,
        pbar           = pbar,
        visualizer     = visualizer
    )
    
    save_checkpoint(
        frame_count = total_frames,
        optimizer   = optimizer,
        policy      = policy,
        save_path   = learning.checkpoint_path,
        is_final    = True
    )
    logger.info("Training finished successfully.")


def update_visualization(
    latest_observation : dict[str, any],
    visualizer         : Visualizer
):
    """
    Update the visualization with the latest observation.
    
    Updates the visualizer's state with the current simulation data
    and renders the updated visualization. Only performs updates if
    a visualizer is provided.
    
    Args:
        latest_observation : The most recent observation from the environment
        visualizer         : The visualization module instance
    """
    if visualizer is not None:
        visualizer.update(latest_observation)
        visualizer.render()