"""
Orchestrates the imitation learning training process for the Thermur project.

This module provides the `TrainingOrchestrator` class, which encapsulates the
entire training pipeline using the `torchrl` library. It is responsible for
executing the main training loop with pre-instantiated components.

The orchestration follows a standard imitation learning (Behavioral Cloning)
paradigm built on a `torchrl` foundation:

1.  An 'expert' controller, defined in `physics.potentials`, generates optimal
    actions within the simulation environment.

2.  A `torchrl.collectors.SyncDataCollector` gathers state-action pairs
    (s, 𝐮_expert) by executing the expert policy in the environment.

3.  These experiences are stored in a `torchrl.data.TensorDictReplayBuffer`.

4.  The GNN policy, π_θ, is trained via supervised learning to minimize the
    Mean Squared Error between its predicted action and the expert's action:
    L(θ) = E[(π_θ(s) - 𝐮_expert)²].

5.  Training progress, including loss and environment metrics, is logged to
    the console and to Weights & Biases, with periodic model checkpointing.
"""
from __future__                  import annotations
from src.configs.pydantic            import TrainConfig, WandbConfig
from src.core.structures             import SwarmDataSpec
from src.envs.thermur                import ThermurEnv
from hydra_zen                     import zen
from hydra_zen.third_party.pydantic import pydantic_parser
from src.models.gnn_policy           import GNNPolicy
from src.ops.loguru                  import logger
from tensordict.tensordict       import TensorDictBase
from torch.optim                 import Optimizer
from torchrl.collectors          import SyncDataCollector
from torchrl.data                import TensorDictReplayBuffer
from torchrl.envs                import EnvBase
from torchrl.modules             import SafeModule
from torchrl.objectives          import LossModule
from tqdm                        import tqdm

import hydra
import os
import torch
import torch.nn as nn
import wandb


class ImitationLoss(LossModule):
    """
    A `torchrl` loss module for supervised imitation learning.

    This class computes the Mean Squared Error (MSE) between the actions
    predicted by the learned GNN policy and the ground-truth actions generated
    by the expert controller. It is designed to be directly used within the
    `torchrl` training framework.
    """
    def __init__(self, policy_network: nn.Module):
        """
        Initializes the ImitationLoss module.

        Args:
            policy_network: The GNN policy network module whose output will be
                            compared against the expert's actions.
        """
        super().__init__()
        self.policy_network = policy_network

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        """
        Computes the imitation loss for a batch of experience.

        The method performs the following steps:
        1.  Converts the `TensorDict` observation into a batch of
            `torch_geometric.data.Data` objects suitable for the GNN.
        2.  Passes the graph data through the policy network to get predicted actions.
        3.  Retrieves the corresponding expert actions from the `TensorDict`.
        4.  Calculates the MSE between the predicted and expert actions.

        Args:
            tensordict: A `TensorDict` containing a batch of training data from
                        the replay buffer. It must contain the observation keys
                        and the `action_expert` key.

        Returns:
            A new `TensorDict` containing a single key, "loss", with the
            computed scalar loss value for the batch.
        """
        pyg_data         = SwarmDataSpec.to_torch_geometric(tensordict)
        predicted_action = self.policy_network(pyg_data)
        expert_action    = tensordict.get("action_expert")
        loss             = nn.functional.mse_loss(predicted_action, expert_action)

        return TensorDictBase(
            source     = {"loss": loss},
            batch_size = tensordict.batch_size
        )


class TrainingOrchestrator:
    """
    Manages the execution of the imitation learning pipeline.

    This class serves as the main driver for a training run. It receives
    pre-instantiated components from Hydra and executes the main training loop,
    orchestrating data collection, model optimization, progress logging, and
    checkpointing.
    """

    def __init__(
        self,
        env           : EnvBase,
        expert_policy : SafeModule,
        policy        : GNNPolicy,
        collector     : SyncDataCollector,
        replay_buffer : TensorDictReplayBuffer,
        loss_module   : ImitationLoss,
        optimizer     : Optimizer,
        train_config  : TrainConfig,
        wandb_config  : WandbConfig,
    ):
        """
        Initializes the orchestrator with pre-instantiated components.

        Args:
            env           : The instantiated environment.
            expert_policy : The expert policy SafeModule.
            policy        : The GNN policy to be trained.
            collector     : The data collector.
            replay_buffer : The replay buffer for storing experiences.
            loss_module   : The imitation loss module.
            optimizer     : The optimizer for training.
            train_config  : Training configuration parameters.
            wandb_config  : Weights & Biases configuration.
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
        
        self.device = torch.device(train_config.device)

        self._setup_logging()
        logger.info("Training orchestrator initialized successfully.")

    def _setup_logging(self):
        """
        Initializes the Weights & Biases logger for experiment tracking.

        This method configures the `wandb` library based on the `WandbConfig`
        settings. If the mode is not 'disabled', it will start a new run,
        logging the entire application configuration for reproducibility.
        """
        if self.wandb_config.mode != "disabled":
            wandb.init(
                project = self.wandb_config.project,
                entity  = self.wandb_config.entity,
                mode    = self.wandb_config.mode,
                config  = {
                    "train": self.train_config.model_dump(),
                    "wandb": self.wandb_config.model_dump(),
                }
            )
            logger.info("Weights & Biases initialized for experiment tracking.")

    def run(self):
        """
        Executes the main training loop.

        This method iterates through the data collector, which yields batches of
        experience from the environment. Each batch is added to the replay buffer.
        The agent then samples from the buffer, computes the imitation loss,
        and performs a gradient-based optimization step. Progress is logged
        periodically to the console and `wandb`, and checkpoints are saved at
        regular intervals.
        """
        logger.info(f"Starting training for {self.train_config.collector.total_frames} frames.")
        pbar = tqdm(total=self.train_config.collector.total_frames)

        total_frames = 0
        for i, data in enumerate(self.collector):

            self.replay_buffer.extend(data.to("cpu"))
            current_frames = data.numel()
            total_frames  += current_frames

            pbar.update(current_frames)

            if total_frames > self.train_config.replay.batch_size:
                batch     = self.replay_buffer.sample().to(self.device)
                loss_dict = self.loss_module(batch)
                loss      = loss_dict["loss"]
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                if i % self.train_config.log_interval == 0:
                    if self.wandb_config.mode != "disabled":
                        wandb.log({"train/loss": loss.item()}, step=total_frames)
                    pbar.set_description(f"Loss: {loss.item():.4f}")

                if i % self.train_config.checkpoint.interval == 0 and i > 0:
                    self._save_checkpoint(total_frames)

        self.collector.shutdown()
        pbar.close()
        self._save_checkpoint(total_frames, is_final=True)
        logger.info("Training finished successfully.")

    def _save_checkpoint(
        self, 
        frame_count : int, 
        is_final    : bool = False
    ):
        """
        Saves a checkpoint of the model and optimizer states.

        Args:
            frame_count : The current number of training frames, used for naming.
            is_final    : If True, saves the checkpoint with a 'final' suffix.
        """
        path = self.train_config.checkpoint.path
        os.makedirs(path, exist_ok=True)
        
        name      = f"final.pt" if is_final else f"checkpoint_{frame_count}.pt"
        save_path = os.path.join(path, name)
        
        torch.save(
            {
                'frame'               : frame_count,
                'model_state_dict'    : self.policy.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss'                : self.loss_module(self.replay_buffer.sample().to(self.device))["loss"],
            }, 
            save_path)
        logger.info(f"Checkpoint saved to {save_path}")


# --------------------------------------------------------------------------
# Hydra Entry Point
# --------------------------------------------------------------------------

if __name__ == "__main__":
    # Register configurations
    from src.configs import register_configs
    register_configs()
    
    @hydra.main(config_path=None, config_name="train", version_base=None)
    def main(cfg):
        """
        Main entry point for the training script.

        This function is decorated with @hydra.main, which enables Hydra to manage
        the configuration and instantiation of all components. The orchestrator
        and all its dependencies are built automatically based on the configuration.

        Args:
            cfg: The Hydra configuration object.
        """
        # Hydra will instantiate the entire object graph
        orchestrator: TrainingOrchestrator = hydra.utils.instantiate(
            cfg.orchestrator,
            _target_wrapper_ = pydantic_parser,
        )
        
        # Run the training
        orchestrator.run()
    
    # Run main
    main()
