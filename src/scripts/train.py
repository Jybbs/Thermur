# File: src/training/orchestrator.py

"""
Orchestrates the imitation learning training process for the Thermur project.

This module provides the `TrainingOrchestrator` class, which encapsulates the
entire training pipeline using the `torchrl` library. It is responsible for
initializing all necessary components—environment, policies, data collectors,
and replay buffers—and executing the main training loop.

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
from __future__            import annotations
from core.structures       import SwarmDataSpec
from envs.thermur          import ThermurEnv
from models.gnn_policy     import GNNPolicy
from ops.config            import AppConfig
from ops.loguru            import logger
from physics.potentials    import ExpertFlockingController
from tensordict.tensordict import TensorDictBase
from torch.optim           import AdamW
from torchrl.collectors    import SyncDataCollector
from torchrl.data          import TensorDictReplayBuffer
from torchrl.modules       import SafeModule
from torchrl.objectives    import LossModule
from tqdm                  import tqdm

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
    Manages the setup and execution of the imitation learning pipeline.

    This class serves as the main driver for a training run. It handles the
    initialization of all components based on a single configuration object,
    and then executes the main training loop, orchestrating data collection,
    model optimization, progress logging, and checkpointing.
    """

    def __init__(self, config: AppConfig):
        """
        Initializes the orchestrator and all its components.

        Args:
            config: The root `AppConfig` object, serving as the single source
                    of truth for all hyperparameters.
        """
        self.config = config
        self.device = torch.device(config.train.device)

        self._setup_logging()
        self._setup_environment()
        self._setup_policies()
        self._setup_data_pipeline()
        self._setup_optimizer()

        logger.info("Training orchestrator initialized successfully.")

    def _setup_logging(self):
        """
        Initializes the Weights & Biases logger for experiment tracking.

        This method configures the `wandb` library based on the `WandbConfig`
        settings. If the mode is not 'disabled', it will start a new run,
        logging the entire application configuration for reproducibility.
        """
        if self.config.wandb.mode != "disabled":
            wandb.init(
                project = self.config.wandb.project,
                entity  = self.config.wandb.entity,
                mode    = self.config.wandb.mode,
                config  = self.config.model_dump()
            )
            logger.info("Weights & Biases initialized for experiment tracking.")

    def _setup_environment(self):
        """
        Initializes the simulation environment.
        """
        self.env = ThermurEnv(self.config)

    def _setup_policies(self):
        """
        Initializes the expert and learner (GNN) policies.

        The expert policy is a handcrafted controller used to generate optimal
        trajectory data. The learner policy is a Graph Neural Network that will
        be trained to imitate this expert.
        """
        # --- Expert Policy ---
        expert_controller = ExpertFlockingController(
            expert_config = self.config.policy.expert,
            agent_config  = self.config.agent
        )

        self.expert_policy = SafeModule(
            module   = expert_controller.compute_nominal_action,
            in_keys  = ["observation"],
            out_keys = ["action_expert"],
            spec     = self.env.action_spec,
        ).to(self.device)

        # --- Learner GNN Policy ---
        self.policy = GNNPolicy(
            in_dim  = self._calculate_gnn_input_dim(),
            out_dim = self.config.swarm.spatial_dims,
            config  = self.config.policy.gnn
        ).to(self.device)

    def _setup_data_pipeline(self):
        """
        Initializes the `torchrl` data collector and replay buffer.

        The collector is responsible for interacting with the environment using
        the expert policy to gather experience. The replay buffer stores this
        experience efficiently for sampling during training updates.
        """
        self.collector = SyncDataCollector(
            create_env_fn    = self.env,
            policy           = self.expert_policy,
            total_frames     = self.config.train.collector.total_frames,
            frames_per_batch = self.config.train.collector.frames_per_batch,
            device           = self.device,
        )

        self.replay_buffer = TensorDictReplayBuffer(
            storage     = "memory",
            batch_size  = self.config.train.replay.batch_size,
            buffer_size = self.config.train.replay.buffer_size,
            prefetch    = self.config.train.replay.prefetch,
        )

    def _setup_optimizer(self):
        """
        Initializes the loss module and the AdamW optimizer.
        """
        self.loss_module = ImitationLoss(self.policy)
        self.optimizer = AdamW(
            self.loss_module.parameters(),
            lr           = self.config.train.learning_rate,
            weight_decay = self.config.train.weight_decay,
        )

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
        logger.info(f"Starting training for {self.config.train.collector.total_frames} frames.")
        pbar = tqdm(total=self.config.train.collector.total_frames)

        total_frames = 0
        for i, data in enumerate(self.collector):

            self.replay_buffer.extend(data.to("cpu"))
            current_frames = data.numel()
            total_frames  += current_frames

            pbar.update(current_frames)

            if total_frames > self.config.train.replay.batch_size:
                batch     = self.replay_buffer.sample().to(self.device)
                loss_dict = self.loss_module(batch)
                loss      = loss_dict["loss"]
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                if i % self.config.train.log_interval == 0:
                    if self.config.wandb.mode != "disabled":
                        wandb.log({"train/loss": loss.item()}, step=total_frames)
                    pbar.set_description(f"Loss: {loss.item():.4f}")

                if i % self.config.train.checkpoint.interval == 0 and i > 0:
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
        path = self.config.train.checkpoint.path
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

    def _calculate_gnn_input_dim(self) -> int:
        """
        Calculates the GNN's input feature dimension dynamically.

        This method introspects the environment's observation specification to
        determine the total number of features for a single agent. This ensures
        the GNN architecture adapts automatically if the observation space
        changes, making the pipeline more robust.

        Returns:
            The integer size of the concatenated node feature vector.
        """
        spec = self.env.observation_spec

        # These are the keys that will be concatenated into the 'x' feature matrix
        feature_keys = [
            "position", "velocity", "temperature", "temperature_grad", "battery"
        ]
        
        total_dims = sum(
            spec[key].shape[-1] for key in feature_keys if key in spec.keys()
        )
        
        return total_dims