"""
Lightning DataModule for experience collection and replay.

This module provides a Lightning-compatible interface for expert demonstration
collection and replay buffer management during imitation learning.
"""
from __future__                  import annotations
from .collector                  import ExperienceCollector
from pytorch_lightning           import LightningDataModule
from torch                       import unique, where
from torchrl.data                import TensorDictReplayBuffer
from torchrl.data.replay_buffers import LazyTensorStorage, SamplerWithoutReplacement
from typing                      import TYPE_CHECKING

import torch as th

if TYPE_CHECKING:
    from config.imitation.training         import ExperienceModel
    from pytorch_lightning                 import LightningModule
    from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
    from thermur.imitation.controller      import MurmurationController
    from thermur.imitation.simulation      import SimulationEnv


class DataModule(LightningDataModule):
    """
    Lightning DataModule for managing expert demonstration data.

    This module orchestrates the data collection pipeline for imitation learning,
    integrating our custom ExperienceCollector with TorchRL's replay buffer.
    It manages trajectory collection, experience storage, and batch sampling
    while ensuring proper lifecycle management within Lightning's training loop.
    """

    def __init__(
        self,
        env        : SimulationEnv,
        experience : ExperienceModel,
        expert     : MurmurationController
    ):
        """
        Initialize the experience module.

        Args:
            env        : The simulation environment for data collection
            experience : Experience data configuration with batch sizes and buffer
                         settings
            expert     : The murmuration controller that generates actions
        """
        super().__init__()
        self.env        = env
        self.experience = experience
        self.expert     = expert

        self.buffer    : TensorDictReplayBuffer | None = None
        self.collector : ExperienceCollector    | None = None

    def setup(self, stage: str | None = None):
        """
        Set up data collection components for the given stage.

        Lightning calls this method to initialize data components. For training,
        this creates the trajectory collector and experience buffer.

        Args:
            stage: The current stage ('fit', 'validate', 'test', or 'predict').
                   Can also be a TrainerFn enum value.
        """
        if not stage or "fit" not in str(stage).lower():
            return
        
        if self.buffer is None:
            self.buffer = TensorDictReplayBuffer(
                batch_size = self.experience.batch_size,
                prefetch   = self.experience.prefetch,
                sampler    = SamplerWithoutReplacement(),
                storage    = LazyTensorStorage(self.experience.buffer_size)
            )
            
        if self.collector is None:
            self.collector = ExperienceCollector(
                env                 = self.env,
                expert              = self.expert,
                frames_per_batch    = self.experience.frames_per_batch,
                max_frames_per_traj = self.experience.max_frames_per_traj,
                total_frames        = self.experience.total_frames
            )

    def teardown(self, stage: str):
        """
        Clean up data collection resources.

        Properly shuts down the data collector when training ends.

        Args:
            stage: The current stage being torn down.
        """
        if stage == "fit" and self.collector is not None:
            self.collector.shutdown()

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        """
        Create the training dataloader.

        Returns a custom dataloader that wraps the TorchRL collector
        and replay buffer, providing batches of experiences for training.

        Returns:
            DataLoader that yields experience batches.
        """
        assert self.buffer    is not None
        assert self.collector is not None
        
        return ExperienceDataLoader(
            buffer     = self.buffer,
            collector  = self.collector,
            experience = self.experience
        )
    
    def val_dataloader(self):
        """
        Create the validation dataloader.
        
        Returns a dataloader that samples from the dedicated validation buffer,
        ensuring proper train/validation separation for better generalization
        monitoring.
        
        Returns:
            ValidationDataLoader that yields batches from the validation buffer.
            The dataloader will handle cases where the buffer is still filling.
        """
        if self.buffer is None:
            return []
        
        return ValidationDataLoader(
            buffer           = self.buffer,
            batch_size       = self.experience.batch_size,
            num_batches      = self.experience.validation_batches,
            validation_split = self.experience.validation_split
        )


class ExperienceDataLoader:
    """
    Custom DataLoader that integrates TorchRL components.

    This dataloader manages the interaction between the trajectory collector
    and replay buffer, yielding batches of experiences for training while
    continuously collecting new trajectories in the background.
    """

    def __init__(
        self,
        buffer     : TensorDictReplayBuffer,
        collector  : ExperienceCollector,
        experience : ExperienceModel,
        pl_module  : LightningModule | None = None
    ):
        """
        Initialize the experience dataloader.

        Args:
            buffer     : The replay buffer for experience storage.
            collector  : The trajectory collector.
            experience : Experience data configuration.
            pl_module  : Optional Lightning module for WandB logging.
        """
        self.buffer     = buffer
        self.collector  = collector
        self.experience = experience
        self.pl_module  = pl_module

    def __iter__(self):
        """
        Iterate over collected experiences.

        Yields batches from the replay buffer while continuously
        collecting new experiences from the environment.
        """
        for data in self.collector:
            self.buffer.extend(data)

            if len(self.buffer) >= self.experience.batch_size:
                yield self.buffer.sample()

    def __len__(self) -> int:
        """
        Return the number of training steps.

        Calculates based on total frames and frames per batch.
        """
        return self.experience.total_frames // self.experience.frames_per_batch


class ValidationDataLoader:
    """
    Trajectory-aware validation dataloader.
    
    Reserves entire trajectories for validation to prevent temporal
    leakage and ensure meaningful generalization metrics.
    """
    
    def __init__(
        self,
        batch_size       : int,
        buffer           : TensorDictReplayBuffer,
        num_batches      : int,
        validation_split : float
    ):
        """
        Initialize validation dataloader with trajectory splitting.
        
        Args:
            batch_size       : Number of samples per batch.
            buffer           : The replay buffer to sample from.
            num_batches      : Number of validation batches to yield.
            validation_split : Fraction of trajectories reserved for validation.
        """
        self.batch_size         = batch_size
        self.buffer             = buffer
        self.num_batches        = num_batches
        self.validation_split   = validation_split
        self.val_sample_indices = th.empty(0, dtype=th.long)
        self.val_trajectory_ids = th.empty(0, dtype=th.long)
        self._update_trajectory_split()
    
    def __iter__(self):
        """
        Yield batches sampled from validation trajectories.
        
        Falls back to random sampling during warmup before trajectory
        assignment is complete.
        """
        if not (buffer_size := len(self.buffer)):
            return
        
        if not self.val_trajectory_ids.numel():
            self._update_trajectory_split()
        
        available_indices = (
            self.val_sample_indices 
            if self.val_sample_indices.numel()
            else th.arange(buffer_size)
        )
        
        if not (n_available := available_indices.numel()):
            yield from (self.buffer.sample() for _ in range(self.num_batches))
            return
        
        batch_size = min(self.batch_size, n_available)
        for _ in range(self.num_batches):
            batch_selection = th.randperm(
                device = available_indices.device,
                n      = n_available,
            )[:batch_size]

            yield self.buffer[available_indices[batch_selection].tolist()]
    
    def __len__(self) -> int:
        """
        Return the number of validation batches.
        
        Always returns the configured number of batches to ensure Lightning
        runs validation even when the buffer is still filling. The __iter__
        method handles empty buffer cases gracefully.
        """
        return self.num_batches

    def _refresh_sample_indices(self):
        """
        Cache indices of samples belonging to validation trajectories.
        
        Uses vectorized comparison to identify which buffer samples belong
        to trajectories reserved for validation. This precomputation enables
        efficient batch sampling without repeated trajectory lookups.
        """
        if (
            not self.val_trajectory_ids.numel() or 
            not len(self.buffer) or
            "trajectory_id" not in (full_data := self.buffer[:])
        ):
            self.val_sample_indices = th.empty(0, dtype=th.long)
            return
        
        all_trajectory_ids = full_data["trajectory_id"].view(-1)
        validation_mask    = (
            all_trajectory_ids.unsqueeze(1) == self.val_trajectory_ids
        ).any(dim=1)
        
        self.val_sample_indices = where(validation_mask)[0]

    def _update_trajectory_split(self):
        """
        Assign trajectories to validation set.
        
        Samples a subset of unique trajectory IDs for consistent
        train/validation separation.
        """
        if not (buffer_size := len(self.buffer)):
            return
        
        probe_data = self.buffer[:min(buffer_size, 1000)]
        if "trajectory_id" not in probe_data:
            return
        
        trajectory_ids      = probe_data["trajectory_id"].view(-1)
        unique_trajectories = unique(trajectory_ids)
        
        if not (n_trajectories := unique_trajectories.numel()):
            return
        
        n_val    = max(1, int(n_trajectories * self.validation_split))
        shuffled = th.randperm(
            device = unique_trajectories.device,
            n      = n_trajectories
        )

        self.val_trajectory_ids = unique_trajectories[shuffled[:n_val]]
        self._refresh_sample_indices()
