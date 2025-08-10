"""
Lightning DataModule for experience collection and replay.

This module wraps TorchRL's trajectory collection and replay buffer components
into a Lightning DataModule, managing the flow of expert demonstrations
during imitation learning.
"""
from __future__                  import annotations
from pytorch_lightning           import LightningDataModule
from torch                       import randint
from torchrl.collectors          import SyncDataCollector
from torchrl.data                import TensorDictReplayBuffer
from torchrl.data.replay_buffers import LazyTensorStorage, SamplerWithoutReplacement
from typing                      import TYPE_CHECKING

if TYPE_CHECKING:
    from config.imitation.lightning        import ExperienceModel
    from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
    from thermur.imitation.controller      import MurmurationController
    from thermur.imitation.simulation      import SimulationEnv


class DataModule(LightningDataModule):
    """
    Lightning DataModule for managing expert demonstration data.

    This module handles the data collection pipeline for imitation learning,
    wrapping TorchRL's SyncDataCollector and TensorDictReplayBuffer into
    Lightning's standardized interface. It manages:

    1. Expert trajectory collection via SyncDataCollector
    2. Experience storage in a replay buffer
    3. Batch sampling for training

    The DataModule ensures proper lifecycle management of data resources
    and provides a clean interface for the Lightning Trainer.
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
        self.collector : SyncDataCollector      | None = None

    def setup(self, stage: str | None = None) -> None:
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
            self.collector = SyncDataCollector(
                create_env_fn       = self.env,
                frames_per_batch    = self.experience.frames_per_batch,
                max_frames_per_traj = self.experience.max_frames_per_traj,
                policy              = self.expert,
                total_frames        = self.experience.total_frames,
                trust_policy        = True
            )

    def teardown(self, stage: str) -> None:
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
        collector  : SyncDataCollector,
        experience : ExperienceModel
    ):
        """
        Initialize the experience dataloader.

        Args:
            buffer     : The replay buffer for experience storage.
            collector  : The trajectory collector.
            experience : Experience data configuration.
        """
        self.buffer     = buffer
        self.collector  = collector
        self.experience = experience

    def __iter__(self):
        """
        Iterate over collected experiences.

        Yields batches from the replay buffer while continuously
        collecting new experiences from the environment.
        """
        for data in self.collector:
            self.buffer.extend(data.cpu())

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
    Validation dataloader that samples from the replay buffer.
    
    Since we're doing behavioral cloning from a fixed expert, we sample
    validation batches from the same replay buffer as training. This helps
    monitor training progress without needing a separate validation set.
    """
    
    def __init__(
        self,
        batch_size       : int,
        buffer           : TensorDictReplayBuffer,
        num_batches      : int,
        validation_split : float
    ):
        """
        Initialize the validation dataloader.
        
        Args:
            batch_size       : Number of samples per batch.
            buffer           : The replay buffer to sample from.
            num_batches      : Number of validation batches to yield.
            validation_split : Fraction of buffer reserved for validation.
        """
        self.batch_size       = batch_size
        self.buffer           = buffer
        self.num_batches      = num_batches
        self.validation_split = validation_split
    
    def __iter__(self):
        """
        Yield validation batches from the replay buffer.
        
        Samples from the portion of the buffer reserved for validation.
        During early training when buffer is filling, yields whatever data
        is available to ensure val/loss metric is computed.
        """
        buffer_len = len(self.buffer)
        
        # Need at least one sample to create a batch
        if buffer_len == 0:
            for _ in range(self.num_batches):
                yield from []
            return
            
        val_size          = max(1, int(buffer_len * self.validation_split))
        val_end_idx       = min(val_size, buffer_len)
        actual_batch_size = min(self.batch_size, val_end_idx)
        
        for _ in range(self.num_batches):
            if actual_batch_size == 1:
                yield self.buffer[[0]]
            else:
                indices = randint(
                    high = val_end_idx,
                    low  = 0,
                    size = (actual_batch_size,)
                )
                yield self.buffer[indices]
    
    def __len__(self) -> int:
        """
        Return the number of validation batches.
        
        Always returns the configured number of batches to ensure Lightning
        runs validation even when the buffer is still filling. The __iter__
        method handles empty buffer cases gracefully.
        """
        return self.num_batches
