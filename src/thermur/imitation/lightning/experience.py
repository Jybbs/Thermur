"""
Lightning DataModule for experience collection and replay.

This module wraps TorchRL's trajectory collection and replay buffer components
into a Lightning DataModule, managing the flow of expert demonstrations
during imitation learning.
"""
from __future__                  import annotations
from pytorch_lightning           import LightningDataModule
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

    def setup(self, stage: str):
        """
        Set up data collection components for the given stage.

        Lightning calls this method to initialize data components. For training,
        this creates the trajectory collector and experience buffer.

        Args:
            stage: The current stage ('fit', 'validate', 'test', or 'predict')
        """
        if stage != "fit":
            return

        self.collector = SyncDataCollector(
            create_env_fn       = self.env,
            frames_per_batch    = self.experience.frames_per_batch,
            max_frames_per_traj = self.experience.max_frames_per_traj,
            policy              = self.expert,
            total_frames        = self.experience.total_frames
        )

        self.buffer = TensorDictReplayBuffer(
            batch_size = self.experience.batch_size,
            prefetch   = self.experience.prefetch,
            sampler    = SamplerWithoutReplacement(),
            storage    = LazyTensorStorage(self.experience.buffer_size)
        )

    def teardown(self, stage: str):
        """
        Clean up data collection resources.

        Properly shuts down the data collector when training ends.

        Args:
            stage: The current stage being torn down
        """
        if stage == "fit" and self.collector:
            self.collector.shutdown()

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        """
        Create the training dataloader.

        Returns a custom dataloader that wraps the TorchRL collector
        and replay buffer, providing batches of experiences for training.

        Returns:
            DataLoader that yields experience batches
        """
        if not (self.buffer and self.collector):
            raise RuntimeError("DataModule not properly set up. Call setup('fit')")

        return ExperienceDataLoader(
            buffer     = self.buffer,
            collector  = self.collector,
            experience = self.experience
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
            buffer     : The replay buffer for experience storage
            collector  : The trajectory collector
            experience : Experience data configuration
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
