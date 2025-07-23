"""
Lightning DataModule for experience collection and replay.

This module wraps TorchRL's trajectory collection and replay buffer components
into a Lightning DataModule, managing the flow of expert demonstrations
during imitation learning.
"""
from config.imitation.schemas.learning        import LearningModel
from pytorch_lightning                        import LightningDataModule
from thermur.imitation.controller.flocking    import ExpertFlockingController
from thermur.imitation.simulation.environment import SimulationEnv
from torch.utils.data                         import DataLoader
from torchrl.collectors                       import SyncDataCollector
from torchrl.data                             import TensorDictReplayBuffer
from torchrl.data.replay_buffers              import LazyTensorStorage, SamplerWithoutReplacement
from typing                                   import Optional


class ExperienceModule(LightningDataModule):
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
        env      : SimulationEnv,
        learning : LearningModel,
        policy   : ExpertFlockingController
    ):
        """
        Initialize the experience module.
        
        Args:
            env      : The flocking environment for trajectory collection
            learning : Learning configuration with batch sizes and buffer settings
            policy   : The expert controller to collect demonstrations from
        """
        super().__init__()
        self.env      = env
        self.learning = learning
        self.policy   = policy
        
        self.buffer    : Optional[TensorDictReplayBuffer] = None
        self.collector : Optional[SyncDataCollector]      = None
    
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
            create_env_fn       = lambda: self.env,
            frames_per_batch    = self.learning.frames_per_batch,
            max_frames_per_traj = self.learning.max_frames_per_traj,
            policy              = self.policy,
            total_frames        = self.learning.total_frames
        )
        
        self.buffer = TensorDictReplayBuffer(
            batch_size = self.learning.batch_size,
            prefetch   = self.learning.prefetch,
            sampler    = SamplerWithoutReplacement(),
            storage    = LazyTensorStorage(self.learning.buffer_size)
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
    
    def train_dataloader(self) -> "ExperienceDataLoader":
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
            buffer    = self.buffer,
            collector = self.collector,
            learning  = self.learning
        )
    


class ExperienceDataLoader(DataLoader):
    """
    Custom DataLoader that integrates TorchRL components.
    
    This dataloader manages the interaction between the trajectory collector
    and replay buffer, yielding batches of experiences for training while
    continuously collecting new trajectories in the background.
    """
    
    def __init__(
        self,
        buffer    : TensorDictReplayBuffer,
        collector : SyncDataCollector,
        learning  : LearningModel
    ):
        """
        Initialize the experience dataloader.
        
        Args:
            buffer    : The replay buffer for experience storage
            collector : The trajectory collector
            learning  : Learning configuration
        """
        self.buffer    = buffer
        self.collector = collector
        self.learning  = learning
    
    def __iter__(self):
        """
        Iterate over collected experiences.
        
        Yields batches from the replay buffer while continuously
        collecting new experiences from the environment.
        """
        for data in self.collector:
            self.buffer.extend(data.cpu())
            
            if len(self.buffer) >= self.learning.batch_size:
                yield self.buffer.sample()
    
    def __len__(self) -> int:
        """
        Return the number of training steps.
        
        Calculates based on total frames and frames per batch.
        """
        return self.learning.total_frames // self.learning.frames_per_batch