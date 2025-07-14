"""
Hydra-zen builders for data collection and storage.

This module provides configuration builders for trajectory collection,
experience replay, and WRF dataset downloads.
"""
from ..schemas                   import WRFModel
from hydra_zen                   import builds
from omegaconf                   import SI
from torchrl.collectors          import SyncDataCollector  
from torchrl.data                import TensorDictReplayBuffer
from torchrl.data.replay_buffers import LazyTensorStorage, SamplerWithoutReplacement


build_experience_buffer = builds(
    TensorDictReplayBuffer,
    batch_size              = SI("${learning.batch_size}"),
    sampler                 = builds(SamplerWithoutReplacement),
    storage                 = builds(
        LazyTensorStorage,
        max_size = SI("${learning.buffer_size}"),
    ),
    populate_full_signature = False,
    zen_dataclass           = {
        "module"   : "src.configs.imitation.factories.data",
        "cls_name" : "ReplayBufferBuild"
    },
)
"""
Builder for experience replay buffer.

Stores and samples agent experiences for training, enabling efficient
batch processing and decorrelated learning updates.
"""

build_dataset = builds(
    WRFModel,
    populate_full_signature = True,
    zen_dataclass           = {
        "module"   : "src.configs.imitation.factories.data",
        "cls_name" : "DatasetBuild"
    }
)
"""
Builder for WRF-Fire dataset configuration.

Manages both dataset acquisition (download, caching) and data structure
definitions (variable names, processing options) in a single configuration.
Supports downloading subsets of large datasets and domain randomization.
"""

build_trajectory = builds(
    SyncDataCollector,
    device                  = SI("${learning.device}"),
    frames_per_batch        = SI("${learning.frames_per_batch}"),
    policy                  = SI("${controller}"),
    simulation              = SI("${simulation}"),
    total_frames            = SI("${learning.total_frames}"),
    populate_full_signature = False,
    zen_dataclass           = {
        "module"   : "src.configs.imitation.factories.dataset",
        "cls_name" : "TrajectoryBuild"
    }
)
"""
Builder for trajectory collection during training.

Manages the interaction loop between expert policy and environment,
collecting demonstration trajectories for imitation learning.
"""
