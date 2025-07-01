"""
Hydra-zen builders for data collection and storage.

This module provides configuration builders for the data collector and
experience replay buffer used in the imitation learning pipeline.
"""
from hydra_zen                   import builds
from omegaconf                   import SI
from torchrl.collectors          import SyncDataCollector  
from torchrl.data                import TensorDictReplayBuffer
from torchrl.data.replay_buffers import LazyTensorStorage, SamplerWithoutReplacement


build_collector = builds(
    SyncDataCollector,
    create_env_fn           = SI("${simulation}"),
    device                  = SI("${learning.device}"),
    policy                  = SI("${expert_policy}"),
    frames_per_batch        = SI("${learning.frames_per_batch}"),
    total_frames            = SI("${learning.total_frames}"),
    populate_full_signature = False,
    zen_dataclass           = {
        "module"   : "src.configs.imitation.factories.data",
        "cls_name" : "CollectorBuild"
    }
)
"""
Builder for synchronous data collection.

Manages the interaction loop between expert policy and environment,
collecting demonstration trajectories for imitation learning.
"""

build_replay_buffer = builds(
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