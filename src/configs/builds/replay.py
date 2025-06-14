"""
Hydra-zen builder for the TorchRL replay buffer.

This module defines the configuration builder for TensorDictReplayBuffer,
which stores and samples experiences for training.
"""
from hydra_zen      import builds
from omegaconf      import SI
from torchrl.data   import TensorDictReplayBuffer


build_replay_buffer = builds(
    TensorDictReplayBuffer,
    storage                 = "memory",
    batch_size              = SI("${replay_buffer.batch_size}"),
    buffer_size             = SI("${replay_buffer.buffer_size}"),
    prefetch                = SI("${replay_buffer.prefetch}"),
    populate_full_signature = False,
    zen_dataclass           = {
        "module"   : "src.configs.builds.replay",
        "cls_name" : "ReplayBufferBuild"
    }
)
