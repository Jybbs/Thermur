"""
Hydra-zen builder for the TorchRL replay buffer.

This module defines the configuration builder for TensorDictReplayBuffer,
which stores and samples experiences for training.
"""
from configs.models import ReplayBufferModel
from hydra_zen      import builds, zen
from torchrl.data   import TensorDictReplayBuffer


build_replay_buffer = builds(
    TensorDictReplayBuffer,
    storage                 = "memory",
    batch_size              = zen(ReplayBufferModel).batch_size,
    buffer_size             = zen(ReplayBufferModel).buffer_size,
    prefetch                = zen(ReplayBufferModel).prefetch,
    populate_full_signature = False,
    zen_dataclass           = {
        "module"   : "src.configs.builds.replay",
        "cls_name" : "ReplayBufferBuild"
    }
)
