"""
Hydra-zen builder for the TorchRL replay buffer.

This module defines the configuration builder for TensorDictReplayBuffer,
which stores and samples experiences for training.
"""
from hydra_zen   import builds, zen
from src.configs import ReplayBufferConfig
from torchrl.data import TensorDictReplayBuffer


replay_buffer_config = builds(
    TensorDictReplayBuffer,
    storage                 = "memory",
    batch_size              = zen(ReplayBufferConfig).batch_size,
    buffer_size             = zen(ReplayBufferConfig).buffer_size,
    prefetch                = zen(ReplayBufferConfig).prefetch,
    populate_full_signature = False,  # TensorDictReplayBuffer has many optional params
    zen_dataclass           = {
        "module"   : "src.configs.builds.replay",
        "cls_name" : "ReplayBufferConfig"
    }
)
