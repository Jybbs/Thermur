"""
Hydra-zen builder for TorchRL replay buffer components.

This module defines the configuration builder for TensorDictReplayBuffer, which
stores transitions for experience replay during training.
"""
from ..schemas                    import ReplayBufferModel, StorageModel
from hydra_zen                    import builds, zen
from torchrl.data                 import TensorDictReplayBuffer
from torchrl.data.replay_buffers  import LazyTensorStorage, SamplerWithoutReplacement


build_replay_buffer = builds(
    TensorDictReplayBuffer,
    sampler                 = builds(SamplerWithoutReplacement),
    storage                 = builds(LazyTensorStorage, **zen(StorageModel)),
    populate_full_signature = False,
    zen_dataclass           = {
        "module"   : "src.configs.factories.replay",
        "cls_name" : "ReplayBufferBuild"
    },
    **zen(ReplayBufferModel)
)
