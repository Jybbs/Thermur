"""
Hydra-zen builders for data sources and trajectory collection.

This module provides configuration builders for environmental data sources,
trajectory collection, and experience replay buffers.
"""
from config.imitation.schemas.dataset import WRFModel
from hydra_zen                        import builds
from omegaconf                        import SI
from torchrl.data                     import TensorDictReplayBuffer
from torchrl.data.replay_buffers      import LazyTensorStorage, SamplerWithoutReplacement


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

build_source = builds(
    WRFModel,
    populate_full_signature = True,
    zen_dataclass           = {
        "module"   : "src.configs.imitation.factories.source",
        "cls_name" : "SourceBuild"
    }
)
"""
Builder for WRF-Fire data source configuration.

Manages environmental data source configuration including variable names,
processing options, and domain randomization settings for the WRF-Fire
NetCDF data that provides temperature, wind, and fire heat flux fields.
"""
