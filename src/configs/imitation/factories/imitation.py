"""
Hydra-zen builders for training components.

This module consolidates configuration builders for the policy network,
optimizer, and loss function - all core components of the training loop.
"""
from ..schemas        import LearningModel
from hydra_zen        import builds, zen
from omegaconf        import SI
from thermur.models   import GNNPolicy
from thermur.training import ImitationLoss
from torch.optim      import AdamW


"""
Builder for the Graph Neural Network policy.

Creates a permutation-equivariant GNN that processes the swarm graph
structure to output nominal control actions u_nom for each agent.
"""
build_policy = builds(
    GNNPolicy,
    learning_config         = zen(LearningModel),
    populate_full_signature = True,
    zen_dataclass           = {
        "module"   : "src.configs.imitation.factories.imitation",
        "cls_name" : "PolicyBuild"
    }
)

"""
Builder for the AdamW optimizer.

Configures adaptive learning with weight decay for stable training
of the GNN policy via behavioral cloning.
"""
build_optimizer = builds(
    AdamW,
    params                  = SI("${policy}"),
    lr                      = SI("${learning.learning_rate}"),
    weight_decay            = SI("${learning.weight_decay}"),
    populate_full_signature = False,
    zen_partial             = True,
    zen_dataclass           = {
        "module"   : "src.configs.imitation.factories.imitation",
        "cls_name" : "OptimizerBuild"
    }
)

"""
Builder for the imitation learning loss.

Implements MSE loss L = ||π_θ(s) - π*(s)||² between learned policy π_θ
and expert demonstrations π* for behavioral cloning.
"""
build_loss = builds(
    ImitationLoss,
    policy_network          = SI("${policy}"),
    populate_full_signature = True,
    zen_dataclass           = {
        "module"   : "src.configs.imitation.factories.imitation",
        "cls_name" : "LossBuild"
    }
)