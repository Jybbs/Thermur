"""
Hydra-zen builders for policy components.

This module defines configuration builders for the GNN policy.
These builders leverage Pydantic validation through the zen() wrapper.
"""
from ..schemas import GNNModel
from hydra_zen import builds, zen
from thermur   import GNNPolicy


build_policy = builds(
    GNNPolicy,
    config                  = zen(GNNModel),
    populate_full_signature = True,
    zen_dataclass           = {
        "module"   : "src.configs.factories.policy", 
        "cls_name" : "PolicyBuild"
    }
)
