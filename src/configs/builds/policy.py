"""
Hydra-zen builders for policy components.

This module defines configuration builders for both the expert flocking
controller and the GNN policy that learns from it. These builders leverage
Pydantic validation through the zen() wrapper.
"""
from hydra_zen   import builds, zen
from src.configs import AgentConfig, ExpertPolicyConfig, GNNConfig
from src.thermur import ExpertFlockingController, GNNPolicy


# Expert controller that generates demonstration data
expert_controller_config = builds(
    ExpertFlockingController,
    expert_config           = zen(ExpertPolicyConfig),
    agent_config            = zen(AgentConfig),
    populate_full_signature = True,
    zen_dataclass           = {
        "module"   : "src.configs.builds.policy",
        "cls_name" : "ExpertControllerConfig"
    }
)


# GNN policy that learns from expert demonstrations
gnn_policy_config = builds(
    GNNPolicy,
    config                  = zen(GNNConfig),
    populate_full_signature = True,
    zen_dataclass           = {
        "module"   : "src.configs.builds.policy", 
        "cls_name" : "GNNPolicyConfig"
    }
)


# For convenience, also export individual configs
expert_policy_config = expert_controller_config
