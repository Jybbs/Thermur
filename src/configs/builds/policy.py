"""
Hydra-zen builders for policy components.

This module defines configuration builders for both the expert flocking
controller and the GNN policy that learns from it. These builders leverage
Pydantic validation through the zen() wrapper.
"""
from configs    import AgentConfig, ExpertPolicyConfig, GNNConfig
from hydra_zen  import builds, zen
from thermur    import ExpertFlockingController, GNNPolicy


# Expert controller that generates demonstration data
build_expert_controller = builds(
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
build_gnn_policy = builds(
    GNNPolicy,
    config                  = zen(GNNConfig),
    populate_full_signature = True,
    zen_dataclass           = {
        "module"   : "src.configs.builds.policy", 
        "cls_name" : "GNNPolicyConfig"
    }
)


# For convenience, also export as expert policy
build_expert_policy = build_expert_controller
