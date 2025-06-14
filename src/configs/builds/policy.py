"""
Hydra-zen builders for policy components.

This module defines configuration builders for both the expert flocking
controller and the GNN policy that learns from it. These builders leverage
Pydantic validation through the zen() wrapper.
"""
from configs.models import AgentModel, ExpertPolicyModel, GNNModel
from hydra_zen      import builds, zen
from thermur        import ExpertFlockingController, GNNPolicy


# Expert controller that generates demonstration data
build_expert_controller = builds(
    ExpertFlockingController,
    expert_config           = zen(ExpertPolicyModel),
    agent_config            = zen(AgentModel),
    populate_full_signature = True,
    zen_dataclass           = {
        "module"   : "src.configs.builds.policy",
        "cls_name" : "ExpertControllerBuild"
    }
)


# GNN policy that learns from expert demonstrations
build_policy = builds(
    GNNPolicy,
    config                  = zen(GNNModel),
    populate_full_signature = True,
    zen_dataclass           = {
        "module"   : "src.configs.builds.policy", 
        "cls_name" : "PolicyBuild"
    }
)
