"""
Main application configuration using Hydra-zen.

This module assembles all the configuration builders into a complete
application config that can be instantiated by Hydra. It demonstrates
the full power of hydra-zen's declarative configuration approach.
"""
from hydra_zen                      import make_config, store, zen
from hydra_zen.third_party.pydantic import pydantic_parser
from omegaconf                      import SI

from src.configs import (
    AgentConfig,
    CollectorConfig,
    EnvironmentConfig,
    LoggingConfig,
    PolicyConfig,
    SafetyConfig,
    SwarmConfig,
    TrainConfig,
    WandbConfig,
)
from src.configs.builds import (
    collector_config,
    env_config,
    expert_policy_config,
    gnn_policy_config,
    loss_config,
    optimizer_config,
    orchestrator_config,
    replay_buffer_config,
    safety_filter_config,
)


# Create the main application config that assembles all components
app_config = make_config(
    # Pydantic configurations (validated via zen())
    agent_config       = zen(AgentConfig),
    collector_config   = zen(CollectorConfig), 
    environment_config = zen(EnvironmentConfig),
    logging_config     = zen(LoggingConfig),
    policy_config      = zen(PolicyConfig),
    safety_config      = zen(SafetyConfig),
    swarm_config       = zen(SwarmConfig),
    train_config       = zen(TrainConfig),
    wandb_config       = zen(WandbConfig),
    
    # Component builders (will be instantiated by Hydra)
    env               = env_config,
    expert_controller = expert_policy_config,
    expert_policy     = SI("${expert_controller}"),  # For now, same as controller
    gnn_policy        = gnn_policy_config,
    collector         = collector_config,
    replay_buffer     = replay_buffer_config,
    loss_module       = loss_config,
    optimizer         = optimizer_config,
    safety_filter     = safety_filter_config,
    
    # The main orchestrator that will be instantiated
    orchestrator = orchestrator_config,
    
    # Hydra defaults
    defaults = ["_self_"],
)


def register_configs():
    """Register all configurations with Hydra's ConfigStore."""
    # Register the main app config
    store(
        app_config,
        name    = "app",
        group   = "config",
        package = "_global_"
    )
    
    # Also register for the train script
    store(
        app_config,
        name    = "train",
        group   = "config",
        package = "_global_"
    )
    
    # Add to hydra store with pydantic parser
    store.add_to_hydra_store(overwrite_ok=True)


# Export for convenience
__all__ = ["app_config", "register_configs"]
