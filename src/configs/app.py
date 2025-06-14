"""
Main application configuration using Hydra-zen.

This module assembles all the configuration builders into a complete
application config that can be instantiated by Hydra. It demonstrates
the full power of hydra-zen's declarative configuration approach.
"""
from configs        import *
from configs.builds import *
from hydra_zen      import make_config, store, zen
from omegaconf      import SI


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
    env               = build_environment,
    expert_controller = build_expert_policy,
    expert_policy     = SI("${expert_controller}"),  # For now, same as controller
    gnn_policy        = build_gnn_policy,
    collector         = build_collector,
    replay_buffer     = build_replay_buffer,
    loss_module       = build_loss,
    optimizer         = build_optimizer,
    safety_filter     = build_safety_filter,
    
    # The main orchestrator that will be instantiated
    orchestrator = build_orchestrator,
    
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
