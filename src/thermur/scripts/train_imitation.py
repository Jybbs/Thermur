"""
Training script for imitation learning using the refactored architecture.

This script demonstrates the clean entry point pattern with Hydra-zen,
directly instantiating only the components needed for training without
an orchestrator intermediary.
"""
from configs                        import register_imitation_training_config
from configs.entry_points           import imitation_train_config
from hydra_zen                      import instantiate, zen
from hydra_zen.third_party.pydantic import pydantic_parser
from thermur                        import configure_loguru, set_seed, train_imitation_learning


def main(cfg):
    """
    Main entry point for imitation learning training.
    
    This function directly instantiates the necessary components and
    runs the training loop, without the overhead of an orchestrator.
    
    Args:
        cfg: The Hydra configuration object containing all builders.
    """
    # Setup logging and seed
    configure_loguru(instantiate(cfg.logging, _parser=pydantic_parser))
    set_seed(instantiate(cfg.hyperparameters, _parser=pydantic_parser).seed)
    
    # Instantiate components directly
    environment       = instantiate(cfg.environment,       _parser=pydantic_parser)
    expert_policy     = instantiate(cfg.expert_policy,     _parser=pydantic_parser)
    policy            = instantiate(cfg.policy,            _parser=pydantic_parser)
    data_collector    = instantiate(cfg.data_collector,    _parser=pydantic_parser)
    experience_buffer = instantiate(cfg.experience_buffer, _parser=pydantic_parser)
    loss_function     = instantiate(cfg.loss_function,     _parser=pydantic_parser)
    optimizer         = instantiate(cfg.optimizer,         _parser=pydantic_parser)
    hyperparameters   = instantiate(cfg.hyperparameters,   _parser=pydantic_parser)
    wandb_config      = instantiate(cfg.wandb,             _parser=pydantic_parser)
    
    # Run the training
    train_imitation_learning(
        environment       = environment,
        expert_policy     = expert_policy,
        policy            = policy,
        data_collector    = data_collector,
        experience_buffer = experience_buffer,
        loss_function     = loss_function,
        optimizer         = optimizer,
        hyperparameters   = hyperparameters,
        wandb_config      = wandb_config,
    )


if __name__ == "__main__":
    # Register configurations with Hydra
    register_imitation_training_config()
    
    # Use hydra-zen's pattern to run with the config
    zen(imitation_train_config).hydra_main(
        config_name  = "imitation_train",
        config_path  = None,
        version_base = None,
    )(main)
