"""
Training script for the Thermur project using Hydra-zen.

This script demonstrates the full power of hydra-zen's declarative configuration.
All components are instantiated by Hydra based on the configuration, and the
TrainingOrchestrator receives fully instantiated objects rather than configs.
"""
from __future__ import annotations

import hydra
from configs                        import register_configs
from hydra_zen                      import instantiate
from hydra_zen.third_party.pydantic import pydantic_parser
from thermur                        import TrainingOrchestrator


@hydra.main(
    config_path  = None,
    config_name  = "train", 
    version_base = None
)
def main(cfg):
    """
    Main entry point for training.
    
    This function is decorated with @hydra.main, enabling Hydra to manage
    the entire configuration and instantiation process. The orchestrator
    and all its dependencies are built automatically based on the declarative
    configuration.
    
    Args:
        cfg: The Hydra configuration object containing all builders.
    """
    # Register configurations with Hydra
    register_configs()
    
    # Hydra instantiates the entire object graph
    # The pydantic_parser ensures Pydantic validation happens during instantiation
    orchestrator: TrainingOrchestrator = instantiate(
        cfg.orchestrator,
        _target_wrapper_ = pydantic_parser,
    )
    
    # Run the training pipeline
    orchestrator.run()


if __name__ == "__main__":
    main()
