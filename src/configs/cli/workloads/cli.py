"""
CLI configuration workload for Thermur.

This module defines the configuration for the CLI system, composing all
necessary schemas and factories to create a Hydra-compatible configuration
that can be used with @hydra.main decorator.
"""
from hydra_zen import builds, make_config, ZenStore

from ..factories import *
from ..schemas import *


# Create the CLI configuration by composing all components
cli_config = make_config(
    # Core CLI configuration
    cli                 = build_cli,
    
    # UI and theme configuration
    theme               = build_theme,
    ui                  = build_ui,
    
    # Message and text configuration
    commands            = build_commands,
    headers             = build_headers,
    messages            = build_messages,
    message_types       = build_message_types,
    prompts             = build_prompts,
    sections            = build_sections,
    status              = build_status,
    
    # System configuration
    system              = build_system,
    validation          = build_validation,
    wandb_display       = build_wandb_display,
    
    # Feature configuration
    explorer            = build_explorer,
    explorer_messages   = build_explorer_messages,
    presets             = build_presets,
    tips                = build_tips,
    training_components = build_training_components,
    
    # Hydra defaults
    defaults = ["_self_"],
)


def register_cli_configs():
    """
    Register CLI configurations with Hydra's ConfigStore.
    
    This allows the CLI configuration to be used with Hydra's
    @hydra.main decorator for configuration management.
    """
    store = ZenStore(overwrite_ok=True)
    
    # Register the CLI config
    store(
        cli_config,
        name    = "cli",
        group   = "config",
        package = "_global_"
    )
    
    # Add to Hydra
    store.add_to_hydra_store()