"""
CLI configuration workload for Thermur.

This module defines the configuration for the CLI system, composing all
necessary schemas and factories to create a Hydra-compatible configuration
that can be used with @hydra.main decorator.
"""
from ..factories.cli import (
    build_cli,
    build_commands,
    build_explorer,
    build_explorer_messages,
    build_headers,
    build_messages,
    build_message_types,
    build_presets,
    build_prompts,
    build_sections,
    build_status,
    build_system,
    build_theme,
    build_tips,
    build_training_components,
    build_ui,
    build_validation,
    build_wandb_display,
)
from hydra_zen   import make_config, ZenStore


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
    
    store(
        cli_config,
        name    = "cli",
        group   = "config",
        package = "_global_"
    )
    
    store.add_to_hydra_store()