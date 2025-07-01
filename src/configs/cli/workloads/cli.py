"""
CLI configuration workload for Thermur.

This module defines the configuration for the CLI system, composing all
necessary schemas and factories to create a Hydra-compatible configuration
that can be used with hydra-zen's instantiate method.
"""
from ..factories.cli import *
from hydra_zen       import make_config, ZenStore

cli_config = make_config(
    cli                 = build_cli,
    commands            = build_commands,
    explorer            = build_explorer,
    explorer_messages   = build_explorer_messages,
    headers             = build_headers,
    messages            = build_messages,
    message_types       = build_message_types,
    presets             = build_presets,
    prompts             = build_prompts,
    sections            = build_sections,
    status              = build_status,
    system              = build_system,
    theme               = build_theme,
    tips                = build_tips,
    training_components = build_training_components,
    ui                  = build_ui,
    validation          = build_validation,
    wandb_display       = build_wandb_display,
    defaults            = ["_self_"],
)


def register_cli_configs():
    """
    Register CLI configurations with Hydra's ConfigStore.
    """
    store = ZenStore(overwrite_ok=True)
    
    store(
        cli_config,
        name    = "cli",
        group   = "config",
        package = "_global_"
    )
    
    store.add_to_hydra_store()