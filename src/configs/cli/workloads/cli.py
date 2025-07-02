"""
CLI configuration workload for Thermur.

This module defines the configuration for the CLI system, composing all
necessary schemas and factories to create a Hydra-compatible configuration
that can be used with hydra-zen's instantiate method.
"""
from ..factories import *
from hydra_zen   import make_config, ZenStore

cli_config = make_config(
    cli                 = build_cli,
    commands            = build_commands,
    messages            = build_messages,
    presets             = build_presets,
    prompts             = build_prompts,
    status              = build_status,
    theme               = build_theme,
    training_components = build_training_components,
    ui                  = build_ui,
    validation          = build_validation,
    wandb_display       = build_wandb_display,
    defaults            = ["_self_"],
)


def register_cli_configs():
    """
    Register CLI configurations with Hydra's ConfigStore.
    
    This function adds the CLI configuration to Hydra's global ConfigStore
    for potential use in Hydra-based workflows. However, the Thermur CLI
    itself directly instantiates cli_config using hydra_zen.instantiate(),
    allowing for immediate configuration loading without Hydra's runtime 
    overhead.
    
    The configuration includes all UI components, command definitions, themes,
    and interactive prompts needed for the Thermur command-line interface.
    """
    store = ZenStore(overwrite_ok=True)
    
    store(
        cli_config,
        name    = "cli",
        group   = "config",
        package = "_global_"
    )
    
    store.add_to_hydra_store()