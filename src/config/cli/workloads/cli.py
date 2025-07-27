"""
CLI configuration workload for Thermur.

This module defines the configuration for the CLI system, composing all
necessary schemas and factories to create a Hydra-compatible configuration
that can be used with hydra-zen's instantiate method.
"""
from config.cli.factories import *
from hydra_zen            import make_config, ZenStore


cli_cfg = make_config(
    display  = build_display,
    download = build_download,
    messages = build_messages,
    presets  = build_presets,
    prompts  = build_prompts,
    secrets  = build_secrets,
    system   = build_system,
    wandb    = build_wandb,
    defaults = ["_self_"],
)


def register_cli_cfgs():
    """
    Register CLI configurations with Hydra's ConfigStore.
    
    This function adds the CLI configuration to Hydra's global ConfigStore
    for potential use in Hydra-based workflows. However, the Thermur CLI
    itself directly instantiates `cli_cfg` using hydra_zen.instantiate(),
    allowing for immediate configuration loading without Hydra's runtime 
    overhead.
    
    The configuration includes all UI components, command definitions, themes,
    and interactive prompts needed for the Thermur command-line interface.
    """
    store = ZenStore(overwrite_ok=True)
    
    store(
        cli_cfg,
        group   = "config",
        name    = "cli",
        package = "_global_"
    )
    
    store.add_to_hydra_store()