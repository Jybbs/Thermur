"""
Core configuration schemas for the Thermur CLI.

This module defines the main CLI configuration model and command-related
schemas that control the overall behavior of the command-line interface.
"""
from pydantic import BaseModel, Field


class CLIModel(BaseModel, extra="forbid"):
    """
    Main configuration model for the Thermur CLI.
    
    This model defines global settings that affect the entire CLI application.
    It serves as the root configuration that gets instantiated by Hydra.
    """
    app_description: str = Field(
        default     = "🔥 Thermur: Advanced thermal-aware drone flock training using "
                      "imitation learning, physics-based constraints, and real-time monitoring",
        description = "Application description shown in help text"
    )
    app_name: str = Field(
        default     = "thermur",
        description = "Application name used by Typer"
    )
    check_updates: bool = Field(
        default     = True,
        description = "Check for Thermur updates on startup"
    )
    debug_mode: bool = Field(
        default     = False,
        description = "Enable debug mode with verbose logging"
    )


class CommandsModel(BaseModel, extra="forbid"):
    """
    Defines available CLI commands and their metadata.
    
    This model centralizes command definitions used to generate help text
    and documentation. Each command has an icon, name, and description.
    """
    available: list[dict[str, str]] = Field(
        default = [
            {"icon": "🚀", "name": "train",     "desc": "Train thermal drone flock behaviors"},
            {"icon": "🔧", "name": "configure", "desc": "Manage training configurations"},
            {"icon": "📋", "name": "info",      "desc": "Display system and configuration details"},
            {"icon": "✅", "name": "validate",  "desc": "Validate configuration and dependencies"},
            {"icon": "🪄", "name": "monitor",   "desc": "Monitor training progress and resources"},
        ],
        description = "List of available commands with metadata"
    )
    examples: list[dict[str, str]] = Field(
        default = [
            {
                "desc"    : "Start interactive training",
                "command" : "thermur train",
                "note"    : "Guides you through configuration"
            },
            {
                "desc"    : "Quick test run",
                "command" : "thermur train --preset quick",
                "note"    : "5 epochs, ideal for testing"
            },
            {
                "desc"    : "Resume from checkpoint",
                "command" : "thermur train --resume latest",
                "note"    : "Continue interrupted training"
            },
            {
                "desc"    : "Monitor training progress",
                "command" : "thermur monitor --project my_project",
                "note"    : "Real-time wandb dashboard"
            },
            {
                "desc"    : "Validate configuration",
                "command" : "thermur validate --config hyperparameters.lr=0.001",
                "note"    : "Check before training"
            },
            {
                "desc"    : "Interactive configuration",
                "command" : "thermur configure",
                "note"    : "GUI-style config builder"
            },
            {
                "desc"    : "Train with overrides",
                "command" : "thermur train --config flock.num_drones=20 environment.max_temp=85",
                "note"    : "Multiple parameters"
            },
        ],
        description = "Example commands for quick start guide"
    )
    override_syntax_help: str = Field(
        default = (
            "# Override examples:\n"
            "hyperparameters.lr=0.001          # Learning rate\n"
            "hyperparameters.batch_size=64     # Batch size\n"
            "flock.num_drones=10               # Number of drones\n"
            "environment.max_temp=85.0         # Temperature limit\n"
            "+experiment=my_custom_setup       # Load experiment"
        ),
        description = "Help text for configuration override syntax"
    )
    override_syntax_title: str = Field(
        default     = "Configuration Override Syntax",
        description = "Title for override syntax help section"
    )


class TrainingComponentsModel(BaseModel, extra="forbid"):
    """
    Training component configuration mapping.
    
    Maps component names to their configuration paths and display names,
    used during the training initialization process.
    """
    component_configs: list[tuple[str, str, str]] = Field(
        default = [
            ("environment",       "simulation",        "🌍 Environment"),
            ("expert_policy",     "expert_policy",     "🎓 Expert Policy"),
            ("policy",            "policy",            "🧠 Learning Policy"),
            ("data_collector",    "data_collector",    "📊 Data Collector"),
            ("experience_buffer", "experience_buffer", "💾 Experience Buffer"),
            ("loss_function",     "loss_function",     "📏 Loss Function"),
            ("optimizer",         "optimizer",         "⚙️  Optimizer"),
            ("hyperparameters",   "hyperparameters",   "🎛️  Hyperparameters"),
            ("wandb_config",      "monitoring.wandb",  "🪄 wandb Tracking"),
        ],
        description = "List of (key, config_path, display_name) for training components"
    )
    visualizer_key: str = Field(
        default     = "visualization",
        description = "Configuration key for the visualizer component"
    )