"""
Application configuration schemas for the Thermur CLI.

This module defines the core application configuration and metadata, including
command definitions, training components, and integration settings.
"""
from pydantic import BaseModel, Field
from typing   import Literal


class CLIModel(BaseModel, extra="forbid"):
    """
    Main CLI configuration and metadata.
    
    This model consolidates all core application settings, command definitions,
    and component configurations into a single cohesive structure. It serves
    as the primary configuration object for the CLI application.
    """
    app_description: str = Field(
        default     = "🔥 Thermur: Advanced thermal-aware drone flock training "
                      "using imitation learning, physics-based constraints, and "
                      "real-time monitoring",
        description = "Application description shown in help text."
    )
    app_name: str = Field(
        default     = "thermur",
        description = "Application name used by Typer."
    )
    commands_available: list[dict[str, str]] = Field(
        default = [
            {
                "icon" : "📥",
                "name" : "download",
                "desc" : "Download WRF-Fire simulation datasets"
            },
            {
                "icon" : "📋",
                "name" : "info",
                "desc" : "Display system and configuration details"
            },
            {
                "icon" : "🎨",
                "name" : "monitor",
                "desc" : "Monitor training progress and resources"
            },
            {
                "icon" : "🚀",
                "name" : "train",
                "desc" : "Train thermal drone flock behaviors"
            },
            {
                "icon" : "✅",
                "name" : "validate",
                "desc" : "Validate configuration and dependencies"
            },
        ],
        description = "List of available commands with metadata."
    )
    commands_examples: list[dict[str, str]] = Field(
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
                "desc"    : "Train with overrides",
                "command" : "thermur train --config flock.num_drones=20 "
                            "environment.max_temp=85",
                "note"    : "Multiple parameters"
            },
            {
                "desc"    : "Download training data",
                "command" : "thermur download",
                "note"    : "Interactive selection of datasets"
            },
            {
                "desc"    : "Download sample data",
                "command" : "thermur download -s",
                "note"    : "Quick start with 468MB download (1.5GB extracted)"
            },
        ],
        description = "Example commands for quick start guide."
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
        description = "Help text for configuration override syntax."
    )
    training_component_configs: list[tuple[str, str, str]] = Field(
        default = [
            ("controller",        "controller",        "🎓 Controller"),
            ("experience_buffer", "experience_buffer", "💾 Experience Buffer"),
            ("loss",              "loss",              "📏 Loss Function"),
            ("optimizer",         "optimizer",         "🔎 Optimizer"),
            ("policy",            "policy",            "🧠 Learning Policy"),
            ("simulation",        "simulation",        "🌍 Simulation"),
            ("trajectory",        "trajectory",        "📊 Trajectory"),
        ],
        description = (
            "List of (key, config_path, display_name) tuples for training "
            "component initialization."
        )
    )


class WandbModel(BaseModel, extra="forbid"):
    """
    Weights & Biases configuration for the CLI.
    
    This model manages wandb project settings and API authentication
    for CLI operations like monitoring and training status display.
    """
    api_key: str = Field(
        default     = "WANDB_API_KEY",
        description = "Environment variable name for wandb API key."
    )
    mode: Literal["online", "offline", "disabled"] = Field(
        default     = "online",
        description = "Wandb tracking mode: online, offline, or disabled."
    )
    project: str = Field(
        default     = "thermur",
        description = "Wandb project name for experiment tracking."
    )