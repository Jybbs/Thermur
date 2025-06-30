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
        default     = "🔥 Thermally-constrained drone swarm training toolkit",
        description = "Application description shown in help text"
    )
    app_name: str = Field(
        default     = "thermur",
        description = "Application name used by Typer"
    )
    auto_fix_permissions: bool = Field(
        default     = True,
        description = "Automatically fix file permissions when needed"
    )
    check_updates: bool = Field(
        default     = True,
        description = "Check for Thermur updates on startup"
    )
    debug_mode: bool = Field(
        default     = False,
        description = "Enable debug mode with verbose logging"
    )
    telemetry_enabled: bool = Field(
        default     = False,
        description = "Enable anonymous usage telemetry"
    )


class CommandsModel(BaseModel, extra="forbid"):
    """
    Defines available CLI commands and their metadata.
    
    This model centralizes command definitions used to generate help text
    and documentation. Each command has an icon, name, and description.
    """
    override_syntax_help: str = Field(
        default = (
            "# Override examples:\n"
            "hyperparameters.lr=0.001          # Learning rate\n"
            "hyperparameters.batch_size=64     # Batch size\n"
            "swarm.num_drones=10               # Number of drones\n"
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
            ("wandb_config",      "monitoring.wandb",  "📊 wandb Tracking"),
        ],
        description = "List of (key, config_path, display_name) for training components"
    )
    visualizer_key: str = Field(
        default     = "visualization",
        description = "Configuration key for the visualizer component"
    )