"""
Simplified CLI configuration schemas.

This module provides configuration models for CLI components that need to be
configurable or contain structured data (arrays/dicts). Single-use strings
have been moved inline to reduce indirection.
"""
from pydantic import BaseModel, Field, PositiveInt
from typing   import Literal


class DisplayModel(BaseModel, extra="forbid"):
    """
    Display configuration for structured UI elements.

    Contains arrays and dictionaries that define the CLI's visual structure
    and may need to be extended or modified.
    """
    commands_available: list[dict[str, str]] = Field(
        default = [
            {
                "desc" : "Display system and configuration details",
                "icon" : "📋",
                "name" : "info",
            },
            {
                "desc" : "Monitor training progress and resources",
                "icon" : "🎨",
                "name" : "monitor",
            },
            {
                "desc" : "Explore and manage training runs",
                "icon" : "🏃",
                "name" : "runs",
            },
            {
                "desc" : "Train thermal drone flock behaviors",
                "icon" : "🚀",
                "name" : "train",
            },
            {
                "desc" : "Validate configuration and dependencies",
                "icon" : "✅",
                "name" : "validate",
            },
        ],
        description = "List of available commands with metadata for the main help display."
    )
    commands_examples: list[dict[str, str]] = Field(
        default = [
            {
                "command" : "thermur train",
                "desc"    : "Start interactive training",
                "note"    : "Guides you through configuration"
            },
            {
                "command" : "thermur train training.optimizer.learning_rate=0.001",
                "desc"    : "Custom learning rate",
                "note"    : "Override specific parameters"
            },
            {
                "command" : "thermur runs",
                "desc"    : "List recent training runs",
                "note"    : "View run history with status"
            },
            {
                "command" : "thermur runs show",
                "desc"    : "Show last run configuration",
                "note"    : "Paginated view of all settings"
            },
            {
                "command" : "thermur monitor",
                "desc"    : "Monitor training progress",
                "note"    : "Real-time wandb dashboard"
            },
            {
                "command" : "thermur validate",
                "desc"    : "Validate configuration",
                "note"    : "Check setup before training"
            },
            {
                "command" : "thermur train controller.expert.agent_count=20 training.optimizer.seed=42",
                "desc"    : "Train with overrides",
                "note"    : "Multiple parameters"
            },
        ],
        description = "Example commands demonstrating common usage patterns for new users."
    )
    fire_gradient: list[str] = Field(
        default = [
            "red3",
            "red1",
            "orange_red1",
            "dark_orange",
            "orange1",
            "yellow1",
            "bright_yellow"
        ],
        description = (
            "Sequential color gradient used for animating fire intensity effects in "
            "the Thermur branding display."
        )
    )
    message_types: dict[str, dict[str, str]] = Field(
        default = {
            'config'  : {'icon': '⚙️ ', 'style': 'config'},
            'debug'   : {'icon': '🐛',  'style': 'debug'},
            'error'   : {'icon': '❌',  'style': 'error'},
            'flock'   : {'icon': '🪿',  'style': 'flock'},
            'info'    : {'icon': 'ℹ️ ', 'style': 'info'},
            'magic'   : {'icon': '🪄',  'style': 'accent'},
            'success' : {'icon': '✅',  'style': 'success'},
            'thermal' : {'icon': '🔥',  'style': 'thermal'},
            'warning' : {'icon': '⚠️ ', 'style': 'warning'},
        },
        description = "Message type configurations mapping semantic types to icons and style names."
    )
    override_examples: str = Field(
        default = (
            "# Override examples:\n"
            "training.optimizer.learning_rate=0.001   # Learning rate\n"
            "training.experience.batch_size=64        # Batch size\n"
            "controller.expert.agent_count=10         # Number of agents\n"
            "training.hardware.precision=32-true      # Training precision\n"
            "cli.wandb.mode=offline                   # W&B logging mode"
        ),
        description = "Multi-line help text showing example Hydra-style configuration overrides."
    )
    progress_bar_length: PositiveInt = Field(
        default     = 20,
        description = "Character width for rendering progress bars in terminal output."
    )
    questionary_style: dict[str, str] = Field(
        default = {
            'answer'      : 'fg:#4ecdc4 bold',
            'disabled'    : 'fg:#808080 italic',
            'highlighted' : 'fg:#ff6b6b bold',
            'instruction' : 'fg:#f38181',
            'pointer'     : 'fg:#ffe66d bold',
            'question'    : 'fg:#ff6b6b bold',
            'selected'    : 'fg:#4ecdc4',
            'separator'   : 'fg:#95e1d3',
            'text'        : 'fg:#ffffff',
        },
        description = "Color and style configuration for interactive questionary prompts."
    )
    styles: dict[str, str] = Field(
        default = {
            'accent'    : 'bright_magenta',
            'bright'    : 'bright_white',
            'config'    : 'bright_blue',
            'debug'     : 'italic grey70',
            'dim'       : 'grey30',
            'drone'     : 'yellow',
            'error'     : 'bold red',
            'flock'     : 'bold bright_cyan',
            'highlight' : 'on dark_blue',
            'info'      : 'cyan',
            'muted'     : 'grey50',
            'success'   : 'bold green',
            'thermal'   : 'bold orange_red1',
            'warning'   : 'bold yellow',
        },
        description = "Named Rich styles mapping semantic names to color and formatting definitions."
    )
    system_components: dict[str, str] = Field(
        default = {
            'cuda'     : '🎮 CUDA',
            'dataset'  : '📥 Dataset',
            'disk'     : '💿 Disk',
            'gpu'      : '💎 GPU',
            'memory'   : '💾 Memory',
            'platform' : '🖥️  Platform',
            'python'   : '🐍 Python',
            'torch'    : '✨ PyTorch',
            'thermur'  : '🔥 Thermur',
        },
        description = "Display names with emoji icons for system components shown in info command."
    )
    system_table_columns: list[dict[str, str | int]] = Field(
        default = [
            {
                "header" : "Component",
                "style"  : "bright_cyan",
                "width"  : 15
            },
            {
                "header" : "Status",
                "style"  : "bright_white",
                "width"  : 40
            },
        ],
        description = "Column definitions including headers, styles, and widths for system info table."
    )
    training_component_cfgs: list[tuple[str, str, str]] = Field(
        default = [
            ("datamodule", "_system.datamodule", "📦 Data Module"),
            ("policy",     "_system.policy",     "🧠 Learning Policy"),
            ("trainer",    "_system.trainer",    "🌩️  Lightning Trainer"),
        ],
        description = (
            "Component initialization tuples containing configuration key, hydra path, "
            "and display name for training setup workflow."
        )
    )


class WandbModel(BaseModel, extra="forbid"):
    """
    Configuration for Weights & Biases experiment tracking.

    W&B provides comprehensive experiment tracking for machine learning workflows,
    including metric logging, hyperparameter tracking, model versioning, and
    visualization. This configuration controls how Lightning integrates with W&B
    during training and how the CLI accesses W&B for monitoring.

    The mode parameter allows flexible deployment:
    - "online": Full cloud synchronization for collaborative work
    - "offline": Local logging for air-gapped environments

    The API key is read from an environment variable to avoid hardcoding
    credentials in configuration files.
    """
    log_model: Literal["all", "false"] | bool = Field(
        default     = "all",
        description = (
            "Model checkpoint logging policy - 'all' saves every checkpoint, "
            "'false' or False disables model logging to save bandwidth."
        )
    )
    mode: Literal["online", "offline"] = Field(
        default     = "online",
        description = (
            "W&B tracking mode where 'online' syncs runs to cloud dashboard and "
            "'offline' stores all data locally for air-gapped environments."
        )
    )
    notes: str | None = Field(
        default     = None,
        description = (
            "Detailed description of the run, like a commit message. Use this to capture "
            "context, experimental setup, or purpose that helps recall what this run was "
            "testing. Appears in W&B UI Overview tab."
        )
    )
    project: str = Field(
        default     = "thermur-imitation",
        description = (
            "W&B project name for organizing experiments - groups related training "
            "runs for easier comparison and analysis."
        )
    )
    quiet: bool = Field(
        default     = True,
        description = (
            "Reduce W&B console output verbosity. When True, suppresses non-critical "
            "messages like sync reminders while keeping important status updates."
        )
    )
    run_name: str | None = Field(
        default     = None,
        description = (
            "Explicit run name (e.g., 'IM001', 'murmuration-test-v2'). Propagates to "
            "WandB, Hydra output directories, and checkpoint paths. If not set, "
            "WandB auto-generates a random name like 'cosmic-sunset-42'."
        )
    )
