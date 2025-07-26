"""
User interface and display configuration schemas.

This module defines configuration for the Rich terminal interface including
themes, styles, messages, and display components used by the ThermurUI helper.
"""
from pydantic import BaseModel, Field, PositiveInt


class DisplayModel(BaseModel, extra="forbid"):
    """
    Unified display and theme configuration.
    
    This model combines all visual presentation settings including color themes,
    UI component configurations, and display layouts. It provides a centralized
    configuration for the Rich terminal interface.
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
                "note"    : "Minimal setup for rapid testing"
            },
            {
                "desc"    : "Custom learning rate",
                "command" : "thermur train learning.learning_rate=0.001",
                "note"    : "Override specific parameters"
            },
            {
                "desc"    : "Monitor training progress",
                "command" : "thermur monitor",
                "note"    : "Real-time wandb dashboard"
            },
            {
                "desc"    : "Validate configuration",
                "command" : "thermur validate",
                "note"    : "Check setup before training"
            },
            {
                "desc"    : "Train with overrides",
                "command" : "thermur train flock.agent_count=20 learning.seed=42",
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
                "note"    : "Quick start with sample dataset"
            },
        ],
        description = "Example commands for quick start guide."
    )
    fire_gradient: list[str] = Field(
        default     = [
            "red3", 
            "red1", 
            "orange_red1", 
            "dark_orange", 
            "orange1", 
            "yellow1", 
            "bright_yellow"
        ],
        description = "Color gradient for fire effects."
    )
    styles: dict[str, str] = Field(
        default = {
            'thermal'   : 'bold orange_red1',
            'flock'     : 'bold bright_cyan', 
            'drone'     : 'yellow',
            'success'   : 'bold green',
            'warning'   : 'bold yellow',
            'error'     : 'bold red',
            'info'      : 'cyan',
            'accent'    : 'bright_magenta',
            'muted'     : 'grey50',
            'bright'    : 'bright_white',
            'dim'       : 'grey30',
            'highlight' : 'on dark_blue',
            'config'    : 'bright_blue',
            'debug'     : 'italic grey70',
        },
        description = "Named styles for consistent theming."
    )
    message_types: dict[str, dict[str, str]] = Field(
        default = {
            'info'    : {'icon': 'ℹ️ ', 'style': 'info'},
            'warning' : {'icon': '⚠️ ', 'style': 'warning'},
            'error'   : {'icon': '❌',  'style': 'error'},
            'success' : {'icon': '✅',  'style': 'success'},
            'thermal' : {'icon': '🔥',  'style': 'thermal'},
            'flock'   : {'icon': '🦅',  'style': 'flock'},
            'config'  : {'icon': '⚙️ ', 'style': 'config'},
            'debug'   : {'icon': '🐛',  'style': 'debug'},
        },
        description = "Message type configurations with icons and styles."
    )
    progress_bar_length: PositiveInt = Field(
        default     = 20,
        description = "Character width for progress bars."
    )
    progress_style: str = Field(
        default     = "thermal",
        description = "Style for progress indicators."
    )
    progress_unfilled_color: str = Field(
        default     = "grey30",
        description = "Color for unfilled progress bar segments."
    )
    resource_details_template: str = Field(
        default     = "{:.1f}{} / {:.1f}{}",
        description = "Format template for resource usage display."
    )
    system_components: dict[str, str] = Field(
        default = {
            'platform' : '🖥️  Platform',
            'python'   : '🐍 Python',
            'thermur'  : '🔥 Thermur',
            'torch'    : '⚡ PyTorch',
            'cuda'     : '🎮 CUDA',
            'mujoco'   : '🏃 MuJoCo',
            'gpu'      : '💎 GPU',
            'memory'   : '💾 Memory',
            'disk'     : '💿 Disk',
            'dataset'  : '📥 Dataset',
        },
        description = "System component display names with icons."
    )
    system_logic: dict[str, dict[str, str | bool]] = Field(
        default = {
            'platform'   : {'key': 'platform', 'format': '{}'},
            'python'     : {'key': 'python',   'format': '{}'},
            'thermur'    : {'key': 'thermur',  'format': 'v{}'},
            'torch'      : {'key': 'torch',    'format': '{}'},
            'cuda'       : {'key': 'cuda',     'format': '{}'},
            'mujoco'     : {'key': 'mujoco',   'format': 'v{}'},
            'gpu'        : {'key': 'gpu_name', 'format': '{}'},
            'memory'     : {
                'is_resource' : True, 
                'available'   : 'memory_available', 
                'total'       : 'memory_total'
            },
            'disk'       : {
                'is_resource' : True, 
                'available'   : 'disk_available', 
                'total'       : 'disk_total'
            },
            'dataset'    : {
                'key'    : 'dataset_size',
                'format' : '{:.1f} GB ({} files)',
                'count'  : 'dataset_count'
            },
        },
        description = "Logic for formatting system component values."
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
        description = "Column configuration for system table."
    )
    system_table_settings: dict[str, str | bool] = Field(
        default     = {"show_edge": True},
        description = "Settings for system information table."
    )
    training_component_configs: list[tuple[str, str, str]] = Field(
        default = [
            ("data_module", "data_module", "📦 Data Module"),
            ("policy",      "policy",      "🧠 Learning Policy"),
            ("trainer",     "trainer",     "⚡ Lightning Trainer"),
        ],
        description = (
            "List of (key, config_path, display_name) tuples for training "
            "component initialization."
        )
    )


class MessagesModel(BaseModel, extra="forbid"):
    """
    Comprehensive message configuration for all CLI output.
    
    This model consolidates all user-facing messages including general messages,
    status indicators, and validation feedback. Messages are organized by category
    for easier maintenance while maintaining a flat access pattern for performance.
    """
    browser_fail_template: str = Field(
        default     = "Failed to open browser: {e}",
        description = "Error message when browser launch fails."
    )
    browser_launch_template: str = Field(
        default     = (
            "Opening dashboard for project: "
            "[bright_cyan]{project}[/bright_cyan]"
        ),
        description = "Message shown when launching browser."
    )
    browser_manual_template: str = Field(
        default     = "Please visit manually: {url}",
        description = "Message with manual URL when browser fails."
    )
    browser_success: str = Field(
        default     = "Dashboard opened in your default browser!",
        description = "Success message after browser launch."
    )
    components_initialized: str = Field(
        default     = "All components initialized successfully!",
        description = "Message after successful component initialization."
    )
    dry_run_complete: str = Field(
        default     = "Dry run complete. Configuration validated successfully.",
        description = "Completion message for dry-run mode."
    )
    dry_run_config_display: str = Field(
        default     = "Final configuration that would be used:",
        description = "Message before showing dry-run configuration."
    )
    dry_run_header: str = Field(
        default     = "[bold yellow]DRY RUN MODE[/bold yellow] - No training will occur",
        description = "Header message for dry-run mode."
    )
    loading_components: str = Field(
        default     = "Loading training components...",
        description = "Status message during component loading."
    )
    monitoring_dynamics: str = Field(
        default     = "Monitoring thermal constraints and flock dynamics",
        description = "Message about training monitoring."
    )
    ready_to_train: str = Field(
        default     = "Ready to train some thermal flocks? 🔥",
        description = "Encouraging message to start training."
    )
    skipping_checks: str = Field(
        default     = "Skipping system checks (--force enabled)",
        description = "Message when force flag skips validation."
    )
    status: dict[str, str] = Field(
        default = {
            "checking_reqs"            : "[thermal]Checking system requirements...[/thermal]",
            "init_modules"             : "Initializing core modules...",
            "instantiating_components" : "Instantiating components...",
            "launching_browser"        : "[flock]Launching browser...[/flock]",
            "loading_config_sys"       : "Loading configuration system...",
            "preparing_hydra"          : "Preparing Hydra runtime...",
            "ready_to_train"           : "Ready to train!",
            "registering_configs"      : "Registering configurations...",
            "setup_component_template" : "Setting up {display_name}...",
            "validating_config"        : "[accent]Validating configuration...[/accent]",
        },
        description = "Status messages for progress indicators."
    )
    track_wandb: str = Field(
        default     = "Track progress in your wandb dashboard",
        description = "Tip about wandb tracking."
    )
    training_cancelled: str = Field(
        default     = "Training cancelled by user.",
        description = "Message when user cancels training."
    )
    training_complete_sub: str = Field(
        default     = "Your thermal flock has learned to fly",
        description = "Subtitle for training completion."
    )
    training_failed_template: str = Field(
        default     = "Training failed: {e}",
        description = "Error message template for training failure."
    )
    training_interrupted: str = Field(
        default     = "Training interrupted by user.",
        description = "Message when training is interrupted."
    )
    validation: dict[str, str] = Field(
        default = {
            "all_passed"              : "All validations passed!",
            "config_fail"             : "Configuration validation failed:",
            "config_issues"           : "Configuration issues found:",
            "config_passed"           : "Configuration validation passed!",
            "force_override"          : "Use --force to override or fix the issues above.",
            "gpu_unavailable"         : "GPU not available - training will be slower on CPU",
            "invalid_override_format" : "Invalid override format (missing '=')",
            "invalid_override_key"    : "Invalid override key format",
            "review_issues"           : "Review the issues above before training",
            "system_ready"            : "Your system is ready for training",
            "with_warnings"           : "Validation completed with warnings",
        },
        description = "Validation and diagnostic messages."
    )
    wandb: dict[str, str] = Field(
        default = {
            "dashboard" : "🎨 Dashboard: [link={}]{}[/link]",
            "login"     : "Run 'wandb login' to authenticate",
            "no_auth"   : "🎨 wandb: Not authenticated • Run 'wandb login'",
            "open"      : "Opening dashboard for project: [bright_cyan]{}[/]",
            "ready"     : "🎨 wandb: Ready • User: {}",
            "required"  : "wandb authentication required to access dashboard",
        },
        description = "Wandb integration display messages."
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