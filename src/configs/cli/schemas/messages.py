"""
Message and text configuration schemas for the Thermur CLI.

This module defines models for all user-facing strings, prompts, status
messages, and section headers used throughout the CLI interface.
"""
from pydantic import BaseModel, Field


class MessagesModel(BaseModel, extra="forbid"):
    """
    Message text and templates for CLI output.
    
    This model centralizes all user-facing strings, from simple status
    updates to formatted error messages, to ensure consistency.
    """
    browser_fail_template: str = Field(
        default     = "Failed to open browser: {e}",
        description = "Error message when browser launch fails"
    )
    browser_launch_template: str = Field(
        default     = "Opening dashboard for project: [bright_cyan]{project}[/bright_cyan]",
        description = "Message shown when launching browser"
    )
    browser_manual_template: str = Field(
        default     = "Please visit manually: {url}",
        description = "Message with manual URL when browser fails"
    )
    browser_success: str = Field(
        default     = "Dashboard opened in your default browser!",
        description = "Success message after browser launch"
    )
    components_initialized: str = Field(
        default     = "All components initialized successfully!",
        description = "Message after successful component initialization"
    )
    loading_components: str = Field(
        default     = "Loading training components...",
        description = "Status message during component loading"
    )
    monitoring_dynamics: str = Field(
        default     = "Monitoring thermal constraints and flock dynamics",
        description = "Message about training monitoring"
    )
    ready_to_train: str = Field(
        default     = "Ready to train some thermal flocks? 🔥",
        description = "Encouraging message to start training"
    )
    skipping_checks: str = Field(
        default     = "Skipping system checks (--force enabled)",
        description = "Message when force flag skips validation"
    )
    track_wandb: str = Field(
        default     = "Track progress in your wandb dashboard",
        description = "Tip about wandb tracking"
    )
    training_cancelled: str = Field(
        default     = "Training cancelled by user.",
        description = "Message when user cancels training"
    )
    training_complete_sub: str = Field(
        default     = "Your thermal flock has learned to fly",
        description = "Subtitle for training completion"
    )
    training_failed_template: str = Field(
        default     = "Training failed: {e}",
        description = "Error message template for training failure"
    )
    training_interrupted: str = Field(
        default     = "Training interrupted by user.",
        description = "Message when training is interrupted"
    )
    wandb_unavailable: str = Field(
        default     = "wandb monitoring not available - please authenticate first",
        description = "Message when wandb is not available"
    )


class PromptsModel(BaseModel, extra="forbid"):
    """
    Configures user interaction prompts and messages.
    
    This model contains all the text templates used for user interaction,
    including confirmation prompts, help messages, and interactive guidance.
    The prompts use string formatting placeholders for dynamic content.
    """
    config_not_found_error: str = Field(
        default     = "Configuration '{config}' not found in available workloads",
        description = "Error message template for missing configurations"
    )
    confirm_override_prompt: str = Field(
        default     = "Apply {count} override(s)?",
        description = "Confirmation prompt for applying overrides"
    )
    override_prompt: str = Field(
        default     = "Enter override value for {field_name}:",
        description = "Prompt template for configuration overrides"
    )
    package_missing_error: str = Field(
        default     = "Required package '{package}' is not installed",
        description = "Error message template for missing packages"
    )
    python_version_error: str = Field(
        default     = "Python {current} detected, but {required} or higher is required",
        description = "Error message template for Python version mismatch"
    )
    questionary_style: list[tuple[str, str]] = Field(
        default = [
            ('question',    'fg:#ff6b6b bold'),
            ('answer',      'fg:#4ecdc4 bold'),
            ('pointer',     'fg:#ffe66d bold'),
            ('highlighted', 'fg:#ff6b6b bold'),
            ('selected',    'fg:#4ecdc4'),
            ('separator',   'fg:#95e1d3'),
            ('instruction', 'fg:#f38181'),
            ('text',        'fg:#ffffff'),
            ('disabled',    'fg:#808080 italic'),
        ],
        description = "Questionary prompt styling configuration"
    )
    workload_selection_prompt: str = Field(
        default     = "Select a workload configuration:",
        description = "Prompt shown when user needs to select a workload"
    )


class StatusModel(BaseModel, extra="forbid"):
    """
    Text for status indicators shown during processing.
    
    These strings are used in progress bars and spinners to provide
    real-time feedback to the user about what the application is doing.
    """
    checking_reqs: str = Field(
        default     = "[thermal]Checking system requirements...[/thermal]",
        description = "Status text during requirement checking"
    )
    init_modules: str = Field(
        default     = "Initializing core modules...",
        description = "Status text during module initialization"
    )
    instantiating_components: str = Field(
        default     = "Instantiating components...",
        description = "Status text during component instantiation"
    )
    launching_browser: str = Field(
        default     = "[flock]Launching browser...[/flock]",
        description = "Status text during browser launch"
    )
    loading_config_sys: str = Field(
        default     = "Loading configuration system...",
        description = "Status text during config system load"
    )
    preparing_hydra: str = Field(
        default     = "Preparing Hydra runtime...",
        description = "Status text during Hydra preparation"
    )
    ready_to_train: str = Field(
        default     = "Ready to train!",
        description = "Status text when training is ready"
    )
    registering_configs: str = Field(
        default     = "Registering configurations...",
        description = "Status text during config registration"
    )
    setup_component_template: str = Field(
        default     = "Setting up {display_name}...",
        description = "Template for component setup status"
    )
    validating_config: str = Field(
        default     = "[accent]Validating configuration...[/accent]",
        description = "Status text during config validation"
    )