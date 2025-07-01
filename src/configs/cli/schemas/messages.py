"""
Message and text configuration schemas for the Thermur CLI.

This module defines models for all user-facing strings, prompts, status
messages, and section headers used throughout the CLI interface.
"""
from pydantic import BaseModel, Field


class HeadersModel(BaseModel, extra="forbid"):
    """
    Titles and subtitles for different CLI sections.
    
    These constants are used to create the main styled panels that introduce
    a command's function to the user.
    """
    config_gen_title: str = Field(
        default     = "Generated Configuration Overrides",
        description = "Title for generated config overrides display"
    )
    info_title: str = Field(
        default     = "Thermur System Information",
        description = "Title for info command header"
    )
    main_title: str = Field(
        default     = "Welcome to Thermur",
        description = "Title for main welcome screen"
    )
    monitor_subtitle_template: str = Field(
        default     = "Project: {project}",
        description = "Subtitle template for monitor command"
    )
    monitor_title: str = Field(
        default     = "wandb Monitoring",
        description = "Title for monitor command header"
    )
    train_subtitle: str = Field(
        default     = "Thermally-constrained drone flock imitation learning",
        description = "Subtitle for train command header"
    )
    train_title: str = Field(
        default     = "Thermur Training System",
        description = "Title for train command header"
    )
    validate_subtitle: str = Field(
        default     = "Pre-flight checks for training",
        description = "Subtitle for validate command header"
    )
    validate_title: str = Field(
        default     = "System Validation",
        description = "Title for validate command header"
    )


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
    config_gen_add_cmd: str = Field(
        default     = "Add these to your training command:",
        description = "Instructions for using generated overrides"
    )
    config_gen_use_ind: str = Field(
        default     = "Or use them individually:",
        description = "Alternative instructions for overrides"
    )
    loading_components: str = Field(
        default     = "Loading training components...",
        description = "Status message during component loading"
    )
    monitoring_dynamics: str = Field(
        default     = "Monitoring thermal constraints and flock dynamics",
        description = "Message about training monitoring"
    )
    no_config_changes: str = Field(
        default     = "No configuration changes made.",
        description = "Message when no config overrides are generated"
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
    training_complete_header: str = Field(
        default     = "Training Complete! 🎉",
        description = "Header for training completion"
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
        default     = "wandb not available - install with 'pip install wandb'",
        description = "Message when wandb is not installed"
    )
    wandb_details_api_key: str = Field(
        default     = "Run 'wandb login' to connect",
        description = "Details when API key exists but not logged in"
    )
    wandb_details_connected: str = Field(
        default     = "Logged in as: {user}",
        description = "Details when wandb is connected"
    )
    wandb_details_not_connected: str = Field(
        default     = "Run 'wandb login' to authenticate",
        description = "Details when wandb is installed but not connected"
    )
    wandb_details_not_installed: str = Field(
        default     = "Install with: pip install wandb",
        description = "Details when wandb is not installed"
    )
    wandb_status_api_key: str = Field(
        default     = "[yellow]API key found[/yellow]",
        description = "Status when API key exists but not logged in"
    )
    wandb_status_connected: str = Field(
        default     = "[green]Connected[/green]",
        description = "Status when wandb is connected"
    )
    wandb_status_not_connected: str = Field(
        default     = "[red]Not connected[/red]",
        description = "Status when wandb is installed but not connected"
    )
    wandb_status_not_installed: str = Field(
        default     = "[red]Not installed[/red]",
        description = "Status when wandb is not installed"
    )


class MessageTypesModel(BaseModel, extra="forbid"):
    """
    Message type definitions with icons and styles.
    
    Maps message types to their visual representation in the CLI,
    including icons and color styles for consistent messaging.
    """
    config_icon: str = Field(
        default     = "⚙️",
        description = "Icon for configuration messages"
    )
    config_style: str = Field(
        default     = "accent",
        description = "Style for configuration messages"
    )
    error_icon: str = Field(
        default     = "🚨",
        description = "Icon for error messages"
    )
    error_style: str = Field(
        default     = "error",
        description = "Style for error messages"
    )
    flock_icon: str = Field(
        default     = "🪽",
        description = "Icon for flock messages"
    )
    flock_style: str = Field(
        default     = "flock",
        description = "Style for flock messages"
    )
    info_icon: str = Field(
        default     = "💡",
        description = "Icon for info messages"
    )
    info_style: str = Field(
        default     = "info",
        description = "Style for info messages"
    )
    step_icon: str = Field(
        default     = "🔥",
        description = "Icon for step messages"
    )
    step_style: str = Field(
        default     = "thermal",
        description = "Style for step messages"
    )
    success_icon: str = Field(
        default     = "✅",
        description = "Icon for success messages"
    )
    success_style: str = Field(
        default     = "success",
        description = "Style for success messages"
    )
    thermal_icon: str = Field(
        default     = "🔥",
        description = "Icon for thermal messages"
    )
    thermal_style: str = Field(
        default     = "thermal",
        description = "Style for thermal messages"
    )
    tip_icon: str = Field(
        default     = "💭",
        description = "Icon for tip messages"
    )
    tip_style: str = Field(
        default     = "muted",
        description = "Style for tip messages"
    )
    warning_icon: str = Field(
        default     = "🌡️",
        description = "Icon for warning messages"
    )
    warning_style: str = Field(
        default     = "warning",
        description = "Style for warning messages"
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


class SectionsModel(BaseModel, extra="forbid"):
    """
    Titles for different content sections printed to the console.
    
    These are used to create styled horizontal rules that visually break up
    the content of a command's output.
    """
    available_commands: str = Field(
        default     = "Available Commands",
        description = "Section title for command list"
    )
    building_components: str = Field(
        default     = "Building Training Components",
        description = "Section title during component creation"
    )
    config_check: str = Field(
        default     = "Configuration Check",
        description = "Section title for configuration validation"
    )
    config_setup: str = Field(
        default     = "Configuration Setup",
        description = "Section title for config setup phase"
    )
    config_system: str = Field(
        default     = "Configuration System",
        description = "Section title for config system info"
    )
    features: str = Field(
        default     = "Features",
        description = "Section title for feature list"
    )
    getting_started: str = Field(
        default     = "Getting Started",
        description = "Section title for getting started guide"
    )
    init_training: str = Field(
        default     = "Initializing Training",
        description = "Section title when starting training"
    )
    integration_check: str = Field(
        default     = "Integration Check",
        description = "Section title for integration validation"
    )
    integration_status: str = Field(
        default     = "Integration Status",
        description = "Section title for integration info"
    )
    quick_start: str = Field(
        default     = "Common Commands",
        description = "Section title for quick start guide"
    )
    system_validation: str = Field(
        default     = "System Validation",
        description = "Section title for system checks"
    )
    training_started: str = Field(
        default     = "Training Started",
        description = "Section title when training begins"
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