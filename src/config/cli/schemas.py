"""
Configuration schemas for CLI components.

This module provides all configuration models for the Thermur CLI system. These schemas
control user interaction, display formatting, system validation, and data management.
Unlike the training configuration, these schemas are instantiated directly rather than
through Hydra's runtime, as they configure the CLI framework itself.

The schemas are organized alphabetically and follow consistent patterns for validation
and defaults. They are consumed by CLI helpers (ThermurUI, CLIPrompts, SystemInspector,
GlobusManager) to provide a rich terminal experience.
"""
from pathlib           import Path
from pydantic          import BaseModel, computed_field, Field, PositiveInt, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing            import Literal, Optional

SECRETS_DIR = Path.home() / ".config" / "thermur" / "secrets"


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


class DownloadModel(BaseModel, extra="forbid"):
    """
    Download and management configuration for the CLI.
    
    This model contains settings for file downloads, display options, and
    caching behavior, separate from the dataset schema used for training.
    """
    globus_client_id: str = Field(
        default     = "ac349f52-8197-4a41-8d6d-5ae1c879273f",
        description = "Native app client ID for Globus OAuth2 authentication flow."
    )
    globus_dataset_path: str = Field(
        default     = "/1/published/publication_309/submitted_data",
        description = "Path to the WRF-Fire dataset within the FRDR Globus endpoint."
    )
    globus_endpoint_id: str = Field(
        default     = "f163c1b3-9c88-42f6-a7bb-5839ed6c4063",
        description = "UUID of the FRDR Globus endpoint hosting WRF-Fire simulations."
    )
    globus_scopes: str = Field(
        default     = "urn:globus:auth:scope:transfer.api.globus.org:all",
        description = "OAuth2 scopes required for Globus transfer operations."
    )
    recommended_files: list[dict[str, str]] = Field(
        default = [
            {
                "file" : "wrfout_W3F1R0",
                "desc" : "Light wind (3m/s) over short grass. Represents prescribed burns "
                         "or early-season fires in grasslands with stable atmospheric conditions."
            },
            {
                "file" : "wrfout_W5F7R4",  
                "desc" : "Moderate wind (5m/s) through brushy forest understory. Models typical "
                         "wildfire conditions with mixed vegetation and moderate atmospheric mixing."
            },
            {
                "file" : "wrfout_W8F13R6",
                "desc" : "Strong wind (8m/s) through heavy dead trees and branches. Simulates "
                         "post-logging or storm damage areas with deep atmospheric mixing."
            },
            {
                "file" : "wrfout_W12F4R8",
                "desc" : "Extreme wind (12m/s) in dense shrubland. Represents high-risk fire "
                         "weather in Mediterranean climates with strong temperature inversions."
            }
        ],
        description = "Recommended starter files with condition descriptions."
    )
    sample_data_path: Path = Field(
        default     = Path("data/samples/wrf_sample.nc"),
        description = "Local path where sample NetCDF file will be stored after extraction."
    )
    sample_data_url: str = Field(
        default     = "https://huggingface.co/datasets/Jybbs/sfire-samples/resolve/main/samples.tar.gz",
        description = "Hugging Face direct download URL for sample data tar.gz file."
    )
    sample_extract_dir: Path = Field(
        default     = Path("data"),
        description = "Directory where sample tar.gz will be extracted."
    )
    source: Literal["sample", "wrf-sfire", ""] = Field(
        default     = "",
        description = "Data source to download: 'sample' for quick start, 'wrf-sfire' for full dataset."
    )
    transfer_timeout: int = Field(
        default     = 86400,
        description = "Maximum seconds to wait for transfer completion (default: 24 hours)."
    )
    wrf_sfire_dir: Path = Field(
        default     = Path("data/wrf-sfire"),
        description = "Local directory for storing WRF-SFIRE dataset files from Globus."
    )


class GlobusSecrets(BaseSettings):
    """
    Secure storage for Globus OAuth2 tokens.
    
    Uses Pydantic's BaseSettings with secrets_dir for automatic persistence.
    Each token field is stored as a separate file in the secrets directory,
    with the filename matching the field name.
    """
    refresh_token: Optional[SecretStr] = Field(
        default     = None,
        description = "Long-lived token used to obtain new access tokens"
    )
    scope: Optional[str] = Field(
        default     = None,
        description = "Space-delimited OAuth2 scopes granted by this token"
    )
    secrets_path: Path = Field(
        default     = SECRETS_DIR,
        description = "Directory for storing secret files"
    )
    
    model_config = SettingsConfigDict(
        case_sensitive = False,
        secrets_dir    = str(SECRETS_DIR) if SECRETS_DIR.exists() else None
    )
    
    @computed_field
    @property
    def is_valid(self) -> bool:
        """
        Check if all required token fields are present.
        """
        return all(
            getattr(self, field) is not None 
            for field in ['refresh_token', 'scope']
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
    override_syntax_help: str = Field(
        default = (
            "# Override examples:\n"
            "optimizer.learning_rate=0.001     # Learning rate\n"
            "experience.batch_size=64          # Batch size\n"
            "flock.agent_count=10              # Number of agents\n"
            "hardware.precision=32-true        # Training precision\n"
            "wandb.mode=offline                # W&B logging mode"
        ),
        description = "Help text for configuration override syntax."
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


class PromptsModel(BaseModel, extra="forbid"):
    """
    Interactive prompt and dialog configuration.
    
    This model contains all settings for user interaction including error messages,
    prompt templates, and questionary styling. It manages the conversational flow
    of the CLI application.
    """
    config_not_found_error: str = Field(
        default     = "Configuration '{config}' not found in available workloads",
        description = "Error message template for missing configurations."
    )
    confirm_override_prompt: str = Field(
        default     = "Apply {count} override(s)?",
        description = "Confirmation prompt for applying overrides."
    )
    override_prompt: str = Field(
        default     = "Enter override value for {field_name}:",
        description = "Prompt template for configuration overrides."
    )
    package_missing_error: str = Field(
        default     = "Required package '{package}' is not installed",
        description = "Error message template for missing packages."
    )
    python_version_error: str = Field(
        default     = "Python {current} detected, but {required} or higher is required",
        description = "Error message template for Python version mismatch."
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
        description = "Questionary prompt styling configuration."
    )
    workload_selection_prompt: str = Field(
        default     = "Select a workload configuration:",
        description = "Prompt shown when user needs to select a workload."
    )


class SystemModel(BaseModel, extra="forbid"):
    """
    System inspection and validation configuration.
    
    This model defines system requirements and validation rules used
    by the SystemInspector to check environment compatibility.
    """
    cuda_preferred: bool = Field(
        default     = True,
        description = "Whether CUDA GPU acceleration is preferred for training."
    )
    dataset_validation: dict[str, float] = Field(
        default = {
            "min_size_gb"    : 0.1,
            "max_size_gb"    : 10000.0,
            "warning_size_gb": 100.0,
        },
        description = "Dataset size validation thresholds in gigabytes."
    )
    mujoco_min_version: str = Field(
        default     = "2.3.0",
        description = "Minimum MuJoCo version required for physics simulation."
    )
    python_min_version: tuple[int, int] = Field(
        default     = (3, 9),
        description = "Minimum Python version required as (major, minor) tuple."
    )
    required_packages: list[str] = Field(
        default = [
            "torch",
            "pytorch_lightning",
            "torchrl",
            "mujoco",
            "hydra-core",
            "wandb",
            "rich",
            "typer",
        ],
        description = "List of required Python packages for system validation."
    )


